"""Angle-sensor I2C bench: does a load make the reads fail or stall?

Reads the two angle sensors with the daemon's own AngleCapture (same buses,
same 100 Hz loop) for a long time, with or without the RPi camera running next
to it, and logs every slow or failed I2C read with its errno. Angle holes (an
EIO then a 1 s ETIMEDOUT on one bus) come about once per 15-40 min of
recording, so give each condition at least an hour (--duration, default 3600 s).

Run on the Pi with the grabette venv, daemon STOPPED (it holds the cameras and
reads the sensors itself), from the daemon's directory (its .env is read):
    sudo systemctl stop grabette
    .venv/bin/python benchmarks/bench_i2c_angles.py --mode i2c            # A: I2C alone
    .venv/bin/python benchmarks/bench_i2c_angles.py --mode camera         # B: + camera, no encoder
    .venv/bin/python benchmarks/bench_i2c_angles.py --mode record         # C: + H.264 recordings, as in an episode
    .venv/bin/python benchmarks/bench_i2c_angles.py --mode orbbec-record  # E: depth camera recordings, no RPi camera
    .venv/bin/python benchmarks/bench_i2c_angles.py --mode orbbec         # F: depth camera streaming, nothing recorded
    sudo systemctl start grabette

E records the depth camera as an episode does (x264 and FFV1 ffmpeg encoders,
files on the SD card, deleted after each segment) with the daemon's Orbbec
settings; F keeps only its USB stream and power draw, to tell those apart from
the encoding load. On 2026-11-10, C gave no I2C error in 2 h on two grabettes
while full episodes (D) gave one every 14-37 min.

The daemon itself (D: full episodes) is driven by benchmarks/run_episodes.py.

Writes ~/benchmarks/<UTC start>_i2c_<mode>/:
  events.csv    every read over --slow-ms or failed, and every loop gap over
                100 ms (UTC, CLOCK_BOOTTIME, sensor, duration, errno)
  summary.json  reads, loop rate, errors by errno, slow reads, max read,
                shared-IRQ count
  kernel.log    kernel messages of the run (i2c-bcm2835 "unexpected interrupt")
  bench.log     everything printed during the run

Ctrl-C ends the run early and still writes the summary. Works on any branch:
it only uses AngleCapture.init_sensors/_read_angle_raw, VideoCapture and
OrbbecCapture.
"""

import argparse
import csv
import errno
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLOCK_BOOTTIME = getattr(time, "CLOCK_BOOTTIME", 7)
SENSORS = ("distal", "proximal")
LOOP_GAP_S = 0.1  # same threshold as the daemon's "Angle loop stalled"
REPORT_EVERY_S = 60

stop = threading.Event()
log_file = None  # bench.log, once the output folder exists


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc):%H:%M:%S} {msg}"
    print(line, flush=True)
    if log_file is not None:
        log_file.write(line + "\n")


def i2c_irq_count() -> dict | None:
    """The /proc/interrupts line shared by the I2C controllers (IRQ 41 on a
    Pi 4, with the camera's i2c-0): its name and total count, all CPUs."""
    try:
        lines = Path("/proc/interrupts").read_text().splitlines()
    except OSError:
        return None
    ncpu = len(lines[0].split())
    for line in lines[1:]:
        if "i2c" in line:
            fields = line.split()
            return {"irq": fields[0].rstrip(":"),
                    "count": sum(int(x) for x in fields[1:1 + ncpu] if x.isdigit()),
                    "devices": " ".join(fields[1 + ncpu:])}
    return None


def daemon_active() -> bool:
    try:
        return subprocess.run(["systemctl", "is-active", "--quiet", "grabette"]).returncode == 0
    except FileNotFoundError:
        return False


def journal(since_epoch: float, *args: str) -> str:
    try:
        r = subprocess.run(["journalctl", "--no-pager", "-o", "short-precise", "--utc",
                            "--since", f"@{int(since_epoch)}", *args],
                           capture_output=True, text=True, timeout=60)
        return r.stdout or r.stderr
    except (OSError, subprocess.SubprocessError) as e:
        return f"journalctl failed: {e}\n"


def error_name(exc: BaseException) -> str:
    """ETIMEDOUT (the driver gave up waiting), EREMOTEIO (no ACK from the
    sensor), EIO (bus error)... or the exception type when there is no errno."""
    code = getattr(exc, "errno", None)
    if isinstance(code, int):
        return errno.errorcode.get(code, f"errno {code}")
    return type(exc).__name__


def timed_read(cap, i2c) -> tuple[float, str | None, str]:
    """(seconds, error name or None, detail) of one read, as the daemon does it."""
    t = time.monotonic()
    try:
        cap._read_angle_raw(i2c)
    except Exception as e:
        return time.monotonic() - t, error_name(e), f"{type(e).__name__}: {e}"
    return time.monotonic() - t, None, ""


# ── Camera load (modes camera and record) ─────────────────────────────


def make_camera(sync):
    """The camera exactly as the daemon builds it on this branch."""
    from grabette.backend.rpi import FPS
    from grabette.hardware.camera import VideoCapture
    return VideoCapture(sync, fps=FPS)


def camera_only(sync, stats: Counter) -> None:
    """Camera streaming in the daemon's configuration, nothing encoded or written
    (init_camera configures and starts picamera2 without an encoder)."""
    cam = make_camera(sync)
    try:
        cam.init_camera()
        log("camera streaming")
        stop.wait()
    except Exception as e:
        # Without the camera this run would be mode i2c under another name.
        stats["camera_failed"] += 1
        log(f"camera failed, stopping the run: {type(e).__name__}: {e}")
        stop.set()
    finally:
        cam.close()


def recording(sync, out_dir: Path, segment_s: float, pause_s: float, stats: Counter) -> None:
    """Back-to-back recordings through the daemon's VideoCapture (encoder,
    writes, mux, checks), deleted after each one."""
    seg_dir = out_dir / "segments"
    seg_dir.mkdir()
    n = 0
    while not stop.is_set():
        n += 1
        cam = make_camera(sync)
        try:
            cam.init_camera()
            cam.start_recording(seg_dir / f"seg{n:03d}.mp4")
            log(f"recording segment {n}")
            stop.wait(segment_s)
            cam.stop()
            stats["segments_ok"] += 1
        except Exception as e:
            stats["segments_failed"] += 1
            log(f"segment {n} failed: {type(e).__name__}: {e}")
            try:
                cam.close()
            except Exception:
                pass
        for f in seg_dir.iterdir():
            f.unlink()
        stop.wait(pause_s)
    shutil.rmtree(seg_dir, ignore_errors=True)


# ── Depth camera load (modes orbbec and orbbec-record) ────────────────


def orbbec_load(sync, out_dir: Path, segment_s: float, pause_s: float,
                record: bool, stats: Counter) -> None:
    try:
        _orbbec_load(sync, out_dir, segment_s, pause_s, record, stats)
    except Exception as e:
        # Without the depth camera this run would be mode i2c under another name.
        stats["orbbec_failed"] += 1
        log(f"orbbec failed, stopping the run: {type(e).__name__}: {e}")
        stop.set()


def _orbbec_load(sync, out_dir, segment_s, pause_s, record, stats) -> None:
    """The daemon's OrbbecCapture: its pipeline always streams (live view);
    with `record`, back-to-back recordings deleted after each one."""
    from grabette.hardware.orbbec import OrbbecCapture

    kwargs = {}
    try:
        from grabette.config import Settings
        settings = Settings()
        kwargs = dict(ir_exposure_us=settings.orbbec_ir_exposure_us,
                      ir_gain=settings.orbbec_ir_gain,
                      rotate_180=bool(settings.orbbec_rotate_180))
    except Exception as e:
        log(f"settings not read ({type(e).__name__}: {e}), Orbbec defaults")
    cam = OrbbecCapture(sync, **kwargs)
    cam.init_device()
    log(f"orbbec streaming {kwargs}")
    try:
        if not record:
            stop.wait()
            return
        seg_dir = out_dir / "segments"
        seg_dir.mkdir()
        n = 0
        while not stop.is_set():
            n += 1
            seg = seg_dir / f"seg{n:03d}"
            cam.wait_until_ready()  # as the daemon does before t=0
            try:
                cam.start_recording(seg)
                log(f"recording segment {n}")
                stop.wait(segment_s)
                st = cam.stop_recording()
                stats["segments_ok"] += 1
                stats["dropped_frames"] += st.get("dropped_frames") or 0
                gap = round(st.get("left_max_gap_ms") or 0)
                stats["max_left_gap_ms"] = max(stats["max_left_gap_ms"], gap)
                log(f"segment {n}: {st.get('left_frames')} left frames, "
                    f"max gap {gap} ms, {st.get('dropped_frames')} dropped")
            except Exception as e:
                stats["segments_failed"] += 1
                log(f"segment {n} failed: {type(e).__name__}: {e}")
                try:
                    cam.stop_recording()
                except Exception:
                    pass
            shutil.rmtree(seg, ignore_errors=True)
            stop.wait(pause_s)
        shutil.rmtree(seg_dir, ignore_errors=True)
    finally:
        cam.shutdown()


# ── I2C reads ─────────────────────────────────────────────────────────


def read_loop(cap, rate_hz: float, slow_s: float, writer, stats: dict) -> None:
    period = 1.0 / rate_hz
    buses = (cap.i2c_bus_1, cap.i2c_bus_2)
    prev = None
    last_report = time.monotonic()
    while not stop.is_set():
        start = time.monotonic()
        stats["loops"] += 1
        if prev is not None and start - prev > LOOP_GAP_S:
            stats["loop_gaps"] += 1
            writer.writerow([utc_now(), f"{time.clock_gettime(CLOCK_BOOTTIME):.6f}",
                             "loop", "", f"{(start - prev) * 1000:.1f}", "gap", ""])
        prev = start
        for i, i2c in enumerate((cap._i2c_1, cap._i2c_2)):
            seconds, err, detail = timed_read(cap, i2c)
            s = stats["sensors"][SENSORS[i]]
            s["reads"] += 1
            s["max_read_ms"] = max(s["max_read_ms"], seconds * 1000)
            if seconds > 0.1:
                s["reads_over_100ms"] += 1
            if seconds > 0.5:
                s["reads_over_500ms"] += 1
            if err is not None:
                s["errors"][err] += 1
            if err is not None or seconds > slow_s:
                writer.writerow([utc_now(), f"{time.clock_gettime(CLOCK_BOOTTIME):.6f}",
                                 SENSORS[i], buses[i], f"{seconds * 1000:.1f}",
                                 err or "slow", detail])
                if err is not None or seconds > 0.5:
                    log(f"{SENSORS[i]} /dev/i2c-{buses[i]}: {seconds * 1000:.0f} ms {err or ''} {detail}")
        if start - last_report > REPORT_EVERY_S:
            last_report = start
            log(" | ".join(f"{n}: {s['reads']} reads, {sum(s['errors'].values())} err, "
                           f"{s['reads_over_500ms']} >500 ms, max {s['max_read_ms']:.0f} ms"
                           for n, s in stats["sensors"].items()))
        sleep = period - (time.monotonic() - start)
        if sleep > 0:
            time.sleep(sleep)


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", required=True,
                   choices=("i2c", "camera", "record", "orbbec", "orbbec-record"))
    p.add_argument("--duration", type=float, default=3600, help="seconds (default 3600)")
    p.add_argument("--rate", type=float, default=100, help="I2C read loop, Hz (default 100)")
    p.add_argument("--slow-ms", type=float, default=50,
                   help="log reads slower than this (default 50; normal is 1-15 ms)")
    p.add_argument("--segment", type=float, default=120,
                   help="record, orbbec-record: length of each recording, s (default 120)")
    p.add_argument("--pause", type=float, default=5,
                   help="record, orbbec-record: pause between recordings, s (default 5)")
    p.add_argument("--out", type=Path, default=Path.home() / "benchmarks")
    p.add_argument("--allow-daemon", action="store_true",
                   help="run even though the daemon is active")
    args = p.parse_args()

    if daemon_active() and not args.allow_daemon:
        sys.exit("the daemon holds the cameras and reads the sensors: "
                 "sudo systemctl stop grabette (or --allow-daemon)")

    from grabette.hardware.angle import AngleCapture
    from grabette.hardware.sync import SyncManager

    t0_epoch = time.time()
    out_dir = args.out / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_i2c_{args.mode}"
    out_dir.mkdir(parents=True)
    global log_file
    log_file = open(out_dir / "bench.log", "a", buffering=1)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    timer = threading.Timer(args.duration, stop.set)
    timer.daemon = True
    timer.start()

    irq_start = i2c_irq_count()
    summary: dict = {"kind": "i2c_bench", "mode": args.mode, "start_utc": utc_now(),
                     "args": {k: str(v) for k, v in vars(args).items()}}
    load_stats: Counter = Counter()
    log(f"mode {args.mode} for {args.duration:.0f} s -> {out_dir}")

    sync = SyncManager()
    sync.start()
    cap = AngleCapture(sync, sample_rate_hz=int(args.rate))
    cap.init_sensors()
    load = None
    if args.mode == "camera":
        load = threading.Thread(target=camera_only, args=(sync, load_stats), daemon=True)
    elif args.mode == "record":
        load = threading.Thread(target=recording, daemon=True,
                                args=(sync, out_dir, args.segment, args.pause, load_stats))
    elif args.mode in ("orbbec", "orbbec-record"):
        load = threading.Thread(target=orbbec_load, daemon=True, args=(
            sync, out_dir, args.segment, args.pause, args.mode == "orbbec-record", load_stats))
    if load is not None:
        load.start()
    stats = {"loops": 0, "loop_gaps": 0, "sensors": {n: {
        "bus": b, "reads": 0, "errors": Counter(), "reads_over_100ms": 0,
        "reads_over_500ms": 0, "max_read_ms": 0.0}
        for n, b in zip(SENSORS, (cap.i2c_bus_1, cap.i2c_bus_2))}}
    with open(out_dir / "events.csv", "w", newline="", buffering=1) as f:
        w = csv.writer(f)
        w.writerow(["utc", "boottime_s", "sensor", "bus", "ms", "kind", "detail"])
        try:
            read_loop(cap, args.rate, args.slow_ms / 1000, w, stats)
        finally:
            stop.set()
            if load is not None:
                load.join(timeout=60)
            cap._close()
    for s in stats["sensors"].values():
        s["errors"] = dict(s["errors"])
        s["max_read_ms"] = round(s["max_read_ms"], 1)
    summary.update(stats)

    summary["end_utc"] = utc_now()
    summary["duration_s"] = round(time.time() - t0_epoch, 1)
    summary["loop_hz"] = round(stats["loops"] / summary["duration_s"], 2) if summary["duration_s"] else None
    summary["load"] = dict(load_stats)
    irq_end = i2c_irq_count()
    if irq_start and irq_end:
        summary["i2c_irq"] = {**irq_end, "count": irq_end["count"] - irq_start["count"]}
    kernel = journal(t0_epoch, "-k")
    (out_dir / "kernel.log").write_text(kernel)
    summary["kernel_i2c_lines"] = sum("i2c" in l.lower() for l in kernel.splitlines())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
