"""Bench for the ~1 s holes in the angle stream: under which load do they come?

Reads the two angle sensors the way the daemon does (same AngleCapture code,
100 Hz) for a long time, with or without the RPi camera running next to it,
and logs every slow or failed I2C read. Holes come about once per 20 min of
recording, so give each condition about an hour (--duration, default 3600 s).

Run on the Pi with the grabette venv, daemon STOPPED for the first three modes
(it holds the camera and reads the sensors itself):
    sudo systemctl stop grabette
    .venv/bin/python scripts/bench_i2c_angles.py --mode i2c       # A: I2C alone
    .venv/bin/python scripts/bench_i2c_angles.py --mode camera    # B: + camera, no encoder
    .venv/bin/python scripts/bench_i2c_angles.py --mode record    # C: + H.264 recording, as in an episode

Mode `episodes` (D) drives the RUNNING daemon instead: it starts and stops
episodes through its API and reads the daemon's own angle warnings from the
journal afterwards. Select a throwaway task in the dashboard first (episodes
pile up there), and run it once per branch to compare (develop vs this one):
    sudo systemctl start grabette
    .venv/bin/python scripts/bench_i2c_angles.py --mode episodes

Writes ~/i2c_bench/<UTC start>_<mode>/:
  events.csv    every read over --slow-ms or failed, and every loop gap over
                100 ms (UTC, CLOCK_BOOTTIME, sensor, duration, errno)
  summary.json  reads, errors by errno, slow reads, max read, shared-IRQ count
  kernel.log    kernel messages of the run (i2c-bcm2835 "unexpected interrupt")
  daemon.log    mode episodes only: the daemon's angle lines

Ctrl-C ends the run early and still writes the summary.
"""

import argparse
import csv
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLOCK_BOOTTIME = getattr(time, "CLOCK_BOOTTIME", 7)
SENSORS = ("distal", "proximal")
LOOP_GAP_S = 0.1  # same threshold as the daemon's "Angle loop stalled"
REPORT_EVERY_S = 60

stop = threading.Event()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%H:%M:%S} {msg}", flush=True)


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


# ── Camera load (modes camera and record) ─────────────────────────────


def camera_only(stats: Counter) -> None:
    try:
        _camera_only()
    except Exception as e:
        # Without the camera this run would be mode i2c under another name.
        stats["camera_failed"] += 1
        log(f"camera failed, stopping the run: {type(e).__name__}: {e}")
        stop.set()


def _camera_only() -> None:
    """The daemon's camera configuration, streaming, nothing encoded or written."""
    from picamera2 import Picamera2

    from grabette.hardware.camera import CameraCapture as C

    picam2 = Picamera2()
    duration = int(1_000_000 / C.DEFAULT_FPS)
    picam2.configure(picam2.create_video_configuration(
        main={"size": C.DEFAULT_RESOLUTION, "format": "YUV420"},
        controls={"FrameDurationLimits": (duration, duration)},
        buffer_count=C.DEFAULT_BUFFER_COUNT, queue=False))
    picam2.start()
    log("camera streaming")
    try:
        stop.wait()
    finally:
        picam2.stop()
        picam2.close()


def recording(sync, out_dir: Path, segment_s: float, pause_s: float, stats: Counter) -> None:
    """Back-to-back recordings through the daemon's CameraCapture (writer
    thread, mux, ffprobe), deleted after each one."""
    from grabette.hardware.camera import CameraCapture

    seg_dir = out_dir / "segments"
    seg_dir.mkdir()
    n = 0
    while not stop.is_set():
        n += 1
        cam = CameraCapture(sync)
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


# ── I2C reads (modes i2c, camera, record) ─────────────────────────────


def read_loop(cap, rate_hz: float, slow_s: float, writer, stats: dict) -> None:
    period = 1.0 / rate_hz
    buses = (cap.i2c_bus_1, cap.i2c_bus_2)
    prev = None
    last_report = time.monotonic()
    while not stop.is_set():
        start = time.monotonic()
        if prev is not None and start - prev > LOOP_GAP_S:
            stats["loop_gaps"] += 1
            writer.writerow([utc_now(), f"{time.clock_gettime(CLOCK_BOOTTIME):.6f}",
                             "loop", "", f"{(start - prev) * 1000:.1f}", "gap", ""])
        prev = start
        for i, i2c in enumerate((cap._i2c_1, cap._i2c_2)):
            r = cap._timed_read(i2c)
            s = stats["sensors"][SENSORS[i]]
            s["reads"] += 1
            s["max_read_ms"] = max(s["max_read_ms"], r.seconds * 1000)
            if r.seconds > 0.1:
                s["reads_over_100ms"] += 1
            if r.seconds > 0.5:
                s["reads_over_500ms"] += 1
            if r.error is not None:
                s["errors"][r.error] += 1
            if r.error is not None or r.seconds > slow_s:
                writer.writerow([utc_now(), f"{time.clock_gettime(CLOCK_BOOTTIME):.6f}",
                                 SENSORS[i], buses[i], f"{r.seconds * 1000:.1f}",
                                 r.error or "slow", r.detail])
                if r.error is not None or r.seconds > 0.5:
                    log(f"{SENSORS[i]} /dev/i2c-{buses[i]}: {r.seconds * 1000:.0f} ms "
                        f"{r.error or ''} {r.detail}")
        if start - last_report > REPORT_EVERY_S:
            last_report = start
            log(" | ".join(f"{n}: {s['reads']} reads, {sum(s['errors'].values())} err, "
                           f"{s['reads_over_500ms']} >500 ms, max {s['max_read_ms']:.0f} ms"
                           for n, s in stats["sensors"].items()))
        sleep = period - (time.monotonic() - start)
        if sleep > 0:
            time.sleep(sleep)


# ── Episodes through the daemon (mode episodes) ───────────────────────


def post(base: str, path: str) -> dict:
    req = urllib.request.Request(base + path, data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read() or b"{}")


def episodes_loop(base: str, segment_s: float, pause_s: float, stats: Counter) -> None:
    while not stop.is_set():
        try:
            started = post(base, "/api/episodes/start")
            if started.get("status") != "capturing":
                log(f"start did not capture ({started}); is this device in a group?")
                stats["episodes_failed"] += 1
                stop.wait(10)
                continue
            log(f"episode {started.get('episode_id')}")
            stop.wait(segment_s)
            post(base, "/api/episodes/stop")
            stats["episodes_ok"] += 1
        except (urllib.error.URLError, OSError, ValueError) as e:
            stats["episodes_failed"] += 1
            detail = e.read().decode(errors="replace") if isinstance(e, urllib.error.HTTPError) else ""
            log(f"episode failed: {e} {detail}")
            try:
                post(base, "/api/episodes/stop")
            except Exception:
                pass
        stop.wait(pause_s)


def daemon_summary(text: str) -> dict:
    lines = text.splitlines()
    return {"angle_loop_stalls": sum("Angle loop stalled" in l for l in lines),
            "failed_reads_logged": sum("read failed" in l for l in lines),
            "capture_summaries": [l.split("Angle: ", 1)[1] for l in lines
                                  if "Angle: " in l and " reads, " in l]}


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", choices=("i2c", "camera", "record", "episodes"), required=True)
    p.add_argument("--duration", type=float, default=3600, help="seconds (default 3600)")
    p.add_argument("--rate", type=float, default=100, help="I2C read loop, Hz (default 100)")
    p.add_argument("--slow-ms", type=float, default=50,
                   help="log reads slower than this (default 50; normal is 1-15 ms)")
    p.add_argument("--segment", type=float, default=120,
                   help="record/episodes: length of each recording, s (default 120)")
    p.add_argument("--pause", type=float, default=5,
                   help="record/episodes: pause between recordings, s (default 5)")
    p.add_argument("--api", default="http://127.0.0.1:8000", help="episodes: daemon URL")
    p.add_argument("--out", type=Path, default=Path.home() / "i2c_bench")
    p.add_argument("--allow-daemon", action="store_true",
                   help="run i2c/camera/record even though the daemon is active")
    args = p.parse_args()

    if args.mode == "episodes" and not daemon_active():
        sys.exit("mode episodes drives the daemon: sudo systemctl start grabette")
    if args.mode != "episodes" and daemon_active() and not args.allow_daemon:
        sys.exit("the daemon holds the camera and reads the sensors: "
                 "sudo systemctl stop grabette (or --allow-daemon)")

    t0_epoch = time.time()
    out_dir = args.out / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{args.mode}"
    out_dir.mkdir(parents=True)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    timer = threading.Timer(args.duration, stop.set)
    timer.daemon = True
    timer.start()

    irq_start = i2c_irq_count()
    summary: dict = {"mode": args.mode, "start_utc": utc_now(), "args": {
        k: str(v) for k, v in vars(args).items()}}
    load_stats: Counter = Counter()
    log(f"mode {args.mode} for {args.duration:.0f} s -> {out_dir}")

    if args.mode == "episodes":
        episodes_loop(args.api.rstrip("/"), args.segment, args.pause, load_stats)
    else:
        from grabette.hardware.angle import AngleCapture
        from grabette.hardware.sync import SyncManager

        sync = SyncManager()
        sync.start()
        cap = AngleCapture(sync, sample_rate_hz=int(args.rate))
        cap.init_sensors()
        load = None
        if args.mode == "camera":
            load = threading.Thread(target=camera_only, args=(load_stats,), daemon=True)
        elif args.mode == "record":
            load = threading.Thread(target=recording, daemon=True,
                                    args=(sync, out_dir, args.segment, args.pause, load_stats))
        if load is not None:
            load.start()
        stats = {"loop_gaps": 0, "sensors": {n: {
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
    summary["load"] = dict(load_stats)
    irq_end = i2c_irq_count()
    if irq_start and irq_end:
        summary["i2c_irq"] = {**irq_end, "count": irq_end["count"] - irq_start["count"]}
    kernel = journal(t0_epoch, "-k")
    (out_dir / "kernel.log").write_text(kernel)
    summary["kernel_i2c_lines"] = sum("i2c" in l.lower() for l in kernel.splitlines())
    if args.mode == "episodes":
        daemon = journal(t0_epoch, "-u", "grabette")
        angle_lines = [l for l in daemon.splitlines()
                       if "Angle" in l or "i2c" in l.lower()]
        (out_dir / "daemon.log").write_text("\n".join(angle_lines) + "\n")
        summary["daemon"] = daemon_summary(daemon)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
