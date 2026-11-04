"""Find the holes in an episode's streams and what the Pi was doing during them.

Runs anywhere (Pi or workstation), stdlib only:
    python3 scripts/analyze_frame_gaps.py <episode or folder>...
    python3 scripts/analyze_frame_gaps.py ~/Downloads/sync-review-* --monitor ~/capture_monitor/20261105_101500
    python3 scripts/analyze_frame_gaps.py /data/episodes --journal      # on the Pi

For every episode found (a folder with frame_timestamps.json) it reports:
  1. the holes of each stream: RPi camera, depth camera (left IR), angles;
  2. for each RPi camera hole, whether the depth camera and the angles kept
     going. If they did, the daemon was alive and only the camera chain
     stalled; if everything stopped, the whole process or Pi froze;
  3. with raw_video.frames.json (PR 239), whether the hole is in the sensor
     timestamps of the frames the encoder received, i.e. the frames never
     reached the encoder (dropped by libcamera for want of a free buffer);
  4. with --monitor (output of monitor_capture.py), the Pi's load during the
     hole next to its usual level over the episode, and the kernel messages;
  5. with --journal (on the Pi), the daemon's log lines during the hole.

Placing a hole on the monitor's clock needs the episode's t=0: exact from
raw_video.frames.json (sensor timestamps are CLOCK_BOOTTIME), else from
metadata sync.capture_started_utc (group starts), else (--journal) from the
"capture started" log line, which comes ~0.1 s after t=0.
"""

import argparse
import bisect
import csv
import json
import statistics
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Context shown around a hole, and the daemon's detection threshold.
MARGIN_MS = 300.0
DAEMON_THRESHOLD_MS = 250.0


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def streams_of(ep: Path) -> dict[str, list[float]]:
    """Per-stream timestamps (ms on the episode's t=0 timeline)."""
    out = {}
    rgb = load_json(ep / "frame_timestamps.json")
    if isinstance(rgb, list):
        out["rgb"] = [float(t) for t in rgb]
    for name in ("dcam_left_timestamps.json", "oakd_left_timestamps.json"):
        d = load_json(ep / name)
        if d and d.get("samples"):
            out["depth"] = [s["host_ms"] for s in d["samples"]]
            break
    ang = load_json(ep / "angle_data.json")
    if ang and ang.get("samples"):
        out["angle"] = [s["cts"] for s in ang["samples"]]
    return out


def holes(ts: list[float], threshold: float) -> list[tuple[float, float]]:
    return [(a, b) for a, b in zip(ts, ts[1:]) if b - a > threshold]


def worst_gap(ts: list[float], a: float, b: float) -> float | None:
    """Largest interval of `ts` overlapping [a, b]."""
    i = max(bisect.bisect_left(ts, a) - 1, 0)
    j = min(bisect.bisect_right(ts, b) + 1, len(ts))
    seg = ts[i:j]
    return max((y - x for x, y in zip(seg, seg[1:])), default=None)


# ── t=0 on an absolute clock ────────────────────────────────────────────

def t0_boottime(ep: Path) -> float | None:
    """CLOCK_BOOTTIME (s) of t=0, from PR 239's per-frame sensor timestamps."""
    d = load_json(ep / "raw_video.frames.json")
    frames = (d or {}).get("frames") or []
    vals = [f["sensor_timestamp_ns"] / 1e9 - f["timestamp_ms"] / 1000 for f in frames[:50]
            if f.get("sensor_timestamp_ns") is not None and f.get("timestamp_ms") is not None]
    return statistics.median(vals) if vals else None


def t0_utc(ep: Path, meta: dict, use_journal: bool) -> tuple[datetime | None, str]:
    s = (meta.get("sync") or {}).get("capture_started_utc")
    if s:
        return datetime.fromisoformat(s), "metadata sync.capture_started_utc"
    if use_journal:
        out = journal(["--grep", f"capture started.*{ep.name}", "-n", "1"])
        for line in out.splitlines():
            ts = parse_iso(line.split(" ", 1)[0])
            if ts:
                return ts, "journal 'capture started' line (≈ t=0 + 0.1 s)"
    return None, ""


def parse_iso(s: str) -> datetime | None:
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def journal(args: list[str]) -> str:
    try:
        return subprocess.run(["journalctl", "--no-pager", "-o", "short-iso-precise", *args],
                              capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


# ── monitor_capture.py output ───────────────────────────────────────────

class Monitor:
    NUM = ("cpu_busy", "iowait", "psi_cpu_some_ms", "psi_io_some_ms", "psi_io_full_ms",
           "psi_memory_some_ms", "dirty_kb", "writeback_kb", "disk_write_kb",
           "disk_write_ms", "disk_inflight")

    def __init__(self, folder: Path):
        csv_path = folder if folder.is_file() else folder / "monitor.csv"
        self.rows = []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                r["boottime_s"] = float(r["boottime_s"])
                r["utc"] = parse_iso(r["utc"])
                for k in self.NUM:
                    r[k] = float(r[k] or 0)
                self.rows.append(r)
        self.boot = [r["boottime_s"] for r in self.rows]
        self.kernel = []
        klog = csv_path.parent / "kernel.log"
        if klog.exists():
            for line in klog.read_text(errors="replace").splitlines():
                ts = parse_iso(line.split(" ", 1)[0])
                if ts:
                    self.kernel.append((ts, line))
        # Offset boottime → UTC, to place a UTC-anchored episode.
        self.utc_minus_boot = statistics.median(
            r["utc"].timestamp() - r["boottime_s"] for r in self.rows[:: max(len(self.rows) // 200, 1)]
        ) if self.rows else 0.0

    def window(self, b0: float, b1: float) -> list[dict]:
        return self.rows[bisect.bisect_left(self.boot, b0):bisect.bisect_right(self.boot, b1)]

    @staticmethod
    def summary(rows: list[dict]) -> dict:
        if not rows:
            return {}
        s = {k: max(r[k] for r in rows) for k in Monitor.NUM}
        s["disk_write_kb"] = sum(r["disk_write_kb"] for r in rows)
        s["n"] = len(rows)
        s["d_threads"] = sorted({t for r in rows for t in r["d_threads"].split("|") if t})
        s["throttled"] = sorted({r["throttled"] for r in rows if r["throttled"]})
        s["temp"] = sorted({r["temp"] for r in rows if r["temp"]})
        return s


def chance(ep_rows: list[dict], n: int, test) -> float | None:
    """Fraction of the episode's n-sample windows for which test(window) holds:
    how often the hole's reading turns up by chance, anywhere in the take."""
    wins = [ep_rows[i:i + n] for i in range(0, max(len(ep_rows) - n + 1, 0))]
    return sum(map(test, wins)) / len(wins) if wins else None


def print_monitor(mon: Monitor, b_start: float, b_end: float, a: float, b: float, t0b: float):
    hole = mon.window(t0b + (a - MARGIN_MS) / 1000, t0b + (b + MARGIN_MS) / 1000)
    if not hole:
        print("      monitor: no sample at that time (was it running?)")
        return
    s = mon.summary(hole)
    ep_rows = mon.window(b_start, b_end)
    n = len(hole)
    print(f"      monitor ({n} samples, hole ±{MARGIN_MS:.0f} ms)    max in hole | episode mean | "
          f"same max elsewhere")
    labels = [("psi_io_full_ms", "I/O stall, all tasks (ms/sample)"),
              ("psi_io_some_ms", "I/O stall, some task (ms/sample)"),
              ("psi_cpu_some_ms", "CPU stall (ms/sample)"),
              ("psi_memory_some_ms", "memory stall (ms/sample)"),
              ("cpu_busy", "CPU busy %"), ("iowait", "iowait %"),
              ("dirty_kb", "dirty page cache kB"), ("writeback_kb", "writeback kB"),
              ("disk_write_ms", "disk busy writing (ms/sample)"),
              ("disk_inflight", "disk requests in flight")]
    for k, label in labels:
        mean = statistics.fmean(r[k] for r in ep_rows) if ep_rows else 0.0
        if s[k] > 0:
            c = chance(ep_rows, n, lambda w, k=k, m=s[k]: max(r[k] for r in w) >= m)
            # Rarely reached elsewhere → the hole's reading is unusual.
            cs = "-" if c is None else f"{100 * c:.0f}% of windows" + ("  <<<" if c < 0.1 else "")
        else:
            cs = ""
        print(f"        {label:36s} {s[k]:9.1f} | {mean:12.1f} | {cs}")
    c = chance(ep_rows, n, lambda w: sum(r["disk_write_kb"] for r in w) >= s["disk_write_kb"])
    print(f"        {'kB written to disk (sum)':36s} {s['disk_write_kb']:9.0f} | {'':12s} | "
          + ("" if c is None or not s["disk_write_kb"] else f"{100 * c:.0f}% of windows"
             + ("  <<<" if c < 0.1 else "")))
    for t in s["d_threads"]:
        c = chance(ep_rows, n, lambda w, t=t: any(t in r["d_threads"].split("|") for r in w))
        share = sum(t in r["d_threads"].split("|") for r in hole) / n
        print(f"        blocked (D) thread {t!r}: {100 * share:.0f}% of hole samples, "
              f"present in {'-' if c is None else f'{100 * c:.0f}%'} of episode windows")
    if s["throttled"] and s["throttled"] != ["0x0"]:
        print(f"        throttled: {s['throttled']}  temp: {s['temp']}")
    u0 = datetime.fromtimestamp(t0b + mon.utc_minus_boot + (a - MARGIN_MS) / 1000, timezone.utc)
    u1 = datetime.fromtimestamp(t0b + mon.utc_minus_boot + (b + MARGIN_MS) / 1000, timezone.utc)
    for ts, line in mon.kernel:
        if u0 <= ts <= u1:
            print(f"        kernel: {line}")


# ── report ──────────────────────────────────────────────────────────────

def episode_time(ep: Path) -> datetime | None:
    """Creation time from the episode id (YYYYMMDD_HHMMSS in UTC, see
    grabette.task.episode_id_for). The parent folder holds it for a
    left/right download layout."""
    for name in (ep.name, ep.parent.name):
        try:
            return datetime.strptime(name, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def outside_monitor(ep: Path, mon: Monitor) -> bool:
    """True when the episode was recorded while the monitor wasn't running.
    By wall clock: CLOCK_BOOTTIME restarts at every reboot, so an older
    episode's sensor times could fall in the monitor's range by coincidence."""
    t = episode_time(ep)
    if t is None or not mon.rows:
        return False
    margin = timedelta(minutes=2)
    return not (mon.rows[0]["utc"] - margin <= t <= mon.rows[-1]["utc"] + margin)


def analyze(ep: Path, args, mon: Monitor | None) -> list[float] | None:
    """RGB hole sizes; None when skipped (outside the monitor's session)."""
    if mon is not None and outside_monitor(ep, mon):
        return None
    meta = load_json(ep / "metadata.json") or {}
    st = streams_of(ep)
    rgb = st.get("rgb") or []
    if len(rgb) < 2:
        return []
    step = statistics.median(b - a for a, b in zip(rgb, rgb[1:]))
    all_holes = {name: holes(ts, args.threshold) for name, ts in st.items()}
    flagged = meta.get("frame_drops")
    align = meta.get("video_timestamp_alignment")
    found = any(all_holes.values())
    # With a monitor, every episode of its session is listed, hole or not.
    if not found and not args.all and mon is None:
        return []

    peak = (align or {}).get("writer_queue_peak")
    print(f"\n{ep}  ({meta.get('duration_seconds', '?')} s, rgb every {step:.1f} ms"
          f"{', frame_drops=' + str(flagged) if flagged else ''}"
          f"{', alignment valid=' + str(align.get('valid')) if align else ''}"
          f"{', writer_queue_peak=' + str(peak) if peak is not None else ''})")
    if not found:
        print("    no hole")
        return []

    # Absolute anchoring for the monitor / journal.
    t0b = t0_boottime(ep)
    t0u, how = (None, "")
    if t0b is not None:
        how = "raw_video.frames.json sensor timestamps (exact)"
    else:
        t0u, how = t0_utc(ep, meta, args.journal)
        if t0u is not None and mon is not None:
            t0b = t0u.timestamp() - mon.utc_minus_boot
    if t0u is None and t0b is not None and mon is not None:
        t0u = datetime.fromtimestamp(t0b + mon.utc_minus_boot, timezone.utc)
    if (mon or args.journal) and how:
        print(f"    t=0 from {how}")
    elif mon or args.journal:
        print("    t=0 unknown: cannot place the holes on the monitor/journal clock "
              "(needs PR 239, a group start, or --journal on the Pi)")

    for name, hs in all_holes.items():
        if name != "rgb":
            for a, b in hs:
                print(f"    {name} hole {b - a:.0f} ms at {a / 1000:.2f}-{b / 1000:.2f} s")

    sizes = []
    for a, b in all_holes.get("rgb", []):
        sizes.append(b - a)
        pos = a / rgb[-1] if rgb[-1] else 0
        where = "start" if pos < 0.1 else "end" if pos > 0.9 else "middle"
        mark = " (flagged by daemon)" if b - a > DAEMON_THRESHOLD_MS else ""
        print(f"  ▶ rgb hole {b - a:.0f} ms at {a / 1000:.2f}-{b / 1000:.2f} s ({where}), "
              f"≈{round((b - a) / step) - 1} frames lost{mark}")
        for other in ("depth", "angle"):
            ts = st.get(other)
            if not ts:
                continue
            g = worst_gap(ts, a, b)
            inside = sum(a < t < b for t in ts)
            verdict = "kept going" if g is not None and g < (b - a) / 2 else "STOPPED TOO"
            print(f"      {other:5s}: {verdict:11s} (largest interval {g:.0f} ms, {inside} samples in the hole)")
        if align is not None:
            d = load_json(ep / "raw_video.frames.json") or {}
            frames = d.get("frames") or []
            k = next((i for i, f in enumerate(frames) if f.get("timestamp_ms", 0) > a), None)
            if k:
                f0, f1 = frames[k - 1], frames[k]
                pts = (f1["encoder_pts_us"] - f0["encoder_pts_us"]) / 1000
                print(f"      encoder: frames {k - 1}->{k} consecutive in the video; sensor "
                      f"timestamps {(f1['sensor_timestamp_ns'] - f0['sensor_timestamp_ns']) / 1e6:.0f} ms "
                      f"apart, PTS {pts:.0f} ms apart → the lost frames never reached the encoder")
        if mon is not None and t0b is not None:
            print_monitor(mon, t0b, t0b + rgb[-1] / 1000, a, b, t0b)
        if args.journal and t0u is not None:
            since = t0u + timedelta(milliseconds=a - 2000)
            until = t0u + timedelta(milliseconds=b + 1000)
            fmt = "%Y-%m-%d %H:%M:%S"
            out = journal(["--utc", "--since", since.strftime(fmt) + " UTC",
                           "--until", until.strftime(fmt) + " UTC"])
            # Dashboard/API access lines are just noise here.
            lines = [l for l in out.splitlines()
                     if l and not l.startswith("--") and "HTTP/1.1" not in l]
            print(f"      journal {since:%H:%M:%S}-{until:%H:%M:%S} UTC: {len(lines)} lines")
            for l in lines[-40:]:
                print(f"        {l}")
    return sizes


def find_episodes(paths: list[Path]) -> list[Path]:
    eps = []
    for p in paths:
        if not p.exists():
            # An unmatched shell glob arrives here verbatim.
            print(f"warning: {p} does not exist")
            continue
        if (p / "frame_timestamps.json").exists():
            eps.append(p)
        else:
            found = sorted(f.parent for f in p.rglob("frame_timestamps.json"))
            if not found:
                print(f"warning: no episode (frame_timestamps.json) under {p}")
            eps += found
    return eps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="episode folders or folders containing them")
    p.add_argument("--threshold", type=float, default=150.0,
                   help="report holes longer than this, ms (default 150; the daemon flags >250)")
    p.add_argument("--monitor", type=Path, help="monitor_capture.py output folder or monitor.csv")
    p.add_argument("--journal", action="store_true", help="read the daemon log (on the Pi)")
    p.add_argument("--all", action="store_true", help="also list episodes without holes")
    args = p.parse_args()

    mon = Monitor(args.monitor) if args.monitor else None
    eps = find_episodes(args.paths)
    sizes, with_hole, skipped = [], 0, 0
    for ep in eps:
        s = analyze(ep, args, mon)
        if s is None:
            skipped += 1
            continue
        sizes += s
        with_hole += bool(s)
    if skipped:
        print(f"\n{skipped} episode streams skipped: recorded while the monitor wasn't running")
    print(f"\n{len(eps) - skipped} episode streams, {with_hole} with an rgb hole > {args.threshold:.0f} ms, "
          f"{sum(x > DAEMON_THRESHOLD_MS for x in sizes)} holes > {DAEMON_THRESHOLD_MS:.0f} ms"
          + (f" (sizes: {', '.join(f'{x:.0f}' for x in sorted(sizes))} ms)" if sizes else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
