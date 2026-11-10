"""Measure a set of recorded episodes: every check of the Synchro Grabette review.

Stdlib only; runs on the Pi or on a workstation, on episodes laid out any way:
    ~/grabette-data/episodes/<id>/            (on the Pi)
    <download>/<id>/                          (single-device download)
    <download>/<id>/<left|right|...>/         (multi-device raw download)

    python3 benchmarks/episode_report.py ~/grabette-data/episodes --run ~/benchmarks/<run> \
        --label "develop @ 1707d4e" --out report-develop.json
    python3 benchmarks/episode_report.py ~/Downloads/<session>-rawdl-<...> \
        --chrono ~/Downloads/<session>-rawdl-<...>/chrono_sync/results.json --out report.json

Then compare two reports with benchmarks/compare_reports.py.

Per episode and stream:
  rgb     holes > max(30 ms, 1.5 x median interval), split into the start-up
          burst (first 1.5 s) and the rest; holes > 150 / 250 ms (the daemon
          flags > 250 ms); frames lost; frame_timestamps.json count vs
          metadata frame_count; video_timestamp_alignment (valid, unwritten
          camera frames, reordered pictures, writer_queue_peak, how the PTS
          offset was found and how many offsets matching alone accepted)
          when present.
  depth   holes > max(100 ms, 1.5 x median); frames the camera numbered but
          the host never received (seq jumps); dropped_frames (encoder queue);
          IR / depth frame counts.
  angle   loop rate (Hz); holes > 100 ms (loop stalls) and > 500 ms (the ~1 s
          I2C timeouts).
  start   start_capture_ms, wake_skew_ms, capture_started_utc - scheduled
          (sync block), first sample of each stream relative to t=0.
  end     last sample of each stream: spread between streams, and streams
          that stopped more than 0.5 s before the others.
  decode  with --decode: frames ffprobe actually decodes in raw_video.mp4,
          dcam_left.mp4 and dcam_depth.mkv, against their timestamp counts
          (an out-of-order picture is silently dropped by the decoder).
Across devices (same episode id on several devices): spread of
capture_started_utc, i.e. the t=0 gap the alignment would assume.
Optional inputs: --run (a run_episodes.py folder: selects its episodes,
counts the daemon / kernel log lines, the daemon's stop phases, how long each
stop call took, and the Pi's clock synchronisation), --chrono (measure_chrono_sync.py
results: RPi-Orbbec offset and drift, real t=0 gap, capture_started error),
--i2c (bench_i2c_angles.py summaries).
"""

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

STARTUP_MS = 1500.0          # the start-up burst of small RGB losses
RGB_HOLE_MS = 30.0           # >= one frame lost at ~49 fps
DAEMON_FLAG_MS = 250.0       # MAX_FRAME_GAP_MS: the daemon flags frame_drops
DEPTH_HOLE_MS = 100.0
ANGLE_STALL_MS = 100.0       # the daemon's "Angle loop stalled" threshold
ANGLE_TIMEOUT_MS = 500.0     # the ~1 s I2C timeouts
LATE_START_MS = 1000.0       # start_capture_ms above this: a visibly late start
EARLY_STOP_MS = 500.0        # a stream whose last sample is this much before the others'
WRITER_QUEUE_FULL = 300
CLOCK_OFFSET_MS = 5.0        # NTP offset above this: capture_started_utc is off by as much

# Daemon / kernel log lines worth counting, by kind (substring match).
LOG_PATTERNS = {
    "angle_loop_stalled": "Angle loop stalled",
    "angle_read_failed": "read failed on /dev/i2c",
    "frame_hole_flagged": "hole between two frames",
    "depth_not_ready": "not ready after",
    "alignment_error": "FrameAlignmentError",
    "traceback": "Traceback",
}
KERNEL_PATTERNS = {
    "i2c_unexpected_interrupt": "Got unexpected interrupt",
    "i2c_other": "i2c",
    "usb_reset": "reset high-speed USB",
    "usb_disconnect": "USB disconnect",
    "mmc_error": "mmc0: ",
    "under_voltage": "Under-voltage",
}
# "RpiBackend capture stopped [ms: angle_stop=12 camera_stop=1650 ... total_awaited=2100]"
STOP_LINE = re.compile(r"capture stopped \[ms: ([^\]]*?)\s+total_awaited=([\d.]+)\]")


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def median_step(ts: list[float]) -> float | None:
    return statistics.median(b - a for a, b in zip(ts, ts[1:])) if len(ts) > 1 else None


def gaps(ts: list[float], threshold: float) -> list[tuple[float, float]]:
    return [(a, b) for a, b in zip(ts, ts[1:]) if b - a > threshold]


def episode_time(name: str) -> datetime | None:
    try:
        return datetime.strptime(name, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_utc(s) -> datetime | None:
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# ── Finding episodes ───────────────────────────────────────────────────

def find_episodes(paths: list[Path]) -> list[Path]:
    """Folders holding one device's take of an episode."""
    out = []
    for p in paths:
        if not p.exists():
            print(f"warning: {p} does not exist", file=sys.stderr)
            continue
        if (p / "frame_timestamps.json").exists():
            out.append(p)
        else:
            out += sorted(f.parent for f in p.rglob("frame_timestamps.json"))
    return out


def episode_id(folder: Path) -> str:
    """The episode id: the folder's name, or its parent's in a multi-device
    download (<id>/<left|right>/)."""
    for name in (folder.name, folder.parent.name):
        if episode_time(name):
            return name
    return folder.name


# ── Per-episode measures ───────────────────────────────────────────────

def rgb_measures(ts: list[float], meta: dict) -> dict:
    step = median_step(ts)
    if step is None:
        return {"frames": len(ts)}
    thr = max(RGB_HOLE_MS, 1.5 * step)
    holes = gaps(ts, thr)
    start = ts[0]
    after = [(a, b) for a, b in holes if a - start >= STARTUP_MS]
    span_s = (ts[-1] - ts[0]) / 1000
    out = {
        "frames": len(ts),
        "fps": round((len(ts) - 1) / span_s, 2) if span_s > 0 else None,
        "first_ms": round(ts[0], 1),
        "last_ms": round(ts[-1], 1),
        "holes_startup": len(holes) - len(after),
        "holes_after_startup": len(after),
        "frames_lost_after_startup": sum(max(round((b - a) / step) - 1, 1) for a, b in after),
        "holes_over_150ms": sum(b - a > 150 for a, b in holes),
        "holes_over_250ms": sum(b - a > DAEMON_FLAG_MS for a, b in holes),
        "max_gap_ms": round(max(b - a for a, b in zip(ts, ts[1:])), 1),
        "frame_count_matches": meta.get("frame_count") in (None, len(ts)),
    }
    align = meta.get("video_timestamp_alignment")
    if isinstance(align, dict):
        out["alignment_valid"] = align.get("valid")
        for k in ("unwritten_camera_frame_count", "reordered_output_count",
                  "writer_queue_peak", "camera_buffer_count",
                  "pts_offset_source", "pts_offset_candidates"):
            if k in align:
                out[k] = align[k]
    return out


def depth_measures(ep: Path, meta: dict) -> dict | None:
    for name in ("dcam_left_timestamps.json", "oakd_left_timestamps.json"):
        samples = (load_json(ep / name) or {}).get("samples")
        if samples:
            break
    else:
        return None
    ts = [s["host_ms"] for s in samples]
    step = median_step(ts) or 33.3
    holes = gaps(ts, max(DEPTH_HOLE_MS, 1.5 * step))
    seqs = [s["seq"] for s in samples if "seq" in s]
    dcam = meta.get("dcam") or {}
    out = {
        "frames": len(ts),
        "first_ms": round(ts[0], 1),
        "last_ms": round(ts[-1], 1),
        "holes": len(holes),
        "max_gap_ms": round(max((b - a for a, b in zip(ts, ts[1:])), default=0), 1),
        # Numbered by the camera but never received by the host.
        "seq_lost": sum(max(b - a - 1, 0) for a, b in zip(seqs, seqs[1:])) if len(seqs) == len(ts) else None,
        "dropped_frames": dcam.get("dropped_frames"),
    }
    if "left_frames" in dcam and "depth_frames" in dcam:
        out["ir_depth_counts_match"] = dcam["left_frames"] == dcam["depth_frames"]
    return out


def angle_measures(ep: Path) -> dict | None:
    samples = (load_json(ep / "angle_data.json") or {}).get("samples")
    if not samples or len(samples) < 2:
        return None
    ts = [s["cts"] for s in samples]
    span_s = (ts[-1] - ts[0]) / 1000
    return {
        "samples": len(ts),
        "rate_hz": round((len(ts) - 1) / span_s, 2) if span_s > 0 else None,
        "first_ms": round(ts[0], 1),
        "last_ms": round(ts[-1], 1),
        "holes_over_100ms": len(gaps(ts, ANGLE_STALL_MS)),
        "holes_over_500ms": len(gaps(ts, ANGLE_TIMEOUT_MS)),
        "max_gap_ms": round(max(b - a for a, b in zip(ts, ts[1:])), 1),
    }


def start_measures(meta: dict) -> dict:
    sync = meta.get("sync") or {}
    out = {k: sync[k] for k in ("start_capture_ms", "wake_skew_ms") if k in sync}
    sched, started = parse_utc(sync.get("scheduled_start_utc")), parse_utc(sync.get("capture_started_utc"))
    if sched and started:
        out["started_minus_scheduled_ms"] = round((started - sched).total_seconds() * 1000, 1)
    out["capture_started_utc"] = sync.get("capture_started_utc")
    out["scheduled_start_utc"] = sync.get("scheduled_start_utc")
    return out


def end_measures(streams: dict) -> dict | None:
    """Do the streams stop together? Last sample of each, in ms after t=0."""
    last = {k: v["last_ms"] for k, v in streams.items() if v and v.get("last_ms") is not None}
    if len(last) < 2:
        return None
    latest = max(last.values())
    return {"last_ms": last,
            "spread_ms": round(latest - min(last.values()), 1),
            "stopped_early": sorted(k for k, v in last.items() if latest - v > EARLY_STOP_MS)}


def decoded_frames(path: Path) -> int | None:
    """Frames ffprobe decodes (not packets: an out-of-order picture is a
    packet the decoder drops without an error)."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=600)
        return int(r.stdout.strip().split(",")[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def decode_measures(ep: Path, n_rgb: int) -> dict:
    out = {}
    left = (load_json(ep / "dcam_left_timestamps.json") or {}).get("samples")
    depth = (load_json(ep / "dcam_depth_timestamps.json") or {}).get("samples")
    for key, name, expected in (("rgb", "raw_video.mp4", n_rgb),
                                ("depth_left", "dcam_left.mp4", len(left) if left else None),
                                ("depth", "dcam_depth.mkv", len(depth) if depth else None)):
        f = ep / name
        if expected is None or not f.exists():
            continue
        n = decoded_frames(f)
        out[key] = {"decoded": n, "timestamps": expected, "matches": n == expected if n is not None else None}
    return out


def measure_episode(ep: Path, decode: bool = False) -> dict:
    meta = load_json(ep / "metadata.json") or {}
    rgb_ts = load_json(ep / "frame_timestamps.json")
    rgb_ts = [float(t) for t in rgb_ts] if isinstance(rgb_ts, list) else []
    rgb = rgb_measures(rgb_ts, meta)
    duration = meta.get("duration_seconds")
    if duration is None and len(rgb_ts) > 1:
        duration = (rgb_ts[-1] - rgb_ts[0]) / 1000
    depth, angle = depth_measures(ep, meta), angle_measures(ep)
    return {
        "episode": episode_id(ep),
        "device": meta.get("device_id") or ep.name,
        "folder": str(ep),
        "duration_s": duration,
        "frame_drops": meta.get("frame_drops") or [],
        "rgb": rgb,
        "depth": depth,
        "angle": angle,
        "start": start_measures(meta),
        "end": end_measures({"rgb": rgb, "depth": depth, "angle": angle}),
        "decode": decode_measures(ep, len(rgb_ts)) if decode else None,
        "stop_call_s": None,  # filled from --run
    }


# ── Aggregates ─────────────────────────────────────────────────────────

def values(eps, *path):
    out = []
    for e in eps:
        v = e
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(v)
    return out


def summarize(eps: list[dict]) -> dict:
    """One device's (or all devices') figures, as compared between branches.

    Counts of rare events are kept with their exposure (minutes or episodes),
    so compare_reports.py can tell a real change from chance."""
    minutes = sum(e["duration_s"] or 0 for e in eps) / 60
    with_depth = [e for e in eps if e["depth"]]
    depth_min = sum(e["duration_s"] or 0 for e in with_depth) / 60
    with_angle = [e for e in eps if e["angle"]]
    angle_min = sum(e["duration_s"] or 0 for e in with_angle) / 60
    after_min = sum(max((e["duration_s"] or 0) - STARTUP_MS / 1000, 0) for e in eps) / 60

    def med(xs):
        return round(statistics.median(xs), 2) if xs else None

    def mx(xs):
        return round(max(xs), 1) if xs else None

    def mn(xs):
        return round(min(xs), 1) if xs else None

    aligned = [e for e in eps if "alignment_valid" in e["rgb"]]
    decoded = [e for e in eps if e.get("decode")]
    with_end = [e for e in eps if e.get("end")]
    s = {
        "episodes": len(eps),
        "minutes": round(minutes, 2),
        # Hard invariants: expected 0 on every branch.
        "invariants": {
            "alignment_invalid": sum(e["rgb"].get("alignment_valid") is False for e in aligned),
            "unwritten_camera_frames": sum(values(eps, "rgb", "unwritten_camera_frame_count")),
            "frame_count_mismatch": sum(e["rgb"].get("frame_count_matches") is False for e in eps),
            "writer_queue_full": sum(v >= WRITER_QUEUE_FULL for v in values(eps, "rgb", "writer_queue_peak")),
            "depth_dropped_frames": sum(values(eps, "depth", "dropped_frames")),
            "ir_depth_count_mismatch": sum((e["depth"] or {}).get("ir_depth_counts_match") is False for e in eps),
            # Only with --decode; "episodes_decoded" below says over how many.
            "decoded_frame_mismatch": sum(any(v.get("matches") is False for v in e["decode"].values())
                                          for e in decoded),
        },
        "episodes_decoded": len(decoded),
        "rgb": {
            "episodes_with_alignment": len(aligned),
            "holes_after_startup": {"count": sum(values(eps, "rgb", "holes_after_startup")),
                                    "minutes": round(after_min, 2)},
            "frames_lost_after_startup": sum(values(eps, "rgb", "frames_lost_after_startup")),
            "holes_startup_per_episode": med(values(eps, "rgb", "holes_startup")),
            "holes_over_150ms": {"count": sum(values(eps, "rgb", "holes_over_150ms")), "minutes": round(minutes, 2)},
            "holes_over_250ms": {"count": sum(values(eps, "rgb", "holes_over_250ms")), "minutes": round(minutes, 2)},
            "episodes_flagged_frame_drops": sum("rgb" in e["frame_drops"] for e in eps),
            "max_gap_ms": mx(values(eps, "rgb", "max_gap_ms")),
            "fps_median": med(values(eps, "rgb", "fps")),
            "first_frame_ms_median": med(values(eps, "rgb", "first_ms")),
            "first_frame_ms_max": mx(values(eps, "rgb", "first_ms")),
            "reordered_pictures": sum(values(eps, "rgb", "reordered_output_count")),
            "writer_queue_peak_max": mx(values(eps, "rgb", "writer_queue_peak")),
            # Since ad4eb7b (absent before): the PTS offset came from matching
            # alone, and episodes where matching alone was ambiguous (> 1).
            "pts_offset_matched": sum(e["rgb"].get("pts_offset_source") == "matched" for e in eps),
            "pts_offset_ambiguous": sum(v > 1 for v in values(eps, "rgb", "pts_offset_candidates")),
        },
        "depth": {
            "episodes": len(with_depth),
            "holes": {"count": sum(values(with_depth, "depth", "holes")), "minutes": round(depth_min, 2)},
            "seq_lost": {"count": sum(values(with_depth, "depth", "seq_lost")), "minutes": round(depth_min, 2)},
            "max_gap_ms": mx(values(with_depth, "depth", "max_gap_ms")),
            "first_frame_ms_abs_max": mx([abs(v) for v in values(with_depth, "depth", "first_ms")]),
        },
        "angle": {
            "episodes": len(with_angle),
            "rate_hz_median": med(values(with_angle, "angle", "rate_hz")),
            "rate_hz_min": mn(values(with_angle, "angle", "rate_hz")),
            "holes_over_100ms": {"count": sum(values(with_angle, "angle", "holes_over_100ms")),
                                 "minutes": round(angle_min, 2)},
            "holes_over_500ms": {"count": sum(values(with_angle, "angle", "holes_over_500ms")),
                                 "minutes": round(angle_min, 2)},
            "max_gap_ms": mx(values(with_angle, "angle", "max_gap_ms")),
        },
        "start": {
            "start_capture_ms_median": med(values(eps, "start", "start_capture_ms")),
            "start_capture_ms_max": mx(values(eps, "start", "start_capture_ms")),
            "late_starts": {"count": sum(v > LATE_START_MS for v in values(eps, "start", "start_capture_ms")),
                            "episodes": len(values(eps, "start", "start_capture_ms"))},
            "started_minus_scheduled_ms_median": med(values(eps, "start", "started_minus_scheduled_ms")),
            "started_minus_scheduled_ms_max": mx(values(eps, "start", "started_minus_scheduled_ms")),
            "wake_skew_ms_abs_max": mx([abs(v) for v in values(eps, "start", "wake_skew_ms")]),
        },
        "end": {
            "episodes": len(with_end),
            "spread_ms_median": med(values(with_end, "end", "spread_ms")),
            "spread_ms_max": mx(values(with_end, "end", "spread_ms")),
            "streams_stopped_early": {"count": sum(bool(e["end"]["stopped_early"]) for e in with_end),
                                      "episodes": len(with_end)},
        },
        "stop": {
            "stop_call_s_median": med(values(eps, "stop_call_s")),
            "stop_call_s_max": mx(values(eps, "stop_call_s")),
        },
    }
    return s


def group_measures(eps: list[dict]) -> list[dict]:
    """Episodes recorded by several devices: the t=0 gap capture_started_utc
    implies (measured for real only by the chronometer bench)."""
    by_id: dict[str, list[dict]] = {}
    for e in eps:
        by_id.setdefault(e["episode"], []).append(e)
    out = []
    for eid, members in sorted(by_id.items()):
        if len(members) < 2:
            continue
        starts = {m["device"]: parse_utc(m["start"].get("capture_started_utc")) for m in members}
        known = {d: t for d, t in starts.items() if t}
        sched = {m["start"].get("scheduled_start_utc") for m in members}
        row = {"episode": eid, "devices": sorted(starts), "same_scheduled_start": len(sched) == 1 and None not in sched}
        if len(known) >= 2:
            ts = sorted(known.values())
            row["capture_started_spread_ms"] = round((ts[-1] - ts[0]).total_seconds() * 1000, 1)
        out.append(row)
    return out


def chrono_measures(path: Path) -> dict | None:
    """Figures from measure_chrono_sync.py's results.json."""
    rows = load_json(path)
    if not isinstance(rows, list):
        return None
    rpi_orbbec, drift, gap_err, gaps_rgb = [], [], [], []
    for row in rows:
        for dev in (row.get("devices") or {}).values():
            if dev.get("rpi_minus_orbbec_ms") is not None:
                rpi_orbbec.append(dev["rpi_minus_orbbec_ms"])
            if dev.get("rpi_minus_orbbec_drift_ms") is not None:
                drift.append(dev["rpi_minus_orbbec_drift_ms"])
            for k, v in dev.items():
                if k.startswith("t0_gap_vs_") and isinstance(v, dict):
                    if v.get("capture_started_error_ms") is not None:
                        gap_err.append(v["capture_started_error_ms"])
                    if v.get("rgb_ms") is not None:
                        gaps_rgb.append(v["rgb_ms"])
    if not rpi_orbbec and not gap_err:
        return None
    r = lambda x: round(x, 2)
    return {
        "episodes": len(rows),
        "rpi_minus_orbbec_ms_median": r(statistics.median(rpi_orbbec)) if rpi_orbbec else None,
        "rpi_minus_orbbec_ms_spread": r(max(rpi_orbbec) - min(rpi_orbbec)) if rpi_orbbec else None,
        "rpi_minus_orbbec_drift_ms_max": r(max(drift)) if drift else None,
        "capture_started_error_ms_abs_max": r(max(map(abs, gap_err))) if gap_err else None,
        "capture_started_error_ms_median": r(statistics.median(gap_err)) if gap_err else None,
        "t0_gap_ms_abs_max": r(max(map(abs, gaps_rgb))) if gaps_rgb else None,
    }


def count_lines(text: str, patterns: dict[str, str]) -> dict[str, int]:
    lines = text.splitlines()
    return {k: sum(p in l for l in lines) for k, p in patterns.items()}


def stop_phases(text: str) -> dict | None:
    """The daemon's stop timings ("capture stopped [ms: ...]"), median and max per phase."""
    phases: dict[str, list[float]] = {}
    for m in STOP_LINE.finditer(text):
        for kv in m.group(1).split():
            k, _, v = kv.partition("=")
            try:
                phases.setdefault(k, []).append(float(v))
            except ValueError:
                pass
        phases.setdefault("total_awaited", []).append(float(m.group(2)))
    if not phases:
        return None
    return {k: {"median": round(statistics.median(v)), "max": round(max(v)), "n": len(v)}
            for k, v in phases.items()}


def clock_check(clock: dict | None) -> dict | None:
    """Worst state of the Pi's clock over a run (start and end)."""
    if not clock:
        return None
    states = [c for c in clock.values() if isinstance(c, dict)]
    offsets = [abs(c["offset_ms"]) for c in states if c.get("offset_ms") is not None]
    synced = [c.get("ntp_synchronized") for c in states]
    out = {"ntp_synchronized": None if None in synced else all(synced),
           "offset_ms_abs_max": round(max(offsets), 3) if offsets else None}
    out["ok"] = out["ntp_synchronized"] is not False and (out["offset_ms_abs_max"] or 0) <= CLOCK_OFFSET_MS
    return out


def run_inputs(run_dir: Path) -> tuple[set[str] | None, dict, dict]:
    """Episode ids, log counts and per-episode entries of a run_episodes.py folder."""
    run = load_json(run_dir / "run.json") or {}
    entries = {e["episode_id"]: e for e in run.get("episodes", []) if e.get("episode_id")}
    logs = {"run": run.get("label") or run_dir.name, "requested_episodes": len(run.get("episodes", [])),
            "failed_episodes": sum(1 for e in run.get("episodes", []) if e.get("error")),
            "clock": clock_check(run.get("clock"))}
    for name, patterns in (("daemon.log", LOG_PATTERNS), ("kernel.log", KERNEL_PATTERNS)):
        f = run_dir / name
        if f.exists():
            text = f.read_text(errors="replace")
            logs[name.split(".")[0]] = count_lines(text, patterns)
            if name == "daemon.log":
                logs["stop_phases_ms"] = stop_phases(text)
    return (set(entries) or None), logs, entries


# ── Output ─────────────────────────────────────────────────────────────

def print_summary(report: dict) -> None:
    def rate(c):
        return f"{c['count']} ({c['count'] / c['minutes'] * 60:.1f}/h)" if c.get("minutes") else str(c["count"])

    print(f"\n{report['label']}: {len(report['episodes'])} episode takes")
    for dev, s in list(report["devices"].items()) + [("ALL", report["overall"])]:
        inv = {k: v for k, v in s["invariants"].items() if v}
        print(f"\n  {dev}: {s['episodes']} episodes, {s['minutes']:.1f} min"
              + (f"   INVARIANTS BROKEN: {inv}" if inv else "   invariants ok"))
        g, d, a, st = s["rgb"], s["depth"], s["angle"], s["start"]
        print(f"    rgb    holes after start-up {rate(g['holes_after_startup'])}, start-up holes/episode "
              f"{g['holes_startup_per_episode']}, >150 ms {rate(g['holes_over_150ms'])}, "
              f">250 ms {rate(g['holes_over_250ms'])}, max {g['max_gap_ms']} ms, "
              f"first frame {g['first_frame_ms_median']} ms (max {g['first_frame_ms_max']})")
        if d["episodes"]:
            print(f"    depth  holes {rate(d['holes'])}, seq lost {rate(d['seq_lost'])}, max {d['max_gap_ms']} ms, "
                  f"first frame within {d['first_frame_ms_abs_max']} ms of t=0")
        if a["episodes"]:
            print(f"    angle  {a['rate_hz_median']} Hz (min {a['rate_hz_min']}), >100 ms {rate(a['holes_over_100ms'])}, "
                  f">500 ms {rate(a['holes_over_500ms'])}, max {a['max_gap_ms']} ms")
        print(f"    start  start_capture {st['start_capture_ms_median']} ms (max {st['start_capture_ms_max']}), "
              f"late {st['late_starts']['count']}/{st['late_starts']['episodes']}, "
              f"started-scheduled {st['started_minus_scheduled_ms_median']} ms (max {st['started_minus_scheduled_ms_max']})")
        en, sp = s["end"], s["stop"]
        if en["episodes"]:
            print(f"    end    streams' last samples within {en['spread_ms_median']} ms (max {en['spread_ms_max']}), "
                  f"a stream stopped early in {en['streams_stopped_early']['count']}/{en['episodes']}"
                  + (f", stop call {sp['stop_call_s_median']} s (max {sp['stop_call_s_max']})"
                     if sp["stop_call_s_median"] is not None else ""))
        if g["pts_offset_matched"] or g["pts_offset_ambiguous"]:
            print(f"    pts    offset from matching alone {g['pts_offset_matched']}, "
                  f"ambiguous by matching alone {g['pts_offset_ambiguous']}")
        if s["episodes_decoded"]:
            print(f"    decode {s['episodes_decoded']} episodes decoded, "
                  f"{s['invariants']['decoded_frame_mismatch']} with frames ≠ timestamps")
    if report["groups"]:
        spreads = [g["capture_started_spread_ms"] for g in report["groups"] if "capture_started_spread_ms" in g]
        print(f"\n  groups: {len(report['groups'])} multi-device episodes, capture_started spread "
              f"max {max(spreads) if spreads else '—'} ms (real gap: chronometer bench)")
    if report.get("chrono"):
        print(f"  chrono: {report['chrono']}")
    if report.get("logs"):
        print(f"  logs: {report['logs']}")
    for b in report.get("i2c_bench") or []:
        print(f"  i2c bench {b['mode']}: {b.get('loop_hz')} Hz, errors {b.get('errors')}, "
              f">500 ms reads {b.get('reads_over_500ms')}, {b.get('duration_s')} s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("paths", nargs="+", type=Path, help="episode folders, or folders containing them")
    p.add_argument("--run", type=Path, action="append", default=[],
                   help="run_episodes.py output folder: keep only its episodes, count its log lines")
    p.add_argument("--chrono", type=Path, help="measure_chrono_sync.py results.json")
    p.add_argument("--decode", action="store_true",
                   help="count the frames ffprobe decodes in each video (slow on a Pi: "
                        "~1 min per 2 min episode; fine on a workstation)")
    p.add_argument("--i2c", type=Path, action="append", default=[], help="bench_i2c_angles.py summary.json")
    p.add_argument("--since", help="keep episodes recorded at or after this UTC time (YYYYMMDD_HHMMSS)")
    p.add_argument("--until", help="keep episodes recorded before this UTC time (YYYYMMDD_HHMMSS)")
    p.add_argument("--label", default="", help="what was measured, e.g. 'develop @ 1707d4e'")
    p.add_argument("--out", type=Path, help="write the JSON report here")
    args = p.parse_args()

    if args.decode and not shutil.which("ffprobe"):
        sys.exit("--decode needs ffprobe")
    keep_ids, logs, entries = None, {}, {}
    for run_dir in args.run:
        ids, lg, ent = run_inputs(run_dir)
        if ids:
            keep_ids = (keep_ids or set()) | ids
        entries.update(ent)
        logs[lg.pop("run")] = lg

    eps = []
    for f in find_episodes(args.paths):
        eid = episode_id(f)
        if keep_ids is not None and eid not in keep_ids:
            continue
        if args.since and eid < args.since or args.until and eid >= args.until:
            continue
        e = measure_episode(f, decode=args.decode)
        e["stop_call_s"] = (entries.get(eid) or {}).get("stop_call_s")
        eps.append(e)
    if not eps:
        print("No episode found (a folder with frame_timestamps.json)", file=sys.stderr)
        return 1

    devices: dict[str, list[dict]] = {}
    for e in eps:
        devices.setdefault(e["device"], []).append(e)
    report = {
        "kind": "episode_report",
        "version": 2,
        "label": args.label or ", ".join(str(p) for p in args.paths),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thresholds_ms": {"rgb_hole": RGB_HOLE_MS, "startup": STARTUP_MS, "depth_hole": DEPTH_HOLE_MS,
                          "angle_stall": ANGLE_STALL_MS, "angle_timeout": ANGLE_TIMEOUT_MS,
                          "late_start": LATE_START_MS, "early_stop": EARLY_STOP_MS,
                          "clock_offset": CLOCK_OFFSET_MS},
        "overall": summarize(eps),
        "devices": {d: summarize(v) for d, v in sorted(devices.items())},
        "groups": group_measures(eps),
        "chrono": chrono_measures(args.chrono) if args.chrono else None,
        "logs": logs or None,
        "i2c_bench": [],
        "episodes": eps,
    }
    for f in args.i2c:
        b = load_json(f) or {}
        sensors = b.get("sensors") or {}
        report["i2c_bench"].append({
            "mode": b.get("mode"), "duration_s": b.get("duration_s"), "loop_hz": b.get("loop_hz"),
            "errors": sum(sum((s.get("errors") or {}).values()) for s in sensors.values()),
            "reads_over_500ms": sum(s.get("reads_over_500ms", 0) for s in sensors.values()),
            "kernel_i2c_lines": b.get("kernel_i2c_lines"),
        })
    print_summary(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\nReport: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
