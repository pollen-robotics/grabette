"""Compare two episode reports: does the candidate branch regress against the baseline?

    python3 benchmarks/compare_reports.py report-develop.json report-pr.json [--out compare.md]

Prints a Markdown table (paste it in the PR) and exits 1 if anything regressed.

How each figure is judged:
  invariant  must be 0 (invalid alignment, unwritten camera frames, frame
             count mismatch, writer queue full, depth frames dropped...).
             Regression when the candidate breaks one the baseline kept.
  rate       rare events counted over a duration (holes, I2C timeouts) or a
             number of episodes (late starts). A Poisson test asks whether the
             candidate's count is too high for the baseline's rate: regression
             when that would happen by chance less than 5 % of the time
             (improvement, the other way round). With few minutes recorded,
             small differences are reported as "not significant", not hidden.
  value      a median or maximum (start-up delays, angle rate, chronometer
             offsets) with a tolerance: regression when it moves the wrong way
             by more than the tolerance, or crosses an absolute limit.

The two reports should come from the same grabette(s) and the same protocol
(see benchmarks/README.md); the device lists are printed to check.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ALPHA = 0.05

# (path in the summary, label, kind, options)
#   rate:  options = None
#   value: options = (direction, tolerance, absolute limit or None)
#          direction "up" = higher is worse, "down" = lower is worse, "abs" = |x| higher is worse
METRICS = [
    (("invariants", "alignment_invalid"), "RGB: episodes with invalid timestamp alignment", "invariant", None),
    (("invariants", "unwritten_camera_frames"), "RGB: camera frames never written", "invariant", None),
    (("invariants", "frame_count_mismatch"), "RGB: timestamps ≠ frames", "invariant", None),
    (("invariants", "writer_queue_full"), "RGB: writer queue full (300)", "invariant", None),
    (("invariants", "depth_dropped_frames"), "Depth: frames dropped by the encoder queue", "invariant", None),
    (("invariants", "ir_depth_count_mismatch"), "Depth: IR / depth counts differ", "invariant", None),
    (("invariants", "decoded_frame_mismatch"), "Videos: decoded frames ≠ timestamps (--decode)", "invariant", None),
    (("rgb", "holes_over_250ms"), "RGB: holes > 250 ms (flagged)", "rate", None),
    (("rgb", "holes_over_150ms"), "RGB: holes > 150 ms", "rate", None),
    (("rgb", "holes_after_startup"), "RGB: holes > 30 ms after the first 1.5 s", "rate", None),
    (("rgb", "holes_startup_per_episode"), "RGB: holes in the first 1.5 s, per episode (median)", "value", ("up", 2, None)),
    (("rgb", "first_frame_ms_median"), "RGB: first frame after t=0, ms (median)", "value", ("up", 30, None)),
    (("rgb", "fps_median"), "RGB: frame rate (median)", "value", ("down", 1.0, None)),
    (("rgb", "reordered_pictures"), "RGB: pictures put back in order (info)", "value", ("up", None, None)),
    (("rgb", "writer_queue_peak_max"), "RGB: writer queue peak (max)", "value", ("up", 100, 250)),
    (("rgb", "pts_offset_matched"), "RGB: PTS offset from matching alone, episodes (info)", "value", ("up", None, None)),
    (("rgb", "pts_offset_ambiguous"), "RGB: episodes matching alone could not decide (info)", "value", ("up", None, None)),
    (("depth", "holes"), "Depth: holes > 100 ms", "rate", None),
    (("depth", "seq_lost"), "Depth: frames numbered by the camera, never received", "rate", None),
    (("depth", "first_frame_ms_abs_max"), "Depth: first frame vs t=0, |ms| (max)", "value", ("up", 30, None)),
    (("angle", "holes_over_500ms"), "Angles: holes > 500 ms (I2C timeouts)", "rate", None),
    (("angle", "holes_over_100ms"), "Angles: holes > 100 ms (loop stalls)", "rate", None),
    (("angle", "rate_hz_median"), "Angles: loop rate, Hz (median)", "value", ("down", 5.0, None)),
    (("start", "start_capture_ms_median"), "Start: start_capture, ms (median)", "value", ("up", 50, None)),
    (("start", "late_starts"), "Start: starts later than 1 s", "rate", None),
    (("start", "started_minus_scheduled_ms_median"), "Start: capture_started − scheduled, ms (median)", "value", ("abs", 20, None)),
    (("end", "streams_stopped_early"), "End: a stream stopped > 0.5 s before the others", "rate", None),
    (("end", "spread_ms_median"), "End: spread of the streams' last samples, ms (median)", "value", ("up", 100, None)),
    (("stop", "stop_call_s_median"), "Stop: stop call, s (median)", "value", ("up", 0.5, None)),
    (("stop", "stop_call_s_max"), "Stop: stop call, s (max)", "value", ("up", 2.0, 10.0)),
    (("chrono", "rpi_minus_orbbec_ms_median"), "Sync: RPi − Orbbec, ms (median, chronometer)", "value", ("abs", 5, 15)),
    (("chrono", "rpi_minus_orbbec_drift_ms_max"), "Sync: RPi − Orbbec drift within an episode, ms (max)", "value", ("up", 10, None)),
    (("chrono", "capture_started_error_ms_abs_max"), "Sync: capture_started error vs chronometer, |ms| (max)", "value", ("up", 5, 15)),
]


def get(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    term = total = math.exp(-lam)
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return min(total, 1.0)


def judge_rate(base: dict, cand: dict) -> tuple[str, str]:
    """Is the candidate's count unlikely under the baseline's rate?"""
    kb, kc = base["count"], cand["count"]
    eb = base.get("minutes", base.get("episodes")) or 0
    ec = cand.get("minutes", cand.get("episodes")) or 0
    if not eb or not ec:
        return "n/a", "no exposure"
    # Half an event when the baseline saw none: 0 in 10 min is not a rate of 0.
    expected = max(kb, 0.5) / eb * ec
    p_worse = 1 - poisson_cdf(kc - 1, expected)
    p_better = poisson_cdf(kc, kb / eb * ec) if kb else 1.0
    note = f"expected ≈ {expected:.1f} at the baseline rate"
    if kc > expected and p_worse < ALPHA:
        return "REGRESSION", f"{note}; p = {p_worse:.3f}"
    if kc < expected and p_better < ALPHA:
        return "better", f"{note}; p = {p_better:.3f}"
    if kc > expected:
        return "worse, not significant", f"{note}; p = {p_worse:.2f}"
    return "ok", note


def judge_value(b, c, opts) -> tuple[str, str]:
    direction, tol, limit = opts
    if c is None:
        return "n/a", "not measured on the candidate"
    worse_by = {"up": lambda: c - b, "down": lambda: b - c, "abs": lambda: abs(c) - abs(b)}[direction]
    mag = abs(c) if direction == "abs" else c
    if limit is not None and mag > limit:
        return "REGRESSION", f"over the limit of {limit}"
    if b is None:
        return "ok" if limit is not None else "n/a", "no baseline value"
    if tol is None:
        return "info", ""
    d = worse_by()
    if d > tol:
        return "REGRESSION", f"worse by {d:.1f} (tolerance {tol})"
    if d < -tol:
        return "better", f"better by {-d:.1f}"
    return "ok", f"within ±{tol}"


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, dict) and "count" in v:
        exp = v.get("minutes")
        if exp:
            return f"{v['count']} in {exp:.0f} min ({v['count'] / exp * 60:.1f}/h)"
        return f"{v['count']} / {v.get('episodes', '?')} episodes"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def summary_of(report: dict, device: str | None) -> dict:
    s = dict(report["devices"][device]) if device else dict(report["overall"])
    s["chrono"] = report.get("chrono") or {}
    return s


def compare(base: dict, cand: dict, device: str | None = None) -> tuple[list[dict], bool]:
    sb, sc = summary_of(base, device), summary_of(cand, device)
    rows, regressed = [], False
    mismatch = any(get(s, ("invariants", "frame_count_mismatch")) for s in (sb, sc))
    for path, label, kind, opts in METRICS:
        b, c = get(sb, path), get(sc, path)
        if b is None and c is None:
            continue
        if path[0] == "rgb" and path[1].startswith("first_frame") and mismatch:
            # Without PR 239's alignment the timestamps don't belong to the
            # written frames: their first value says nothing about the video.
            rows.append({"metric": label, "baseline": fmt(b), "candidate": fmt(c),
                         "verdict": "n/a", "why": "timestamps ≠ frames on one side: not comparable"})
            continue
        if path == ("invariants", "decoded_frame_mismatch") and not (
                sb.get("episodes_decoded") and sc.get("episodes_decoded")):
            continue
        if kind == "invariant":
            if not c:
                verdict, why = "ok", ""
            elif b:
                verdict, why = "still broken", "also broken on the baseline"
            else:
                verdict, why = "REGRESSION", "broken on the candidate only"
        elif kind == "rate":
            verdict, why = judge_rate(b, c) if b and c else ("n/a", "missing on one side")
        else:
            verdict, why = judge_value(b, c, opts)
        regressed |= verdict == "REGRESSION"
        rows.append({"metric": label, "baseline": fmt(b), "candidate": fmt(c), "verdict": verdict, "why": why})
    return rows, regressed


def clock_warnings(report: dict, side: str) -> list[str]:
    out = []
    for run, lg in (report.get("logs") or {}).items():
        c = (lg or {}).get("clock")
        if c and not c.get("ok"):
            out.append(f"> ⚠️ {side} run '{run}': clock not synchronised (NTP synchronised: "
                       f"{c.get('ntp_synchronized')}, offset up to {c.get('offset_ms_abs_max')} ms): "
                       "capture_started_utc and group starts are off by as much.")
    return out


def markdown(base: dict, cand: dict, rows: list[dict], regressed: bool) -> str:
    def devs(r):
        return ", ".join(f"{d} ({s['episodes']} ep., {s['minutes']:.0f} min)" for d, s in r["devices"].items())
    icon = {"REGRESSION": "❌", "better": "✅", "ok": "✓", "still broken": "⚠️",
            "worse, not significant": "≈", "info": "ℹ️", "n/a": "—"}
    out = [f"### Benchmark: {'❌ regression' if regressed else '✅ no regression'}", "",
           f"- **Baseline**: {base['label']} — {devs(base)}",
           f"- **Candidate**: {cand['label']} — {devs(cand)}", ""]
    if set(base["devices"]) != set(cand["devices"]):
        out += ["> ⚠️ Not the same grabettes on both sides: hardware differences can look like regressions.", ""]
    warnings = clock_warnings(base, "Baseline") + clock_warnings(cand, "Candidate")
    if warnings:
        out += warnings + [""]
    if bool(base["overall"].get("episodes_decoded")) != bool(cand["overall"].get("episodes_decoded")):
        out += ["> ℹ️ --decode was used on one side only: decoded frame counts are not compared.", ""]
    low = min(base["overall"]["minutes"], cand["overall"]["minutes"])
    if low < 30:
        out += [f"> ℹ️ Only {low:.0f} min recorded on one side: rare events (angle timeouts, long holes) "
                "need about an hour per side to be judged.", ""]
    out += ["| Check | Baseline | Candidate | Verdict | |", "|---|---|---|---|---|"]
    out += [f"| {r['metric']} | {r['baseline']} | {r['candidate']} | {icon.get(r['verdict'], '')} {r['verdict']} | {r['why']} |"
            for r in rows]
    return "\n".join(out) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("baseline", type=Path, help="episode_report.py JSON of the reference (e.g. develop)")
    p.add_argument("candidate", type=Path, help="episode_report.py JSON of the branch under review")
    p.add_argument("--device", help="compare one device's figures instead of all devices together")
    p.add_argument("--out", type=Path, help="also write the Markdown here")
    args = p.parse_args()
    base, cand = (json.loads(f.read_text()) for f in (args.baseline, args.candidate))
    if args.device and (args.device not in base["devices"] or args.device not in cand["devices"]):
        sys.exit(f"{args.device} is not in both reports")
    rows, regressed = compare(base, cand, args.device)
    md = markdown(base, cand, rows, regressed)
    print(md)
    if args.out:
        args.out.write_text(md)
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
