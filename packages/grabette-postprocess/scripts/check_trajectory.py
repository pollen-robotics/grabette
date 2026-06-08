#!/usr/bin/env python3
"""
Validate SLAM trajectory quality.

Detects common failure modes: IMU drift, relocalization jumps, zigzagging,
and unrealistic motion. Works on individual episodes or entire datasets.

Usage:
    # Check a single episode
    uv run python scripts/check_trajectory.py test_data/test_HF/episodes/20260331_095957

    # Check all episodes in a dataset
    uv run python scripts/check_trajectory.py test_data/test_HF/episodes

    # Verbose output with per-frame details
    uv run python scripts/check_trajectory.py test_data/test_HF/episodes -v
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
import pandas as pd


@dataclass
class TrajectoryReport:
    """Quality report for a single trajectory."""
    name: str
    n_frames: int = 0
    n_tracked: int = 0
    n_lost: int = 0
    tracking_pct: float = 0.0
    duration_s: float = 0.0
    total_distance_m: float = 0.0
    median_step_mm: float = 0.0
    max_step_mm: float = 0.0
    n_jumps: int = 0          # frames with step > jump_threshold
    median_angle_deg: float = 0.0
    drift_score: float = 0.0  # 0=no drift, higher=more drift-like
    method: str = "?"
    frame_skip: int = 1
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verdict: str = "UNKNOWN"


def analyze_trajectory(
    traj_path: Path,
    meta_path: Path | None = None,
    jump_threshold_mm: float = 50.0,
    max_reasonable_speed_ms: float = 2.0,
) -> TrajectoryReport:
    """Analyze a trajectory CSV and produce a quality report.

    Args:
        traj_path: path to camera_trajectory.csv or mapping_camera_trajectory.csv
        meta_path: optional path to slam_metadata.json
        jump_threshold_mm: flag frames with displacement above this (mm)
        max_reasonable_speed_ms: maximum plausible gripper speed (m/s)
    """
    name = traj_path.parent.name
    report = TrajectoryReport(name=name)

    df = pd.read_csv(traj_path)
    report.n_frames = len(df)
    report.n_lost = int(df["is_lost"].sum())
    report.n_tracked = report.n_frames - report.n_lost
    report.tracking_pct = 100.0 * report.n_tracked / report.n_frames if report.n_frames > 0 else 0.0

    if report.n_tracked < 2:
        report.errors.append(f"Only {report.n_tracked} tracked frames")
        report.verdict = "FAIL"
        return report

    # Load metadata if available
    if meta_path and meta_path.is_file():
        with open(meta_path) as f:
            meta = json.load(f)
        report.method = meta.get("method", "?")
        report.frame_skip = meta.get("frame_skip", 1)

    # Extract tracked positions and timestamps
    tracked = df[~df["is_lost"].astype(bool)]
    pos = tracked[["x", "y", "z"]].values
    ts = tracked["timestamp"].values
    report.duration_s = ts[-1] - ts[0]

    # Per-frame displacement
    displacements = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dt = np.diff(ts)
    dt[dt == 0] = 1e-6  # avoid division by zero

    report.total_distance_m = float(np.sum(displacements))
    report.median_step_mm = float(np.median(displacements) * 1000)
    report.max_step_mm = float(np.max(displacements) * 1000)
    report.n_jumps = int(np.sum(displacements > jump_threshold_mm / 1000))

    # Direction changes (angle between consecutive segments)
    segments = np.diff(pos, axis=0)
    norms = np.linalg.norm(segments, axis=1)
    angles = []
    for i in range(len(segments) - 1):
        if norms[i] > 1e-6 and norms[i + 1] > 1e-6:
            cos = np.dot(segments[i], segments[i + 1]) / (norms[i] * norms[i + 1])
            angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    if angles:
        report.median_angle_deg = float(np.median(angles))

    # Drift score: ratio of total distance to displacement.
    # Pure drift = straight line: distance ≈ displacement → ratio ≈ 1
    # Real motion: lots of back-and-forth → ratio >> 1
    # But also: IMU drift has very smooth trajectory (low angle changes)
    # and unrealistically high speed
    displacement = np.linalg.norm(pos[-1] - pos[0])
    if displacement > 1e-6:
        distance_ratio = report.total_distance_m / displacement
    else:
        distance_ratio = report.total_distance_m * 100  # large

    avg_speed = report.total_distance_m / max(report.duration_s, 0.1)

    # Drift-like: high speed + straight trajectory (low distance_ratio + low angles)
    # Real motion: moderate speed + complex trajectory (high distance_ratio + varied angles)
    if avg_speed > max_reasonable_speed_ms and distance_ratio < 3.0:
        report.drift_score = avg_speed / max_reasonable_speed_ms
    else:
        report.drift_score = 0.0

    # ---- Checks ----

    # Speed check
    if avg_speed > max_reasonable_speed_ms:
        report.errors.append(
            f"Unrealistic avg speed: {avg_speed:.2f} m/s "
            f"(total {report.total_distance_m:.2f}m in {report.duration_s:.1f}s)"
        )

    # Jump check
    if report.n_jumps > report.n_tracked * 0.1:
        report.warnings.append(
            f"{report.n_jumps} jumps > {jump_threshold_mm:.0f}mm "
            f"({100*report.n_jumps/report.n_tracked:.0f}% of frames)"
        )

    # Drift detection: high median step + low angle variation = IMU dead-reckoning
    if report.median_step_mm > 15 and report.median_angle_deg < 5:
        report.errors.append(
            f"Likely IMU drift: med_step={report.median_step_mm:.1f}mm, "
            f"med_angle={report.median_angle_deg:.1f}° (straight-line motion)"
        )

    # Zigzag detection: many large jumps with direction reversals
    if report.n_jumps > 5 and report.median_angle_deg > 90:
        report.errors.append(
            f"Zigzag pattern: {report.n_jumps} jumps with "
            f"med_angle={report.median_angle_deg:.1f}° (repeated relocalization failures)"
        )

    # Tracking rate
    if report.tracking_pct < 50:
        report.warnings.append(f"Low tracking: {report.tracking_pct:.1f}%")

    # Verdict
    if report.errors:
        report.verdict = "BAD"
    elif report.warnings:
        report.verdict = "WARN"
    else:
        report.verdict = "GOOD"

    return report


def find_episodes(path: Path) -> list[Path]:
    """Find episode directories containing trajectory CSVs."""
    path = path.expanduser().absolute()
    episodes = []
    if (path / "camera_trajectory.csv").is_file() or (path / "mapping_camera_trajectory.csv").is_file():
        # Single episode
        episodes.append(path)
    else:
        # Dataset directory: find all subdirs with trajectories
        for traj in sorted(path.rglob("camera_trajectory.csv")):
            episodes.append(traj.parent)
        for traj in sorted(path.rglob("mapping_camera_trajectory.csv")):
            if traj.parent not in episodes:
                episodes.append(traj.parent)
    return sorted(set(episodes))


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True, help="Show detailed per-episode diagnostics")
@click.option("--jump_threshold", type=float, default=50.0,
              help="Jump detection threshold in mm (default: 50)")
@click.option("--max_speed", type=float, default=2.0,
              help="Max reasonable gripper speed in m/s (default: 2.0)")
def main(path, verbose, jump_threshold, max_speed):
    """Validate SLAM trajectory quality for episodes or datasets."""
    episodes = find_episodes(Path(path))

    if not episodes:
        print(f"No trajectories found under {path}")
        return

    print(f"Checking {len(episodes)} trajectories\n")

    counts = {"GOOD": 0, "WARN": 0, "BAD": 0, "FAIL": 0}

    for ep_dir in episodes:
        traj = ep_dir / "camera_trajectory.csv"
        if not traj.is_file():
            traj = ep_dir / "mapping_camera_trajectory.csv"
        meta = ep_dir / "slam_metadata.json"

        report = analyze_trajectory(
            traj, meta,
            jump_threshold_mm=jump_threshold,
            max_reasonable_speed_ms=max_speed,
        )
        counts[report.verdict] = counts.get(report.verdict, 0) + 1

        # Format output
        icon = {"GOOD": "OK", "WARN": "WARN", "BAD": "BAD", "FAIL": "FAIL"}[report.verdict]
        stats = (f"{report.n_tracked}/{report.n_frames} "
                 f"dist={report.total_distance_m:.2f}m "
                 f"med={report.median_step_mm:.1f}mm "
                 f"jumps={report.n_jumps} "
                 f"[{report.method}]")
        print(f"  {icon:4s} {report.name}  {stats}")

        if verbose or report.verdict in ("BAD", "FAIL"):
            for e in report.errors:
                print(f"         {e}")
            for w in report.warnings:
                print(f"         {w}")

    # Summary
    print(f"\n{'='*50}")
    total = sum(counts.values())
    for verdict in ["GOOD", "WARN", "BAD", "FAIL"]:
        if counts[verdict] > 0:
            print(f"  {verdict}: {counts[verdict]}/{total}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
