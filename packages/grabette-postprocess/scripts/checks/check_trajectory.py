#!/usr/bin/env python3
"""
Validate SLAM trajectory quality.

Detects common failure modes: IMU drift, relocalization jumps, zigzagging,
and unrealistic motion. Works on individual episodes or entire datasets.

The analysis logic lives in grabette_postprocess.checks.trajectory so it can be
reused by the HF Space pipeline; this file is just the CLI.

Usage:
    # Check a single episode
    uv run python scripts/checks/check_trajectory.py test_data/test_HF/episodes/20260331_095957

    # Check all episodes in a dataset
    uv run python scripts/checks/check_trajectory.py test_data/test_HF/episodes

    # Verbose output with per-frame details
    uv run python scripts/checks/check_trajectory.py test_data/test_HF/episodes -v
"""

from pathlib import Path

import click

from grabette_postprocess.checks.trajectory import check_trajectory
from grabette_postprocess.episode_manager import find_trajectory_episodes


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True, help="Show detailed per-episode diagnostics")
@click.option("--jump_threshold", type=float, default=50.0,
              help="Jump detection threshold in mm (default: 50)")
@click.option("--max_speed", type=float, default=2.0,
              help="Max reasonable gripper speed in m/s (default: 2.0)")
def main(path, verbose, jump_threshold, max_speed):
    """Validate SLAM trajectory quality for episodes or datasets."""
    episodes = find_trajectory_episodes(Path(path))

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

        report = check_trajectory(
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
