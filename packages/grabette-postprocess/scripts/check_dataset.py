#!/usr/bin/env python3
"""Quick health check on a dataset directory (OAK + Arducam hardware).

For each episode, checks the recordings produced by the current rig:
  - Arducam observation camera : raw_video.mp4 (+ frame_timestamps.json)
  - OAK RGBD                   : oakd_left.mp4 / oakd_right.mp4 / oakd_depth/
                                 (+ *_timestamps.json) and oakd_calib_offline.json
  - OAK IMU                    : oakd_imu.json (accel + gyro + rotation)
  - Gripper                    : angle_data.json (joint angles)
  - SLAM outputs (if present)  : camera_trajectory.csv (+ slam_metadata.json)

Counts are cross-checked against metadata.json. The check logic lives in
grabette_postprocess.episode_check so it can be reused by the HF Space pipeline;
this file is just the CLI.

Usage:
    uv run python scripts/check_dataset.py ~/data/dataset
"""

from pathlib import Path

import click

from grabette_postprocess.episode_check import check_episode


@click.command()
@click.argument("dataset_dir", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True, help="Show per-episode info lines too")
def main(dataset_dir, verbose):
    """Check dataset health: Arducam video, OAK RGBD/IMU, gripper angles, SLAM outputs."""
    dataset_dir = Path(dataset_dir).expanduser().absolute()

    # An episode is any dir containing an OAK recording (oakd_imu.json is the anchor).
    episodes = sorted({p.parent for p in dataset_dir.rglob("oakd_imu.json")})
    if not episodes and (dataset_dir / "oakd_imu.json").is_file():
        episodes = [dataset_dir]  # dataset_dir is itself a single episode
    if not episodes:
        print(f"No episodes (oakd_imu.json) found under {dataset_dir}")
        return

    print(f"Checking {len(episodes)} episode(s) in {dataset_dir}\n")

    n_errors = n_warnings = n_ok = 0
    for ep_dir in episodes:
        s = check_episode(ep_dir)
        label = s["name"]
        traj = f"  {s['trajectory']}" if "trajectory" in s else ""

        if s["errors"]:
            n_errors += len(s["errors"])
            head = "ERROR"
        elif s["warnings"]:
            n_warnings += len(s["warnings"])
            head = "WARN "
        else:
            n_ok += 1
            head = "OK   "
        print(f"  {head} {label}{traj}")

        if verbose or s["errors"] or s["warnings"]:
            if verbose and s["info"]:
                print(f"         {'  '.join(s['info'])}")
            for e in s["errors"]:
                print(f"         ✗ {e}")
            for w in s["warnings"]:
                print(f"         ! {w}")

    print()
    parts = []
    if n_errors:
        parts.append(f"{n_errors} error(s)")
    if n_warnings:
        parts.append(f"{n_warnings} warning(s)")
    parts.append(f"{n_ok}/{len(episodes)} clean")
    print(", ".join(parts))


if __name__ == "__main__":
    main()
