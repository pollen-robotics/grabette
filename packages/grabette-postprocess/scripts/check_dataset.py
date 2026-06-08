#!/usr/bin/env python3
"""Quick health check on a dataset directory.

Checks IMU sample counts, video metadata, and existing SLAM outputs
for each episode. Flags obvious problems.

Usage:
    uv run python scripts/check_dataset.py ~/data/dataset
"""

import json
import sys
from pathlib import Path

import click
import av


def check_episode(ep_dir: Path) -> dict:
    """Check one episode directory, return a status dict."""
    status = {"name": ep_dir.name, "errors": [], "warnings": []}

    # Video
    video = ep_dir / "raw_video.mp4"
    if not video.is_file():
        status["errors"].append("missing raw_video.mp4")
        return status

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        status["video_frames"] = stream.frames
        status["video_fps"] = float(stream.average_rate)
        status["video_duration"] = float(stream.duration * stream.time_base)
        status["video_res"] = f"{stream.width}x{stream.height}"

    # Cross-check video frame count with metadata
    meta_path = ep_dir / "metadata.json"
    if meta_path.is_file():
        with open(meta_path) as f:
            capture_meta = json.load(f)
        meta_frames = capture_meta.get("frame_count", 0)
        diff = meta_frames - status["video_frames"]
        if diff > 1:
            status["warnings"].append(
                f"metadata says {meta_frames} frames but video has {status['video_frames']} "
                f"({diff} frames lost during muxing)"
            )

    # IMU
    imu_path = ep_dir / "imu_data.json"
    if not imu_path.is_file():
        status["errors"].append("missing imu_data.json")
        return status

    with open(imu_path) as f:
        imu = json.load(f)

    streams = imu.get("1", {}).get("streams", {})
    for stream_name in ["ACCL", "GYRO"]:
        samples = streams.get(stream_name, {}).get("samples", [])
        n = len(samples)
        status[f"{stream_name.lower()}_samples"] = n

        expected = int(status["video_duration"] * 200)  # 200Hz
        if n < expected * 0.5:
            status["errors"].append(
                f"{stream_name}: {n} samples (expected ~{expected} for {status['video_duration']:.1f}s at 200Hz)"
            )
        elif n < expected * 0.8:
            status["warnings"].append(
                f"{stream_name}: {n} samples (expected ~{expected})"
            )

        # Check for duplicate values (stale reads)
        if n > 10:
            dupes = sum(1 for i in range(1, min(n, 200))
                        if samples[i]["value"] == samples[i-1]["value"])
            dupe_pct = 100 * dupes / min(n, 200)
            if dupe_pct > 30:
                status["warnings"].append(
                    f"{stream_name}: {dupe_pct:.0f}% duplicate values in first 200 samples"
                )

    # ANGL stream
    angl_samples = streams.get("ANGL", {}).get("samples", [])
    status["angl_samples"] = len(angl_samples)

    # Frame timestamps (if available)
    frame_ts_path = ep_dir / "frame_timestamps.json"
    if frame_ts_path.is_file():
        import numpy as np
        with open(frame_ts_path) as f:
            frame_ts = json.load(f)
        if len(frame_ts) >= 2:
            intervals = np.diff(frame_ts)
            expected_interval = 1000.0 / status["video_fps"]  # ms
            drops = intervals > expected_interval * 1.5
            n_drops = int(np.sum(drops))
            if n_drops > 0:
                total_missed = sum(round(intervals[i] / expected_interval) - 1
                                  for i in np.where(drops)[0])
                status["frame_drops"] = n_drops
                status["warnings"].append(
                    f"{n_drops} frame drops ({int(total_missed)} frames missed)"
                )
            # Check if video frame count matches timestamps
            if abs(len(frame_ts) - status["video_frames"]) > 1:
                status["warnings"].append(
                    f"frame_timestamps ({len(frame_ts)}) != video frames ({status['video_frames']})"
                )

    # SLAM outputs
    for name in ["camera_trajectory.csv", "mapping_camera_trajectory.csv"]:
        traj = ep_dir / name
        if traj.is_file():
            import pandas as pd
            df = pd.read_csv(traj)
            tracked = len(df) - int(df["is_lost"].sum())
            status["trajectory"] = f"{tracked}/{len(df)} ({100*tracked/len(df):.1f}%)"
            break

    meta = ep_dir / "slam_metadata.json"
    if meta.is_file():
        with open(meta) as f:
            m = json.load(f)
        status["slam_method"] = m.get("method", "?")
        status["frame_skip"] = m.get("frame_skip", "?")

    return status


@click.command()
@click.argument("dataset_dir", type=click.Path(exists=True))
def main(dataset_dir):
    """Check dataset health: IMU data, video, SLAM outputs."""
    dataset_dir = Path(dataset_dir).expanduser().absolute()

    # Find all episode directories
    episodes = sorted([
        p.parent for p in dataset_dir.rglob("raw_video.mp4")
    ])

    if not episodes:
        print(f"No episodes found under {dataset_dir}")
        return

    print(f"Checking {len(episodes)} episodes in {dataset_dir}\n")

    n_errors = 0
    for ep_dir in episodes:
        s = check_episode(ep_dir)
        label = s["name"]

        # Build status line
        parts = []
        if "video_res" in s:
            parts.append(f'{s["video_res"]}')
        if "video_duration" in s:
            parts.append(f'{s["video_duration"]:.1f}s')
        if "accl_samples" in s:
            parts.append(f'IMU:{s["accl_samples"]}')
        if "angl_samples" in s:
            parts.append(f'ANGL:{s["angl_samples"]}')
        if "trajectory" in s:
            method = s.get("slam_method", "")
            parts.append(f'traj:{s["trajectory"]} [{method}]')

        status_str = "  ".join(parts)

        if s["errors"]:
            n_errors += len(s["errors"])
            print(f"  ERROR  {label}  {status_str}")
            for e in s["errors"]:
                print(f"           {e}")
        elif s["warnings"]:
            print(f"  WARN   {label}  {status_str}")
            for w in s["warnings"]:
                print(f"           {w}")
        else:
            print(f"  OK     {label}  {status_str}")

    if n_errors > 0:
        print(f"\n{n_errors} error(s) found")
    else:
        print(f"\nAll {len(episodes)} episodes OK")


if __name__ == "__main__":
    main()
