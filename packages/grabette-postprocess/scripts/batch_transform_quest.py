#!/usr/bin/env python3
"""
Batch-transform Quest trajectories to camera frame for all episodes in a dataset.

Applies a saved Quest→camera calibration to each episode's r_hand_traj.json,
producing camera_trajectory.csv files compatible with the rest of the pipeline.

Usage:
    uv run python scripts/batch_transform_quest.py \
        -i ~/data/dataset \
        -c config/quest_to_camera_calibration.json

    # Force reprocess existing trajectories
    uv run python scripts/batch_transform_quest.py \
        -i ~/data/dataset \
        -c config/quest_to_camera_calibration.json \
        --force
"""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Load Quest→camera calibration. Returns (R_world, t, scale, time_offset_s, R_body)."""
    with open(path) as f:
        cal = json.load(f)
    R_world = Rotation.from_quat(cal["rotation_quaternion_xyzw"]).as_matrix()
    t = np.array(cal["translation"])
    s = cal["scale"]
    time_offset = cal["time_offset_ms"] / 1000.0
    if "body_rotation_quaternion_xyzw" in cal:
        R_body = Rotation.from_quat(cal["body_rotation_quaternion_xyzw"]).as_matrix()
    else:
        R_body = np.eye(3)
    return R_world, t, s, time_offset, R_body


def transform_episode(
    episode_dir: Path,
    R_world: np.ndarray,
    t: np.ndarray,
    s: float,
    time_offset: float,
    R_body: np.ndarray | None = None,
    quest_filename: str = "r_hand_traj.json",
) -> int:
    """Transform one episode's Quest trajectory to camera frame.

    Returns number of frames written, or 0 if quest file not found.
    """
    if R_body is None:
        R_body = np.eye(3)

    quest_path = episode_dir / quest_filename
    if not quest_path.is_file():
        return 0

    with open(quest_path) as f:
        data = json.load(f)

    # Extract timestamps, positions, rotations
    timestamps = np.array([e["timestamp_ms"] for e in data], dtype=np.float64)
    timestamps = (timestamps - timestamps[0]) / 1000.0 + time_offset

    positions = []
    quaternions = []
    for entry in data:
        pose = np.array(entry["pose"]).reshape(4, 4)
        # Detect layout from last row
        if not (abs(pose[3, 0]) < 1e-6 and abs(pose[3, 1]) < 1e-6 and abs(pose[3, 3] - 1.0) < 1e-6):
            pose = pose.T
        pos = pose[:3, 3]
        rot = pose[:3, :3]

        # Apply Quest→camera transform
        # Position: p_cam = s * R_world @ p_quest + t
        # Orientation: R_cam = R_world @ R_quest @ R_body
        pos_cam = s * (R_world @ pos) + t
        rot_cam = R_world @ rot @ R_body
        quat = Rotation.from_matrix(rot_cam).as_quat()  # xyzw

        positions.append(pos_cam)
        quaternions.append(quat)

    positions = np.array(positions)
    quaternions = np.array(quaternions)

    # Save as camera_trajectory.csv
    df = pd.DataFrame({
        "frame_idx": np.arange(len(timestamps)),
        "timestamp": timestamps,
        "state": 2,
        "is_lost": False,
        "is_keyframe": False,
        "x": positions[:, 0],
        "y": positions[:, 1],
        "z": positions[:, 2],
        "q_x": quaternions[:, 0],
        "q_y": quaternions[:, 1],
        "q_z": quaternions[:, 2],
        "q_w": quaternions[:, 3],
    })
    df.to_csv(episode_dir / "camera_trajectory.csv", index=False)

    # Save metadata
    meta = {
        "method": "quest",
        "tracking_pct": 100.0,
        "tracked_frames": len(timestamps),
        "total_frames": len(timestamps),
        "quest_file": quest_filename,
    }
    with open(episode_dir / "slam_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    return len(timestamps)


@click.command()
@click.option("-i", "--input_dir", required=True, type=click.Path(exists=True),
              help="Dataset directory containing episode subdirectories")
@click.option("-c", "--calibration", required=True, type=click.Path(exists=True),
              help="Quest→camera calibration JSON")
@click.option("--quest-file", default="r_hand_traj.json",
              help="Quest trajectory filename in each episode (default: r_hand_traj.json)")
@click.option("--force", "-f", is_flag=True, default=False,
              help="Overwrite existing camera_trajectory.csv")
def main(input_dir, calibration, quest_file, force):
    """Batch-transform Quest trajectories to camera frame."""
    input_dir = Path(input_dir).expanduser().absolute()

    # Load calibration
    print(f"Loading calibration from {calibration}...")
    R_world, t, s, time_offset, R_body = load_calibration(Path(calibration))
    print(f"  Scale: {s:.4f}, time offset: {time_offset*1000:.1f}ms")

    # Find episodes with quest data
    episodes = sorted([
        p.parent for p in input_dir.glob(f"*/{quest_file}")
    ])
    print(f"Found {len(episodes)} episodes with {quest_file}")

    if not episodes:
        raise click.ClickException(f"No {quest_file} found under {input_dir}")

    n_processed = 0
    n_skipped = 0
    for ep_dir in episodes:
        if not force and (ep_dir / "camera_trajectory.csv").is_file():
            print(f"  Skipping {ep_dir.name} (camera_trajectory.csv exists)")
            n_skipped += 1
            continue

        n_frames = transform_episode(ep_dir, R_world, t, s, time_offset, R_body, quest_file)
        if n_frames > 0:
            print(f"  {ep_dir.name}: {n_frames} frames")
            n_processed += 1
        else:
            print(f"  {ep_dir.name}: no quest data")

    print(f"\nDone: {n_processed} processed, {n_skipped} skipped")


if __name__ == "__main__":
    main()
