"""Trajectory data layer: CSV parsing, quaternion conversion, and joint-angle
interpolation. The SLAM trajectory *quality* check lives in
grabette_postprocess.checks.trajectory (it builds on load_trajectory_csv here)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def load_trajectory_csv(path: Path) -> pd.DataFrame:
    """Load SLAM trajectory CSV.

    Columns: frame_idx, timestamp, state, is_lost, is_keyframe,
             x, y, z, q_x, q_y, q_z, q_w
    """
    return pd.read_csv(path)


def quaternion_to_axis_angle(qx: np.ndarray, qy: np.ndarray,
                             qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    """Convert quaternions to compact axis-angle (rotation vector).

    Args:
        qx, qy, qz, qw: arrays of shape (N,)

    Returns:
        (N, 3) rotation vectors (axis * angle in radians)
    """
    quats = np.stack([qx, qy, qz, qw], axis=-1)
    return Rotation.from_quat(quats, scalar_first=False).as_rotvec()


def trajectory_to_poses(df: pd.DataFrame) -> np.ndarray:
    """Convert trajectory DataFrame to (N, 6) pose array [x, y, z, ax, ay, az].

    Lost frames get all zeros.

    Args:
        df: trajectory DataFrame from load_trajectory_csv()

    Returns:
        (N, 6) float32 array: position + axis-angle
    """
    n = len(df)
    poses = np.zeros((n, 6), dtype=np.float32)

    tracked = ~df['is_lost'].astype(bool)
    if tracked.any():
        pos = df.loc[tracked, ['x', 'y', 'z']].values
        rotvec = quaternion_to_axis_angle(
            df.loc[tracked, 'q_x'].values,
            df.loc[tracked, 'q_y'].values,
            df.loc[tracked, 'q_z'].values,
            df.loc[tracked, 'q_w'].values,
        )
        poses[tracked, :3] = pos
        poses[tracked, 3:] = rotvec

    return poses


def _load_angle_stream(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (cts_seconds, values) for the gripper joint-angle stream.

    Schema (angle_data.json):
        {"samples": [{"cts": <ms>, "value": [distal, proximal]}]}
    cts is already relative to recording start. Values keep their native
    [distal, proximal] order (the caller swaps).
    """
    samples = data["samples"]
    cts = np.array([s["cts"] for s in samples]) * 1e-3
    vals = np.array([s["value"] for s in samples])
    return cts, vals


def interpolate_angles(angle_json_path: Path,
                       video_timestamps: np.ndarray) -> np.ndarray:
    """Interpolate the gripper joint-angle stream to video/trajectory timestamps.

    Reads angle_data.json (flat schema). The stream stores value=[distal,
    proximal]; this returns [proximal, distal] to match the kinematic chain order.

    Args:
        angle_json_path: path to angle_data.json
        video_timestamps: (N,) array of timestamps in seconds (recording-relative)

    Returns:
        (N, 2) float32 array: [proximal, distal] in radians
    """
    with open(angle_json_path) as f:
        data = json.load(f)

    angl_cts, angl_vals = _load_angle_stream(data)

    n = len(video_timestamps)
    angles = np.zeros((n, 2), dtype=np.float32)
    # Interpolate each axis, then swap distal/proximal -> proximal/distal
    for i, axis in enumerate([1, 0]):  # proximal=index1, distal=index0
        angles[:, i] = np.interp(video_timestamps, angl_cts, angl_vals[:, axis])

    return angles
