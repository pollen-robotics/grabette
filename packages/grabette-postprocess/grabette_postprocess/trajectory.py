"""Trajectory CSV parsing, quaternion conversion, and joint angle interpolation."""

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


def interpolate_angles(imu_json_path: Path,
                       video_timestamps: np.ndarray) -> np.ndarray:
    """Interpolate ANGL stream (100Hz) to video frame timestamps.

    The raw ANGL stream stores value=[distal, proximal]. This function swaps
    them to return [proximal, distal] which matches the kinematic chain order.

    Args:
        imu_json_path: path to raw imu_data.json (not resampled — ANGL is stripped there)
        video_timestamps: (N,) array of video frame timestamps in seconds

    Returns:
        (N, 2) float32 array: [proximal, distal] in radians
    """
    with open(imu_json_path) as f:
        data = json.load(f)

    angl_samples = data['1']['streams']['ANGL']['samples']

    # ANGL timestamps are in ms, convert to seconds
    angl_cts = np.array([s['cts'] for s in angl_samples]) * 1e-3
    angl_vals = np.array([s['value'] for s in angl_samples])  # [distal, proximal]

    # Zero-base ANGL timestamps to match video timestamps (same as CORI in LoadTelemetry)
    # Video timestamps are frame_idx/fps, starting at 0
    # ANGL cts are absolute; we need to align them.
    # The ACCL stream's first timestamp is subtracted from IMU timestamps in LoadTelemetry.
    # ANGL timestamps use the same clock, so subtract the same offset.
    # We load ACCL to find the offset.
    accl_samples = data['1']['streams']['ACCL']['samples']
    imu_start_t = accl_samples[0]['cts'] * 1e-3
    angl_cts = angl_cts - imu_start_t

    n = len(video_timestamps)
    angles = np.zeros((n, 2), dtype=np.float32)

    # Interpolate each axis, then swap distal/proximal -> proximal/distal
    for i, axis in enumerate([1, 0]):  # proximal=index1, distal=index0
        angles[:, i] = np.interp(video_timestamps, angl_cts, angl_vals[:, axis])

    return angles


def load_gravity(path: Path) -> np.ndarray:
    """Load 3x3 gravity rotation matrix from CSV.

    Returns:
        (3, 3) float64 array
    """
    return np.loadtxt(path, delimiter=',')
