#!/usr/bin/env python3
"""
Compare SLAM trajectory with an external reference trajectory (e.g. Meta Quest).

Finds the rigid transformation (rotation + translation + optional scale) that
best aligns the two trajectories using Umeyama alignment, then reports the
residual error (ATE — Absolute Trajectory Error).

The two trajectories may be in completely different coordinate frames and have
different time bases. The script:
1. Aligns timestamps via cross-correlation of velocity profiles
2. Interpolates to common timestamps
3. Finds the SE3 (or Sim3) transform via Umeyama alignment
4. Reports ATE statistics and optionally saves a plot

Usage:
    uv run python scripts/compare_trajectories.py \
        --slam /path/to/camera_trajectory.csv \
        --reference /path/to/r_hand_traj.json \
        --plot comparison.png
"""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def load_slam_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load SLAM trajectory CSV.

    Returns (timestamps_s, positions_Nx3) for tracked frames only.
    """
    df = pd.read_csv(path)
    tracked = df[~df["is_lost"].astype(bool)]
    timestamps = tracked["timestamp"].values
    positions = tracked[["x", "y", "z"]].values
    return timestamps, positions


def load_quest_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Meta Quest hand trajectory JSON (list of {timestamp_ms, pose: [16 floats]}).

    Returns (timestamps_s, positions_Nx3). Timestamps are zero-based.
    """
    with open(path) as f:
        data = json.load(f)

    timestamps = np.array([entry["timestamp_ms"] for entry in data], dtype=np.float64)
    timestamps = (timestamps - timestamps[0]) / 1000.0  # zero-based, seconds

    # Extract position from 4x4 column-major pose matrix
    positions = []
    for entry in data:
        pose = np.array(entry["pose"]).reshape(4, 4)
        # Position is the translation column (last column, rows 0-2)
        # But check: could be row-major. Let's check the last row.
        if abs(pose[3, 0]) < 1e-6 and abs(pose[3, 1]) < 1e-6 and abs(pose[3, 3] - 1.0) < 1e-6:
            # Last row is [0, 0, 0, 1] → row-major, translation in column 3
            positions.append(pose[:3, 3])
        else:
            # Try column-major (transpose)
            pose = pose.T
            positions.append(pose[:3, 3])

    return timestamps, np.array(positions)


def align_timestamps(
    t1: np.ndarray, pos1: np.ndarray,
    t2: np.ndarray, pos2: np.ndarray,
) -> float:
    """Find the time offset between two trajectories by cross-correlating velocity profiles.

    Returns offset such that t2_aligned = t2 + offset.
    """
    # Compute velocity norms
    dt1 = np.diff(t1)
    dt1[dt1 == 0] = 1e-6
    vel1 = np.linalg.norm(np.diff(pos1, axis=0), axis=1) / dt1

    dt2 = np.diff(t2)
    dt2[dt2 == 0] = 1e-6
    vel2 = np.linalg.norm(np.diff(pos2, axis=0), axis=1) / dt2

    # Midpoint timestamps for velocity
    t1_mid = (t1[:-1] + t1[1:]) / 2
    t2_mid = (t2[:-1] + t2[1:]) / 2

    # Resample to uniform grid
    dt = 0.01  # 100Hz
    t_start = 0
    t_end = min(t1_mid[-1], t2_mid[-1])
    t_uniform = np.arange(t_start, t_end, dt)

    v1 = np.interp(t_uniform, t1_mid, vel1)
    v2 = np.interp(t_uniform, t2_mid, vel2)

    # Normalize
    v1 = (v1 - np.mean(v1))
    v1_std = np.std(v1)
    if v1_std > 0:
        v1 /= v1_std
    v2 = (v2 - np.mean(v2))
    v2_std = np.std(v2)
    if v2_std > 0:
        v2 /= v2_std

    # Cross-correlate (search ±5 seconds)
    max_lag = int(5.0 / dt)
    lags = np.arange(-max_lag, max_lag + 1)
    n = len(t_uniform)
    corr = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            a, b = v1[lag:], v2[:n - lag]
        else:
            a, b = v1[:n + lag], v2[-lag:]
        if len(a) > 0:
            corr[i] = np.mean(a * b)

    best_idx = np.argmax(corr)
    best_lag = lags[best_idx] * dt
    return best_lag


def umeyama_alignment(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama alignment: find R, t, s such that dst ≈ s * R @ src + t.

    Args:
        src: (N, 3) source points
        dst: (N, 3) destination points
        with_scale: estimate scale factor

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scale factor (1.0 if with_scale=False)
    """
    assert src.shape == dst.shape
    n, d = src.shape

    # Centroids
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    # Centered points
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Covariance
    sigma = dst_c.T @ src_c / n

    # SVD
    U, D, Vt = np.linalg.svd(sigma)

    # Handle reflection
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[d - 1, d - 1] = -1

    R = U @ S @ Vt

    if with_scale:
        var_src = np.sum(src_c ** 2) / n
        s = np.trace(np.diag(D) @ S) / var_src if var_src > 0 else 1.0
    else:
        s = 1.0

    t = mu_dst - s * R @ mu_src

    return R, t, s


@click.command()
@click.option("--slam", required=True, type=click.Path(exists=True),
              help="SLAM trajectory CSV (camera_trajectory.csv)")
@click.option("--reference", required=True, type=click.Path(exists=True),
              help="Reference trajectory JSON (e.g. r_hand_traj.json)")
@click.option("--no-scale", is_flag=True, default=False,
              help="Don't estimate scale (assume both are metric)")
@click.option("--plot", "-p", type=click.Path(), default=None,
              help="Save comparison plot to file")
def main(slam, reference, no_scale, plot):
    """Compare SLAM trajectory with a reference (e.g. Meta Quest)."""

    # Load trajectories
    print("Loading SLAM trajectory...")
    slam_ts, slam_pos = load_slam_trajectory(Path(slam))
    print(f"  {len(slam_ts)} tracked frames, {slam_ts[-1]:.1f}s")

    print("Loading reference trajectory...")
    ref_ts, ref_pos = load_quest_trajectory(Path(reference))
    print(f"  {len(ref_ts)} samples, {ref_ts[-1]:.1f}s")

    # Align timestamps
    print("\nAligning timestamps via velocity cross-correlation...")
    time_offset = align_timestamps(slam_ts, slam_pos, ref_ts, ref_pos)
    print(f"  Time offset: {time_offset*1000:.1f}ms")

    ref_ts_aligned = ref_ts + time_offset

    # Interpolate reference to SLAM timestamps
    overlap_start = max(slam_ts[0], ref_ts_aligned[0])
    overlap_end = min(slam_ts[-1], ref_ts_aligned[-1])
    mask = (slam_ts >= overlap_start) & (slam_ts <= overlap_end)
    common_ts = slam_ts[mask]
    slam_common = slam_pos[mask]

    ref_common = np.zeros_like(slam_common)
    for axis in range(3):
        ref_common[:, axis] = np.interp(common_ts, ref_ts_aligned, ref_pos[:, axis])

    print(f"  Overlap: {len(common_ts)} frames, {common_ts[-1]-common_ts[0]:.1f}s")

    # Umeyama alignment: find transform from SLAM to reference
    print("\nUmeyama alignment (SLAM → reference)...")
    R, t, s = umeyama_alignment(slam_common, ref_common, with_scale=not no_scale)
    print(f"  Scale: {s:.4f}")
    print(f"  Rotation (axis-angle): {Rotation.from_matrix(R).as_rotvec() * 180 / np.pi} deg")
    print(f"  Translation: {t}")

    # Apply transform
    slam_aligned = s * (R @ slam_common.T).T + t

    # Compute ATE (Absolute Trajectory Error)
    errors = np.linalg.norm(slam_aligned - ref_common, axis=1)
    ate_rmse = np.sqrt(np.mean(errors ** 2))
    ate_mean = np.mean(errors)
    ate_median = np.median(errors)
    ate_max = np.max(errors)

    print(f"\n{'='*50}")
    print(f"  Absolute Trajectory Error (ATE)")
    print(f"  RMSE:   {ate_rmse*1000:.1f} mm")
    print(f"  Mean:   {ate_mean*1000:.1f} mm")
    print(f"  Median: {ate_median*1000:.1f} mm")
    print(f"  Max:    {ate_max*1000:.1f} mm")
    print(f"  Scale:  {s:.4f}")
    print(f"{'='*50}")

    if ate_rmse < 0.01:
        print(f"\n  RESULT: EXCELLENT (RMSE < 10mm)")
    elif ate_rmse < 0.03:
        print(f"\n  RESULT: GOOD (RMSE < 30mm)")
    elif ate_rmse < 0.10:
        print(f"\n  RESULT: FAIR (RMSE < 100mm)")
    else:
        print(f"\n  RESULT: POOR (RMSE >= 100mm)")

    # Optional plot
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(16, 10))

            # 3D trajectories
            ax1 = fig.add_subplot(2, 2, 1, projection="3d")
            ax1.plot(*slam_aligned.T, label="SLAM (aligned)", alpha=0.8)
            ax1.plot(*ref_common.T, label="Reference", alpha=0.8)
            ax1.set_xlabel("X (m)")
            ax1.set_ylabel("Y (m)")
            ax1.set_zlabel("Z (m)")
            ax1.set_title("3D Trajectories (aligned)")
            ax1.legend()

            # Per-axis comparison
            ax2 = fig.add_subplot(2, 2, 2)
            for i, label in enumerate(["X", "Y", "Z"]):
                ax2.plot(common_ts, slam_aligned[:, i], label=f"SLAM {label}", linestyle="-")
                ax2.plot(common_ts, ref_common[:, i], label=f"Ref {label}", linestyle="--")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Position (m)")
            ax2.set_title("Per-axis comparison")
            ax2.legend(fontsize=7)

            # Error over time
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.plot(common_ts, errors * 1000)
            ax3.axhline(y=ate_rmse * 1000, color="r", linestyle="--",
                       label=f"RMSE: {ate_rmse*1000:.1f}mm")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Error (mm)")
            ax3.set_title("ATE over time")
            ax3.legend()

            # Error histogram
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.hist(errors * 1000, bins=50, edgecolor="black")
            ax4.axvline(x=ate_rmse * 1000, color="r", linestyle="--",
                       label=f"RMSE: {ate_rmse*1000:.1f}mm")
            ax4.set_xlabel("Error (mm)")
            ax4.set_ylabel("Count")
            ax4.set_title("Error distribution")
            ax4.legend()

            plt.tight_layout()
            plt.savefig(plot, dpi=150)
            print(f"\nPlot saved to: {plot}")
        except ImportError:
            print("\nWarning: matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()
