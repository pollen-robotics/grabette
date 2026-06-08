#!/usr/bin/env python3
"""
Transform a Meta Quest trajectory into the SLAM camera reference frame.

Three modes:
  1. With --slam: compute the transform from a SLAM trajectory (Umeyama alignment)
  2. With --calibration + single file: apply a pre-computed transform from a calibration file
  3. With --calibration + directory: batch mode — process all subdirectories that contain
     r_hand_traj.json and write camera_trajectory.csv into each

Usage:
    # Compute transform from SLAM trajectory
    uv run python scripts/transform_quest_trajectory.py \
        --slam camera_trajectory.csv \
        --quest r_hand_traj.json \
        -o quest_camera.csv

    # Apply saved calibration (no SLAM needed)
    uv run python scripts/transform_quest_trajectory.py \
        --quest r_hand_traj.json \
        --calibration config/quest_to_camera_calibration.json \
        -o quest_camera.csv

    # Batch mode: apply calibration to all episodes under a directory
    uv run python scripts/transform_quest_trajectory.py \
        --quest /data/episodes/ \
        --calibration config/quest_to_camera_calibration.json

    # Compute and save calibration for future use
    uv run python scripts/transform_quest_trajectory.py \
        --slam camera_trajectory.csv \
        --quest r_hand_traj.json \
        -o quest_camera.csv \
        --save-calibration config/quest_to_camera_calibration.json
"""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def load_slam_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SLAM trajectory. Returns (timestamps, positions, quaternions) for tracked frames."""
    df = pd.read_csv(path)
    tracked = df[~df["is_lost"].astype(bool)]
    return (tracked["timestamp"].values,
            tracked[["x", "y", "z"]].values,
            tracked[["q_x", "q_y", "q_z", "q_w"]].values)


def load_quest_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Quest trajectory. Returns (timestamps_s, positions, rotation_matrices)."""
    with open(path) as f:
        data = json.load(f)

    timestamps = np.array([e["timestamp_ms"] for e in data], dtype=np.float64)
    timestamps = (timestamps - timestamps[0]) / 1000.0

    positions = []
    rotations = []
    for entry in data:
        pose = np.array(entry["pose"]).reshape(4, 4)
        if abs(pose[3, 0]) < 1e-6 and abs(pose[3, 1]) < 1e-6 and abs(pose[3, 3] - 1.0) < 1e-6:
            positions.append(pose[:3, 3])
            rotations.append(pose[:3, :3])
        else:
            pose = pose.T
            positions.append(pose[:3, 3])
            rotations.append(pose[:3, :3])

    return timestamps, np.array(positions), np.array(rotations)


def align_timestamps(t1, pos1, t2, pos2) -> float:
    """Find time offset via velocity cross-correlation."""
    dt1 = np.diff(t1); dt1[dt1 == 0] = 1e-6
    vel1 = np.linalg.norm(np.diff(pos1, axis=0), axis=1) / dt1
    dt2 = np.diff(t2); dt2[dt2 == 0] = 1e-6
    vel2 = np.linalg.norm(np.diff(pos2, axis=0), axis=1) / dt2

    t1_mid = (t1[:-1] + t1[1:]) / 2
    t2_mid = (t2[:-1] + t2[1:]) / 2

    dt = 0.01
    t_end = min(t1_mid[-1], t2_mid[-1])
    t_uniform = np.arange(0, t_end, dt)
    v1 = np.interp(t_uniform, t1_mid, vel1)
    v2 = np.interp(t_uniform, t2_mid, vel2)

    v1 = v1 - np.mean(v1); std1 = np.std(v1)
    if std1 > 0: v1 /= std1
    v2 = v2 - np.mean(v2); std2 = np.std(v2)
    if std2 > 0: v2 /= std2

    max_lag = int(5.0 / dt)
    lags = np.arange(-max_lag, max_lag + 1)
    n = len(t_uniform)
    corr = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0: a, b = v1[lag:], v2[:n - lag]
        else: a, b = v1[:n + lag], v2[-lag:]
        if len(a) > 0: corr[i] = np.mean(a * b)

    return lags[np.argmax(corr)] * dt


def umeyama_alignment(src, dst, with_scale=True):
    """Find R, t, s such that dst ~ s * R @ src + t."""
    n, d = src.shape
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    sigma = dst_c.T @ src_c / n
    U, D, Vt = np.linalg.svd(sigma)
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


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Load saved calibration. Returns (R_world, t, scale, time_offset_s, R_body)."""
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


def save_calibration(path: Path, R, t, s, time_offset_s, ate_rmse, source=""):
    """Save calibration to JSON."""
    quat = Rotation.from_matrix(R).as_quat().tolist()  # xyzw
    cal = {
        "description": "Transform from Meta Quest right-hand controller frame to SLAM camera frame",
        "source": source,
        "ate_rmse_mm": round(ate_rmse * 1000, 1),
        "scale": float(s),
        "rotation_quaternion_xyzw": quat,
        "translation": t.tolist(),
        "time_offset_ms": round(time_offset_s * 1000, 1),
        "note": "Apply as: p_cam = scale * R @ p_quest + t",
    }
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)
        f.write("\n")
    print(f"Calibration saved to: {path}")


def apply_transform(quest_ts, quest_pos, quest_rots, R_world, t, s, time_offset,
                    R_body=None):
    """Apply rigid transform to Quest trajectory. Returns (aligned_ts, positions, quaternions).

    Position: p_cam = s * R_world @ p_quest + t
    Orientation: R_cam = R_world @ R_quest @ R_body
    """
    if R_body is None:
        R_body = np.eye(3)
    ts_aligned = quest_ts + time_offset
    pos_transformed = s * (R_world @ quest_pos.T).T + t
    quats = []
    for R_quest in quest_rots:
        R_cam = R_world @ R_quest @ R_body
        quats.append(Rotation.from_matrix(R_cam).as_quat())  # xyzw
    return ts_aligned, pos_transformed, np.array(quats)


def save_trajectory_csv(path: Path, timestamps, positions, quaternions):
    """Save trajectory in camera_trajectory.csv format."""
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
    df.to_csv(path, index=False)
    return df


def process_single(quest_path: Path, output_path: Path, slam, calibration,
                   save_calibration, no_scale):
    """Process a single Quest trajectory file."""
    # Load Quest trajectory
    print("Loading Quest trajectory...")
    quest_ts, quest_pos, quest_rots = load_quest_trajectory(quest_path)
    print(f"  {len(quest_ts)} samples, {quest_ts[-1]:.1f}s")

    R_body = np.eye(3)

    if calibration:
        # Mode 2: Apply saved calibration
        print(f"Loading calibration from {calibration}...")
        R, t, s, time_offset, R_body = load_calibration(Path(calibration))
        if no_scale:
            s = 1.0
        print(f"  Scale: {s:.4f}, time offset: {time_offset*1000:.1f}ms")

    else:
        # Mode 1: Compute transform from SLAM trajectory
        print("Loading SLAM trajectory...")
        slam_ts, slam_pos, slam_quats = load_slam_trajectory(Path(slam))
        print(f"  {len(slam_ts)} tracked frames, {slam_ts[-1]:.1f}s")

        print("Aligning timestamps...")
        time_offset = align_timestamps(slam_ts, slam_pos, quest_ts, quest_pos)
        print(f"  Time offset: {time_offset*1000:.1f}ms")

        quest_ts_aligned = quest_ts + time_offset
        overlap_start = max(slam_ts[0], quest_ts_aligned[0])
        overlap_end = min(slam_ts[-1], quest_ts_aligned[-1])
        mask = (slam_ts >= overlap_start) & (slam_ts <= overlap_end)
        slam_common = slam_pos[mask]
        slam_quats_common = slam_quats[mask]
        common_ts = slam_ts[mask]

        quest_common = np.zeros_like(slam_common)
        for axis in range(3):
            quest_common[:, axis] = np.interp(common_ts, quest_ts_aligned, quest_pos[:, axis])

        print("Computing position alignment (Umeyama)...")
        R, t, s = umeyama_alignment(quest_common, slam_common, with_scale=not no_scale)

        transformed = s * (R @ quest_common.T).T + t
        ate = np.sqrt(np.mean(np.linalg.norm(transformed - slam_common, axis=1)**2))
        print(f"  Scale: {s:.4f}")
        print(f"  ATE RMSE: {ate*1000:.1f}mm")

        # Compute body frame rotation offset: R_cam = R_world @ R_quest @ R_body
        # => R_body = (R_world @ R_quest)^T @ R_slam
        print("Computing orientation offset...")
        offsets = []
        for i in range(len(common_ts)):
            idx = np.argmin(np.abs(quest_ts_aligned - common_ts[i]))
            if abs(quest_ts_aligned[idx] - common_ts[i]) > 0.05:
                continue
            R_slam = Rotation.from_quat(slam_quats_common[i]).as_matrix()
            R_quest_i = quest_rots[idx]
            R_combined = R @ R_quest_i
            offsets.append(R_combined.T @ R_slam)

        if offsets:
            offset_quats = Rotation.from_matrix(np.array(offsets)).as_quat()
            mean_quat = offset_quats.mean(axis=0)
            mean_quat /= np.linalg.norm(mean_quat)
            R_body = Rotation.from_quat(mean_quat).as_matrix()
            angle_std = np.std(np.linalg.norm(
                Rotation.from_matrix(np.array(offsets)).as_rotvec()
                - Rotation.from_matrix(np.array(offsets)).as_rotvec().mean(axis=0),
                axis=1))
            print(f"  Body rotation offset: {Rotation.from_matrix(R_body).as_rotvec() * 180/np.pi} deg")
            print(f"  Orientation std: {np.degrees(angle_std):.2f} deg ({len(offsets)} pairs)")

        if save_calibration:
            cal = {
                "description": "Transform from Meta Quest right-hand controller frame to SLAM camera frame",
                "source": f"Umeyama alignment + orientation offset from {Path(slam).parent.name}",
                "ate_rmse_mm": round(ate * 1000, 1),
                "orientation_std_deg": round(float(np.degrees(angle_std)), 2) if offsets else None,
                "scale": float(s),
                "rotation_quaternion_xyzw": Rotation.from_matrix(R).as_quat().tolist(),
                "translation": t.tolist(),
                "body_rotation_quaternion_xyzw": Rotation.from_matrix(R_body).as_quat().tolist(),
                "time_offset_ms": round(time_offset * 1000, 1),
                "note": "Position: p_cam = scale * R_world @ p_quest + t. Orientation: R_cam = R_world @ R_quest @ R_body",
            }
            with open(Path(save_calibration), "w") as f:
                json.dump(cal, f, indent=2)
                f.write("\n")
            print(f"  Calibration saved to: {save_calibration}")

    # Apply transform
    print("Transforming Quest trajectory...")
    ts_out, pos_out, quats_out = apply_transform(
        quest_ts, quest_pos, quest_rots, R, t, s, time_offset, R_body,
    )

    df = save_trajectory_csv(output_path, ts_out, pos_out, quats_out)
    print(f"Saved {len(df)} frames to {output_path}")
    print(f"  Time range: {ts_out[0]:.2f} - {ts_out[-1]:.2f}s")


@click.command()
@click.option("--quest", required=True, type=click.Path(exists=True),
              help="Quest trajectory JSON (r_hand_traj.json) or directory for batch mode")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output CSV in camera frame (single-file mode only)")
@click.option("--slam", type=click.Path(exists=True), default=None,
              help="SLAM trajectory CSV (for computing transform)")
@click.option("--calibration", "-c", type=click.Path(exists=True), default=None,
              help="Saved calibration JSON (alternative to --slam)")
@click.option("--save-calibration", type=click.Path(), default=None,
              help="Save computed calibration to this file")
@click.option("--no-scale", is_flag=True, default=False,
              help="Don't correct scale")
@click.option("--force", "-f", is_flag=True, default=False,
              help="Overwrite existing camera_trajectory.csv files (batch mode only)")
def main(quest, output, slam, calibration, save_calibration, no_scale, force):
    """Transform Quest trajectory into camera reference frame.

    If --quest is a directory and --calibration is provided, runs in batch mode:
    finds all subdirectories containing r_hand_traj.json and writes
    camera_trajectory.csv into each one.
    """
    if slam is None and calibration is None:
        raise click.UsageError("Provide either --slam (to compute transform) or --calibration (to apply saved transform)")

    quest_path = Path(quest)

    # Batch mode: --quest is a directory and --calibration is provided
    if quest_path.is_dir():
        if calibration is None:
            raise click.UsageError("Batch mode (--quest is a directory) requires --calibration")
        if output is not None:
            raise click.UsageError("--output cannot be used in batch mode; output is written as camera_trajectory.csv in each episode directory")

        episode_dirs = sorted([p.parent for p in quest_path.glob("*/r_hand_traj.json")])
        print(f"Found {len(episode_dirs)} directories with r_hand_traj.json")
        if not episode_dirs:
            raise click.ClickException(f"No r_hand_traj.json found under {quest_path}")

        skipped = 0
        processed = 0
        failed = 0
        for ep_dir in episode_dirs:
            out_file = ep_dir / "camera_trajectory.csv"
            if out_file.exists() and not force:
                print(f"[skip] {ep_dir.name} (camera_trajectory.csv exists, use --force to overwrite)")
                skipped += 1
                continue
            print(f"\n[{processed + failed + 1}/{len(episode_dirs) - skipped}] Processing {ep_dir.name}...")
            try:
                process_single(ep_dir / "r_hand_traj.json", out_file,
                               slam=None, calibration=calibration,
                               save_calibration=None, no_scale=no_scale)
                processed += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                failed += 1

        print(f"\nDone: {processed} processed, {skipped} skipped, {failed} failed")
        return

    # Single-file mode
    if output is None:
        raise click.UsageError("--output / -o is required in single-file mode")

    process_single(quest_path, Path(output), slam=slam, calibration=calibration,
                   save_calibration=save_calibration, no_scale=no_scale)


if __name__ == "__main__":
    main()
