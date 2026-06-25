#!/usr/bin/env python3
"""
Visualize OAK-D RGBD SLAM trajectory using Rerun.

Usage:
    uv run python scripts/rgbd_slam/visualize_rgbd_trajectory.py <episode_dir>
    uv run python scripts/rgbd_slam/visualize_rgbd_trajectory.py <episode_dir> --video-skip 3
"""

import json
import sys
import time
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation


def _gravity_align(positions: np.ndarray, quaternions: np.ndarray,
                   oak_dir: Path, n_samples: int = 50,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate trajectory so gravity (in WORLD frame) points to world -Z.

    Critical: we have to compute gravity in the *world frame*, not the
    camera frame, before computing the align rotation. With rtabmap's
    IMU-init, the first pose's orientation is non-identity (world ≠ camera
    at t=0), so camera-frame gravity and world-frame gravity differ by
    that initial rotation.

    Pipeline:
      1. Average first `n_samples` accel readings → gravity in IMU frame.
      2. Transform to optical camera frame via calib's imu_to_cam.
      3. Rotate through the first pose's orientation to get gravity in
         the trajectory's world frame.
      4. Find shortest-arc rotation R_align that maps world-gravity to (0,0,-1).
      5. Apply R_align (left-multiply) to all poses.

    Returns (positions, quaternions, g_world_unit) where g_world_unit is the
    measured gravity direction in the original (pre-align) world frame.
    """
    imu_acc = pd.read_csv(oak_dir / "imu_acc.csv")
    g_imu = imu_acc[["ax", "ay", "az"]].iloc[:n_samples].mean().to_numpy()

    calib = json.loads((oak_dir / "calib_offline.json").read_text())
    R_imu_to_cam = np.array(calib["imu_to_cam"])[:3, :3]
    # Accelerometer at rest = SPECIFIC FORCE (points UP). Negate for physical
    # gravity (DOWN) so the align-to-(0,0,-1) below gives a Z-UP world.
    g_cam = -(R_imu_to_cam @ g_imu)
    g_cam_unit = g_cam / np.linalg.norm(g_cam)

    # Project gravity into world frame using the FIRST pose's orientation.
    R_world_from_cam0 = Rotation.from_quat(quaternions[0])
    g_world_unit = R_world_from_cam0.apply(g_cam_unit)

    # Align world gravity to (0, 0, -1).
    R_align, _ = Rotation.align_vectors([[0.0, 0.0, -1.0]], [g_world_unit])

    new_pos = R_align.apply(positions)
    new_quats = (R_align * Rotation.from_quat(quaternions)).as_quat()
    return new_pos, new_quats, g_world_unit



def _load_oak_imu(oak_dir: Path) -> dict | None:
    """Load IMU from OAK imu_acc.csv / imu_gyro.csv (timestamp_ns, x, y, z)."""
    acc_path = oak_dir / "imu_acc.csv"
    gyro_path = oak_dir / "imu_gyro.csv"
    if not acc_path.is_file() or not gyro_path.is_file():
        return None

    def _read(path):
        samples = []
        with open(path) as f:
            next(f)  # skip header
            for line in f:
                p = line.strip().split(',')
                if len(p) < 4:
                    continue
                samples.append({
                    'timestamp': float(p[0]) * 1e-9,
                    'value': [float(p[1]), float(p[2]), float(p[3])],
                })
        return samples

    accel = _read(acc_path)
    gyro = _read(gyro_path)
    if not accel and not gyro:
        return None
    return {'accel': accel, 'gyro': gyro}


def _log_oak_imu(imu_data: dict, t0: float):
    rr.log("imu/accelerometer/x", rr.SeriesLines(colors=[255, 0, 0], names="accel_x"), static=True)
    rr.log("imu/accelerometer/y", rr.SeriesLines(colors=[0, 255, 0], names="accel_y"), static=True)
    rr.log("imu/accelerometer/z", rr.SeriesLines(colors=[0, 0, 255], names="accel_z"), static=True)
    rr.log("imu/gyroscope/x", rr.SeriesLines(colors=[255, 128, 0], names="gyro_x"), static=True)
    rr.log("imu/gyroscope/y", rr.SeriesLines(colors=[128, 255, 0], names="gyro_y"), static=True)
    rr.log("imu/gyroscope/z", rr.SeriesLines(colors=[0, 128, 255], names="gyro_z"), static=True)

    for s in imu_data['accel']:
        rr.set_time("time", timestamp=s['timestamp'] - t0)
        v = s['value']
        rr.log("imu/accelerometer/x", rr.Scalars(float(v[0])))
        rr.log("imu/accelerometer/y", rr.Scalars(float(v[1])))
        rr.log("imu/accelerometer/z", rr.Scalars(float(v[2])))

    for s in imu_data['gyro']:
        rr.set_time("time", timestamp=s['timestamp'] - t0)
        v = s['value']
        rr.log("imu/gyroscope/x", rr.Scalars(float(v[0])))
        rr.log("imu/gyroscope/y", rr.Scalars(float(v[1])))
        rr.log("imu/gyroscope/z", rr.Scalars(float(v[2])))



@click.command()
@click.argument('episode_dir', type=click.Path(exists=True))
@click.option('--show-video/--no-video', default=True, help='Show camera frames')
@click.option('--video-skip', default=5, help='Show every Nth frame')
@click.option('--app-id', default='grabette_viz', help='Rerun application ID')
@click.option('--gravity-align/--no-gravity-align', default=False,
              help='Re-align trajectory so first IMU gravity maps to world -Z. '
                   'Normally NOT needed — run_oak_slam already does this when '
                   'producing camera_trajectory.csv. Use for legacy CSVs that '
                   'predate the in-pipeline gravity align.')
def main(episode_dir, show_video, video_skip, app_id, gravity_align):
    """Visualize OAK SLAM trajectory from a processed episode directory."""
    episode_dir = Path(episode_dir)
    oak_dir = episode_dir / "oak"

    traj_csv = episode_dir / "camera_trajectory.csv"
    if not traj_csv.exists():
        print(f"Error: No camera_trajectory.csv in {episode_dir}")
        sys.exit(1)

    # --- Load trajectory ---
    print(f"Loading trajectory from {traj_csv.name}...")
    df_all = pd.read_csv(traj_csv)
    df_valid = df_all[~df_all['is_lost'].astype(bool)].copy()
    n_total, n_tracked = len(df_all), len(df_valid)

    print(f"\n=== SLAM Statistics ===")
    print(f"  Frames:   {n_tracked}/{n_total} tracked ({100*n_tracked/n_total:.1f}%)")
    print(f"  Lost:     {n_total - n_tracked} frames")
    if n_tracked > 0:
        duration = df_valid.iloc[-1]['timestamp'] - df_valid.iloc[0]['timestamp']
        print(f"  Duration: {duration:.2f}s")

    positions = df_valid[['x', 'y', 'z']].to_numpy().copy()
    quaternions = df_valid[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy().copy()

    if n_tracked == 0:
        print("Error: No valid poses found!")
        sys.exit(1)

    g_unit_for_log = None  # gravity direction in trajectory frame (pre-align)
    if gravity_align:
        if not oak_dir.is_dir():
            print("Error: --gravity-align requires the oak/ subdir (run convert_episode_to_oak.py first)")
            sys.exit(1)
        positions, quaternions, g_unit_for_log = _gravity_align(
            positions, quaternions, oak_dir,
        )
        print(f"\n=== Gravity Alignment ===")
        print(f"  Measured g in SLAM camera frame (unit): "
              f"[{g_unit_for_log[0]:+.3f}, {g_unit_for_log[1]:+.3f}, {g_unit_for_log[2]:+.3f}]")
        print(f"  Rotated so that direction now maps to [0, 0, -1] (world -Z)")
        # Also rotate the position/quaternion columns we still need from df_valid
        df_valid = df_valid.copy()
        df_valid.loc[:, ['x', 'y', 'z']] = positions
        df_valid.loc[:, ['q_x', 'q_y', 'q_z', 'q_w']] = quaternions

    print(f"\n=== Trajectory Statistics ===")
    for ax, name in enumerate(['X', 'Y', 'Z']):
        lo, hi = positions[:, ax].min(), positions[:, ax].max()
        print(f"  {name}: [{lo:.4f}, {hi:.4f}]  range={hi-lo:.4f}")
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    print(f"  Total path length: {np.sum(distances):.3f} m")
    print(f"  Displacement:      {np.linalg.norm(positions[-1] - positions[0]):.3f} m\n")

    # Normalize timestamps to start from 0 so Rerun timeline is readable
    t0 = float(df_all['timestamp'].iloc[0])
    df_all = df_all.copy()
    df_all['timestamp'] = df_all['timestamp'] - t0

    # --- Load OAK IMU ---
    imu_data = _load_oak_imu(oak_dir) if oak_dir.is_dir() else None
    if imu_data:
        print(f"IMU: {len(imu_data['accel'])} accel, {len(imu_data['gyro'])} gyro samples")
    else:
        print("IMU: not found (expected oak/imu_acc.csv and oak/imu_gyro.csv)")

    # --- Open OAK left video (what the SLAM actually saw) ---
    # We use oakd_left.mp4 — NOT raw_video.mp4 — because:
    #   1. raw_video.mp4 is the RPi fisheye camera, not the SLAM input.
    #   2. oakd_left.mp4 is the rectified SLAM input; its frames are 1:1
    #      with the trajectory (encoder order == SLAM frame_idx), so no
    #      timestamp arithmetic is needed.
    video_path = episode_dir / "oakd_left.mp4"
    print(f"Looking for video at: {video_path}  (exists={video_path.exists()})")
    video_cap = None
    video_fps = None
    if show_video and video_path.exists():
        video_cap = cv2.VideoCapture(str(video_path))
        if not video_cap.isOpened():
            print("Warning: Could not open oakd_left.mp4")
            video_cap = None
        else:
            video_fps = video_cap.get(cv2.CAP_PROP_FPS)
            n_vframes = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"OAK left video: {n_vframes} frames at {video_fps:.2f} fps ({vid_w}x{vid_h})")
    elif show_video:
        print("Video: oakd_left.mp4 not found, camera feed disabled")

    # --- Initialize Rerun ---
    rr.init(app_id, spawn=True)
    time.sleep(0.5)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    if imu_data:
        print("Logging IMU data...")
        _log_oak_imu(imu_data, t0)

    # --- Blueprint ---
    try:
        top_views = [rrb.Spatial3DView(name="3D View", origin="/world")]
        if video_cap is not None:
            top_views.append(rrb.Spatial2DView(name="Camera", origin="/camera_feed"))
        bottom_views = []
        if imu_data:
            bottom_views.append(rrb.TimeSeriesView(name="Accelerometer", origin="/imu/accelerometer"))
            bottom_views.append(rrb.TimeSeriesView(name="Gyroscope", origin="/imu/gyroscope"))

        if bottom_views:
            blueprint = rrb.Blueprint(
                rrb.Vertical(rrb.Horizontal(*top_views), rrb.Horizontal(*bottom_views))
            )
        else:
            blueprint = rrb.Blueprint(rrb.Horizontal(*top_views))
        rr.send_blueprint(blueprint)
    except Exception as e:
        print(f"Warning: Could not send blueprint: {e}")

    # --- Static elements ---
    axis_len = 0.5
    rr.log("world/axes", rr.Arrows3D(
        origins=[[0, 0, 0]] * 3,
        vectors=[[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    ), static=True)
    rr.log("world/trajectory_full", rr.LineStrips3D(positions, colors=[0, 255, 0]), static=True)
    rr.log("world/start", rr.Points3D(positions[0], colors=[0, 255, 0], radii=0.01), static=True)
    rr.log("world/end", rr.Points3D(positions[-1], colors=[255, 0, 0], radii=0.01), static=True)

    # Gravity arrow in WORLD frame: rotate the camera-frame gravity vector
    # by the camera's first orientation to express it in world coords. After
    # `--gravity-align`, this should be ~(0, 0, -1) by construction (since the
    # align step now operates on world-frame gravity, not camera-frame).
    g_world = None
    if oak_dir.is_dir():
        try:
            imu_acc = pd.read_csv(oak_dir / "imu_acc.csv")
            g_imu = imu_acc[["ax", "ay", "az"]].iloc[:50].mean().to_numpy()
            calib = json.loads((oak_dir / "calib_offline.json").read_text())
            R_imu_to_cam = np.array(calib["imu_to_cam"])[:3, :3]
            # Negate: accel at rest is specific force (UP); physical gravity is DOWN.
            g_cam = -(R_imu_to_cam @ g_imu)
            g_cam /= np.linalg.norm(g_cam)
            R_world_from_cam = Rotation.from_quat(quaternions[0])
            g_world = R_world_from_cam.apply(g_cam)
        except Exception as e:
            print(f"(gravity arrow disabled: {e})")
    if g_world is not None:
        print(f"  Gravity in WORLD frame (should be ~(0,0,-1) with --gravity-align): "
              f"[{g_world[0]:+.3f}, {g_world[1]:+.3f}, {g_world[2]:+.3f}]")
        rr.log("world/gravity", rr.Arrows3D(
            origins=[positions[0].tolist()],
            vectors=[(g_world * 0.3).tolist()],
            colors=[[255, 255, 0]],
            labels=["g (world)"],
        ), static=True)

    # --- Build pose lookup ---
    pose_map = {}
    for _, row in df_valid.iterrows():
        pose_map[int(row['frame_idx'])] = {
            'pos': np.array([row['x'], row['y'], row['z']]),
            'quat': np.array([row['q_x'], row['q_y'], row['q_z'], row['q_w']]),
        }

    # --- Animate ---
    print(f"Visualizing {n_total} frames (skip={video_skip})...")
    trajectory_so_far = []
    cam_axis_len = 0.1

    for frame_i in range(n_total):
        if frame_i % video_skip != 0:
            continue

        row = df_all.iloc[frame_i]
        frame_idx = int(row['frame_idx'])
        t = float(row['timestamp'])
        rr.set_time("time", timestamp=t)

        if frame_idx in pose_map:
            p = pose_map[frame_idx]
            pos, quat = p['pos'], p['quat']

            rr.log("world/camera", rr.Transform3D(translation=pos.tolist(), quaternion=quat.tolist()))
            rr.log("world/current_position", rr.Points3D(pos, colors=[255, 0, 0], radii=0.005))

            trajectory_so_far.append(pos)
            if len(trajectory_so_far) > 1:
                rr.log("world/trajectory_history",
                       rr.LineStrips3D(np.array(trajectory_so_far), colors=[0, 128, 255]))

            rot = Rotation.from_quat(quat)
            rr.log("world/camera_axes", rr.Arrows3D(
                origins=[pos, pos, pos],
                vectors=[rot.apply([cam_axis_len, 0, 0]),
                         rot.apply([0, cam_axis_len, 0]),
                         rot.apply([0, 0, cam_axis_len])],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ))

        if video_cap is not None:
            # oakd_left.mp4 frames are 1:1 with SLAM trajectory frame_idx —
            # both came from the same StereoDepth output in encoding order.
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video_cap.read()
            if ret:
                frame_disp = cv2.resize(frame, (640, 400), interpolation=cv2.INTER_AREA)
                rr.log("camera_feed", rr.Image(frame_disp))

        if frame_i % (video_skip * 10) == 0:
            n_vframes_logged = frame_i // video_skip + 1
            print(f"  Frame {frame_i}/{n_total}  video_idx={frame_idx if video_cap else '-'}  logged={n_vframes_logged}", end='\r')

    print(f"\nVisualization complete.")
    print(f"  Green line: full trajectory (static)")
    print(f"  Blue line:  trajectory up to current time")
    print(f"  RGB arrows: camera X/Y/Z axes")

    if video_cap is not None:
        video_cap.release()

    print("\nPress Ctrl+C to exit...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
