#!/usr/bin/env python3
"""
Visualize SLAM trajectory in 3D using Rerun.

The trajectory from grabette_slam is already gravity-aligned (Z-up, gravity = -Z)
after IMU initialization.

Usage:
    python scripts/visualize_trajectory.py <episode_dir>
    python scripts/visualize_trajectory.py test_data/grabette9 --video-skip 5
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


# Default intrinsics at 960x720 (fallback if video resolution unknown)
_DEFAULT_FX = 389.16
_DEFAULT_FY = 388.22
_DEFAULT_CX = 471.00
_DEFAULT_CY = 366.17
_DEFAULT_W = 960
_DEFAULT_H = 720

# Intrinsics at native 1296x972 (from rpi_bmi088_slam_settings.yaml)
_NATIVE_FX = 525.37
_NATIVE_FY = 524.10
_NATIVE_CX = 635.85
_NATIVE_CY = 494.33
_NATIVE_W = 1296
_NATIVE_H = 972


def _load_imu_streams(imu_json_path: Path) -> dict | None:
    """Load raw IMU streams (ACCL, GYRO, ANGL) from GoPro-format JSON.

    Returns dict with 'accel', 'gyro', 'angle' lists of {timestamp, value},
    or None if file doesn't exist.
    """
    if not imu_json_path.is_file():
        return None

    with open(imu_json_path) as f:
        raw = json.load(f)

    streams = raw.get('1', {}).get('streams', {})
    result = {'accel': [], 'gyro': [], 'angle': []}

    for stream_key, result_key in [('ACCL', 'accel'), ('GYRO', 'gyro'), ('ANGL', 'angle')]:
        if stream_key in streams and 'samples' in streams[stream_key]:
            for s in streams[stream_key]['samples']:
                result[result_key].append({
                    'timestamp': s['cts'] / 1000.0,
                    'value': s['value'],
                })

    if not result['accel'] and not result['gyro']:
        return None
    return result


def _log_imu_data(imu_data: dict):
    """Log IMU time series to Rerun."""
    # Configure series styles (static)
    rr.log("imu/accelerometer/x", rr.SeriesLines(colors=[255, 0, 0], names="accel_x"), static=True)
    rr.log("imu/accelerometer/y", rr.SeriesLines(colors=[0, 255, 0], names="accel_y"), static=True)
    rr.log("imu/accelerometer/z", rr.SeriesLines(colors=[0, 0, 255], names="accel_z"), static=True)
    rr.log("imu/gyroscope/x", rr.SeriesLines(colors=[255, 128, 0], names="gyro_x"), static=True)
    rr.log("imu/gyroscope/y", rr.SeriesLines(colors=[128, 255, 0], names="gyro_y"), static=True)
    rr.log("imu/gyroscope/z", rr.SeriesLines(colors=[0, 128, 255], names="gyro_z"), static=True)

    for sample in imu_data['accel']:
        rr.set_time("time", timestamp=sample['timestamp'])
        v = sample['value']
        rr.log("imu/accelerometer/x", rr.Scalars(float(v[0])))
        rr.log("imu/accelerometer/y", rr.Scalars(float(v[1])))
        rr.log("imu/accelerometer/z", rr.Scalars(float(v[2])))

    for sample in imu_data['gyro']:
        rr.set_time("time", timestamp=sample['timestamp'])
        v = sample['value']
        rr.log("imu/gyroscope/x", rr.Scalars(float(v[0])))
        rr.log("imu/gyroscope/y", rr.Scalars(float(v[1])))
        rr.log("imu/gyroscope/z", rr.Scalars(float(v[2])))

    if imu_data['angle']:
        rr.log("sensors/angle/sensor_1", rr.SeriesLines(colors=[255, 0, 128], names="angle_1"), static=True)
        rr.log("sensors/angle/sensor_2", rr.SeriesLines(colors=[0, 200, 200], names="angle_2"), static=True)
        for sample in imu_data['angle']:
            rr.set_time("time", timestamp=sample['timestamp'])
            v = sample['value']
            rr.log("sensors/angle/sensor_1", rr.Scalars(float(v[0])))
            rr.log("sensors/angle/sensor_2", rr.Scalars(float(v[1])))


@click.command()
@click.argument('episode_dir', type=click.Path(exists=True))
@click.option('--show-video/--no-video', default=True, help='Show video frames')
@click.option('--video-skip', default=5, help='Show every Nth video frame')
@click.option('--reference', '-r', type=click.Path(exists=True), default=None,
              help='Reference trajectory CSV to overlay (e.g. quest_in_slam_frame.csv)')
@click.option('--app-id', default='grabette_viz', help='Rerun application ID')
def main(episode_dir, show_video, video_skip, reference, app_id):
    """Visualize SLAM trajectory from a processed episode directory."""

    episode_dir = Path(episode_dir)

    # Find trajectory CSV
    traj_csv = episode_dir / "mapping_camera_trajectory.csv"
    if not traj_csv.exists():
        traj_csv = episode_dir / "camera_trajectory.csv"
    if not traj_csv.exists():
        print(f"Error: No trajectory CSV found in {episode_dir}")
        sys.exit(1)

    video_path = episode_dir / "raw_video.mp4"
    imu_json = episode_dir / "imu_data.json"

    # --- Load SLAM metadata (for frame_skip) ---
    frame_skip = 1
    slam_meta_path = episode_dir / "slam_metadata.json"
    if slam_meta_path.is_file():
        with open(slam_meta_path) as f:
            slam_meta = json.load(f)
        frame_skip = slam_meta.get("frame_skip", 1)
        if frame_skip > 1:
            print(f"SLAM metadata: frame_skip={frame_skip}")

    # --- Load trajectory ---
    print(f"Loading trajectory from {traj_csv.name}...")
    df_all = pd.read_csv(traj_csv)
    df_valid = df_all[~df_all['is_lost'].astype(bool)].copy()
    n_total = len(df_all)
    n_tracked = len(df_valid)

    # --- SLAM statistics ---
    print(f"\n=== SLAM Statistics ===")
    print(f"  Frames:   {n_tracked}/{n_total} tracked ({100*n_tracked/n_total:.1f}%)")
    print(f"  Lost:     {n_total - n_tracked} frames")
    if n_tracked > 0:
        first_tracked = int(df_valid.iloc[0]['frame_idx'])
        last_tracked = int(df_valid.iloc[-1]['frame_idx'])
        print(f"  Range:    frame {first_tracked} to {last_tracked}")
        duration = df_valid.iloc[-1]['timestamp'] - df_valid.iloc[0]['timestamp']
        print(f"  Duration: {duration:.2f}s")

    positions = df_valid[['x', 'y', 'z']].to_numpy().copy()
    quaternions = df_valid[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy().copy()

    if n_tracked == 0:
        print("Error: No valid poses found!")
        sys.exit(1)

    # --- Trajectory statistics ---
    print(f"\n=== Trajectory Statistics ===")
    print(f"  Position range (meters):")
    for ax, name in enumerate(['X', 'Y', 'Z']):
        lo, hi = positions[:, ax].min(), positions[:, ax].max()
        print(f"    {name}: [{lo:.4f}, {hi:.4f}]  range={hi-lo:.4f}")
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    print(f"  Total path length: {np.sum(distances):.3f} m")
    print(f"  Displacement:      {np.linalg.norm(positions[-1] - positions[0]):.3f} m")
    print()

    # --- Load IMU ---
    imu_data = _load_imu_streams(imu_json)
    if imu_data:
        print(f"IMU: {len(imu_data['accel'])} accel, {len(imu_data['gyro'])} gyro, "
              f"{len(imu_data['angle'])} angle samples")

    # --- Initialize Rerun ---
    rr.init(app_id, spawn=True)
    time.sleep(0.5)

    # Set world coordinate system: gravity-aligned, Z-up (matches ORB-SLAM3 after IMU init)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Log IMU data first (fills the timeline)
    if imu_data:
        print("Logging IMU data...")
        _log_imu_data(imu_data)

    # --- Blueprint ---
    try:
        top_views = [
            rrb.Spatial3DView(name="3D View", origin="/world"),
            rrb.Spatial2DView(name="Camera", origin="/camera_feed"),
        ]
        bottom_views = []
        if imu_data and imu_data['angle']:
            bottom_views.append(rrb.TimeSeriesView(name="Angle Sensors", origin="/sensors/angle"))
        if imu_data:
            bottom_views.append(rrb.TimeSeriesView(name="Accelerometer", origin="/imu/accelerometer"))
            bottom_views.append(rrb.TimeSeriesView(name="Gyroscope", origin="/imu/gyroscope"))

        if bottom_views:
            blueprint = rrb.Blueprint(
                rrb.Vertical(
                    rrb.Horizontal(*top_views),
                    rrb.Horizontal(*bottom_views),
                ),
            )
        else:
            blueprint = rrb.Blueprint(rrb.Horizontal(*top_views))
        rr.send_blueprint(blueprint)
    except Exception as e:
        print(f"Warning: Could not send blueprint: {e}")

    # --- Static elements ---
    # World axes at origin
    axis_len = 0.5
    rr.log("world/axes", rr.Arrows3D(
        origins=[[0, 0, 0]] * 3,
        vectors=[[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    ), static=True)

    # Full trajectory line (green)
    rr.log("world/trajectory_full", rr.LineStrips3D(positions, colors=[0, 255, 0]))
    rr.log("world/start", rr.Points3D(positions[0], colors=[0, 255, 0], radii=0.01))
    rr.log("world/end", rr.Points3D(positions[-1], colors=[255, 0, 0], radii=0.01))

    # Reference trajectory (orange) if provided
    if reference:
        print(f"Loading reference trajectory from {reference}...")
        df_ref = pd.read_csv(reference)
        df_ref_valid = df_ref[~df_ref['is_lost'].astype(bool)]
        ref_positions = df_ref_valid[['x', 'y', 'z']].to_numpy()
        if len(ref_positions) > 1:
            rr.log("world/reference_full", rr.LineStrips3D(ref_positions, colors=[255, 165, 0]))
            rr.log("world/ref_start", rr.Points3D(ref_positions[0], colors=[255, 165, 0], radii=0.01))
            rr.log("world/ref_end", rr.Points3D(ref_positions[-1], colors=[255, 100, 0], radii=0.01))
            print(f"  Reference: {len(ref_positions)} frames (orange)")
        else:
            print(f"  Warning: reference has < 2 tracked frames")

    # --- Open video ---
    video_cap = None
    if show_video and video_path.exists():
        video_cap = cv2.VideoCapture(str(video_path))
        if not video_cap.isOpened():
            print("Warning: Could not open video")
            video_cap = None
        else:
            fps = video_cap.get(cv2.CAP_PROP_FPS)
            total_vframes = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video: {total_vframes} frames at {fps:.2f} fps")

    # --- Build pose lookup for valid frames ---
    pose_map = {}
    for _, row in df_valid.iterrows():
        pose_map[int(row['frame_idx'])] = {
            'pos': np.array([row['x'], row['y'], row['z']]),
            'quat': np.array([row['q_x'], row['q_y'], row['q_z'], row['q_w']]),
        }

    # Intrinsics for pinhole projection — match SLAM processing resolution
    if video_cap is not None:
        vid_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if vid_w == _NATIVE_W and vid_h == _NATIVE_H:
            fx, fy, cx, cy = _NATIVE_FX, _NATIVE_FY, _NATIVE_CX, _NATIVE_CY
            disp_w, disp_h = _NATIVE_W, _NATIVE_H
        else:
            fx, fy, cx, cy = _DEFAULT_FX, _DEFAULT_FY, _DEFAULT_CX, _DEFAULT_CY
            disp_w, disp_h = _DEFAULT_W, _DEFAULT_H
    else:
        fx, fy, cx, cy = _DEFAULT_FX, _DEFAULT_FY, _DEFAULT_CX, _DEFAULT_CY
        disp_w, disp_h = _DEFAULT_W, _DEFAULT_H

    # --- Animate ---
    print(f"Visualizing {n_total} frames (skip={video_skip})...")
    trajectory_so_far = []
    cam_axis_len = 0.1

    for frame_i in range(n_total):
        if frame_i % video_skip != 0:
            continue

        row = df_all.iloc[frame_i]
        frame_idx = int(row['frame_idx'])
        rr.set_time("time", timestamp=row['timestamp'])

        if frame_idx in pose_map:
            p = pose_map[frame_idx]
            pos = p['pos']
            quat = p['quat']

            # Camera pose in world
            rr.log("world/camera", rr.Transform3D(
                translation=pos.tolist(),
                quaternion=quat.tolist(),
            ))
            rr.log("world/current_position", rr.Points3D(pos, colors=[255, 0, 0], radii=0.005))

            # Progressive trajectory
            trajectory_so_far.append(pos)
            if len(trajectory_so_far) > 1:
                rr.log("world/trajectory_history",
                       rr.LineStrips3D(np.array(trajectory_so_far), colors=[0, 128, 255]))

            # Camera frame axes
            rot = Rotation.from_quat(quat)
            cam_x = rot.apply([cam_axis_len, 0, 0])
            cam_y = rot.apply([0, cam_axis_len, 0])
            cam_z = rot.apply([0, 0, cam_axis_len])
            rr.log("world/camera_axes", rr.Arrows3D(
                origins=[pos, pos, pos],
                vectors=[cam_x, cam_y, cam_z],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ))

        # Video frame — frame_idx is the SLAM index, multiply by frame_skip
        # to get the actual video frame position
        if video_cap is not None:
            video_frame_idx = frame_idx * frame_skip
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
            ret, frame = video_cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                # 2D camera feed panel
                rr.log("camera_feed", rr.Image(frame_rgb))

                # Pinhole projection in 3D view
                rr.log("world/camera", rr.Pinhole(
                    resolution=[disp_w, disp_h],
                    focal_length=[fx, fy],
                    principal_point=[cx, cy],
                ))
                rr.log("world/camera", rr.Image(frame_rgb))

        if frame_i % (video_skip * 5) == 0:
            print(f"  Frame {frame_i}/{n_total}", end='\r')

    print(f"\nVisualization complete.")
    print(f"  Coordinate system: Z-up (gravity = -Z), from ORB-SLAM3 IMU init")
    print(f"  Green line: full trajectory")
    print(f"  Blue line: trajectory up to current time")
    print(f"  RGB arrows: camera X(right)/Y(down)/Z(forward) axes")

    if video_cap:
        video_cap.release()

    print("\nPress Ctrl+C to exit...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
