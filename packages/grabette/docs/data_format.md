# Grabette — Data format

What a Grabette recording contains and how the streams are aligned. See the
[README](../README.md) for install and usage.

## Organization

Two-level hierarchy: **sessions** (named groups) containing **episodes** (individual captures).

```
~/grabette-data/
├── sessions.json                       # Session registry
└── episodes/
    └── 20260310_143052/                # One episode
        ├── raw_video.mp4               # Primary RPi cam, H.264 (1296x972 @ 46fps)
        ├── frame_timestamps.json       # Per-frame timestamps for raw_video
        ├── oakd_imu.json               # OAK-D IMU: accel + gyro + rotation vector (200Hz)
        ├── angle_data.json             # AS5600L joint angles (~85–100Hz)
        ├── rpi_camera_intrinsics.json  # Fisheye KB8 calibration for the primary cam
        ├── frames.json                 # URDF-derived frame transforms, incl. T_camera_in_oak_l
        ├── oakd_left.mp4               # OAK-D stereo left (H.264)
        ├── oakd_right.mp4              # OAK-D stereo right (H.264)
        ├── oakd_depth.mkv              # OAK-D depth stream
        ├── oakd_*_timestamps.json      # Per-stream timestamps
        ├── oakd_calib.json             # OAK-D factory EEPROM dump
        ├── oakd_calib_offline.json     # Flat fx/fy/cx/cy/baseline/imu_to_cam for SLAM
        ├── oakd_clock_pairs.json       # OAK-D ↔ SyncManager clock alignment
        └── metadata.json               # Duration, counts, hand, angle_convention, device_id, urdf
```

## Calibration & geometry

Added by the rpi backend at capture stop:

- **Camera intrinsics** — `rpi_camera_intrinsics.json`, copied from `config/rpi_camera_intrinsics.json` (KannalaBrandt8 fisheye model, ~0.32px reproj). Ships as a single canonical file for all devices; per-device calibration is a separate open task.
- **Camera ↔ OAK-D geometry** — `frames.json`, computed from `urdf/grabette_{hand}/robot.urdf` at capture stop. Contains each frame's 4×4 transform in the `grip_r` link frame (`camera`, `oak_l`, `oak_r`, `gripper_center`, `thumb_tip`) plus the pre-composed `T_camera_in_oak_l` (so SLAM poses produced in the `oak_l` frame can be re-expressed in the primary camera frame without URDF parsing on the consumer side).
- **URDF traceability** — `metadata.json.urdf` records which URDF was used (`grabette_left` / `grabette_right`).
- **Angle sensor offsets** — captured by `scripts/calibrate_angles.py`, stored in `~/.grabette/angle_calibration.json`.

## IMU format

`oakd_imu.json` — the OAK-D SR onboard IMU stream, written as `{"samples": [...]}` with interleaved accelerometer, gyroscope, and rotation-vector packets (accel in m/s², gyro in rad/s), timestamped on the shared capture clock. `convert_episode_to_oak.py` (in [grabette-postprocess](../../grabette-postprocess)) expands it into `imu_acc.csv` / `imu_gyro.csv` / `imu_rotation.csv` for SLAM.

> The legacy GoPro-style `imu_data.json` (`ACCL`/`GYRO` streams) is the older casquette/V1 format and is **not** produced by the OAK-D recording — the mock backend still emits it for development.

## Capture synchronization

All sensor streams share a common `SyncManager` clock based on `time.monotonic()`:

- **Camera**: SensorTimestamp from picamera2 (same SoC hardware clock — no drift)
- **IMU**: depthai timestamps from the OAK-D pipeline, mapped onto the SyncManager clock at sample arrival
- **Contention prevention**: `_capturing` flag blocks daemon I2C reads during recording
- **Stop order**: IMU/depth first, then camera (camera stop includes ffmpeg muxing)
- **IMU brackets video**: IMU starts before first frame, stops before last — required by the downstream SLAM/VIO pipeline

## Data pipeline

```
RPi (camera + OAK-D + AS5600L)
  → Grabette service (capture, manage sessions)
  → HuggingFace dataset repo (upload episodes)
  → Cloud SLAM/VIO processing
  → Training dataset + 6DoF trajectories
```

Downstream processing (SLAM → LeRobot dataset) lives in
[grabette-postprocess](../../grabette-postprocess).
