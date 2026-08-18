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
        ├── dcam_imu.json               # depth-cam IMU: accel + gyro + rotation (200Hz, OAK-D only)
        ├── angle_data.json             # AS5600L joint angles (~85–100Hz)
        ├── rpi_camera_intrinsics.json  # Fisheye KB8 calibration for the primary cam
        ├── frames.json                 # URDF-derived frame transforms, incl. T_camera_in_oak_l
        ├── dcam_left.mp4               # depth-cam left, rectified mono H.264 (default 640×400)
        ├── dcam_right.mp4              # depth-cam right, rectified (OAK-D only; never consumed)
        ├── dcam_depth.mkv              # depth, lossless FFV1 16-bit uint16 mm (PNG dir dcam_depth/ kept only if packing fails)
        ├── dcam_mask.png               # Body mask applied to the depth stream
        ├── dcam_*_timestamps.json      # Per-stream timestamps (left / right / depth)
        ├── dcam_calib.json             # camera calibration dump (OAK-D EEPROM / Orbbec params)
        ├── dcam_calib_offline.json     # Flat fx/fy/cx/cy/baseline (+imu_to_cam if the camera has an IMU)
        ├── dcam_clock_pairs.json       # device ↔ SyncManager clock alignment
        └── metadata.json               # Duration, counts, hand, angle_convention, device_id, urdf, depth_camera
```

The `dcam_*` files are present when the depth camera was enabled for the capture
(it's toggled on demand, default off to save battery); SLAM needs them, so a
recording intended for the pipeline is made with the camera on.

**`dcam_` means "depth camera"**, not any particular model — the same layout is
produced whether the rig carries an OAK-D SR or an Orbbec Gemini 305
(`GRABETTE_DEPTH_CAMERA`). Two things differ by camera rather than by format:
the Gemini has no IMU, so it writes no `dcam_imu.json` and no `imu_to_cam` in
the calibration, and it writes no `dcam_right.mp4` (the OAK-D does, but nothing
downstream consumes it).

> **Legacy names.** Episodes recorded before this rename use `oakd_*` and
> `oak_mask.png`, and are *not* being rewritten — plenty are already on the Hub.
> Every reader resolves through `episode_files.resolve()`, which prefers the
> canonical name and falls back to the legacy one, so both layouts process
> identically. Never hardcode either name in new code.

## Calibration & geometry

Added by the rpi backend at capture stop:

- **Camera intrinsics** — `rpi_camera_intrinsics.json`, copied from `config/rpi_camera_intrinsics.json` (KannalaBrandt8 fisheye model, ~0.32px reproj). Ships as a single canonical file for all devices; per-device calibration is a separate open task.
- **Camera ↔ depth-camera geometry** — `frames.json`, computed from `urdf/grabette_{hand}/robot.urdf` at capture stop. Contains each frame's 4×4 transform in the `grip_r` link frame (`camera`, `oak_l`, `oak_r`, `gripper_center`, `thumb_tip`) plus the pre-composed `T_camera_in_oak_l` (so SLAM poses produced in the `oak_l` frame can be re-expressed in the primary camera frame without URDF parsing on the consumer side).
- **URDF traceability** — `metadata.json.urdf` records which URDF was used (`grabette_left` / `grabette_right`).
- **Which camera recorded it** — `metadata.json.depth_camera`, e.g.
  `{"model": "gemini305", "name": "Orbbec Gemini 305", "serial": "CV275610003C",
  "firmware": "1.0.70", "link": "USB3.2", "imu": null}`. `model` is the
  `GRABETTE_DEPTH_CAMERA` value; the rest is read from the device and is
  best-effort. This exists because the `dcam_*` filenames are deliberately
  vendor-neutral — without it a mixed dataset gives no hint which trajectories
  came from which hardware, and the two cameras differ in IMU availability and
  frame-drop rate. Absent on episodes recorded before it was added; readers get
  `{}` from `episode_files.camera_info()` rather than a guess.
- **Angle sensor offsets** — captured by `scripts/calibrate_angles.py`, stored in `~/.grabette/angle_calibration.json`.

## IMU format

`dcam_imu.json` — the OAK-D SR onboard IMU stream, written as `{"samples": [...]}` with interleaved accelerometer, gyroscope, and rotation-vector packets (accel in m/s², gyro in rad/s), timestamped on the shared capture clock. `convert_episode_to_oak.py` (in [grabette-postprocess](../../grabette-postprocess)) expands it into `imu_acc.csv` / `imu_gyro.csv` / `imu_rotation.csv` for SLAM.

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
