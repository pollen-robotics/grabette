# grabette-data

SLAM orchestration and LeRobot dataset generation for the GRABETTE project.

Takes raw episode recordings (video + IMU) from [Grabette](https://github.com/SteveNguyen/grabette), runs ORB-SLAM3 in Docker to produce camera trajectories, and converts everything into a [LeRobot v3](https://huggingface.co/docs/lerobot) dataset (Parquet + MP4) ready for policy training.

Trajectories can come from two sources:
- **ORB-SLAM3** — visual-inertial SLAM from camera + IMU
- **Meta Quest** — external tracking via Quest controller, transformed to camera frame

## Data flow

```
Episode directory (from Grabette)
├── raw_video.mp4          1296x972 @ ~50fps
├── imu_data.json          ACCL 200Hz, GYRO 200Hz, ANGL 100Hz
├── r_hand_traj.json       (optional) Meta Quest controller trajectory
└── metadata.json

    │  SLAM path:  create_map.py / batch_slam.py
    │  Quest path: transform_quest_trajectory.py
    ▼

├── camera_trajectory.csv       trajectory (from SLAM or Quest)
├── slam_metadata.json          SLAM run info (if SLAM was used)
├── imu_data_resampled.json     uniform 200Hz, ANGL stripped (SLAM only)
├── slam_mask.png               device body mask (SLAM only)
├── gravity.csv                 3x3 rotation matrix (SLAM only)
└── biases.csv                  IMU biases (SLAM only)

    │  generate_dataset.py
    ▼

LeRobot v3 dataset/
├── meta/info.json, stats.json, tasks.parquet, episodes/
├── data/chunk-NNN/file-NNN.parquet
└── videos/observation.images.cam0/chunk-NNN/file-NNN.mp4
```

## Setup

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo>
cd grabette-data
uv sync
```

The SLAM step requires Docker with the pre-built ORB-SLAM3 image:

```bash
docker pull pollenrobotics/orbslam3-headless
```

## Quick start

### A. SLAM-based workflow

```bash
# 0. (Optional) Check/update calibration on a new device
uv run python scripts/check_calibration.py \
  --video /path/to/charuco_video.mp4
# If poor (> 1.0px): recalibrate and update SLAM settings
uv run python scripts/calibrate_camera.py \
  --video /path/to/charuco_video.mp4

# 1. Create map from the mapping video
uv run python scripts/create_map.py \
  -i ~/data/dataset/mapping \
  --retries 3

# 2. Batch localize all episodes (with automatic mapping retry)
uv run python scripts/batch_slam.py \
  -i ~/data/dataset \
  -m ~/data/dataset/mapping/map/map_atlas.osa \
  -n 4

# 3. Validate trajectories
uv run python scripts/check_trajectory.py ~/data/dataset

# 4. Generate LeRobot dataset
uv run python scripts/generate_dataset.py \
  -i ~/data/dataset \
  --repo_id user/dataset-name \
  --task "task description" \
  --root ~/lerobot_datasets
```

### B. Quest-based workflow

When using a Meta Quest controller as external tracker (bypasses SLAM):

```bash
# One-time calibration: find Quest→camera transform from a recording
# where both SLAM and Quest are available
uv run python scripts/transform_quest_trajectory.py \
  --slam good_recording/camera_trajectory.csv \
  --quest good_recording/r_hand_traj.json \
  -o /dev/null \
  --save-calibration config/quest_to_camera_calibration.json

# For each episode: apply saved calibration
uv run python scripts/transform_quest_trajectory.py \
  --quest episode/r_hand_traj.json \
  --calibration config/quest_to_camera_calibration.json \
  -o episode/camera_trajectory.csv

# Validate + generate dataset (same as SLAM workflow)
uv run python scripts/check_trajectory.py ~/data/dataset
uv run python scripts/generate_dataset.py \
  -i ~/data/dataset \
  --repo_id user/dataset-name \
  --task "task description" \
  --root ~/lerobot_datasets \
  --quest-camera
```

### Common final steps

```bash
# Visualize a trajectory (with optional reference overlay)
uv run python scripts/visualize_trajectory.py ~/data/dataset/some_episode
uv run python scripts/visualize_trajectory.py ~/data/dataset/some_episode \
  --reference quest_in_slam_frame.csv

# Push to HuggingFace Hub
uv run python scripts/push_to_hub.py \
  --repo_id user/dataset-name \
  --root ~/lerobot_datasets
```

## Usage details

### 1. Calibration

#### Check existing calibration

Verify that existing camera intrinsics fit a new device by detecting ChArUco corners and computing fisheye reprojection error. Useful when assembling a new device with the same lens model.

```bash
uv run python scripts/check_calibration.py \
  --video /path/to/charuco_video.mp4
```

Thresholds: median < 1.0px = good, 1.0–2.0px = marginal, > 2.0px = recalibrate.

#### Recalibrate

If the check shows poor results (> 1.0px), recalibrate from a ChArUco video. Record a ~30s video slowly moving the camera over the standard ChArUco board (10x8, 19mm squares, DICT_ARUCO_ORIGINAL) from various angles and distances.

```bash
uv run python scripts/calibrate_camera.py \
  --video /path/to/charuco_video.mp4
```

This runs OpenCV fisheye calibration, saves `config/camera_intrinsics.json`, and automatically updates `config/rpi_bmi088_slam_settings.yaml` with the new intrinsics (scaled to the YAML's target resolution if needed). The IMU parameters and camera-IMU extrinsics (`T_b_c1`) are preserved — only camera intrinsics (fx, fy, cx, cy, k1-k4) are updated.

To calibrate without modifying the SLAM settings:

```bash
uv run python scripts/calibrate_camera.py \
  --video /path/to/charuco_video.mp4 --dry-run
```

**Note:** The camera-IMU extrinsic transform is fixed from the physical mounting (back-to-back, 180° rotation, 11.15mm offset) and does not need recalibration unless the hardware changes.

#### Quest-to-camera calibration

When using Meta Quest tracking, the Quest→camera rigid transform must be calibrated once per device setup (the Quest controller handle is physically attached to the device).

Record one session where both SLAM and Quest tracking work, then compute the transform:

```bash
uv run python scripts/transform_quest_trajectory.py \
  --slam good_recording/camera_trajectory.csv \
  --quest good_recording/r_hand_traj.json \
  -o /dev/null \
  --save-calibration config/quest_to_camera_calibration.json
```

This finds the rotation, translation, and scale via Umeyama alignment. The calibration file can then be applied to all future Quest recordings from the same device setup.

### 2. Trajectory extraction

#### SLAM (ORB-SLAM3)

**Create map** from a mapping video (two-pass SLAM with retries):

```bash
uv run python scripts/create_map.py \
  -i ~/data/dataset/mapping \
  --retries 3
```

**Batch localize** episodes against the map:

```bash
uv run python scripts/batch_slam.py \
  -i ~/data/dataset \
  -m ~/data/dataset/mapping/map/map_atlas.osa \
  -n 4
```

Batch SLAM runs in two phases:

1. **Localization** — each episode localizes against the shared map. If SLAM loses tracking and resets the map, the container is killed immediately.

2. **Mapping retry** — episodes that failed or tracked below `--min_tracking_pct` (default 50%) are retried in full mapping mode (independent map per episode).

**Mapping-only mode** — skip localization, run independent SLAM on each episode:

```bash
uv run python scripts/batch_slam.py \
  -i ~/data/dataset \
  --mapping-only -n 4
```

Options:
- `--min_tracking_pct 50.0` — retry threshold (default 50%)
- `--no-retry` — disable mapping retry (localization only)
- `--mapping-only` — skip localization, run independent mapping on each episode
- `--max_lost_frames 60` — terminate localization after N consecutive lost frames
- `--force` / `-f` — reprocess episodes that already have trajectories

#### Meta Quest

Apply the saved Quest→camera calibration to each episode:

```bash
uv run python scripts/transform_quest_trajectory.py \
  --quest episode/r_hand_traj.json \
  --calibration config/quest_to_camera_calibration.json \
  -o episode/camera_trajectory.csv
```

The output is in `camera_trajectory.csv` format, directly compatible with all downstream tools.

### 3. Validate data and trajectories

#### Check dataset health

Verify IMU sample counts, video metadata, and flag obvious data problems:

```bash
uv run python scripts/check_dataset.py ~/data/dataset
```

#### Check camera-IMU synchronization

Correlate optical flow with gyroscope data to detect timing offsets:

```bash
uv run python scripts/check_sync.py ~/data/dataset/mapping
uv run python scripts/check_sync.py ~/data/dataset/mapping --plot sync.png
```

Thresholds: < 20ms = good, 20–50ms = marginal, > 50ms = will break SLAM.

#### Validate trajectory quality

Detect IMU drift, relocalization jumps, zigzagging, and unrealistic motion:

```bash
uv run python scripts/check_trajectory.py ~/data/dataset
uv run python scripts/check_trajectory.py ~/data/dataset -v  # verbose
```

#### Compare SLAM vs reference trajectory

Compute Absolute Trajectory Error (ATE) between SLAM and an external reference:

```bash
uv run python scripts/compare_trajectories.py \
  --slam camera_trajectory.csv \
  --reference r_hand_traj.json \
  --plot comparison.png
```

### Frame rate and resolution

By default, SLAM processes at **native resolution (1296x972)** and **25fps** (`--frame_skip 2`), which gives better tracking than the previous 960x720 @ 50fps. The 50fps recording is preserved — decimation happens at processing time.

- **25fps (default)**: wider baseline between frames improves triangulation and map quality. Best for typical manipulation speeds.
- **50fps** (`--frame_skip 1`): use when recording fast motions where 25fps might cause blur between frames.

The previous 960x720 settings are available as a fallback: `-s config/rpi_bmi088_slam_settings_960x720.yaml`.

### Deterministic mode

The `--deterministic` flag forces tracking to wait for LocalMapping after each frame, making results fully reproducible.

Trade-offs:
- **Slower**: tracking blocks on LocalMapping after each frame
- **Reproducible**: identical results across runs — no need for retries
- **Simpler**: `create_map` with `--deterministic` forces `retries=0` and `parallel=1`

### 4. Generate LeRobot dataset

Converts trajectories + raw data into a LeRobot v3 dataset.

```bash
uv run python scripts/generate_dataset.py \
  --input_dir ~/data/dataset \
  --repo_id myuser/grabette-demo \
  --task "cup manipulation" \
  --root ~/lerobot_datasets
```

#### Dataset features

| Feature | dtype | shape | Source |
|---------|-------|-------|--------|
| `observation.images.cam0` | video | (3, 720, 960) | raw_video.mp4 (resized) |
| `observation.pose` | float32 | (6,) | trajectory: [x, y, z, ax, ay, az] |
| `observation.joints` | float32 | (2,) | ANGL stream: [proximal, distal] rad |
| `action` | float32 | (2,) | joints[t+1] (next-step angle) |

Poses are gravity-aligned (Z-up) directly from ORB-SLAM3's IMU initialization. The 6D pose is position + axis-angle rotation.

### 5. Push dataset to Hugging Face Hub

```bash
# Login (one-time)
huggingface-cli login

# Push
uv run python scripts/push_to_hub.py \
  --repo_id pollenrobotics/grabette-demo \
  --root ~/lerobot_datasets

# Or as private
uv run python scripts/push_to_hub.py \
  --repo_id pollenrobotics/grabette-demo \
  --root ~/lerobot_datasets \
  --private
```

### 6. Visualize trajectory

Interactive 3D visualization with [Rerun](https://rerun.io/): trajectory, camera frustum, video overlay, and IMU time series.

```bash
uv run python scripts/visualize_trajectory.py ~/data/dataset/episode
uv run python scripts/visualize_trajectory.py ~/data/dataset/episode --video-skip 1

# Overlay a reference trajectory (e.g. Quest in camera frame)
uv run python scripts/visualize_trajectory.py ~/data/dataset/episode \
  --reference quest_in_slam_frame.csv
```

## Project structure

```
grabette-data/
├── pyproject.toml
├── config/
│   ├── rpi_bmi088_slam_settings.yaml          # SLAM settings (native 1296x972)
│   ├── rpi_bmi088_slam_settings_960x720.yaml  # SLAM settings (legacy 960x720)
│   └── quest_to_camera_calibration.json        # Quest→camera rigid transform
├── grabette_data/
│   ├── imu.py           # IMU deduplication + resampling to uniform 200Hz
│   ├── mask.py          # Device body mask polygon (auto-scales to resolution)
│   ├── slam.py          # Docker SLAM orchestration (run_slam, create_map, batch_slam)
│   ├── trajectory.py    # CSV parsing, quaternion→axis-angle, ANGL interpolation
│   └── dataset.py       # LeRobot v3 dataset builder
├── scripts/
│   ├── calibrate_camera.py             # Fisheye calibration + update SLAM settings
│   ├── check_calibration.py            # Verify intrinsics on new device
│   ├── check_dataset.py                # Dataset health check (IMU, video, files)
│   ├── check_sync.py                   # Camera-IMU synchronization check
│   ├── check_trajectory.py             # Trajectory quality validation
│   ├── compare_trajectories.py         # SLAM vs reference ATE comparison
│   ├── transform_quest_trajectory.py   # Quest trajectory → camera frame
│   ├── create_map.py                   # Two-pass SLAM mapping
│   ├── batch_slam.py                   # Batch localization/mapping
│   ├── generate_dataset.py             # SLAM/Quest outputs → LeRobot v3
│   ├── push_to_hub.py                  # Upload dataset to Hugging Face Hub
│   └── visualize_trajectory.py         # Rerun 3D visualization
└── test_data/
```

## Hardware

- **Camera**: Raspberry Pi camera module, 1296x972 @ 50fps, fisheye lens (KannalaBrandt8)
- **IMU**: BMI088 (Bosch), 200Hz, raw gyro + accel
- **Mounting**: back-to-back, camera and IMU centers aligned along z-axis, 11.15mm apart
- **Angle sensors**: two joint encoders at 100Hz (proximal + distal)
- **External tracking** (optional): Meta Quest controller, ~30Hz
