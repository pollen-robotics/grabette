# grabette-postprocess

SLAM/VIO orchestration and LeRobot dataset generation for the GRABETTE project.

Takes raw episode recordings (OAK-D stereo + depth + IMU, or RPi video + Quest
tracking) from [Grabette](https://github.com/SteveNguyen/grabette), produces a
camera trajectory, and converts everything into a
[LeRobot v3](https://huggingface.co/docs/lerobot) dataset (Parquet + MP4) ready
for policy training.

Trajectories can come from two sources:
- **OAK-D offline VSLAM** — RTAB-Map RGBD-inertial odometry, run in Docker
- **Meta Quest** — external tracking via Quest controller, transformed to camera frame

> **Note:** The legacy RPi-fisheye + ORB-SLAM3 ("arducam") path was removed —
> Grabette V2 moved to the OAK-D + RTAB-Map stack (see
> `packages/grabette/docs/rgbd_branch_status.md` and
> `packages/casquette/docs/bimanual_fusion.md` for the rationale).

## Data flow

```
Episode directory (from Grabette)
├── oakd_left.mp4 + oakd_depth/        OAK-D stereo + depth
├── oakd_*_timestamps.json             frame timestamps
├── oakd_imu.json                      accel/gyro/rotation
├── oakd_calib_offline.json            intrinsics + imu_to_cam
├── r_hand_traj.json                   (optional) Meta Quest controller trajectory
└── metadata.json

    │  OAK path:   convert_episode_to_oak.py → run_oak_slam.py
    │  Quest path: transform_quest_trajectory.py
    ▼

└── camera_trajectory.csv              trajectory (absolute poses, gravity-aligned Z-up)

    │  generate_dataset.py
    ▼

LeRobot v3 dataset/
├── meta/info.json, stats.json, tasks.parquet, episodes/
├── data/chunk-NNN/file-NNN.parquet
└── videos/observation.images.cam0/chunk-NNN/file-NNN.mp4
```

## Setup

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The OAK SLAM step requires Docker with a locally built image:

```bash
docker build -t pollenrobotics/oak-vslam docker/oak_vslam/
```

## Quick start

### A. OAK-D SLAM workflow

```bash
# 1. Convert the compact recording into the per-file oak/ layout the C++ expects
uv run python scripts/rgbd_slam/convert_episode_to_oak.py -i ~/data/dataset/episode

# 2. Run offline RTAB-Map VSLAM → camera_trajectory.csv
uv run python scripts/rgbd_slam/run_oak_slam.py -i ~/data/dataset/episode

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

# Validate + generate dataset (same as the OAK workflow)
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
uv run python scripts/rgbd_slam/visualize_rgbd_trajectory.py ~/data/dataset/some_episode

# Push to HuggingFace Hub
uv run python scripts/push_to_hub.py \
  --repo_id user/dataset-name \
  --root ~/lerobot_datasets
```

## Usage details

### 1. Trajectory extraction

#### OAK-D SLAM (RTAB-Map)

The recording (`grabette/hardware/oakd.py`) stores compact mp4 + JSON sidecars.
`convert_episode_to_oak.py` expands these into the `oak/` layout
(`frames/`, `depth/`, `timestamps.csv`, `imu_acc.csv`, `imu_gyro.csv`,
`imu_rotation.csv`, `calib_offline.json`) consumed by the C++ `offline_vslam`
binary.

`run_oak_slam.py` then runs RTAB-Map odometry in Docker, integrates the
frame-to-frame deltas into absolute poses, gravity-aligns the trajectory
(world Z-up, estimated robustly from the accel stream), and writes
`camera_trajectory.csv`.

#### Meta Quest

Apply the saved Quest→camera calibration to each episode:

```bash
uv run python scripts/transform_quest_trajectory.py \
  --quest episode/r_hand_traj.json \
  --calibration config/quest_to_camera_calibration.json \
  -o episode/camera_trajectory.csv
```

The output is in `camera_trajectory.csv` format, directly compatible with all
downstream tools.

### 2. Validate data and trajectories

```bash
# Dataset health: IMU sample counts, video metadata, flag obvious problems
uv run python scripts/check_dataset.py ~/data/dataset

# Camera-IMU synchronization (optical flow vs gyro). <20ms good, >50ms bad
uv run python scripts/check_sync.py ~/data/dataset/episode --plot sync.png

# Trajectory quality: drift, relocalization jumps, zigzagging, unrealistic motion
uv run python scripts/check_trajectory.py ~/data/dataset -v

# SLAM vs reference (e.g. Quest) Absolute Trajectory Error
uv run python scripts/compare_trajectories.py \
  --slam camera_trajectory.csv \
  --reference r_hand_traj.json \
  --plot comparison.png
```

### 3. Generate LeRobot dataset

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
| `observation.images.cam0` | video | (3, 720, 960) | episode video (resized) |
| `observation.images.cam1` | video | (3, H, W) | Quest POV camera (`--quest-camera`, optional) |
| `action` | float32 | (8,) | `[x, y, z, ax, ay, az, proximal, distal]`, next-step target |

Poses are gravity-aligned (Z-up). The pose component is position +
axis-angle rotation; the gripper component is the two joint angles.

### 4. Push dataset to Hugging Face Hub

```bash
huggingface-cli login      # one-time

uv run python scripts/push_to_hub.py \
  --repo_id pollenrobotics/grabette-demo \
  --root ~/lerobot_datasets
  # add --private for a private repo
```

### 5. Visualize trajectory

Interactive 3D visualization with [Rerun](https://rerun.io/): trajectory,
camera frustum, video overlay, and IMU time series.

```bash
uv run python scripts/visualize_trajectory.py ~/data/dataset/episode
uv run python scripts/rgbd_slam/visualize_rgbd_trajectory.py ~/data/dataset/episode
```

## Project structure

```
grabette-postprocess/
├── pyproject.toml
├── config/
│   └── quest_to_camera_calibration.json   # Quest→camera rigid transform
├── grabette_postprocess/
│   ├── trajectory.py    # CSV parsing, quaternion→axis-angle, ANGL interpolation
│   ├── oak_slam.py      # OAK-D RTAB-Map orchestration (delta integration + gravity align)
│   └── dataset.py       # LeRobot v3 dataset builder
├── docker/
│   └── oak_vslam/       # RTAB-Map offline_vslam C++ + Dockerfile
└── scripts/
    ├── rgbd_slam/
    │   ├── convert_episode_to_oak.py   # recording → oak/ layout
    │   ├── run_oak_slam.py             # offline OAK-D VSLAM
    │   └── visualize_rgbd_trajectory.py
    ├── transform_quest_trajectory.py   # Quest trajectory → camera frame
    ├── batch_transform_quest.py
    ├── check_dataset.py                # dataset health check
    ├── check_sync.py                   # camera-IMU synchronization check
    ├── check_trajectory.py             # trajectory quality validation
    ├── compare_trajectories.py         # SLAM vs reference ATE comparison
    ├── generate_dataset.py             # trajectories → LeRobot v3
    ├── push_to_hub.py                  # upload dataset to Hugging Face Hub
    └── visualize_trajectory.py         # Rerun 3D visualization
```

## Hardware

- **Camera/depth**: OAK-D SR (stereo + on-board depth)
- **IMU**: BNO086 (accel + gyro + fused rotation)
- **Angle sensors**: two joint encoders (proximal + distal)
- **External tracking** (optional): Meta Quest controller, ~30Hz
```