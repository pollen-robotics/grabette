# grabette-postprocess

SLAM/VIO orchestration and LeRobot dataset generation for the GRABETTE project.

Takes raw episode recordings (OAK-D stereo + depth + IMU) from
[Grabette](https://github.com/pollen-robotics/grabette), runs offline visual-inertial
SLAM to produce a camera trajectory, and converts everything into a
[LeRobot v3](https://huggingface.co/docs/lerobot) dataset (Parquet + MP4) ready
for policy training.

Trajectories are produced by **OAK-D offline VSLAM** — RTAB-Map RGBD-inertial
odometry, run in Docker.


## Data flow

```
Episode directory (from Grabette)
├── oakd_left.mp4 + oakd_depth/        OAK-D stereo + depth
├── oakd_*_timestamps.json             frame timestamps
├── oakd_imu.json                      accel/gyro/rotation
├── oakd_calib_offline.json            intrinsics + imu_to_cam
└── metadata.json

    │  convert_episode_to_oak.py → run_oak_slam.py
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

```bash
# 1. Convert the compact recording into the per-file oak/ layout the C++ expects
uv run python scripts/pipeline/convert_episode_to_oak.py -i ~/data/dataset/episode

# 2. Run offline RTAB-Map VSLAM → camera_trajectory.csv
uv run python scripts/pipeline/run_oak_slam.py -i ~/data/dataset/episode

# 3. Validate trajectories
uv run python scripts/checks/check_trajectory.py -i ~/data/dataset

# 4. Generate LeRobot dataset
uv run python scripts/pipeline/generate_dataset.py \
  -i ~/data/dataset \
  --repo_id user/dataset-name \
  --task "task description" \
  --root ~/lerobot_datasets

# 5. Visualize a trajectory
uv run python scripts/visualize/visualize_rgbd_trajectory.py -i ~/data/dataset/episode

# 6. Push to HuggingFace Hub
uv run python scripts/pipeline/push_to_hub.py \
  --repo_id user/dataset-name \
  --root ~/lerobot_datasets
```

## Usage details

### 1. Trajectory extraction (OAK-D SLAM, RTAB-Map)

The recording (`grabette/hardware/oakd.py`) stores compact mp4 + JSON sidecars.
`convert_episode_to_oak.py` expands these into the `oak/` layout
(`frames/`, `depth/`, `timestamps.csv`, `imu_acc.csv`, `imu_gyro.csv`,
`imu_rotation.csv`, `calib_offline.json`) consumed by the C++ `offline_vslam`
binary.

`run_oak_slam.py` then runs RTAB-Map odometry in Docker, integrates the
frame-to-frame deltas into absolute poses, gravity-aligns the trajectory
(world Z-up, estimated robustly from the accel stream), and writes
`camera_trajectory.csv`.

### 2. Validate data and trajectories

```bash
# Dataset health: IMU sample counts, video metadata, flag obvious problems
uv run python scripts/checks/check_dataset.py -i ~/data/dataset

# Camera-IMU synchronization (optical flow vs gyro). <20ms good, >50ms bad
uv run python scripts/checks/check_sync.py -i ~/data/dataset/episode --plot sync.png

# Trajectory quality: drift, relocalization jumps, zigzagging, unrealistic motion
uv run python scripts/checks/check_trajectory.py -i ~/data/dataset -v
```

### 3. Generate LeRobot dataset

Converts trajectories + raw data into a LeRobot v3 dataset.

```bash
uv run python scripts/pipeline/generate_dataset.py \
  --input_dir ~/data/dataset \
  --repo_id myuser/grabette-demo \
  --task "cup manipulation" \
  --root ~/lerobot_datasets
```

#### Dataset features

| Feature | dtype | shape | Source |
|---------|-------|-------|--------|
| `observation.images.cam0` | video | (3, 720, 960) | episode video (resized) |
| `action` | float32 | (8,) | `[x, y, z, ax, ay, az, proximal, distal]`, next-step target |

Poses are gravity-aligned (Z-up). The pose component is position +
axis-angle rotation; the gripper component is the two joint angles.

### 4. Push dataset to Hugging Face Hub

```bash
huggingface-cli login      # one-time

uv run python scripts/pipeline/push_to_hub.py \
  --repo_id pollenrobotics/grabette-demo \
  --root ~/lerobot_datasets
  # add --private for a private repo
```

### 5. Visualize trajectory

Interactive 3D visualization with [Rerun](https://rerun.io/): the SLAM
trajectory, the OAK-left camera feed, and IMU time series.

```bash
uv run python scripts/visualize/visualize_rgbd_trajectory.py -i ~/data/dataset/episode
```

## Project structure

```
grabette-postprocess/
├── pyproject.toml
├── grabette_postprocess/
│   ├── trajectory.py       # CSV parsing, quaternion→axis-angle, ANGL interpolation
│   ├── convert.py          # recording → oak/ layout expansion
│   ├── oak_slam.py         # OAK-D RTAB-Map orchestration (delta integration + gravity align)
│   ├── dataset.py          # LeRobot v3 dataset builder
│   ├── episode_manager.py  # episode discovery / dropping
│   └── checks/             # validation logic (recording, sync, trajectory)
├── docker/
│   └── oak_vslam/       # RTAB-Map offline_vslam C++ + Dockerfile
└── scripts/                           # local CLIs (mirror the post-processing pipeline)
    ├── pipeline/                       # the pipeline stages, in order
    │   ├── convert_episode_to_oak.py   # recording → oak/ layout
    │   ├── run_oak_slam.py             # offline OAK-D VSLAM → camera_trajectory.csv
    │   ├── generate_dataset.py         # trajectories → LeRobot v3
    │   └── push_to_hub.py              # upload dataset to Hugging Face Hub
    ├── checks/                         # validation / QA
    │   ├── check_dataset.py            # dataset health check
    │   ├── check_sync.py               # camera-IMU synchronization check
    │   └── check_trajectory.py         # trajectory quality validation
    └── visualize/
        └── visualize_rgbd_trajectory.py  # Rerun 3D visualization
```

## Hardware

- **Camera/depth**: OAK-D SR (stereo + on-board depth)
- **IMU**: BNO086 (accel + gyro + fused rotation)
- **Angle sensors**: two joint encoders (proximal + distal)
```