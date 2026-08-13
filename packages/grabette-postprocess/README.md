# grabette-postprocess

SLAM/VIO orchestration and LeRobot dataset generation for the GRABETTE project.

Takes raw episode recordings (OAK-D stereo + depth + IMU) from
[Grabette](../grabette), runs offline visual-inertial
SLAM to produce a camera trajectory, and converts everything into a
[LeRobot v3](https://huggingface.co/docs/lerobot) dataset (Parquet + MP4) ready
for policy training.

Trajectories are produced by **OAK-D offline VSLAM** — RTAB-Map RGBD-inertial
odometry, run in Docker.


## Data flow

```
Episode directory (from Grabette)
├── oakd_left.mp4 + oakd_depth.mkv     OAK-D stereo + depth
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

Requires [uv](https://docs.astral.sh/uv/). The non-LeRobot conversion and SLAM
tools support Python >= 3.11; dataset generation, publishing, and visualization
require Python >= 3.12 because that is LeRobot 0.6's minimum.

> Part of the uv **workspace**: a bare `uv sync` here would build the *entire
> monorepo* environment. Always pass `--package` (root README → Development).

```bash
uv sync --package grabette-postprocess
```

The OAK SLAM step requires Docker with a locally built image:

```bash
docker build -t pollenrobotics/oak-vslam docker/oak_vslam/
```

## Quick start

The pipeline expects `-i` to point at a **dataset directory** containing one or
more episode subdirectories (`<dataset>/<episode>/…`), except for the
per-episode scripts (`convert_episode_to_oak.py`, `run_oak_slam.py`,
`visualize_rgbd_trajectory.py`) which take a **single episode directory**.

```bash
# 1. Sanity-check the recording (no Docker, fast) — sample counts, file inventory
uv run python scripts/checks/check_dataset.py -i ~/data/dataset

# 2. Convert the compact recording into the per-file oak/ layout the C++ expects
uv run python scripts/pipeline/convert_episode_to_oak.py -i ~/data/dataset/episode

# 3. Run offline RTAB-Map VSLAM → camera_trajectory.csv
uv run python scripts/pipeline/run_oak_slam.py -i ~/data/dataset/episode

# 4. Validate trajectories (drift, relocalization jumps, motion realism)
uv run python scripts/checks/check_trajectory.py -i ~/data/dataset -v

# 5. Generate LeRobot v3 dataset (angles are np.interp'd onto trajectory timestamps here)
uv run python scripts/pipeline/generate_dataset.py \
  -i ~/data/dataset \
  --repo_id user/dataset-name \
  --task "task description" \
  --root ~/lerobot_datasets

# 6. Visualize the SLAM trajectory in Rerun (3D poses + camera + IMU)
uv run python scripts/visualize/visualize_rgbd_trajectory.py \
  -i ~/data/dataset/episode --gravity-align --video-skip 2

# 7. Visualize the generated LeRobot dataset (action + video, episode-by-episode)
uv run lerobot-dataset-viz \
  --repo-id user/dataset-name --root ~/lerobot_datasets --episode-index 0

# 8. Push to HuggingFace Hub
uv run python scripts/pipeline/push_to_hub.py \
  --repo_id user/dataset-name \
  --root ~/lerobot_datasets
```

Notes:
- `--repo_id` accepts any `owner/name` string; for local-only runs the owner
  half doesn't need to correspond to a real HF account until you actually push
  (step 8).
- Rerun's `--gravity-align` re-orients the world so Z points up (uses the
  oak/ imu_acc stream), and `--video-skip 2` renders every other frame for
  faster loading. Both are optional.

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
  -i ~/data/dataset \
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

To train a policy, the [DiffusionPolicy](../../integrations/DiffusionPolicy)
integration's `convert_dataset.py` turns this 8D **absolute** action into the
11D **camera-local delta** action (6D rotation) + 2D gripper state the policy
actually trains on.

### 4. Push dataset to Hugging Face Hub

```bash
huggingface-cli login      # one-time

uv run python scripts/pipeline/push_to_hub.py \
  --repo_id pollenrobotics/grabette-demo \
  --root ~/lerobot_datasets
  # add --private for a private repo
```

### 5. Visualize

Two independent viewers, depending on what you want to inspect.

**Trajectory (Rerun)** — interactive 3D of the SLAM output, with the OAK-left
camera feed animated along the trajectory and IMU time series alongside.
Requires steps 2 + 3 (needs `camera_trajectory.csv` and, with
`--gravity-align`, the `oak/` subdir produced by `convert_episode_to_oak.py`):

```bash
uv run python scripts/visualize/visualize_rgbd_trajectory.py \
  -i ~/data/dataset/episode --gravity-align --video-skip 2
```

`--gravity-align` re-orients so Z is world-up; `--video-skip N` renders every
Nth frame (higher = faster load). Both optional.

**Generated LeRobot dataset (`lerobot-dataset-viz`)** — walk the produced
dataset episode-by-episode with the LeRobot viewer. Shows the camera stream
alongside the `action` vector; useful for confirming the end-to-end pipeline
(SLAM poses + gripper angles) matches expectations:

```bash
uv run lerobot-dataset-viz \
  --repo-id user/dataset-name \
  --root ~/lerobot_datasets \
  --episode-index 0
```

`--repo-id` is the same string you passed to `generate_dataset.py`.
`--episode-index` picks the episode inside the dataset. See
`lerobot-dataset-viz --help` for `--mode local/distant` and other flags.

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
- **IMU**: OAK-D SR onboard IMU — BNO086 (accel + gyro + fused rotation vector)
- **Angle sensors**: two joint encoders (proximal + distal)
