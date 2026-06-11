---
title: Grabette SLAM Pipeline
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
hf_oauth: true
hf_oauth_scopes:
  - read-repos
  - write-repos
  - manage-repos
---

# Grabette SLAM → LeRobot

A HuggingFace Space that turns a raw OAK-D recording dataset into a
[LeRobot](https://github.com/huggingface/lerobot) dataset and pushes it to the Hub.

1. **Sign in** with your Hugging Face account (OAuth).
2. Give a **source** dataset `repo_id` (raw OAK-D recording on the Hub), a
   **target** `repo_id` to create, and a task description.
3. For each episode the Space runs, in-process:
   - `convert_episode` — expand the recording into the `oak/` layout
   - `run_oak_slam` — RGBD-inertial odometry via the bundled `offline_vslam` binary
   - `build_dataset` — assemble a LeRobot v3 dataset
   - `push_to_hub` — upload under your account
4. A link + embedded view of the
   [LeRobot visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)
   is shown for the result (the dataset must be public to be visualized).

## Why a Docker Space

The SLAM step is a compiled C++ binary (`offline_vslam`, built on RTAB-Map).
Locally it runs in the `pollenrobotics/oak-vslam` Docker image, but Spaces cannot
run Docker-in-Docker. So this Space **is** that image: it builds the binary at
image-build time and `run_oak_slam(..., binary=...)` calls it directly.

## Self-contained build

The `grabette-postprocess` package (with `offline_vslam.cpp`) is vendored into
the build context as `./grabette-postprocess` — no git clone, no private-repo
secret. `deploy.sh` assembles that layout from the working tree and uploads it.

```bash
HF="uv run --project ../../packages/grabette-postprocess hf" \
  ./deploy.sh pollen-robotics/grabette-slam
```

> Note: RTAB-Map is compiled with `--parallel 2` to stay within the HF build
> runner's memory (full parallelism OOMs it).
