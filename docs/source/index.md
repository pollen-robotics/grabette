# Grabette 🤏

**Grabette is an open-source toolkit for collecting robotic manipulation demonstrations and turning them into training-ready datasets.**

A Grabette rig records synchronized **camera + depth** streams (plus IMU with the OAK-D SR) from a hand-held or gripper-mounted device, recovers the camera trajectory with SLAM, and exports a [LeRobot](https://huggingface.co/docs/lerobot) dataset ready for policy learning. You demonstrate the task with your own hand; no robot is involved in the recording, and the resulting dataset is **robot-agnostic**.

<video controls src="https://github.com/user-attachments/assets/6db9dd7b-1762-4004-8a76-ce76323499ba"></video>

## How it works

Data collection is three steps:

| 1. Record | 2. Process | 3. Train |
| :--- | :--- | :--- |
| Grab an object while the handheld device captures camera, depth and finger-joint angles. Start and stop with the [physical button](./data_collection.md#start-and-stop-a-recording); organize tasks and sessions in the [Fleet Space](./data_collection.md). | Offline RGB-D SLAM recovers the camera trajectory (visual-inertial when an IMU is present), then everything is assembled into a LeRobot v3 dataset in the [SLAM Space](./data_collection.md#build-the-lerobot-dataset). | Feed the dataset to your policy of choice. The repository ships [Diffusion Policy and π0.5](./training.md) integrations as worked examples. |

## The devices

| Device | What it is | Runs on |
| :--- | :--- | :--- |
| **Grabette** | The hand-held data-collection device: RPi camera, a depth camera (Orbbec Gemini 305 by default, or Luxonis OAK-D SR with IMU), two finger-joint encoders, one button. This is what you record with. | Raspberry Pi 4 |
| **Gripette** | The robot-mounted motorized gripper — the same fingers, driven by two servos, so a robot can reproduce what you demonstrated. | Raspberry Pi Zero 2W |
| **Casquette** *(WIP)* | A head-mounted point-of-view camera, for recording the scene from the operator's viewpoint. | Raspberry Pi Zero 2W |

## Where to go next

- **[Getting started](./get_started.md)** — power, WiFi and the dashboard.
- **[Data collection](./data_collection.md)** — the Fleet Space, and how to record a good dataset.
- **[Build your own](./build_your_own.md)** — parts, assembly and software install.
- **[Training](./training.md)** — Diffusion Policy and π0.5.
- **[Gripette](./gripette.md)** — the robot-mounted gripper.
- **[FAQ](./faq.md)** — common questions and troubleshooting.

Grabette is developed by [Pollen Robotics](https://pollen-robotics.com/) and released under the Apache-2.0 licence. The code lives at [pollen-robotics/grabette](https://github.com/pollen-robotics/grabette).
