# Grabette 🤏

**Grabette is an open-source toolkit for collecting robotic manipulation demonstrations and turning them into training-ready datasets.**

A Grabette rig records synchronized **camera + depth + IMU** streams from a hand-held or gripper-mounted device, recovers the camera trajectory with SLAM, and exports a [LeRobot](https://huggingface.co/docs/lerobot) dataset ready for policy learning. You demonstrate the task with your own hand; no robot is involved in the recording, and the resulting dataset is **robot-agnostic**.

<video controls src="https://github.com/user-attachments/assets/6db9dd7b-1762-4004-8a76-ce76323499ba"></video>

## How it works

Data collection is three steps, and each one has a page in this documentation:

| 1. Record | 2. Process | 3. Train |
| :--- | :--- | :--- |
| Grab an object while the handheld device captures camera, depth, IMU and finger-joint angles. Start and stop with the physical button or from the [dashboard](./dashboard.md). | Offline visual-inertial SLAM recovers the camera trajectory, then everything is assembled into a LeRobot v3 dataset — [locally](./get_started.md#turn-recordings-into-a-lerobot-dataset) or in the [SLAM Space](./spaces.md#grabette-slam--lerobot). | Feed the dataset to your policy of choice. The repository ships Diffusion Policy and π0.5 integrations as worked examples. |

## Build the hardware

Grabette is built from off-the-shelf parts, 3D-printed components and a Raspberry Pi.

- 📋 **[Bill of Materials](https://docs.google.com/spreadsheets/d/e/2PACX-1vQ3LyyWI-CiplVPtgrWkmLRYjdDqYhbVJXYt8PNa71FDzbTSMVj1YGV0Zpo5PJeBGJURaz8nZt1_v-8/pubhtml)** — the complete parts list, shared by Grabette and Gripette.
- 🧩 **[CAD on Onshape](https://cad.onshape.com/documents/0c6175c392788391992ff2ec/w/9f773e5f0eeae1577ae36a05/e/13a89fef2591d863bb0bf186)** — the full assembly.
- 🔩 **Assembly guides** — [Grabette](https://github.com/pollen-robotics/grabette/blob/develop/packages/grabette/assembly/Grabette_Assembly.pdf) · [Gripette](https://github.com/pollen-robotics/grabette/blob/develop/packages/gripette/assembly/Gripette_Assembly.pdf), with matching 3D-print guides in the same folders.

No hardware yet? The device software runs in **mock mode** on any laptop, so you can explore the dashboard before you build anything — see [Getting started](./get_started.md).

## The devices

| Device | What it is | Runs on |
| :--- | :--- | :--- |
| **Grabette** | The hand-held data-collection device: RPi camera, OAK-D SR depth camera + IMU, two finger-joint encoders, one button. This is what you record with. | Raspberry Pi 4 |
| **Gripette** | The robot-mounted motorized gripper — the same fingers, driven by two servos, so a robot can reproduce what you demonstrated. | Raspberry Pi Zero 2W |
| **Casquette** *(WIP)* | A head-mounted point-of-view camera, for recording the scene from the operator's viewpoint. | Raspberry Pi Zero 2W |

## Where to go next

- **[Getting started](./get_started.md)** — from an empty SD card to your first LeRobot dataset.
- **[The dashboard](./dashboard.md)** — the web interface you record and manage episodes from.
- **[Hugging Face Spaces](./spaces.md)** — run SLAM in the cloud and drive a fleet of devices.
- **[FAQ](./faq.md)** — common questions and troubleshooting.

Grabette is developed by [Pollen Robotics](https://pollen-robotics.com/) and released under the Apache-2.0 licence. The code lives at [pollen-robotics/grabette](https://github.com/pollen-robotics/grabette).
