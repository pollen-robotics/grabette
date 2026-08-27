# The Grabette Project

<img src='https://github.com/pollen-robotics/grabette/raw/develop/docs/images/grabette_logo_small.png' align='left' width='20%'/><br> 

**Grabette is an open-source toolkit for collecting robotic manipulation demonstrations and turning them into training-ready datasets.**

A Grabette rig records synchronized **camera + depth + IMU** streams from a hand-held or gripper-mounted device, recovers the camera trajectory with SLAM, and exports a [LeRobot](https://huggingface.co/docs/lerobot) dataset ready for policy learning. You demonstrate the task with your own hand; no robot is involved in the recording, and the resulting dataset is **robot-agnostic**.

<video controls src="https://github.com/user-attachments/assets/6db9dd7b-1762-4004-8a76-ce76323499ba"></video>

## How it works

Data collection is three steps, and each one has a page in this documentation:

| 1. Record | 2. Process | 3. Train |
| :--- | :--- | :--- |
| Grab an object while the handheld device captures camera, depth, IMU and finger-joint angles. Start and stop with the physical button or from the [dashboard](./grabette/usage.md). | Offline visual-inertial SLAM recovers the camera trajectory, then everything is assembled into a LeRobot v3 dataset — [locally or in the SLAM Space](./grabette/data_collection.md). | Feed the dataset to your policy of choice. The repository ships [Diffusion Policy and π0.5](./grabette/training.md) integrations as worked examples. |

## Build the hardware

Grabette is built from off-the-shelf parts, 3D-printed components and a Raspberry Pi.

- 📋 **[Bill of Materials](https://docs.google.com/spreadsheets/d/e/2PACX-1vQ3LyyWI-CiplVPtgrWkmLRYjdDqYhbVJXYt8PNa71FDzbTSMVj1YGV0Zpo5PJeBGJURaz8nZt1_v-8/pubhtml)** — the complete parts list, shared by Grabette and Gripette.
- 🧩 **[CAD on Onshape](https://cad.onshape.com/documents/0c6175c392788391992ff2ec/w/9f773e5f0eeae1577ae36a05/e/13a89fef2591d863bb0bf186)** — the full assembly.
- 🔩 **Assembly guides** — [Grabette](https://github.com/pollen-robotics/grabette/blob/develop/packages/grabette/assembly/Grabette_Assembly.pdf) · [Gripette](https://github.com/pollen-robotics/grabette/blob/develop/packages/gripette/assembly/Gripette_Assembly.pdf), with matching 3D-print guides in the same folders.

No hardware yet? The device software runs in **mock mode** on any laptop, so you can explore the dashboard before you build anything — see [Getting Started](./grabette/getting_started.md).

## The devices

| Device | What it is | Runs on |
| :--- | :--- | :--- |
| **[Grabette](./grabette/getting_started.md)** | The hand-held data-collection device: RPi camera, OAK-D SR depth camera + IMU, two finger-joint encoders, one button. This is what you record with. | Raspberry Pi 4 |
| **[Gripette](./gripette/getting_started.md)** | The robot-mounted motorized gripper — the same fingers, driven by two servos, so a robot can reproduce what you demonstrated. | Raspberry Pi Zero 2W |
| **[Casquette](./casquette/casquette.md)** *(WIP)* | A head-mounted point-of-view camera, for recording the scene from the operator's viewpoint. | Raspberry Pi Zero 2W |

## Where to go next

### Grabette — the hand-held recorder

- **[Getting Started](./grabette/getting_started.md)** — assembly, flashing the Pi, services, angle calibration, and the Bluetooth WiFi tool.
- **[Usage](./grabette/usage.md)** — powering the device on and off, charging it, reaching the dashboard, and running a recording.
- **[Data Collection](./grabette/data_collection.md)** — grabette-fleet, sessions, tasks, uploading a dataset, and the bimanual setup.
- **[Training](./grabette/training.md)** — Diffusion Policy and π0.5 on Grabette data.

### Gripette — the robot-mounted gripper

- **[Getting Started](./gripette/getting_started.md)** — assembly, flashing the Pi, services, angle calibration, and the Bluetooth WiFi tool.
- **[Set up on a robotic arm](./gripette/set_up.md)** — mounting and wiring the gripper to an arm.
- **[Usage](./gripette/usage.md)** — driving the gripper day to day.

### Casquette

- **[Casquette](./casquette/casquette.md)** *(WIP)* — the head-mounted point-of-view capture, still experimental.

Grabette is developed by [Pollen Robotics](https://pollen-robotics.com/) and released under the Apache-2.0 licence. The code lives at [pollen-robotics/grabette](https://github.com/pollen-robotics/grabette).
