# Meet Grabette

**Grabette is an open-source toolkit for collecting robotic manipulation demonstrations and turning them into training-ready datasets.**

A Grabette rig records synchronized **camera + depth + IMU + joints** streams from a hand-held device, recovers the camera trajectory with SLAM, and exports a [LeRobot](https://huggingface.co/docs/lerobot) dataset ready for policy learning. 

You demonstrate the task with your own hand; no robot is involved in the recording, and the resulting dataset is **robot-agnostic**.

## What it's made of

<img src="https://github.com/pollen-robotics/grabette/raw/develop/docs/images/grabette_label.png" width="80%"/><br> 

| Part | Role |
| :--- | :--- |
| Raspberry Pi 4 | Runs the daemon and the dashboard |
| RPi camera module | The recorded RGB stream |
| OAK-D SR | Depth + IMU — **mandatory** on Grabette, this is what SLAM runs on |
| Two joint encoders | Finger-joint angles |
| One button | Start / stop a recording without touching a screen |

## How it works

Data collection is three steps :

| 1. Record | 2. Process | 3. Train |
| :--- | :--- | :--- |
| Grab an object while the handheld device captures camera, depth, IMU and finger-joint angles. Start and stop with the physical button or from the [dashboard](./usage.md). | Offline visual-inertial SLAM recovers the camera trajectory, then everything is assembled into a LeRobot v3 dataset — [locally or in the SLAM Space](./data_collection.md). | Feed the dataset to your policy of choice. The repository ships [Diffusion Policy and π0.5](./training.md) integrations as worked examples. |
