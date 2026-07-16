# Third-Party Notices

GRABETTE is © Pollen Robotics and released under the Apache License 2.0 (see
[`LICENSE`](LICENSE)). It incorporates, depends on, or derives from the
third-party components below, each under its own license. This file is provided
for attribution and license compliance.

---

### RTAB-Map — BSD-3-Clause (Mathieu Labbe, IntRoLab, Université de Sherbrooke)
<https://github.com/introlab/rtabmap>

Built from source (v0.23.1) inside the SLAM Docker image
(`packages/grabette-postprocess/docker/oak_vslam/`); the offline odometry binary
links RTAB-Map's core libraries.

### Luxonis DepthAI — MIT
<https://github.com/luxonis/depthai-core>

- `packages/grabette-postprocess/docker/oak_vslam/offline_vslam.cpp` adapts the
  processing logic of DepthAI's `RTABMapVIO` node (offline reimplementation; no
  DepthAI code is linked at build time).
- The `depthai` Python API is a runtime dependency of the on-device capture
  service (`packages/grabette`).

### OpenArm & openarm_can — Apache-2.0 (Enactic, Inc.)
<https://github.com/enactic/openarm> · <https://github.com/enactic/openarm_can>

- The robot description in `integrations/openarm/openarm_gripette_model` is
  derived from the OpenArm arm model, adapted to carry the Gripette end-effector.
- `openarm_can` is a dependency of `integrations/openarm/openarm_gripette`.
- `integrations/openarm/openarm_gripette/examples/calibrate_arm_no_gripper.py` is
  adapted from OpenArm's zero-position calibration procedure.

### LeRobot — Apache-2.0 (Hugging Face, Inc.)
<https://github.com/huggingface/lerobot>

- Used as a dependency for LeRobot-format dataset generation and policy training.
- `integrations/openarm/openarm_gripette/openarm_gripette/openarm7_follower.py`
  and `config_openarm7_follower.py` subclass LeRobot's `OpenArmFollower` and
  retain the original Hugging Face copyright headers.

### onshape-to-robot — MIT (Rhoban)
<https://github.com/Rhoban/onshape-to-robot>

Used to generate the OpenArm-Gripette URDF / MuJoCo model from Onshape CAD.

### 6-D rotation representation — Zhou et al., CVPR 2019
*"On the Continuity of Rotation Representations in Neural Networks."*

`integrations/DiffusionPolicy/rotation.py` is a Pollen Robotics implementation of
this representation (Apache-2.0), vendored to avoid depending on a fork.

### reachy_mini — Apache-2.0 (Pollen Robotics)

Portions of `packages/grabette` (`bluetooth/bluetooth_service.py`, `auth.py`) are
adapted from Pollen Robotics' reachy_mini project.
