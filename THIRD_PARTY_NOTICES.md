# Third-Party Notices

GRABETTE is © Pollen Robotics and released under the Apache License 2.0 (see
[`LICENSE`](LICENSE)). It incorporates, depends on, or derives from the
third-party components below, each under its own license. This file is provided
for attribution and license compliance.

---

### Luxonis DepthAI — MIT
<https://github.com/luxonis/depthai-core>

The `depthai` Python API is a runtime dependency of the on-device capture
service (`packages/grabette`).

### OpenArm — Apache-2.0 (Enactic, Inc.)
<https://github.com/enactic/openarm>

The robot description in `packages/gripette/urdf/openarm_gripette_right` is
derived from the OpenArm arm model, adapted to carry the Gripette end-effector.

### onshape-to-robot — MIT (Rhoban)
<https://github.com/Rhoban/onshape-to-robot>

Used to generate the Grabette and Gripette URDF / MuJoCo models from Onshape CAD.

### reachy_mini — Apache-2.0 (Pollen Robotics)

Portions of `packages/grabette` (`bluetooth/bluetooth_service.py`, `auth.py`) are
adapted from Pollen Robotics' reachy_mini project.
