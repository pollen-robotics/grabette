# OpenArm-Gripette model

URDF + MuJoCo (MJCF) description and mesh assets for the OpenArm right arm with the Gripette gripper as end-effector, generated from Onshape via [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot). Consumed as a workspace package by [`openarm_gripette_simu`](../openarm_gripette_simu) (and, for shared kinematics, `openarm_gripette`).

## Contents

```
openarm_gripette_model/openarm_right/
├── robot.urdf        # URDF (config output_format = urdf)
├── robot.xml         # MuJoCo MJCF (post-processed — see below)
├── scene.xml         # MuJoCo scene wrapper (offscreen render size 1296×972)
├── assets/           # STL meshes (simplify_stls = true)
└── config.json       # onshape-to-robot config: Onshape doc URL, robot_name, output_format, simplify_stls
```

## Regenerating from Onshape

Requires `onshape-to-robot` with valid Onshape API credentials (see its docs). From this package directory:

```bash
uv run onshape-to-robot-mujoco openarm_right   # pull CAD from Onshape → robot.urdf / robot.xml / assets
uv run python postprocess_mujoco.py            # REQUIRED after every generation (see below)
```

`postprocess_mujoco.py` re-applies patches that the raw Onshape export does **not** produce, so the model works in the sim. It edits:

- **robot.xml** — strips `meshdir` and prefixes mesh filenames with `assets/` (so the model is `<include>`-able from scene files in other directories); sets `contype`/`conaffinity` on the collision geom class (no self-collision between adjacent links); raises armature (0.005 → 0.1) and actuator gains (kp 50 → 500) for stiffer position tracking; adds the Gripette camera next to the camera site (180° pitch correction, `fovy=130`).
- **scene.xml** — adds `offwidth="1296" offheight="972"` to `<global>` for offscreen rendering.

> A re-export can silently regress collision classes, gains, axes, and joint ranges. After regenerating, verify the model in the sim (a `cartesian_square` run + a grasp episode) before trusting it — don't assume the export is drop-in.
