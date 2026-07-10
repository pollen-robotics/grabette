# OpenArm Gripette Simulation

MuJoCo simulation of the OpenArm right arm + Gripette gripper with gRPC control interfaces, policy evaluation support, and synthetic data collection in LeRobot format.

## Install

`placo` (used for kinematics) dynamically links the system `urdfdom` libraries,
which are **not** installed by `uv`. On Debian/Ubuntu, install them first:

```bash
sudo apt install -y liburdfdom-sensor4.0 liburdfdom-model4.0 liburdfdom-world4.0
# or the catch-all: sudo apt install -y liburdfdom-dev
```

Without them, `import placo` (and the sim) fails with
`ImportError: liburdfdom_sensor.so.4.0: cannot open shared object file`.

Then sync the Python environment:

> Part of the uv **workspace**: a bare `uv sync` here would build the *entire
> monorepo* environment. Always pass `--package` (root README → Development).

```bash
uv sync --package openarm-gripette-simu                  # base install
uv sync --package openarm-gripette-simu --extra dataset  # + lerobot (dataset collection)
uv sync --package openarm-gripette-simu --extra dev      # + grpcio-tools (proto stubs)
```

The robot model comes from the workspace package
[`openarm_gripette_model`](../openarm_gripette_model), resolved locally from the
monorepo via `[tool.uv.sources]` (`openarm-gripette-model = { workspace = true }`)
— no external fetch.

## Run the server

```bash
# With 3D viewer
uv run python -m openarm_gripette_simu --scene scenes/table_red_cube.xml \
    --initial-joints 0.0 0 0 1.57 0 0 0

# Headless
uv run python -m openarm_gripette_simu --scene scenes/table_red_cube.xml --headless
```

Two gRPC services on separate ports:

| Service | Port | Description |
|---------|------|-------------|
| GripperService | 50051 | Camera stream + gripper motor control (same API as real Gripette) |
| ArmService | 50052 | Delta Cartesian arm control, episode reset, success detection |

## gRPC API

### GripperService (port 50051)

Identical to the real [Gripette](../../../packages/gripette) gRPC API.

| RPC | Description |
|-----|-------------|
| `StreamState` | 50Hz JPEG camera frames + motor positions + timestamp |
| `SendMotorCommand(m1, m2)` | Set gripper joint goals (rad) |
| `ReadMotors` | Read gripper joint positions (rad) |
| `SetTorque(enable)` | No-op in simulation |
| `Ping` | Health check |

### ArmService (port 50052)

| RPC | Description |
|-----|-------------|
| `SendCartesianDelta(dx, dy, dz, dr6d[6])` | Apply delta to end-effector pose (meters + 6D rotation). **Deltas are camera-LOCAL — see below.** |
| `GetArmState` | Current camera-frame pose (xyz + r6d) and joint positions |
| `Reset(joint_positions=[])` | Reset episode: randomize cube + arm (empty joints = random start) |
| `GetSuccessStatus` | Returns `goal_reached` (cube touched) + displacement |
| `Ping` | Health check |

The 6D rotation representation (Zhou et al., CVPR 2019) matches the action space used by diffusion policies trained with LeRobot.

#### `SendCartesianDelta` is camera-local (integrator semantics)

`(dx, dy, dz)` and `dr6d` are interpreted in the **integrator's current
camera-frame target**, not in the world frame. The server maintains an
internal `(_target_pos, _target_r6d)` initialized from FK on startup (and
re-synced after `Reset`), and each call updates it as:

```python
R_target = rotation_6d_to_matrix(self._target_r6d)
self._target_pos = self._target_pos + R_target @ (dx, dy, dz)   # local → world
self._target_r6d = rotation_matrix_to_6d(R_target @ rotation_6d_to_matrix(dr6d))
# IK solves to the new (target_pos, target_r6d) and commands the arm joints.
```

This matches `convert_dataset.py` in lerobot, which builds the per-frame
action as `R[t].T @ Δpos_world` and `R[t].T @ R[t+1]`. The full rationale is
in `examples/openarm_gripette/README.md` → "Frame Convention" on the lerobot
side. To verify the integrator end-to-end against a running server, use the
**lerobot-side** `cartesian_square.py` (not the one in this repo's
`examples/`, see "Examples" below).

### Evaluation loop pattern

```python
import grpc
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc

stub = arm_pb2_grpc.ArmServiceStub(grpc.insecure_channel("localhost:50052"))

for episode in range(n_episodes):
    stub.Reset(arm_pb2.ResetRequest())  # randomized start + cube
    for step in range(max_steps):
        state = stub.GetArmState(arm_pb2.GetArmStateRequest())
        # ... policy inference ...
        stub.SendCartesianDelta(arm_pb2.CartesianDelta(...))
        status = stub.GetSuccessStatus(arm_pb2.SuccessStatusRequest())
        if status.goal_reached:
            break
```

### View the camera stream

```bash
uv run python -c "
import cv2, grpc, numpy as np
from openarm_gripette_simu.proto import gripper_pb2, gripper_pb2_grpc
stub = gripper_pb2_grpc.GripperServiceStub(grpc.insecure_channel('localhost:50051'))
for frame in stub.StreamState(gripper_pb2.StreamRequest()):
    img = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow('Gripette', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
"
```

## Synthetic data collection

Generates diverse reach-and-touch episodes in LeRobot v3 format or as raw numpy arrays.

Trajectory types (mixed randomly per episode):
- **Direct reach** (40%): cube visible at start, go straight to it
- **Scan then reach** (35%): camera looking away, sweep to find cube, then reach
- **Offset reach** (25%): start displaced sideways, re-center then reach

Each episode randomizes:
- Cube XY position (±6cm in X, ±20cm in Y) and yaw (full rotation)
- Arm start joints (±0.08 rad per joint)
- Cube placement uses MuJoCo collision detection to guarantee no contact with the arm at spawn

```bash
# LeRobot format (requires: uv sync --extra dataset)
uv run python examples/collect_reach_dataset.py \
    --repo-id user/simu_reach --episodes 500

# Raw npy/jpg format (no lerobot dependency)
uv run python examples/collect_reach_dataset.py \
    --raw --root data/raw_reach --episodes 500

# Push to HuggingFace Hub
uv run python examples/collect_reach_dataset.py \
    --repo-id user/simu_reach --episodes 500 --push

# Visual debug
uv run python examples/collect_reach_dataset.py \
    --repo-id local/test --episodes 5 --viewer
```

Dataset features match the Grabette training pipeline:
- `observation.state`: `[11]` float32 — `[x, y, z, r6d_0..5, proximal_deg, distal_deg]`
- `observation.images.cam0`: video, 972×1296 fisheye
- `action`: `[11]` float32 — same 11D (absolute, next-step target)

All rates aligned at **50fps** to match real Grabette data (20ms per frame).

## Grasp data collection

`examples/collect_grasp_dataset.py` generates scripted grasp-and-lift demos of the
arm picking a cylinder ("can") off the table, in LeRobot format, for training a
visuomotor grasp policy. The gripper is driven free-floating (welded mocap) along
an approach → descend → close → hold → lift trajectory; only successful episodes
(cube lifted ≥ 5 cm) are saved.

```bash
uv run python examples/collect_grasp_dataset.py \
    --episodes 200 --repo_id user/sim_grabette_grasp --push_to_hub --private
```

Dataset format (matches the real Grabette recordings):
- `observation.images.cam0`: **h264** video, **960×720** fisheye (distorted at the
  calibration resolution then downscaled), **30 fps**.
- `action`: `[8]` float32 — `[x, y, z, ax, ay, az, proximal, distal]`: the absolute
  `oak_l` (SLAM/control) pose + gripper joint **position** at the next step.
  `DiffusionPolicy/convert_dataset.py` turns these into the camera-local deltas the
  policy trains on.

Grasp-pose sampling (tunables in `examples/grabette_trajectory.py`):
- `GRASP_TILT_RANGE_DEG = (65, 75)` — tilt off vertical (0 = top-down). Diagonal
  grasps; top-down is both hardest to reach and near the wrist-roll singularity.
- `GRASP_AZIMUTH_RANGE_DEG = (10, 30)` — grasp yaw. Positive is the right arm's
  natural side; negative wraps the elbow into the pedestal.
- `GRASP_OFFSET_BODY` z (0.090) — how deep the can sits in the V (smaller = deeper).

Episode-type mix (CLI fractions): `--release_fraction` (start closed→open),
`--recover_fraction` (deliberate near-miss → reposition → re-grasp); the remainder
are normal grasps.

Two prefilters keep only **arm-executable** demos (data-gen is free-floating, so it
must check the arm explicitly):
- `--ik_filter` (default on) — rejects per-frame poses the OpenArm can't reach.
- `--pedestal_filter` (default on) — rejects configs that collide with the mounting
  column (shared `pedestal.py` clearance query — the same one the eval gate uses).

`scenes/grabette_grasp.xml` (the free-floating grasp scene) is regenerated from the
model by `scenes/gen_grabette_grasp.py` after a model re-export.

**Verify a collected dataset** with `examples/replay_dataset.py` (needs a display):

```bash
# Free-floating gripper + the recorded cam0 frames (check the data itself)
uv run python examples/replay_dataset.py --repo_id user/sim_grabette_grasp

# Replay the recorded oak_l poses on the FULL ARM (IK + physics) and report grasp
# success — confirms the saved episodes are arm-executable
uv run python examples/replay_dataset.py --repo_id user/sim_grabette_grasp --arm
```

## Evaluate a trained policy

`examples/evaluate.py` loads a trained policy (Diffusion / ACT / Pi0…, the type
is auto-detected from its `config.json`) and drives the arm over gRPC for N
episodes, reporting grasp success. The policy must have been trained on the
camera-local delta format produced by the `DiffusionPolicy` integration's
`convert_dataset.py` (2D gripper state, 11D delta action).

```bash
uv sync --package openarm-gripette-simu --extra eval   # adds lerobot + scipy

# Terminal 1 — arm grasp scene, headless
uv run python -m openarm_gripette_simu --scene scenes/table_grasp.xml --headless

# Terminal 2 — run the policy
uv run python examples/evaluate.py \
    --checkpoint <user>/<model>-best \
    --num_episodes 30 --n_action_steps 8
```

- `--checkpoint` — local path or HF repo id.
- `--n_action_steps 8` — committed grasp (lower = more reactive approach, but can hesitate on the trigger).
- `--debug` — show the camera feed; `--log_gripper` — print the gripper command vs observed state each step.
- `--clamp_pos_mm` / `--clamp_rot_deg` — cap per-step Cartesian deltas (stability test).

### Eval safety gates (`arm_servicer`)

`SendCartesianDelta` validates each IK solution before applying it (holding the
last-good command + rolling back the integrator target if it fails):
- **Pedestal gate** — rejects a config that moves a guarded link within 2 cm of the
  mounting column. Moves that *increase* clearance are always allowed, so the arm
  can never deadlock inside the margin.
- **Singularity guard** — rejects an IK solution that jumps more than
  `JOINT_JUMP_LIMIT` (0.5 rad) from the last command — the signature of a
  near-singular "explosion" (it also stops the arm flailing into the column).

Both use the shared `pedestal.py` clearance query against the `pedestal_box` proxy
geom in `scenes/table_grasp.xml` (the same query the data-gen `--pedestal_filter`
uses, so demos and eval enforce the identical constraint). Per-episode the servicer
logs `[grasp-diag]` (commanded-vs-actual grasp gap) and `[grasp-orient]` (grasp
tilt/azimuth) at `GetSuccessStatus`, for diagnosing systematic misses.

## Scenes

Scene XML files live in `scenes/`. They include the robot model and add environment elements.

| Scene | Description |
|-------|-------------|
| `scenes/table_red_cube.xml` | Wooden table with a movable red cube (reach task) |
| `scenes/table_grasp.xml` | Arm + table + grasp cylinder + `pedestal_box` proxy (grasp eval / replay) |
| `scenes/grabette_grasp.xml` | Free-floating gripper + cylinder (grasp data generation); regenerated by `scenes/gen_grabette_grasp.py` |

To create a new scene, copy an existing one as a template. The robot is included via `<include file="...robot.xml"/>`. The `Simulation` class injects the correct `meshdir` so scenes can live anywhere.

## Examples

Standalone demos (no gRPC — each script spawns its own MuJoCo simulation):

```bash
uv run python examples/cartesian_square.py   # Standalone Cartesian square (NOT a gRPC client)
uv run python examples/joint_control.py      # Move joints one by one
uv run python examples/gripper_demo.py       # Gripper open/close cycle
uv run python examples/table_scene.py        # Table scene with red cube
uv run python examples/grpc_client_demo.py   # gRPC client (requires server running)
```

> **Two `cartesian_square.py` files exist.** The one in *this* repo
> (`openarm_gripette_simu/examples/cartesian_square.py`) is **standalone**:
> it imports `Simulation`/`Kinematics` directly and runs its own MuJoCo
> instance with world-frame waypoints. It is **not** a gRPC client — running
> `python -m openarm_gripette_simu --scene ...` in another terminal has no
> effect on it. The canonical end-to-end smoke test of the camera-local
> delta convention is the *other* `cartesian_square.py` in the lerobot
> repo: `examples/openarm_gripette/cartesian_square.py`. That one is a gRPC
> client and tests the integrator semantics described above.

## Camera

The simulated Gripette camera matches the real camera calibration:
- Resolution: 1296×972
- Lens model: KannalaBrandt8 fisheye
- MuJoCo renders a wide-FOV (130°) pinhole image, then remaps with real distortion coefficients

Camera rendering is done in the main thread (alongside physics and viewer) and cached; gRPC streams read the cached frame to avoid GL context conflicts.

## Timing

| Component | Rate |
|-----------|------|
| Physics | 500Hz (dt=0.002s) |
| Camera render | 50Hz |
| Gripper gRPC stream | 50Hz |
| Dataset FPS | 50 |
| Viewer sync | 60Hz |

## Regenerating proto stubs

```bash
uv sync --package openarm-gripette-simu --extra dev
uv run python -m grpc_tools.protoc -I proto \
    --python_out=openarm_gripette_simu/proto \
    --grpc_python_out=openarm_gripette_simu/proto \
    --pyi_out=openarm_gripette_simu/proto \
    proto/gripper.proto proto/arm.proto
```

Then fix the imports in generated `*_pb2_grpc.py` files: change `import xxx_pb2` to `from . import xxx_pb2`.
