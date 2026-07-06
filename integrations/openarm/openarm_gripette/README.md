# openarm_gripette

Real-hardware driver for the OpenArm-Gripette follower (7-DoF arm, gripette gripper on its own service). Wraps upstream LeRobot's `OpenArmFollower` as a 7-joint variant that leaves the gripper off the CAN bus, and exposes the arm over the same gRPC `ArmService` API as the simulator.

For the wider package map, see [`../README.md`](../README.md).

## What's in the package

```
openarm_gripette/
├── openarm7_follower.py           # OpenArm7Follower — subclass of lerobot.robots.OpenArmFollower
├── config_openarm7_follower.py    # @RobotConfig.register_subclass("openarm7_follower")
├── grpc_server_real.py            # ArmService gRPC server (mirrors simulator's API)
└── examples/
    ├── calibrate_arm_no_gripper.py  # per-joint calibration via openarm-can
    ├── read_arm_state.py            # observability
    ├── reset_arm.py                 # smooth interpolation to home
    ├── set_arm_pose.py              # absolute-target test
    ├── set_gripper_pose.py          # gripette gRPC test
    ├── view_camera.py               # gripette camera stream test
    ├── cartesian_sinusoid.py        # sinusoid delta demo
    └── cartesian_square.py          # square-trajectory smoke test (do this BEFORE any policy run)
```

## Install

The package is part of the monorepo `uv` workspace:

```bash
cd /path/to/grabette
uv sync --package openarm-gripette
```

Deps (auto): `lerobot[damiao]` from upstream PyPI, `openarm-gripette-simu` (local workspace, provides proto + kinematics + rotation), `grpcio`, `numpy`.

## Prerequisites (once per physical arm)

1. **CAN bus setup** — see LeRobot's [Damiao guide](https://huggingface.co/docs/lerobot/damiao):
   ```bash
   lerobot-setup-can --mode=setup --interfaces=can0
   lerobot-setup-can --mode=test  --interfaces=can0
   ```

2. **Firmware-zero calibration** — motor zeros live in the Damiao motor firmware, set via OpenArm's official calibration tool (**not** LeRobot's calibration — `OpenArm7Follower.is_calibrated` returns `True` unconditionally):
   ```bash
   uv run python examples/calibrate_arm_no_gripper.py --can_port can0 --side right
   ```

3. **Direction check** — for each joint 1..7, command 30° alone and verify the physical direction matches the simulator. If a motor is wired reversed, set `joint_signs = {"joint_i": -1.0}` in the config so read/write symmetrically flip.

## Start the gRPC arm server

The server exposes `ArmService` on the CAN-connected machine:

```bash
uv run python -m openarm_gripette.grpc_server_real \
    --can_port can0 --side right --arm_port 50052
```

Client machines (running eval / teleop) connect to `arm_addr = <robot-ip>:50052` and to the gripette's own gRPC service (`packages/gripette`) for the gripper.

## Verification sequence (before any policy run)

Run these from the client machine, in this order. Each has a clear pass/fail signal:

```bash
# 1. Can you read the state?
uv run python examples/read_arm_state.py --arm_addr <robot-ip>:50052

# 2. Can you home the arm smoothly?
uv run python examples/reset_arm.py --arm_addr <robot-ip>:50052

# 3. Does an absolute pose command track?
uv run python examples/set_arm_pose.py --arm_addr <robot-ip>:50052

# 4. Do delta Cartesian commands trace a square? (Trips workspace + IK-jump issues.)
uv run python examples/cartesian_square.py --arm_addr <robot-ip>:50052 --size 0.02
uv run python examples/cartesian_square.py --arm_addr <robot-ip>:50052 --size 0.10
```

If step 4 fails at 10 cm, don't run a policy — you have a URDF / home-pose / IK-jump issue that will silently produce garbage trajectories.

## Using with the LeRobot CLI

If you want `--robot.type=openarm7_follower` to work with `lerobot-record`, `lerobot-teleoperate`, etc., add an explicit import at the top of your script:

```python
import openarm_gripette  # registers OpenArm7Follower with draccus
```

Alternatively, LeRobot's plugin auto-discovery (`register_third_party_plugins`) will find this package if it's installed under a distribution name starting with `lerobot_robot_`. We chose the plain `openarm_gripette` name for readable direct imports; if you need auto-discovery in a shared install, publish under `lerobot_robot_openarm_gripette` or add a shim distribution.

## Design notes

- `OpenArm7Follower.get_observation` / `send_action` apply a per-joint sign flip (`config.joint_signs`) symmetrically on read and write — the URDF convention is preserved upstream regardless of individual motor wiring.
- LeRobot's file-based calibration (`homing_offset`, `drive_mode`, `range_*`) is vestigial for this arm: the Damiao bus never reads those fields, motor zeros live in firmware. We disable LeRobot calibration entirely to avoid prompts and accidental zero overwrites.
- `grpc_server_real.py` uses the same `Kinematics` (Placo, from `openarm_gripette_simu`) as the simulator. FK/IK are bit-for-bit identical; the only difference between sim and real is the CAN-side backend.
- `Reset` on the real arm interpolates to the home pose (can't teleport a real arm). Cube-randomization arguments to `Reset` are no-ops on real — they return dummy cube coordinates for API compat with the simulator.
- `GetSuccessStatus` on the real server always returns `goal_reached=False` — there's no cube tracking on hardware. Success determination is up to the client.
