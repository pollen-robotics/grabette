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
    ├── set_arm_torque.py            # enable/disable arm torque (--off = arm falls!)
    ├── disable_all_motors_can.py    # software e-stop, direct CAN (no server needed)
    ├── _torque_guard.py             # shared Ctrl+C/crash handler: torque off on abort
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

## Prerequisites

1. **CAN bus setup** — *once per boot / re-plug*: the interface configuration does not survive a power cycle or unplugging the CAN adapter. See LeRobot's [Damiao guide](https://huggingface.co/docs/lerobot/damiao):
   ```bash
   uv run lerobot-setup-can --mode=setup --interfaces=can0
   uv run lerobot-setup-can --mode=test  --interfaces=can0
   ```

2. **Firmware-zero calibration** — *once per physical arm*: the zeros are written into the Damiao motor firmware, so re-run only after replacing a motor or mechanically disassembling the arm. Uses OpenArm's official calibration procedure (**not** LeRobot's calibration — `OpenArm7Follower.is_calibrated` returns `True` unconditionally).

   ```bash
   uv run python examples/calibrate_arm_no_gripper.py --canport can0 --arm-side right_arm
   ```
   The tool uses OpenArm's [`openarm_can`](https://github.com/enactic/openarm_can) bindings — a regular dependency of this package (pinned git tag, compiled from C++ at install time), so the standard `uv sync --package openarm-gripette` provides it. The only system requirement is a C++ compiler.
   > ⚠️ Upstream LeRobot's `OpenArmFollower.connect()` **re-zeros all motors at the current pose on every connect** (its own calibration convention). `OpenArm7Follower` overrides `connect()` to skip that — but if you ever ran a server version without this override, your firmware zeros were silently overwritten at each server start: re-run the calibration once.

3. **Direction check** — *once per physical arm*: for each joint 1..7, command 30° alone and verify the physical direction matches the simulator. If a motor is wired reversed, set `joint_signs = {"joint_i": -1.0}` in the config so read/write symmetrically flip.

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

# 2. Can you home the arm smoothly? Requires a waypoint source (--preset or
#    repeated --waypoint_deg); add --dry_run first to preview the plan.
uv run python examples/reset_arm.py --arm_addr <robot-ip>:50052 --preset home_right_over_table

# 3. Does an absolute pose command track? (no --joints → default home pose)
uv run python examples/set_arm_pose.py --arm_addr <robot-ip>:50052

# 4. Do delta Cartesian commands trace a square? (Trips workspace + IK-jump issues.)
#    Tiny 2 cm square first, then a 10 cm one. Default --loops 0 runs forever; Ctrl+C to stop.
uv run python examples/cartesian_square.py --arm_addr <robot-ip>:50052 --tiny --loops 1
uv run python examples/cartesian_square.py --arm_addr <robot-ip>:50052 --half_size 0.05 --loops 1
```

If step 4 fails at 10 cm, don't run a policy — you have a URDF / home-pose / IK-jump issue that will silently produce garbage trajectories.

## Torque safety

- **Every motion example disables arm torque on Ctrl+C or crash** (via `ArmService.SetTorque`): the motors freewheel and **the arm falls under gravity** — be ready to catch it or let it drop safely. Pass `--keep_torque` to leave the arm holding instead. Normal completion always leaves torque on.
- Manual control:
  ```bash
  uv run python examples/set_arm_torque.py --arm_addr <robot-ip>:50052 --off   # arm falls!
  uv run python examples/set_arm_torque.py --arm_addr <robot-ip>:50052 --on    # enabled but limp — Reset to home next
  ```
- Stopping the **server** (Ctrl+C) also disables torque on disconnect. The Damiao disable frames are fire-and-forget (a motor that misses one silently stays powered), so the server sends a redundant double-pass; if a joint still holds torque after the server is gone, run the direct-CAN software e-stop **on the CAN machine, with the server stopped**:
  ```bash
  uv run python examples/disable_all_motors_can.py --can_port can0
  ```
  It also reports which motors ACK — a motor that never ACKs has a wiring/communication problem.
- None of this replaces the hardware e-stop — keep it within reach whenever the arm is powered.
- `set_arm_torque.py --on` now **holds the current pose** (the interpolator target is set to the measured joints), so a hand-placed arm stays where you put it. Use with `evaluate.py --no_reset` to start an episode from a hand-chosen pose.

## After a collision: recovering undetected motors

A hard impact can make one or more motors disappear from detection. Recovery,
in order of least-invasive (field-validated after a table strike that took out
joints 1 and 7):

1. **Full power cycle** (PSU off ≥30 s, not just software) — Damiao motors
   latch protection faults (over-current from the impact) that only clear on
   power-off. This alone often brings motors back.
2. **A motor with a RED LED has power and is latching a fault** — it is NOT a
   dead cable. Clear the latch over CAN without another power cycle (server
   stopped, on the CAN machine):
   ```bash
   uv run python examples/clear_motor_fault.py --can_port can0 --motor 7
   ```
   This sends the Damiao clear-error frame (`[0xFF]*7 + 0xFB`, same protocol
   family as enable/disable) and health-checks every motor with REFRESH. An
   over-current latch — the typical post-impact fault — clears immediately
   (validated: red LED off, motor detected again).
3. **Reseat CAN/power connectors** at the affected motors and along the
   harness near the impact path. If an END-of-chain motor is flaky, check the
   CAN terminator's seating — a loose terminator degrades detection bus-wide.
4. **Power off and rotate the joint by hand**: smooth = mechanics fine (the
   fault was electrical); grinding/notchy/blocked = gearbox damage.
5. If the red LED returns after every clear + power cycle, read the error
   register with the Damiao USB debug tool — persistent encoder faults mean
   hardware replacement.

After recovery, verify calibration hasn't shifted (`read_arm_state.py` at a
known pose) before the first torque-on.

## Using with the LeRobot CLI

If you want `--robot.type=openarm7_follower` to work with `lerobot-record`, `lerobot-teleoperate`, etc., add an explicit import at the top of your script:

```python
import openarm_gripette  # registers OpenArm7Follower with draccus
```

Alternatively, LeRobot's plugin auto-discovery (`register_third_party_plugins`) will find this package if it's installed under a distribution name starting with `lerobot_robot_`. We chose the plain `openarm_gripette` name for readable direct imports; if you need auto-discovery in a shared install, publish under `lerobot_robot_openarm_gripette` or add a shim distribution.

## Design notes

- `OpenArm7Follower.get_observation` / `send_action` apply a per-joint sign flip (`config.joint_signs`) symmetrically on read and write — the URDF convention is preserved upstream regardless of individual motor wiring.
- LeRobot's file-based calibration (`homing_offset`, `drive_mode`, `range_*`) is vestigial for this arm: the Damiao bus never reads those fields, motor zeros live in firmware. We disable LeRobot calibration entirely to avoid prompts and accidental zero overwrites.
- `OpenArm7Follower.connect` overrides the upstream method for one reason: upstream calls `bus.set_zero_position()` on every connect (writing the current pose into the motor firmware as zero), which destroys the OpenArm official calibration whenever the server starts with the arm away from the calibration pose. Our override is byte-identical otherwise (bus + cameras + configure + enable torque).
- `grpc_server_real.py` uses the same `Kinematics` (Placo, from `openarm_gripette_simu`) as the simulator. FK/IK are bit-for-bit identical; the only difference between sim and real is the CAN-side backend.
- `Reset` on the real arm interpolates to the home pose (can't teleport a real arm). Cube-randomization arguments to `Reset` are no-ops on real — they return dummy cube coordinates for API compat with the simulator.
- `GetSuccessStatus` on the real server always returns `goal_reached=False` — there's no cube tracking on hardware. Success determination is up to the client.
