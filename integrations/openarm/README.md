# OpenArm-Gripette integration

The **OpenArm-Gripette** is our custom robot platform: a 7-DoF [OpenArm](https://openarm.dev) follower with the Grabette team's [Gripette](../../packages/gripette/) gripper as end-effector, instead of the stock OpenArm gripper. It's built to be teleoperated via the [Grabette](../../packages/grabette/) motion-capture glove and used as the target platform for imitation-learning policies trained on Grabette-collected data.

Three sibling packages make it up:

| Package | Purpose |
|---------|---------|
| [`openarm_gripette_model/`](openarm_gripette_model/) | URDF + MuJoCo XML + meshes for the OpenArm-right + gripette assembly |
| [`openarm_gripette_simu/`](openarm_gripette_simu/) | MuJoCo simulator — arm + gripper gRPC servicers, grasp scenes, Placo-based kinematics, proto definitions |
| [`openarm_gripette/`](openarm_gripette/) | **Real-hardware** driver + gRPC server for the OpenArm 7-DoF arm side (CAN bus). Consumes upstream LeRobot's `OpenArmFollower` base class |

## Design principle: same gRPC interface for sim and real

The whole point of the layout is that a **single eval script targets either sim or real** via `--arm_addr` / `--gripper_addr`. Both sides implement the same `arm.proto` + `gripper.proto`:

```
Eval / teleop client  ──── ArmService (gRPC) ─────►  SIM:  openarm_gripette_simu.arm_servicer
                                                    REAL: openarm_gripette.grpc_server_real
Eval / teleop client  ──── GripperService (gRPC) ─►  SIM:  openarm_gripette_simu.gripper_servicer
                                                    REAL: gripette's own gRPC service (packages/gripette)
```

The **proto stubs, kinematics, and rotation utilities** live in `openarm_gripette_simu/` and are shared with `openarm_gripette/` (real-hardware side) so the two implementations are bit-for-bit interchangeable. This "little coupling" is the deliberate cross-package dependency: `openarm_gripette` depends on `openarm_gripette_simu` for those utilities only, not for the MuJoCo runtime.

## Dependency graph

```
openarm_gripette_model  (URDF + meshes, no code deps)
     ▲
     │
openarm_gripette_simu  (MuJoCo sim + proto + kinematics + rotation)
     ▲
     │
openarm_gripette  (real hardware + shared proto/kinematics from _simu)
     │
     └── depends on: lerobot[damiao] (upstream, PyPI)
```

## Known wart

`openarm_gripette` transitively pulls MuJoCo (via `openarm_gripette_simu`) even for real-robot-only deployments. Real-hardware users don't need it. The wart is intentional for the initial release — flagged for future refactor into an `openarm_gripette_core` (proto + kinematics + rotation, no MuJoCo) + `openarm_gripette_simu` (MuJoCo scenes + servicers, depends on `_core`). Not blocking.

## Getting started

Point of entry depends on what you want to do:

- **Teleoperate the real robot from a Grabette glove** — see [`openarm_gripette/README.md`](openarm_gripette/README.md)
- **Run a trained policy on the real robot** — see [`openarm_gripette/README.md`](openarm_gripette/README.md) (server side) + [`integrations/DiffusionPolicy/README.md`](../DiffusionPolicy/README.md) (client side, eval script)
- **Same, but in simulation first** — see [`openarm_gripette_simu/README.md`](openarm_gripette_simu/README.md)
- **Regenerate the URDF/MuJoCo model** — see [`openarm_gripette_model/README.md`](openarm_gripette_model/README.md)
