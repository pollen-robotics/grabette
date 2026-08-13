"""openarm_gripette — real-hardware driver for the OpenArm-Gripette follower.

Wraps LeRobot's `OpenArmFollower` (upstream) as a 7-DoF variant that leaves the
gripper off the CAN bus — the Gripette is driven separately over its own gRPC
service. See `grpc_server_real.py` for the arm-side gRPC server that mirrors
the simulator's `ArmService` API for eval scripts that target sim or real
interchangeably.

Importing this module also registers `openarm7_follower` as a LeRobot config
choice (via `@RobotConfig.register_subclass`), so `--robot.type=openarm7_follower`
works after `import openarm_gripette` in any script that dispatches to LeRobot.
"""

from openarm_gripette.config_openarm7_follower import (
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
    OpenArm7FollowerConfig,
)
from openarm_gripette.openarm7_follower import OpenArm7Follower

__all__ = [
    "LEFT_DEFAULT_JOINTS_LIMITS",
    "RIGHT_DEFAULT_JOINTS_LIMITS",
    "OpenArm7Follower",
    "OpenArm7FollowerConfig",
]
