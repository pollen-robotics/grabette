"""Move the robot arm smoothly to a specified joint configuration.

Works with the simulator OR the real-robot gRPC server. Calls the Reset RPC,
which handles the smooth interpolation server-side (3 seconds by default).

Usage:
  # Move to a specific 7-joint configuration (radians)
  uv run python examples/openarm_gripette/set_arm_pose.py \\
      --arm_addr localhost:50052 \\
      --joints 0.0 0.0 0.0 1.57 0.0 0.0 0.0

  # Move to the default home pose (no --joints argument)
  uv run python examples/openarm_gripette/set_arm_pose.py \\
      --arm_addr localhost:50052

  # Move to a pose specified in degrees (added convenience)
  uv run python examples/openarm_gripette/set_arm_pose.py \\
      --arm_addr localhost:50052 \\
      --joints_deg 0 0 0 90 0 0 0

Joint order (simulator convention):
  [r_arm_pitch, r_arm_roll, r_arm_yaw, r_elbow, r_wrist_yaw, r_wrist_roll, r_wrist_pitch]
"""

import argparse
import logging
import math

import grpc
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Move robot arm to a specified pose")
    p.add_argument(
        "--arm_addr",
        type=str,
        default="localhost:50052",
        help="ArmService gRPC address",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--joints",
        type=float,
        nargs=7,
        metavar="RAD",
        help="7 joint positions in radians",
    )
    group.add_argument(
        "--joints_deg",
        type=float,
        nargs=7,
        metavar="DEG",
        help="7 joint positions in degrees (converted to radians)",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    # Build joint positions list (empty = use server default / home)
    joints = []
    if args.joints is not None:
        joints = args.joints
    elif args.joints_deg is not None:
        joints = [math.radians(d) for d in args.joints_deg]

    if joints:
        logger.info(f"Target joints (rad): {[round(j, 4) for j in joints]}")
    else:
        logger.info("Target: server default home pose")

    logger.info(f"Connecting to {args.arm_addr}")
    channel = grpc.insecure_channel(args.arm_addr)
    stub = arm_pb2_grpc.ArmServiceStub(channel)

    # Verify connection
    ping = stub.Ping(arm_pb2.ArmPingRequest())
    logger.info(f"Server responded: {ping.status} (uptime: {ping.uptime_seconds:.1f}s)")

    # Send Reset with target joints (blocks until the interpolation finishes)
    logger.info("Sending Reset (smooth interpolation, ~3s)...")
    response = stub.Reset(arm_pb2.ResetRequest(joint_positions=joints))

    if response.success:
        logger.info("Done. Arm at target position.")
    else:
        logger.error(f"Reset failed: {response.error}")

    channel.close()


if __name__ == "__main__":
    main()
