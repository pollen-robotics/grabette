"""Send goal positions to the Gripette's 2-DOF gripper.

Talks to the GripperService (same proto for sim or real Gripette).
Two motors: proximal (motor1) and distal (motor2), both in radians.

By dataset convention (from `check_action_means.py`):
  Fully open:   motor1 =  0.0,  motor2 =  0.0   (~ both joints unfolded)
  Fully closed: motor1 = -1.5,  motor2 = -2.1   (V-pocket closed on object)

Usage:
  # Send explicit goals (radians)
  uv run python examples/set_gripper_pose.py \\
      --gripper_addr localhost:50051 --motors -1.5 -2.1

  # Or in degrees (added convenience)
  uv run python examples/set_gripper_pose.py \\
      --gripper_addr localhost:50051 --motors_deg -85 -120

  # Presets
  uv run python examples/set_gripper_pose.py --gripper_addr ... --open
  uv run python examples/set_gripper_pose.py --gripper_addr ... --close

  # Just read the current motor positions (no command)
  uv run python examples/set_gripper_pose.py \\
      --gripper_addr localhost:50051 --read

  # Enable / disable torque (free-move / hold)
  uv run python examples/set_gripper_pose.py \\
      --gripper_addr localhost:50051 --torque off
"""

import argparse
import logging
import math
import sys

import grpc
from openarm_gripette_simu.proto import gripper_pb2, gripper_pb2_grpc

logger = logging.getLogger(__name__)

# Dataset-canonical extremes (from check_action_means.py).
OPEN_GOAL = (0.0, 0.0)
CLOSE_GOAL = (-1.5, -2.1)


def parse_args():
    p = argparse.ArgumentParser(description="Send goal positions to the Gripette gripper")
    p.add_argument(
        "--gripper_addr", type=str, default="localhost:50051",
        help="GripperService gRPC address (real Gripette or simulator).",
    )

    cmd = p.add_mutually_exclusive_group()
    cmd.add_argument(
        "--motors", type=float, nargs=2, metavar=("M1_RAD", "M2_RAD"),
        help="Motor goal positions in radians (proximal, distal).",
    )
    cmd.add_argument(
        "--motors_deg", type=float, nargs=2, metavar=("M1_DEG", "M2_DEG"),
        help="Motor goal positions in degrees.",
    )
    cmd.add_argument("--open", action="store_true",
                     help=f"Preset: send {OPEN_GOAL} rad (fully open).")
    cmd.add_argument("--close", action="store_true",
                     help=f"Preset: send {CLOSE_GOAL} rad (fully closed on object).")
    cmd.add_argument("--read", action="store_true",
                     help="Read and print current motor positions, send no command.")

    p.add_argument(
        "--torque", choices=["on", "off"], default=None,
        help="Enable or disable torque before/after the command. 'off' lets you "
             "freely move the fingers by hand; 'on' makes them hold position.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress the post-command read-back (faster for scripted use).",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    # Resolve goal in radians (if any).
    goal = None
    if args.motors is not None:
        goal = tuple(args.motors)
    elif args.motors_deg is not None:
        goal = tuple(math.radians(v) for v in args.motors_deg)
    elif args.open:
        goal = OPEN_GOAL
    elif args.close:
        goal = CLOSE_GOAL

    logger.info(f"Connecting to {args.gripper_addr}")
    channel = grpc.insecure_channel(args.gripper_addr)
    stub = gripper_pb2_grpc.GripperServiceStub(channel)

    try:
        ping = stub.Ping(gripper_pb2.PingRequest(), timeout=5.0)
    except grpc.RpcError as e:
        logger.error(f"Cannot reach GripperService at {args.gripper_addr}: {e}")
        channel.close()
        sys.exit(1)
    logger.info(f"GripperService: {ping.status} (uptime: {ping.uptime_seconds:.1f}s)")

    try:
        # Optional torque toggle BEFORE sending the goal (only meaningful for
        # 'on'; 'off' typically belongs at the end).
        if args.torque == "on":
            stub.SetTorque(gripper_pb2.TorqueCommand(enable=True))
            logger.info("Torque enabled (motors will hold position).")

        # Send goal if one was specified.
        if goal is not None:
            m1, m2 = goal
            logger.info(
                f"Sending motor goal: motor1={m1:+.4f} rad ({math.degrees(m1):+.1f}°), "
                f"motor2={m2:+.4f} rad ({math.degrees(m2):+.1f}°)"
            )
            resp = stub.SendMotorCommand(
                gripper_pb2.MotorCommand(motor1_goal=m1, motor2_goal=m2)
            )
            if not resp.success:
                logger.error(f"SendMotorCommand failed: {resp.error}")
                channel.close()
                sys.exit(2)

        # Read back current positions (skip with --quiet, or always print if --read).
        if args.read or not args.quiet:
            state = stub.ReadMotors(gripper_pb2.ReadMotorsRequest(), timeout=5.0)
            logger.info(
                f"Current: motor1={state.motor1_position:+.4f} rad "
                f"({math.degrees(state.motor1_position):+.1f}°), "
                f"motor2={state.motor2_position:+.4f} rad "
                f"({math.degrees(state.motor2_position):+.1f}°)"
            )

        # Optional torque-off AFTER the read-back (so you can free-move from here).
        if args.torque == "off":
            stub.SetTorque(gripper_pb2.TorqueCommand(enable=False))
            logger.info("Torque disabled (fingers are now free to move by hand).")

    finally:
        channel.close()


if __name__ == "__main__":
    main()
