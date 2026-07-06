"""Multi-waypoint safe reset for the OpenArm.

Visits a sequence of joint configurations in order, using ArmService.Reset for
each hop. The server smoothly interpolates between the current pose and each
waypoint over RESET_DURATION_S seconds (~3s), so the motion between waypoints
is always a straight line in joint space — we just control the *endpoints*.

Useful when a direct move to the home pose would collide with the environment
(e.g., a table in front of the arm): define intermediate waypoints that trace
a safe path around the obstacle.

Waypoints can be specified via a named preset (easy to edit below), or via
repeated --waypoint_deg flags on the CLI.

Usage:
  # Named preset (edit the PRESETS dict below to tune for your setup)
  uv run python examples/reset_arm.py \\
      --arm_addr 192.168.10.147:50052 --preset home_right_over_table

  # Explicit waypoints on the CLI (7 space-separated joint angles in degrees,
  # repeat the flag for each waypoint). Negative values work fine this way.
  uv run python examples/reset_arm.py \\
      --arm_addr 192.168.10.147:50052 \\
      --waypoint_deg -10 0 0 0 0 0 0 \\
      --waypoint_deg -10 0 0 100 0 0 0 \\
      --waypoint_deg 0 0 0 90 0 0 0

  # Dry run: print the planned sequence without sending anything
  uv run python examples/reset_arm.py \\
      --preset home_right_over_table --dry_run
"""

import argparse
import logging
import math
import time

import grpc
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc

from _torque_guard import abort_torque_off, add_keep_torque_arg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Presets — joint configurations in DEGREES.
# Order: joint_1..joint_7 (same as the simulator KIN_ARM_JOINT_NAMES order).
# Each preset is a list of waypoints visited in order; the last one is the
# final resting pose. Edit freely to match your table / workspace geometry.
# ---------------------------------------------------------------------------
PRESETS: dict[str, list[list[float]]] = {
    # Lifts the elbow progressively so the EE stays above the table while the
    # shoulder rotates into position. Tune the intermediate waypoints for your
    # actual table height / arm mounting.
    "home_right_over_table": [
        [-30.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0],
        [-30.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0],
        [-30.0, 0.0, 0.0, 120.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0],
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="Multi-waypoint safe reset for the OpenArm")
    p.add_argument("--arm_addr", type=str, default="localhost:50052")
    p.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        default=None,
        help=f"Named waypoint sequence. Available: {', '.join(PRESETS.keys())}",
    )
    p.add_argument(
        "--waypoint_deg",
        type=float,
        nargs=7,
        action="append",
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Waypoint as 7 space-separated joint angles in degrees. "
        "Repeat the flag for each waypoint (visited in order). "
        "Negative values are supported (unlike a comma-joined single string).",
    )
    p.add_argument(
        "--pause_s",
        type=float,
        default=0.3,
        help="Seconds to pause between waypoints (lets motion settle).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the planned waypoint sequence without sending any commands.",
    )
    add_keep_torque_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    # Resolve waypoints: --waypoint_deg takes precedence over --preset.
    waypoints_deg: list[list[float]]
    if args.waypoint_deg:
        waypoints_deg = args.waypoint_deg
        source = "cli"
    elif args.preset:
        waypoints_deg = PRESETS[args.preset]
        source = f"preset {args.preset!r}"
    else:
        raise SystemExit("Must specify either --preset or --waypoint_deg (at least one).")

    logger.info(f"Plan ({source}): {len(waypoints_deg)} waypoint(s)")
    for i, wp in enumerate(waypoints_deg):
        logger.info(f"  [{i + 1}/{len(waypoints_deg)}]  deg={wp}")

    if args.dry_run:
        logger.info("Dry run — exiting without sending commands.")
        return

    # Convert to radians for the RPC.
    waypoints_rad = [[math.radians(v) for v in wp] for wp in waypoints_deg]

    logger.info(f"Connecting to {args.arm_addr}")
    channel = grpc.insecure_channel(args.arm_addr)
    stub = arm_pb2_grpc.ArmServiceStub(channel)

    ping = stub.Ping(arm_pb2.ArmPingRequest())
    logger.info(f"Server responded: {ping.status} (uptime: {ping.uptime_seconds:.1f}s)")

    try:
        for i, joints in enumerate(waypoints_rad):
            logger.info(f"Waypoint [{i + 1}/{len(waypoints_rad)}] — sending Reset (server interpolates ~3s)")
            response = stub.Reset(arm_pb2.ResetRequest(joint_positions=joints))
            if not response.success:
                logger.error(f"Reset failed at waypoint {i + 1}: {response.error}")
                return
            if i < len(waypoints_rad) - 1 and args.pause_s > 0:
                time.sleep(args.pause_s)
        logger.info("Done — arm at final waypoint.")
    except KeyboardInterrupt:
        logger.warning("Interrupted — arm may be mid-motion.")
        abort_torque_off(stub, args.keep_torque)
    except Exception:
        abort_torque_off(stub, args.keep_torque)
        raise
    finally:
        channel.close()


if __name__ == "__main__":
    main()
