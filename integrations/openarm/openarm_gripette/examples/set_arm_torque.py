"""Enable/disable motor torque on the whole arm via the ArmService.

DANGER: --off makes the motors freewheel — the arm FALLS under gravity.
Hold the arm or make sure it can drop safely before disabling.

After --on the arm is enabled but holds nothing (limp) until the next
command; use reset_arm.py to bring it home smoothly from wherever it hangs.

In simulation SetTorque is a no-op (returns success, does nothing).

Usage:
  uv run python examples/set_arm_torque.py --arm_addr <robot-ip>:50052 --off
  uv run python examples/set_arm_torque.py --arm_addr <robot-ip>:50052 --on
"""

import argparse
import logging

import grpc
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Enable/disable arm motor torque")
    p.add_argument("--arm_addr", type=str, default="localhost:50052", help="ArmService gRPC address")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--on", action="store_true", help="Enable torque (arm stays limp until next command)")
    group.add_argument("--off", action="store_true", help="Disable torque — THE ARM FALLS")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    channel = grpc.insecure_channel(args.arm_addr)
    stub = arm_pb2_grpc.ArmServiceStub(channel)
    try:
        if args.off:
            logger.warning("Disabling torque — the arm will fall under gravity!")
        response = stub.SetTorque(arm_pb2.SetTorqueRequest(enable=args.on), timeout=5.0)
        if response.success:
            logger.info(f"Torque {'ENABLED' if args.on else 'DISABLED'}.")
        else:
            logger.error(f"SetTorque failed: {response.error}")
            raise SystemExit(1)
    finally:
        channel.close()


if __name__ == "__main__":
    main()
