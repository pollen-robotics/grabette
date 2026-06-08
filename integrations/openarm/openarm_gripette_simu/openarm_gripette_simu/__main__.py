"""Launch the simulation gRPC server.

Usage:
    uv run python -m openarm_gripette_simu
    uv run python -m openarm_gripette_simu --scene scenes/table_red_cube.xml
    uv run python -m openarm_gripette_simu --headless
    uv run python -m openarm_gripette_simu --initial-joints -1.0 0 0 2.44 0 0 0
"""

import argparse
import logging

from .server import SimulationServer, GRIPPER_PORT, ARM_PORT


def main():
    parser = argparse.ArgumentParser(description="OpenArm Gripette simulation server")
    parser.add_argument("--scene", type=str, default=None, help="Path to scene XML file")
    parser.add_argument("--gripper-port", type=int, default=GRIPPER_PORT)
    parser.add_argument("--arm-port", type=int, default=ARM_PORT)
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument(
        "--initial-joints", type=float, nargs=7, default=None,
        metavar="RAD",
        help="Initial arm joint positions (7 values in rad)",
    )
    parser.add_argument(
        "--gripper-hold-open-duration", type=float, default=0.0,
        help="Seconds after a reset (keyboard or RPC) during which incoming "
             "gripper commands are ignored and the gripper is forced open. "
             "Originally needed because the v3 model couldn't predict "
             "closed→open transitions; the v4 (release+hover) model can, so "
             "the default is now 0 (no override). Bump to 1.5 if you ever "
             "train a model again that gets stuck closed after a reset.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    server = SimulationServer(
        scene_xml=args.scene,
        initial_arm_joints=args.initial_joints,
        gripper_hold_open_duration=args.gripper_hold_open_duration,
    )
    server.run(
        gripper_port=args.gripper_port,
        arm_port=args.arm_port,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
