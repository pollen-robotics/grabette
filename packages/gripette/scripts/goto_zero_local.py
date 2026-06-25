"""Local goto-zero — talks directly to hardware, no gRPC.

Sends both motors to position 0 (the upper end of the gripette's range,
i.e. the fully-open position) and prints the feedback. Run on the Pi when
you want to reset the gripper without bringing up the gRPC stack.

If the systemd service is running it will hold /dev/serial0; stop it first:
    sudo systemctl stop gripette

Usage:
    uv run python scripts/goto_zero_local.py
    uv run python scripts/goto_zero_local.py --hold        # leave torque on
    uv run python scripts/goto_zero_local.py --settle 2.5  # custom wait time
"""

import argparse
import time

from gripette.config import settings
from gripette.hardware.motors import MotorController

DEFAULT_SETTLE = 1.5  # seconds — a full-range move at default servo speed takes <1s


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hold", action="store_true",
                        help="Leave torque enabled after reaching zero.")
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE,
                        help=f"Seconds to wait for motion to complete (default: {DEFAULT_SETTLE})")
    args = parser.parse_args()

    motors = MotorController(
        port=settings.motor_port,
        baudrate=settings.motor_baudrate,
        id_1=settings.motor_id_1,
        id_2=settings.motor_id_2,
        limits=(
            (settings.motor1_min, settings.motor1_max),
            (settings.motor2_min, settings.motor2_max),
        ),
    )

    print(f"Starting motors on {settings.motor_port}...")
    motors.start()

    pos = motors.read_positions()
    print(f"Current: ({pos[0]:+.3f}, {pos[1]:+.3f}) rad")

    motors.set_torque(True)
    print(f"Torque on — going to (0, 0), waiting {args.settle:.1f}s...")
    motors.write_goal_positions(0.0, 0.0)
    time.sleep(args.settle)

    pos = motors.read_positions()
    print(f"Final:   ({pos[0]:+.3f}, {pos[1]:+.3f}) rad")

    if args.hold:
        print("Torque held on (--hold). Use scripts/motor_test_local.py or your own script to drive further.")
    else:
        motors.set_torque(False)
        print("Torque off")

    motors.stop()


if __name__ == "__main__":
    main()
