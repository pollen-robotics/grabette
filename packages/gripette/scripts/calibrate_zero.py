"""Remote zero-calibration over gRPC. Prints env-file lines for the Pi.

Reads the robot-frame position at the user-defined zero pose and prints the
offset DELTA the user should add to /etc/gripette/env. Service stays up —
this is a no-stop diagnostic / iterative-tuning tool.

It prints env-file lines rather than writing them: the gripper's filesystem
isn't accessible from here. For a one-shot calibration that writes the env
file directly, run scripts/calibrate_zero_local.py ON the Pi.

Math (per motor):
    robot_at_zero = (encoder_at_zero - offset_old) * sign
    encoder_at_zero = robot_at_zero * sign + offset_old
    offset_new = encoder_at_zero
    delta = offset_new - offset_old = robot_at_zero * sign

So the printed delta works whether the gripette is freshly installed
(offset_old = 0, new = delta) or being re-calibrated (just add to existing).

Usage:
    uv run python scripts/calibrate_zero.py <target> --hand {left,right}
    uv run python scripts/calibrate_zero.py 192.168.1.36 --hand right
"""

import argparse
import sys
import time

from gripette.client import GripperClient
from gripette.config import settings

SAMPLES = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target",
                        help=f"Gripette endpoint as HOST or HOST:PORT (port defaults to {settings.port})")
    parser.add_argument("--hand", choices=["left", "right"], required=True,
                        help="Hand this gripette is configured as. Determines the sign "
                             "mapping for the delta calculation. Must match the GRIPPER_HAND "
                             "value on the Pi.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the 'press enter when ready' prompt.")
    args = parser.parse_args()

    target = args.target if ":" in args.target else f"{args.target}:{settings.port}"
    sign1, sign2 = (+1, +1) if args.hand == "right" else (-1, -1)

    with GripperClient(target) as g:
        print(f"Connected to {target}  (hand={args.hand}, signs=({sign1:+d}, {sign2:+d}))")
        try:
            g.torque_off()
            print("Torque OFF — gripper is back-drivable.\n")
        except RuntimeError as e:
            print(f"ERROR: torque_off failed: {e}", file=sys.stderr)
            sys.exit(1)

        print("Move the gripper to the TRUE zero position (fully open).")
        if not args.yes:
            try:
                input("Press ENTER when ready (Ctrl-C to cancel): ")
            except (KeyboardInterrupt, EOFError):
                print("\nCanceled.")
                sys.exit(1)

        print(f"\nReading positions (averaging {SAMPLES} samples)...")
        readings = []
        for _ in range(SAMPLES):
            readings.append(g.read_motors())
            time.sleep(0.05)
        m1 = sum(r[0] for r in readings) / len(readings)
        m2 = sum(r[1] for r in readings) / len(readings)

    print(f"  m1 robot-frame reading: {m1:+.4f} rad")
    print(f"  m2 robot-frame reading: {m2:+.4f} rad\n")

    delta1 = m1 * sign1
    delta2 = m2 * sign2

    print(f"  Δ offset m1 = {m1:+.4f} * {sign1:+d} = {delta1:+.6f}")
    print(f"  Δ offset m2 = {m2:+.4f} * {sign2:+d} = {delta2:+.6f}\n")

    print("Update /etc/gripette/env on the Pi by ADDING these deltas to the")
    print("existing GRIPPER_MOTOR*_OFFSET values. If the file has no offset")
    print("lines yet, set them to the deltas directly:")
    print()
    print(f"  GRIPPER_MOTOR1_OFFSET=<existing> + {delta1:+.6f}")
    print(f"  GRIPPER_MOTOR2_OFFSET=<existing> + {delta2:+.6f}")
    print()
    print("Then on the Pi:  sudo systemctl restart gripette")


if __name__ == "__main__":
    main()
