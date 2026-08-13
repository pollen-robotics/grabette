"""Stream motor positions remotely via gRPC.

Lightweight read-only diagnostic — useful for checking that motors are
responding, sanity-checking the sign convention, or watching live positions
while manually back-driving the gripper.

Usage:
    uv run python scripts/read_motors.py <host:port>
    uv run python scripts/read_motors.py 192.168.1.36
    uv run python scripts/read_motors.py 192.168.1.36 --torque-off   # back-drivable
    uv run python scripts/read_motors.py 192.168.1.36 --once         # single read, exit
    uv run python scripts/read_motors.py 192.168.1.36 --hz 30        # faster poll
"""

import argparse
import math
import sys
import time

from gripette.client import GripperClient
from gripette.config import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target",
                        help=f"Gripette endpoint as HOST or HOST:PORT (port defaults to {settings.port})")
    parser.add_argument("--torque-off", action="store_true",
                        help="Disable torque first so the gripper is back-drivable.")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Polling rate in Hz (default: 10)")
    parser.add_argument("--once", action="store_true",
                        help="Print one reading and exit (for scripting).")
    args = parser.parse_args()

    target = args.target if ":" in args.target else f"{args.target}:{settings.port}"
    dt = 1.0 / args.hz

    with GripperClient(target) as g:
        print(f"Connected to {target}")
        # Limits printed up front as a reference — handy for the inverted-motor
        # diagnostic, where you want to compare observed extremes to the
        # configured min/max.
        print(f"Configured ranges (rad): "
              f"m1=[{settings.motor1_min:+.3f}, {settings.motor1_max:+.3f}], "
              f"m2=[{settings.motor2_min:+.3f}, {settings.motor2_max:+.3f}]")

        if args.torque_off:
            g.torque_off()
            print("Torque off — gripper is back-drivable.")

        if args.once:
            m1, m2 = g.read_motors()
            print(f"m1={m1:+.4f} rad ({math.degrees(m1):+.2f}°)  "
                  f"m2={m2:+.4f} rad ({math.degrees(m2):+.2f}°)")
            return

        print(f"Polling at {args.hz:g} Hz — Ctrl-C to stop.")
        print(f"{'m1 (rad)':>10} {'m1 (°)':>8}   {'m2 (rad)':>10} {'m2 (°)':>8}")

        try:
            next_time = time.monotonic()
            while True:
                m1, m2 = g.read_motors()
                sys.stdout.write(
                    f"\r{m1:+10.4f} {math.degrees(m1):+8.2f}   "
                    f"{m2:+10.4f} {math.degrees(m2):+8.2f}"
                )
                sys.stdout.flush()
                next_time += dt
                sleep_for = next_time - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    # We're behind schedule (network or service slow); resync
                    # so we don't busy-loop catching up.
                    next_time = time.monotonic()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
