"""Move both motors to position 0 (fully open) via gRPC.

Usage:
    uv run python scripts/goto_zero.py <host:port>
    uv run python scripts/goto_zero.py 192.168.1.36:50051

For a local version that talks to /dev/serial0 directly (no gRPC), use
scripts/goto_zero_local.py.
"""

import argparse
import time

from gripette.client import GripperClient
from gripette.config import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target",
                        help=f"Gripette endpoint as HOST or HOST:PORT (port defaults to {settings.port})")
    args = parser.parse_args()

    target = args.target if ":" in args.target else f"{args.target}:{settings.port}"

    with GripperClient(target) as g:
        print(f"Connected to {target}")
        g.torque_on()

        g.move(0.0, 0.0)
        time.sleep(1.0)
        fb1, fb2 = g.read_motors()
        print(f'Positions: {fb1} {fb2}')
        g.torque_off()
        print("Torque off")


if __name__ == "__main__":
    main()
