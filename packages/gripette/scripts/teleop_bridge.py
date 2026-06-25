"""Teleoperation bridge: read angles from grabette, send to gripette.

Reads proximal/distal angles from grabette's REST API and forwards them
as motor commands to the gripper. Run with --dry-run first to check
sign conventions without moving motors.

Usage:
    uv run python scripts/teleop_bridge.py --grabette HOST[:PORT] --gripper HOST:PORT [--dry-run]
    uv run python scripts/teleop_bridge.py --grabette 192.168.1.35 --gripper 192.168.1.36:50051

--grabette: HOST or HOST:PORT of the grabette REST API (default port 8000).
--gripper:  HOST:PORT of the gripette gRPC server.
"""

import argparse
import json
import math
import time
import urllib.request

from gripette.client import GripperClient
from gripette.config import settings

LOOP_HZ = 20  # bridge rate


def read_grabette_angles(url: str) -> tuple[float, float]:
    """Read proximal and distal angles (radians) from grabette REST API."""
    with urllib.request.urlopen(url, timeout=1) as resp:
        data = json.loads(resp.read())
    angle = data["angle"]
    return (angle["proximal"], angle["distal"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grabette", required=True,
                        help="Grabette REST host as HOST or HOST:PORT (port defaults to 8000)")
    parser.add_argument("--gripper", required=True,
                        help=f"Gripette gRPC endpoint as HOST or HOST:PORT (port defaults to {settings.port})")
    parser.add_argument("--dry-run", action="store_true", help="Print only, don't move motors")
    args = parser.parse_args()

    grabette_host = args.grabette if ":" in args.grabette else f"{args.grabette}:8000"
    grabette_url = f"http://{grabette_host}/api/state"
    gripper_target = args.gripper if ":" in args.gripper else f"{args.gripper}:{settings.port}"
    dt = 1.0 / LOOP_HZ

    with GripperClient(gripper_target) as g:
        print(f"Gripper connected: {g.ping()}")

        if not args.dry_run:
            g.torque_on()
            print("Torque on")

        print(f"Bridge running at {LOOP_HZ}Hz {'(DRY RUN)' if args.dry_run else ''}")
        print("Press Ctrl+C to stop\n")
        print(f"{'proximal':>10} {'distal':>10} {'→ m1':>10} {'→ m2':>10}")

        next_time = time.monotonic()
        try:
            while True:
                try:
                    proximal, distal = read_grabette_angles(grabette_url)
                except Exception:
                    # Network hiccup — skip this iteration
                    next_time += dt
                    sleep_dur = next_time - time.monotonic()
                    if sleep_dur > 0:
                        time.sleep(sleep_dur)
                    continue

                # Motor mapping — adjust signs here after testing
                # Start with direct mapping, same sign
                m1_goal = proximal
                m2_goal = distal

                print(f"{proximal:10.3f} {distal:10.3f} {m1_goal:10.3f} {m2_goal:10.3f}", end="\r")

                if not args.dry_run:
                    try:
                        g.move(m1_goal, m2_goal)
                    except RuntimeError as e:
                        # Limit violation — print but don't crash
                        print(f"\nLimit: {e}")

                next_time += dt
                sleep_dur = next_time - time.monotonic()
                if sleep_dur > 0:
                    time.sleep(sleep_dur)

        except KeyboardInterrupt:
            print("\nStopped")
        finally:
            if not args.dry_run:
                g.torque_off()
                print("Torque off")


if __name__ == "__main__":
    main()
