"""Teleoperation bridge: read angles from grabette, send to gripette.

Reads proximal/distal angles from grabette's REST API and forwards them
as motor commands to the gripper. Run with --dry-run first to check
sign conventions without moving motors.

Usage:
    uv run python scripts/teleop_bridge.py --dry-run      # just print, no motor commands
    uv run python scripts/teleop_bridge.py                 # actually move motors

Requires grabette running on 192.168.1.35:8000 and gripette on 192.168.1.36:50051.
"""

import argparse
import json
import math
import time
import urllib.request

from gripette.client import GripperClient

GRABETTE_URL = "http://192.168.1.35:8000/api/state"
GRIPPER_TARGET = "192.168.1.36:50051"
LOOP_HZ = 20  # bridge rate


def read_grabette_angles() -> tuple[float, float]:
    """Read proximal and distal angles (radians) from grabette REST API."""
    with urllib.request.urlopen(GRABETTE_URL, timeout=1) as resp:
        data = json.loads(resp.read())
    angle = data["angle"]
    return (angle["proximal"], angle["distal"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print only, don't move motors")
    args = parser.parse_args()

    dt = 1.0 / LOOP_HZ

    with GripperClient(GRIPPER_TARGET) as g:
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
                    proximal, distal = read_grabette_angles()
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
