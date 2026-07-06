"""Read and print the current arm state (joints + EE pose) from the gRPC server.

Works with the simulator OR the real-robot server.

Usage:
  # One-shot read (default)
  uv run python examples/openarm_gripette/read_arm_state.py \\
      --arm_addr localhost:50052

  # Continuous monitoring at 10 Hz
  uv run python examples/openarm_gripette/read_arm_state.py \\
      --arm_addr localhost:50052 --watch --fps 10
"""

import argparse
import math
import time

import grpc
from openarm_gripette_simu.kinematics import ARM_JOINT_NAMES
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc


def parse_args():
    p = argparse.ArgumentParser(description="Read and print OpenArm state via gRPC")
    p.add_argument("--arm_addr", type=str, default="localhost:50052", help="ArmService gRPC address")
    p.add_argument("--watch", action="store_true", help="Continuously print state (default: one-shot)")
    p.add_argument("--fps", type=float, default=5.0, help="Refresh rate in watch mode")
    return p.parse_args()


def format_state(state) -> str:
    """Format an ArmState response as a human-readable multi-line string."""
    lines = []
    lines.append(f"  EE position (m):  x={state.x:+.4f}  y={state.y:+.4f}  z={state.z:+.4f}")
    lines.append(f"  EE rotation (6D): {[round(v, 4) for v in state.r6d]}")
    lines.append("  Joint positions:")
    for i, name in enumerate(ARM_JOINT_NAMES):
        rad = state.joint_positions[i]
        deg = math.degrees(rad)
        lines.append(f"    [{i}] {name:16s} {rad:+8.4f} rad   ({deg:+7.2f}°)")
    return "\n".join(lines)


def main():
    args = parse_args()

    channel = grpc.insecure_channel(args.arm_addr)
    stub = arm_pb2_grpc.ArmServiceStub(channel)

    ping = stub.Ping(arm_pb2.ArmPingRequest())
    print(f"Connected to {args.arm_addr} (server uptime: {ping.uptime_seconds:.1f}s)\n")

    try:
        if not args.watch:
            state = stub.GetArmState(arm_pb2.GetArmStateRequest())
            print(format_state(state))
        else:
            dt = 1.0 / args.fps
            print(f"Watching at {args.fps} Hz. Press Ctrl+C to stop.\n")
            while True:
                state = stub.GetArmState(arm_pb2.GetArmStateRequest())
                # Move cursor up to overwrite the previous print (10 lines of output)
                print("\033[F" * 10, end="")
                print(format_state(state))
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        channel.close()


if __name__ == "__main__":
    main()
