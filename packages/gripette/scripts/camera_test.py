"""Test camera stream: measure framerate and save a sample frame.

Usage:
    uv run python scripts/camera_test.py <host:port>
    uv run python scripts/camera_test.py 192.168.1.36:50051
"""

import argparse
import time

from gripette.client import GripperClient
from gripette.config import settings

NUM_FRAMES = 50


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target",
                        help=f"Gripette endpoint as HOST or HOST:PORT (port defaults to {settings.port})")
    args = parser.parse_args()

    target = args.target if ":" in args.target else f"{args.target}:{settings.port}"

    with GripperClient(target) as g:
        print(f"Connected to {target}")
        print(f"Streaming {NUM_FRAMES} frames...")

        sizes = []
        t0 = time.monotonic()

        for i, frame in enumerate(g.stream()):
            sizes.append(len(frame.jpeg_data))

            # Save first frame for color check
            if i == 0:
                with open("camera_test.jpg", "wb") as f:
                    f.write(frame.jpeg_data)
                print(f"Saved camera_test.jpg ({len(frame.jpeg_data)} bytes)")

            if i + 1 >= NUM_FRAMES:
                break

        elapsed = time.monotonic() - t0
        fps = NUM_FRAMES / elapsed
        avg_size = sum(sizes) / len(sizes)

        print(f"\nResults:")
        print(f"  Frames: {NUM_FRAMES}")
        print(f"  Elapsed: {elapsed:.2f}s")
        print(f"  FPS: {fps:.1f}")
        print(f"  Avg JPEG size: {avg_size / 1024:.0f} KB")
        print(f"  Bandwidth: {avg_size * fps / 1024:.0f} KB/s")


if __name__ == "__main__":
    main()
