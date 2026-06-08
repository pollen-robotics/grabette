"""Test camera stream: measure framerate and save a sample frame.

Usage:
    uv run python scripts/camera_test.py [host:port]
"""

import sys
import time

from gripette.client import GripperClient

TARGET = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.36:50051"
NUM_FRAMES = 50


def main():
    with GripperClient(TARGET) as g:
        print(f"Connected to {TARGET}")
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
