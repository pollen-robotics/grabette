"""Display the live camera feed from the Gripette's GripperService.

Uses matplotlib for display (interactive mode) so it works with LeRobot's
`opencv-python-headless` — no need to install `opencv-python` separately.
cv2 is still used to decode the JPEG bytes; only the GUI path is matplotlib.

Usage:
  uv run python examples/view_camera.py \\
      --gripper_addr localhost:50051

Press 'q' in the figure window or Ctrl+C in the terminal to quit.
"""

import argparse
import os
import time

import cv2  # must be imported before matplotlib so we can clear its Qt plugin hijack

# cv2 (opencv-python-headless wheel) ships its own incomplete Qt plugin dir and
# sets QT_QPA_PLATFORM_PLUGIN_PATH on import. That directory is missing the
# 'xcb' platform plugin on Linux, which makes PyQt5 fail with
# "Could not load the Qt platform plugin xcb". Clearing the env var lets PyQt5
# fall back to its own bundled Qt plugins, which include xcb.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import grpc  # noqa: E402
import matplotlib  # noqa: E402

# Force an interactive backend. uv-managed Python defaults to the non-GUI
# 'Agg' backend when matplotlib is imported as a library. QtAgg (requires
# PyQt5) is more portable than TkAgg on uv's managed Python, which ships
# a version of Tk that often mismatches matplotlib's _tkagg expectations.
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from openarm_gripette_simu.proto import gripper_pb2, gripper_pb2_grpc  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="View the Gripette camera feed")
    p.add_argument(
        "--gripper_addr", type=str, default="localhost:50051", help="GripperService gRPC address"
    )
    p.add_argument(
        "--show_gripper_state",
        action="store_true",
        help="Overlay motor1/motor2 positions on the frame",
    )
    return p.parse_args()


def main():
    args = parse_args()
    channel = grpc.insecure_channel(args.gripper_addr)
    stub = gripper_pb2_grpc.GripperServiceStub(channel)

    ping = stub.Ping(gripper_pb2.PingRequest())
    print(f"Connected to {args.gripper_addr} (server uptime: {ping.uptime_seconds:.1f}s)")
    print("Close the figure window (or Ctrl+C) to quit.\n")

    # Interactive matplotlib window. First frame sets up the axes; subsequent
    # frames just swap the image array — much faster than redrawing.
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_axis_off()
    image_handle = None
    fps_text = None

    # 'q' in the window quits, matching cv2 convention.
    stop_flag = {"stop": False}

    def on_key(event):
        if event.key in ("q", "escape"):
            stop_flag["stop"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda _: stop_flag.update(stop=True))

    last_tick = time.perf_counter()
    fps_ema = 0.0
    try:
        for frame in stub.StreamState(gripper_pb2.StreamRequest()):
            if stop_flag["stop"]:
                break
            img_bgr = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            now = time.perf_counter()
            dt = now - last_tick
            last_tick = now
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if dt > 0 else fps_ema

            if image_handle is None:
                image_handle = ax.imshow(img_rgb)
                h, w = img_rgb.shape[:2]
                fps_text = ax.text(
                    10,
                    25,
                    "",
                    color="lime",
                    fontsize=10,
                    fontfamily="monospace",
                    bbox={"facecolor": "black", "alpha": 0.5, "pad": 2},
                )
                ax.set_title(f"Gripette camera ({w}x{h})")
            else:
                image_handle.set_data(img_rgb)

            overlay = f"{fps_ema:5.1f} fps"
            if args.show_gripper_state:
                m = frame.motor_state
                overlay += f"  m1={m.motor1_position:+.3f}  m2={m.motor2_position:+.3f}"
            fps_text.set_text(overlay)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("\nStopped.")
    except grpc.RpcError as e:
        print(f"gRPC error: {e}")
    finally:
        plt.close(fig)
        channel.close()


if __name__ == "__main__":
    main()
