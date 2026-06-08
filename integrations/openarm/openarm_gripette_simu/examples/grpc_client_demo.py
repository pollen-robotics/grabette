"""gRPC client demo: move the arm in a square, cycle the gripper, display camera.

Usage:
    # First, start the server in another terminal:
    uv run python -m openarm_gripette_simu --scene scenes/table_red_cube.xml \
        --initial-joints 0 0 0 1.57 0 0 0

    # Then run this client:
    uv run python examples/grpc_client_demo.py
"""

import time
import threading
import numpy as np
import cv2
import grpc

from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc
from openarm_gripette_simu.proto import gripper_pb2, gripper_pb2_grpc

ARM_ADDR = "localhost:50052"
GRIPPER_ADDR = "localhost:50051"


def camera_display_thread(grip_stub):
    """Stream and display the Gripette camera in a loop (runs in background)."""
    try:
        for frame in grip_stub.StreamState(gripper_pb2.StreamRequest()):
            img = cv2.imdecode(
                np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR
            )
            cv2.imshow("Gripette camera", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except grpc.RpcError:
        pass


def main():
    arm_stub = arm_pb2_grpc.ArmServiceStub(grpc.insecure_channel(ARM_ADDR))
    grip_stub = gripper_pb2_grpc.GripperServiceStub(grpc.insecure_channel(GRIPPER_ADDR))

    # Check connection
    arm_stub.Ping(arm_pb2.ArmPingRequest())
    grip_stub.Ping(gripper_pb2.PingRequest())
    print("Connected to simulation server.")

    # Start camera display in background
    cam_thread = threading.Thread(target=camera_display_thread, args=(grip_stub,), daemon=True)
    cam_thread.start()

    # Read initial state
    state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
    print(f"Initial EE position: [{state.x:.3f}, {state.y:.3f}, {state.z:.3f}]")

    # --- Square trajectory via delta commands ---
    # 5cm half-side in YZ plane, well above the table (z>0.48 at all corners)
    half = 0.05
    n_steps = 50
    square = [
        (0, half, 0),      # +Y
        (0, 0, -half),     # -Z
        (0, -half, 0),     # -Y
        (0, 0, half),      # +Z (back to start)
    ]

    print("\nMoving arm in a square (press 'q' in camera window to quit)...")
    for cycle in range(3):
        for dx, dy, dz in square:
            step_dx = dx / n_steps
            step_dy = dy / n_steps
            step_dz = dz / n_steps
            for _ in range(n_steps):
                arm_stub.SendCartesianDelta(arm_pb2.CartesianDelta(
                    dx=step_dx, dy=step_dy, dz=step_dz, dr6d=[0.0] * 6,
                ))
                time.sleep(0.02)

        state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
        print(f"  cycle {cycle + 1}: EE at [{state.x:.3f}, {state.y:.3f}, {state.z:.3f}]")

    # --- Gripper open/close ---
    print("\nCycling gripper...")
    for cycle in range(3):
        # Close
        for t in np.linspace(0, 1, 30):
            grip_stub.SendMotorCommand(gripper_pb2.MotorCommand(
                motor1_goal=-1.48 * t,
                motor2_goal=-2.02 * t,
            ))
            time.sleep(0.02)

        # Open
        for t in np.linspace(1, 0, 30):
            grip_stub.SendMotorCommand(gripper_pb2.MotorCommand(
                motor1_goal=-1.48 * t,
                motor2_goal=-2.02 * t,
            ))
            time.sleep(0.02)

        motors = grip_stub.ReadMotors(gripper_pb2.ReadMotorsRequest())
        print(f"  cycle {cycle + 1}: motors=({motors.motor1_position:.2f}, {motors.motor2_position:.2f})")

    # --- Hold and display ---
    print("\nDone. Camera still streaming, press 'q' to quit.")
    cam_thread.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
