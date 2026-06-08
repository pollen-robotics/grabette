"""GripperClient — synchronous Python client for remote gripper control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import grpc

from .proto import gripper_pb2, gripper_pb2_grpc


@dataclass
class Frame:
    """A single gripper frame with camera image and motor state."""

    jpeg_data: bytes
    motor1: float
    motor2: float
    timestamp_ms: float
    sequence: int


class GripperClient:
    """Synchronous gRPC client for the gripper service.

    Usage:
        with GripperClient("192.168.1.X:50051") as g:
            print(g.ping())
            g.move(1.0, -0.5)
            for frame in g.stream():
                print(frame.sequence, len(frame.jpeg_data))
    """

    def __init__(self, target: str = "localhost:50051"):
        self._target = target
        self._channel: grpc.Channel | None = None
        self._stub: gripper_pb2_grpc.GripperServiceStub | None = None

    def __enter__(self) -> GripperClient:
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    def connect(self) -> None:
        self._channel = grpc.insecure_channel(self._target)
        self._stub = gripper_pb2_grpc.GripperServiceStub(self._channel)

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def ping(self) -> dict:
        """Health check. Returns {"status": str, "uptime_seconds": float}."""
        resp = self._stub.Ping(gripper_pb2.PingRequest())
        return {"status": resp.status, "uptime_seconds": resp.uptime_seconds}

    def torque_on(self) -> bool:
        """Enable motor torque."""
        resp = self._stub.SetTorque(gripper_pb2.TorqueCommand(enable=True))
        if not resp.success:
            raise RuntimeError(f"Torque command failed: {resp.error}")
        return True

    def torque_off(self) -> bool:
        """Disable motor torque (motors can be moved by hand)."""
        resp = self._stub.SetTorque(gripper_pb2.TorqueCommand(enable=False))
        if not resp.success:
            raise RuntimeError(f"Torque command failed: {resp.error}")
        return True

    def read_motors(self) -> tuple[float, float]:
        """Read motor positions (radians). Lightweight, no camera."""
        resp = self._stub.ReadMotors(gripper_pb2.ReadMotorsRequest())
        return (resp.motor1_position, resp.motor2_position)

    def move(self, motor1: float, motor2: float) -> bool:
        """Send goal positions (radians) to both motors. Returns success."""
        resp = self._stub.SendMotorCommand(
            gripper_pb2.MotorCommand(motor1_goal=motor1, motor2_goal=motor2)
        )
        if not resp.success:
            raise RuntimeError(f"Motor command failed: {resp.error}")
        return True

    def stream(self) -> Iterator[Frame]:
        """Yield Frame objects from the 10Hz camera+motor stream."""
        for msg in self._stub.StreamState(gripper_pb2.StreamRequest()):
            yield Frame(
                jpeg_data=msg.jpeg_data,
                motor1=msg.motor_state.motor1_position,
                motor2=msg.motor_state.motor2_position,
                timestamp_ms=msg.timestamp_ms,
                sequence=msg.sequence,
            )
