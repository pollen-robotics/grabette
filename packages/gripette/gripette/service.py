"""GripperServicer — implements the gRPC RPCs defined in gripper.proto."""

import logging
import time

from .hardware.camera import CameraCapture
from .hardware.motors import MotorController
from .hardware.sync import SyncManager
from .proto import gripper_pb2, gripper_pb2_grpc

logger = logging.getLogger(__name__)

# Target streaming rate
STREAM_HZ = 10
STREAM_INTERVAL = 1.0 / STREAM_HZ


class GripperServicer(gripper_pb2_grpc.GripperServiceServicer):
    """Implements StreamState, SendMotorCommand, and Ping RPCs."""

    def __init__(
        self,
        camera: CameraCapture,
        motors: MotorController,
        sync: SyncManager,
    ):
        self._camera = camera
        self._motors = motors
        self._sync = sync

    def StreamState(self, request, context):
        """Server-streaming: yields GripperFrame at ~10Hz."""
        logger.info("StreamState: client connected")
        sequence = 0
        next_time = time.monotonic()

        while context.is_active():
            # Capture JPEG then read motor positions in tight sequence
            jpeg_data = self._camera.capture_jpeg()
            pos1, pos2 = self._motors.read_positions()
            timestamp_ms = self._sync.get_timestamp_ms()

            frame = gripper_pb2.GripperFrame(
                jpeg_data=jpeg_data,
                motor_state=gripper_pb2.MotorState(
                    motor1_position=pos1,
                    motor2_position=pos2,
                ),
                timestamp_ms=timestamp_ms,
                sequence=sequence,
            )
            yield frame
            sequence += 1

            # Accumulator-pattern sleep to avoid drift
            next_time += STREAM_INTERVAL
            sleep_duration = next_time - time.monotonic()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

        logger.info("StreamState: client disconnected after %d frames", sequence)

    def SendMotorCommand(self, request, context):
        """Unary: send goal positions to motors."""
        try:
            self._motors.write_goal_positions(request.motor1_goal, request.motor2_goal)
            return gripper_pb2.MotorCommandResponse(success=True)
        except Exception as e:
            logger.exception("Motor command failed")
            return gripper_pb2.MotorCommandResponse(success=False, error=str(e))

    def ReadMotors(self, request, context):
        """Unary: read motor positions (lightweight, no camera)."""
        pos1, pos2 = self._motors.read_positions()
        return gripper_pb2.MotorState(
            motor1_position=pos1,
            motor2_position=pos2,
        )

    def SetTorque(self, request, context):
        """Unary: enable/disable motor torque."""
        try:
            self._motors.set_torque(request.enable)
            return gripper_pb2.TorqueResponse(success=True)
        except Exception as e:
            logger.exception("Torque command failed")
            return gripper_pb2.TorqueResponse(success=False, error=str(e))

    def Ping(self, request, context):
        """Unary: health check."""
        uptime = self._sync.get_timestamp_ms() / 1000.0
        return gripper_pb2.PingResponse(status="ok", uptime_seconds=uptime)
