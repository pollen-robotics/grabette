"""GripperServicer — implements the gRPC RPCs defined in gripper.proto."""

import logging
import os
import time

import grpc

from .config import settings
from .hardware.camera import CameraCapture
from .hardware.motors import MotorController
from .hardware.sync import SyncManager
from .proto import gripper_pb2, gripper_pb2_grpc

logger = logging.getLogger(__name__)

# A wedged camera stack does not recover in-process; a service restart does
# (field-validated). After this many consecutive capture failures, exit so
# systemd (Restart=on-failure) brings the service back on its own.
MAX_CONSECUTIVE_CAMERA_FAILURES = 3


class GripperServicer(gripper_pb2_grpc.GripperServiceServicer):
    """Implements StreamState, SendMotorCommand, and Ping RPCs."""

    def __init__(
        self,
        camera: CameraCapture,
        motors: MotorController,
        sync: SyncManager,
        stream_hz: float = 10.0,
    ):
        self._camera = camera
        self._motors = motors
        self._sync = sync
        self._stream_interval = 1.0 / stream_hz
        self._camera_failures = 0
        # Last torque-limit fractions applied, to dedup the per-command write.
        self._last_torque_limit: tuple[float, float] | None = None
        # Apply the ceiling up front so the guarantee holds from boot, not from
        # the first command: between SetTorque(enable) and the first goal the
        # servo would otherwise hold at whatever limit it had stored.
        ceiling = float(settings.torque_ceiling)
        self._motors.set_torque_limit([ceiling, ceiling])
        self._last_torque_limit = (ceiling, ceiling)
        logger.info("Torque ceiling %.2f applied at startup", ceiling)

    def StreamState(self, request, context):
        """Server-streaming: yields GripperFrame at the configured rate."""
        logger.info("StreamState: client connected")
        sequence = 0
        next_time = time.monotonic()

        while context.is_active():
            # Capture JPEG then read motor positions in tight sequence
            try:
                jpeg_data = self._camera.capture_jpeg()
                self._camera_failures = 0
            except Exception as e:
                # A dead camera must kill the stream LOUDLY — the client must
                # never keep acting on a frozen or absent image. (Also frees
                # this worker thread: a blocked capture used to pin it, and
                # with max_workers=4 a few reconnects starved the server.)
                self._camera_failures += 1
                logger.exception(
                    "StreamState: camera failure %d/%d — aborting stream",
                    self._camera_failures, MAX_CONSECUTIVE_CAMERA_FAILURES,
                )
                if self._camera_failures >= MAX_CONSECUTIVE_CAMERA_FAILURES:
                    # os._exit, not sys.exit: sys.exit from a gRPC worker only
                    # kills the thread, and the graceful shutdown path can
                    # itself hang on the wedged pipeline. Hard nonzero exit →
                    # systemd restarts. Gripper motors hold their last goal
                    # through the ~10 s gap (firmware-side torque).
                    logger.critical(
                        "Camera wedged (%d consecutive failures) — exiting for systemd restart",
                        self._camera_failures,
                    )
                    os._exit(1)
                context.abort(grpc.StatusCode.INTERNAL, f"camera failure: {e}")
            pos1, pos2 = self._motors.read_positions()
            timestamp_ms = self._sync.get_timestamp_ms()

            load1, load2 = self._motors.get_present_load()
            frame = gripper_pb2.GripperFrame(
                jpeg_data=jpeg_data,
                motor_state=gripper_pb2.MotorState(
                    motor1_position=pos1,
                    motor2_position=pos2,
                    motor1_load=load1,
                    motor2_load=load2,
                ),
                timestamp_ms=timestamp_ms,
                sequence=sequence,
            )
            yield frame
            sequence += 1

            # Accumulator-pattern sleep to avoid drift
            next_time += self._stream_interval
            sleep_duration = next_time - time.monotonic()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

        logger.info("StreamState: client disconnected after %d frames", sequence)

    def SendMotorCommand(self, request, context):
        """Unary: send goal positions to motors (and optional torque limit)."""
        try:
            self._apply_torque_limit(request)
            self._motors.write_goal_positions(request.motor1_goal, request.motor2_goal)
            return gripper_pb2.MotorCommandResponse(success=True)
        except Exception as e:
            logger.exception("Motor command failed")
            return gripper_pb2.MotorCommandResponse(success=False, error=str(e))

    def _apply_torque_limit(self, request):
        """Apply the per-command torque limit, capped by settings.torque_ceiling.

        A value of 0 means "unset" (proto default). Unset now resolves to the
        CEILING rather than being left alone: the position limits are the real
        collision angles, so a full close is meant to drive into the stop and be
        halted by torque. Leaving an unset limit untouched meant full torque on a
        fresh boot, which would grind into that collision — the protection
        depended on every client remembering to pass a limit.

        Requests above the ceiling are clamped, not rejected: a caller asking for
        more effort than allowed should still get a working grasp.

        Deduped so a constant per-command value isn't re-deposited at stream rate.
        """
        ceiling = float(settings.torque_ceiling)
        req = (request.motor1_torque_limit, request.motor2_torque_limit)
        tl = tuple(ceiling if v <= 0.0 else min(v, ceiling) for v in req)
        if tl != self._last_torque_limit:
            self._motors.set_torque_limit([tl[0], tl[1]])
            self._last_torque_limit = tl

    def ReadMotors(self, request, context):
        """Unary: read motor positions + load (lightweight, no camera)."""
        pos1, pos2 = self._motors.read_positions()
        load1, load2 = self._motors.get_present_load()
        return gripper_pb2.MotorState(
            motor1_position=pos1,
            motor2_position=pos2,
            motor1_load=load1,
            motor2_load=load2,
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
