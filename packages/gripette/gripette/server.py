"""gRPC server bootstrap and hardware lifecycle."""

import logging
import signal
from concurrent import futures

import grpc

from .config import settings
from .hardware.camera import CameraCapture
from .hardware.motors import MotorController
from .hardware.sync import SyncManager
from .proto import gripper_pb2_grpc
from .service import GripperServicer

logger = logging.getLogger(__name__)


def serve() -> None:
    """Initialize hardware, start gRPC server, block until shutdown."""
    # Hardware init
    sync = SyncManager()
    camera = CameraCapture(
        resolution=(settings.camera_resolution_w, settings.camera_resolution_h),
        quality=settings.jpeg_quality,
        mode=settings.camera_mode,
        # Ask the sensor for the stream's target rate (video mode only) —
        # otherwise the pipeline's ~30 fps default caps the stream.
        framerate=settings.stream_hz,
    )
    motors = MotorController(
        port=settings.motor_port,
        baudrate=settings.motor_baudrate,
        id_1=settings.motor_id_1,
        id_2=settings.motor_id_2,
        limits=(
            (settings.motor1_min, settings.motor1_max),
            (settings.motor2_min, settings.motor2_max),
        ),
        signs=(settings.motor1_sign, settings.motor2_sign),
        offsets=(settings.motor1_offset, settings.motor2_offset),
    )

    camera.start()
    motors.start()
    sync.start()

    # gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = GripperServicer(camera, motors, sync, stream_hz=settings.stream_hz)
    gripper_pb2_grpc.add_GripperServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{settings.host}:{settings.port}")
    server.start()
    logger.info("gRPC server listening on %s:%d", settings.host, settings.port)

    # Graceful shutdown on SIGTERM/SIGINT
    stop_event = server.wait_for_termination

    def _shutdown(signum, frame):
        logger.info("Received signal %s, shutting down...", signum)
        server.stop(grace=2)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        server.wait_for_termination()
    finally:
        motors.stop()
        camera.stop()
        sync.reset()
        logger.info("Server shut down cleanly")
