"""Real RPi hardware backend using picamera2 + BMI088 IMU."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from grabette.backend.base import Backend
from grabette.models import CaptureStatus, IMUSample, SensorState
from grabette.output import write_imu_json

logger = logging.getLogger(__name__)

FPS = 46
IMU_HZ = 200


class RpiBackend(Backend):
    """Backend using real RPi camera and BMI088 IMU hardware."""

    def __init__(self, enable_angle: bool = False) -> None:
        self._running = False
        self._start_time: float | None = None
        self._capturing = False
        self._capture_session_dir: Path | None = None
        self._capture_start: float | None = None
        self._enable_angle = enable_angle

        self._sync = None
        self._camera = None
        self._imu = None
        self._angle = None

    async def start(self) -> None:
        from grabette.hardware.sync import SyncManager
        from grabette.hardware.camera import VideoCapture
        from grabette.hardware.imu import BMI088Capture

        self._sync = SyncManager()
        self._camera = VideoCapture(self._sync, fps=FPS)
        self._imu = BMI088Capture(self._sync, sample_rate_hz=IMU_HZ)

        logger.info("Initializing camera...")
        self._camera.init_camera()

        logger.info("Initializing IMU...")
        self._imu.init_sensor()

        if self._enable_angle:
            try:
                from grabette.hardware.angle import AngleCapture
                self._angle = AngleCapture(self._sync)
                self._angle.init_sensors()
                logger.info("Angle sensors initialized")
            except Exception:
                logger.warning("Angle sensors not available, continuing without them")
                self._angle = None

        self._running = True
        self._start_time = time.time()
        logger.info("RpiBackend started")

    async def stop(self) -> None:
        if self._capturing:
            await self.stop_capture()
        self._running = False
        self._start_time = None
        logger.info("RpiBackend stopped")

    def get_state(self) -> SensorState:
        imu = None
        if self._capturing:
            # During capture, read latest from capture buffer (no I2C contention)
            if self._imu and self._imu._samples.accel:
                last_accel = self._imu._samples.accel[-1]
                last_gyro = self._imu._samples.gyro[-1] if self._imu._samples.gyro else {"cts": 0, "value": [0, 0, 0]}
                imu = IMUSample(
                    timestamp_ms=last_accel["cts"],
                    accel=tuple(last_accel["value"]),
                    gyro=tuple(last_gyro["value"]),
                )
        elif self._imu and self._imu._bmi088:
            # When idle, read directly from sensor
            try:
                accel = self._imu._bmi088.read_accel()
                gyro = self._imu._bmi088.read_gyro()
                imu = IMUSample(
                    timestamp_ms=time.time() * 1000,
                    accel=accel,
                    gyro=gyro,
                )
            except Exception:
                pass
        return SensorState(imu=imu, capture=self.get_capture_status())

    async def start_capture(self, session_dir: Path) -> None:
        if self._capturing:
            raise RuntimeError("Already capturing")

        self._capture_session_dir = session_dir
        self._capture_start = time.time()

        # Start synchronized capture
        self._sync.start()
        self._imu.start_capture()
        if self._angle:
            self._angle.start_capture()
        self._camera.start_recording(session_dir / "raw_video.mp4")

        self._capturing = True
        logger.info("RpiBackend capture started → %s", session_dir)

    async def stop_capture(self) -> CaptureStatus:
        if not self._capturing:
            raise RuntimeError("Not capturing")

        self._capturing = False

        # Stop in reverse order
        frame_timestamps = self._camera.stop()
        imu_samples = self._imu.stop()
        angle_samples = None
        if self._angle:
            angle_data = self._angle.stop()
            angle_samples = angle_data.samples if angle_data.samples else None

        duration = time.time() - self._capture_start if self._capture_start else 0.0
        status = CaptureStatus(
            is_capturing=False,
            session_id=self._capture_session_dir.name if self._capture_session_dir else None,
            duration_seconds=round(duration, 2),
            frame_count=len(frame_timestamps),
            imu_sample_count=len(imu_samples.accel),
        )

        # Write output files
        if self._capture_session_dir:
            write_imu_json(
                imu_samples.accel,
                imu_samples.gyro,
                FPS,
                self._capture_session_dir / "imu_data.json",
                angle_samples=angle_samples,
            )
            meta = {
                "duration_seconds": status.duration_seconds,
                "frame_count": status.frame_count,
                "imu_sample_count": status.imu_sample_count,
                "fps": FPS,
                "imu_hz": IMU_HZ,
                "backend": "rpi",
            }
            (self._capture_session_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        self._sync.reset()

        # Re-initialize hardware for next capture
        from grabette.hardware.imu import BMI088Capture
        from grabette.hardware.camera import VideoCapture
        self._camera = VideoCapture(self._sync, fps=FPS)
        self._imu = BMI088Capture(self._sync, sample_rate_hz=IMU_HZ)
        self._camera.init_camera()
        self._imu.init_sensor()

        self._capture_session_dir = None
        self._capture_start = None
        logger.info("RpiBackend capture stopped")
        return status

    def get_capture_status(self) -> CaptureStatus:
        duration = 0.0
        if self._capture_start:
            duration = time.time() - self._capture_start

        frame_count = self._camera.frame_count if self._camera else 0
        imu_count = self._imu.sample_count[0] if self._imu else 0

        return CaptureStatus(
            is_capturing=self._capturing,
            session_id=self._capture_session_dir.name if self._capture_session_dir else None,
            duration_seconds=round(duration, 2),
            frame_count=frame_count,
            imu_sample_count=imu_count,
        )

    @property
    def is_capturing(self) -> bool:
        return self._capturing

    def get_frame_jpeg(self) -> bytes | None:
        """Capture a JPEG frame from picamera2."""
        if self._camera and self._camera._picam2:
            try:
                import io
                buf = io.BytesIO()
                self._camera._picam2.capture_file(buf, format="jpeg")
                return buf.getvalue()
            except Exception as e:
                logger.debug("Failed to capture JPEG: %s", e)
        return None
