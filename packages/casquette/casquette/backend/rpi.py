"""Real RPi hardware backend using picamera2 + BMI088 IMU."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from casquette.backend.base import Backend
from casquette.config import settings
from casquette.models import CaptureStatus, IMUSample, SensorState
from casquette.output import write_imu_json

logger = logging.getLogger(__name__)

FPS = 46
IMU_HZ = 200


class RpiBackend(Backend):
    """Backend using real RPi camera and BMI088 IMU hardware."""

    def __init__(self, imu_i2c_bus: int = 1) -> None:
        self._running = False
        self._start_time: float | None = None
        self._capturing = False
        self._capture_session_dir: Path | None = None
        self._imu_i2c_bus = imu_i2c_bus
        self._wall_clock_start: str | None = None

        self._sync = None
        self._camera = None
        self._imu = None

    async def start(self) -> None:
        from casquette.hardware.sync import SyncManager
        from casquette.hardware.camera import VideoCapture
        from casquette.hardware.imu import BMI088Capture

        self._sync = SyncManager()
        self._camera = VideoCapture(
            self._sync, fps=FPS, exposure_us=settings.camera_exposure_us,
        )
        self._imu = BMI088Capture(self._sync, sample_rate_hz=IMU_HZ, i2c_bus=self._imu_i2c_bus)

        logger.info("Initializing camera...")
        self._camera.init_camera()

        # IMU init is best-effort. The HAT-integrated BMI088 may be absent,
        # mis-wired, or returning unexpected chip IDs during early bring-up
        # of a fresh hardware build. Failing here used to kill the whole
        # daemon (and hence the camera stream + REST API). We log + continue
        # with self._imu = None instead — get_state() already handles that
        # path (returns imu=None), and start_capture() will refuse if the
        # caller tries to actually record without an IMU.
        logger.info("Initializing IMU...")
        try:
            self._imu.init_sensor()
        except Exception as e:
            logger.warning(
                "IMU init failed (%s) — continuing in camera-only mode. "
                "Capture endpoints will refuse until the IMU is fixed.",
                e,
            )
            self._imu = None

        self._running = True
        self._start_time = time.time()
        logger.info("RpiBackend started (IMU=%s)", "ok" if self._imu else "DISABLED")

    async def stop(self) -> None:
        if self._capturing:
            await self.stop_capture()
        self._running = False
        self._start_time = None
        logger.info("RpiBackend stopped")

    def get_state(self) -> SensorState:
        imu = None

        if self._capturing:
            if self._imu and self._imu._samples.accel:
                last_accel = self._imu._samples.accel[-1]
                last_gyro = self._imu._samples.gyro[-1] if self._imu._samples.gyro else {"cts": 0, "value": [0, 0, 0]}
                imu = IMUSample(
                    timestamp_ms=last_accel["cts"],
                    accel=tuple(last_accel["value"]),
                    gyro=tuple(last_gyro["value"]),
                )
        else:
            if self._imu and self._imu._bmi088:
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
        if self._imu is None:
            raise RuntimeError(
                "Cannot capture: IMU is not available "
                "(init failed at startup — camera stream still works, "
                "but recording requires the IMU)."
            )

        self._capture_session_dir = session_dir
        self._wall_clock_start = datetime.now(timezone.utc).isoformat()

        # Set flag BEFORE starting streams to prevent I2C contention
        self._capturing = True

        self._sync.start()
        self._imu.start_capture()
        self._camera.start_recording(session_dir / "raw_video.mp4")

        logger.info("RpiBackend capture started → %s", session_dir)

    async def stop_capture(self) -> CaptureStatus:
        if not self._capturing:
            raise RuntimeError("Not capturing")

        duration_ms = self._sync.get_timestamp_ms()

        # Stop IMU before camera (camera stop includes ffmpeg muxing)
        imu_samples = self._imu.stop()
        frame_timestamps = self._camera.stop()

        self._capturing = False

        duration = round(duration_ms / 1000.0, 2)

        # Compute actual video FPS
        actual_fps = float(FPS)
        video_span_ms = 0.0
        if len(frame_timestamps) >= 2:
            video_span_ms = frame_timestamps[-1] - frame_timestamps[0]
            if video_span_ms > 0:
                actual_fps = round((len(frame_timestamps) - 1) / (video_span_ms / 1000.0), 3)

        # Diagnostic: compare video and IMU durations
        imu_span_ms = 0.0
        if len(imu_samples.accel) >= 2:
            imu_span_ms = imu_samples.accel[-1]["cts"] - imu_samples.accel[0]["cts"]
        if video_span_ms > 0 and imu_span_ms > 0:
            drift_pct = abs(video_span_ms - imu_span_ms) / video_span_ms * 100
            logger.info("Sync diagnostic: video=%.1fms (%d frames, %.1f fps), "
                        "IMU=%.1fms (%d samples), drift=%.2f%%",
                        video_span_ms, len(frame_timestamps), actual_fps,
                        imu_span_ms, len(imu_samples.accel), drift_pct)
            if drift_pct > 2.0:
                logger.warning("Video-IMU drift %.2f%% exceeds 2%% threshold", drift_pct)

        status = CaptureStatus(
            is_capturing=False,
            session_id=self._capture_session_dir.name if self._capture_session_dir else None,
            duration_seconds=duration,
            frame_count=len(frame_timestamps),
            imu_sample_count=len(imu_samples.accel),
        )

        # Write output files
        if self._capture_session_dir:
            write_imu_json(
                imu_samples.accel,
                imu_samples.gyro,
                actual_fps,
                self._capture_session_dir / "imu_data.json",
            )
            meta = {
                "duration_seconds": status.duration_seconds,
                "frame_count": status.frame_count,
                "imu_sample_count": status.imu_sample_count,
                "fps": actual_fps,
                "imu_hz": IMU_HZ,
                "backend": "rpi",
                "device_id": settings.device_id,
                "wall_clock_start_utc": self._wall_clock_start,
            }
            (self._capture_session_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        self._sync.reset()

        # Re-initialize hardware for next capture
        from casquette.hardware.imu import BMI088Capture
        from casquette.hardware.camera import VideoCapture
        self._camera = VideoCapture(
            self._sync, fps=FPS, exposure_us=settings.camera_exposure_us,
        )
        self._imu = BMI088Capture(self._sync, sample_rate_hz=IMU_HZ, i2c_bus=self._imu_i2c_bus)
        self._camera.init_camera()
        self._imu.init_sensor()

        self._capture_session_dir = None
        self._wall_clock_start = None
        logger.info("RpiBackend capture stopped")
        return status

    def get_capture_status(self) -> CaptureStatus:
        duration = 0.0
        if self._capturing and self._sync and self._sync.is_started:
            duration = self._sync.get_timestamp_ms() / 1000.0

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
        """Capture a JPEG frame. Returns None during active capture."""
        if self._capturing:
            return None
        if self._camera and self._camera._picam2:
            try:
                import io
                buf = io.BytesIO()
                self._camera._picam2.capture_file(buf, format="jpeg")
                return buf.getvalue()
            except Exception as e:
                logger.debug("Failed to capture JPEG: %s", e)
        return None

    def get_camera_exposure_us(self) -> int:
        if self._camera is None:
            return 0
        return int(getattr(self._camera, "exposure_us", 0))

    def set_camera_exposure_us(self, us: int) -> int:
        if self._camera is None:
            raise RuntimeError("Camera not initialized")
        if self._capturing:
            # Changing exposure during a recording would create a
            # discontinuity in the video — refuse and let the caller
            # stop the capture first.
            raise RuntimeError("Cannot change exposure while capturing")
        return self._camera.set_exposure_us(int(us))
