"""Real RPi hardware backend using picamera2 + BMI088 IMU."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

from grabette.backend.base import Backend
from grabette.models import AngleSample, CaptureStatus, IMUSample, SensorState
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
            self._init_angle_sensors()

        self._running = True
        self._start_time = time.time()
        logger.info("RpiBackend started")

    def _init_angle_sensors(self) -> None:
        try:
            from grabette.hardware.angle import AngleCapture
            self._angle = AngleCapture(self._sync)
            self._angle.init_sensors()
            logger.info("Angle sensors initialized")
        except Exception:
            logger.warning("Angle sensors not available, continuing without them")
            self._angle = None

    async def stop(self) -> None:
        if self._capturing:
            await self.stop_capture()
        self._running = False
        self._start_time = None
        logger.info("RpiBackend stopped")

    def get_state(self) -> SensorState:
        imu = None
        angle = None

        if self._capturing:
            # During capture, read from capture buffers (no I2C contention)
            if self._imu and self._imu._samples.accel:
                last_accel = self._imu._samples.accel[-1]
                last_gyro = self._imu._samples.gyro[-1] if self._imu._samples.gyro else {"cts": 0, "value": [0, 0, 0]}
                imu = IMUSample(
                    timestamp_ms=last_accel["cts"],
                    accel=tuple(last_accel["value"]),
                    gyro=tuple(last_gyro["value"]),
                )
            if self._angle and self._angle._samples.samples:
                last = self._angle._samples.samples[-1]
                angle = AngleSample(
                    timestamp_ms=last["cts"],
                    proximal=last["value"][1],
                    distal=last["value"][0],
                )
        else:
            # When idle, read directly from sensors
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
            if self._angle and self._angle._i2c_1 and self._angle._i2c_2:
                try:
                    raw1 = self._angle._read_angle_raw(self._angle._i2c_1)
                    raw2 = self._angle._read_angle_raw(self._angle._i2c_2)
                    cal1 = self._angle._normalize_angle(raw1 - self._angle._offset_1_deg)
                    cal2 = self._angle._normalize_angle(raw2 - self._angle._offset_2_deg)
                    angle = AngleSample(
                        timestamp_ms=time.time() * 1000,
                        proximal=math.radians(cal2),
                        distal=math.radians(cal1),
                    )
                except Exception:
                    pass

        return SensorState(imu=imu, angle=angle, capture=self.get_capture_status())

    async def start_capture(self, session_dir: Path) -> None:
        if self._capturing:
            raise RuntimeError("Already capturing")

        self._capture_session_dir = session_dir

        # Set flag BEFORE starting streams so the daemon poll loop
        # (get_state) reads from capture buffers instead of doing
        # direct I2C reads that would contend with capture threads.
        self._capturing = True

        # Start synchronized capture — all streams share the same
        # SyncManager t=0 reference (time.monotonic based).
        self._sync.start()
        self._imu.start_capture()
        if self._angle:
            self._angle.start_capture()
        self._camera.start_recording(session_dir / "raw_video.mp4")

        logger.info("RpiBackend capture started → %s", session_dir)

    async def stop_capture(self) -> CaptureStatus:
        if not self._capturing:
            raise RuntimeError("Not capturing")

        # Keep _capturing = True until ALL streams have stopped.
        # This prevents the daemon poll loop (get_state) from doing
        # direct I2C reads while capture threads are still running.

        # Grab sync-clock duration before stopping streams (monotonic,
        # same clock used by all stream timestamps — no wall-clock drift).
        duration_ms = self._sync.get_timestamp_ms()

        # Stop IMU/angle BEFORE camera.  camera.stop() runs ffmpeg muxing
        # which takes ~1-2s — if IMU is still running during muxing, it
        # accumulates extra samples that extend the IMU duration well past
        # the video duration (causes 7-9% IMU-video clock drift).
        imu_samples = self._imu.stop()
        angle_samples = None
        angle_count = 0
        if self._angle:
            angle_data = self._angle.stop()
            angle_count = len(angle_data.samples)
            angle_samples = angle_data.samples if angle_data.samples else None
        frame_timestamps = self._camera.stop()

        # NOW safe to clear flag — all streams stopped, no I2C contention.
        self._capturing = False

        duration = round(duration_ms / 1000.0, 2)

        # Compute actual video FPS from frame timestamps
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
            frame_count=self._camera.frame_count,
            imu_sample_count=len(imu_samples.accel),
            angle_sample_count=angle_count,
        )

        # Write output files
        if self._capture_session_dir:
            write_imu_json(
                imu_samples.accel,
                imu_samples.gyro,
                actual_fps,
                self._capture_session_dir / "imu_data.json",
                angle_samples=angle_samples,
            )
            meta = {
                "duration_seconds": status.duration_seconds,
                "frame_count": status.frame_count,
                "imu_sample_count": status.imu_sample_count,
                "angle_sample_count": status.angle_sample_count,
                "fps": actual_fps,
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
        if self._enable_angle:
            self._init_angle_sensors()

        self._capture_session_dir = None
        logger.info("RpiBackend capture stopped")
        return status

    def get_capture_status(self) -> CaptureStatus:
        duration = 0.0
        if self._capturing and self._sync and self._sync.is_started:
            duration = self._sync.get_timestamp_ms() / 1000.0

        frame_count = self._camera.frame_count if self._camera else 0
        imu_count = self._imu.sample_count[0] if self._imu else 0
        angle_count = self._angle.sample_count if self._angle else 0

        return CaptureStatus(
            is_capturing=self._capturing,
            session_id=self._capture_session_dir.name if self._capture_session_dir else None,
            duration_seconds=round(duration, 2),
            frame_count=frame_count,
            imu_sample_count=imu_count,
            angle_sample_count=angle_count,
        )

    @property
    def is_capturing(self) -> bool:
        return self._capturing

    def get_frame_jpeg(self) -> bytes | None:
        """Capture a JPEG frame from picamera2.

        Returns None during active capture to avoid competing with the
        H.264 encoder for camera resources (preserves frame timing and
        IMU synchronization).
        """
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
