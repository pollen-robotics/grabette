"""JPEG snapshot capture from RPi camera.

Simpler than grabette's VideoCapture — no H.264, no recording.

Fails LOUDLY. On the robot, a broken camera must never look like a working
one (the policy would act on garbage), so:
  - no automatic mock fallback — the mock is explicit opt-in via
    GRIPPER_MOCK_CAMERA=1, for dev machines without the hardware;
  - every capture has a timeout: a wedged libcamera stack (field-observed
    at boot: start() succeeds, sensor never delivers) raises CameraError
    instead of blocking forever.
"""

import io
import logging
import threading

logger = logging.getLogger(__name__)

# A healthy pipeline delivers the first frame in well under a second (~2 s
# worst case in still mode with AE settling); 5 s of nothing means the sensor
# is not delivering at all.
CAPTURE_TIMEOUT_S = 5.0


class CameraError(RuntimeError):
    """Camera produced no frame — stack wedged, or hardware/driver absent."""


class CameraCapture:
    """Thread-safe JPEG snapshot capture."""

    def __init__(
        self,
        resolution: tuple[int, int] = (1296, 972),
        quality: int = 70,
        mode: str = "still",
        framerate: float | None = None,
        mock: bool = False,
    ):
        self.resolution = resolution
        self.quality = quality
        self.mode = mode
        self.framerate = framerate
        self._lock = threading.Lock()
        self._picam2 = None
        self._mock = mock

    def start(self) -> None:
        if self._mock:
            logger.warning(
                "MOCK camera explicitly enabled (GRIPPER_MOCK_CAMERA) — "
                "serving generated placeholder frames"
            )
            return
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise CameraError(
                "picamera2 not importable — broken camera stack? "
                "(for dev without camera hardware, set GRIPPER_MOCK_CAMERA=1)"
            ) from e
        self._picam2 = Picamera2()
        if self.mode == "video":
            # Continuous pipeline on a binned sensor mode: full FOV on the
            # RPi cameras, much faster per-frame capture than still mode
            # (which re-reads the full-res sensor every frame). The video
            # pipeline defaults to ~30 fps; capture_array blocks until the
            # next sensor frame, so the SENSOR rate caps the stream rate —
            # request it explicitly (OV5647 binned mode supports up to ~42).
            controls = {}
            if self.framerate:
                dur_us = int(1_000_000 / self.framerate)
                controls["FrameDurationLimits"] = (dur_us, dur_us)
            config = self._picam2.create_video_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls=controls,
            )
        else:
            config = self._picam2.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"},
            )
        self._picam2.configure(config)
        self._picam2.start()
        logger.info("Camera started: %dx%d (%s mode)", *self.resolution, self.mode)

    def capture_jpeg(self) -> bytes:
        """Capture a single JPEG frame. Thread-safe.

        Raises CameraError if the sensor delivers nothing within
        CAPTURE_TIMEOUT_S — never blocks forever, never fakes a frame.
        """
        if self._mock:
            return _generate_mock_jpeg(self.resolution)
        with self._lock:
            # capture_array is thread-safe with the lock. Plain (wait=True)
            # capture_array blocks FOREVER when the stack is wedged; the
            # job + bounded wait turns that into a loud error.
            # (Picamera2.wait(job, timeout=...) requires picamera2 >= 0.3.10.)
            job = self._picam2.capture_array("main", wait=False)
            try:
                array = self._picam2.wait(job, timeout=CAPTURE_TIMEOUT_S)
            except TimeoutError as e:
                raise CameraError(
                    f"no frame from sensor within {CAPTURE_TIMEOUT_S:.0f}s — "
                    "camera stack wedged; restart the gripette service"
                ) from e
        # Encode to JPEG outside the lock (CPU-bound, no hardware contention)
        return _encode_jpeg(array, self.quality)

    def stop(self) -> None:
        if self._picam2 is None:
            return
        picam2, self._picam2 = self._picam2, None
        # On a wedged pipeline stop()/close() can block forever, stalling
        # shutdown until systemd's SIGKILL (observed as ~90 s service stops).
        # Run them bounded and best-effort — process exit releases the device
        # regardless.
        t = threading.Thread(target=lambda: (picam2.stop(), picam2.close()), daemon=True)
        t.start()
        t.join(timeout=5.0)
        if t.is_alive():
            logger.error("Camera stop() wedged — abandoning cleanup, process exit will release it")
        else:
            logger.info("Camera stopped")


def _encode_jpeg(array, quality: int) -> bytes:
    """Encode a numpy RGB array to JPEG bytes."""
    # Use simplejpeg if available (faster, installed with picamera2), else PIL
    try:
        import simplejpeg
        # picamera2 RGB888 format is actually BGR from the ISP
        return simplejpeg.encode_jpeg(array, quality=quality, colorspace="BGR")
    except ImportError:
        from PIL import Image
        # PIL expects RGB, so swap channels
        img = Image.fromarray(array[:, :, ::-1])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def _generate_mock_jpeg(resolution: tuple[int, int]) -> bytes:
    """Generate a small placeholder JPEG for local dev without a camera."""
    # Minimal valid JPEG — a tiny gray image
    from PIL import Image
    img = Image.new("RGB", resolution, color=(64, 64, 64))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    return buf.getvalue()
