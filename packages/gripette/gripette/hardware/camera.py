"""JPEG snapshot capture from RPi camera.

Simpler than grabette's VideoCapture — no H.264, no recording.
Falls back to a mock (generated placeholder JPEG) when picamera2 is unavailable.
"""

import io
import logging
import threading

logger = logging.getLogger(__name__)

try:
    from picamera2 import Picamera2

    _HAS_PICAMERA2 = True
except ImportError:
    _HAS_PICAMERA2 = False


class CameraCapture:
    """Thread-safe JPEG snapshot capture."""

    def __init__(self, resolution: tuple[int, int] = (1296, 972), quality: int = 70):
        self.resolution = resolution
        self.quality = quality
        self._lock = threading.Lock()
        self._picam2 = None
        self._mock = not _HAS_PICAMERA2

    def start(self) -> None:
        if self._mock:
            logger.warning("picamera2 not available — using mock camera")
            return
        self._picam2 = Picamera2()
        config = self._picam2.create_still_configuration(
            main={"size": self.resolution, "format": "RGB888"},
        )
        self._picam2.configure(config)
        self._picam2.start()
        logger.info("Camera started: %dx%d", *self.resolution)

    def capture_jpeg(self) -> bytes:
        """Capture a single JPEG frame. Thread-safe."""
        if self._mock:
            return _generate_mock_jpeg(self.resolution)
        with self._lock:
            # capture_array is thread-safe with the lock
            array = self._picam2.capture_array("main")
        # Encode to JPEG outside the lock (CPU-bound, no hardware contention)
        return _encode_jpeg(array, self.quality)

    def stop(self) -> None:
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2.close()
            self._picam2 = None
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
