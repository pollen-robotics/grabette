"""Video capture using picamera2 with H.264 encoding.

Ported from grabette-capture/grabette_capture/video.py.
"""

import gc
import logging
import subprocess
from pathlib import Path

from .sync import SyncManager

logger = logging.getLogger(__name__)


class VideoCapture:
    """Captures video from CSI camera using picamera2.

    Default configuration:
        - Resolution: 1296x972 (native OV5647 binned mode)
        - Frame rate: 46 fps (CFR)
        - Codec: H.264 at ~5 Mbps
    """

    DEFAULT_RESOLUTION = (1296, 972)
    DEFAULT_FPS = 46
    DEFAULT_BITRATE = 5_000_000

    def __init__(
        self,
        sync_manager: SyncManager,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        fps: int = DEFAULT_FPS,
        bitrate: int = DEFAULT_BITRATE,
        preview: bool = False,
    ):
        self.sync = sync_manager
        self.resolution = resolution
        self.fps = fps
        self.bitrate = bitrate
        self.preview = preview

        self._picam2 = None
        self._encoder = None
        self._output_path: Path | None = None
        self._h264_path: Path | None = None
        self._frame_timestamps: list[float] = []
        self._recording = False
        self._first_sensor_ts: int | None = None
        self._sync_offset_ms: float = 0.0

    def init_camera(self) -> None:
        """Initialize picamera2 with CFR configuration."""
        from picamera2 import Picamera2, Preview
        from picamera2.encoders import H264Encoder

        self._picam2 = Picamera2()
        frame_duration_us = int(1_000_000 / self.fps)

        if self.preview:
            video_config = self._picam2.create_video_configuration(
                main={"size": self.resolution, "format": "YUV420"},
                lores={"size": (640, 480), "format": "YUV420"},
                display="lores",
                controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
            )
        else:
            video_config = self._picam2.create_video_configuration(
                main={"size": self.resolution, "format": "YUV420"},
                controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
            )
        self._picam2.configure(video_config)
        self._encoder = H264Encoder(bitrate=self.bitrate)

        if self.preview:
            try:
                self._picam2.start_preview(Preview.QTGL)
            except Exception:
                try:
                    self._picam2.start_preview(Preview.DRM)
                except Exception:
                    logger.warning("Could not start preview")

        self._picam2.start()

    def _on_frame(self, request) -> None:
        if self._recording:
            metadata = request.get_metadata()
            sensor_ts_ns = metadata.get("SensorTimestamp")

            if sensor_ts_ns is not None:
                if self._first_sensor_ts is None:
                    self._first_sensor_ts = sensor_ts_ns
                    self._sync_offset_ms = self.sync.get_timestamp_ms()
                ts = (sensor_ts_ns - self._first_sensor_ts) / 1_000_000.0 + self._sync_offset_ms
            else:
                ts = self.sync.get_timestamp_ms()

            self._frame_timestamps.append(ts)

    def start_recording(self, output_path: Path) -> None:
        if self._recording:
            raise RuntimeError("Video capture already running")
        if self._picam2 is None:
            raise RuntimeError("Camera not initialized. Call init_camera() first.")
        if not self.sync.is_started:
            raise RuntimeError("SyncManager must be started before video capture")

        self._output_path = Path(output_path)
        self._h264_path = self._output_path.with_suffix(".h264")
        self._frame_timestamps = []
        self._first_sensor_ts = None
        self._sync_offset_ms = 0.0

        self._picam2.post_callback = self._on_frame
        self._recording = True
        gc.disable()  # Prevent GC pauses from dropping frames during recording
        self._picam2.start_encoder(self._encoder, str(self._h264_path))

    def stop(self) -> list[float]:
        if not self._recording:
            return self._frame_timestamps

        self._recording = False
        self._picam2.stop_encoder()
        gc.enable()
        if self.preview:
            try:
                self._picam2.stop_preview()
            except Exception:
                pass
        self._picam2.stop()
        self._picam2.close()
        self._picam2 = None
        self._encoder = None

        self._mux_to_mp4()
        return self._frame_timestamps

    def _mux_to_mp4(self) -> None:
        if self._h264_path is None or self._output_path is None:
            return
        if not self._h264_path.exists():
            raise RuntimeError(f"H.264 file not found: {self._h264_path}")

        actual_fps = self.fps
        if len(self._frame_timestamps) >= 2:
            duration_ms = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if duration_ms > 0:
                actual_fps = (len(self._frame_timestamps) - 1) / (duration_ms / 1000.0)

        cmd = [
            "ffmpeg", "-y", "-fflags", "+genpts",
            "-r", str(actual_fps), "-i", str(self._h264_path),
            "-c", "copy", "-video_track_timescale", "90000",
            str(self._output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg muxing failed: {result.stderr}")
        self._h264_path.unlink()

    @property
    def frame_count(self) -> int:
        return len(self._frame_timestamps)
