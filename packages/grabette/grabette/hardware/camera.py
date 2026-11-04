"""Video capture using picamera2 with H.264 encoding.

Ported from grabette-capture/grabette_capture/video.py.
"""

import gc
import json
import logging
import subprocess
from pathlib import Path

from grabette.output import write_json_atomic

from .frame_timing import FrameTiming
from .sync import SyncManager

logger = logging.getLogger(__name__)


class FrameAlignmentError(RuntimeError):
    """The saved video cannot safely be paired with its capture timestamps."""

    def __init__(self, message: str, *, episode_id: str | None = None):
        super().__init__(message)
        # Set by the backend only after all streams and capture state are down.
        # Stop consumers can then retain this exact failed take in the registry.
        self.episode_id = episode_id


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
        self._frame_count: int = 0
        self._timing = FrameTiming(self.sync)
        self.timestamp_alignment: dict = {}
        self._has_b_frames = None

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
        owner = self

        class SensorTimestampEncoder(H264Encoder):
            def _timestamp(self, request):
                try:
                    pts_us = super()._timestamp(request)
                    owner._timing.submitted(pts_us, request.get_metadata()["SensorTimestamp"])
                    return pts_us
                except Exception as exc:
                    # An exception in an encoder worker must not make a saved,
                    # matching prefix look like a complete successful capture.
                    owner._timing.record_error(f"Cannot timestamp encoder input: {exc}")
                    raise

        self._encoder = SensorTimestampEncoder(bitrate=self.bitrate)

        if self.preview:
            try:
                self._picam2.start_preview(Preview.QTGL)
            except Exception:
                try:
                    self._picam2.start_preview(Preview.DRM)
                except Exception:
                    logger.warning("Could not start preview")

        self._picam2.start()

    def close(self) -> None:
        """Release a camera opened (or half-opened, when init_camera failed)
        outside a recording; stop() does it after one."""
        picam2, self._picam2, self._encoder = self._picam2, None, None
        if picam2 is not None:
            try:
                picam2.close()
            except Exception as e:
                logger.debug("Closing picamera2: %s", e)

    def start_recording(self, output_path: Path) -> None:
        if self._recording:
            raise RuntimeError("Video capture already running")
        if self._picam2 is None:
            raise RuntimeError("Camera not initialized. Call init_camera() first.")
        if not self.sync.is_started:
            raise RuntimeError("SyncManager must be started before video capture")

        self._output_path = Path(output_path)
        self._h264_path = self._output_path.with_suffix(".h264")
        from picamera2.outputs import FileOutput

        self._timing = FrameTiming(self.sync)
        self._frame_timestamps = self._timing.timestamps
        self.timestamp_alignment = {}
        self._frame_count = 0
        self._has_b_frames = None
        timing = self._timing

        class TimestampedFileOutput(FileOutput):
            def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
                try:
                    super().outputframe(frame, keyframe, timestamp, packet, audio)
                except Exception as exc:
                    # Surface worker-thread I/O failures in stop(), after the
                    # encoder has released its buffers and other streams stop.
                    timing.record_error(f"Video output write failed: {exc}")
                if self.dead:
                    timing.record_error("Video output reported a failed write")

            def outputtimestamp(self, timestamp):
                # FileOutput calls this after writing/flushing a video picture,
                # including pictures drained during stop_encoder().
                timing.written(timestamp)

        output = TimestampedFileOutput(str(self._h264_path))

        self._recording = True
        gc.disable()  # Prevent GC pauses from dropping frames during recording
        try:
            self._picam2.start_encoder(self._encoder, output)
        except Exception:
            self._recording = False
            gc.enable()
            output.close()
            raise

    def stop(self) -> list[float]:
        if not self._recording:
            return self._frame_timestamps

        self._recording = False
        try:
            self._picam2.stop_encoder()
        finally:
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
        self.timestamp_alignment = self._timing.result(self._frame_count)
        if self._has_b_frames != 0:
            self.timestamp_alignment["valid"] = False
            self.timestamp_alignment["errors"].append(
                f"Cannot verify video presentation order: has_b_frames={self._has_b_frames}")
        try:
            write_json_atomic(self._output_path.with_suffix(".frames.json"), {
                "alignment": self.timestamp_alignment, "frames": self._timing.frames,
            })
        except OSError as exc:
            self.timestamp_alignment["valid"] = False
            self.timestamp_alignment["errors"].append(f"Cannot save frame provenance: {exc}")
        if not self.timestamp_alignment["valid"]:
            raise FrameAlignmentError(str(self.timestamp_alignment["errors"]))
        # Only discard the intermediate stream after successful validation.
        self._h264_path.unlink()
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
        self._frame_count = self._count_frames_ffprobe()

    def _count_frames_ffprobe(self) -> int:
        if self._output_path is None or not self._output_path.exists():
            return 0
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-count_packets",
                    "-show_entries", "stream=nb_read_packets,has_b_frames",
                    "-of", "json",
                    str(self._output_path),
                ],
                capture_output=True, text=True, timeout=30,
            )
            result.check_returncode()
            stream = json.loads(result.stdout)["streams"][0]
            self._has_b_frames = stream.get("has_b_frames")
            return int(stream["nb_read_packets"])
        except Exception:
            return 0

    @property
    def frame_timestamps(self) -> list[float]:
        """Finalized timestamps; available for diagnostics if validation fails."""
        return list(self._frame_timestamps)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_open(self) -> bool:
        """True while the picamera2 device is initialized (not yet closed)."""
        return self._picam2 is not None
