"""Video capture using picamera2 with H.264 encoding.

Ported from grabette-capture/grabette_capture/video.py.
"""

import gc
import heapq
import logging
import queue
import subprocess
import threading
from pathlib import Path

from grabette.output import write_json_atomic

from .frame_timing import FrameTiming
from .sync import SyncManager

logger = logging.getLogger(__name__)

# Encoded pictures waiting to be written: ~6 s at 50 fps, ~4 MB of H.264
# (a 2 s SD stall, 102 pictures, was seen on 2026-11-04). See _h264_file_output.
OUTPUT_QUEUE_FRAMES = 300

# Pictures the writer holds back to put them in PTS order. The Pi's hardware
# encoder occasionally hands two consecutive pictures back swapped (2 in 6934
# on 2026-11-09): written as-is, the decoder silently drops the late one and
# the video loses sync. Its pictures are encoded in capture order (no
# B-frames), so PTS order is exactly the encoded stream order. 3 pictures
# (60 ms) covers a picture up to 3 places late; later than that, the take is
# invalid as before.
REORDER_WINDOW = 3


def _slice_type(frame) -> int | None:
    """H.264 slice_type (0 P, 1 B, 2 I, ...) of the first picture slice in an
    Annex B access unit, or None if there is none. Only called for the rare
    pictures that arrive out of order."""
    data = bytes(frame)
    pos = 0
    while (pos := data.find(b"\x00\x00\x01", pos)) != -1:
        pos += 3
        if pos < len(data) and data[pos] & 0x1F in (1, 5):  # non-IDR / IDR slice
            rbsp = data[pos + 1:pos + 17].replace(b"\x00\x00\x03", b"\x00\x00")
            bits = "".join(f"{b:08b}" for b in rbsp)
            values, i = [], 0
            for _ in range(2):  # ue(v) first_mb_in_slice, then ue(v) slice_type
                zeros = 0
                while i < len(bits) and bits[i] == "0":
                    zeros += 1
                    i += 1
                if i + zeros >= len(bits):
                    return None
                values.append(int(bits[i:i + zeros + 1], 2) - 1)
                i += zeros + 1
            return values[1] % 5
    return None


def _h264_file_output(path: Path, timing: FrameTiming):
    """An H.264 file output that reports each written picture to `timing`.

    It derives from picamera2's public `Output` base (start/stop/outputframe)
    rather than from `FileOutput`, so it does not depend on FileOutput's
    internals (`outputtimestamp`, `dead`, `recording`, `_write`), which vary
    between picamera2 versions. It writes the file itself, on a thread of its
    own: picamera2's encoder thread both writes the output and releases the
    camera buffers, so an SD-card write stall used to starve libcamera
    (6 buffers, ~120 ms) and drop frames.
    """
    from picamera2.outputs import Output

    class H264FileOutput(Output):
        def __init__(self):
            super().__init__()
            self.queue_peak = 0
            self.reordered_count = 0  # pictures that arrived after a later one
            self._file = open(path, "wb")
            self._have_keyframe = False
            self._held = []  # (pts, arrival, item), a min-heap: see REORDER_WINDOW
            self._arrivals = 0
            self._max_pts = None
            self._last_written_pts = None
            self._queue = queue.Queue(maxsize=OUTPUT_QUEUE_FRAMES)
            self._writer = threading.Thread(
                target=self._write_queued, name="h264-writer", daemon=True)
            self._writer.start()

        # Older picamera2 versions call this with (frame, keyframe, timestamp)
        # only; newer ones add packet and audio. Accept both.
        def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
            if audio:
                return  # No audio is configured; never write it into the H.264.
            # Encoder thread: hand over and return at once. A full queue
            # blocks, which drops at the camera as before rather than
            # losing an encoded picture mid-stream.
            self._queue.put((frame, keyframe, timestamp))
            self.queue_peak = max(self.queue_peak, self._queue.qsize())

        def _write_queued(self):
            # Nothing may escape this loop: a dead writer would fill the queue,
            # block the encoder and stall the camera.
            while (item := self._queue.get()) is not None:
                try:
                    self._reorder(item)
                except Exception as exc:
                    timing.record_error(f"Video output reordering failed: {exc}")
                finally:
                    self._queue.task_done()
            self._flush_held()

        def _reorder(self, item):
            pts = item[2]
            if pts is None:
                # Can't be placed: write it where it came; timing reports it.
                self._flush_held()
                self._write_safely(item)
                return
            if self._max_pts is not None and pts < self._max_pts:
                self.reordered_count += 1
                logger.warning("H.264 encoder output out of order: PTS %d after %d, "
                               "put back in order", pts, self._max_pts)
                if _slice_type(item[0]) == 1:
                    # B-frames are legitimately out of PTS order: sorting them
                    # would break the stream. The Pi's encoder makes none.
                    timing.record_error(f"B-frame in H.264 output at PTS {pts}")
            else:
                self._max_pts = pts
            heapq.heappush(self._held, (pts, self._arrivals, item))
            self._arrivals += 1
            if len(self._held) > REORDER_WINDOW:
                self._write_safely(heapq.heappop(self._held)[2])

        def _flush_held(self):
            while self._held:
                self._write_safely(heapq.heappop(self._held)[2])

        def _write_safely(self, item):
            pts = item[2]
            if pts is not None and self._last_written_pts is not None and pts < self._last_written_pts:
                timing.record_error(f"Encoder output PTS {pts} arrived more than "
                                    f"{REORDER_WINDOW} pictures late")
            if pts is not None:
                self._last_written_pts = pts
            try:
                self._write(*item)
            except Exception as exc:
                # Surface writer-thread I/O failures in stop(), after the
                # encoder has released its buffers and other streams stop.
                timing.record_error(f"Video output write failed: {exc}")

        def _write(self, frame, keyframe, timestamp):
            # A stream must start on a keyframe: earlier pictures can't be decoded.
            if not self._have_keyframe:
                if not keyframe:
                    return
                self._have_keyframe = True
            self._file.write(frame)
            self._file.flush()
            # Only now is the picture in the file: give it its capture time.
            timing.written(timestamp)

        def _finish_writes(self):
            if self._writer.is_alive():
                self._queue.put(None)
                self._writer.join()

        def stop(self):
            # The encoder calls this after draining its last pictures.
            self._finish_writes()
            super().stop()
            self.close()

        def close(self):
            self._finish_writes()
            if not self._file.closed:
                try:
                    self._file.close()
                except OSError as exc:
                    timing.record_error(f"Video output close failed: {exc}")

    return H264FileOutput()


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
    # picamera2's default is 6, with one more held back for capture_* (queue=True).
    # A short stall of the thread that returns them then drops frames at libcamera.
    DEFAULT_BUFFER_COUNT = 10

    def __init__(
        self,
        sync_manager: SyncManager,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        fps: int = DEFAULT_FPS,
        bitrate: int = DEFAULT_BITRATE,
        preview: bool = False,
        buffer_count: int = DEFAULT_BUFFER_COUNT,
    ):
        self.sync = sync_manager
        self.resolution = resolution
        self.fps = fps
        self.bitrate = bitrate
        self.preview = preview
        self.buffer_count = buffer_count

        self._picam2 = None
        self._encoder = None
        self._output_path: Path | None = None
        self._h264_path: Path | None = None
        self._frame_timestamps: list[float] = []
        self._recording = False
        self._frame_count: int = 0
        self._timing = FrameTiming(self.sync)
        self.timestamp_alignment: dict = {}
        self._output = None

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
                buffer_count=self.buffer_count,
                queue=False,
            )
        else:
            video_config = self._picam2.create_video_configuration(
                main={"size": self.resolution, "format": "YUV420"},
                controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
                buffer_count=self.buffer_count,
                queue=False,
            )
        self._picam2.configure(video_config)
        logger.info("Camera configured with %d buffers", self.buffer_count)
        # Capture times come from the public post_callback (see FrameTiming).
        # Without it nothing could be timestamped: refuse the camera now
        # rather than invalidate every take.
        if not hasattr(self._picam2, "post_callback"):
            raise RuntimeError("Incompatible picamera2: Picamera2.post_callback is missing")
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
        self._timing = FrameTiming(self.sync)
        self._frame_timestamps = self._timing.timestamps
        self.timestamp_alignment = {}
        self._frame_count = 0

        output = _h264_file_output(self._h264_path, self._timing)
        self._output = output

        self._recording = True
        gc.disable()  # Prevent GC pauses from dropping frames during recording
        # Set before the encoder starts, so it sees every frame the encoder gets.
        self._picam2.post_callback = self._on_camera_frame
        try:
            self._picam2.start_encoder(self._encoder, output)
        except Exception:
            self._picam2.post_callback = None
            self._recording = False
            gc.enable()
            output.close()
            raise

    def _on_camera_frame(self, request) -> None:
        # picamera2's request thread, before the encoder: never raise here.
        try:
            sensor_ns = request.get_metadata().get("SensorTimestamp")
        except Exception as exc:
            self._timing.record_error(f"Cannot read camera frame metadata: {exc}")
            return
        self._timing.camera_frame(sensor_ns)

    def stop(self) -> list[float]:
        if not self._recording:
            return self._frame_timestamps

        self._recording = False
        try:
            self._picam2.stop_encoder()
        finally:
            self._picam2.post_callback = None
            gc.enable()
        self._timing.finish()
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
        # How close the SD card came to stalling the camera (OUTPUT_QUEUE_FRAMES = full).
        self.timestamp_alignment["writer_queue_peak"] = getattr(self._output, "queue_peak", None)
        # Pictures the encoder handed back out of order, put back in PTS order.
        self.timestamp_alignment["reordered_output_count"] = getattr(self._output, "reordered_count", None)
        self.timestamp_alignment["camera_buffer_count"] = self.buffer_count
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
                    "-show_entries", "stream=nb_read_packets",
                    "-of", "csv=p=0",
                    str(self._output_path),
                ],
                capture_output=True, text=True, timeout=30,
            )
            result.check_returncode()
            val = result.stdout.strip()
            return int(val) if val.isdigit() else 0
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
