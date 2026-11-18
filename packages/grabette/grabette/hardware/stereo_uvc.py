"""UVC side-by-side stereo capture (SKL-4689 / Waveshare AR0144), no on-board depth.

Satisfies hardware.depth_camera.DepthCameraCapture, so RpiBackend drives it like
the OAK-D and the Gemini 305. Structure mirrors orbbec.py: an always-on reader
thread started by init_device(), caching a preview frame and writing to disk
only while `_recording` is set.

Differences from the Gemini 305, all measured on a Pi 4 (scripts/stereo_bench.py):

1. **No depth on the device.** The camera sends one MJPG frame holding both eyes
   side by side (hardware-synced, since they share a buffer). The raw MJPG bytes
   are written untouched, with no decode or encode on the Pi: SKL at 3040x1080
   holds 30.0 fps at ~13 MB/s for 7% of one core. Rectification and SGBM depth
   run offline from dcam_calib_stereo.json, which then produces the usual
   dcam_left.mp4 / dcam_depth.mkv.

2. **Plain file, not an ffmpeg pipe.** Piping into ffmpeg stalled the first
   write ~1.2 s while ffmpeg started, dropping ~35 frames. Concatenated JPEGs
   are already a stream ffmpeg and OpenCV read directly (`-f mjpeg`).

3. **Kernel timestamps.** CAP_PROP_POS_MSEC is the V4L2 buffer timestamp on
   CLOCK_MONOTONIC (the SyncManager clock), ~34 ms before read() returns, so
   frames are dated at arrival rather than after Python scheduling.

4. **No IMU** on these cameras: get_latest_imu() is None forever.

Output layout:
    dcam_stereo.mjpeg              raw MJPG, side by side, as sent by the camera
    dcam_stereo_timestamps.json    per-frame {seq, device_us, host_ms}
    dcam_calib_stereo.json         stereokit calibration (K, D, R, T, rectification)
    dcam_calib.json                profile, mode, exposure, video node
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .sync import SyncManager

logger = logging.getLogger(__name__)

_CALIB_DIR = Path(__file__).parent / "stereo_calib"

# From stereokit/profiles.py. `card_match` is a substring of the V4L2 card name,
# matched before any mode is requested: both cameras accept several of the same
# frame sizes, so "first node that accepts the size" could bind the wrong one.
# The SKL sends its halves as [right|left].
PROFILES = {
    "skl2mp220": {
        "card_match": "SPCA2100",
        "swap_eyes": True,
        "modes": {"3040x1080": (3040, 1080, 30)},
    },
    "ar0144": {
        "card_match": "CCB Camera",
        "swap_eyes": False,
        "modes": {"2560x720": (2560, 720, 30)},
    },
}

# Preview is decoded at 1/4 scale (libjpeg DCT scaling, cheap) and only one
# frame in six, i.e. ~5 fps. The recording path never decodes.
_PREVIEW_EVERY = 6

# A recording with a longer stretch than this between frames (or any failed
# read) gets a warning at stop. 0.2 s = 6 frames at 30 fps.
_GAP_WARN_S = 0.2

# Frames read after init before wait_until_ready() reports ready. Exposure is
# fixed, so there is no auto-exposure to converge; this only skips the first
# frames after stream-on.
_READY_FRAMES = 10


def find_video_node(card_match: str) -> tuple[int, str] | None:
    """(index, card name) of the first /dev/videoN whose card name matches.

    A UVC camera exposes a capture node and a metadata node with the same card
    name; the capture node enumerates first.
    """
    nodes = sorted(Path("/sys/class/video4linux").glob("video*"),
                   key=lambda p: int(p.name.removeprefix("video")))
    for p in nodes:
        try:
            name = (p / "name").read_text().strip()
        except OSError:
            continue
        if card_match in name:
            return int(p.name.removeprefix("video")), name
    return None


def split_eyes(frame: np.ndarray, swap: bool) -> tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side frame into (left, right)."""
    mid = frame.shape[1] // 2
    a, b = frame[:, :mid], frame[:, mid:]
    return (b, a) if swap else (a, b)


class StereoUvcCapture:
    """Records raw side-by-side MJPG from a UVC stereo camera."""

    def __init__(
        self,
        sync_manager: SyncManager,
        profile: str = "skl2mp220",
        mode: str = "3040x1080",
        exposure: int = 300,
        gamma: int = 0,
    ) -> None:
        self.sync = sync_manager
        self.profile_name = profile
        self.profile = PROFILES[profile]
        self.mode = mode
        self.width, self.height, self.fps = self.profile["modes"][mode]
        # UVC exposure_time_absolute, in 100 us units (300 = 30 ms). Must stay
        # under the frame period: auto-exposure on the SKL at 3040x1080 picks
        # ~60 ms and throttles 30 fps to 16.6.
        self.exposure = exposure
        # UVC gamma, 100-300; 0 = leave the camera's value. With no gain
        # control on these cameras, this is how a shorter exposure is
        # brightened, on the camera and before JPEG compression.
        self.gamma = gamma

        self._cap: cv2.VideoCapture | None = None
        self._node: tuple[int, str] | None = None
        self._calibration: dict | None = None

        self._output_dir: Path | None = None
        self._file = None
        self._files_lock = threading.Lock()
        self._recording = False
        self._initialized = False
        self._timestamps: list[dict] = []
        self._frames_read = 0
        # Recording health, reset per recording. A stall used to be silent:
        # one episode lost its last 2.4 s with nothing in the log.
        self._read_failures = 0
        self._last_frame_mono = 0.0
        self._max_gap_s = 0.0

        self._latest_preview: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ init

    def init_device(self) -> None:
        """Load calibration, open the camera, start the reader thread."""
        # Without calibration the episode cannot be turned into depth, so fail
        # here (RpiBackend latches a hardware error) rather than record it.
        calib_path = _CALIB_DIR / f"{self.profile_name}_{self.mode}.json"
        self._calibration = json.loads(calib_path.read_text())

        self._node = find_video_node(self.profile["card_match"])
        if self._node is None:
            raise RuntimeError(
                f"No {self.profile_name} camera (card name containing "
                f"{self.profile['card_match']!r}) on /dev/video*")
        index, name = self._node

        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{index} ({name})")
        # FOURCC before size, or V4L2 ignores it. YUYV at this size is 1 fps.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Hand back the raw MJPG buffer instead of a decoded image.
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual
        cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        if self.gamma:
            cap.set(cv2.CAP_PROP_GAMMA, self.gamma)
        got =(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if got != (self.width, self.height):
            cap.release()
            raise RuntimeError(
                f"/dev/video{index} negotiated {got[0]}x{got[1]}, "
                f"wanted {self.width}x{self.height}")
        self._cap = cap

        logger.info("Stereo camera ready: %s on /dev/video%d, %s@%d, exposure %d, gamma %s",
                    name, index, self.mode, self.fps, self.exposure,
                    self.gamma or "camera default")

        self._frames_read = 0
        self._initialized = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------ recording on/off

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """Block until a few frames have arrived, or until timeout."""
        if not self._initialized:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._frames_read >= _READY_FRAMES:
                return True
            time.sleep(0.02)
        logger.warning("Stereo camera not ready after %.1fs — starting capture anyway",
                       timeout)
        return False

    def start_recording(self, output_dir: Path) -> None:
        if not self._initialized:
            raise RuntimeError("StereoUvcCapture not initialized. Call init_device() first.")
        if self._recording:
            raise RuntimeError("StereoUvcCapture already recording")
        if not self.sync.is_started:
            raise RuntimeError("SyncManager must be started before StereoUvcCapture")

        self._output_dir = Path(output_dir).absolute()
        self._output_dir.mkdir(parents=True, exist_ok=True)

        (self._output_dir / "dcam_calib_stereo.json").write_text(
            json.dumps(self._calibration, indent=2))
        (self._output_dir / "dcam_calib.json").write_text(
            json.dumps(self._device_info(), indent=2))

        self._timestamps.clear()
        self._read_failures = 0
        self._max_gap_s = 0.0
        self._last_frame_mono = time.monotonic()
        with self._files_lock:
            self._file = open(self._output_dir / "dcam_stereo.mjpeg", "wb")
            self._recording = True

        logger.info("StereoUvcCapture recording → %s", self._output_dir)

    def stop_recording(self) -> dict:
        """Stop disk writes, close the file, dump timestamps. Camera keeps streaming."""
        if not self._recording:
            return {}

        with self._files_lock:
            self._recording = False
            f, self._file = self._file, None
        f.close()
        # Include the tail: frames that stop arriving before the stop are a
        # gap too, and the one that went unnoticed was exactly that.
        max_gap_s = max(self._max_gap_s, time.monotonic() - self._last_frame_mono)

        if self._output_dir:
            (self._output_dir / "dcam_stereo_timestamps.json").write_text(
                json.dumps({"samples": self._timestamps}))

        stats = {
            "stereo_frames": len(self._timestamps),
            "read_failures": self._read_failures,
            "max_gap_s": round(max_gap_s, 3),
            # No IMU. The key is always present because rpi.py reads
            # oakd_stats.get("imu_samples", 0) into metadata.json.
            "imu_samples": 0,
        }
        if self._read_failures or max_gap_s > _GAP_WARN_S:
            logger.warning("StereoUvcCapture: %d failed reads, longest gap %.2f s "
                           "without a frame", self._read_failures, max_gap_s)
        logger.info("StereoUvcCapture recording stopped: %s", stats)
        return stats

    # ---------------------------------------------------------------- reader

    def _reader_loop(self) -> None:
        """Read raw MJPG continuously; write while recording; refresh preview."""
        seq = 0
        while not self._stop_event.is_set():
            ok, buf = self._cap.read()
            if not ok:
                if not self._stop_event.is_set():
                    logger.debug("stereo camera read failed")
                    if self._recording:
                        self._read_failures += 1
                continue
            if self._recording:
                now = time.monotonic()
                self._max_gap_s = max(self._max_gap_s, now - self._last_frame_mono)
                self._last_frame_mono = now
            # Kernel buffer timestamp (CLOCK_MONOTONIC), read before anything
            # else touches the capture.
            pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            self._frames_read += 1

            if self._frames_read % _PREVIEW_EVERY == 0:
                self._update_preview(buf)

            with self._files_lock:
                if not self._recording or self._file is None:
                    continue
                self._file.write(buf.tobytes())
                self._timestamps.append({
                    "seq": seq,
                    "device_us": int(pos_ms * 1000),
                    "host_ms": self.sync.monotonic_s_to_ms(pos_ms / 1000.0),
                })
                seq += 1
        logger.info("stereo reader stopped")

    def _update_preview(self, buf: np.ndarray) -> None:
        img = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_4)
        if img is not None:
            left, _ = split_eyes(img, self.profile["swap_eyes"])
            self._latest_preview = left  # atomic reference swap

    # ------------------------------------------------------------- live view

    def get_latest_imu(self) -> dict | None:
        """Always None — these cameras have no IMU."""
        return None

    def get_depth_jpeg(self, quality: int = 80) -> bytes | None:
        """Left eye at 1/4 scale. There is no live depth: it is computed offline."""
        img = self._latest_preview
        if img is None:
            return None
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return jpg.tobytes() if ok else None

    # -------------------------------------------------------------- shutdown

    def shutdown(self) -> None:
        """Stop the reader thread and release the camera."""
        if not self._initialized:
            return
        if self._recording:
            try:
                self.stop_recording()
            except Exception:
                pass

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        # UVC controls persist on the device after release: give auto-exposure
        # back so other tools are not left with our fixed value.
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        if self.gamma:
            self._cap.set(cv2.CAP_PROP_GAMMA, 100)  # camera default
        self._cap.release()
        self._cap = None
        self._initialized = False
        logger.info("StereoUvcCapture shut down")

    def _device_info(self) -> dict:
        index, name = self._node if self._node else (None, None)
        return {
            "name": name,
            "video_node": index,
            "profile": self.profile_name,
            "mode": self.mode,
            "fps": self.fps,
            "exposure_100us": self.exposure,
            "gamma": self.gamma or None,
            "swap_eyes": self.profile["swap_eyes"],
        }

    def camera_info(self) -> dict:
        """Identity of the stereo camera for episode provenance; `imu` is None."""
        info = {k: v for k, v in self._device_info().items() if v is not None}
        info["imu"] = None
        info["image_orientation"] = "as_captured"
        return info

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def imu_sample_count(self) -> int:
        """Always 0 — no IMU on these cameras."""
        return 0
