"""Orbbec Gemini 305 capture, the IMU-less counterpart to oakd.py.

Exists so the rig is not single-sourced on Luxonis. Satisfies
hardware.depth_camera.DepthCameraCapture, so RpiBackend drives it identically.

Structure deliberately mirrors OakdCapture: an always-on pipeline started by
init_device(), drainer threads that cache the latest frame for live preview and
only write to disk while `_recording` is set.

Two differences from the OAK-D, both verified on hardware:

1. **No IMU.** The 305 exposes only COLOR/DEPTH/LEFT_IR/RIGHT_IR. No
   `oakd_imu.json` is written and `get_latest_imu()` returns None forever.
   Phase 0 measured IMU-free odometry as indistinguishable from re-running the
   same pipeline, and convert.py/offline_vslam both tolerate the absent CSVs.

2. **No on-board encoder.** The OAK-D H.264-encoded on its own ASIC; here the
   Pi does it. Measured on a Pi 4: libx264 ultrafast single-threaded runs at
   4.07x realtime (111% of one core) producing ~54 MB/min, so it is pinned to
   one thread and left on the CPU — picamera2 keeps the hardware V4L2 encoder
   for raw_video.mp4 and the two never contend.

Streams LEFT_IR (Y8) + Depth (Y16) with D2C off. Verified by stereo NCC on real
frames: LEFT_IR is the rectified left image, row-rectified against RIGHT_IR and
sharing depth's pixel grid, so no host undistortion is needed.

Output layout matches oakd.py's `oakd_*` names on purpose — convert.py,
checks/*, dataset.py and every existing HuggingFace dataset key on them:
    oakd_left.mp4               H.264, rectified left (from LEFT_IR)
    oakd_depth/<seq>.png        uint16 mm, packed to oakd_depth.mkv on stop
    oakd_left_timestamps.json   per-frame device_us + host_ms
    oakd_depth_timestamps.json
    oakd_calib_offline.json     fx/fy/cx/cy/baseline (no imu_to_cam)
    oakd_calib.json             device info + intrinsics/distortion dump
    oakd_clock_pairs.json       first device_us <-> host_ms pair per stream
    oak_mask.png                body mask (copied from hardware/oak_mask.png)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .sync import SyncManager

logger = logging.getLogger(__name__)

_MASK_PATH = Path(__file__).parent / "oak_mask.png"

# Raw depth above this (in millimetres, after depth_scale) is discarded. The 305
# is rated 4-100cm and returns progressively noisier values further out; 65535
# raw is its saturation marker, which unscaled would become a phantom 6.5 m
# return. 0 must remain the ONLY invalid value because wait_until_ready() gates
# on (depth > 0).mean() and the body mask zeroes pixels by multiplication.
_MAX_VALID_DEPTH_MM = 10_000


class OrbbecCapture:
    """Captures rectified left mono (H.264) + depth from a Gemini 305 over USB3."""

    DEFAULT_FPS = 30
    DEFAULT_DEPTH_RESOLUTION = (640, 400)
    DEFAULT_BITRATE = "8M"
    DEFAULT_DEPTH_PNG_COMPRESSION = 1

    PREVIEW_DEPTH_MIN_MM = 200
    PREVIEW_DEPTH_MAX_MM = 3000

    def __init__(
        self,
        sync_manager: SyncManager,
        fps: int = DEFAULT_FPS,
        depth_resolution: tuple[int, int] = DEFAULT_DEPTH_RESOLUTION,
        bitrate: str = DEFAULT_BITRATE,
        enable_depth: bool = True,
        depth_png_compression: int = DEFAULT_DEPTH_PNG_COMPRESSION,
    ) -> None:
        self.sync = sync_manager
        self.fps = fps
        self.depth_resolution = depth_resolution
        self.bitrate = bitrate
        self.enable_depth = enable_depth
        self.depth_png_compression = depth_png_compression

        # The SDK collects a temporary Context mid-call and then raises
        # 'NULL pointer passed for argument "deviceMgr"', so the Context and the
        # device list have to stay referenced for the lifetime of the device.
        self._ctx = None
        self._devlist = None
        self._device = None
        self._pipeline = None

        self._output_dir: Path | None = None
        self._recording = False
        self._initialized = False

        self._left_ts: list[dict] = []
        self._depth_ts: list[dict] = []
        self._clock_pairs: list[dict] = []

        self._encoder: subprocess.Popen | None = None
        self._files_lock = threading.Lock()

        self._latest_depth: np.ndarray | None = None
        self._depth_scale = 1.0

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._calibration_json: dict | None = None
        self._calib_offline: dict | None = None
        self._mask: np.ndarray | None = None

    # ------------------------------------------------------------------ init

    def init_device(self) -> None:
        """Connect, sync clocks, read calibration, START the pipeline, drain."""
        import pyorbbecsdk as ob

        self._ctx = ob.Context()
        self._devlist = self._ctx.query_devices()
        if self._devlist.get_count() == 0:
            raise RuntimeError("No Orbbec device found (check the udev rules — "
                               "see `make install-udev-orbbec`)")
        self._device = self._devlist.get_device_by_index(0)
        info = self._device.get_device_info()

        # Align the device clock to the host BEFORE streaming so
        # get_global_timestamp_us() lands on the host timeline. This is the
        # analog of depthai's clock sync; without it the only host-side stamp is
        # get_system_timestamp_us(), which is ~28 ms late because it marks
        # delivery rather than capture.
        self._device.timer_sync_with_host()

        logger.info("Gemini 305 ready: %s id=%s fw=%s conn=%s",
                    info.get_name(), info.get_serial_number(),
                    info.get_firmware_version(), info.get_connection_type())

        self._load_mask()

        self._pipeline = ob.Pipeline(self._device)
        config = ob.Config()
        w, h = self.depth_resolution
        depth_profile = self._pick_profile(ob.OBSensorType.DEPTH_SENSOR, "Y16", w, h)
        config.enable_stream(depth_profile)
        config.enable_stream(
            self._pick_profile(ob.OBSensorType.LEFT_IR_SENSOR, "Y8", w, h))

        # Hardware frame sync — depth and LEFT_IR then carry an identical device
        # timestamp and frame index (measured: 0 us skew).
        self._pipeline.enable_frame_sync()
        self._pipeline.start(config)

        # Intrinsics must be read after start(): they track the streaming
        # resolution, and the per-unit values differ from the datasheet's
        # nominal table.
        self._calibration_json = self._dump_calibration()
        self._calib_offline = self._dump_calib_offline()

        self._initialized = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

        logger.info("OrbbecCapture pipeline running (idle, awaiting start_recording)")

    def _pick_profile(self, sensor, fmt: str, w: int, h: int):
        """Exact width/height/format/fps profile, or a clear error listing options."""
        plist = self._pipeline.get_stream_profile_list(sensor)
        options = []
        for i in range(plist.get_count()):
            p = plist.get_stream_profile_by_index(i)
            if str(p.get_format()) != f"OBFormat.{fmt}":
                continue
            options.append((p.get_width(), p.get_height(), p.get_fps()))
            if (p.get_width(), p.get_height()) == (w, h) and p.get_fps() == self.fps:
                return p
        raise RuntimeError(
            f"No {fmt} profile at {w}x{h}@{self.fps} for {sensor}; "
            f"available: {sorted(set(options))}"
        )

    def _load_mask(self) -> None:
        if not _MASK_PATH.exists():
            logger.warning("oak_mask.png not found at %s — capturing without mask",
                           _MASK_PATH)
            return
        mask = cv2.imread(str(_MASK_PATH), cv2.IMREAD_GRAYSCALE)
        want = (self.depth_resolution[1], self.depth_resolution[0])
        if mask is not None and mask.shape == want:
            self._mask = mask
        else:
            logger.warning("oak_mask.png shape %s != depth_resolution %s — mask disabled",
                           None if mask is None else mask.shape, self.depth_resolution)

    def _dump_calibration(self) -> dict:
        """Device info + every calibration set, the analog of the OAK-D EEPROM dump."""
        try:
            info = self._device.get_device_info()
            out: dict = {
                "name": info.get_name(),
                "serial_number": info.get_serial_number(),
                "firmware_version": info.get_firmware_version(),
                "connection_type": str(info.get_connection_type()),
                "baseline_mm": float(self._device.get_baseline().baseline),
                "sets": [],
            }
            cplist = self._device.get_calibration_camera_param_list()
            for i in range(cplist.get_count()):
                cp = cplist.get_camera_param(i)
                di, dd = cp.depth_intrinsic, cp.depth_distortion
                out["sets"].append({
                    "depth_intrinsic": {
                        "width": di.width, "height": di.height,
                        "fx": di.fx, "fy": di.fy, "cx": di.cx, "cy": di.cy,
                    },
                    "depth_distortion": {
                        k: getattr(dd, k) for k in
                        ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
                    },
                })
            return out
        except Exception as e:
            logger.warning("Could not read Orbbec calibration: %s", e)
            return {}

    def _dump_calib_offline(self) -> dict:
        """Flat intrinsics for offline SLAM.

        No `imu_to_cam` key — there is no IMU. offline_vslam probes for it with
        contains() and falls back to identity.
        """
        try:
            cam = self._pipeline.get_camera_param()
            intr = cam.depth_intrinsic
            return {
                "width": int(intr.width),
                "height": int(intr.height),
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "cx": float(intr.cx),
                "cy": float(intr.cy),
                "baseline": float(self._device.get_baseline().baseline) / 1000.0,
            }
        except Exception as e:
            logger.warning("Could not extract Orbbec offline calib: %s", e)
            return {}

    # ------------------------------------------------------ recording on/off

    def wait_until_ready(self, timeout: float = 5.0,
                         min_depth_coverage: float = 0.05) -> bool:
        """Block until depth has converged past warmup, or until timeout."""
        if not self._initialized:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            depth = self._latest_depth
            if depth is not None and float((depth > 0).mean()) >= min_depth_coverage:
                return True
            time.sleep(0.02)
        logger.warning("Gemini 305 not ready after %.1fs — starting capture anyway",
                       timeout)
        return False

    def start_recording(self, output_dir: Path) -> None:
        if not self._initialized:
            raise RuntimeError("OrbbecCapture not initialized. Call init_device() first.")
        if self._recording:
            raise RuntimeError("OrbbecCapture already recording")
        if not self.sync.is_started:
            raise RuntimeError("SyncManager must be started before OrbbecCapture")

        self._output_dir = Path(output_dir).absolute()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_depth:
            (self._output_dir / "oakd_depth").mkdir(parents=True, exist_ok=True)

        if self._calibration_json:
            (self._output_dir / "oakd_calib.json").write_text(
                json.dumps(self._calibration_json, indent=2))
        if self._calib_offline:
            (self._output_dir / "oakd_calib_offline.json").write_text(
                json.dumps(self._calib_offline, indent=2))
        if self._mask is not None:
            try:
                shutil.copyfile(_MASK_PATH, self._output_dir / "oak_mask.png")
            except OSError as e:
                logger.warning("Could not copy oak_mask.png to session dir: %s", e)

        self._left_ts.clear()
        self._depth_ts.clear()
        self._clock_pairs.clear()

        with self._files_lock:
            self._encoder = self._spawn_encoder(self._output_dir / "oakd_left.mp4")
            self._recording = True

        logger.info("OrbbecCapture recording → %s", self._output_dir)

    def _spawn_encoder(self, mp4_path: Path) -> subprocess.Popen:
        """ffmpeg reading raw Y8 on stdin, writing H.264 mp4.

        Pinned to one thread: measured 4.07x realtime on a single Pi 4 core, so
        one is plenty and it leaves the other three for capture, picamera2 and
        the daemon. `-preset ultrafast` was as small on disk as `veryfast`
        (54 MB/min either way) for a fraction of the CPU.

        Unlike the OAK-D path there is no keyframe gating to worry about: the
        encoder is started fresh here, so one input frame is exactly one output
        frame and the mp4 frame count always matches the timestamps sidecar.
        """
        w, h = self.depth_resolution
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
            "-r", str(self.fps), "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-threads", "1", "-b:v", self.bitrate,
            "-pix_fmt", "yuv420p", str(mp4_path),
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def stop_recording(self) -> dict:
        """Stop disk writes, close the encoder, dump sidecars. Pipeline keeps running."""
        if not self._recording:
            return {}

        with self._files_lock:
            self._recording = False
            enc, self._encoder = self._encoder, None

        if enc is not None:
            try:
                enc.stdin.close()
                enc.wait(timeout=30)
                if enc.returncode != 0:
                    err = enc.stderr.read().decode(errors="replace")[-300:]
                    logger.error("ffmpeg encode failed: %s", err)
            except Exception as e:
                logger.error("Error finalising encoder: %s", e)
                enc.kill()

        if self.enable_depth:
            self._pack_depth_video()

        if self._output_dir:
            (self._output_dir / "oakd_left_timestamps.json").write_text(
                json.dumps({"samples": self._left_ts}))
            if self.enable_depth:
                (self._output_dir / "oakd_depth_timestamps.json").write_text(
                    json.dumps({"samples": self._depth_ts}))
            (self._output_dir / "oakd_clock_pairs.json").write_text(
                json.dumps({"pairs": self._clock_pairs}))

        stats = {
            "left_frames": len(self._left_ts),
            "depth_frames": len(self._depth_ts) if self.enable_depth else None,
            # No IMU on this device. The key is always present because rpi.py
            # reads oakd_stats.get("imu_samples", 0) into metadata.json.
            "imu_samples": 0,
        }
        logger.info("OrbbecCapture recording stopped: %s", stats)
        return stats

    # ---------------------------------------------------------------- writer

    def _writer_loop(self) -> None:
        """Drain frames continuously; cache for preview; write only while recording."""
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, self.depth_png_compression]
        n = 0
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(1000)
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.debug("orbbec wait_for_frames error: %s", e)
                continue
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_left_ir_frame()
            if depth_frame is None or ir_frame is None:
                continue

            depth_mm = self._to_millimetres(depth_frame)
            if self._mask is not None and depth_mm.shape == self._mask.shape:
                depth_mm = depth_mm * (self._mask > 0).astype(depth_mm.dtype)
            self._latest_depth = depth_mm  # atomic reference swap for preview

            if not self._recording:
                continue

            # Both streams are hardware-synced to the same device timestamp, so
            # one stamp covers the pair.
            device_us = int(depth_frame.get_timestamp_us())
            host_ms = self.sync.monotonic_s_to_ms(
                depth_frame.get_global_timestamp_us() / 1_000_000.0)
            seq = int(depth_frame.get_index())

            if not self._clock_pairs:
                self._clock_pairs.append({
                    "stream": "left", "seq": seq,
                    "device_us": device_us, "host_ms": host_ms,
                })

            gray = np.frombuffer(ir_frame.get_data(), dtype=np.uint8).reshape(
                ir_frame.get_height(), ir_frame.get_width())

            with self._files_lock:
                if not self._recording or self._encoder is None:
                    continue
                try:
                    self._encoder.stdin.write(gray.tobytes())
                except (BrokenPipeError, ValueError):
                    logger.error("encoder pipe closed unexpectedly")
                    continue
                self._left_ts.append(
                    {"seq": seq, "device_us": device_us, "host_ms": host_ms})
                if self.enable_depth:
                    cv2.imwrite(
                        str(self._output_dir / "oakd_depth" / f"{seq:08d}.png"),
                        depth_mm, png_params)
                    self._depth_ts.append(
                        {"seq": seq, "device_us": device_us, "host_ms": host_ms})
                n += 1
        logger.info("orbbec writer: %d frames recorded", n)

    def _to_millimetres(self, depth_frame) -> np.ndarray:
        """Raw Y16 -> uint16 millimetres, with 0 as the only invalid marker.

        `depth_scale` is 0.1 on this device, i.e. raw units are tenths of a
        millimetre. Writing raw values through unscaled makes RTAB-Map read every
        depth 10x too far, silently — verified against fx*B/Z by stereo matching.
        """
        scale = depth_frame.get_depth_scale()
        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            depth_frame.get_height(), depth_frame.get_width())
        mm = raw.astype(np.float32) * scale
        mm[raw == 65535] = 0          # saturation marker, not a 6.5 m return
        mm[mm > _MAX_VALID_DEPTH_MM] = 0
        return mm.astype(np.uint16)

    # ------------------------------------------------------------- live view

    def get_latest_imu(self) -> dict | None:
        """Always None — the Gemini 305 has no IMU. See the module docstring."""
        return None

    def get_depth_jpeg(self, quality: int = 80) -> bytes | None:
        """Latest depth as a colorized JPEG (turbo colormap, 0.2-3 m)."""
        depth = self._latest_depth
        if depth is None:
            return None
        d_min, d_max = self.PREVIEW_DEPTH_MIN_MM, self.PREVIEW_DEPTH_MAX_MM
        mask = (depth >= d_min) & (depth <= d_max)
        d_clip = np.clip(depth, d_min, d_max).astype(np.float32)
        d_norm = (255.0 * (d_max - d_clip) / (d_max - d_min)).astype(np.uint8)
        d_norm[~mask] = 0
        colorized = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
        ok, buf = cv2.imencode(".jpg", colorized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    # -------------------------------------------------------------- shutdown

    def _pack_depth_video(self) -> None:
        """Pack oakd_depth/*.png into one lossless FFV1 16-bit oakd_depth.mkv.

        Same contract as oakd.py's version: capture order, bit-identical once
        decoded, PNGs kept on any failure since convert/episode_check accept
        either layout. Lossless is right here — unlike the left stream, depth
        values are measurements the SLAM reads numerically.
        """
        if not self._output_dir or not self._depth_ts:
            return
        depth_dir = self._output_dir / "oakd_depth"
        if not depth_dir.is_dir():
            return
        out = self._output_dir / "oakd_depth.mkv"
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                stage = Path(tmp)
                for i, s in enumerate(self._depth_ts):
                    src = depth_dir / f'{int(s["seq"]):08d}.png'
                    if not src.exists():
                        logger.warning("depth pack: missing PNG for seq %s — keeping PNGs",
                                       s.get("seq"))
                        return
                    (stage / f"{i:06d}.png").symlink_to(src.resolve())
                result = subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", f"{float(self.fps):.3f}",
                    "-start_number", "0", "-i", str(stage / "%06d.png"),
                    "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray16le", str(out),
                ], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("depth pack ffmpeg failed: %s", result.stderr[-300:])
                if out.exists():
                    out.unlink()
                return
            shutil.rmtree(depth_dir)
            logger.info("orbbec depth packed → %s (%d frames)", out.name, len(self._depth_ts))
        except Exception as e:
            logger.warning("depth pack failed (%s) — keeping PNGs", e)

    def shutdown(self) -> None:
        """Stop the pipeline and exit the drainer thread."""
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

        try:
            self._pipeline.stop()
        except Exception as e:
            logger.warning("pipeline stop error: %s", e)

        self._pipeline = None
        self._device = None
        self._devlist = None
        self._ctx = None
        self._initialized = False
        logger.info("OrbbecCapture shut down")

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def imu_sample_count(self) -> int:
        """Always 0 — no IMU on this device."""
        return 0
