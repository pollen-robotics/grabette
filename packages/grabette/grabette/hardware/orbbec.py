"""Orbbec Gemini 305 capture, the IMU-less counterpart to oakd.py.

Exists so the rig is not single-sourced on Luxonis. Satisfies
hardware.depth_camera.DepthCameraCapture, so RpiBackend drives it identically.

Structure deliberately mirrors OakdCapture: an always-on pipeline started by
init_device(), drainer threads that cache the latest frame for live preview and
only write to disk while `_recording` is set.

Two differences from the OAK-D, both verified on hardware:

1. **No IMU.** The 305 exposes only COLOR/DEPTH/LEFT_IR/RIGHT_IR. No
   `dcam_imu.json` is written and `get_latest_imu()` returns None forever.
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

Output layout matches oakd.py's `dcam_*` names on purpose: the postprocess
pipeline reads whichever camera produced the episode without caring which:
    dcam_left.mp4               H.264, rectified left (from LEFT_IR)
    dcam_depth.mkv              uint16 mm, lossless FFV1, encoded live
    dcam_left_timestamps.json   per-frame device_us + host_ms
    dcam_depth_timestamps.json
    dcam_calib_offline.json     fx/fy/cx/cy/baseline (no imu_to_cam)
    dcam_calib.json             device info + intrinsics/distortion dump
    dcam_clock_pairs.json       first device_us <-> host_ms pair per stream
    dcam_mask.png                body mask (copied from hardware/dcam_mask.png)
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

_MASK_PATH = Path(__file__).parent / "dcam_mask.png"

# Raw depth above this (in millimetres, after depth_scale) is discarded. The 305
# is rated 4-100cm and returns progressively noisier values further out; 65535
# raw is its saturation marker, which unscaled would become a phantom 6.5 m
# return. 0 must remain the ONLY invalid value because wait_until_ready() gates
# on (depth > 0).mean() and the body mask zeroes pixels by multiplication.
_MAX_VALID_DEPTH_MM = 10_000

# While NOT recording the pipeline still runs, purely to keep a live preview
# frame warm. Converting every frame for that is waste: measured on a Pi 4, the
# always-on loop alone accounted for ~18 C of steady-state temperature
# (78.8 C enabled-and-idle vs 60.8 C disabled), which pushed the board into
# thermal throttling before a capture even began. The preview WebSocket runs at
# ~15 fps and nobody is watching most of the time, so convert one frame in six
# (5 fps) when idle. Recording converts every frame, as it must.
_IDLE_PREVIEW_EVERY = 6

# One Context for the process, created lazily and never torn down.
#
# ob.Context() is not a cheap handle: each one spawns a device-watcher thread
# and native state that is NOT released when the Python object is dropped.
# Because the daemon powers the camera down after each capture (the keepalive)
# and re-initialises it for the next one, constructing a Context per
# init_device() leaked ~14-22 MB and a thread or two per cycle. Measured on the
# Pi: RSS climbed 324 -> 457 MB over six enable/disable cycles and reached
# 1.66 GB in normal use, at which point memory pressure started costing ~50% of
# the depth frames. The SDK log made it plain -- "Create PollingDeviceWatcher"
# once per Context, 16 times in one session.
_CONTEXT = None
_CONTEXT_LOCK = threading.Lock()


def _shared_context():
    """The process-wide Orbbec Context, created on first use."""
    global _CONTEXT
    with _CONTEXT_LOCK:
        if _CONTEXT is None:
            import pyorbbecsdk as ob
            _CONTEXT = ob.Context()
            logger.info("Created the process-wide Orbbec Context")
        return _CONTEXT


class OrbbecCapture:
    """Captures rectified left mono (H.264) + depth from a Gemini 305 over USB3."""

    DEFAULT_FPS = 30
    DEFAULT_DEPTH_RESOLUTION = (640, 400)
    DEFAULT_BITRATE = "8M"

    PREVIEW_DEPTH_MIN_MM = 200
    PREVIEW_DEPTH_MAX_MM = 3000

    def __init__(
        self,
        sync_manager: SyncManager,
        fps: int = DEFAULT_FPS,
        depth_resolution: tuple[int, int] = DEFAULT_DEPTH_RESOLUTION,
        bitrate: str = DEFAULT_BITRATE,
        enable_depth: bool = True,
        ir_exposure_us: int = 0,
        ir_gain: int = 0,
    ) -> None:
        self.sync = sync_manager
        self.fps = fps
        self.depth_resolution = depth_resolution
        self.bitrate = bitrate
        self.enable_depth = enable_depth
        # 0 = leave auto-exposure alone (the default, unchanged behaviour).
        # Non-zero disables AE and pins exposure/gain — see _apply_exposure.
        self.ir_exposure_us = ir_exposure_us
        self.ir_gain = ir_gain

        # The SDK collects a temporary Context mid-call and then raises
        # 'NULL pointer passed for argument "deviceMgr"', so the Context and the
        # device list have to stay referenced for the lifetime of the device.
        # The Context itself is process-wide (see _shared_context); this is just
        # a local alias to keep it reachable.
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
        self._depth_encoder: subprocess.Popen | None = None
        self._files_lock = threading.Lock()

        self._latest_depth: np.ndarray | None = None
        self._depth_scale = 1.0
        # Orbbec's get_global_timestamp_us() is CLOCK_REALTIME (epoch), but
        # SyncManager.monotonic_s_to_ms() expects CLOCK_MONOTONIC — depthai's
        # getTimestamp() is monotonic, which is what it was written for. Feeding
        # epoch straight in yields host_ms ~1.79e12, which downstream collapses
        # to a single value at float precision and makes every episode look
        # zero-length. Measured once at init and applied per frame.
        self._epoch_to_monotonic = 0.0

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._calibration_json: dict | None = None
        self._calib_offline: dict | None = None
        self._mask: np.ndarray | None = None
        self._mask_mul: np.ndarray | None = None

    # ------------------------------------------------------------------ init

    def init_device(self) -> None:
        """Connect, sync clocks, read calibration, START the pipeline, drain."""
        import pyorbbecsdk as ob

        self._ctx = _shared_context()
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
        # Offset between the epoch clock the SDK stamps with and the monotonic
        # clock SyncManager runs on. Sampled tightly so scheduling noise between
        # the two reads stays sub-millisecond.
        self._epoch_to_monotonic = time.monotonic() - time.time()

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

        self._apply_exposure()

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

    def _apply_exposure(self) -> None:
        """Optionally pin IR exposure/gain instead of letting auto-exposure run.

        Default AE in a dim room picks ~15.6 ms of integration with gain at its
        minimum of 16 — nearly half the 33 ms frame interval. That is the right
        trade for a static camera (low noise) and the wrong one for a moving
        rig: frames smear, and tracking losses in real recordings correlated
        with a drop in image sharpness while depth coverage stayed healthy.

        Pinning a shorter exposure forces the noise up instead. Measured on a
        static scene, 3000 us with gain 160 holds the same mean brightness as
        AE's 15600/16 while costing ~9 points of depth coverage (59.6% -> 50.7%).
        Whether that trade wins depends on how fast the rig actually moves, so
        this stays OFF by default and is opt-in per device.
        """
        if not self.ir_exposure_us:
            return
        try:
            import pyorbbecsdk as ob
            P = ob.OBPropertyID
            self._device.set_bool_property(P.OB_PROP_IR_AUTO_EXPOSURE_BOOL, False)
            self._device.set_int_property(P.OB_PROP_IR_EXPOSURE_INT, self.ir_exposure_us)
            if self.ir_gain:
                self._device.set_int_property(P.OB_PROP_IR_GAIN_INT, self.ir_gain)
            logger.info("IR exposure pinned: %d us, gain %s",
                        self.ir_exposure_us, self.ir_gain or "unchanged")
        except Exception as e:
            logger.warning("Could not pin IR exposure (leaving AE on): %s", e)

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
            logger.warning("dcam_mask.png not found at %s — capturing without mask",
                           _MASK_PATH)
            return
        mask = cv2.imread(str(_MASK_PATH), cv2.IMREAD_GRAYSCALE)
        want = (self.depth_resolution[1], self.depth_resolution[0])
        if mask is not None and mask.shape == want:
            self._mask = mask
            # Precomputed once. Doing `(mask > 0).astype(...)` per frame meant a
            # full-frame compare plus a cast 30 times a second for a value that
            # never changes.
            self._mask_mul = (mask > 0).astype(np.uint16)
        else:
            logger.warning("dcam_mask.png shape %s != depth_resolution %s — mask disabled",
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


        if self._calibration_json:
            (self._output_dir / "dcam_calib.json").write_text(
                json.dumps(self._calibration_json, indent=2))
        if self._calib_offline:
            (self._output_dir / "dcam_calib_offline.json").write_text(
                json.dumps(self._calib_offline, indent=2))
        if self._mask is not None:
            try:
                shutil.copyfile(_MASK_PATH, self._output_dir / "dcam_mask.png")
            except OSError as e:
                logger.warning("Could not copy dcam_mask.png to session dir: %s", e)

        self._left_ts.clear()
        self._depth_ts.clear()
        self._clock_pairs.clear()

        with self._files_lock:
            self._encoder = self._spawn_encoder(self._output_dir / "dcam_left.mp4")
            if self.enable_depth:
                self._depth_encoder = self._spawn_depth_encoder(
                    self._output_dir / "dcam_depth.mkv")
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

    def _spawn_depth_encoder(self, mkv_path: Path) -> subprocess.Popen:
        """ffmpeg reading raw uint16 depth on stdin, writing lossless FFV1.

        Encoding live replaces the OAK-D path's write-PNGs-then-pack-at-stop
        approach. A 640x400 uint16 PNG per frame cost more than the whole rest of
        the loop on a Pi 4 and was the main reason two thirds of frames were
        dropped once picamera2 was encoding concurrently. Piping to ffmpeg moves
        the compression onto its own process, drops the separate packing pass,
        and produces exactly the same dcam_depth.mkv.

        Lossless is right here, unlike the left stream: depth values are
        measurements RTAB-Map reads numerically.
        """
        w, h = self.depth_resolution
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{w}x{h}",
            "-r", str(self.fps), "-i", "-",
            "-c:v", "ffv1", "-level", "3", "-threads", "1", str(mkv_path),
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
            denc, self._depth_encoder = self._depth_encoder, None

        # Both encoders finalise in parallel; each is an independent process.
        for proc, label in ((enc, "left"), (denc, "depth")):
            if proc is None:
                continue
            try:
                proc.stdin.close()
                proc.wait(timeout=60)
                if proc.returncode != 0:
                    err = proc.stderr.read().decode(errors="replace")[-300:]
                    logger.error("ffmpeg %s encode failed: %s", label, err)
            except Exception as e:
                logger.error("Error finalising %s encoder: %s", label, e)
                proc.kill()

        if self._output_dir:
            (self._output_dir / "dcam_left_timestamps.json").write_text(
                json.dumps({"samples": self._left_ts}))
            if self.enable_depth:
                (self._output_dir / "dcam_depth_timestamps.json").write_text(
                    json.dumps({"samples": self._depth_ts}))
            (self._output_dir / "dcam_clock_pairs.json").write_text(
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
        """Drain frames continuously; cache for preview; write only while recording.

        Per-frame work is kept deliberately small — an integer depth conversion
        and two pipe writes. Anything heavier here (a float conversion, a PNG
        encode) stalls the loop past the 33 ms frame interval and the SDK drops
        frames, which is exactly what happened in the first on-device recordings.
        """
        n = 0
        idle_n = 0
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

            recording = self._recording
            if not recording:
                # Idle: keep the preview warm at ~5 fps instead of 30, and skip
                # the conversion entirely on the frames in between.
                idle_n += 1
                if idle_n % _IDLE_PREVIEW_EVERY:
                    continue

            depth_mm = self._to_millimetres(depth_frame)
            if self._mask_mul is not None and depth_mm.shape == self._mask_mul.shape:
                depth_mm = depth_mm * self._mask_mul
            self._latest_depth = depth_mm  # atomic reference swap for preview

            if not recording:
                continue

            # Both streams are hardware-synced to the same device timestamp, so
            # one stamp covers the pair.
            device_us = int(depth_frame.get_timestamp_us())
            host_ms = self.sync.monotonic_s_to_ms(
                depth_frame.get_global_timestamp_us() / 1_000_000.0
                + self._epoch_to_monotonic)
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
                    self._left_ts.append(
                        {"seq": seq, "device_us": device_us, "host_ms": host_ms})
                    if self._depth_encoder is not None:
                        self._depth_encoder.stdin.write(depth_mm.tobytes())
                        self._depth_ts.append(
                            {"seq": seq, "device_us": device_us, "host_ms": host_ms})
                except (BrokenPipeError, ValueError):
                    logger.error("encoder pipe closed unexpectedly")
                    continue
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
        # Integer path, not float. scale is 0.1 on this device, so millimetres
        # are raw // 10 — one uint16 pass instead of allocating a 1 MB float32
        # and walking it four times. At 30 fps on a Pi 4 that difference is the
        # gap between keeping up and dropping two thirds of the frames.
        step = int(round(1.0 / scale)) if scale > 0 else 1
        mm = raw // step if step > 1 else raw.copy()
        np.putmask(mm, raw == 65535, 0)      # saturation marker, not 6.5 m
        np.putmask(mm, mm > _MAX_VALID_DEPTH_MM, 0)
        return mm

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
        # Drop only this instance's alias. The Context stays alive for the
        # process — recreating it per cycle is exactly what leaked.
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
