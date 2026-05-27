"""OAK-D SR capture using depthai v3.

v4 (rgbd branch): always-on pipeline + per-capture writers + live preview.

After init_device(), the OAK-D pipeline runs continuously (until shutdown()).
Drainer/writer threads always pull frames and cache the latest one for live
view; they only write to disk when `_recording=True` (start_recording /
stop_recording toggles).

Streams:
- 2× Camera (CAM_B/CAM_C) → 2× VideoEncoder (H.264) → host queues
  → .h264 elementary stream → muxed to .mp4 on stop_recording
- StereoDepth (PresetMode = ROBOTICS) fed from a SEPARATE lower-res
  (GRAY8) output of each camera → uint16 depth → PNG sequence on disk
- BNO086 IMU at ~200 Hz → JSON

Frames are encoded from `stereo.rectifiedLeft/Right` so the mp4s are already
SLAM-ready (rectified + undistorted). On-device `setImageOrientation(ROTATE_180)`
flips both sensors before stereo, since the OAK-D SR is mounted upside-down on
the grabette.

Output layout per episode (resolution = depth_resolution, default 640×400):
    oakd_left.mp4               H.264 mono, rectified (CAM_B = mono global-shutter)
    oakd_right.mp4              H.264 mono, rectified (CAM_C = mono global-shutter)
    oakd_depth/<seq>.png        uint16 mm (mask-applied if oak_mask.png present)
    oakd_*_timestamps.json      per-stream device_us + host_ms
    oakd_imu.json               accel + gyro + rotation_vector
    oakd_calib.json             full factory calibration (eeprom dump)
    oakd_calib_offline.json     flat fx/fy/cx/cy/baseline/imu_to_cam for offline SLAM
    oak_mask.png                body mask (copied from hardware/oak_mask.png)
    oakd_clock_pairs.json       first device_us ↔ host_ms pair per stream
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


def _device_us(ts) -> int:
    """Convert depthai's timedelta-style timestamp to integer microseconds."""
    return int(ts.total_seconds() * 1_000_000)


class OakdCapture:
    """Captures stereo mono (H.264) + depth + IMU from OAK-D SR over USB3.

    Lifecycle:
        init_device() → pipeline starts, drainer threads run continuously
        start_recording(dir) → writers also write to disk
        stop_recording() → writers stop writing; pipeline keeps running
        shutdown() → pipeline stops, threads exit
    """

    DEFAULT_FPS = 30
    DEFAULT_RESOLUTION = (1280, 800)
    DEFAULT_DEPTH_RESOLUTION = (640, 400)
    DEFAULT_IMU_HZ = 200
    DEFAULT_BITRATE_BPS = 8_000_000
    DEFAULT_KEYFRAME_EVERY = 30
    DEFAULT_DEPTH_PNG_COMPRESSION = 1

    # Depth visualization range (mm) for live JPEG preview
    PREVIEW_DEPTH_MIN_MM = 200
    PREVIEW_DEPTH_MAX_MM = 3000

    def __init__(
        self,
        sync_manager: SyncManager,
        fps: int = DEFAULT_FPS,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        depth_resolution: tuple[int, int] = DEFAULT_DEPTH_RESOLUTION,
        imu_rate_hz: int = DEFAULT_IMU_HZ,
        bitrate_bps: int = DEFAULT_BITRATE_BPS,
        keyframe_every: int = DEFAULT_KEYFRAME_EVERY,
        enable_depth: bool = True,
        depth_png_compression: int = DEFAULT_DEPTH_PNG_COMPRESSION,
    ) -> None:
        self.sync = sync_manager
        self.fps = fps
        self.resolution = resolution
        self.depth_resolution = depth_resolution
        self.imu_rate_hz = imu_rate_hz
        self.bitrate_bps = bitrate_bps
        self.keyframe_every = keyframe_every
        self.enable_depth = enable_depth
        self.depth_png_compression = depth_png_compression

        self._pipeline = None
        self._left_q = None
        self._right_q = None
        self._depth_q = None
        self._imu_q = None

        self._output_dir: Path | None = None
        self._recording = False
        self._initialized = False

        # Per-capture buffers (cleared on start_recording)
        self._left_ts: list[dict] = []
        self._right_ts: list[dict] = []
        self._depth_ts: list[dict] = []
        self._imu_samples: list[dict] = []
        self._clock_pairs: list[dict] = []

        # H.264 file handles, opened on start_recording, closed on stop_recording
        self._left_h264_path: Path | None = None
        self._right_h264_path: Path | None = None
        self._left_h264_fp = None
        self._right_h264_fp = None
        self._files_lock = threading.Lock()  # protects file-handle swap on start/stop

        # Latest-frame cache for live preview (lock-free reads via reference swap)
        self._latest_depth: np.ndarray | None = None  # uint16 (H,W)
        # Latest IMU values for live dashboard (always-on, independent of recording).
        # Each is (host_ms, (x, y, z)) or None until first packet arrives.
        self._latest_accel: tuple[float, tuple[float, float, float]] | None = None
        self._latest_gyro: tuple[float, tuple[float, float, float]] | None = None

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

        self._calibration_json: dict | None = None
        self._calib_offline: dict | None = None
        # 8-bit GRAY mask (depth_resolution-sized) that blacks out the
        # grabette body from SLAM frames. Loaded in init_device.
        self._mask: np.ndarray | None = None

    # ------------------------------------------------------------------ init

    def init_device(self) -> None:
        """Connect, read calibration, build pipeline, START it, launch drainers."""
        import depthai as dai

        logger.info("Connecting to OAK-D for calibration read...")
        with dai.Device() as device:
            self._device_id = device.getDeviceId()
            self._product_name = device.getProductName()
            self._usb_speed = str(device.getUsbSpeed())
            self._imu_type = str(device.getConnectedIMU())
            self._calibration_json = self._dump_calibration(device)
            self._calib_offline = self._dump_calib_offline(device, self.depth_resolution)
            logger.info(
                "OAK-D ready: %s id=%s usb=%s imu=%s",
                self._product_name, self._device_id, self._usb_speed, self._imu_type,
            )

        # Body mask (8-bit GRAY, depth_resolution-sized). Applied to depth on
        # host before PNG write; also copied to session dir so downstream
        # SLAM can mask the mp4 frames.
        if _MASK_PATH.exists():
            mask = cv2.imread(str(_MASK_PATH), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape == (self.depth_resolution[1], self.depth_resolution[0]):
                self._mask = mask
            else:
                logger.warning(
                    "oak_mask.png shape %s != depth_resolution %s — mask disabled",
                    None if mask is None else mask.shape, self.depth_resolution,
                )
        else:
            logger.warning("oak_mask.png not found at %s — capturing without mask", _MASK_PATH)

        # --- Build pipeline ---
        # Stereo input resolution dictates everything downstream:
        # cameras → (rotated 180° on-device) → stereo → rectifiedLeft/Right → H.264
        #                                            ↘ depth → host
        # The OAK-D SR is physically mounted upside-down on the grabette;
        # ROTATE_180_DEG on each sensor makes rectified frames + depth right-side-up
        # for downstream SLAM. Calibration (intrinsics, imu_to_cam) is computed
        # by depthai with the rotation already applied.
        self._pipeline = dai.Pipeline()

        camB = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        camB.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        leftStereoIn = camB.requestOutput(
            self.depth_resolution, type=dai.ImgFrame.Type.GRAY8, fps=self.fps,
        )

        camC = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        camC.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        rightStereoIn = camC.requestOutput(
            self.depth_resolution, type=dai.ImgFrame.Type.GRAY8, fps=self.fps,
        )

        # StereoDepth — exposes rectifiedLeft / rectifiedRight / depth.
        # PresetMode.ROBOTICS is tuned for manipulation/close-range use.
        stereo = self._pipeline.create(dai.node.StereoDepth).build(
            left=leftStereoIn,
            right=rightStereoIn,
            presetMode=dai.node.StereoDepth.PresetMode.ROBOTICS,
        )
        stereo.setOutputSize(*self.depth_resolution)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setRectifyEdgeFillColor(0)
        stereo.enableDistortionCorrection(True)
        stereo.initialConfig.setLeftRightCheckThreshold(10)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)

        # H.264-encode the rectified outputs so the mp4 frames are SLAM-ready
        # (already stereo-aligned and undistorted).
        leftEnc = self._pipeline.create(dai.node.VideoEncoder).build(
            input=stereo.rectifiedLeft,
            bitrate=self.bitrate_bps,
            frameRate=float(self.fps),
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
            keyframeFrequency=self.keyframe_every,
        )
        leftEnc.setNumBFrames(0)
        self._left_q = leftEnc.out.createOutputQueue(maxSize=32, blocking=False)

        rightEnc = self._pipeline.create(dai.node.VideoEncoder).build(
            input=stereo.rectifiedRight,
            bitrate=self.bitrate_bps,
            frameRate=float(self.fps),
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
            keyframeFrequency=self.keyframe_every,
        )
        rightEnc.setNumBFrames(0)
        self._right_q = rightEnc.out.createOutputQueue(maxSize=32, blocking=False)

        if self.enable_depth:
            self._depth_q = stereo.depth.createOutputQueue(maxSize=8, blocking=False)

        imu = self._pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(
            [
                dai.IMUSensor.ACCELEROMETER_RAW,
                dai.IMUSensor.GYROSCOPE_RAW,
                dai.IMUSensor.ROTATION_VECTOR,
            ],
            self.imu_rate_hz,
        )
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        self._imu_q = imu.out.createOutputQueue(maxSize=200, blocking=False)

        # --- Start pipeline (continuous) ---
        self._pipeline.start()
        self._initialized = True

        # Launch drainer threads that run for the lifetime of the device
        self._threads = [
            threading.Thread(
                target=self._writer_loop_video,
                args=(self._left_q, "left", "_left_h264_fp", self._left_ts),
                daemon=True,
            ),
            threading.Thread(
                target=self._writer_loop_video,
                args=(self._right_q, "right", "_right_h264_fp", self._right_ts),
                daemon=True,
            ),
            threading.Thread(
                target=self._writer_loop_imu,
                daemon=True,
            ),
        ]
        if self.enable_depth:
            self._threads.append(threading.Thread(
                target=self._writer_loop_depth,
                daemon=True,
            ))
        for t in self._threads:
            t.start()

        logger.info("OakdCapture pipeline running (idle, awaiting start_recording)")

    @staticmethod
    def _dump_calibration(device) -> dict:
        try:
            handler = device.readCalibration()
            if not hasattr(handler, "eepromToJson"):
                return {}
            data = handler.eepromToJson()
            if isinstance(data, str):
                return json.loads(data)
            if isinstance(data, dict):
                return data
            return {}
        except Exception as e:
            logger.warning("Could not read OAK-D calibration: %s", e)
            return {}

    @staticmethod
    def _dump_calib_offline(device, resolution: tuple[int, int]) -> dict:
        """Flat intrinsics + IMU extrinsics for the rgbd-for-slam offline pipeline.

        Schema matches what grabette-data/docker/oak_vslam expects:
        width, height, fx, fy, cx, cy, baseline (m), imu_to_cam (4x4).

        Note: depthai returns factory intrinsics for the un-rotated sensor.
        We apply setImageOrientation(ROTATE_180) on-device, but for OAK-D SR
        the lens is well-centered and the cx/cy offset is sub-pixel — this
        matches the convention used by the grabette-data rgbd-for-slam pipeline.
        """
        import depthai as dai
        try:
            calib = device.readCalibration()
            w, h = resolution
            intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, w, h)
            imu_extr = calib.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_B, True)
            baseline_m = calib.getBaselineDistance(
                dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_B
            ) / 100.0
            return {
                "width": w,
                "height": h,
                "fx": intr[0][0],
                "fy": intr[1][1],
                "cx": intr[0][2],
                "cy": intr[1][2],
                "baseline": baseline_m,
                "imu_to_cam": imu_extr,
            }
        except Exception as e:
            logger.warning("Could not extract OAK-D offline calib: %s", e)
            return {}

    # ------------------------------------------------------ recording on/off

    def wait_until_ready(
        self, timeout: float = 5.0, min_depth_coverage: float = 0.05,
    ) -> bool:
        """Block until the OAK-D is producing valid frames, or until timeout.

        "Valid" means the IMU is streaming AND the depth map has converged
        past cold-boot warmup (more than `min_depth_coverage` of pixels are
        non-zero). The first frames after init_device() are autoexposure /
        stereo warmup and are unusable for SLAM, so callers gate the recording
        clock on this.

        Returns True if ready within the timeout, False on timeout (the caller
        proceeds anyway — a late-but-recording capture beats a hung one).
        """
        if not self._initialized:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            depth = self._latest_depth
            if self._latest_accel is not None and depth is not None:
                if float((depth > 0).mean()) >= min_depth_coverage:
                    return True
            time.sleep(0.02)
        logger.warning("OAK-D not ready after %.1fs — starting capture anyway", timeout)
        return False

    def start_recording(self, output_dir: Path) -> None:
        if not self._initialized:
            raise RuntimeError("OakdCapture not initialized. Call init_device() first.")
        if self._recording:
            raise RuntimeError("OakdCapture already recording")
        if not self.sync.is_started:
            raise RuntimeError("SyncManager must be started before OakdCapture")

        self._output_dir = Path(output_dir).absolute()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_depth:
            (self._output_dir / "oakd_depth").mkdir(parents=True, exist_ok=True)

        self._left_h264_path = self._output_dir / "oakd_left.h264"
        self._right_h264_path = self._output_dir / "oakd_right.h264"

        if self._calibration_json:
            (self._output_dir / "oakd_calib.json").write_text(
                json.dumps(self._calibration_json, indent=2)
            )
        if self._calib_offline:
            (self._output_dir / "oakd_calib_offline.json").write_text(
                json.dumps(self._calib_offline, indent=2)
            )
        # Copy mask alongside the episode so downstream SLAM can mask mp4 frames
        # (we can't mask H.264-encoded streams in-flight).
        if self._mask is not None:
            try:
                shutil.copyfile(_MASK_PATH, self._output_dir / "oak_mask.png")
            except OSError as e:
                logger.warning("Could not copy oak_mask.png to session dir: %s", e)

        # Clear buffers
        self._left_ts.clear()
        self._right_ts.clear()
        self._depth_ts.clear()
        self._imu_samples.clear()
        self._clock_pairs.clear()

        # Open H.264 files under lock so the drainer threads pick them up safely
        with self._files_lock:
            self._left_h264_fp = open(self._left_h264_path, "wb")
            self._right_h264_fp = open(self._right_h264_path, "wb")
            self._recording = True

        logger.info("OakdCapture recording → %s", self._output_dir)

    def stop_recording(self) -> dict:
        """Stop disk writes, mux H.264 → mp4, dump JSON sidecars. Pipeline keeps running."""
        if not self._recording:
            return {}

        # Flip flag first so drainers stop trying to write
        with self._files_lock:
            self._recording = False
            for fp in (self._left_h264_fp, self._right_h264_fp):
                try:
                    if fp:
                        fp.flush()
                        fp.close()
                except Exception:
                    pass
            self._left_h264_fp = None
            self._right_h264_fp = None

        # Mux raw .h264 → .mp4 with actual fps inferred from timestamps
        for h264_path, ts_buffer, name in [
            (self._left_h264_path, self._left_ts, "left"),
            (self._right_h264_path, self._right_ts, "right"),
        ]:
            self._mux_h264_to_mp4(h264_path, ts_buffer, name)

        # Sidecars
        if self._output_dir:
            (self._output_dir / "oakd_left_timestamps.json").write_text(
                json.dumps({"samples": self._left_ts})
            )
            (self._output_dir / "oakd_right_timestamps.json").write_text(
                json.dumps({"samples": self._right_ts})
            )
            if self.enable_depth:
                (self._output_dir / "oakd_depth_timestamps.json").write_text(
                    json.dumps({"samples": self._depth_ts})
                )
            (self._output_dir / "oakd_imu.json").write_text(
                json.dumps({"samples": self._imu_samples})
            )
            (self._output_dir / "oakd_clock_pairs.json").write_text(
                json.dumps({"pairs": self._clock_pairs})
            )

        stats = {
            "left_frames": len(self._left_ts),
            "right_frames": len(self._right_ts),
            "depth_frames": len(self._depth_ts) if self.enable_depth else None,
            "imu_samples": len(self._imu_samples),
        }
        logger.info("OakdCapture recording stopped: %s", stats)
        return stats

    # ---------------------------------------------------------------- writers

    def _writer_loop_video(self, q, name: str, fp_attr: str, ts_buffer: list[dict]) -> None:
        """Always pull from queue. Append to .h264 file only when recording.

        Recording is gated to begin on the first I-frame: the warm pipeline means
        start_recording() usually lands mid-GOP, and ffmpeg's mux drops the leading
        inter-frames before the first IDR as undecodable — which would leave the
        mp4 shorter than its timestamps and misaligned with depth. Starting the
        .h264 (and its timestamps) on a keyframe keeps mp4 frame count == timestamps.
        """
        import depthai as dai
        IFRAME = dai.EncodedFrame.FrameType.I
        n = 0
        was_recording = False
        seen_keyframe = False
        skipped = 0
        while True:
            try:
                if not q.has():
                    if self._stop_event.is_set():
                        break
                    time.sleep(0.001)
                    continue
                pkt = q.tryGet()
            except Exception:
                break
            if pkt is None:
                continue

            # Reset the keyframe gate on each recording-start edge.
            recording = self._recording
            if recording and not was_recording:
                seen_keyframe = False
                skipped = 0
            was_recording = recording

            # Always drain, but only record if recording.
            if not recording:
                continue

            # Wait for the first I-frame before writing anything. Safety net: if
            # no keyframe shows within ~2 GOPs (e.g. getFrameType() unreliable),
            # start anyway — an mp4 with some leading loss beats recording nothing.
            if not seen_keyframe:
                if pkt.getFrameType() != IFRAME and skipped < 2 * self.keyframe_every:
                    skipped += 1
                    continue
                seen_keyframe = True
                if skipped:
                    logger.info("oakd %s: skipped %d pre-keyframe packet(s) at record start",
                                name, skipped)

            host_ms = self.sync.get_timestamp_ms()
            seq = pkt.getSequenceNum()
            device_us = _device_us(pkt.getTimestampDevice())

            if not self._clock_pairs:
                self._clock_pairs.append({
                    "stream": name, "seq": int(seq),
                    "device_us": device_us, "host_ms": host_ms,
                })

            with self._files_lock:
                fp = getattr(self, fp_attr, None)
                if fp is not None and self._recording:
                    fp.write(pkt.getData())
                    ts_buffer.append({
                        "seq": int(seq),
                        "device_us": device_us,
                        "host_ms": host_ms,
                    })
                    n += 1
        logger.info("oakd %s writer: %d packets recorded", name, n)

    def _writer_loop_depth(self) -> None:
        """Always pull depth; cache latest for live view; PNG to disk when recording."""
        n = 0
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, self.depth_png_compression]
        warned_shape = False
        while True:
            try:
                if not self._depth_q.has():
                    if self._stop_event.is_set():
                        break
                    time.sleep(0.001)
                    continue
                frame = self._depth_q.tryGet()
            except Exception:
                break
            if frame is None:
                continue

            depth = frame.getCvFrame()
            if self._mask is not None:
                if depth.shape == self._mask.shape:
                    # Zero out pixels where mask==0 (the grabette body region).
                    # Multiplying by a uint8 boolean is faster than bitwise_and
                    # for uint16 depth and avoids dtype promotion.
                    depth = depth * (self._mask > 0).astype(depth.dtype)
                elif not warned_shape:
                    logger.warning(
                        "Depth shape %s != mask shape %s — mask skipped. "
                        "Check StereoDepth.setOutputSize / subpixel settings.",
                        depth.shape, self._mask.shape,
                    )
                    warned_shape = True
            # Cache latest frame for live preview (atomic reference swap)
            self._latest_depth = depth

            if not self._recording:
                continue

            host_ms = self.sync.get_timestamp_ms()
            seq = frame.getSequenceNum()
            device_us = _device_us(frame.getTimestampDevice())

            cv2.imwrite(
                str(self._output_dir / "oakd_depth" / f"{seq:08d}.png"),
                depth, png_params,
            )
            self._depth_ts.append({
                "seq": int(seq), "device_us": device_us, "host_ms": host_ms,
            })
            n += 1
        logger.info("oakd depth writer: %d frames recorded", n)

    def _writer_loop_imu(self) -> None:
        n_acc = n_gyr = n_rot = 0
        while True:
            try:
                if not self._imu_q.has():
                    if self._stop_event.is_set():
                        break
                    time.sleep(0.001)
                    continue
                msg = self._imu_q.tryGet()
            except Exception:
                break
            if msg is None:
                continue
            try:
                # time.time() (not sync) because the pipeline runs continuously,
                # but SyncManager only ticks during a capture session.
                live_ms = time.time() * 1000.0
                for packet in msg.packets:
                    if hasattr(packet, "acceleroMeter") and packet.acceleroMeter:
                        a = packet.acceleroMeter
                        self._latest_accel = (live_ms, (a.x, a.y, a.z))
                    if hasattr(packet, "gyroscope") and packet.gyroscope:
                        g = packet.gyroscope
                        self._latest_gyro = (live_ms, (g.x, g.y, g.z))

                if not self._recording:
                    continue

                host_ms = self.sync.get_timestamp_ms()
                for packet in msg.packets:
                    if hasattr(packet, "acceleroMeter") and packet.acceleroMeter:
                        a = packet.acceleroMeter
                        self._imu_samples.append({
                            "kind": "accel",
                            "device_us": _device_us(a.getTimestampDevice()),
                            "host_ms": host_ms,
                            "value": [a.x, a.y, a.z],
                        })
                        n_acc += 1
                    if hasattr(packet, "gyroscope") and packet.gyroscope:
                        g = packet.gyroscope
                        self._imu_samples.append({
                            "kind": "gyro",
                            "device_us": _device_us(g.getTimestampDevice()),
                            "host_ms": host_ms,
                            "value": [g.x, g.y, g.z],
                        })
                        n_gyr += 1
                    if hasattr(packet, "rotationVector") and packet.rotationVector:
                        r = packet.rotationVector
                        self._imu_samples.append({
                            "kind": "rotation",
                            "device_us": _device_us(r.getTimestampDevice()),
                            "host_ms": host_ms,
                            "value": [r.i, r.j, r.k, r.real],
                            "accuracy": getattr(r, "rotationVectorAccuracy", None),
                        })
                        n_rot += 1
            except Exception as e:
                logger.debug("oakd imu writer error: %s", e)
        logger.info("oakd imu recorded: accel=%d gyro=%d rotation=%d", n_acc, n_gyr, n_rot)

    # ------------------------------------------------------------- live view

    def get_latest_imu(self) -> dict | None:
        """Latest accel + gyro sample for live dashboard.

        Returns None until both first accel and first gyro packets have arrived.
        Timestamp is host-side time.time()*1000 at packet arrival, since the
        SyncManager only ticks during a capture session.
        """
        acc = self._latest_accel
        gyr = self._latest_gyro
        if acc is None or gyr is None:
            return None
        return {
            "timestamp_ms": max(acc[0], gyr[0]),
            "accel": acc[1],
            "gyro": gyr[1],
        }

    def get_depth_jpeg(self, quality: int = 80) -> bytes | None:
        """Return latest depth as colorized JPEG (turbo colormap, 0.2-3m).

        Returns None if no depth frame has arrived yet.

        The frame is rotated 180° for display only — recorded depth on disk
        keeps the pipeline orientation so calibration intrinsics stay valid.
        """
        depth = self._latest_depth  # atomic read
        if depth is None:
            return None
        d_min, d_max = self.PREVIEW_DEPTH_MIN_MM, self.PREVIEW_DEPTH_MAX_MM
        mask = (depth >= d_min) & (depth <= d_max)
        d_clip = np.clip(depth, d_min, d_max).astype(np.float32)
        # Close = bright
        d_norm = (255.0 * (d_max - d_clip) / (d_max - d_min)).astype(np.uint8)
        d_norm[~mask] = 0
        colorized = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
        colorized = cv2.rotate(colorized, cv2.ROTATE_180)
        ok, buf = cv2.imencode(".jpg", colorized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    # ------------------------------------------------------------- shutdown

    def _mux_h264_to_mp4(self, h264_path: Path, ts_buffer: list[dict], name: str) -> None:
        if h264_path is None or not h264_path.exists() or h264_path.stat().st_size == 0:
            logger.warning("oakd %s: no h264 data, skipping mux", name)
            return
        mp4_path = h264_path.with_suffix(".mp4")
        actual_fps = float(self.fps)
        if len(ts_buffer) >= 2:
            duration_us = ts_buffer[-1]["device_us"] - ts_buffer[0]["device_us"]
            if duration_us > 0:
                actual_fps = (len(ts_buffer) - 1) / (duration_us / 1_000_000.0)
        cmd = [
            "ffmpeg", "-y", "-fflags", "+genpts",
            "-r", f"{actual_fps:.3f}", "-i", str(h264_path),
            "-c", "copy", "-video_track_timescale", "90000",
            str(mp4_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            h264_path.unlink()
        except OSError:
            pass
        if result.returncode != 0:
            logger.error("ffmpeg muxing failed for %s: %s", name, result.stderr[-300:])

    def shutdown(self) -> None:
        """Stop the pipeline and exit all drainer threads."""
        if not self._initialized:
            return
        if self._recording:
            try:
                self.stop_recording()
            except Exception:
                pass

        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5.0)

        try:
            self._pipeline.stop()
            self._pipeline.wait()
        except Exception as e:
            logger.warning("pipeline stop error: %s", e)

        self._initialized = False
        logger.info("OakdCapture shut down")

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def imu_sample_count(self) -> int:
        """Live count of IMU samples appended during the current recording.
        Reset to 0 between recordings."""
        return len(self._imu_samples)
