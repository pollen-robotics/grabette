"""OAK-D SR capture using depthai v3.

v3: stereo mono (CAM_B + CAM_C) → on-device H.264 encoder → host writers
    + StereoDepth (uint16 depth, ROBOTICS preset) → uint16 PNG sequence
    + BNO086 IMU (raw accel/gyro + rotation vector)

The H.264 elementary streams are written to .h264 files during capture, then
muxed to .mp4 on stop() so the result is a proper MP4 container. Depth is
saved as a uint16 PNG sequence (no codec preserves uint16 well). Per-frame
timestamps live in JSON sidecars (the mp4 fps metadata alone isn't enough
for accurate offline sync).

Output layout per episode:
    oakd_left.mp4               H.264 mono left, 1280x800
    oakd_right.mp4              H.264 mono right, 1280x800
    oakd_left_timestamps.json   {"samples": [{"seq", "device_us", "host_ms"}]}
    oakd_right_timestamps.json  same shape
    oakd_depth/<seq>.png        uint16 depth maps, 1280x800, mm units
    oakd_depth_timestamps.json  per-frame device/host timestamps
    oakd_imu.json               accel + gyro + rotation_vector samples
    oakd_calib.json             factory intrinsics + extrinsics
    oakd_clock_pairs.json       first device_us ↔ host_ms pair per stream
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path

import cv2

from .sync import SyncManager

logger = logging.getLogger(__name__)


def _device_us(ts) -> int:
    """Convert depthai's timedelta-style timestamp to integer microseconds."""
    return int(ts.total_seconds() * 1_000_000)


class OakdCapture:
    """Captures stereo mono (H.264) + IMU from OAK-D SR over USB3.

    The pipeline runs on the OAK-D's RVC2; the host pulls H.264 packets and
    IMU samples into queues, and writer threads drain them to disk.
    """

    DEFAULT_FPS = 30
    DEFAULT_RESOLUTION = (1280, 800)
    DEFAULT_DEPTH_RESOLUTION = (640, 400)  # Half-res depth keeps RVC2 from bottlenecking
    DEFAULT_IMU_HZ = 200
    DEFAULT_BITRATE_BPS = 8_000_000   # 8 Mbps per camera; near-lossless for SLAM
    DEFAULT_KEYFRAME_EVERY = 30       # 1 I-frame every N frames
    DEFAULT_DEPTH_PNG_COMPRESSION = 1  # uint16 PNG compression (0-9); 1 = fast

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

        # In-memory buffers populated by writer threads
        self._left_ts: list[dict] = []
        self._right_ts: list[dict] = []
        self._depth_ts: list[dict] = []
        self._imu_samples: list[dict] = []
        self._clock_pairs: list[dict] = []

        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

        self._calibration_json: dict | None = None

    # ------------------------------------------------------------------ init

    def init_device(self) -> None:
        """Connect briefly to read calibration, then build the pipeline."""
        import depthai as dai

        logger.info("Connecting to OAK-D for calibration read...")
        with dai.Device() as device:
            self._device_id = device.getDeviceId()
            self._product_name = device.getProductName()
            self._usb_speed = str(device.getUsbSpeed())
            self._imu_type = str(device.getConnectedIMU())
            self._calibration_json = self._dump_calibration(device)
            logger.info(
                "OAK-D ready: %s id=%s usb=%s imu=%s",
                self._product_name, self._device_id, self._usb_speed, self._imu_type,
            )

        # --- Pipeline ---
        self._pipeline = dai.Pipeline()

        # Left camera + H.264 encoder
        camB = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        leftRaw = camB.requestOutput(
            self.resolution, type=dai.ImgFrame.Type.NV12, fps=self.fps,
        )
        leftEnc = self._pipeline.create(dai.node.VideoEncoder).build(
            input=leftRaw,
            bitrate=self.bitrate_bps,
            frameRate=float(self.fps),
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
            keyframeFrequency=self.keyframe_every,
        )
        leftEnc.setNumBFrames(0)  # no reordering — preserves stream order in mp4
        self._left_q = leftEnc.out.createOutputQueue(maxSize=32, blocking=False)

        # Right camera + H.264 encoder
        camC = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        rightRaw = camC.requestOutput(
            self.resolution, type=dai.ImgFrame.Type.NV12, fps=self.fps,
        )
        rightEnc = self._pipeline.create(dai.node.VideoEncoder).build(
            input=rightRaw,
            bitrate=self.bitrate_bps,
            frameRate=float(self.fps),
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
            keyframeFrequency=self.keyframe_every,
        )
        rightEnc.setNumBFrames(0)
        self._right_q = rightEnc.out.createOutputQueue(maxSize=32, blocking=False)

        # StereoDepth: request SEPARATE lower-resolution camera outputs to
        # feed it. Stereo matching cost scales with input pixels — feeding
        # 1280x800 inputs maxes the RVC2 stereo matcher and backpressures
        # the cameras (limiting H.264 to ~18fps). A 640x400 stereo input
        # is plenty for SLAM depth and runs comfortably at 30fps.
        # PresetMode.ROBOTICS is tuned for manipulation/close-range use.
        if self.enable_depth:
            leftStereoIn = camB.requestOutput(
                self.depth_resolution, type=dai.ImgFrame.Type.GRAY8, fps=self.fps,
            )
            rightStereoIn = camC.requestOutput(
                self.depth_resolution, type=dai.ImgFrame.Type.GRAY8, fps=self.fps,
            )
            stereo = self._pipeline.create(dai.node.StereoDepth).build(
                left=leftStereoIn,
                right=rightStereoIn,
                presetMode=dai.node.StereoDepth.PresetMode.ROBOTICS,
            )
            stereo.setOutputSize(*self.depth_resolution)
            self._depth_q = stereo.depth.createOutputQueue(maxSize=8, blocking=False)

        # IMU (BNO086 on this device — 9-axis with onboard fusion)
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

    @staticmethod
    def _dump_calibration(device) -> dict:
        """Dump factory calibration to a JSON-serializable dict."""
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

    # ----------------------------------------------------------------- start

    def start_recording(self, output_dir: Path) -> None:
        if self._recording:
            raise RuntimeError("OakdCapture already recording")
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call init_device() first.")
        if not self.sync.is_started:
            raise RuntimeError("SyncManager must be started before OakdCapture")

        self._output_dir = Path(output_dir).absolute()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_depth:
            (self._output_dir / "oakd_depth").mkdir(parents=True, exist_ok=True)

        # Stream files (raw H.264 elementary streams; muxed to mp4 on stop)
        self._left_h264 = self._output_dir / "oakd_left.h264"
        self._right_h264 = self._output_dir / "oakd_right.h264"

        if self._calibration_json:
            (self._output_dir / "oakd_calib.json").write_text(
                json.dumps(self._calibration_json, indent=2)
            )

        # Clear buffers
        self._left_ts.clear()
        self._right_ts.clear()
        self._depth_ts.clear()
        self._imu_samples.clear()
        self._clock_pairs.clear()
        self._stop_event.clear()

        self._pipeline.start()
        self._recording = True

        self._threads = [
            threading.Thread(
                target=self._writer_loop_video,
                args=(self._left_q, "left", self._left_h264, self._left_ts),
                daemon=True,
            ),
            threading.Thread(
                target=self._writer_loop_video,
                args=(self._right_q, "right", self._right_h264, self._right_ts),
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

        logger.info("OakdCapture started → %s", self._output_dir)

    # -------------------------------------------------------------- writers

    def _writer_loop_video(
        self, q, name: str, h264_path: Path, ts_buffer: list[dict],
    ) -> None:
        """Pull H.264 packets from the encoder queue, append to .h264 file."""
        n = 0
        with open(h264_path, "wb") as f:
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

                host_ms = self.sync.get_timestamp_ms()
                seq = pkt.getSequenceNum()
                device_us = _device_us(pkt.getTimestampDevice())

                if not self._clock_pairs:
                    self._clock_pairs.append({
                        "stream": name,
                        "seq": int(seq),
                        "device_us": device_us,
                        "host_ms": host_ms,
                    })

                f.write(pkt.getData())
                ts_buffer.append({
                    "seq": int(seq),
                    "device_us": device_us,
                    "host_ms": host_ms,
                })
                n += 1
        logger.info("oakd %s writer: %d packets", name, n)

    def _writer_loop_depth(self) -> None:
        """Pull depth frames (uint16) from queue, save as PNG sequence."""
        out_dir = self._output_dir / "oakd_depth"
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, self.depth_png_compression]
        n = 0
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

            host_ms = self.sync.get_timestamp_ms()
            seq = frame.getSequenceNum()
            device_us = _device_us(frame.getTimestampDevice())

            # uint16 depth (mm). cv2 preserves bit depth when filename is .png.
            depth = frame.getCvFrame()
            cv2.imwrite(
                str(out_dir / f"{seq:08d}.png"), depth, png_params,
            )
            self._depth_ts.append({
                "seq": int(seq),
                "device_us": device_us,
                "host_ms": host_ms,
            })
            n += 1
        logger.info("oakd depth writer: %d frames", n)

    def _writer_loop_imu(self) -> None:
        """Pull IMU batches from queue, flatten into per-sample dicts."""
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
            try:
                if msg is None:
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
        logger.info("oakd imu: accel=%d gyro=%d rotation=%d", n_acc, n_gyr, n_rot)

    # ------------------------------------------------------------------ stop

    def stop(self) -> dict:
        """Stop streaming, flush writers, mux H.264 → mp4, dump JSON sidecars."""
        if not self._recording:
            return {}

        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5.0)

        try:
            self._pipeline.stop()
            self._pipeline.wait()
        except Exception as e:
            logger.warning("pipeline stop error: %s", e)

        self._recording = False

        # Mux raw .h264 streams into proper .mp4 containers.
        # Compute actual fps from timestamps for the muxer.
        for h264_path, ts_buffer, name in [
            (self._left_h264, self._left_ts, "left"),
            (self._right_h264, self._right_ts, "right"),
        ]:
            self._mux_h264_to_mp4(h264_path, ts_buffer, name)

        # JSON sidecars
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
        logger.info("OakdCapture stopped: %s", stats)
        return stats

    def _mux_h264_to_mp4(
        self, h264_path: Path, ts_buffer: list[dict], name: str,
    ) -> None:
        """Mux raw H.264 elementary stream into .mp4. Uses ffmpeg, like VideoCapture."""
        if not h264_path.exists() or h264_path.stat().st_size == 0:
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

    @property
    def is_recording(self) -> bool:
        return self._recording
