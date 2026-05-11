"""OAK-D SR capture using depthai v3.

Minimal v1: stereo mono (CAM_B + CAM_C) + IMU (raw accel/gyro/rotation vector).
Frames saved as PNG sequence per camera; IMU and timestamps as JSON.
No depth and no on-device H.264 encoding yet (deferred to v2).

Output layout per episode:
    oakd_left/<seq>.png         GRAY8, 1280x800
    oakd_right/<seq>.png        GRAY8, 1280x800
    oakd_left_timestamps.json   {"samples": [{"seq", "device_us", "host_ms"}]}
    oakd_right_timestamps.json  same shape
    oakd_imu.json               raw accel + gyro + rotation vector + per-sample timestamps
    oakd_calib.json             factory intrinsics + extrinsics
    oakd_clock_pairs.json       periodic (host_ms, device_us) pairs for offline drift fit
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .sync import SyncManager

logger = logging.getLogger(__name__)


def _device_us(ts) -> int:
    """Convert depthai's timedelta-style timestamp to integer microseconds."""
    return int(ts.total_seconds() * 1_000_000)


class OakdCapture:
    """Captures stereo mono + IMU from OAK-D SR over USB3.

    Pipeline runs on the OAK-D; host pulls frames/IMU into queues and
    writer threads drain them to disk. The device clock and the SyncManager
    monotonic clock are paired periodically for offline alignment.
    """

    DEFAULT_FPS = 30
    DEFAULT_RESOLUTION = (1280, 800)
    DEFAULT_IMU_HZ = 200
    DEFAULT_PNG_COMPRESSION = 1  # 0..9; 1 is fast, files larger

    def __init__(
        self,
        sync_manager: SyncManager,
        fps: int = DEFAULT_FPS,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        imu_rate_hz: int = DEFAULT_IMU_HZ,
        png_compression: int = DEFAULT_PNG_COMPRESSION,
    ) -> None:
        self.sync = sync_manager
        self.fps = fps
        self.resolution = resolution
        self.imu_rate_hz = imu_rate_hz
        self.png_compression = png_compression

        self._pipeline = None
        self._left_q = None
        self._right_q = None
        self._imu_q = None

        self._output_dir: Path | None = None
        self._recording = False

        # In-memory buffers populated by writer threads
        self._left_ts: list[dict] = []
        self._right_ts: list[dict] = []
        self._imu_samples: list[dict] = []
        self._clock_pairs: list[dict] = []

        # Writer threads
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

        # Calibration (captured once at init)
        self._calibration_json: dict | None = None

    # ------------------------------------------------------------------ init

    def init_device(self) -> None:
        """Connect to OAK-D and build the pipeline (no streaming yet)."""
        import depthai as dai

        logger.info("Connecting to OAK-D...")
        # A short-lived Device connection just to read calibration once.
        # The real pipeline below holds its own device connection.
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

        # Build pipeline (doesn't start the device yet — that happens on pipeline.start())
        self._pipeline = dai.Pipeline()

        camB = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        leftOut = camB.requestOutput(
            self.resolution, type=dai.ImgFrame.Type.GRAY8, fps=self.fps,
        )
        self._left_q = leftOut.createOutputQueue(maxSize=8, blocking=False)

        camC = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        rightOut = camC.requestOutput(
            self.resolution, type=dai.ImgFrame.Type.GRAY8, fps=self.fps,
        )
        self._right_q = rightOut.createOutputQueue(maxSize=8, blocking=False)

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
            # depthai v3 returns a dict already; some versions return a JSON string.
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
        (self._output_dir / "oakd_left").mkdir(parents=True, exist_ok=True)
        (self._output_dir / "oakd_right").mkdir(parents=True, exist_ok=True)

        # Write calibration once
        if self._calibration_json:
            (self._output_dir / "oakd_calib.json").write_text(
                json.dumps(self._calibration_json, indent=2)
            )

        # Clear buffers
        self._left_ts.clear()
        self._right_ts.clear()
        self._imu_samples.clear()
        self._clock_pairs.clear()
        self._stop_event.clear()

        # Start pipeline (connects to device, begins streaming)
        self._pipeline.start()
        self._recording = True

        # Launch writer threads
        self._threads = [
            threading.Thread(
                target=self._writer_loop_video,
                args=(self._left_q, "left", self._left_ts),
                daemon=True,
            ),
            threading.Thread(
                target=self._writer_loop_video,
                args=(self._right_q, "right", self._right_ts),
                daemon=True,
            ),
            threading.Thread(
                target=self._writer_loop_imu,
                daemon=True,
            ),
        ]
        for t in self._threads:
            t.start()

        logger.info("OakdCapture started → %s", self._output_dir)

    # -------------------------------------------------------------- writers

    def _writer_loop_video(
        self, q, name: str, ts_buffer: list[dict],
    ) -> None:
        """Pull frames from a queue, write as PNG, record timestamps.

        Exits when stop_event is set AND queue is drained, or when the
        queue is closed (pipeline stop).
        """
        out_dir = self._output_dir / f"oakd_{name}"
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression]
        n = 0
        while True:
            try:
                if not q.has():
                    if self._stop_event.is_set():
                        break
                    time.sleep(0.001)
                    continue
                frame = q.tryGet()
            except Exception:
                # MessageQueue closed (pipeline stopped) — exit cleanly
                break

            if frame is None:
                continue

            host_ms = self.sync.get_timestamp_ms()
            seq = frame.getSequenceNum()
            device_us = _device_us(frame.getTimestampDevice())

            # Save first clock pair (device <-> sync mapping)
            if not self._clock_pairs:
                self._clock_pairs.append({
                    "stream": name,
                    "seq": int(seq),
                    "device_us": device_us,
                    "host_ms": host_ms,
                })

            img = frame.getCvFrame()
            cv2.imwrite(
                str(out_dir / f"{seq:08d}.png"), img, png_params,
            )
            ts_buffer.append({
                "seq": int(seq),
                "device_us": device_us,
                "host_ms": host_ms,
            })
            n += 1
        logger.info("oakd %s writer: %d frames", name, n)

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
                    # Accel
                    if hasattr(packet, "acceleroMeter") and packet.acceleroMeter:
                        a = packet.acceleroMeter
                        self._imu_samples.append({
                            "kind": "accel",
                            "device_us": _device_us(a.getTimestampDevice()),
                            "host_ms": host_ms,
                            "value": [a.x, a.y, a.z],
                        })
                        n_acc += 1
                    # Gyro
                    if hasattr(packet, "gyroscope") and packet.gyroscope:
                        g = packet.gyroscope
                        self._imu_samples.append({
                            "kind": "gyro",
                            "device_us": _device_us(g.getTimestampDevice()),
                            "host_ms": host_ms,
                            "value": [g.x, g.y, g.z],
                        })
                        n_gyr += 1
                    # Rotation vector (quaternion + accuracy)
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
        """Stop streaming, flush writers, dump JSON files."""
        if not self._recording:
            return {}

        # Signal writers to flush remaining queue contents and exit
        self._stop_event.set()
        # Allow a brief moment for the queues to drain
        for t in self._threads:
            t.join(timeout=5.0)

        try:
            self._pipeline.stop()
            self._pipeline.wait()
        except Exception as e:
            logger.warning("pipeline stop error: %s", e)

        self._recording = False

        # Write JSON sidecars
        if self._output_dir:
            (self._output_dir / "oakd_left_timestamps.json").write_text(
                json.dumps({"samples": self._left_ts})
            )
            (self._output_dir / "oakd_right_timestamps.json").write_text(
                json.dumps({"samples": self._right_ts})
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
            "imu_samples": len(self._imu_samples),
        }
        logger.info("OakdCapture stopped: %s", stats)
        return stats

    @property
    def is_recording(self) -> bool:
        return self._recording
