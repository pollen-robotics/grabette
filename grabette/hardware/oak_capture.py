"""OAK-D capture: rectified left + depth + IMU → PNG + CSV files."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

WIDTH  = 640
HEIGHT = 400
FPS    = 30
IMU_RATE = 200


class OakCapture:
    """Records rect_left + depth + IMU from an OAK-D device in a daemon thread.

    Output layout inside session_dir/oak/:
        frames/{:06d}.png   — GRAY8 rectified left
        depth/{:06d}.png    — CV_16UC1 depth in mm
        timestamps.csv      — idx,timestamp_ns
        imu_acc.csv         — timestamp_ns,ax,ay,az
        imu_gyro.csv        — timestamp_ns,wx,wy,wz
        calib_offline.json  — intrinsics + IMU extrinsics for offline_vslam
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._pipeline = None          # set inside thread, guarded by _lock
        self._pipeline_started = False # True only after pipeline.start() returns
        self._lock = threading.Lock()
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Public API — matches the pattern of BMI088Capture / AngleCapture
    # ------------------------------------------------------------------

    def start_capture(self, session_dir: Path) -> None:
        oak_dir = session_dir / "oak"
        (oak_dir / "frames").mkdir(parents=True, exist_ok=True)
        (oak_dir / "depth").mkdir(exist_ok=True)
        self._frame_count = 0
        self._pipeline_started = False

        self._thread = threading.Thread(
            target=self._run, args=(oak_dir,), daemon=True, name="oak-capture"
        )
        self._thread.start()
        logger.info("OakCapture started → %s", oak_dir)

    def stop(self) -> int:
        """Stop the pipeline and wait for the thread to finish. Returns frame count."""
        with self._lock:
            pipeline = self._pipeline
            started = self._pipeline_started
        if pipeline is not None and started:
            pipeline.stop()
        if self._thread is not None:
            self._thread.join(timeout=10)
        logger.info("OakCapture stopped — %d frames", self._frame_count)
        return self._frame_count

    # ------------------------------------------------------------------
    # Capture thread
    # ------------------------------------------------------------------

    def _run(self, oak_dir: Path) -> None:
        try:
            import cv2
            import depthai as dai
        except ImportError as e:
            logger.error("OAK dependency not installed (%s) — OAK capture disabled", e)
            return

        try:
            with dai.Pipeline() as pipeline:
                with self._lock:
                    self._pipeline = pipeline

                # --- Stereo cameras ---
                left = pipeline.create(dai.node.Camera).build(
                    dai.CameraBoardSocket.CAM_B, sensorFps=FPS)
                right = pipeline.create(dai.node.Camera).build(
                    dai.CameraBoardSocket.CAM_C, sensorFps=FPS)

                # --- StereoDepth (same config as record_vslam.py) ---
                stereo = pipeline.create(dai.node.StereoDepth)
                stereo.setExtendedDisparity(False)
                stereo.setLeftRightCheck(True)
                stereo.setRectifyEdgeFillColor(0)
                stereo.enableDistortionCorrection(True)
                stereo.initialConfig.setLeftRightCheckThreshold(10)
                stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
                left.requestOutput((WIDTH, HEIGHT)).link(stereo.left)
                right.requestOutput((WIDTH, HEIGHT)).link(stereo.right)

                # --- OAK built-in IMU ---
                imu = pipeline.create(dai.node.IMU)
                imu.enableIMUSensor(
                    [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
                    IMU_RATE,
                )
                imu.setBatchReportThreshold(1)
                imu.setMaxBatchReports(10)

                # --- Output queues ---
                q_rect  = stereo.rectifiedLeft.createOutputQueue(maxSize=8,  blocking=False)
                q_depth = stereo.depth.createOutputQueue(maxSize=8,  blocking=False)
                q_imu   = imu.out.createOutputQueue(maxSize=50, blocking=False)

                pipeline.start()
                with self._lock:
                    self._pipeline_started = True

                # --- Calibration ---
                calib      = pipeline.getDefaultDevice().readCalibration()
                intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, WIDTH, HEIGHT)
                imu_extr   = calib.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_B, True)
                baseline   = calib.getBaselineDistance(
                    dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_B) / 100.0

                (oak_dir / "calib_offline.json").write_text(json.dumps({
                    "width":      WIDTH,
                    "height":     HEIGHT,
                    "fx":         intrinsics[0][0],
                    "fy":         intrinsics[1][1],
                    "cx":         intrinsics[0][2],
                    "cy":         intrinsics[1][2],
                    "baseline":   baseline,
                    "imu_to_cam": imu_extr,
                }, indent=2))

                # --- CSV writers ---
                with (
                    open(oak_dir / "timestamps.csv", "w") as ts_f,
                    open(oak_dir / "imu_acc.csv",    "w") as acc_f,
                    open(oak_dir / "imu_gyro.csv",   "w") as gyro_f,
                ):
                    ts_f.write("idx,timestamp_ns\n")
                    acc_f.write("timestamp_ns,ax,ay,az\n")
                    gyro_f.write("timestamp_ns,wx,wy,wz\n")

                    pending_rect:  dict = {}
                    pending_depth: dict = {}
                    frame_idx = 0

                    while pipeline.isRunning():
                        rect  = q_rect.tryGet()
                        if rect  is not None:
                            pending_rect[rect.getSequenceNum()]   = rect
                        depth = q_depth.tryGet()
                        if depth is not None:
                            pending_depth[depth.getSequenceNum()] = depth

                        for seq in sorted(set(pending_rect) & set(pending_depth)):
                            r = pending_rect.pop(seq)
                            d = pending_depth.pop(seq)
                            stamp_ns = int(r.getTimestampDevice().total_seconds() * 1e9)

                            cv2.imwrite(
                                str(oak_dir / "frames" / f"{frame_idx:06d}.png"),
                                r.getCvFrame(),
                                [cv2.IMWRITE_PNG_COMPRESSION, 0],
                            )
                            cv2.imwrite(
                                str(oak_dir / "depth" / f"{frame_idx:06d}.png"),
                                d.getCvFrame(),
                                [cv2.IMWRITE_PNG_COMPRESSION, 0],
                            )
                            ts_f.write(f"{frame_idx},{stamp_ns}\n")
                            frame_idx += 1

                        # Drop unmatched frames older than 10 frames
                        cutoff = frame_idx - 10
                        pending_rect  = {k: v for k, v in pending_rect.items()  if k > cutoff}
                        pending_depth = {k: v for k, v in pending_depth.items() if k > cutoff}

                        imu_data = q_imu.tryGet()
                        if imu_data is not None:
                            for pkt in imu_data.packets:
                                acc  = pkt.acceleroMeter
                                gyro = pkt.gyroscope
                                acc_f.write(
                                    f"{int(acc.getTimestampDevice().total_seconds()*1e9)}"
                                    f",{acc.x},{acc.y},{acc.z}\n"
                                )
                                gyro_f.write(
                                    f"{int(gyro.getTimestampDevice().total_seconds()*1e9)}"
                                    f",{gyro.x},{gyro.y},{gyro.z}\n"
                                )

                        time.sleep(0.002)

                self._frame_count = frame_idx

        except Exception:
            logger.exception("OakCapture thread crashed")
        finally:
            with self._lock:
                self._pipeline = None
