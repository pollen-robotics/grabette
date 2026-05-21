"""Live BasaltVIO teleop mode for OAK-D SR.

Builds a minimal OAK pipeline (stereo + IMU + BasaltVIO) that runs
continuously and produces camera-local pose deltas. Designed for low Pi
CPU load and ~20 Hz pose output rate; suitable for live teleoperation.

Conventions match the offline pipeline (see grabette-data/docker/oak_vslam):
- Camera local frame: OpenCV optical (X right, Y down, Z forward).
- Delta: T_prev⁻¹ · T_curr — camera-local frame, per LeRobot §10.3.
- World is BasaltVIO's internal frame at start (NOT gravity-aligned by
  default). Position drifts ~few cm over 30 s static; rotation drift is
  small. For delta-based teleop the absolute world frame doesn't matter.

Lifecycle mirrors OakdCapture's:
    init_device() → build pipeline (no start)
    start()       → start pipeline + drainer thread
    stop()        → stop drainer + pipeline
    shutdown()    → full cleanup

Live access is via properties (latest_pose, latest_delta) — daemon
consumers can poll at any rate. No callback API for now; pollable
properties are sufficient for the planned WebSocket stream.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import depthai as dai
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

WIDTH, HEIGHT = 640, 400
FPS = 30
IMU_HZ = 200

# Right-multiply each Basalt pose's rotation by this to re-express the
# camera's local basis in OpenCV optical (X right, Y down, Z forward).
# Basalt emits FRD (X forward, Y right, Z down); we want RDF.
#   RDF +X (right)   = FRD +Y → (0, 1, 0)
#   RDF +Y (down)    = FRD +Z → (0, 0, 1)
#   RDF +Z (forward) = FRD +X → (1, 0, 0)
BASALT_TO_OPTICAL = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=np.float64)


@dataclass
class Pose:
    """6-DoF camera pose, camera frame = OpenCV optical."""
    t_host: float                  # seconds since start()
    translation: np.ndarray        # (3,) world-frame position
    quaternion: np.ndarray         # (4,) [qx, qy, qz, qw] camera in optical


@dataclass
class Delta:
    """Frame-to-frame delta, in the camera's previous local frame."""
    t_host: float
    dx: float
    dy: float
    dz: float
    dqx: float
    dqy: float
    dqz: float
    dqw: float


class OakdTeleop:
    """Live BasaltVIO-based teleop SLAM for OAK-D SR.

    Owns the OAK device exclusively while active; the daemon must stop any
    other OakdCapture/recording pipeline before calling init_device() here.
    """

    def __init__(self, fps: int = FPS, imu_hz: int = IMU_HZ) -> None:
        self.fps = fps
        self.imu_hz = imu_hz
        self._pipeline: dai.Pipeline | None = None
        self._pose_q = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # State
        self._initialized = False
        self._latest_pose: Pose | None = None
        self._latest_delta: Delta | None = None
        self._prev_pose_mat: np.ndarray | None = None
        self._t_start: float | None = None
        self._n_poses = 0
        self._pose_arrival_times: list[float] = []

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def init_device(self) -> None:
        """Build the OAK pipeline (does NOT start it)."""
        if self._initialized:
            return
        logger.info("OakdTeleop: building pipeline")

        self._pipeline = dai.Pipeline()

        camB = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        camC = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        leftIn = camB.requestOutput(
            (WIDTH, HEIGHT), type=dai.ImgFrame.Type.GRAY8, fps=self.fps,
        )
        rightIn = camC.requestOutput(
            (WIDTH, HEIGHT), type=dai.ImgFrame.Type.GRAY8, fps=self.fps,
        )

        imu = self._pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(
            [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
            self.imu_hz,
        )
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)

        vio = self._pipeline.create(dai.node.BasaltVIO)
        # Critical: tell Basalt the IMU rate. Without this, the IMU integration
        # is timestep-blind and the trajectory looks like "pure IMU drift".
        vio.setImuUpdateRate(self.imu_hz)
        leftIn.link(vio.left)
        rightIn.link(vio.right)
        imu.out.link(vio.imu)

        self._pose_q = vio.transform.createOutputQueue(maxSize=8, blocking=False)
        self._initialized = True
        logger.info("OakdTeleop: pipeline built")

    def start(self) -> None:
        """Start the pipeline and the drainer thread."""
        if not self._initialized:
            raise RuntimeError("OakdTeleop.init_device() must be called first")
        if self._thread is not None and self._thread.is_alive():
            logger.warning("OakdTeleop: already running")
            return

        # Reset state for a clean start
        self._stop_event.clear()
        self._t_start = time.monotonic()
        self._n_poses = 0
        self._pose_arrival_times.clear()
        self._latest_pose = None
        self._latest_delta = None
        self._prev_pose_mat = None

        logger.info("OakdTeleop: starting pipeline")
        self._pipeline.start()
        self._thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="oakd-teleop",
        )
        self._thread.start()
        logger.info("OakdTeleop: running")

    def stop(self) -> None:
        """Stop the drainer thread and pipeline. Does NOT teardown the device."""
        if self._thread is None or not self._thread.is_alive():
            return
        logger.info("OakdTeleop: stopping")
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        try:
            self._pipeline.stop()
        except Exception as e:
            logger.warning("OakdTeleop: pipeline stop error: %s", e)
        logger.info("OakdTeleop: stopped (%d poses processed)", self._n_poses)

    def shutdown(self) -> None:
        """Fully tear down: stop, then release the pipeline."""
        self.stop()
        self._pipeline = None
        self._pose_q = None
        self._initialized = False

    # ── Live accessors ─────────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def latest_pose(self) -> Pose | None:
        """Most recent absolute pose (camera in BasaltVIO world)."""
        return self._latest_pose

    @property
    def latest_delta(self) -> Delta | None:
        """Most recent camera-local delta (per LeRobot §10.3 convention)."""
        return self._latest_delta

    def stats(self) -> dict:
        """Real-time framerate and pose-count stats."""
        if len(self._pose_arrival_times) < 2:
            return {"n_poses": self._n_poses, "mean_hz": 0.0,
                    "p95_inter_pose_ms": 0.0}
        intervals = np.diff(self._pose_arrival_times[-60:])  # last ~3 seconds
        return {
            "n_poses": self._n_poses,
            "mean_hz": round(1.0 / float(intervals.mean()), 2)
                       if intervals.mean() > 0 else 0.0,
            "p95_inter_pose_ms": round(float(np.percentile(intervals, 95)) * 1000, 2),
        }

    # ── Internals ──────────────────────────────────────────────────────────────

    def _drain_loop(self) -> None:
        """Background thread: pulls poses from OAK, computes deltas, caches."""
        while not self._stop_event.is_set():
            try:
                p = self._pose_q.tryGet()
            except Exception:
                break
            if p is None:
                time.sleep(0.001)
                continue

            t_host = time.monotonic() - (self._t_start or 0.0)
            tr = p.getTranslation()
            tx, ty, tz = tr.x, tr.y, tr.z
            qd = p.getQuaternion()
            qx, qy, qz, qw = qd.qx, qd.qy, qd.qz, qd.qw

            # Build pose matrix, then fix the camera basis FRD → optical.
            T_curr = np.eye(4)
            T_curr[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix() @ BASALT_TO_OPTICAL
            T_curr[:3, 3] = [tx, ty, tz]
            quat_optical = Rotation.from_matrix(T_curr[:3, :3]).as_quat()

            # Camera-local delta = prev⁻¹ · curr
            if self._prev_pose_mat is None:
                d = np.eye(4)
            else:
                d = np.linalg.inv(self._prev_pose_mat) @ T_curr
            dq = Rotation.from_matrix(d[:3, :3]).as_quat()

            self._latest_pose = Pose(
                t_host=t_host,
                translation=np.array([tx, ty, tz]),
                quaternion=np.asarray(quat_optical),
            )
            self._latest_delta = Delta(
                t_host=t_host,
                dx=float(d[0, 3]), dy=float(d[1, 3]), dz=float(d[2, 3]),
                dqx=float(dq[0]), dqy=float(dq[1]),
                dqz=float(dq[2]), dqw=float(dq[3]),
            )
            self._prev_pose_mat = T_curr
            self._n_poses += 1
            self._pose_arrival_times.append(time.monotonic())
