"""Simulation gRPC server.

Physics and camera rendering run in the main thread (with optional viewer).
gRPC servers run in background threads and read cached camera frames.
  - GripperService (port 50051): same API as the real Gripette
  - ArmService (port 50052): delta Cartesian arm control
"""

import logging
import sys
import time
from concurrent import futures
from pathlib import Path
import threading

import grpc
import mujoco
import mujoco.viewer
import numpy as np

from .simulation import Simulation
from .kinematics import Kinematics, CAMERA_FRAME
from .gripper_servicer import GripperServicer
from .arm_servicer import ArmServicer
from .proto import gripper_pb2_grpc, arm_pb2_grpc

# Examples directory holds the training trajectory module. The viewer reset
# shortcuts sample home/cube positions from the SAME distribution
# `collect_grasp_dataset.py` uses, which is what the policy was trained on.
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

# GLFW key codes used by the MuJoCo viewer's key_callback. We avoid
# letter keys that the viewer already binds (A=camera, C=contacts, etc.)
# by using R (free) plus the digit row.
_KEY_R = 82  # 'R' — reset both arm and cube
_KEY_1 = 49  # '1' — reset arm only
_KEY_2 = 50  # '2' — reset cube only

logger = logging.getLogger(__name__)

GRIPPER_PORT = 50051
ARM_PORT = 50052
VIEWER_FPS = 60
CAMERA_FPS = 50  # camera rendering rate, matches real data (50fps)


class SimulationServer:
    """MuJoCo simulation with gRPC interfaces for the Gripette and arm."""

    def __init__(self, scene_xml: str | Path | None = None, initial_arm_joints=None,
                 gripper_hold_open_duration: float = 0.0):
        self._sim = Simulation(scene_xml)
        self._kin = Kinematics()
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        # rng for the reset shortcuts
        self._rng = np.random.default_rng()
        # Hold-open grace period after a reset. Originally added (default
        # 1.5 s) to work around the v3 model getting stuck commanding the
        # gripper closed after a reset, because that training distribution
        # had no closed→open transitions. The v4 model (with release +
        # hover episodes) handles the transition correctly, so the default
        # is now 0 (no override). Pass --gripper-hold-open-duration=1.5
        # at the CLI to re-enable.
        self._gripper_hold_open_until = 0.0
        self._gripper_hold_open_duration = float(gripper_hold_open_duration)
        # Remember the seed (or default ARM_IK_SEED) for IK warm-starts on reset.
        self._initial_arm_joints = (
            np.asarray(initial_arm_joints, dtype=float) if initial_arm_joints is not None
            else np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.5])
        )

        # Cached camera frame — rendered in the main thread, read by gRPC
        self._camera_frame = self._sim.render_camera()

        if initial_arm_joints is not None:
            self._sim.reset_arm(np.asarray(initial_arm_joints, dtype=float))
            logger.info(f"Arm initialized to {list(initial_arm_joints)}")

        # Initialize gripper to OPEN (0 rad). Without this, the position
        # actuator's default ctrl + gravity drag the unactuated fingers
        # toward the closed-bound joint range, and the simulator boots with
        # gripper qpos around (-1.5, -2.1) — i.e. fully CLOSED. This is
        # severely out of distribution for any policy trained on demos that
        # always start with the gripper open: the model conditions on a
        # closed gripper and predicts "lift / hold" actions instead of
        # "approach / close". Match the dataset's start state explicitly.
        self._sim.set_joint_commands(
            np.array([0.0, 0.0]), joint_names=["proximal", "distal"],
        )
        # Also zero the gripper qpos so the very first observation reads
        # "open", before physics has a chance to settle.
        self._sim.reset_joints(
            np.array([0.0, 0.0]), joint_names=["proximal", "distal"],
        )
        logger.info("Gripper initialized to OPEN (proximal=0, distal=0)")

    # ------------------------------------------------------------------
    # Viewer keyboard shortcuts: random reset to a training-distribution pose
    # ------------------------------------------------------------------

    def _get_cube_qadr(self) -> int:
        return self._sim.model.jnt_qposadr[
            mujoco.mj_name2id(self._sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
        ]

    def _get_cube_xy(self) -> tuple[float, float]:
        qadr = self._get_cube_qadr()
        return float(self._sim.data.qpos[qadr]), float(self._sim.data.qpos[qadr + 1])

    def _set_cube_xy(self, cx: float, cy: float):
        qadr = self._get_cube_qadr()
        cube_jnt = mujoco.mj_name2id(self._sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
        qveladr = self._sim.model.jnt_dofadr[cube_jnt]
        # Hard-coded cube z (table top + half cube height); matches grabette_trajectory.CUBE_START_Z.
        cube_z = 0.41
        self._sim.data.qpos[qadr:qadr + 3] = (cx, cy, cube_z)
        self._sim.data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        self._sim.data.qvel[qveladr:qveladr + 6] = 0.0
        mujoco.mj_forward(self._sim.model, self._sim.data)

    def _sample_home_arm_joints(self, cube_xy: tuple[float, float]) -> np.ndarray | None:
        """Sample a feasible home pose around the given cube and IK to arm joints.

        Returns None if no feasible home is found within a few attempts.
        """
        from grabette_trajectory import (  # noqa: E402
            sample_home_pose, body_pose_to_camera_pose, pose_T,
        )
        cx, cy = cube_xy
        for _ in range(50):
            home_xyz, home_quat, _ = sample_home_pose(self._rng, cx, cy)
            cam_xyz, cam_quat = body_pose_to_camera_pose(home_xyz, home_quat)
            T_target = pose_T(cam_xyz, cam_quat)
            joints = self._kin.inverse(
                T_target, current_joint_positions=self._initial_arm_joints.copy(),
                n_iter=300, frame=CAMERA_FRAME,
            )
            T_actual = self._kin.forward(joints, frame=CAMERA_FRAME)
            if np.linalg.norm(T_actual[:3, 3] - cam_xyz) < 0.02:
                return joints
        return None

    def _sample_cube_xy(self) -> tuple[float, float]:
        from grabette_trajectory import CUBE_X_RANGE, CUBE_Y_RANGE  # noqa: E402
        return (float(self._rng.uniform(*CUBE_X_RANGE)),
                float(self._rng.uniform(*CUBE_Y_RANGE)))

    def reset_cube_random(self):
        """Place the cube at a random training-distribution position."""
        with self._lock:
            cx, cy = self._sample_cube_xy()
            self._set_cube_xy(cx, cy)
            logger.info(f"[reset_cube] cube → ({cx:+.3f}, {cy:+.3f})")

    def reset_arm_random(self):
        """Reset the arm to a random training-distribution home pose around
        the CURRENT cube position. Re-opens the gripper. Re-syncs the
        arm-servicer's internal Cartesian target so the next
        SendCartesianDelta starts from the new arm pose, not the stale one."""
        with self._lock:
            cube_xy = self._get_cube_xy()
            joints = self._sample_home_arm_joints(cube_xy)
            if joints is None:
                logger.warning("[reset_arm] no feasible home pose found")
                return
            self._sim.reset_arm(joints)
            self._sim.set_joint_commands(np.array([0.0, 0.0]),
                                          joint_names=["proximal", "distal"])
            self._sim.reset_joints(np.array([0.0, 0.0]),
                                    joint_names=["proximal", "distal"])
            if hasattr(self, "_arm_servicer"):
                self._arm_servicer._sync_target_from_sim()
            self._gripper_hold_open_until = time.monotonic() + self._gripper_hold_open_duration
            logger.info(f"[reset_arm] arm → home for cube ({cube_xy[0]:+.3f}, {cube_xy[1]:+.3f})")

    def reset_episode_random(self) -> tuple[float, float, float] | None:
        """Reset cube + arm to a fresh, mutually consistent training-distribution config.

        Returns ``(cube_x, cube_y, cube_z)`` on success or ``None`` if no
        feasible home pose was found for the sampled cube. The cube is at
        the standard CUBE_START_Z height (= table top + half cube height)."""
        with self._lock:
            for _attempt in range(5):
                cx, cy = self._sample_cube_xy()
                self._set_cube_xy(cx, cy)
                joints = self._sample_home_arm_joints((cx, cy))
                if joints is not None:
                    break
            else:
                logger.warning("[reset_episode] no feasible home pose found in 5 cube draws")
                return None
            self._sim.reset_arm(joints)
            self._sim.set_joint_commands(np.array([0.0, 0.0]),
                                          joint_names=["proximal", "distal"])
            self._sim.reset_joints(np.array([0.0, 0.0]),
                                    joint_names=["proximal", "distal"])
            if hasattr(self, "_arm_servicer"):
                self._arm_servicer._sync_target_from_sim()
            self._gripper_hold_open_until = time.monotonic() + self._gripper_hold_open_duration
            logger.info(f"[reset_episode] cube → ({cx:+.3f}, {cy:+.3f}) + arm to home")
            return (float(cx), float(cy), 0.41)

    def _key_callback(self, keycode: int):
        """MuJoCo viewer key handler — R / 1 / 2 reset the scene."""
        if keycode == _KEY_R:
            self.reset_episode_random()
        elif keycode == _KEY_1:
            self.reset_arm_random()
        elif keycode == _KEY_2:
            self.reset_cube_random()

    def _physics_loop(self):
        """Step physics at the model's timestep rate (background thread)."""
        dt = self._sim.model.opt.timestep
        t_wall = time.perf_counter()
        step = 0
        while self._running:
            with self._lock:
                self._sim.step()
            step += 1
            t_target = t_wall + step * dt
            t_now = time.perf_counter()
            if t_target > t_now:
                time.sleep(t_target - t_now)

    def _start_grpc(self, gripper_port: int, arm_port: int):
        """Start the gRPC servers in background thread pools."""
        self._gripper_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        gripper_pb2_grpc.add_GripperServiceServicer_to_server(
            GripperServicer(self._sim, self, self._lock, self._start_time),
            self._gripper_server,
        )
        self._gripper_server.add_insecure_port(f"0.0.0.0:{gripper_port}")

        self._arm_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        # Keep a reference to the servicer so the keyboard reset shortcuts
        # can re-sync its internal Cartesian target after a manual arm reset.
        # Otherwise the next SendCartesianDelta adds the delta to a stale
        # target and IK snaps the arm back to its old position.
        # The servicer also gets a back-reference to the server so the
        # Reset RPC can call reset_episode_random() — same training-
        # distribution sampling as the keyboard 'R' shortcut.
        self._arm_servicer = ArmServicer(self._sim, self._kin, self._lock, self._start_time, server=self)
        arm_pb2_grpc.add_ArmServiceServicer_to_server(
            self._arm_servicer,
            self._arm_server,
        )
        self._arm_server.add_insecure_port(f"0.0.0.0:{arm_port}")

        self._gripper_server.start()
        self._arm_server.start()
        logger.info(f"Gripper gRPC server on port {gripper_port}")
        logger.info(f"Arm gRPC server on port {arm_port}")

    def get_camera_frame(self):
        """Get the latest cached camera frame (thread-safe)."""
        return self._camera_frame

    def start(self, gripper_port: int = GRIPPER_PORT, arm_port: int = ARM_PORT):
        """Start gRPC servers + physics in background (non-blocking, headless)."""
        self._start_grpc(gripper_port, arm_port)
        self._running = True
        self._physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self._physics_thread.start()
        logger.info("Physics loop running (background).")

    def stop(self):
        """Stop physics and gRPC servers."""
        self._running = False
        if hasattr(self, '_physics_thread'):
            self._physics_thread.join(timeout=2)
        self._gripper_server.stop(grace=2)
        self._arm_server.stop(grace=2)
        logger.info("Server stopped.")

    def run(
        self,
        gripper_port: int = GRIPPER_PORT,
        arm_port: int = ARM_PORT,
        headless: bool = False,
    ):
        """Run the simulation (blocking). Call from the main thread.

        Physics, camera rendering, and viewer sync all happen in the main thread.
        gRPC servers run in background threads and read cached camera frames.
        """
        self._start_grpc(gripper_port, arm_port)

        dt = self._sim.model.opt.timestep
        viewer_interval = max(1, int(1.0 / (VIEWER_FPS * dt)))
        camera_interval = max(1, int(1.0 / (CAMERA_FPS * dt)))

        viewer = None
        if not headless:
            viewer = mujoco.viewer.launch_passive(
                self._sim.model, self._sim.data,
                key_callback=self._key_callback,
                show_left_ui=False, show_right_ui=False,
            )
            logger.info("Viewer launched. Reset shortcuts: R=both, 1=arm only, 2=cube only.")

        logger.info("Physics loop running. Press Ctrl+C to stop.")
        self._running = True
        t_wall = time.perf_counter()
        step = 0

        try:
            while self._running and (viewer is None or viewer.is_running()):
                with self._lock:
                    self._sim.step()

                    # Render camera in the main thread at CAMERA_FPS
                    if step % camera_interval == 0:
                        self._camera_frame = self._sim.render_camera()

                step += 1

                if viewer is not None and step % viewer_interval == 0:
                    viewer.sync()

                t_target = t_wall + step * dt
                t_now = time.perf_counter()
                if t_target > t_now:
                    time.sleep(t_target - t_now)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self._gripper_server.stop(grace=2)
            self._arm_server.stop(grace=2)
            if viewer is not None:
                viewer.close()
            logger.info("Server stopped.")
