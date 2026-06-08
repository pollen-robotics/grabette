"""ArmServicer — gRPC service for delta Cartesian arm control.

Maintains an internal Cartesian target (position + orientation). Delta commands
accumulate on this target, avoiding drift from physics errors. IK solves for
the accumulated target, and MuJoCo tracks the resulting joint commands.

Supports episode reset with cube/arm randomization and success detection.
"""

import logging
import time
import threading
import numpy as np
import mujoco

from .kinematics import Kinematics, GRIPPER_FRAME
from .rotation import rotation_matrix_to_6d, rotation_6d_to_matrix
from .proto import arm_pb2, arm_pb2_grpc

logger = logging.getLogger(__name__)

# Default arm start for randomized reset
START_JOINTS = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0])

# Cube nominal position (matches table_red_cube.xml)
CUBE_NOMINAL_X = 0.40
CUBE_NOMINAL_Y = -0.15
CUBE_Z = 0.415

# Randomization ranges
CUBE_X_NOISE = 0.06
CUBE_Y_NOISE = 0.2
CUBE_YAW_NOISE = np.pi
ARM_JOINT_NOISE = 0.08

# Table bounds (from table_red_cube.xml)
TABLE_X_MIN = 0.165
TABLE_X_MAX = 0.735
TABLE_Y_MIN = -0.285
TABLE_Y_MAX = 0.285

# Success threshold: cube displacement in XY (meters)
CUBE_MOVED_THRESHOLD = 0.005


class ArmServicer(arm_pb2_grpc.ArmServiceServicer):

    def __init__(self, sim, kin: Kinematics, lock: threading.Lock, start_time: float,
                 server=None):
        self._sim = sim
        self._kin = kin
        self._lock = lock
        self._start_time = start_time
        self._rng = np.random.default_rng()
        # Optional back-reference to the SimulationServer. If set, the Reset
        # RPC can delegate to server.reset_episode_random() which samples a
        # cube position + arm home pose from the SAME training distribution
        # the dataset used. Falls back to the legacy local-noise reset if
        # not provided (so tests that construct ArmServicer alone keep working).
        self._server = server

        # Cube initial position for success tracking (set on reset)
        self._cube_start_xy = np.array([CUBE_NOMINAL_X, CUBE_NOMINAL_Y])

        # Internal Cartesian target — initialized from current FK
        self._sync_target_from_sim()

    def _sync_target_from_sim(self):
        """Initialize the internal target from the current sim state."""
        arm_joints = self._sim.get_arm_positions()
        T = self._kin.forward(arm_joints)
        self._target_pos = T[:3, 3].copy()
        self._target_r6d = rotation_matrix_to_6d(T[:3, :3]).copy()

    def _cube_contacts_robot(self):
        """Check if the cube is in contact with any robot geom."""
        for i in range(self._sim.data.ncon):
            c = self._sim.data.contact[i]
            n1 = self._sim.model.geom(c.geom1).name
            n2 = self._sim.model.geom(c.geom2).name
            is_cube = "red_cube" in n1 or "red_cube" in n2
            is_env = ("table" in n1 or "leg" in n1 or "floor" in n1 or
                      "table" in n2 or "leg" in n2 or "floor" in n2)
            if is_cube and not is_env:
                return True
        return False

    def _has_cube(self) -> bool:
        """Return True iff the loaded scene has a `red_cube_joint`."""
        return mujoco.mj_name2id(
            self._sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint"
        ) >= 0

    def _randomize_cube(self):
        """Randomize cube position using MuJoCo collision detection to avoid robot contact."""
        cube_jnt_id = mujoco.mj_name2id(self._sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
        if cube_jnt_id < 0:
            # No cube in this scene — nothing to randomize. Keep cube_start_xy
            # at whatever nominal value it was initialized to so GetSuccessStatus
            # doesn't crash on a missing body lookup either.
            logger.info("Scene has no `red_cube_joint`; skipping cube randomization.")
            return (
                float(self._cube_start_xy[0]),
                float(self._cube_start_xy[1]),
                CUBE_Z,
            )
        cube_qadr = self._sim.model.jnt_qposadr[cube_jnt_id]
        cube_dof_adr = self._sim.model.jnt_dofadr[cube_jnt_id]

        while True:
            cube_x = np.clip(
                CUBE_NOMINAL_X + self._rng.uniform(-CUBE_X_NOISE, CUBE_X_NOISE),
                TABLE_X_MIN + 0.02, TABLE_X_MAX - 0.02,
            )
            cube_y = np.clip(
                CUBE_NOMINAL_Y + self._rng.uniform(-CUBE_Y_NOISE, CUBE_Y_NOISE),
                TABLE_Y_MIN + 0.02, TABLE_Y_MAX - 0.02,
            )
            yaw = self._rng.uniform(-CUBE_YAW_NOISE, CUBE_YAW_NOISE)

            self._sim.data.qpos[cube_qadr:cube_qadr + 3] = [cube_x, cube_y, CUBE_Z]
            self._sim.data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self._sim.data.qvel[cube_dof_adr:cube_dof_adr + 6] = 0
            mujoco.mj_forward(self._sim.model, self._sim.data)

            if not self._cube_contacts_robot():
                break

        self._cube_start_xy = np.array([cube_x, cube_y])
        return cube_x, cube_y, CUBE_Z

    def _get_state_from_sim(self):
        """Read actual arm state from the simulation."""
        arm_joints = self._sim.get_arm_positions()
        T = self._kin.forward(arm_joints)
        pos = T[:3, 3]
        r6d = rotation_matrix_to_6d(T[:3, :3])
        return pos, r6d, arm_joints

    def SendCartesianDelta(self, request, context):
        try:
            delta_pos = np.array([request.dx, request.dy, request.dz])
            delta_r6d = np.array(request.dr6d)

            if len(delta_r6d) != 6:
                return arm_pb2.ArmCommandResponse(
                    success=False, error=f"dr6d must have 6 values, got {len(delta_r6d)}"
                )

            with self._lock:
                # Camera-LOCAL frame deltas (Stage-6 convention) applied to
                # the INTEGRATOR target, not to the FK-current pose. The
                # incoming (dx, dy, dz) is a position offset expressed in
                # the integrator's current rotation basis, and (dr6d) is
                # the 6D form of R_delta such that
                #
                #     R_target_next = R_target_now @ R_delta
                #     pos_target_next = pos_target_now + R_target_now @ delta_pos
                #
                # Applying the delta to the INTEGRATOR (the cumulative
                # commanded pose) and NOT to the FK-current pose is
                # essential when the IK can leave orientation error in
                # the actual arm pose. Our Placo solver is configured
                # with position weight 100x orientation weight (to keep
                # position accurate), so the actual FK rotation can be up
                # to 30° off the commanded rotation. If we applied
                # local-frame deltas through that drifted FK rotation,
                # the position delta direction would also drift — the
                # arm would consistently miss the target by a slowly
                # accumulating error in the +/-X+/-Y plane. The integrator
                # is self-consistent with the dataset's trajectory by
                # construction (it reproduces pose[t+1] = pose[t] + apply(delta_t)
                # exactly given pose[0]), so the arm just has to track the
                # integrator without affecting subsequent commands.
                R_target = rotation_6d_to_matrix(self._target_r6d)

                # Position: target_pos += R_target @ delta_local
                delta_pos_world = R_target @ delta_pos
                self._target_pos = self._target_pos + delta_pos_world

                # Rotation: R_target_new = R_target @ R_delta_local
                R_delta = rotation_6d_to_matrix(delta_r6d)
                R_target_new = R_target @ R_delta
                self._target_r6d = rotation_matrix_to_6d(R_target_new).copy()

                # IK to the new integrator target.
                T_target = np.eye(4)
                T_target[:3, :3] = R_target_new
                T_target[:3, 3] = self._target_pos

                arm_joints = self._sim.get_arm_positions()
                target_joints = self._kin.inverse(T_target, current_joint_positions=arm_joints)
                self._sim.set_arm_commands(target_joints)

            return arm_pb2.ArmCommandResponse(success=True)

        except Exception as e:
            logger.exception("Cartesian delta command failed")
            return arm_pb2.ArmCommandResponse(success=False, error=str(e))

    def GetArmState(self, request, context):
        with self._lock:
            pos, r6d, arm_joints = self._get_state_from_sim()

        return arm_pb2.ArmState(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            r6d=r6d.tolist(),
            joint_positions=arm_joints.tolist(),
        )

    def Reset(self, request, context):
        """Reset the episode: teleport arm + randomize cube.

        If a server reference is available AND the request doesn't pin the
        joint positions, sample a cube + home pose from the SAME training
        distribution the dataset was generated with. Otherwise fall back to
        the legacy local-noise reset around START_JOINTS.
        """
        try:
            # Preferred path: training-distribution random reset (matches
            # collect_grasp_dataset.py exactly).
            if self._server is not None and len(request.joint_positions) != 7:
                result = self._server.reset_episode_random()
                if result is None:
                    return arm_pb2.ResetResponse(
                        success=False,
                        error="reset_episode_random failed to find a feasible home pose",
                    )
                cx, cy, cz = result
                self._cube_start_xy = np.array([cx, cy])
                logger.info(f"Reset (training-dist): cube=[{cx:.3f}, {cy:.3f}]")
                return arm_pb2.ResetResponse(success=True, cube_x=cx, cube_y=cy, cube_z=cz)

            # Legacy path: cube via local noise around CUBE_NOMINAL, arm via
            # local noise around START_JOINTS. Used when no server reference,
            # or when the caller pins joints explicitly.
            with self._lock:
                cx, cy, cz = self._randomize_cube()

                if len(request.joint_positions) == 7:
                    joints = np.array(request.joint_positions)
                else:
                    joints = START_JOINTS + self._rng.uniform(
                        -ARM_JOINT_NOISE, ARM_JOINT_NOISE, size=7
                    )

                self._sim.reset_arm(joints)
                self._sim.data.qvel[:] = 0
                mujoco.mj_forward(self._sim.model, self._sim.data)
                self._sync_target_from_sim()

            logger.info(f"Reset (legacy): arm={joints.round(3).tolist()}, cube=[{cx:.3f}, {cy:.3f}]")
            return arm_pb2.ResetResponse(
                success=True, cube_x=cx, cube_y=cy, cube_z=cz,
            )
        except Exception as e:
            logger.exception("Reset failed")
            return arm_pb2.ResetResponse(success=False, error=str(e))

    def GetSuccessStatus(self, request, context):
        """Check if the cube was touched (moved from its initial position).

        Returns goal_reached=False / displacement=0 if the scene has no cube.
        """
        with self._lock:
            if not self._has_cube():
                return arm_pb2.SuccessStatusResponse(
                    goal_reached=False,
                    cube_displacement=0.0,
                )
            cube_xy = self._sim.data.body("red_cube").xpos[:2].copy()

        displacement = float(np.linalg.norm(cube_xy - self._cube_start_xy))
        goal_reached = displacement > CUBE_MOVED_THRESHOLD

        return arm_pb2.SuccessStatusResponse(
            goal_reached=goal_reached,
            cube_displacement=displacement,
        )

    def Ping(self, request, context):
        uptime = time.monotonic() - self._start_time
        return arm_pb2.ArmPingResponse(status="ok", uptime_seconds=uptime)
