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

from .kinematics import Kinematics, CONTROL_FRAME
from .pedestal import PedestalClearance, PEDESTAL_MARGIN
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

# --- Pedestal collision gate (Route 1) ----------------------------------------
# Before committing an IK solution in SendCartesianDelta, reject (hold last-good)
# any arm config that moves a guarded link within PEDESTAL_MARGIN of the pedestal
# column. The clearance query (box / margin / guarded links) lives in pedestal.py
# and is SHARED with the data-gen IK filter so they enforce the same constraint.

# Singularity guard: near a kinematic singularity the IK produces an exploding
# joint jump for a tiny Cartesian step (the arm "flails"). If a proposed config
# jumps more than this (rad, any single joint) from the current one, hold
# last-good instead of commanding the explosion. ~0.5 rad/command is already well
# past the arm's velocity limit (10 rad/s -> ~0.33 rad at 30Hz), so normal motion
# never trips it; only the explosions do. Tunable: lower if explosions slip
# through, raise if fast legitimate moves get held.
JOINT_JUMP_LIMIT = 0.5

# --- Grasp-point undershoot diagnostic ----------------------------------------
# GRASP_OFFSET_OAKL is where the cube sits in the oak_l (control) frame when
# correctly grasped — the cube position in the gripper-root frame at the recorded
# grasp pose. MUST match grabette_trajectory.GRASP_OFFSET_BODY (the free-floating
# capture used oak_l == grabette_root). Used to log how far a COMMANDED oak_l pose
# would place the cube from where it actually is — the systematic-undershoot probe.
GRASP_OFFSET_OAKL = np.array([0.0027, 0.05123, 0.090])
# Gripper finger-pointing axis in the oak_l frame: oak_l-local +Z, which points
# straight DOWN at a vertical "top grasp" (matches grabette_trajectory's
# GRIPPER_DOWN_QUAT / grip_quat convention — verified to reproduce the dataset's
# sampled tilt exactly). Its angle from world straight-down is the grasp TILT:
# ~0deg = top grasp; the dataset sampled ~10-35deg. Used to log whether the policy
# reproduces the trained tilt variation or collapses to top grasps.
TILT_AXIS_OAKL = np.array([0.0, 0.0, 1.0])

# Success = cube LIFTED by at least this (meters), matching the grasp-and-lift
# task the dataset is collected/trained on. MUST equal
# grabette_trajectory.LIFT_SUCCESS_THRESHOLD so eval and data collection use the
# same definition of success (a mere horizontal nudge must NOT count).
LIFT_SUCCESS_THRESHOLD = 0.05
# Kept as a secondary diagnostic only (how far the cube slid in XY).
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
        self._cube_start_z = CUBE_Z   # lift is measured against this

        # Internal Cartesian target — initialized from current FK
        self._sync_target_from_sim()

        # Pedestal collision gate (see SendCartesianDelta).
        self._setup_pedestal_gate()
        # Grasp-point undershoot diagnostic (per-episode closest approach).
        self._reset_grasp_diag()

    def _setup_pedestal_gate(self):
        """Build the shared pedestal clearance query (disabled if the scene has
        no `pedestal_box` geom)."""
        self._ped = PedestalClearance(self._sim.model)
        self._pedestal_reject_count = 0
        self._pedestal_last_log = 0.0
        self._singularity_reject_count = 0
        self._singularity_last_log = 0.0
        if self._ped.enabled:
            logger.info("Pedestal collision gate ENABLED: %d arm collision geoms, %.0f mm margin.",
                        len(self._ped.arm_geom_ids), PEDESTAL_MARGIN * 1000)
        else:
            logger.warning("No `pedestal_box` geom in scene; pedestal collision gate DISABLED.")

    def _pedestal_clearance(self, arm_joints) -> float:
        """Signed min distance from a guarded arm link to the pedestal box (see
        pedestal.PedestalClearance)."""
        return self._ped.clearance(arm_joints)

    def _reset_grasp_diag(self):
        """Per-episode closest-approach tracking for the grasp-point diagnostic."""
        self._diag_min_gap = np.inf
        self._diag_snapshot = None

    def _update_grasp_diag(self, T_target, arm_joints):
        """Track the closest a COMMANDED oak_l pose comes to actually grasping the
        cube: compare where the command says the cube should be (oak_l target
        applied to GRASP_OFFSET_OAKL) to the real cube. Records the snapshot at the
        episode's closest approach (logged in GetSuccessStatus). Cheap; called per
        command under the lock."""
        if not self._has_cube():
            return
        cube = self._sim.data.body("red_cube").xpos.copy()
        exp_cmd = T_target[:3, :3] @ GRASP_OFFSET_OAKL + T_target[:3, 3]
        gap = float(np.linalg.norm(exp_cmd - cube))
        if gap < self._diag_min_gap:
            self._diag_min_gap = gap
            # Where the ACHIEVED arm pose puts the grasp point (separates a
            # policy/perception shortfall from an IK/tracking shortfall).
            T_ach = self._kin.forward(arm_joints, frame=CONTROL_FRAME)
            exp_ach = T_ach[:3, :3] @ GRASP_OFFSET_OAKL + T_ach[:3, 3]

            # Grasp orientation: tilt of the approach axis from world straight-down
            # (0 = top grasp; dataset ~10-35deg) + its azimuth (heading).
            def _tilt_azim(R):
                v = R @ TILT_AXIS_OAKL
                v = v / (np.linalg.norm(v) + 1e-12)
                tilt = np.degrees(np.arccos(np.clip(-v[2], -1.0, 1.0)))
                azim = np.degrees(np.arctan2(v[1], v[0]))
                return float(tilt), float(azim)
            tilt_cmd, azim_cmd = _tilt_azim(T_target[:3, :3])
            tilt_ach, _ = _tilt_azim(T_ach[:3, :3])

            self._diag_snapshot = {
                "gap_world": (exp_cmd - cube),                       # commanded gap, world
                "gap_local": T_target[:3, :3].T @ (exp_cmd - cube),  # in oak_l frame (z = approach/depth)
                "gap_achieved": float(np.linalg.norm(exp_ach - cube)),
                "oakl_track_mm": float(np.linalg.norm(T_ach[:3, 3] - T_target[:3, 3]) * 1000),
                "tilt_cmd": tilt_cmd, "tilt_ach": tilt_ach, "azim_cmd": azim_cmd,
            }

    def _sync_target_from_sim(self):
        """Initialize the internal target from the current sim state."""
        arm_joints = self._sim.get_arm_positions()
        T = self._kin.forward(arm_joints, frame=CONTROL_FRAME)
        self._target_pos = T[:3, 3].copy()
        self._target_r6d = rotation_matrix_to_6d(T[:3, :3]).copy()
        # Last successfully-COMMANDED arm config — the singularity guard measures
        # the IK jump against this (command-to-command), not the lagging actual
        # arm, so tracking error doesn't masquerade as an explosion.
        self._last_good_joints = arm_joints.copy()

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
        self._cube_start_z = CUBE_Z
        return cube_x, cube_y, CUBE_Z

    def _get_state_from_sim(self):
        """Read actual arm state from the simulation."""
        arm_joints = self._sim.get_arm_positions()
        T = self._kin.forward(arm_joints, frame=CONTROL_FRAME)
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

                # Save the pre-delta target so the pedestal gate can roll back.
                prev_pos = self._target_pos.copy()
                prev_r6d = self._target_r6d.copy()

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
                target_joints = self._kin.inverse(T_target, current_joint_positions=arm_joints,
                                                  frame=CONTROL_FRAME)

                # Singularity guard: a near-singular target makes IK explode into a
                # huge joint jump. Compare to the LAST COMMANDED config (not the
                # lagging actual arm, which would conflate tracking error with an
                # explosion). Hold last-good + roll back the integrator so the arm
                # freezes at the last good config (and doesn't fling itself into the
                # pedestal) instead of flailing; it resumes when the policy commands
                # away from the singular region.
                joint_jump = float(np.max(np.abs(target_joints - self._last_good_joints)))
                if joint_jump > JOINT_JUMP_LIMIT:
                    self._target_pos = prev_pos
                    self._target_r6d = prev_r6d
                    self._singularity_reject_count += 1
                    now = time.monotonic()
                    if now - self._singularity_last_log > 1.0:
                        logger.warning("Singularity guard: holding arm (%d rejections); IK joint "
                                       "jump %.2f rad > %.2f (near-singular target).",
                                       self._singularity_reject_count, joint_jump, JOINT_JUMP_LIMIT)
                        self._singularity_last_log = now
                    return arm_pb2.ArmCommandResponse(success=True)

                # Pedestal collision gate (anti-deadlock): reject a proposed config
                # only if it's within PEDESTAL_MARGIN of the column AND moves
                # CLOSER than the current config. Moves that keep or increase
                # clearance are always allowed — so if the arm ends up inside the
                # margin (reset, a singularity jump, or physical overshoot, since
                # the box proxy is non-physical), it can still command its way out
                # instead of freezing. On reject: HOLD last-good + roll back the
                # integrator target so it doesn't drift deeper.
                proposed_clear = self._pedestal_clearance(target_joints)
                if proposed_clear < PEDESTAL_MARGIN and proposed_clear < self._pedestal_clearance(arm_joints):
                    self._target_pos = prev_pos
                    self._target_r6d = prev_r6d
                    self._pedestal_reject_count += 1
                    now = time.monotonic()
                    if now - self._pedestal_last_log > 1.0:
                        logger.warning("Pedestal gate: holding arm (%d rejections); "
                                       "target moves closer to the column (clearance %.1f mm).",
                                       self._pedestal_reject_count, proposed_clear * 1000.0)
                        self._pedestal_last_log = now
                    return arm_pb2.ArmCommandResponse(success=True)

                self._sim.set_arm_commands(target_joints)
                self._last_good_joints = target_joints.copy()

                # Diagnostic: track how close the COMMANDED pose comes to a grasp.
                self._update_grasp_diag(T_target, arm_joints)

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
                self._cube_start_z = cz
                self._reset_grasp_diag()
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
                self._reset_grasp_diag()

            logger.info(f"Reset (legacy): arm={joints.round(3).tolist()}, cube=[{cx:.3f}, {cy:.3f}]")
            return arm_pb2.ResetResponse(
                success=True, cube_x=cx, cube_y=cy, cube_z=cz,
            )
        except Exception as e:
            logger.exception("Reset failed")
            return arm_pb2.ResetResponse(success=False, error=str(e))

    def GetSuccessStatus(self, request, context):
        """Check if the cube was GRASPED AND LIFTED — the actual task.

        goal_reached = (cube rose by >= LIFT_SUCCESS_THRESHOLD above its start
        z), matching the dataset/collector definition. A mere horizontal nudge
        no longer counts as success. ``cube_displacement`` reports the vertical
        lift (cube_z - start_z) so the client can see how high it got; this RPC
        is instantaneous, so the client should query it at episode end (after
        any settle) to judge a sustained grasp.

        Returns goal_reached=False / displacement=0 if the scene has no cube.
        """
        with self._lock:
            if not self._has_cube():
                return arm_pb2.SuccessStatusResponse(
                    goal_reached=False,
                    cube_displacement=0.0,
                )
            cube_z = float(self._sim.data.body("red_cube").xpos[2])

        lift = cube_z - self._cube_start_z
        goal_reached = lift > LIFT_SUCCESS_THRESHOLD

        # Grasp-point undershoot diagnostic: how close did the COMMANDED oak_l
        # pose come to a correct grasp, and is any shortfall in the command
        # (policy/perception) or in tracking (IK)?
        snap = self._diag_snapshot
        if snap is not None:
            gw = snap["gap_world"] * 1000.0
            gl = snap["gap_local"] * 1000.0
            logger.info(
                "[grasp-diag] closest COMMANDED grasp gap=%.1f mm "
                "(world=[%+.1f %+.1f %+.1f], oak_l-local=[%+.1f %+.1f %+.1f], "
                "local z=approach/depth); ACHIEVED gap=%.1f mm; oak_l tracking err=%.1f mm. "
                "Large COMMANDED gap => policy/perception undershoot; small commanded "
                "but large achieved => IK/tracking.",
                self._diag_min_gap * 1000.0, gw[0], gw[1], gw[2], gl[0], gl[1], gl[2],
                snap["gap_achieved"] * 1000.0, snap["oakl_track_mm"],
            )
            logger.info(
                "[grasp-orient] gripper tilt from vertical: commanded=%.1f deg "
                "simulated=%.1f deg (0=top grasp; dataset sampled ~10-35 deg); "
                "commanded azimuth=%.1f deg. A persistently small commanded tilt "
                "=> the policy collapsed to top grasps despite the trained variation.",
                snap["tilt_cmd"], snap["tilt_ach"], snap["azim_cmd"],
            )

        return arm_pb2.SuccessStatusResponse(
            goal_reached=goal_reached,
            cube_displacement=lift,
        )

    def Ping(self, request, context):
        uptime = time.monotonic() - self._start_time
        return arm_pb2.ArmPingResponse(status="ok", uptime_seconds=uptime)
