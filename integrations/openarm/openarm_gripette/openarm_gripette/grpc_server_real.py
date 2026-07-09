# ruff: noqa: N802  # gRPC servicer methods use PascalCase (protobuf convention)
"""gRPC ArmService server for a real OpenArm (7-DOF, no gripper on CAN).

This server exposes the same ArmService API as `openarm_gripette_simu`, but drives
a real Pollen OpenArm via CAN. It uses the Placo `Kinematics` class and URDF from
`openarm_gripette_model` for bit-for-bit FK/IK compatibility with the simulator.

The **Gripette has its own gRPC service** (same API as `gripper.proto`) running
independently. The eval client connects to two separate endpoints:

  - ArmService     → this server (arm control via CAN)
  - GripperService → the Gripette's own service (camera + gripper motors)

So this script implements ONLY the ArmService.

Behavioral notes:
  - SendCartesianDelta: accumulates on an internal target, runs IK, sends to CAN.
    Identical semantics to the simulator.
  - Reset: interpolates smoothly to the home pose (can't teleport a real arm).
    Cube randomization is a no-op — returns dummy cube coords for API compat.
  - GetSuccessStatus: always returns goal_reached=False (no cube tracking here).

Prerequisites: CAN bus setup + firmware-zero calibration — see README.md.

Usage (on the CAN-connected machine):
  uv run python -m openarm_gripette.grpc_server_real \\
      --can_port can0 --side right --arm_port 50052

Then, from the inference machine, point the eval / teleop client at
`--arm_addr <robot-ip>:50052` (see integrations/DiffusionPolicy/README.md).
"""

import argparse
import logging
import threading
import time
from concurrent import futures

import grpc
import numpy as np

# Kinematics + proto stubs from the simulator package (ensures bit-for-bit match)
from openarm_gripette_simu import Kinematics
from openarm_gripette_simu.kinematics import ARM_JOINT_NAMES as KIN_ARM_JOINT_NAMES
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc
from openarm_gripette_simu.rotation import rotation_6d_to_matrix, rotation_matrix_to_6d

from openarm_gripette import OpenArm7Follower, OpenArm7FollowerConfig

logger = logging.getLogger(__name__)

# Default safe home pose (matches simulator START_JOINTS).
HOME_JOINTS_RAD = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0])

# Reset interpolation: move from current to home over this many seconds.
RESET_DURATION_S = 3.0
RESET_HZ = 50

# LeRobot's OpenArm driver uses degrees; simulator/policy use radians.
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi


# ---------------------------------------------------------------------------
# Arm wrapper: unit conversion + joint-name mapping
# ---------------------------------------------------------------------------


class ArmInterface:
    """Adapter between the simulator's joint API (radians, r_arm_* names) and
    LeRobot's OpenArm7Follower (degrees, joint_1..joint_7 names).

    Caches the most recent joint read with a short TTL so that GetArmState and
    SendCartesianDelta called within the same control cycle share a single CAN
    refresh. Without this, every inference step triggers two full refreshes
    (one for client-side state, one for IK), doubling bus load and packet drops.
    """

    def __init__(
        self,
        robot: OpenArm7Follower,
        arm_joint_map: dict[str, str],
        state_cache_ttl_s: float = 0.02,
    ):
        self._robot = robot
        self._arm_joint_map = arm_joint_map  # sim_name -> lerobot_name
        self._lerobot_arm_names = [arm_joint_map[n] for n in KIN_ARM_JOINT_NAMES]
        self._lock = threading.Lock()
        self._state_cache_ttl = state_cache_ttl_s
        self._cached_positions_rad: np.ndarray | None = None
        self._cached_positions_ts: float = 0.0

    def get_positions(self) -> np.ndarray:
        """Read arm joint positions in radians, in simulator order (r_arm_pitch, ...).

        Returns a cached value if the last refresh is within state_cache_ttl_s.
        """
        now = time.monotonic()
        if (
            self._cached_positions_rad is not None
            and (now - self._cached_positions_ts) < self._state_cache_ttl
        ):
            return self._cached_positions_rad.copy()
        with self._lock:
            # Double-checked: another thread may have refreshed while we waited.
            now = time.monotonic()
            if (
                self._cached_positions_rad is not None
                and (now - self._cached_positions_ts) < self._state_cache_ttl
            ):
                return self._cached_positions_rad.copy()
            obs = self._robot.get_observation()
            positions_deg = np.array(
                [obs[f"{name}.pos"] for name in self._lerobot_arm_names], dtype=np.float64
            )
            self._cached_positions_rad = positions_deg * DEG_TO_RAD
            self._cached_positions_ts = time.monotonic()
            return self._cached_positions_rad.copy()

    def send_command_rad(self, joint_angles_rad: np.ndarray):
        """Send arm joint commands (radians, in simulator order)."""
        joint_angles_deg = joint_angles_rad * RAD_TO_DEG
        action = {
            f"{self._lerobot_arm_names[i]}.pos": float(joint_angles_deg[i])
            for i in range(len(joint_angles_rad))
        }
        with self._lock:
            self._robot.send_action(action)

    def set_torque(self, enable: bool):
        """Enable/disable torque on all arm motors (Damiao enable/disable).

        Disable = motors freewheel; the arm falls under gravity.

        The Damiao enable/disable frames are fire-and-forget in LeRobot: a
        motor that misses the frame stays in its previous state and the driver
        only logs it at DEBUG level (observed in practice: joint_7 kept torque
        after a server shutdown). Send TWO full passes with a pause so a
        single lost frame can't silently leave a motor powered.
        """
        with self._lock:
            for _ in range(2):
                if enable:
                    self._robot.bus.enable_torque(num_retry=2)
                else:
                    self._robot.bus.disable_torque(num_retry=2)
                time.sleep(0.05)


# ---------------------------------------------------------------------------
# ArmService — mirrors openarm_gripette_simu/arm_servicer.py behavior
# ---------------------------------------------------------------------------


class ArmServicer(arm_pb2_grpc.ArmServiceServicer):
    """Implements the same delta-target accumulation pattern as the simulator.

    SendCartesianDelta maintains an internal (_target_pos, _target_r6d). Deltas
    accumulate on that target (not on the current FK pose), avoiding drift from
    mechanical tracking errors. After IK, the target is re-synced to the FK of
    the commanded joints so it stays within a reachable neighborhood.

    Joint-space setpoint interpolation: SendCartesianDelta only computes IK and
    writes the target into a slot. A background thread at interp_hz drives the
    motors using an exponential approach toward that slot, filling in the time
    between sparse policy commands with a smooth joint trajectory. This is the
    standard fix for "smooth Cartesian in -> jerky motors out" with stiff MIT
    gains.
    """

    def __init__(
        self,
        arm: ArmInterface,
        kin: Kinematics,
        start_time: float,
        interp_hz: float = 50.0,
        interp_alpha: float = 0.3,
        max_ik_jump_deg: float = 15.0,
        max_ik_jump_violations: int = 2,
        max_target_lead_mm: float = 50.0,
    ):
        self._arm = arm
        self._kin = kin
        self._start_time = start_time
        self._cmd_lock = threading.Lock()

        # Setpoint interpolator state.
        # `_latest_target_joints` is an atomic slot (reference assignment is
        # atomic under the GIL); `_current_cmd_joints` is only touched by the
        # interp thread, so it needs no lock.
        self._interp_hz = interp_hz
        self._interp_alpha = interp_alpha
        self._latest_target_joints: np.ndarray | None = None
        self._current_cmd_joints: np.ndarray | None = None
        self._interp_enabled = True
        self._interp_running = True

        # IK-jump watchdog state. The threshold is per-joint per-step. A
        # singularity-driven branch flip on a 50 Hz policy loop usually shows
        # up as 30°+ on a single joint in one step, so 15° is a comfortable
        # margin between "normal motion" and "abrupt flip" at typical speeds.
        # Set <= 0 to disable.
        self._max_ik_jump_rad = float(np.deg2rad(max_ik_jump_deg))
        self._max_ik_jump_violations = int(max_ik_jump_violations)
        self._ik_jump_violations = 0

        # Target-runaway ("rubber band") guard. The integrator accumulates
        # deltas OPEN-LOOP on the target — right for tracking accuracy, but
        # under CONTACT (arm pressing the table/object, motors stalled by the
        # per-write safety clamp) the target keeps marching while the arm
        # doesn't, so (a) press torque grows unbounded and (b) a later retreat
        # command must first unwind the accumulated excess before the arm
        # physically moves — corrections look ignored. Cap how far the target
        # position may LEAD the measured FK pose. <= 0 disables.
        self._max_target_lead_m = float(max_target_lead_mm) / 1000.0
        self._lead_clamps = 0

        self._sync_target_from_robot()

        self._interp_thread = threading.Thread(target=self._interp_loop, name="ArmInterpLoop", daemon=True)
        self._interp_thread.start()
        logger.info(
            f"Joint interpolator ON: {self._interp_hz:.0f} Hz, alpha={self._interp_alpha:.2f} "
            f"(e-folding time ~{1000.0 / (self._interp_alpha * self._interp_hz):.0f} ms)"
        )

    def stop(self):
        self._interp_running = False
        if self._interp_thread.is_alive():
            self._interp_thread.join(timeout=2.0)

    def _sync_target_from_robot(self):
        """Reset internal target from the current robot FK pose."""
        arm_joints = self._arm.get_positions()
        tf = self._kin.forward(arm_joints)
        self._target_pos = tf[:3, 3].copy()
        self._target_r6d = rotation_matrix_to_6d(tf[:3, :3]).copy()

    def _interp_loop(self):
        """Drive motors at interp_hz with exponential approach toward latest target."""
        period = 1.0 / self._interp_hz
        while self._interp_running:
            tick = time.monotonic()
            if self._interp_enabled:
                target = self._latest_target_joints  # atomic read (ref assignment)
                if target is not None:
                    if self._current_cmd_joints is None:
                        self._current_cmd_joints = self._arm.get_positions()
                    # next = cur + alpha * (target - cur)
                    self._current_cmd_joints = self._current_cmd_joints + self._interp_alpha * (
                        target - self._current_cmd_joints
                    )
                    try:
                        self._arm.send_command_rad(self._current_cmd_joints)
                    except Exception as e:
                        logger.warning(f"Interp motor write failed: {e}")
            elapsed = time.monotonic() - tick
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def SendCartesianDelta(self, request, context):
        try:
            delta_pos = np.array([request.dx, request.dy, request.dz])
            delta_r6d = np.array(request.dr6d)

            if len(delta_r6d) != 6:
                return arm_pb2.ArmCommandResponse(
                    success=False, error=f"dr6d must have 6 values, got {len(delta_r6d)}"
                )

            with self._cmd_lock:
                # Once the IK-jump watchdog latched (or torque is off), the
                # interpolator is disabled and motion is frozen until Reset.
                # Reject outright instead of running IK: solving against the
                # stale pre-latch joint solution just re-trips the watchdog
                # with confusing repeated "would change by N°" errors while
                # the arm isn't moving at all.
                if not self._interp_enabled:
                    return arm_pb2.ArmCommandResponse(
                        success=False,
                        error="motion frozen (IK-jump watchdog latch or torque off) — call Reset to re-home and re-arm",
                    )

                # Camera-LOCAL frame deltas (Stage-6 convention) applied to
                # the INTEGRATOR target. See sim arm_servicer.SendCartesianDelta
                # for the full math explanation. Crucial: deltas are applied
                # to (_target_pos, _target_r6d), not to the FK-read pose, so
                # the commanded trajectory exactly reproduces the dataset
                # trajectory regardless of arm tracking error.
                R_target = rotation_6d_to_matrix(self._target_r6d)

                prev_target_pos = self._target_pos.copy()
                delta_pos_world = R_target @ delta_pos
                self._target_pos = self._target_pos + delta_pos_world

                R_delta = rotation_6d_to_matrix(delta_r6d)
                R_target_new = R_target @ R_delta
                self._target_r6d = rotation_matrix_to_6d(R_target_new).copy()

                # Target-runaway guard, TRIP-ONLY: if this delta would put the
                # target more than the cap ahead of the MEASURED pose (arm in
                # contact / stalled while the open-loop integrator marches on),
                # REJECT the command and roll it back — exactly like the
                # IK-jump watchdog. Never silently modify the target: pulling
                # it toward the measured pose creates a delayed-feedback loop
                # on the reference that corrupts the whole trajectory
                # (measured to drag the arm downward with gravity sag).
                # Deltas that REDUCE the lead (retreats) always pass, so
                # contact stays bounded and corrections act immediately.
                if self._max_target_lead_m > 0:
                    fk_meas = self._kin.forward(self._arm.get_positions())
                    lead_prev = float(np.linalg.norm(prev_target_pos - fk_meas[:3, 3]))
                    lead_new = float(np.linalg.norm(self._target_pos - fk_meas[:3, 3]))
                    if lead_new > self._max_target_lead_m and lead_new > lead_prev:
                        self._lead_clamps += 1
                        if self._lead_clamps <= 3 or self._lead_clamps % 50 == 0:
                            logger.warning(
                                f"Target-lead trip #{self._lead_clamps}: rejecting delta — "
                                f"target would be {lead_new * 1000:.0f} mm ahead of the "
                                f"measured pose (cap {self._max_target_lead_m * 1000:.0f} mm). "
                                f"Arm in CONTACT or not tracking; only lead-reducing "
                                f"commands accepted until it catches up."
                            )
                        self._target_pos = prev_target_pos
                        self._target_r6d = rotation_matrix_to_6d(R_target).copy()
                        return arm_pb2.ArmCommandResponse(
                            success=False,
                            error=f"target lead {lead_new * 1000:.0f}mm > "
                                  f"{self._max_target_lead_m * 1000:.0f}mm cap (contact?)",
                        )

                target_tf = np.eye(4)
                target_tf[:3, :3] = R_target_new
                target_tf[:3, 3] = self._target_pos

                # IK seed: prefer the LAST COMMANDED joint config (continuity with
                # the previous IK solution) over the MEASURED joint config (which
                # lags behind by the interpolator e-folding time, ~67 ms at the
                # 50 Hz / alpha=0.3 default). Seeding from measured joints on real
                # makes Placo's frame-task (position weight 100x orientation) flip
                # the wrist toward whichever yaw value matches the lagged seed,
                # which compounds into visible yaw drift edge-by-edge in
                # cartesian_square. Sim doesn't hit this because MuJoCo position
                # controllers track commanded joints tightly, so measured ≈
                # commanded and the seed is already smooth.
                if self._latest_target_joints is not None:
                    ik_seed = self._latest_target_joints
                else:
                    ik_seed = self._arm.get_positions()
                try:
                    target_joints = self._kin.inverse(target_tf, current_joint_positions=ik_seed)
                except Exception as e:
                    # IK blew up (e.g. QP NaN near a singularity). Roll the
                    # delta back — without this the failed delta stays in the
                    # integrator and the target drifts away from every
                    # subsequent solvable command.
                    self._target_pos = prev_target_pos
                    self._target_r6d = rotation_matrix_to_6d(R_target).copy()
                    logger.error(f"IK solve failed ({e}); delta rolled back.")
                    return arm_pb2.ArmCommandResponse(success=False, error=f"IK failed: {e}")

                # IK-jump watchdog: refuse the update if any joint would change
                # by more than `_max_ik_jump_rad` in this single step. Singular
                # configurations (typical wrist alignment) can make Placo flip
                # to a different IK branch within a single inference step, and
                # the per-joint motor clamp + interpolator only soften the speed
                # of that flip, they don't prevent it. Comparing the new
                # solution to the last accepted one catches the branch flip
                # directly and trips the integrator before it propagates.
                if self._latest_target_joints is not None and self._max_ik_jump_rad > 0:
                    delta_joints = target_joints - self._latest_target_joints
                    max_jump = float(np.max(np.abs(delta_joints)))
                    if max_jump > self._max_ik_jump_rad:
                        bad_idx = int(np.argmax(np.abs(delta_joints)))
                        bad_name = KIN_ARM_JOINT_NAMES[bad_idx]
                        logger.error(
                            f"IK-jump watchdog tripped: joint '{bad_name}' "
                            f"(idx {bad_idx}) would change by "
                            f"{np.rad2deg(delta_joints[bad_idx]):+.1f}° in one step "
                            f"(limit {np.rad2deg(self._max_ik_jump_rad):.1f}°). "
                            f"Rejecting command; integrator NOT updated. "
                            f"Likely singularity branch flip — re-home or "
                            f"raise --max_ik_jump_deg if intentional."
                        )
                        self._ik_jump_violations += 1
                        if self._ik_jump_violations >= self._max_ik_jump_violations:
                            logger.error(
                                f"{self._ik_jump_violations} consecutive "
                                f"IK-jump violations — disabling interpolator "
                                f"and reverting integrator target to last "
                                f"FK pose for safety."
                            )
                            self._interp_enabled = False
                            self._sync_target_from_robot()
                        # Undo this Cartesian delta on the integrator so we don't
                        # leak it into the next step's cumulative target.
                        self._target_pos -= delta_pos_world
                        self._target_r6d = rotation_matrix_to_6d(R_target).copy()
                        return arm_pb2.ArmCommandResponse(
                            success=False,
                            error=f"IK jump on '{bad_name}': "
                                  f"{np.rad2deg(delta_joints[bad_idx]):+.1f}°",
                        )
                    # Healthy step — reset the consecutive-violation counter.
                    self._ik_jump_violations = 0

                # Re-sync the integrator to what the accepted solution ACTUALLY
                # achieves (the class docstring always promised this; it was
                # never implemented). Exact IK -> FK(IK(target)) == target and
                # this is a no-op, so faithful replay is unchanged. Compromised
                # IK (joint limits, singular region, solver tolerance) -> the
                # target stays on the reachable manifold instead of marching
                # open-loop into unreachable space — measured as 70-120 mm of
                # cmd-vs-meas divergence pinned at the trip-guard cap. Uses
                # COMMANDED joints (deterministic), never measured ones — no
                # feedback of tracking error/gravity sag into the reference.
                fk_cmd = self._kin.forward(target_joints)
                self._target_pos = fk_cmd[:3, 3].copy()
                self._target_r6d = rotation_matrix_to_6d(fk_cmd[:3, :3]).copy()

                # Hand off to the interpolator — no direct motor write.
                self._latest_target_joints = target_joints.copy()

            return arm_pb2.ArmCommandResponse(success=True)

        except Exception as e:
            logger.exception("SendCartesianDelta failed")
            return arm_pb2.ArmCommandResponse(success=False, error=str(e))

    def GetArmState(self, request, context):
        """Return the current arm state (FK of measured joints)."""
        arm_joints = self._arm.get_positions()
        tf = self._kin.forward(arm_joints)
        pos = tf[:3, 3]
        r6d = rotation_matrix_to_6d(tf[:3, :3])
        return arm_pb2.ArmState(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            r6d=r6d.tolist(),
            joint_positions=arm_joints.tolist(),
        )

    def Reset(self, request, context):
        """Move the arm smoothly to the home (or specified) joint configuration.

        Pauses the setpoint interpolator for the duration of the linear ramp,
        then resyncs its state to the final pose so it resumes cleanly.

        Cube randomization is a no-op (no physical cube); dummy cube coords are
        returned for API compatibility with the simulator.
        """
        try:
            if len(request.joint_positions) == 7:
                target_joints = np.array(request.joint_positions, dtype=np.float64)
            else:
                target_joints = HOME_JOINTS_RAD.copy()

            self._interp_enabled = False
            try:
                with self._cmd_lock:
                    start_joints = self._arm.get_positions()

                    num_steps = int(RESET_DURATION_S * RESET_HZ)
                    dt = 1.0 / RESET_HZ

                    logger.info(
                        f"Reset: interpolating over {RESET_DURATION_S}s "
                        f"from {start_joints.round(3).tolist()} to {target_joints.round(3).tolist()}"
                    )

                    for i in range(1, num_steps + 1):
                        alpha = i / num_steps
                        interp = start_joints * (1 - alpha) + target_joints * alpha
                        self._arm.send_command_rad(interp)
                        time.sleep(dt)

                    # Resync interp state so it picks up from here without a jump.
                    self._current_cmd_joints = target_joints.copy()
                    self._latest_target_joints = target_joints.copy()
                    self._sync_target_from_robot()
                    # Re-homing un-latches the IK-jump watchdog: fresh count.
                    self._ik_jump_violations = 0
            finally:
                self._interp_enabled = True

            return arm_pb2.ResetResponse(
                success=True,
                cube_x=0.0,
                cube_y=0.0,
                cube_z=0.0,
                error="cube randomization skipped (real robot)",
            )

        except Exception as e:
            logger.exception("Reset failed")
            return arm_pb2.ResetResponse(success=False, error=str(e))

    def GetSuccessStatus(self, request, context):
        """No cube tracking on a real robot — always returns goal_reached=False."""
        return arm_pb2.SuccessStatusResponse(goal_reached=False, cube_displacement=0.0)

    def SetTorque(self, request, context):
        """Enable/disable motor torque on the whole arm.

        Disable: pauses the setpoint interpolator FIRST (so it stops streaming
        MIT commands), clears its targets, then sends the Damiao disable — the
        motors freewheel and the arm FALLS under gravity.

        Enable: re-enables torque, resyncs the Cartesian integrator to the
        current (hanging) pose, and leaves the interpolator with no target —
        the arm stays limp-but-enabled until the next Reset / delta command,
        which then starts from the measured pose (no jump to a stale target).
        """
        try:
            with self._cmd_lock:
                if request.enable:
                    self._arm.set_torque(True)
                    self._latest_target_joints = None
                    self._current_cmd_joints = None
                    self._sync_target_from_robot()
                    self._ik_jump_violations = 0
                    self._interp_enabled = True
                    logger.info("Torque ENABLED — arm holds nothing until the next command (Reset to home).")
                else:
                    self._interp_enabled = False
                    self._latest_target_joints = None
                    self._current_cmd_joints = None
                    self._arm.set_torque(False)
                    logger.warning("Torque DISABLED — motors freewheeling, arm falls under gravity.")
            return arm_pb2.ArmCommandResponse(success=True)
        except Exception as e:
            logger.exception("SetTorque failed")
            return arm_pb2.ArmCommandResponse(success=False, error=str(e))

    def Ping(self, request, context):
        uptime = time.monotonic() - self._start_time
        return arm_pb2.ArmPingResponse(status="ok", uptime_seconds=uptime)


# ---------------------------------------------------------------------------
# CLI + server setup
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="gRPC ArmService for real OpenArm (7-DOF, no gripper)")
    p.add_argument("--can_port", type=str, default="can0", help="CAN interface name")
    p.add_argument("--side", type=str, default="right", choices=["left", "right"])
    p.add_argument(
        "--ik_orientation_weight",
        type=float,
        default=10.0,
        help="Placo frame-task orientation weight. Default 10.0 on real (sim "
        "uses 1.0). 100:1 position-priority leaks rotation into wrist_yaw "
        "when translating image-right or image-down at the standard home pose "
        "— observable in cartesian_square as a visible camera yaw on edges "
        "perpendicular to the optical axis. Bumping to 10 keeps orientation "
        "held at the cost of <1 mm position error per step. Try 50+ only if "
        "10 still shows yaw drift; trades more position accuracy.",
    )
    p.add_argument(
        "--ik_position_weight",
        type=float,
        default=100.0,
        help="Placo frame-task position weight. Default 100.0 (same as sim).",
    )
    p.add_argument(
        "--max_relative_target",
        type=float,
        default=8.0,
        help="Max per-step joint motion in degrees (safety limit). Each command "
        "then does an extra CAN sync_read to check current position — costs one "
        "full bus transaction. Pass a large value (e.g. 180) to effectively "
        "disable the clamp while still paying the read cost, or remove the "
        "`max_relative_target` kwarg in the config to skip it entirely.",
    )
    p.add_argument("--arm_port", type=int, default=50052, help="gRPC listen port")
    p.add_argument(
        "--kp_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to all MIT position_kp values at startup. "
        "Useful for taming a stiff controller without editing the config: "
        "0.5 halves all kp (softer tracking, smoother under sparse commands), "
        "2.0 doubles them (stiffer).",
    )
    p.add_argument(
        "--kd_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to all MIT position_kd values at startup. "
        "Typically scale kd ~ sqrt(kp_scale) to preserve damping ratio; in "
        "practice leave at 1.0 first and tune from there.",
    )
    p.add_argument(
        "--interp_hz",
        type=float,
        default=50.0,
        help="Joint-space setpoint interpolator rate (Hz). The interpolator "
        "thread sends MIT commands at this rate, filling in the time between "
        "sparse Cartesian delta RPCs with a smooth joint trajectory.",
    )
    p.add_argument(
        "--interp_alpha",
        type=float,
        default=0.3,
        help="Exponential approach rate per interp tick: next = cur + alpha * (target - cur). "
        "Smaller = smoother + more lag (e-folding time = 1 / (alpha * interp_hz)). "
        "Typical range 0.1 (heavy smoothing, ~100ms lag at 50Hz) to 0.5 (light smoothing, ~40ms lag).",
    )
    p.add_argument(
        "--max_ik_jump_deg",
        type=float,
        default=15.0,
        help="IK-jump watchdog: max per-joint change between two consecutive "
        "Placo IK solutions, in degrees. If any joint exceeds this in one step, "
        "the Cartesian delta is REJECTED (integrator rolled back). This is the "
        "specific guard against singularity-driven branch flips that the OOD "
        "Cartesian watchdog and per-step motor clamp can't catch. "
        "Set <= 0 to disable. Typical: 10–20°. Default: 15.",
    )
    p.add_argument(
        "--max_ik_jump_violations",
        type=int,
        default=2,
        help="After N consecutive IK-jump rejections, disable the interpolator "
        "and re-sync the integrator to the current FK pose. Forces the operator "
        "to home before further motion. Default: 2.",
    )
    p.add_argument(
        "--max_target_lead_mm",
        type=float,
        default=80.0,
        help="Contact guard (TRIP-ONLY): reject any delta that would put the "
        "integrator TARGET more than this far (mm) ahead of the MEASURED pose "
        "AND increase the lead. Bounds press force under contact without ever "
        "modifying the reference (a continuous pull-back corrupts the "
        "trajectory). Lead-reducing deltas (retreats) always pass. Costs one "
        "extra bus read per Cartesian command. <= 0 disables. Default: 80 "
        "(above normal full-speed tracking lag).",
    )
    p.add_argument(
        "--arm_joint_map",
        type=str,
        nargs="+",
        default=[
            "r_arm_pitch=joint_1",
            "r_arm_roll=joint_2",
            "r_arm_yaw=joint_3",
            "r_elbow=joint_4",
            "r_wrist_yaw=joint_5",
            "r_wrist_roll=joint_6",
            "r_wrist_pitch=joint_7",
        ],
        help="Map simulator joint names to LeRobot motor names (format: sim_name=lerobot_name)",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    arm_joint_map = dict(item.split("=") for item in args.arm_joint_map)
    logger.info(f"Arm joint map (sim → lerobot): {arm_joint_map}")

    for sim_name in KIN_ARM_JOINT_NAMES:
        if sim_name not in arm_joint_map:
            raise ValueError(f"Missing mapping for simulator joint '{sim_name}' in --arm_joint_map")

    # ---- Robot setup (arm only, no gripper on CAN) ----
    logger.info(f"Connecting to OpenArm on {args.can_port}, side={args.side}")
    robot_config = OpenArm7FollowerConfig(
        port=args.can_port,
        side=args.side,
        can_interface="socketcan",
        max_relative_target=args.max_relative_target,
        cameras={},  # no cameras on this driver — camera comes via Gripette's gRPC
    )
    if args.kp_scale != 1.0:
        robot_config.position_kp = [v * args.kp_scale for v in robot_config.position_kp]
        logger.info(
            f"Scaled MIT position_kp by {args.kp_scale}: {[round(v, 2) for v in robot_config.position_kp]}"
        )
    if args.kd_scale != 1.0:
        robot_config.position_kd = [v * args.kd_scale for v in robot_config.position_kd]
        logger.info(
            f"Scaled MIT position_kd by {args.kd_scale}: {[round(v, 2) for v in robot_config.position_kd]}"
        )
    robot = OpenArm7Follower(robot_config)
    # calibrate=False: the server never modifies calibration. If the firmware
    # zeros or the calibration file need updating, run `lerobot-calibrate`
    # explicitly before starting the server.
    robot.connect(calibrate=False)
    logger.info("Robot connected (7-DOF arm, gripper is external)")

    arm_iface = ArmInterface(robot, arm_joint_map)

    # ---- Kinematics (same URDF as simulator, but with stricter orientation
    #      weight on real to keep the wrist locked through optical-axis
    #      translation; see --ik_orientation_weight in CLI help) ----
    logger.info(
        f"Loading kinematics (placo + URDF from openarm_gripette_model), "
        f"frame-task weights pos={args.ik_position_weight} orient={args.ik_orientation_weight}"
    )
    kin = Kinematics(
        position_weight=args.ik_position_weight,
        orientation_weight=args.ik_orientation_weight,
    )
    logger.info("Kinematics loaded")

    # ---- gRPC server ----
    start_time = time.monotonic()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = ArmServicer(
        arm_iface,
        kin,
        start_time,
        interp_hz=args.interp_hz,
        interp_alpha=args.interp_alpha,
        max_ik_jump_deg=args.max_ik_jump_deg,
        max_ik_jump_violations=args.max_ik_jump_violations,
        max_target_lead_mm=args.max_target_lead_mm,
    )
    arm_pb2_grpc.add_ArmServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{args.arm_port}")
    server.start()
    logger.info(f"ArmService listening on port {args.arm_port}")
    logger.info("Press Ctrl+C to stop")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.stop(grace=2.0)
        servicer.stop()
        # Belt and braces: explicit double-pass torque disable BEFORE
        # disconnect. robot.disconnect() also disables, but with single
        # fire-and-forget frames per motor — one lost frame leaves that motor
        # powered and holding (seen on joint_7/wrist_pitch). The arm FALLS.
        try:
            arm_iface.set_torque(False)
            logger.info("Torque disabled on all motors (double-pass).")
        except Exception as e:
            logger.error(f"Explicit torque disable failed: {e} — check the arm, some motors may still be powered!")
        robot.disconnect()
        logger.info("Robot disconnected. Goodbye.")


if __name__ == "__main__":
    main()
