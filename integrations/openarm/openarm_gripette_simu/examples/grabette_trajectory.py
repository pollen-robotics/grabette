"""Trajectory helpers for the virtual-Grabette grasp scripts.

Stage 3 upgrade: produces a smoother, more "human-like" approach for both
manual_grasp_test.py and collect_grasp_dataset.py. Highlights:

- Random "human holding pose" home: in front of the cube on the -X side,
  pitched 35-65 deg below horizontal, with mild yaw jitter.
- Smoothstep (cosine ease-in-out) interpolation in position; SLERP in
  orientation. Multi-waypoint approach with an arc-shaped mid-point.
- Quaternions throughout are MuJoCo (w, x, y, z); SLERP is done in
  scipy / LeRobotRotation (x, y, z, w) form internally and converted at
  the boundary.

The mocap is the source of truth for "where the gripper is": writing
data.mocap_pos / data.mocap_quat is what drives the welded grabette body
in scenes/grabette_grasp.xml.

============================================================================
Frame convention (PIPELINE-WIDE — must match the real Grabette / deployment)
============================================================================
Every recorded pose, every IK target, and every action delta is expressed
in the **CAMERA SITE pose** with the position in **WORLD COORDINATES** that
are **Z-up, gravity-aligned**.

  * Position (x, y, z): origin is arbitrary (the SLAM origin in the real
    pipeline; the MuJoCo world origin in sim). Axes are gravity-aligned
    with +Z up. Position component of the recorded pose is the camera
    site's position in this frame.
  * Orientation (rotvec / 6D): the camera site's rotation matrix expressed
    *with respect to the same gravity-aligned world*. NOT relative to the
    camera's own instantaneous frame.

Implications for sim:
  * The MuJoCo mocap pose drives the GRIPPER_BASE body (= grabette_root),
    not the camera site. Recording the mocap pose directly would be wrong.
  * Use `body_pose_to_camera_pose` to turn a body pose into a camera pose
    before:
      - feeding it to Placo IK (which targets the camera frame),
      - writing it into the dataset as `action[:6]`,
      - comparing against another camera-frame pose.
  * The feasibility of an episode is decided by whether the *camera site*
    can reach the planned camera-frame poses. Not the body. Not the gripper
    site. The historical confusion between body / gripper-site / camera
    frames cost us several stages of recalibrated home ranges.

Reference: README.md in lerobot/examples/openarm_gripette/ — "End-effector
position: (x, y, z) in meters, Z-up, gravity-aligned". `eval_on_robot.py`
also asserts "Both are Z-up, gravity-aligned".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- Cube placement (shared with both call sites).
# Originally a 6 cm x 10 cm region. Widened to ~15 cm x 20 cm (Stage 5c,
# Config C) after empirically measuring the arm-feasible envelope: with the
# home pose constrained to the arm-reachable slice (see HOME_*_RANGE below),
# the cube itself can sit over a much larger area without losing IK
# feasibility — and more cube-position diversity is exactly what the policy
# needs to learn.
CUBE_X_RANGE = (0.35, 0.50)
CUBE_Y_RANGE = (-0.22, -0.02)

# --- Train/eval split: a 2-cm strip on the +y edge is held out as the eval
# distribution. Train sees ~89% of the cube area; eval is OOD on cube_y so
# success there is a direct generalisation signal (memorisation can't help).
EVAL_CUBE_X_RANGE = CUBE_X_RANGE
EVAL_CUBE_Y_RANGE = (-0.04, -0.02)
TRAIN_CUBE_X_RANGE = CUBE_X_RANGE
TRAIN_CUBE_Y_RANGE = (CUBE_Y_RANGE[0], EVAL_CUBE_Y_RANGE[0])  # (-0.22, -0.04)
CUBE_HALF_HEIGHT = 0.06
TABLE_TOP_Z = 0.35
CUBE_START_Z = TABLE_TOP_Z + CUBE_HALF_HEIGHT  # 0.41 m


# --- Per-frame counts for the "moving" phases (recorded at 50 fps).
# Used both by the dataset collector and the IK feasibility checker — keep
# them aligned. The static phases (settle/close/hold/retract-final-settle)
# hold the same pose, so they trivially pass IK feasibility if the endpoints
# pass. We only need to densely check the moving segments.
FRAMES_APPROACH = 80
FRAMES_DESCEND = 25
FRAMES_LIFT = 30


# --- Approach / lift geometry ---
APPROACH_HEIGHT = 0.05    # 5 cm above grasp pose for pre-grasp
LIFT_HEIGHT = 0.10        # 10 cm above grasp pose for the lift target
RETRACT_EXTRA = 0.10      # extra 10 cm above home for the retract endpoint

# Stage 4: distance the sentry waypoint sits BACK along the gripper's local
# approach axis (body -Y, away from the cube). For a vertical grasp this is
# directly above the grasp by this amount; for a tilted grasp it sits up-and-
# behind so the final segment is a clean linear push along the approach axis.
SENTRY_OFFSET = 0.10      # 10 cm back along approach axis


# --- "Home pose" sampling region (Stage 5d after the body↔site frame fix) ---
# Position: relative to cube_xy except home_z (above table top).
#
# IMPORTANT history: through Stage 5b/5c we calibrated home ranges against an
# IK call that was using the gripper SITE as the target while we passed the
# BODY pose — so the "feasible envelope" was off by 14 cm in body Y. After
# fixing the frame conversion (see body_pose_to_camera_pose) the actual
# arm-reachable envelope for the BODY is much smaller and located very
# close to the cube: a few cm in front, slightly to the -y side, near table
# height, with the gripper pitched steeply downward. The "human reaching
# from chest height" intuition no longer fits — the arm is short.
#
# These bounds come from a 5000-sample search against the corrected IK; the
# IK filter still acts as a final guard, so they're loose-enough to keep
# diversity but tight-enough to give >=10% IK-filter acceptance.
#
# CRITICAL: these bounds enforce *separation* from the cube. Pre-Stage-5e
# bounds were derived purely from arm IK feasibility and admitted home
# poses where the gripper sits essentially on the cube (home_x_offset=0,
# home_z=5cm above table). A non-trivial fraction of training episodes
# starting at the cube taught the policy to predict "close + lift" from
# any open-gripper start, regardless of cube position.
#
# The arm's reachable envelope unfortunately doesn't allow much separation
# (the workspace is roughly a 15 cm radius around the cube), so we settle
# for ~5 cm minimum horizontal offset + 10 cm minimum above-table — enough
# that every training episode shows a clear approach motion in the
# recorded video, while keeping IK-filter acceptance usable.
HOME_X_OFFSET_RANGE = (-0.20, -0.05)     # 5-20 cm behind the cube in -X
HOME_Y_OFFSET_RANGE = (-0.25, -0.05)     # 5-25 cm in -Y from the cube
HOME_Z_ABOVE_TABLE_RANGE = (0.30, 0.40)  # 0.65-0.75 m world Z. Stage-5g: dropped the 0.40-0.45 upper slice — at home_z ≥ 0.42 the arm operates near joint limits and the policy's residual orientation error during descend is large enough to miss grasps. v4 eval failures clustered there. Visible diversity loss is small; stability gain matters.

# Orientation: gripper finger axis tilted forward-down at this pitch below
# horizontal. Pitch=90 deg -> straight down. Steep angles only.
HOME_PITCH_RANGE_DEG = (40.0, 80.0)
# Yaw is asymmetric — the camera site convention only reaches positive-yaw
# orientations within the arm's workspace.
HOME_YAW_RANGE_DEG = (0.0, +25.0)
HOME_ROLL_RANGE_DEG = (0.0, 0.0)        # roll disabled per spec

# Pre-grasp pose: fingers fully down (pitch=90, yaw=0, roll=0)
PRE_GRASP_PITCH_DEG = 90.0
PRE_GRASP_YAW_DEG = 0.0
PRE_GRASP_ROLL_DEG = 0.0

# Stage 4 grasp orientation sampling (random per-episode).
#  * GRASP_TILT: pitch FROM VERTICAL of the gripper finger axis.
#    0 deg -> straight down (pre-Stage-4 behaviour).
#    90 deg -> fully horizontal (forbidden — never go past 60).
#  * GRASP_AZIMUTH: rotation around world +Z of the tilt direction.
#    0 deg -> tilt towards world +X (toward the cube from the home side).
#    The home is sampled on the cube's -X side, so positive +X tilt means
#    the gripper "leans toward" the cube as a human reaching forward would.
# Diagonal grasp tilt range (off-vertical). Original spec was 30-60 deg, but
# the V-pocket grasp geometry can't sustain >25 deg under gravity during the
# close-and-hold phase — cube slides out the open end. 10-25 deg is the
# pragmatic compromise: visibly diagonal motion vs the prior pure top-down,
# but geometrically close enough to vertical that the V-pocket still holds.
GRASP_TILT_RANGE_DEG = (15.0, 30.0)
# Camera-frame convention: the arm-feasible azimuth half is positive (+15 to
# +40°), opposite of the body/site-frame number we had in Stage 5b. Symmetric
# ±60° sampling collapses to a tight strip on the +y side under IK filter,
# so the sampler is restricted to that strip directly.
GRASP_AZIMUTH_RANGE_DEG = (+15.0, +40.0)


# --- Body-frame offset from grabette_root to the cube center when grasping ---
# (cube sits in the open V-pocket between the open proximal/distal fingers)
GRASP_OFFSET_BODY = np.array([0.0, 0.20, 0.023])


# --- Gripper joint commands (rad) ---
PROXIMAL_OPEN = 0.0
PROXIMAL_CLOSED = -np.pi / 2
DISTAL_OPEN = 0.0
DISTAL_CLOSED = -2.0 * np.pi / 3


# --- Canonical "fingers fully down" mocap quaternion (MuJoCo w,x,y,z).
# A -90 deg rotation around world +X: local +Y -> world -Z (fingers down),
# local +Z -> world +Y (V-pocket points along world +Y), local +X -> +X.
GRIPPER_DOWN_QUAT = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0])


# --- Success threshold ---
LIFT_SUCCESS_THRESHOLD = 0.05


# =============================================================================
# Quaternion helpers (MuJoCo w,x,y,z convention)
# =============================================================================

def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R @ v


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2 in (w, x, y, z) ordering."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a (w, x, y, z) quaternion from an axis-angle rotation."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two unit quaternions in MuJoCo (w, x, y, z) ordering.

    Robust to opposite-hemisphere pairs (flips q1 if dot < 0) and falls back
    to normalized linear interpolation when the angle is near zero.
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    omega = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_o = np.sin(omega)
    s0 = np.sin((1.0 - t) * omega) / sin_o
    s1 = np.sin(t * omega) / sin_o
    return s0 * q0 + s1 * q1


def grip_quat(pitch_rad: float, yaw_rad: float, roll_rad: float = 0.0) -> np.ndarray:
    """Build a (w, x, y, z) MuJoCo quaternion for the gripper at the given
    pitch/yaw/roll, keeping the same axis convention as GRIPPER_DOWN_QUAT.

    Args:
        pitch_rad: angle of the finger axis below horizontal. 90 deg -> fully
            down (= GRIPPER_DOWN_QUAT). 0 deg -> horizontal pointing +X.
        yaw_rad:   rotation around world +Z. 0 -> finger forward = +X.
        roll_rad:  rotation around the finger axis (small, kept at 0 by default).

    Construction (rotations applied to vectors in world frame, post-multiplied):
        R_total = R_z(yaw) * R_y(pitch - 90 deg) * R_x(-90 deg)
                  [yaw   ] * [pitch correction ] * [GRIPPER_DOWN]
        Optional roll is applied as a final rotation around the finger axis.
    """
    base = GRIPPER_DOWN_QUAT  # R_x(-90 deg)
    pitch_corr = quat_axis_angle(np.array([0.0, 1.0, 0.0]), pitch_rad - np.pi / 2)
    yaw_rot = quat_axis_angle(np.array([0.0, 0.0, 1.0]), yaw_rad)

    q = quat_mul(yaw_rot, quat_mul(pitch_corr, base))

    if roll_rad != 0.0:
        # Roll about the current finger axis (world frame): rotate local +Y by q.
        finger_world = quat_apply(q, np.array([0.0, 1.0, 0.0]))
        roll_rot = quat_axis_angle(finger_world, roll_rad)
        q = quat_mul(roll_rot, q)

    # Re-normalise (numerical drift after several products)
    return q / np.linalg.norm(q)


# =============================================================================
# Easing
# =============================================================================

def smoothstep(t: float) -> float:
    """Cubic Hermite ease-in-out, t in [0, 1]. Zero velocity at endpoints."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# =============================================================================
# High-level pose helpers
# =============================================================================

def grasp_pos_for_cube(cube_xyz: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """World-space mocap position so the cube sits in the gripper V-pocket.

    grabette_root must be at cube_xyz - R(quat) * GRASP_OFFSET_BODY so that
    cube_in_body == GRASP_OFFSET_BODY when the gripper is in the OPEN state.
    """
    return cube_xyz - quat_apply(quat, GRASP_OFFSET_BODY)


def sample_home_pose(rng: np.random.Generator, cube_x: float, cube_y: float):
    """Sample a random "human holding" home pose for the Grabette.

    Returns:
        home_xyz: (3,) world position of grabette_root mocap target
        home_quat: (4,) MuJoCo (w, x, y, z) quaternion (forward-down tilt)
        debug:    dict with the sampled scalar params (for logging)
    """
    home_x = cube_x + rng.uniform(*HOME_X_OFFSET_RANGE)
    home_y = cube_y + rng.uniform(*HOME_Y_OFFSET_RANGE)
    home_z = TABLE_TOP_Z + rng.uniform(*HOME_Z_ABOVE_TABLE_RANGE)

    pitch_deg = rng.uniform(*HOME_PITCH_RANGE_DEG)
    yaw_deg = rng.uniform(*HOME_YAW_RANGE_DEG)
    if HOME_ROLL_RANGE_DEG[0] == HOME_ROLL_RANGE_DEG[1]:
        roll_deg = HOME_ROLL_RANGE_DEG[0]
    else:
        roll_deg = rng.uniform(*HOME_ROLL_RANGE_DEG)

    home_xyz = np.array([home_x, home_y, home_z])
    home_quat = grip_quat(np.deg2rad(pitch_deg),
                          np.deg2rad(yaw_deg),
                          np.deg2rad(roll_deg))
    debug = {
        "pitch_deg": float(pitch_deg),
        "yaw_deg": float(yaw_deg),
        "roll_deg": float(roll_deg),
    }
    return home_xyz, home_quat, debug


def pre_grasp_quat() -> np.ndarray:
    """Mocap quat for the pre-grasp / grasp poses (fingers fully down).

    Stage-3-and-earlier helper. Stage 4 callers should use sample_grasp_pose()
    instead, which produces a randomly tilted diagonal grasp.
    """
    return grip_quat(np.deg2rad(PRE_GRASP_PITCH_DEG),
                     np.deg2rad(PRE_GRASP_YAW_DEG),
                     np.deg2rad(PRE_GRASP_ROLL_DEG))


def sample_grasp_pose(rng: np.random.Generator,
                      tilt_range_deg: tuple[float, float] = GRASP_TILT_RANGE_DEG,
                      azimuth_range_deg: tuple[float, float] = GRASP_AZIMUTH_RANGE_DEG):
    """Sample a random diagonal grasp orientation (Stage 4).

    Convention:
        * tilt_deg: pitch FROM VERTICAL of the gripper's finger axis.
            0 -> straight down (vertical, pre-Stage-4). 90 -> fully horizontal.
            Constrained to <= 60 in practice (never pure horizontal).
        * azimuth_deg: rotation around world +Z of the tilt direction.
            0 -> tilt toward world +X (toward the cube from the home side).

    Internally we reuse grip_quat(pitch_below_horizontal, yaw_around_z) where:
        pitch_below_horizontal = 90 - tilt_from_vertical
        yaw_around_z           = azimuth

    so tilt=0 reproduces the pre-Stage-4 vertical grasp (pitch_below_horizontal
    = 90 deg, identical to GRIPPER_DOWN_QUAT). roll stays 0.

    Returns:
        grasp_quat: (4,) MuJoCo (w, x, y, z) quaternion.
        debug: dict with the sampled scalar params for logging.
    """
    tilt_deg = float(rng.uniform(*tilt_range_deg))
    azimuth_deg = float(rng.uniform(*azimuth_range_deg))
    pitch_below_horizontal_deg = 90.0 - tilt_deg
    grasp_quat = grip_quat(np.deg2rad(pitch_below_horizontal_deg),
                            np.deg2rad(azimuth_deg),
                            0.0)
    debug = {
        "grasp_pitch_deg": tilt_deg,
        "grasp_azimuth_deg": azimuth_deg,
    }
    return grasp_quat, debug


def sentry_pose(grasp_xyz: np.ndarray, grasp_quat: np.ndarray,
                offset: float = SENTRY_OFFSET) -> tuple[np.ndarray, np.ndarray]:
    """Pre-grasp "sentry" waypoint sitting BACK along the gripper's approach axis.

    The gripper's "engagement" / approach axis in body frame is +Y (the V-pocket
    opens toward body +Y; cf. GRASP_OFFSET_BODY = (0, 0.20, 0.023)). To stand
    back from the cube along this axis we offset the sentry by body -Y, then
    rotate that offset into world frame via grasp_quat.

    For a vertical grasp this collapses to "directly above the grasp by `offset`
    metres" — same intuition as the Stage-3 sentry. For a tilted grasp it sits
    up-and-behind so the final segment (sentry -> grasp) is a clean linear
    motion along the gripper's approach direction.

    Sentry orientation matches grasp_quat (no further rotation in the final
    segment).
    """
    back_body = np.array([0.0, -offset, 0.0])
    sentry_xyz = grasp_xyz + quat_apply(grasp_quat, back_body)
    return sentry_xyz, grasp_quat.copy()


def mid_approach_pose(home_xyz: np.ndarray, home_quat: np.ndarray,
                       above_grasp: np.ndarray, grasp_quat: np.ndarray,
                       arc_lift: float = 0.0):
    """Build a mid-waypoint between home and pre-grasp for a curved approach.

    Position: midpoint of home and above_grasp, with z bumped UP so the
    trajectory bows upward (a hand naturally lifts a bit before descending
    to the object). The lift is the larger of the requested arc_lift and a
    z-floor that keeps the mid waypoint at least as high as the higher of
    home/above_grasp — this prevents the curve from sagging into the cube.
    Orientation: already at grasp_quat. Doing the rotation entirely in the
    first segment (home -> mid) and keeping the second segment (mid ->
    above_grasp) at a fixed grasp_quat avoids sweeping the gripper geometry
    through the cube column while the body is rotating.
    """
    base_xyz = 0.5 * (home_xyz + above_grasp)
    z_target = max(home_xyz[2], above_grasp[2]) + arc_lift
    mid_xyz = np.array([base_xyz[0], base_xyz[1], z_target])
    mid_quat = grasp_quat.copy()
    return mid_xyz, mid_quat


# =============================================================================
# 4x4 transform helpers (shared by the IK feasibility checker)
# =============================================================================

def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_T(xyz: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform from (xyz, MuJoCo-quat)."""
    T = np.eye(4)
    T[:3, :3] = quat_to_rot_matrix(quat)
    T[:3, 3] = xyz
    return T


# --- Body (mocap) → "camera" site offset.
# The whole pipeline (real Grabette → dataset → training → deployment) uses
# the camera site as the reference frame. The Grabette mocap drives
# `grabette_root` (= gripper_base body), but every recorded pose, every IK
# target and every action delta is expressed in CAMERA-frame coordinates.
# Source: robot.xml site name="camera" pos="0.0689844 0.0662 0.0278102"
# quat="0.5 -0.5 0.5 0.5", relative to gripper_base.
BODY_TO_CAMERA_POS = np.array([0.0689844, 0.0662, 0.0278102])
BODY_TO_CAMERA_QUAT = np.array([0.5, -0.5, 0.5, 0.5])  # MuJoCo (w, x, y, z)


def body_pose_to_camera_pose(body_xyz: np.ndarray, body_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Grabette body (mocap) pose to the corresponding `"camera"` site pose.

    Use this:
      * when feeding body waypoints into Placo IK with frame=CAMERA_FRAME
        (the only frame the existing kinematics solver targets);
      * when recording the dataset: the deployment pipeline expects camera
        poses, not body poses, so `mocap_state_8d` must apply this before
        writing the action.

    Pure rigid-body composition: camera site is at a fixed offset inside
    gripper_base, so camera_world = body_world * T_body_to_camera.
    """
    cam_xyz = body_xyz + quat_apply(body_quat, BODY_TO_CAMERA_POS)
    cam_quat = quat_mul(body_quat, BODY_TO_CAMERA_QUAT)
    return cam_xyz, cam_quat


# =============================================================================
# Episode plan: cube + home + grasp + derived waypoints
# =============================================================================

@dataclass
class EpisodeWaypoints:
    """All the keyframe poses needed to drive a Grabette grasp episode.

    The collector and the IK feasibility checker both consume this so the
    sampling logic lives in exactly one place. Use `sample_episode_waypoints`
    to populate it.
    """
    cube_xyz: np.ndarray         # (3,) cube center at episode start
    home_xyz: np.ndarray         # (3,) Grabette mocap target at episode start
    home_quat: np.ndarray        # (4,) MuJoCo quat at home
    mid_xyz: np.ndarray          # (3,) arc-lifted mid waypoint
    mid_quat: np.ndarray         # (4,) at mid (= grasp_quat after rotation)
    sentry_xyz: np.ndarray       # (3,) back along approach axis from grasp
    sentry_quat: np.ndarray      # (4,) = grasp_quat
    grasp_xyz: np.ndarray        # (3,) Grabette mocap target with cube in pocket
    grasp_quat: np.ndarray       # (4,) tilted-diagonal grasp orientation
    lift_xyz: np.ndarray         # (3,) grasp + (0, 0, lift_height)
    lift_quat: np.ndarray        # (4,) = grasp_quat
    home_dbg: dict               # sampled scalar params from sample_home_pose
    grasp_dbg: dict              # sampled scalar params from sample_grasp_pose


def sample_episode_waypoints(
    rng: np.random.Generator,
    *,
    cube_xyz: np.ndarray | None = None,
    cube_x_range: tuple[float, float] = CUBE_X_RANGE,
    cube_y_range: tuple[float, float] = CUBE_Y_RANGE,
    lift_height: float = LIFT_HEIGHT,
    arc_lift: float = 0.0,
    sentry_offset: float = SENTRY_OFFSET,
    tilt_range_deg: tuple[float, float] = GRASP_TILT_RANGE_DEG,
    azimuth_range_deg: tuple[float, float] = GRASP_AZIMUTH_RANGE_DEG,
) -> EpisodeWaypoints:
    """Sample a complete episode plan from a single rng draw.

    If ``cube_xyz`` is provided, the cube position is fixed; otherwise it is
    drawn from ``cube_x_range`` / ``cube_y_range`` at the standard
    CUBE_START_Z (defaults to the module-level CUBE_X_RANGE / CUBE_Y_RANGE).
    The home pose, the grasp tilt/azimuth and the derived mid/sentry/lift
    waypoints follow from there.

    Used by the dataset collector AND by the rejection sampler — when the
    sampler rejects a plan, calling this again with the same rng produces a
    new candidate. Pass alternative ``cube_*_range`` to define a held-out
    eval region.
    """
    if cube_xyz is None:
        cube_x = float(rng.uniform(*cube_x_range))
        cube_y = float(rng.uniform(*cube_y_range))
        cube_xyz = np.array([cube_x, cube_y, CUBE_START_Z])
    else:
        cube_xyz = np.asarray(cube_xyz, dtype=float)

    home_xyz, home_quat, home_dbg = sample_home_pose(rng, float(cube_xyz[0]), float(cube_xyz[1]))
    grasp_quat, grasp_dbg = sample_grasp_pose(
        rng, tilt_range_deg=tilt_range_deg, azimuth_range_deg=azimuth_range_deg,
    )
    grasp_xyz = grasp_pos_for_cube(cube_xyz, grasp_quat)
    sentry_xyz, sentry_quat = sentry_pose(grasp_xyz, grasp_quat, sentry_offset)
    mid_xyz, mid_quat = mid_approach_pose(
        home_xyz, home_quat, sentry_xyz, grasp_quat, arc_lift=arc_lift,
    )
    lift_xyz = grasp_xyz + np.array([0.0, 0.0, lift_height])
    lift_quat = grasp_quat.copy()

    return EpisodeWaypoints(
        cube_xyz=cube_xyz,
        home_xyz=home_xyz, home_quat=home_quat,
        mid_xyz=mid_xyz, mid_quat=mid_quat,
        sentry_xyz=sentry_xyz, sentry_quat=sentry_quat,
        grasp_xyz=grasp_xyz, grasp_quat=grasp_quat,
        lift_xyz=lift_xyz, lift_quat=lift_quat,
        home_dbg=home_dbg, grasp_dbg=grasp_dbg,
    )


def episode_target_poses(
    wp: EpisodeWaypoints,
    *,
    n_approach: int = FRAMES_APPROACH,
    n_descend: int = FRAMES_DESCEND,
    n_lift: int = FRAMES_LIFT,
    approach_split: float = 0.60,
) -> list[np.ndarray]:
    """Densely interpolate the episode's moving phases as 4x4 CAMERA poses for IK.

    Mirrors the smoothstep + slerp drive used at runtime by
    `collect_grasp_dataset.py` and `manual_grasp_test.py`. The waypoints
    in ``wp`` are BODY (grabette_root mocap) poses; we apply the
    body→camera transform here so the result can be passed directly to a
    Placo IK call (which targets the camera frame internally) without
    further conversion. The whole pipeline — dataset, training, deployment
    — uses the camera site as the reference frame.

    Only the moving phases (approach, descend, lift) are emitted — the
    static phases (settle / close / hold / retract / final-settle) hold
    one of the keyframe poses already covered, so endpoint feasibility
    is sufficient.

    Returns a list of 4x4 numpy arrays, one per recorded frame, in execution
    order. Pass the list straight to `IKFeasibilityChecker.check_trajectory`.
    """
    n1 = max(int(n_approach * approach_split), 1)
    n2 = max(n_approach - n1, 1)

    poses: list[np.ndarray] = []

    # Approach segment 1: home -> mid (carries the orientation rotation).
    for i in range(n1):
        t = smoothstep((i + 1) / n1)
        body_xyz = (1.0 - t) * wp.home_xyz + t * wp.mid_xyz
        body_quat = slerp_quat(wp.home_quat, wp.mid_quat, t)
        cam_xyz, cam_quat = body_pose_to_camera_pose(body_xyz, body_quat)
        poses.append(pose_T(cam_xyz, cam_quat))

    # Approach segment 2: mid -> sentry (orientation already at grasp_quat).
    for i in range(n2):
        t = smoothstep((i + 1) / n2)
        body_xyz = (1.0 - t) * wp.mid_xyz + t * wp.sentry_xyz
        body_quat = slerp_quat(wp.mid_quat, wp.sentry_quat, t)
        cam_xyz, cam_quat = body_pose_to_camera_pose(body_xyz, body_quat)
        poses.append(pose_T(cam_xyz, cam_quat))

    # Descend: sentry -> grasp (linear push along the approach axis).
    for i in range(n_descend):
        t = smoothstep((i + 1) / n_descend)
        body_xyz = (1.0 - t) * wp.sentry_xyz + t * wp.grasp_xyz
        cam_xyz, cam_quat = body_pose_to_camera_pose(body_xyz, wp.grasp_quat)
        poses.append(pose_T(cam_xyz, cam_quat))

    # Lift: grasp -> grasp + (0, 0, LIFT_HEIGHT) (orientation held).
    for i in range(n_lift):
        t = smoothstep((i + 1) / n_lift)
        body_xyz = (1.0 - t) * wp.grasp_xyz + t * wp.lift_xyz
        cam_xyz, cam_quat = body_pose_to_camera_pose(body_xyz, wp.lift_quat)
        poses.append(pose_T(cam_xyz, cam_quat))

    return poses
