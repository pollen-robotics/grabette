"""Chunk-relative actions: absolute poses <-> offsets from the chunk's reference.

WHAT AND WHY
------------
An action chunk of absolute SLAM poses is re-expressed as offsets from the
chunk's FIRST pose, composed into that pose's own frame:

    translation   a_i = R_ref^-1 (p_i - p_ref)
    rotation      A_i = R_ref^-1 R_i
    gripper       passthrough — absolute, never relative

Two properties earn this over the two alternatives.

Versus per-step deltas (what `convert_dataset.py` bakes in): the supervision is
much better conditioned. Per-step deltas at 50 fps are 2-3 mm against 1-2 mm of
SLAM jitter — measured ~1.4:1 at the grasp. A chunk-length offset is ~cm-scale
against the same noise.

Versus elementwise subtraction (what LeRobot's built-in relative-action processor
does): it is ORIGIN-INVARIANT. SLAM gives every episode an arbitrary origin, and
composing into the reference frame cancels it algebraically —

    (R_g R_ref)^-1 R_g (p_i - p_ref) = R_ref^-1 (p_i - p_ref)

— whereas `p_i - p_ref` alone is a world-frame vector that moves with the
session's yaw (measured: 9.8 mm at 10 deg, 117 mm at 180 deg). Subtraction is
also simply wrong for rotation: it is not composition, and for a 6D encoding the
difference of two rotation matrices' columns is not a rotation at all.

RELOCALISATION JUMPS
--------------------
The cost of a chunk-wide reference: a spurious SLAM jump corrupts every action
after it in the chunk, where a per-step delta representation confines it to one
outlier that the existing despike zeroes. Masking is not available — pi0/pi05
ignore `action_is_pad` — and a processor step cannot drop samples. So jumps are
REPAIRED here: a jump is an artifact of re-acquisition, so splicing out the
discontinuity restores the true continuous motion. Same philosophy as the delta
despike, applied to the pose sequence.

CONVENTIONS
-----------
A chunk row is the 8D raw layout `generate_dataset.py` writes:
`[x, y, z, ax, ay, az, proximal, distal]` — absolute pose (axis-angle) plus
gripper angles in radians.

Rotation output defaults to `rotvec`, which keeps the action dimension at 8.
Relative rotations within a chunk are small (well under a second of hand
motion), so axis-angle is unambiguous here — the continuity argument for a 6D
encoding applies to ABSOLUTE rotations spanning all of SO(3). `rot="r6d"` is
available but changes the action dimension to 11 and therefore needs a matching
feature spec.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# A per-step translation above this is a SLAM re-acquisition, not hand motion:
# the same threshold the pose-smoothing segmentation uses to split segments, and
# the same value as convert_dataset.py's --despike_max_mm.
DEFAULT_JUMP_CAP_M = 0.08

# Per-step ROTATION above this is a glitch. Matches convert_dataset.py's
# --despike_max_deg, whose help records why it is 5 and not 45: at 45 the
# 5-45 deg glitches reached training, "the policy reproduced them at eval,
# amplified through the widened r6d normalization ranges, and tripped the arm
# server's IK-jump watchdog". 5 deg/step is 250 deg/s at 50 fps, above human
# wrist speed (~150 deg/s peak). Rotation glitches are real in this data, so
# repairing translation alone would be a regression against the delta pipeline.
DEFAULT_JUMP_CAP_DEG = 5.0

POSE_DIMS = 6          # x, y, z, ax, ay, az
GRIPPER_DIMS = 2       # proximal, distal


def splice_jumps(
    pos: np.ndarray, jump_cap_m: float = DEFAULT_JUMP_CAP_M
) -> tuple[np.ndarray, int]:
    """Remove relocalisation discontinuities from a position sequence.

    A jump is a single step larger than any real hand motion. It is spurious, so
    the fix is to remove it from everything that follows — which leaves the
    genuine motion on both sides intact and continuous, rather than deleting data.

    The anomalous step is *replaced* by the median of the valid steps, not zeroed.
    An observed jump step is `true_motion + artifact` and the two cannot be
    separated, so exact recovery is impossible; what remains is a choice of
    estimator for the lost motion. Zeroing (the delta-despike convention) leaves a
    residual of one full step — and because the reference is chunk-wide, that
    residual offsets EVERY later action in the chunk rather than one. The local
    median cuts it to jitter scale.

    Returns the repaired positions and how many jumps were spliced.
    """
    if len(pos) < 2:
        return pos.copy(), 0
    steps = np.diff(pos, axis=0)
    mag = np.linalg.norm(steps, axis=1)
    bad = mag > jump_cap_m
    if not bad.any():
        return pos.copy(), 0

    good = steps[~bad]
    fill = np.median(good, axis=0) if len(good) else np.zeros(3)

    # Rebuild the sequence from repaired steps: continuity by construction.
    fixed_steps = steps.copy()
    fixed_steps[bad] = fill
    out = np.empty_like(pos)
    out[0] = pos[0]
    out[1:] = pos[0] + np.cumsum(fixed_steps, axis=0)
    return out, int(bad.sum())


def splice_rotation_jumps(
    rots: Rotation, jump_cap_deg: float = DEFAULT_JUMP_CAP_DEG
) -> tuple[Rotation, int]:
    """Remove orientation discontinuities from a rotation sequence.

    Same repair as `splice_jumps`, in SO(3): per-step relative rotations
    `R_i^-1 R_{i+1}` above the cap are replaced by the median of the valid steps,
    then the sequence is rebuilt by composing forward.

    The median is taken **componentwise on the step rotvecs**, which is legitimate
    here for the reason it is *not* legitimate for the chunk reference: a valid
    step is under 5 deg, and at that scale the Lie algebra is an excellent linear
    approximation (the BCH error term is 1/2|a x b|, ~0.2 deg at 5 deg). The same
    small-angle argument fails for a chunk-wide reference precisely because the
    reference is a large arbitrary hand orientation.

    Returns the repaired rotations and how many steps were spliced.
    """
    if len(rots) < 2:
        return rots, 0
    steps = rots[:-1].inv() * rots[1:]          # body-frame per-step rotations
    mag = steps.magnitude()
    bad = mag > np.deg2rad(jump_cap_deg)
    if not bad.any():
        return rots, 0

    vecs = steps.as_rotvec()
    good = vecs[~bad]
    fill = np.median(good, axis=0) if len(good) else np.zeros(3)
    vecs[bad] = fill

    # Rebuild by composing forward from the (trusted) first orientation.
    fixed = [rots[0]]
    for v in vecs:
        fixed.append(fixed[-1] * Rotation.from_rotvec(v))
    return Rotation.concatenate(fixed), int(bad.sum())


def to_chunk_relative(
    chunk: np.ndarray,
    rot: str = "rotvec",
    jump_cap_m: float = DEFAULT_JUMP_CAP_M,
    jump_cap_deg: float = DEFAULT_JUMP_CAP_DEG,
) -> tuple[np.ndarray, dict]:
    """Absolute pose chunk -> offsets from its first row, in that row's frame.

    Args:
        chunk: (T, 8) absolute `[x,y,z,ax,ay,az,proximal,distal]`.
        rot: `"rotvec"` (keeps 8 dims) or `"r6d"` (11 dims).
        jump_cap_m: per-step translation above which a step is a SLAM jump.
        jump_cap_deg: per-step rotation above which a step is a SLAM jump.

    Returns:
        (T, 8) or (T, 11) relative actions, and a report dict.
    """
    chunk = np.asarray(chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != POSE_DIMS + GRIPPER_DIMS:
        raise ValueError(f"expected (T, 8) chunk, got {chunk.shape}")
    if rot not in ("rotvec", "r6d"):
        raise ValueError(f"rot must be 'rotvec' or 'r6d', got {rot!r}")

    pos, n_jumps = splice_jumps(chunk[:, :3], jump_cap_m)
    rots, n_rot_jumps = splice_rotation_jumps(
        Rotation.from_rotvec(chunk[:, 3:6]), jump_cap_deg
    )
    grip = chunk[:, POSE_DIMS:]

    R_ref_inv = rots[0].inv()
    # Compose, do not subtract: this is what makes the result frame-local and
    # origin-invariant, and it is the only correct relative rotation.
    rel_pos = R_ref_inv.apply(pos - pos[0])
    rel_rot = R_ref_inv * rots

    if rot == "rotvec":
        rot_part = rel_rot.as_rotvec()
    else:
        m = rel_rot.as_matrix()                 # (T, 3, 3)
        rot_part = m[:, :, :2].transpose(0, 2, 1).reshape(len(m), 6)

    out = np.concatenate([rel_pos, rot_part, grip], axis=1)
    return out.astype(np.float32), {
        "n_jumps_spliced": n_jumps,
        "n_rot_jumps_spliced": n_rot_jumps,
        "max_offset_m": float(np.linalg.norm(rel_pos, axis=1).max()),
        "max_rot_deg": float(np.rad2deg(rel_rot.magnitude().max())),
    }


def from_chunk_relative(
    rel: np.ndarray, ref_pose: np.ndarray, rot: str = "rotvec"
) -> np.ndarray:
    """Offsets -> absolute poses, given the reference they are relative to.

    At training time `ref_pose` is the chunk's first row. At inference it is the
    robot's MEASURED pose, which is what "the pose at prediction time" means once
    a real arm is in the loop.

    Args:
        rel: (T, 8) or (T, 11) relative actions.
        ref_pose: (6,) absolute `[x,y,z,ax,ay,az]`.
        rot: the encoding `rel` uses.

    Returns:
        (T, 8) absolute `[x,y,z,ax,ay,az,proximal,distal]`.
    """
    rel = np.asarray(rel, dtype=np.float64)
    ref_pose = np.asarray(ref_pose, dtype=np.float64).reshape(-1)
    n_rot = 3 if rot == "rotvec" else 6
    expected = 3 + n_rot + GRIPPER_DIMS
    if rel.ndim != 2 or rel.shape[1] != expected:
        raise ValueError(f"expected (T, {expected}) for rot={rot!r}, got {rel.shape}")
    if ref_pose.shape[0] != POSE_DIMS:
        raise ValueError(f"expected a 6D reference pose, got {ref_pose.shape}")

    R_ref = Rotation.from_rotvec(ref_pose[3:6])
    pos = R_ref.apply(rel[:, :3]) + ref_pose[:3]

    if rot == "rotvec":
        rel_rot = Rotation.from_rotvec(rel[:, 3:6])
    else:
        rel_rot = Rotation.from_matrix(_r6d_to_matrix(rel[:, 3:9]))
    rots = (R_ref * rel_rot).as_rotvec()

    return np.concatenate([pos, rots, rel[:, 3 + n_rot:]], axis=1).astype(np.float32)


def _r6d_to_matrix(v: np.ndarray) -> np.ndarray:
    """6D -> rotation matrices, by Gram-Schmidt on the first two columns."""
    a1, a2 = v[:, :3], v[:, 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    a2p = a2 - (b1 * a2).sum(axis=1, keepdims=True) * b1
    b2 = a2p / np.linalg.norm(a2p, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)
