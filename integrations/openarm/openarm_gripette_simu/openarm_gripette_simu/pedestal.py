"""Shared arm<->pedestal clearance query.

The arm can swing its upper-arm-and-outward links into the mounting column. We
guard against that in two places — the eval Cartesian-delta gate (arm_servicer)
and the data-gen IK feasibility filter (ik_feasibility) — and they MUST agree on
the box, the margin, and which links are guarded. This module is the single
source of truth: a `PedestalClearance` computes the signed min distance from the
guarded arm links to the `pedestal_box` proxy geom for a given arm config, via
mj_geomDistance (works regardless of the box's contype, which is 0 so it never
perturbs physics).
"""

import mujoco
import numpy as np

from .kinematics import ARM_JOINT_NAMES

# The column proxy geom (a non-physical box wrapping the hfsb6_6060 extrusion;
# defined in scenes/table_grasp.xml).
PEDESTAL_BOX_GEOM = "pedestal_box"
# Clearance to keep from the column (metres).
PEDESTAL_MARGIN = 0.02
# Links that can actually reach the column (the upper arm outward). The two base
# links never approach (>=65mm), per the link-proximity sweep, so they're excluded.
PEDESTAL_ACTIVE_BODIES = (
    "dm_j4340_2_step", "dm_j4310_1_2_step", "j6_a_3_step", "j7_e_2_step",
    "dm_j4310_1_2_step_2", "j8_a_2_step",
    "feetech_sts3215__configuration_default", "distal_soft_tip",
)


class PedestalClearance:
    """Min distance from the guarded arm links to the pedestal box, for a given
    7-DOF arm config. Disabled (returns +inf) if the model has no pedestal_box."""

    def __init__(self, model):
        self.model = model
        self.box_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PEDESTAL_BOX_GEOM)
        self.enabled = self.box_gid >= 0
        if not self.enabled:
            self.arm_geom_ids = []
            return
        active = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                  for b in PEDESTAL_ACTIVE_BODIES}
        active.discard(-1)
        # COLLISION geoms (contype/conaffinity != 0) of the active links only.
        self.arm_geom_ids = [g for g in range(model.ngeom)
                             if model.geom_bodyid[g] in active
                             and (model.geom_contype[g] or model.geom_conaffinity[g])]
        # Scratch MjData for FK-only evaluation (no physics, no live-state touch).
        self._data = mujoco.MjData(model)
        self._qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                      for n in ARM_JOINT_NAMES]
        self._mimic_qadr = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_wrist_roll_mimic")]
        self._wrist_roll_idx = ARM_JOINT_NAMES.index("r_wrist_roll")

    def clearance(self, arm_joints) -> float:
        """Signed min distance (m) from any guarded arm link to the pedestal box
        for `arm_joints` (negative = penetration; capped above at PEDESTAL_MARGIN;
        +inf if disabled)."""
        if not self.enabled:
            return np.inf
        m, d = self.model, self._data
        for adr, val in zip(self._qadr, arm_joints):
            d.qpos[adr] = val
        d.qpos[self._mimic_qadr] = -arm_joints[self._wrist_roll_idx]
        mujoco.mj_kinematics(m, d)
        box, dmin = self.box_gid, PEDESTAL_MARGIN
        for g in self.arm_geom_ids:
            dmin = min(dmin, mujoco.mj_geomDistance(m, d, box, g, PEDESTAL_MARGIN, None))
        return dmin

    def collides(self, arm_joints, margin: float = PEDESTAL_MARGIN) -> bool:
        """True iff `arm_joints` brings a guarded link within `margin` of the box."""
        return self.clearance(arm_joints) < margin
