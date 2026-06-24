"""Placo-based kinematics for the OpenArm right arm.

Provides forward and inverse kinematics targeting the 'camera' frame.
Only computes desired joint positions — no physics, no simulation.
"""

import numpy as np
import placo
from openarm_gripette_model import OPENARM_RIGHT_DIR

# The 7 arm joints in MuJoCo ordering (excludes gripper and mimic joints)
ARM_JOINT_NAMES = [
    "r_arm_pitch",
    "r_arm_roll",
    "r_arm_yaw",
    "r_elbow",
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch",
]

# Frame names
CAMERA_FRAME = "camera"
GRIPPER_FRAME = "gripper_center"
OAKL_FRAME = "oak_l"   # SLAM / control frame the grasp trajectory commands

# THE control frame: the single frame the whole pipeline (recorded data action,
# IK feasibility filter, arm Cartesian-delta control, server reset, replay) must
# agree on. The policy commands this frame; IK targets it. Change it HERE only.
# Must be oak_l for the current free-floating dataset (the scene is rooted at
# oak_l, so the recorded mocap pose IS the oak_l pose — see collect_grasp_dataset
# and the assert there). Switching to CAMERA_FRAME additionally requires the data
# recording + IK targets to apply the oak_l->camera transform.
CONTROL_FRAME = OAKL_FRAME

# --- IK solver robustness knobs (OPT-IN; see Kinematics.__init__) -------------
# The QP decision variable is the joint velocity dq; near a singularity, matching
# the task velocity needs a huge dq. These three knobs were investigated as
# singularity-"explosion" mitigations but DID NOT show a clean win in an offline
# IK harness (the genuine cause is an IK branch-flip when the integrator target
# marches past the reachable workspace, which solver-cost tuning can't fix at
# acceptable position accuracy; damping only bounds the jump at reg>=1 where
# position error blows past 50mm, and the posture task mostly adds branch-flip
# accuracy noise). They are kept as OPT-IN kwargs so they can be re-tuned in the
# REAL MuJoCo eval loop, but default to the original (safe, accurate) behaviour.
#
# A) Damped-least-squares damping: Tikhonov penalty 'IK_REGULARIZATION * ||dq||²'.
#    1e-4 keeps the QP full-rank without perturbing the solution.
IK_REGULARIZATION = 1e-4
# C) Posture task toward a preferred resting config (elbow 90°, others neutral;
#    matches arm_servicer.START_JOINTS). 0.0 = disabled (default).
IK_POSTURE_WEIGHT = 0.0
POSTURE_REFERENCE = {
    "r_arm_pitch": 0.0,
    "r_arm_roll": 0.0,
    "r_arm_yaw": 0.0,
    "r_elbow": np.pi / 2,   # 90°
    "r_wrist_yaw": 0.0,
    "r_wrist_roll": 0.0,
    "r_wrist_pitch": 0.0,
}


class Kinematics:
    """Placo wrapper for FK/IK on the OpenArm right arm."""

    def __init__(self, model_dir=None, position_weight: float = 100.0,
                 orientation_weight: float = 1.0,
                 regularization: float = IK_REGULARIZATION,
                 posture_weight: float = IK_POSTURE_WEIGHT,
                 velocity_limits: bool = False):
        """
        Args:
            model_dir: URDF dir (defaults to OPENARM_RIGHT_DIR).
            position_weight, orientation_weight: Placo frame-task weights
                (default 100:1). Higher orientation_weight (e.g. 10) locks
                rotation more strictly at the cost of position accuracy —
                useful on real hardware where the kinematic chain is close
                to a wrist-roll singularity at the typical home pose and
                position-only priority leaks rotation into the wrist joints.
            regularization: damped-least-squares damping on dq. Default 1e-4
                (keeps the QP full-rank). Raising it bounds dq near singularities
                only at >=1, where position error exceeds 50mm — not recommended.
            posture_weight: weight of the elbow-90° posture task. 0 disables it
                (default). Opt-in / experimental — see the module-level note.
            velocity_limits: per-step joint velocity cap. Default False; enabling
                it under-converges the offline 500-iter solve (accuracy loss).
        """
        model_dir = str(model_dir) if model_dir else str(OPENARM_RIGHT_DIR)
        self.robot = placo.RobotWrapper(model_dir)

        # Set up the IK solver
        self.solver = self.robot.make_solver()
        self.solver.mask_fbase(True)  # base is fixed
        self.solver.enable_joint_limits(True)
        # Optional per-step joint velocity cap (off by default — see kwargs note).
        self.solver.enable_velocity_limits(velocity_limits)
        self.solver.dt = 0.01

        # Mask non-arm joints so IK only moves the 7 arm DOFs
        self.solver.mask_dof("proximal")
        self.solver.mask_dof("distal")
        self.solver.mask_dof("r_wrist_roll_mimic")

        # QP regularization on dq (damped least squares); see IK_REGULARIZATION.
        self.solver.add_regularization_task(regularization)

        # Frame task on camera, with position weighted 100x higher than
        # orientation. Equal weighting (the previous default) caused the QP
        # to compromise — landing 30-200 mm off the position to better
        # match an orientation we may not actually reach. With position
        # priority, the solver nails position first and accepts whatever
        # orientation is achievable. Placo's FrameTask.configure takes
        # (name, type, position_weight, orientation_weight).
        self.robot.update_kinematics()
        T_cam = self.robot.get_T_world_frame(CAMERA_FRAME)
        self._frame_task = self.solver.add_frame_task(CAMERA_FRAME, T_cam)
        self._frame_task.configure(CAMERA_FRAME, "soft", position_weight, orientation_weight)

        # Optional posture task toward the preferred resting config (elbow 90°).
        # Off by default (posture_weight=0); see the module-level note.
        self._posture_task = None
        if posture_weight > 0.0:
            self._posture_task = self.solver.add_joints_task()
            self._posture_task.set_joints(POSTURE_REFERENCE)
            self._posture_task.configure("posture", "soft", posture_weight)

        # Fixed offsets to the camera frame (the solver always tasks the camera),
        # so targets given in other frames can be converted to a camera target.
        T_grip = self.robot.get_T_world_frame(GRIPPER_FRAME)
        self._T_grip_to_cam = np.linalg.inv(T_grip) @ T_cam
        T_oakl = self.robot.get_T_world_frame(OAKL_FRAME)
        self._T_oakl_to_cam = np.linalg.inv(T_oakl) @ T_cam

    def forward(self, joint_positions: np.ndarray, frame: str = CONTROL_FRAME) -> np.ndarray:
        """Compute a frame's pose from arm joint positions.

        Args:
            joint_positions: 7-element array of arm joint angles (rad).
            frame: frame name to compute FK for (default: CONTROL_FRAME).

        Returns:
            4x4 homogeneous transform (world -> frame).
        """
        for i, name in enumerate(ARM_JOINT_NAMES):
            self.robot.set_joint(name, joint_positions[i])
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(frame).copy()

    def inverse(
        self,
        target_pose: np.ndarray,
        current_joint_positions: np.ndarray | None = None,
        n_iter: int = 500,
        frame: str = CONTROL_FRAME,
    ) -> np.ndarray:
        """Solve IK for a target pose of the given frame.

        Args:
            target_pose: 4x4 homogeneous transform (world -> frame).
            current_joint_positions: optional 7-element starting config.
                If None, uses the robot's current state.
            n_iter: number of solver iterations.
            frame: which frame to target ('camera' or 'gripper_center').
                   Gripper targets are converted to camera targets internally.

        Returns:
            7-element array of arm joint angles (rad).
        """
        if current_joint_positions is not None:
            for i, name in enumerate(ARM_JOINT_NAMES):
                self.robot.set_joint(name, current_joint_positions[i])
            self.robot.update_kinematics()

        # Convert a target given in another frame to the camera target the
        # solver tasks, using the fixed offset for that frame.
        if frame == GRIPPER_FRAME:
            cam_target = target_pose @ self._T_grip_to_cam
        elif frame == OAKL_FRAME:
            cam_target = target_pose @ self._T_oakl_to_cam
        else:
            cam_target = target_pose

        self._frame_task.T_world_frame = cam_target

        for _ in range(n_iter):
            self.solver.solve(True)
            self.robot.update_kinematics()

        return self.get_arm_joints()

    def get_arm_joints(self) -> np.ndarray:
        """Read the current arm joint positions from the Placo model."""
        return np.array([self.robot.get_joint(name) for name in ARM_JOINT_NAMES])
