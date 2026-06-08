"""IK feasibility checking and rejection sampling for cross-embodiment data.

Why this module exists
----------------------
Handheld data acquisition rigs (Grabette, UMI, etc.) record the gripper's
SE(3) pose without any constraint that a robot arm can reproduce that pose.
Some recorded poses end up outside the target robot's reachable workspace.
At training time, those frames teach the policy to reach poses the deployed
robot can never achieve, hurting transfer.

The fix is to filter demonstrations through the robot's IK solver before they
enter the dataset (UMI 2.0; Chi et al.). This module provides:

  * `IKFeasibilityChecker` — runs IK on a single 4x4 target pose or on a chain
    of poses (warm-started across the chain) and reports whether each pose is
    within position / rotation tolerance of the target after the solver
    converges.
  * `IKFeasibilityChecker.sample_feasible_trajectory(...)` — rejection
    sampling: keep generating candidate trajectories from a user-supplied
    builder until one fits inside the robot's reachable workspace, or give up
    after `max_attempts`.

Robot-agnostic. The only assumption is that the kinematics object exposes a
Placo-style API:

    forward(joint_positions, frame=...) -> 4x4 transform
    inverse(target_pose, current_joint_positions=..., n_iter=..., frame=...)
        -> joint position vector

Plug a different `Kinematics` instance in and the same checker filters data
for a different arm — same primitive used in cross-embodiment data prefiltering
in the UMI papers.

Example (Grabette pose against the OpenArm)
-------------------------------------------
    from openarm_gripette_simu import Kinematics
    from openarm_gripette_simu.ik_feasibility import IKFeasibilityChecker
    from openarm_gripette_simu.kinematics import GRIPPER_FRAME

    checker = IKFeasibilityChecker(
        Kinematics(),
        frame=GRIPPER_FRAME,
        seed_joints=np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.5]),
        pos_tol_m=0.015,
        rot_tol_deg=5.0,
    )

    res = checker.check_pose(target_T)        # one pose
    traj = checker.check_trajectory(poses)    # chain (warm-started)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class PoseFeasibility:
    """Result of a single-pose IK feasibility check."""
    feasible: bool
    joints: np.ndarray         # IK output (best-effort, even when infeasible)
    pos_err_m: float
    rot_err_deg: float


@dataclass
class TrajectoryFeasibility:
    """Result of a chain-of-poses feasibility check (warm-started IK)."""
    fully_feasible: bool
    n_feasible: int
    n_total: int
    max_pos_err_m: float
    max_rot_err_deg: float
    first_failure_index: int | None  # 0-based index of the first infeasible pose
    final_joints: np.ndarray | None  # IK joint solution at the last checked pose


@dataclass
class RejectionSamplingStats:
    """Aggregate stats for a single rejection-sampling attempt."""
    accepted: bool
    n_attempts: int
    last_feasibility: TrajectoryFeasibility | None


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #


def _rotation_angle_between(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angle (radians) between two 3x3 rotation matrices.

    Computed from `trace(R1 R2^T)`; clipped to avoid arccos NaNs from numerical
    drift just outside [-1, 1].
    """
    R = R1 @ R2.T
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


# --------------------------------------------------------------------------- #
# Checker
# --------------------------------------------------------------------------- #


class IKFeasibilityChecker:
    """Per-pose and per-trajectory IK feasibility test against a target robot.

    Parameters
    ----------
    kinematics : object
        Kinematics solver with `forward(joints, frame=...)` returning a 4x4
        transform and `inverse(target_T, current_joint_positions=...,
        n_iter=..., frame=...)` returning a joint vector.
    frame : str
        Frame name passed through to the kinematics solver (e.g. "gripper").
    seed_joints : np.ndarray
        Default starting configuration for the IK warm-start. Picking a
        sensible seed matters because the IK is non-convex; in practice the
        same seed used by the rest of the codebase (e.g. the manual grasp
        test) is the right choice.
    pos_tol_m : float
        Position tolerance after IK (Euclidean, metres). Default 1.5 cm.
    rot_tol_deg : float
        Rotation tolerance after IK (geodesic angle, degrees). Default 5°.
    n_iter : int
        IK solver iteration budget per pose. Bigger = slower but more reliable.
    """

    def __init__(
        self,
        kinematics,
        *,
        frame: str,
        seed_joints: np.ndarray,
        pos_tol_m: float = 0.015,
        rot_tol_deg: float = 5.0,
        n_iter: int = 200,
    ):
        self.kinematics = kinematics
        self.frame = frame
        self.seed_joints = np.asarray(seed_joints, dtype=float).copy()
        self.pos_tol_m = float(pos_tol_m)
        self.rot_tol_rad = float(np.deg2rad(rot_tol_deg))
        self.n_iter = int(n_iter)

    # ----- per-pose -----

    def check_pose(
        self,
        target_T: np.ndarray,
        *,
        seed_joints: np.ndarray | None = None,
    ) -> PoseFeasibility:
        """Run IK on a single 4x4 target pose; report residuals + verdict.

        The solver returns its best-effort joint configuration even when the
        target is unreachable. We then compute the actual achieved pose via FK
        and compare against the target. A pose is feasible iff position error
        <= pos_tol_m AND rotation error <= rot_tol_deg.
        """
        seed = self.seed_joints if seed_joints is None else seed_joints
        joints = self.kinematics.inverse(
            target_T,
            current_joint_positions=seed,
            n_iter=self.n_iter,
            frame=self.frame,
        )
        T_actual = self.kinematics.forward(joints, frame=self.frame)
        pos_err = float(np.linalg.norm(T_actual[:3, 3] - target_T[:3, 3]))
        rot_err_rad = _rotation_angle_between(T_actual[:3, :3], target_T[:3, :3])
        feasible = (pos_err <= self.pos_tol_m) and (rot_err_rad <= self.rot_tol_rad)
        return PoseFeasibility(
            feasible=feasible,
            joints=joints,
            pos_err_m=pos_err,
            rot_err_deg=float(np.degrees(rot_err_rad)),
        )

    # ----- per-trajectory -----

    def check_trajectory(
        self,
        target_Ts: Sequence[np.ndarray],
        *,
        seed_joints: np.ndarray | None = None,
        early_stop: bool = False,
    ) -> TrajectoryFeasibility:
        """Run IK along a sequence of poses, warm-starting each from the previous solution.

        Warm-starting mirrors how a real control loop would chase the reference
        — we also avoid getting stuck in different local IK minima between
        adjacent poses in a smooth path.

        Parameters
        ----------
        target_Ts : sequence of 4x4 ndarrays
            Pose targets in execution order.
        seed_joints : optional
            Starting configuration for the FIRST pose. Subsequent poses warm-
            start from the previous IK solution.
        early_stop : bool
            If True, return as soon as the first infeasible pose is encountered.
            Useful inside rejection sampling, where we only need a yes/no.
        """
        seed = self.seed_joints if seed_joints is None else seed_joints
        joints = seed.copy()
        n_total = len(target_Ts)
        n_feasible = 0
        max_pos = 0.0
        max_rot_rad = 0.0
        first_fail: int | None = None

        for i, T in enumerate(target_Ts):
            res = self.check_pose(T, seed_joints=joints)
            joints = res.joints  # warm-start the next iteration
            max_pos = max(max_pos, res.pos_err_m)
            max_rot_rad = max(max_rot_rad, np.deg2rad(res.rot_err_deg))
            if res.feasible:
                n_feasible += 1
            else:
                if first_fail is None:
                    first_fail = i
                if early_stop:
                    return TrajectoryFeasibility(
                        fully_feasible=False,
                        n_feasible=n_feasible,
                        n_total=n_total,
                        max_pos_err_m=max_pos,
                        max_rot_err_deg=float(np.degrees(max_rot_rad)),
                        first_failure_index=first_fail,
                        final_joints=joints,
                    )

        return TrajectoryFeasibility(
            fully_feasible=(n_feasible == n_total),
            n_feasible=n_feasible,
            n_total=n_total,
            max_pos_err_m=max_pos,
            max_rot_err_deg=float(np.degrees(max_rot_rad)),
            first_failure_index=first_fail,
            final_joints=joints,
        )

    # ----- rejection sampling -----

    def sample_feasible_trajectory(
        self,
        builder: Callable[[], tuple[Sequence[np.ndarray], object]],
        *,
        max_attempts: int = 50,
        seed_joints: np.ndarray | None = None,
    ) -> tuple[Sequence[np.ndarray] | None, object | None, RejectionSamplingStats]:
        """Rejection-sample a trajectory until it is fully arm-feasible.

        Parameters
        ----------
        builder : callable
            Zero-argument function returning ``(target_Ts, metadata)``. Called
            anew each attempt; the caller is responsible for randomising
            inside (e.g. closure over an rng). ``metadata`` is opaque — passed
            through to the caller untouched.
        max_attempts : int
            Give up after this many attempts.
        seed_joints : optional
            Override the IK warm-start seed for the first pose of each attempt.

        Returns
        -------
        (target_Ts, metadata, stats)
            ``target_Ts`` and ``metadata`` are None if no feasible trajectory
            was found within ``max_attempts``. ``stats.accepted`` reports the
            outcome and ``stats.n_attempts`` reports how many tries were used.
        """
        last_feas: TrajectoryFeasibility | None = None
        for attempt in range(1, max_attempts + 1):
            poses, metadata = builder()
            feas = self.check_trajectory(
                poses, seed_joints=seed_joints, early_stop=True
            )
            last_feas = feas
            if feas.fully_feasible:
                return (
                    poses,
                    metadata,
                    RejectionSamplingStats(
                        accepted=True,
                        n_attempts=attempt,
                        last_feasibility=feas,
                    ),
                )
        return (
            None,
            None,
            RejectionSamplingStats(
                accepted=False,
                n_attempts=max_attempts,
                last_feasibility=last_feas,
            ),
        )
