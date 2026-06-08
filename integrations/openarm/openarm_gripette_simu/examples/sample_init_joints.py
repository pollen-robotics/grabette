"""Sample one training-distribution home pose and print the arm joints.

The default IK seed used by the sim server is just "a good starting point
for IK convergence" — it is NOT a pose the policy ever saw at training
time. If the deployment arm starts at an out-of-distribution gripper
pose, the visual encoder produces OOD features and the policy outputs
near-mean actions (which on this dataset look like a slow lift with
gripper drifting closed) regardless of cube position.

This script samples a home pose from the SAME distribution
`collect_grasp_dataset.py` uses, runs IK to the camera frame, prints the
7 joint values formatted for `--initial-joints`. Pass them to the server.

Usage:
    cd /home/steve/Project/Repo/GRABETTE/openarm_gripette_simu
    uv run python examples/sample_init_joints.py
    uv run python examples/sample_init_joints.py --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grabette_trajectory import (  # noqa: E402
    body_pose_to_camera_pose,
    pose_T,
    sample_episode_waypoints,
)
from openarm_gripette_simu import IKFeasibilityChecker, Kinematics  # noqa: E402
from openarm_gripette_simu.kinematics import CAMERA_FRAME  # noqa: E402

ARM_IK_SEED = np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.5])


def parse_args():
    p = argparse.ArgumentParser(description="Print --initial-joints for a training-distribution home pose.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_attempts", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    kin = Kinematics()

    checker = IKFeasibilityChecker(
        Kinematics(),
        frame=CAMERA_FRAME,
        seed_joints=ARM_IK_SEED,
        pos_tol_m=0.015,
        rot_tol_deg=5.0,
        n_iter=200,
    )

    # Use the same rejection sampler the dataset collector uses, so we land
    # on a plan whose entire trajectory is arm-reachable. We only need the
    # home pose, but it has to come from a fully-feasible plan.
    from grabette_trajectory import episode_target_poses  # noqa: E402

    def builder():
        wp = sample_episode_waypoints(rng)
        return episode_target_poses(wp), wp

    poses, wp, stats = checker.sample_feasible_trajectory(
        builder, max_attempts=args.max_attempts,
    )
    if wp is None:
        print(f"Could not find a feasible plan in {args.max_attempts} attempts.", file=sys.stderr)
        sys.exit(1)

    print(f"Sampled training-distribution home pose:")
    print(f"  cube_xyz   : {wp.cube_xyz}")
    print(f"  home_xyz   : {wp.home_xyz}  (body frame, in world coords)")
    print(f"  home pitch : {wp.home_dbg['pitch_deg']:.1f} deg")
    print(f"  home yaw   : {wp.home_dbg['yaw_deg']:.1f} deg")

    cam_xyz, cam_quat = body_pose_to_camera_pose(wp.home_xyz, wp.home_quat)
    home_T = pose_T(cam_xyz, cam_quat)
    arm_q = kin.inverse(home_T, current_joint_positions=ARM_IK_SEED.copy(),
                         n_iter=300, frame=CAMERA_FRAME)

    # Verify
    T_actual = kin.forward(arm_q, frame=CAMERA_FRAME)
    pos_err_mm = np.linalg.norm(T_actual[:3, 3] - cam_xyz) * 1000
    print(f"  IK residual: {pos_err_mm:.2f} mm (expect <1 mm)")

    print()
    print("Arm joints (rad):")
    print("  " + ", ".join(f"{q:+.4f}" for q in arm_q))
    print()
    print("Pass to the sim server like this:")
    joint_str = " ".join(f"{q:.4f}" for q in arm_q)
    print(f"  uv run python -m openarm_gripette_simu \\")
    print(f"      --scene scenes/table_grasp.xml \\")
    print(f"      --initial-joints {joint_str}")


if __name__ == "__main__":
    main()
