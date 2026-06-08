"""Check whether Grabette demonstration trajectories are reachable by the OpenArm.

The Grabette is mocap-controlled and can pose itself anywhere in space. The
OpenArm has joint limits and a finite reachable workspace — so a Grabette
trajectory might not actually be executable on the arm. This is a problem
because:

  - During training, we collect demonstrations with the Grabette.
  - During deployment, the same trajectory needs to be reproduced by the
    OpenArm with the Gripette mounted on it.

If the trajectory leaves the arm's workspace at any point, the policy may learn
something the real robot can't follow.

This script:
  1. Samples N grasp episodes via `grabette_trajectory.sample_episode_waypoints`
     (no MuJoCo physics — pure waypoint geometry).
  2. Densely interpolates the per-frame target pose along the planned path
     (smoothstep + slerp), exactly mirroring how the dataset is recorded.
  3. Runs `IKFeasibilityChecker.check_trajectory` on each per-frame target.
  4. Reports per-episode and aggregate feasibility.

Same module is used by `collect_grasp_dataset.py --ik_filter` to reject
infeasible plans BEFORE running physics.

Usage:
    uv run python examples/check_grabette_reachable.py --episodes 10
    uv run python examples/check_grabette_reachable.py --episodes 20 --seed 42 --pos_tol 0.02
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Trajectory helpers (shared with the Grabette test/dataset scripts)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grabette_trajectory import (  # noqa: E402
    LIFT_HEIGHT,
    episode_target_poses,
    sample_episode_waypoints,
)
from openarm_gripette_simu import IKFeasibilityChecker, Kinematics  # noqa: E402
from openarm_gripette_simu.kinematics import CAMERA_FRAME  # noqa: E402

# Per-frame pose tolerances. IK solver returns clamped/best-effort joints; we
# read residual from FK.
DEFAULT_POS_TOL = 0.015  # 1.5 cm
DEFAULT_ROT_TOL_DEG = 5.0

# Seed arm joints used as the IK starting pose for the first frame of each
# episode. Same as manual_grasp_test_arm.py uses; gives "gripper roughly
# pointing forward-down" without joint-limit ambiguity.
ARM_IK_SEED = np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.5])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(
        description="Check whether Grabette trajectories are reachable by the OpenArm via IK."
    )
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pos_tol", type=float, default=DEFAULT_POS_TOL,
                   help=f"Per-frame position tolerance in metres (default {DEFAULT_POS_TOL}).")
    p.add_argument("--rot_tol_deg", type=float, default=DEFAULT_ROT_TOL_DEG,
                   help=f"Per-frame rotation tolerance in degrees (default {DEFAULT_ROT_TOL_DEG}).")
    p.add_argument("--lift_height", type=float, default=LIFT_HEIGHT,
                   help="Lift target z above grasp pose (matches the dataset collector).")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    checker = IKFeasibilityChecker(
        Kinematics(),
        frame=CAMERA_FRAME,
        seed_joints=ARM_IK_SEED,
        pos_tol_m=args.pos_tol,
        rot_tol_deg=args.rot_tol_deg,
        n_iter=200,
    )

    print(
        f"Checking {args.episodes} Grabette episode(s) for OpenArm reachability "
        f"(pos_tol={args.pos_tol*1000:.1f}mm, rot_tol={args.rot_tol_deg:.1f}deg)."
    )
    print()
    print(
        f"{'ep':>3} {'cube_xy':>14} {'g_pitch':>8} {'g_azim':>8} "
        f"{'frames':>7} {'feas':>8} {'max_pos':>10} {'max_rot':>10} {'verdict':>10}"
    )

    total_feasible = 0
    total_frames = 0
    n_fully_feasible = 0
    t_start = time.perf_counter()

    for ep in range(args.episodes):
        wp = sample_episode_waypoints(rng, lift_height=args.lift_height)
        poses = episode_target_poses(wp)
        feas = checker.check_trajectory(poses)

        cube_xy = wp.cube_xyz[:2]
        g_pitch = wp.grasp_dbg["grasp_pitch_deg"]
        g_azim = wp.grasp_dbg["grasp_azimuth_deg"]
        frac = feas.n_feasible / max(feas.n_total, 1)
        if feas.fully_feasible:
            n_fully_feasible += 1
        verdict = "OK" if feas.fully_feasible else "PARTIAL" if frac > 0.5 else "BAD"

        print(
            f"{ep:3d} ({cube_xy[0]:+.3f},{cube_xy[1]:+.3f}) "
            f"{g_pitch:7.1f}° {g_azim:+7.1f}° "
            f"{feas.n_total:7d} {feas.n_feasible:4d}/{feas.n_total:<3d} "
            f"{feas.max_pos_err_m*1000:7.1f}mm {feas.max_rot_err_deg:8.1f}° "
            f"{verdict:>10}"
        )

        total_feasible += feas.n_feasible
        total_frames += feas.n_total

    dt = time.perf_counter() - t_start
    print()
    print(
        f"Aggregate: {total_feasible}/{total_frames} frames feasible "
        f"({100.0 * total_feasible / max(total_frames, 1):.1f}%); "
        f"{n_fully_feasible}/{args.episodes} episodes fully feasible "
        f"({100.0 * n_fully_feasible / args.episodes:.1f}%); "
        f"{dt:.1f}s elapsed."
    )

    if n_fully_feasible == args.episodes:
        print("All trajectories reachable — the OpenArm should be able to follow them.")
    elif n_fully_feasible >= args.episodes * 0.7:
        print(
            "Most trajectories reachable. The episodes that fail likely have "
            "extreme home or grasp poses; consider tightening the home/grasp "
            "ranges in grabette_trajectory.py if you want all-feasible data."
        )
    else:
        print(
            "Significant fraction of trajectories unreachable. The Grabette is "
            "going outside the OpenArm's workspace. Use "
            "`collect_grasp_dataset.py --ik_filter` to drop infeasible plans "
            "before recording, or tighten HOME_*_RANGE / "
            "GRASP_TILT_RANGE_DEG / GRASP_AZIMUTH_RANGE_DEG."
        )


if __name__ == "__main__":
    main()
