"""Replay Grabette demonstrations on the simulated OpenArm.

The Grabette collector drives the gripper via a welded mocap — kinematics are
imposed directly. The deployed system drives the gripper via 7 arm joints
under PD control, so a trajectory that passes the IK feasibility check
(per-frame solver residual < 15 mm / 5°) is not necessarily *dynamically*
executable: the arm controller has to track it under physics, the joint
velocities have to stay sane, and the kinematic chain must not self-collide
or hit a joint limit during the move.

This script samples the same plans `collect_grasp_dataset.py` would, then
drives the arm scene (`scenes/table_grasp.xml`) by per-frame IK against the
identical waypoint chain. Per trial we record:
  * success: cube lifted >= LIFT_SUCCESS_THRESHOLD above its starting z
  * max position tracking error (gripper FRAME against the planned target)
  * max rotation tracking error

If most plans accepted by the IK filter also succeed here, the certification
chain (Grabette demos → train → arm replay) is closed. If many fail with
large tracking errors, the demo trajectories are too aggressive for the
arm's real-time tracking and we need to slow the playback or re-plan.

Usage:
    uv run python examples/replay_grabette_on_arm.py --episodes 10
    uv run python examples/replay_grabette_on_arm.py --episodes 10 --viewer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_grabette_reachable import ARM_IK_SEED  # noqa: E402
from grabette_trajectory import (  # noqa: E402
    CUBE_START_Z,
    DISTAL_CLOSED,
    DISTAL_OPEN,
    LIFT_SUCCESS_THRESHOLD,
    PROXIMAL_CLOSED,
    PROXIMAL_OPEN,
    RETRACT_EXTRA,
    body_pose_to_camera_pose,
    episode_target_poses,
    pose_T,
    sample_episode_waypoints,
    slerp_quat,
    smoothstep,
)
from openarm_gripette_simu import IKFeasibilityChecker, Simulation  # noqa: E402
from openarm_gripette_simu.kinematics import CAMERA_FRAME, Kinematics  # noqa: E402

SCENE = Path(__file__).parent.parent / "scenes" / "table_grasp.xml"

# Sub-steps per recorded frame at 50 fps (sim dt = 0.001).
SIM_SUBSTEPS = 20

# Phase frame counts mirror collect_grasp_dataset.py exactly. Same wall-clock
# duration per phase, same number of IK targets — so this replay is a
# faithful re-execution of what the dataset records, only via the arm
# instead of the welded mocap.
FRAMES_INITIAL_SETTLE = 5
FRAMES_APPROACH = 80
FRAMES_DESCEND = 50
FRAMES_PRE_GRIP_SETTLE = 25
FRAMES_CLOSE = 30
FRAMES_HOLD = 50
FRAMES_LIFT = 60
FRAMES_RETRACT = 50
FRAMES_FINAL_SETTLE = 15


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def set_cube_pose(sim: Simulation, cube_qadr: int, x: float, y: float, z: float):
    """Place the cube at (x, y, z) with identity orientation, zero velocity."""
    sim.data.qpos[cube_qadr:cube_qadr + 3] = (x, y, z)
    sim.data.qpos[cube_qadr + 3:cube_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    qveladr = sim.model.jnt_dofadr[cube_jnt_id]
    sim.data.qvel[qveladr:qveladr + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def _rot_err_rad(R_target: np.ndarray, R_actual: np.ndarray) -> float:
    cos_t = (np.trace(R_actual @ R_target.T) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_t, -1.0, 1.0)))


def smooth_segment_targets(start_xyz, start_quat, end_xyz, end_quat, n_frames):
    """Smooth-step + slerp interpolation between two waypoints."""
    out = []
    for i in range(n_frames):
        t = smoothstep((i + 1) / n_frames)
        xyz = (1.0 - t) * start_xyz + t * end_xyz
        quat = slerp_quat(start_quat, end_quat, t)
        out.append((xyz, quat))
    return out


def multiwaypoint_targets(waypoints_pos, waypoints_quat, segment_frames):
    out = []
    for k, n in enumerate(segment_frames):
        out.extend(smooth_segment_targets(
            waypoints_pos[k], waypoints_quat[k],
            waypoints_pos[k + 1], waypoints_quat[k + 1],
            n,
        ))
    return out


def drive_targets(sim: Simulation, kin: Kinematics, arm_q: np.ndarray,
                  targets: list[tuple[np.ndarray, np.ndarray]],
                  gripper_ctrl: tuple[float, float],
                  prox_id: int, dist_id: int,
                  viewer=None):
    """Run per-frame IK against body-frame waypoint targets, sub-stepping physics.

    `targets` is a list of (body_xyz, body_quat) keyframes from the planner.
    The pipeline-wide convention is camera-frame poses (world-positioned,
    gravity-aligned), so we convert each waypoint via body_pose_to_camera_pose
    and pass the result to Placo IK with frame=CAMERA_FRAME. Tracking error
    is reported in camera space (= the convention the dataset records).

    Returns (final arm_q, max camera-pos err, max camera-rot err in radians).
    """
    max_pos = 0.0
    max_rot_rad = 0.0
    for body_xyz, body_quat in targets:
        cam_xyz, cam_quat = body_pose_to_camera_pose(body_xyz, body_quat)
        T_cam_target = pose_T(cam_xyz, cam_quat)
        arm_q = kin.inverse(T_cam_target, current_joint_positions=arm_q,
                             n_iter=50, frame=CAMERA_FRAME)
        sim.set_arm_commands(arm_q)
        sim.data.ctrl[prox_id] = gripper_ctrl[0]
        sim.data.ctrl[dist_id] = gripper_ctrl[1]
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        if viewer is not None:
            viewer.sync()
        T_cam_actual = kin.forward(sim.get_arm_positions(), frame=CAMERA_FRAME)
        pos_err = float(np.linalg.norm(T_cam_actual[:3, 3] - cam_xyz))
        max_pos = max(max_pos, pos_err)
        max_rot_rad = max(max_rot_rad, _rot_err_rad(T_cam_target[:3, :3], T_cam_actual[:3, :3]))
    return arm_q, max_pos, max_rot_rad


def hold_arm(sim: Simulation, arm_q: np.ndarray, n_frames: int,
             gripper_ctrl: tuple[float, float],
             prox_id: int, dist_id: int, viewer=None):
    for _ in range(n_frames):
        sim.set_arm_commands(arm_q)
        sim.data.ctrl[prox_id] = gripper_ctrl[0]
        sim.data.ctrl[dist_id] = gripper_ctrl[1]
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        if viewer is not None:
            viewer.sync()


def close_gripper(sim: Simulation, arm_q: np.ndarray, n_frames: int,
                  prox_id: int, dist_id: int, viewer=None):
    for i in range(n_frames):
        t = (i + 1) / n_frames
        sim.set_arm_commands(arm_q)
        sim.data.ctrl[prox_id] = (1.0 - t) * PROXIMAL_OPEN + t * PROXIMAL_CLOSED
        sim.data.ctrl[dist_id] = (1.0 - t) * DISTAL_OPEN + t * DISTAL_CLOSED
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        if viewer is not None:
            viewer.sync()


# --------------------------------------------------------------------------- #
# Trial
# --------------------------------------------------------------------------- #


def run_trial(rng: np.random.Generator, checker: IKFeasibilityChecker,
              max_ik_attempts: int, use_viewer: bool = False):
    """Sample an arm-feasible plan and replay it on the arm scene.

    Returns dict with: success (bool), max_pos_mm, max_rot_deg, ik_attempts,
    cube_xy, status ("OK", "FAIL", "IK_FAIL").
    """
    # Reuse the dataset collector's rejection sampler so we replay exactly the
    # plans the collector would produce.
    def builder():
        wp = sample_episode_waypoints(rng)
        return episode_target_poses(wp), wp

    _poses, wp, stats = checker.sample_feasible_trajectory(
        builder, max_attempts=max_ik_attempts,
    )
    if wp is None:
        return dict(success=False, max_pos_mm=None, max_rot_deg=None,
                    ik_attempts=stats.n_attempts, cube_xy=None, status="IK_FAIL")

    sim = Simulation(scene_xml=SCENE)
    kin = Kinematics()
    viewer = sim.launch_passive_viewer() if use_viewer else None

    prox_id = sim.model.actuator("proximal").id
    dist_id = sim.model.actuator("distal").id
    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    cube_qadr = sim.model.jnt_qposadr[cube_jnt_id]

    set_cube_pose(sim, cube_qadr, float(wp.cube_xyz[0]), float(wp.cube_xyz[1]), CUBE_START_Z)

    # Initial arm pose: IK-snap to the planned home pose so we start from the
    # same configuration the demo collector started the gripper in. Convert
    # body-frame home target to site frame before calling IK.
    home_cam_xyz, home_cam_quat = body_pose_to_camera_pose(wp.home_xyz, wp.home_quat)
    home_T = pose_T(home_cam_xyz, home_cam_quat)
    home_arm_q = kin.inverse(home_T, current_joint_positions=ARM_IK_SEED.copy(),
                              n_iter=200, frame=CAMERA_FRAME)
    sim.reset_arm(home_arm_q)
    sim.data.ctrl[prox_id] = PROXIMAL_OPEN
    sim.data.ctrl[dist_id] = DISTAL_OPEN

    arm_q = home_arm_q
    max_pos = 0.0
    max_rot_rad = 0.0

    hold_arm(sim, arm_q, FRAMES_INITIAL_SETTLE, (PROXIMAL_OPEN, DISTAL_OPEN),
             prox_id, dist_id, viewer)
    cube_pos_start = sim.data.body("red_cube").xpos.copy()

    # Approach: home -> mid -> sentry (60/40 split, same as collector).
    seg1 = int(FRAMES_APPROACH * 0.60)
    seg2 = FRAMES_APPROACH - seg1
    arm_q, p, r = drive_targets(
        sim, kin, arm_q,
        multiwaypoint_targets(
            [wp.home_xyz, wp.mid_xyz, wp.sentry_xyz],
            [wp.home_quat, wp.mid_quat, wp.sentry_quat],
            [seg1, seg2],
        ),
        (PROXIMAL_OPEN, DISTAL_OPEN), prox_id, dist_id, viewer,
    )
    max_pos = max(max_pos, p); max_rot_rad = max(max_rot_rad, r)

    # Descend: sentry -> grasp.
    arm_q, p, r = drive_targets(
        sim, kin, arm_q,
        smooth_segment_targets(wp.sentry_xyz, wp.sentry_quat, wp.grasp_xyz, wp.grasp_quat,
                                FRAMES_DESCEND),
        (PROXIMAL_OPEN, DISTAL_OPEN), prox_id, dist_id, viewer,
    )
    max_pos = max(max_pos, p); max_rot_rad = max(max_rot_rad, r)

    hold_arm(sim, arm_q, FRAMES_PRE_GRIP_SETTLE, (PROXIMAL_OPEN, DISTAL_OPEN),
             prox_id, dist_id, viewer)
    close_gripper(sim, arm_q, FRAMES_CLOSE, prox_id, dist_id, viewer)
    hold_arm(sim, arm_q, FRAMES_HOLD, (PROXIMAL_CLOSED, DISTAL_CLOSED),
             prox_id, dist_id, viewer)

    # Lift: grasp -> grasp + (0, 0, LIFT_HEIGHT).
    arm_q, p, r = drive_targets(
        sim, kin, arm_q,
        smooth_segment_targets(wp.grasp_xyz, wp.grasp_quat, wp.lift_xyz, wp.lift_quat,
                                FRAMES_LIFT),
        (PROXIMAL_CLOSED, DISTAL_CLOSED), prox_id, dist_id, viewer,
    )
    max_pos = max(max_pos, p); max_rot_rad = max(max_rot_rad, r)

    # Retract: lift -> home + RETRACT_EXTRA, slerp orientation back to home_quat.
    retract_xyz = wp.home_xyz + np.array([0.0, 0.0, RETRACT_EXTRA])
    arm_q, p, r = drive_targets(
        sim, kin, arm_q,
        smooth_segment_targets(wp.lift_xyz, wp.grasp_quat, retract_xyz, wp.home_quat,
                                FRAMES_RETRACT),
        (PROXIMAL_CLOSED, DISTAL_CLOSED), prox_id, dist_id, viewer,
    )
    max_pos = max(max_pos, p); max_rot_rad = max(max_rot_rad, r)

    hold_arm(sim, arm_q, FRAMES_FINAL_SETTLE, (PROXIMAL_CLOSED, DISTAL_CLOSED),
             prox_id, dist_id, viewer)

    cube_final = sim.data.body("red_cube").xpos.copy()
    success = bool((cube_final[2] - cube_pos_start[2]) > LIFT_SUCCESS_THRESHOLD)

    if viewer is not None:
        viewer.close()

    return dict(
        success=success,
        max_pos_mm=max_pos * 1000.0,
        max_rot_deg=float(np.degrees(max_rot_rad)),
        ik_attempts=stats.n_attempts,
        cube_xy=(float(wp.cube_xyz[0]), float(wp.cube_xyz[1])),
        status="OK" if success else "FAIL",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Replay Grabette plans on the simulated arm to check dynamic feasibility."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--max_ik_attempts", type=int, default=200)
    parser.add_argument("--ik_pos_tol", type=float, default=0.015)
    parser.add_argument("--ik_rot_tol_deg", type=float, default=5.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    checker = IKFeasibilityChecker(
        Kinematics(),
        frame=CAMERA_FRAME,
        seed_joints=ARM_IK_SEED,
        pos_tol_m=args.ik_pos_tol,
        rot_tol_deg=args.ik_rot_tol_deg,
        n_iter=200,
    )

    print(f"Replaying {args.episodes} Grabette plans on the arm scene ({SCENE.name})")
    print(f"{'ep':>3} {'cube_xy':>14} {'ik_try':>6} {'max_pos':>9} {'max_rot':>8} {'result':>7}")
    n_success = 0
    n_ik_fail = 0
    pos_errs = []
    rot_errs = []
    t_start = time.perf_counter()
    for ep in range(args.episodes):
        r = run_trial(rng, checker, args.max_ik_attempts, args.viewer)
        if r["status"] == "IK_FAIL":
            n_ik_fail += 1
            print(f"{ep:3d} {'-':>14} {r['ik_attempts']:6d} {'-':>9} {'-':>8} {'IK_FAIL':>7}")
            continue
        cx, cy = r["cube_xy"]
        cube_str = f"({cx:+.3f},{cy:+.3f})"
        print(f"{ep:3d} {cube_str:>14} {r['ik_attempts']:6d} "
              f"{r['max_pos_mm']:7.1f}mm {r['max_rot_deg']:7.1f}° "
              f"{r['status']:>7}")
        if r["success"]:
            n_success += 1
        pos_errs.append(r["max_pos_mm"])
        rot_errs.append(r["max_rot_deg"])

    dt = time.perf_counter() - t_start
    n_eval = args.episodes - n_ik_fail
    rate = (100.0 * n_success / max(n_eval, 1)) if n_eval else 0.0
    print()
    print(f"Replayed {n_eval}/{args.episodes} plans (IK-rejected: {n_ik_fail}); "
          f"{n_success}/{n_eval} succeeded ({rate:.1f}%) in {dt:.1f}s.")
    if pos_errs:
        print(f"Tracking error stats over replayed trials: "
              f"max_pos median={np.median(pos_errs):.1f}mm "
              f"p95={np.percentile(pos_errs, 95):.1f}mm; "
              f"max_rot median={np.median(rot_errs):.1f}° "
              f"p95={np.percentile(rot_errs, 95):.1f}°")
    if rate >= 70.0:
        print("Arm dynamic feasibility looks good — IK-passing demos are also executable.")
    elif rate >= 30.0:
        print("Partial executability. Inspect the failures: are the tracking errors small "
              "(physics/grasp issue) or large (arm can't follow the target rate)?")
    else:
        print("Most replays fail. Consider slowing trajectory playback, adding settling "
              "between segments, or relaxing IK tolerance to admit only the most "
              "comfortably-feasible plans.")


if __name__ == "__main__":
    main()
