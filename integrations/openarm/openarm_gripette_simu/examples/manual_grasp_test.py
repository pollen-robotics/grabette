"""Scripted grasp-and-lift test for the free-floating "virtual Grabette".

Stage 4: random DIAGONAL grasp orientation per trial. The final grasp pose is
no longer pure top-down; instead each trial samples a random tilt off vertical
and azimuth around world +Z. The sentry waypoint sits BACK along the gripper's
local approach axis (body -Y) so the final segment is a clean linear push
along the gripper's approach direction. The lift remains world-+Z aligned.

This mirrors examples/manual_grasp_test_arm.py but drives a 6-DoF mocap
gripper directly, with no arm in the loop. The Grabette body has a
freejoint and is anchored to a mocap target via an <equality><weld>
(see scenes/grabette_grasp.xml). We move the mocap and the dynamic body
follows; the gripper is closed via two position-controlled hinge joints.

Stage 3 changes vs the previous "straight down" version:
  * Random "human holding pose" home far from the cube, with the gripper
    pitched 35-65 deg below horizontal and slight yaw jitter.
  * Smoothstep (cubic ease-in-out) interpolation in position; SLERP in
    orientation. Multi-waypoint approach with an arc-shaped mid-point.
  * The gripper rotates from the tilted home pose down to fingers-fully-
    down at the pre-grasp / grasp pose.

Trajectory phases (recorded-frame counts at 50 fps -> 20 ms / frame):
    1. Initial settle (open, hold home)             ~ 5 frames
    2. Approach: home -> mid_approach -> pre_grasp ~80 frames (smoothstep + slerp)
    3. Descend: pre_grasp -> grasp_pose            ~25 frames
    4. Pre-grip settle                             ~15 frames
    5. Close (ramp gripper ctrl)                   ~25 frames
    6. Hold closed                                 ~25 frames
    7. Lift: grasp_pose -> +10 cm                  ~30 frames
    8. Retract: lift_pose -> home + extra          ~50 frames
    9. Final settle                                ~15 frames
Total ~270 frames per episode (~5.4 s at 50 fps).

Success: cube z lifted by >= 5 cm.

Usage:
    uv run python examples/manual_grasp_test.py --trials 10
    uv run python examples/manual_grasp_test.py --trials 10 --viewer
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

# Make sibling module importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grabette_trajectory import (  # noqa: E402
    CUBE_START_Z,
    CUBE_X_RANGE,
    CUBE_Y_RANGE,
    DISTAL_CLOSED,
    DISTAL_OPEN,
    LIFT_HEIGHT,
    LIFT_SUCCESS_THRESHOLD,
    PROXIMAL_CLOSED,
    PROXIMAL_OPEN,
    RETRACT_EXTRA,
    SENTRY_OFFSET,
    grasp_pos_for_cube,
    mid_approach_pose,
    sample_grasp_pose,
    sample_home_pose,
    sentry_pose,
    slerp_quat,
    smoothstep,
)
from openarm_gripette_simu import Simulation


# Phase step counts (sim timestep is 0.001 s -> 1 step = 1 ms).
# We keep the same wall-clock per phase as the recorded-frame counts in
# collect_grasp_dataset.py: STEPS = recorded_frames * 20 (SIM_SUBSTEPS).
# Total ~ 270 frames * 20 = 5400 sim steps = 5.4 s.
STEPS_INITIAL_SETTLE = 100      # 5 frames
STEPS_APPROACH = 1600           # 80 frames: home -> mid -> sentry (2 segments)
STEPS_DESCEND = 500             # 25 frames: sentry -> grasp_pose (linear approach)
# Match the original test's pre-grip-settle / close / hold timings.
# Slower close gives the cube too much time to escape before the distal
# scoops underneath; the original 200-step close was tuned for ~100% success.
STEPS_PRE_GRIP_SETTLE = 200     # 10 frames @ 20 substeps in dataset
STEPS_CLOSE = 200               # 10 frames: ramp ctrl
STEPS_HOLD = 100                # 5 frames — short hold so cube gets
                                # lifted before slipping out of a tilted
                                # V-pocket
STEPS_LIFT = 600                # 30 frames
STEPS_RETRACT = 1000            # 50 frames
STEPS_FINAL_SETTLE = 300        # 15 frames


def set_cube_pose(sim: Simulation, cube_qadr: int, x: float, y: float, z: float):
    """Place the cube at (x, y, z) with identity orientation, zero velocity."""
    sim.data.qpos[cube_qadr:cube_qadr + 3] = (x, y, z)
    sim.data.qpos[cube_qadr + 3:cube_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    qveladr = sim.model.jnt_dofadr[cube_jnt_id]
    sim.data.qvel[qveladr:qveladr + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def set_grabette_pose(sim: Simulation, mocap_id: int, pos: np.ndarray, quat: np.ndarray):
    """Teleport BOTH the mocap and the welded freejoint body to (pos, quat).

    Used at trial reset only — during runtime we move only the mocap and let
    the weld pull the dynamic body along.
    """
    sim.data.mocap_pos[mocap_id] = pos
    sim.data.mocap_quat[mocap_id] = quat
    free_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "grabette_freejoint")
    qadr = sim.model.jnt_qposadr[free_jnt_id]
    sim.data.qpos[qadr:qadr + 3] = pos
    sim.data.qpos[qadr + 3:qadr + 7] = quat
    qveladr = sim.model.jnt_dofadr[free_jnt_id]
    sim.data.qvel[qveladr:qveladr + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


# -----------------------------------------------------------------------------
# Smooth segment runners (sim-step granularity for the manual test)
# -----------------------------------------------------------------------------

def run_smooth_segment(sim: Simulation, mocap_id: int,
                       start_pos: np.ndarray, start_quat: np.ndarray,
                       end_pos: np.ndarray, end_quat: np.ndarray,
                       n_steps: int, gripper_ctrl: tuple,
                       prox_id: int, dist_id: int,
                       viewer=None):
    """Drive the mocap from (start_pos, start_quat) to (end_pos, end_quat)
    using cubic smoothstep on position and SLERP on orientation, one sim
    step per inner iteration.
    """
    for i in range(n_steps):
        t = (i + 1) / n_steps
        s = smoothstep(t)
        sim.data.mocap_pos[mocap_id] = start_pos + s * (end_pos - start_pos)
        sim.data.mocap_quat[mocap_id] = slerp_quat(start_quat, end_quat, s)
        sim.data.ctrl[prox_id] = gripper_ctrl[0]
        sim.data.ctrl[dist_id] = gripper_ctrl[1]
        sim.step()
        if viewer is not None:
            viewer.sync()


def run_multiwaypoint_smooth(sim: Simulation, mocap_id: int,
                              waypoints_pos: list[np.ndarray],
                              waypoints_quat: list[np.ndarray],
                              segment_steps: list[int],
                              gripper_ctrl: tuple,
                              prox_id: int, dist_id: int,
                              viewer=None):
    """Run a multi-segment smoothstep+slerp path through a list of waypoints.

    waypoints_pos/_quat must have len = N+1; segment_steps must have len = N.
    Each segment runs `segment_steps[k]` sim steps from waypoint k to k+1.
    """
    assert len(waypoints_pos) == len(waypoints_quat)
    assert len(segment_steps) == len(waypoints_pos) - 1
    for k, n in enumerate(segment_steps):
        run_smooth_segment(
            sim, mocap_id,
            waypoints_pos[k], waypoints_quat[k],
            waypoints_pos[k + 1], waypoints_quat[k + 1],
            n, gripper_ctrl, prox_id, dist_id, viewer,
        )


def hold_mocap(sim: Simulation, mocap_id: int, pos: np.ndarray, quat: np.ndarray,
               n_steps: int, gripper_ctrl: tuple, prox_id: int, dist_id: int,
               viewer=None):
    """Hold the mocap pose and gripper command for n_steps."""
    for _ in range(n_steps):
        sim.data.mocap_pos[mocap_id] = pos
        sim.data.mocap_quat[mocap_id] = quat
        sim.data.ctrl[prox_id] = gripper_ctrl[0]
        sim.data.ctrl[dist_id] = gripper_ctrl[1]
        sim.step()
        if viewer is not None:
            viewer.sync()


def ramp_close(sim: Simulation, mocap_id: int, pos: np.ndarray, quat: np.ndarray,
               n_steps: int, prox_id: int, dist_id: int, viewer=None):
    """Ramp gripper ctrl from open to closed while holding mocap fixed."""
    for i in range(n_steps):
        t = (i + 1) / n_steps
        sim.data.mocap_pos[mocap_id] = pos
        sim.data.mocap_quat[mocap_id] = quat
        sim.data.ctrl[prox_id] = (1.0 - t) * PROXIMAL_OPEN + t * PROXIMAL_CLOSED
        sim.data.ctrl[dist_id] = (1.0 - t) * DISTAL_OPEN + t * DISTAL_CLOSED
        sim.step()
        if viewer is not None:
            viewer.sync()


# -----------------------------------------------------------------------------
# Single trial
# -----------------------------------------------------------------------------

def run_trial(scene_xml: Path, rng: np.random.Generator, use_viewer: bool = False):
    """Run a single grasp-and-lift trial.

    Returns:
        (success, cube_xy_start, home_xyz, home_debug, final_cube_z, distance_traveled)
    """
    sim = Simulation(scene_xml=scene_xml)
    viewer_handle = sim.launch_passive_viewer() if use_viewer else None

    # Resolve ids once
    mocap_id = sim.model.body("grabette_mocap").mocapid[0]
    prox_id = sim.model.actuator("proximal").id
    dist_id = sim.model.actuator("distal").id
    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    cube_qadr = sim.model.jnt_qposadr[cube_jnt_id]

    # 1. Random cube position
    cube_x = float(rng.uniform(*CUBE_X_RANGE))
    cube_y = float(rng.uniform(*CUBE_Y_RANGE))
    set_cube_pose(sim, cube_qadr, cube_x, cube_y, CUBE_START_Z)
    cube_xyz_start_planned = np.array([cube_x, cube_y, CUBE_START_Z])

    # 2. Sample a "human holding" home pose (in front of, not above, the cube)
    home_xyz, home_quat, home_dbg = sample_home_pose(rng, cube_x, cube_y)

    # 3. Stage 4: sample a random DIAGONAL grasp orientation per trial
    # (pitch from vertical in [30, 60] deg, azimuth in [-60, +60] deg).
    grasp_quat, grasp_dbg = sample_grasp_pose(rng)
    grasp_xyz = grasp_pos_for_cube(cube_xyz_start_planned, grasp_quat)

    # 3b. Sentry waypoint: stand back along the gripper's approach axis so
    # the final segment (sentry -> grasp) is a clean linear push along that
    # axis. For a vertical grasp this is "10 cm above"; for a diagonal grasp
    # it sits up-and-behind.
    sentry_xyz, sentry_quat = sentry_pose(grasp_xyz, grasp_quat, SENTRY_OFFSET)

    # 3c. Mid-approach waypoint: arcs upward between home and the sentry,
    # carrying the orientation rotation from home_quat to grasp_quat. Doing
    # the rotation in the first segment and keeping the second segment at a
    # fixed grasp_quat avoids sweeping the gripper geometry while rotating.
    mid_xyz, mid_quat = mid_approach_pose(home_xyz, home_quat,
                                          sentry_xyz, grasp_quat,
                                          arc_lift=0.05)

    # 4. Reset Grabette to home and open the gripper
    set_grabette_pose(sim, mocap_id, home_xyz, home_quat)
    sim.data.ctrl[prox_id] = PROXIMAL_OPEN
    sim.data.ctrl[dist_id] = DISTAL_OPEN

    # 5. Settle a few steps so the weld stabilizes at home
    hold_mocap(sim, mocap_id, home_xyz, home_quat,
               STEPS_INITIAL_SETTLE, (PROXIMAL_OPEN, DISTAL_OPEN),
               prox_id, dist_id, viewer_handle)

    # Cube starting position after the settle (for the success check)
    cube_pos_start = sim.data.body("red_cube").xpos.copy()

    # 6. Approach: home -> mid -> sentry
    # Two segments. The first carries the orientation rotation from home_quat
    # to grasp_quat (and bows upward via the arc-lifted mid). The second
    # arrives at the sentry, gripper already aligned to grasp_quat.
    # Allocate ~60% to seg1 (orientation + larger move) and ~40% to seg2.
    approach_seg1 = int(STEPS_APPROACH * 0.60)
    approach_seg2 = STEPS_APPROACH - approach_seg1
    run_multiwaypoint_smooth(
        sim, mocap_id,
        waypoints_pos=[home_xyz, mid_xyz, sentry_xyz],
        waypoints_quat=[home_quat, mid_quat, sentry_quat],
        segment_steps=[approach_seg1, approach_seg2],
        gripper_ctrl=(PROXIMAL_OPEN, DISTAL_OPEN),
        prox_id=prox_id, dist_id=dist_id, viewer=viewer_handle,
    )

    # 7. Descend: sentry -> grasp_xyz (linear push along gripper approach axis,
    # orientation fixed at grasp_quat).
    run_smooth_segment(
        sim, mocap_id,
        sentry_xyz, grasp_quat, grasp_xyz, grasp_quat,
        STEPS_DESCEND, (PROXIMAL_OPEN, DISTAL_OPEN),
        prox_id, dist_id, viewer_handle,
    )

    # 8. Pre-grip settle (let weld converge before closing)
    hold_mocap(sim, mocap_id, grasp_xyz, grasp_quat,
               STEPS_PRE_GRIP_SETTLE, (PROXIMAL_OPEN, DISTAL_OPEN),
               prox_id, dist_id, viewer_handle)

    # 9. Close gripper (mocap stays put)
    ramp_close(sim, mocap_id, grasp_xyz, grasp_quat,
               STEPS_CLOSE, prox_id, dist_id, viewer_handle)

    # 10. Hold closed
    hold_mocap(sim, mocap_id, grasp_xyz, grasp_quat,
               STEPS_HOLD, (PROXIMAL_CLOSED, DISTAL_CLOSED),
               prox_id, dist_id, viewer_handle)

    # 11. Lift (smoothstep, orientation held at grasp_quat)
    lift_target = grasp_xyz + np.array([0.0, 0.0, LIFT_HEIGHT])
    run_smooth_segment(
        sim, mocap_id,
        grasp_xyz, grasp_quat, lift_target, grasp_quat,
        STEPS_LIFT, (PROXIMAL_CLOSED, DISTAL_CLOSED),
        prox_id, dist_id, viewer_handle,
    )

    # 12. Retract: lift_target -> home + 10 cm extra, slerp orientation back
    retract_target = home_xyz + np.array([0.0, 0.0, RETRACT_EXTRA])
    run_smooth_segment(
        sim, mocap_id,
        lift_target, grasp_quat, retract_target, home_quat,
        STEPS_RETRACT, (PROXIMAL_CLOSED, DISTAL_CLOSED),
        prox_id, dist_id, viewer_handle,
    )

    # 13. Final settle
    hold_mocap(sim, mocap_id, retract_target, home_quat,
               STEPS_FINAL_SETTLE, (PROXIMAL_CLOSED, DISTAL_CLOSED),
               prox_id, dist_id, viewer_handle)

    cube_final = sim.data.body("red_cube").xpos.copy()
    success = (cube_final[2] - cube_pos_start[2]) > LIFT_SUCCESS_THRESHOLD

    # Approximate mocap travel distance: sum of straight-line segment lengths.
    # (smoothstep doesn't change the path length much.)
    travel = (
        np.linalg.norm(mid_xyz - home_xyz)
        + np.linalg.norm(sentry_xyz - mid_xyz)
        + np.linalg.norm(grasp_xyz - sentry_xyz)
        + np.linalg.norm(lift_target - grasp_xyz)
        + np.linalg.norm(retract_target - lift_target)
    )

    if viewer_handle is not None:
        viewer_handle.close()

    return (success, (cube_pos_start[0], cube_pos_start[1]),
            tuple(home_xyz.tolist()), home_dbg, grasp_dbg,
            float(cube_final[2]), float(travel))


def main():
    parser = argparse.ArgumentParser(
        description="Scripted grasp-and-lift test for the free-floating "
                    "virtual Grabette (no arm) — Stage 4 diagonal grasp.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--viewer", action="store_true",
                        help="Launch a passive viewer (off by default)")
    parser.add_argument("--scene", type=str, default="scenes/grabette_grasp.xml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scene_path = Path(args.scene).resolve()
    rng = np.random.default_rng(args.seed)

    n_success = 0
    travels = []
    print(f"Running {args.trials} trials on scene {scene_path}")
    print(f"{'trial':>5} {'cube_x':>7} {'cube_y':>7} "
          f"{'home_x':>7} {'home_y':>7} {'home_z':>7} "
          f"{'h_pitch':>7} {'h_yaw':>6} "
          f"{'g_pitch':>7} {'g_azim':>7} "
          f"{'final_z':>8} {'result':>6}")
    for trial in range(args.trials):
        success, (cx, cy), home, home_dbg, grasp_dbg, final_z, travel = run_trial(
            scene_path, rng, use_viewer=args.viewer
        )
        n_success += int(success)
        travels.append(travel)
        result = "OK" if success else "FAIL"
        print(f"{trial:5d} {cx:7.3f} {cy:7.3f} "
              f"{home[0]:7.3f} {home[1]:7.3f} {home[2]:7.3f} "
              f"{home_dbg['pitch_deg']:7.1f} {home_dbg['yaw_deg']:6.1f} "
              f"{grasp_dbg['grasp_pitch_deg']:7.1f} {grasp_dbg['grasp_azimuth_deg']:7.1f} "
              f"{final_z:8.4f} {result:>6}")

    rate = 100.0 * n_success / args.trials
    print(f"\nSuccess rate: {n_success}/{args.trials} ({rate:.1f}%)")
    print(f"Mean traversed distance: {np.mean(travels):.3f} m")


if __name__ == "__main__":
    main()
