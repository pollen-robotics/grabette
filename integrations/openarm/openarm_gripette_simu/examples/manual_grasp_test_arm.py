"""Scripted grasp-and-lift test for the Gripette gripper in MuJoCo
(arm-based variant; for the free-floating Grabette version see
manual_grasp_test.py / collect_grasp_dataset.py).

For each trial: randomize a cube position on the table, IK the arm above the
cube, descend, close the gripper, hold, lift. Success if the cube ends up
at least 5 cm above its starting z. Reports per-trial result and summary.

Usage:
    uv run python examples/manual_grasp_test_arm.py --trials 10
    uv run python examples/manual_grasp_test_arm.py --trials 10 --viewer
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from openarm_gripette_simu import Simulation, Kinematics
from openarm_gripette_simu.kinematics import GRIPPER_FRAME


# Cube placement region. The original 15x15 cm region (x∈[0.40,0.55],
# y∈[-0.20,-0.05]) extends well past the arm's reach when the gripper is
# constrained to point straight down. The reachable workspace at this
# orientation is roughly x∈[0.39,0.45], y∈[-0.20,-0.10]. We use a 6x10 cm
# region inside that, centered slightly behind the seed position.
CUBE_X_RANGE = (0.39, 0.45)
CUBE_Y_RANGE = (-0.20, -0.10)
CUBE_HALF_HEIGHT = 0.06       # cube is 4x4x12 cm, half of 12cm = 0.06m
TABLE_TOP_Z = 0.35            # tabletop top surface (table.pos.z + size.z = 0.34 + 0.01)
CUBE_START_Z = TABLE_TOP_Z + CUBE_HALF_HEIGHT  # 0.41m

# Approach geometry
APPROACH_HEIGHT = 0.05        # 5 cm above the cube center for pre-grasp
LIFT_HEIGHT = 0.10            # 10 cm above the cube's original position for lift

# Gripper joint limits (from robot.xml)
PROXIMAL_OPEN = 0.0
PROXIMAL_CLOSED = -np.pi / 2          # -1.5708 rad
DISTAL_OPEN = 0.0
DISTAL_CLOSED = -2.0 * np.pi / 3      # -2.0944 rad

# Phase step counts (timestep is 0.001 s with current physics → 1ms per step)
STEPS_PRE_SETTLE = 100        # let arm reach the above-cube pose
STEPS_DESCEND = 200           # linear IK from above-cube to cube center
STEPS_CLOSE = 200             # ramp ctrl from open to closed
STEPS_HOLD = 1000             # hold closed (1 s at 1ms timestep)
STEPS_LIFT = 200              # linear IK to lift pose
STEPS_FINAL_SETTLE = 200

# Seed start config for the arm. At this configuration, the gripper is above
# the table region (~ x=0.40, y=-0.135, z=0.43) with gripper z-axis pointing
# nearly straight down. The IK uses this seed AND keeps the seed's rotation
# matrix as the orientation target — that is the natural "gripper down"
# orientation reachable in this part of the workspace. Using a target like
# diag(1,-1,-1) instead drives several joints to their limits.
SEED_JOINTS = np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.5])

# Success threshold: cube must lift by at least 5 cm
LIFT_SUCCESS_THRESHOLD = 0.05


def make_pose(target_xyz: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform with the given rotation R and position target_xyz."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = target_xyz
    return T


def set_cube_pose(sim, cube_qadr, x, y, z):
    """Place the cube at (x, y, z) with identity orientation, zero velocity."""
    sim.data.qpos[cube_qadr:cube_qadr + 3] = (x, y, z)
    sim.data.qpos[cube_qadr + 3:cube_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    # Also zero the freejoint velocity (qvel uses 6 entries for a freejoint)
    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    qveladr = sim.model.jnt_dofadr[cube_jnt_id]
    sim.data.qvel[qveladr:qveladr + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def run_phase_to_xyz(sim, kin, arm_q, target_xyz, R_down, n_steps,
                     gripper_ctrl, viewer=None):
    """Linearly interpolate the gripper position to target_xyz over n_steps.

    Holds gripper orientation at R_down throughout. `gripper_ctrl` is a
    (proximal, distal) pair applied each step. Returns the final arm joint
    positions.
    """
    T_now = kin.forward(arm_q, frame=GRIPPER_FRAME)
    start_xyz = T_now[:3, 3].copy()

    for i in range(n_steps):
        t = (i + 1) / n_steps
        interp_xyz = start_xyz + t * (target_xyz - start_xyz)
        T_target = make_pose(interp_xyz, R_down)
        arm_q = kin.inverse(T_target, current_joint_positions=arm_q,
                            n_iter=50, frame=GRIPPER_FRAME)

        sim.set_arm_commands(arm_q)
        sim.set_joint_commands(np.array([gripper_ctrl[0], gripper_ctrl[1]]),
                               joint_names=["proximal", "distal"])
        sim.step()
        if viewer is not None:
            viewer.sync()
        arm_q = sim.get_arm_positions()

    return arm_q


def run_close_phase(sim, arm_q, n_steps, viewer=None):
    """Ramp gripper ctrl from open to closed while holding the arm in place."""
    for i in range(n_steps):
        t = (i + 1) / n_steps
        prox_cmd = (1.0 - t) * PROXIMAL_OPEN + t * PROXIMAL_CLOSED
        dist_cmd = (1.0 - t) * DISTAL_OPEN + t * DISTAL_CLOSED
        sim.set_arm_commands(arm_q)
        sim.set_joint_commands(np.array([prox_cmd, dist_cmd]),
                               joint_names=["proximal", "distal"])
        sim.step()
        if viewer is not None:
            viewer.sync()


def hold_phase(sim, arm_q, n_steps, gripper_ctrl, viewer=None):
    """Hold the arm and gripper command for n_steps."""
    for _ in range(n_steps):
        sim.set_arm_commands(arm_q)
        sim.set_joint_commands(np.array([gripper_ctrl[0], gripper_ctrl[1]]),
                               joint_names=["proximal", "distal"])
        sim.step()
        if viewer is not None:
            viewer.sync()


def run_trial(scene_xml, rng, use_viewer=False):
    """Run a single grasp-and-lift trial. Returns (success, cube_start_xy, cube_final_z).

    Each trial loads a fresh Simulation so physics state is fully reset.
    """
    sim = Simulation(scene_xml=scene_xml)
    kin = Kinematics()
    viewer_handle = sim.launch_passive_viewer() if use_viewer else None

    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    cube_qadr = sim.model.jnt_qposadr[cube_jnt_id]

    # Use the FK at SEED_JOINTS to define the canonical "gripper down"
    # rotation. This is the orientation reachable at the relevant part of
    # the workspace (gripper z ≈ -world_z, gripper x ≈ -world_y).
    T_seed = kin.forward(SEED_JOINTS, frame=GRIPPER_FRAME)
    R_down = T_seed[:3, :3].copy()

    # 1. Random cube position
    cube_x = rng.uniform(*CUBE_X_RANGE)
    cube_y = rng.uniform(*CUBE_Y_RANGE)
    set_cube_pose(sim, cube_qadr, cube_x, cube_y, CUBE_START_Z)

    # 2. IK to a pose 5 cm above the cube, gripper pointing down
    pre_target = np.array([cube_x, cube_y, CUBE_START_Z + APPROACH_HEIGHT])
    T_pre = make_pose(pre_target, R_down)
    arm_q = kin.inverse(T_pre, current_joint_positions=SEED_JOINTS,
                        n_iter=200, frame=GRIPPER_FRAME)
    sim.reset_arm(arm_q)

    # Open the gripper from the start
    sim.set_joint_commands(np.array([PROXIMAL_OPEN, DISTAL_OPEN]),
                           joint_names=["proximal", "distal"])

    # 3. Settle the arm at the pre-grasp pose
    hold_phase(sim, arm_q, STEPS_PRE_SETTLE, (PROXIMAL_OPEN, DISTAL_OPEN), viewer_handle)

    # Cube starting position after settle (used for lift target reference)
    cube_pos_start = sim.data.body("red_cube").xpos.copy()

    # 4. Descend to the cube center (gripper site at the cube center)
    descend_target = np.array([cube_x, cube_y, CUBE_START_Z])
    arm_q = run_phase_to_xyz(sim, kin, arm_q, descend_target, R_down,
                             STEPS_DESCEND,
                             (PROXIMAL_OPEN, DISTAL_OPEN), viewer_handle)

    # 5. Close the gripper
    run_close_phase(sim, arm_q, STEPS_CLOSE, viewer_handle)

    # 6. Hold closed
    hold_phase(sim, arm_q, STEPS_HOLD, (PROXIMAL_CLOSED, DISTAL_CLOSED), viewer_handle)

    # 7. Lift to 10 cm above the cube's original position
    lift_target = np.array([cube_pos_start[0], cube_pos_start[1],
                            cube_pos_start[2] + LIFT_HEIGHT])
    arm_q = run_phase_to_xyz(sim, kin, arm_q, lift_target, R_down, STEPS_LIFT,
                             (PROXIMAL_CLOSED, DISTAL_CLOSED), viewer_handle)

    # 8. Final settle
    hold_phase(sim, arm_q, STEPS_FINAL_SETTLE, (PROXIMAL_CLOSED, DISTAL_CLOSED), viewer_handle)

    cube_final = sim.data.body("red_cube").xpos.copy()
    success = (cube_final[2] - cube_pos_start[2]) > LIFT_SUCCESS_THRESHOLD
    if viewer_handle is not None:
        viewer_handle.close()
    return success, (cube_pos_start[0], cube_pos_start[1]), cube_final[2]


def main():
    parser = argparse.ArgumentParser(
        description="Scripted grasp-and-lift sim test (arm-based variant; "
                    "for the free-floating Grabette version see "
                    "manual_grasp_test.py / collect_grasp_dataset.py)")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--viewer", action="store_true",
                        help="Launch a passive viewer (off by default)")
    parser.add_argument("--scene", type=str, default="scenes/table_grasp.xml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scene_path = Path(args.scene).resolve()
    rng = np.random.default_rng(args.seed)

    n_success = 0
    print(f"Running {args.trials} trials on scene {scene_path}")
    print(f"{'trial':>5} {'cube_x':>8} {'cube_y':>8} {'final_z':>9} {'result':>8}")
    for trial in range(args.trials):
        success, (cx, cy), final_z = run_trial(scene_path, rng, use_viewer=args.viewer)
        n_success += int(success)
        result = "OK" if success else "FAIL"
        print(f"{trial:5d} {cx:8.3f} {cy:8.3f} {final_z:9.4f} {result:>8}")

    print(f"\nSuccess rate: {n_success}/{args.trials} ({100.0 * n_success / args.trials:.1f}%)")


if __name__ == "__main__":
    main()
