"""Collect scripted grasp-and-lift demonstrations as a LeRobot-format dataset
(arm-based variant; for the free-floating Grabette version see
manual_grasp_test.py / collect_grasp_dataset.py).

Each episode runs a six-phase trajectory (approach -> descend -> close -> hold
-> lift -> settle) that picks up a 4x4x12 cm cube on the table. Successful
episodes (cube lifted >= 5 cm above its starting z) are saved as a single
LeRobotDataset on disk; failures are discarded.

Trajectory and physics are inherited from Stage 1 (`manual_grasp_test_arm.py`):
    * scene: `scenes/table_grasp.xml`
    * IK seed: [1.0, 0, 0, 0.5, 0, 0, 1.5]  (gripper ~ down)
    * orientation target: FK rotation of that seed
    * cube spawn region: x in [0.39, 0.45], y in [-0.20, -0.10]

Schema written here matches `collect_grasp_data.py` (Stage 0):
    * observation.images.cam0: video, (3, 972, 1296) uint8
    * action: float32 (8,) [x, y, z, ax, ay, az, proximal_rad, distal_rad]
    * task: constant string
    * 50 fps. Per-frame action is the absolute pose at the next frame, so
      `convert_dataset.py` can compute deltas from neighbour frames.

Usage:
    uv run python examples/collect_grasp_dataset_arm.py --episodes 10 --repo_id sim_grasp
"""

import argparse
import logging
import time
from pathlib import Path

import mujoco
import numpy as np

from openarm_gripette_simu import Kinematics, Simulation
from openarm_gripette_simu.kinematics import GRIPPER_FRAME

# LeRobot dataset writer + axis-angle helper (same imports as Stage 0).
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.rotation import Rotation as LeRobotRotation

logger = logging.getLogger(__name__)

SCENE = Path(__file__).parent.parent / "scenes" / "table_grasp.xml"

# --- Cube placement (Stage 1 reachable workspace at gripper-down orientation) ---
CUBE_X_RANGE = (0.39, 0.45)
CUBE_Y_RANGE = (-0.20, -0.10)
CUBE_HALF_HEIGHT = 0.06            # cube is 4x4x12 cm
TABLE_TOP_Z = 0.35
CUBE_START_Z = TABLE_TOP_Z + CUBE_HALF_HEIGHT  # 0.41

# --- Approach / lift geometry ---
APPROACH_HEIGHT = 0.05             # 5 cm above cube center for pre-grasp
LIFT_HEIGHT = 0.10                 # 10 cm above starting cube position

# --- Gripper joint commands (rad) ---
PROXIMAL_OPEN = 0.0
PROXIMAL_CLOSED = -np.pi / 2
DISTAL_OPEN = 0.0
DISTAL_CLOSED = -2.0 * np.pi / 3

# --- Stage 1 IK seed: gripper z-axis points ~ down at this configuration. ---
SEED_JOINTS = np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.5])

# --- Recorded-frame budget per phase. 50 fps -> 20 ms per recorded frame.
# At sim timestep 0.001 s that is 20 sim sub-steps per frame. ---
FPS = 50
SIM_SUBSTEPS = 20
FRAMES_APPROACH = 30
FRAMES_DESCEND = 10
FRAMES_CLOSE = 20
FRAMES_HOLD = 20
FRAMES_LIFT = 20
FRAMES_SETTLE = 10

# --- Image dimensions (must match Simulation.render_camera) ---
IMG_HEIGHT = 972
IMG_WIDTH = 1296

# --- Constant task label ---
TASK = "grasp_and_lift_cube"

# --- Success threshold ---
LIFT_SUCCESS_THRESHOLD = 0.05


# ----- LeRobot feature plumbing (matches Stage 0) -----------------------------

def build_features() -> dict:
    """LeRobot feature spec; same contract as `collect_grasp_data.py`.

    convert_dataset.py recognizes the 8D action (axis-angle) and expands to
    11D before computing deltas. observation.state is created downstream.
    """
    return {
        "observation.images.cam0": {
            "dtype": "video",
            "shape": (3, IMG_HEIGHT, IMG_WIDTH),
            "names": ["channels", "height", "width"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "x", "y", "z",
                "ax", "ay", "az",
                "proximal", "distal",
            ],
        },
    }


def fk_state_8d(kin: Kinematics, arm_q: np.ndarray, gripper_rad: np.ndarray) -> np.ndarray:
    """Build the 8D absolute pose [x, y, z, ax, ay, az, proximal, distal] in the camera frame.

    Matches Stage 0: rotation as axis-angle (rotvec), gripper in radians.
    """
    T = kin.forward(arm_q, "camera")
    pos = T[:3, 3]
    rotvec = LeRobotRotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([pos, rotvec, gripper_rad]).astype(np.float32)


def frames_to_actions_8d(frames: list[dict]) -> np.ndarray:
    """action[t] = state[t+1] (last frame repeats). Matches Stage 0 layout."""
    states = np.stack([f["state_8d"] for f in frames]).astype(np.float32)
    actions = np.empty_like(states)
    actions[:-1] = states[1:]
    actions[-1] = states[-1]
    return actions


# ----- Sim helpers ------------------------------------------------------------

def make_pose(target_xyz: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from a rotation R and a position target."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = target_xyz
    return T


def set_cube_pose(sim, cube_qadr, x, y, z):
    """Place the cube at (x, y, z) with identity orientation, zero velocity."""
    sim.data.qpos[cube_qadr:cube_qadr + 3] = (x, y, z)
    sim.data.qpos[cube_qadr + 3:cube_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    qveladr = sim.model.jnt_dofadr[cube_jnt_id]
    sim.data.qvel[qveladr:qveladr + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def record_frame(sim, kin, arm_q, frames):
    """Snapshot current state + camera image into the per-episode buffer.

    State is recorded BEFORE applying any action at this frame. Returns the
    8D state vector for inspection.
    """
    grip_actual = np.array([
        sim.data.joint("proximal").qpos[0],
        sim.data.joint("distal").qpos[0],
    ])
    state_8d = fk_state_8d(kin, arm_q, grip_actual)
    img_rgb = sim.render_camera()                # (H, W, 3) uint8
    frames.append({"state_8d": state_8d, "image": img_rgb})
    return state_8d


# ----- Trajectory phases ------------------------------------------------------

def phase_to_xyz(sim, kin, arm_q, target_xyz, R_down, n_frames, gripper_ctrl, frames, viewer=None):
    """Linearly interpolate the gripper position to target_xyz over n_frames.

    Records one frame per outer iteration (so n_frames frames go to disk).
    Each outer iteration runs SIM_SUBSTEPS physics steps. Holds gripper ctrl
    constant. Returns the final arm joint positions.
    """
    T_now = kin.forward(arm_q, frame=GRIPPER_FRAME)
    start_xyz = T_now[:3, 3].copy()

    for i in range(n_frames):
        # Snapshot BEFORE applying the next command at this frame.
        record_frame(sim, kin, arm_q, frames)

        t = (i + 1) / n_frames
        interp_xyz = start_xyz + t * (target_xyz - start_xyz)
        T_target = make_pose(interp_xyz, R_down)
        arm_q = kin.inverse(T_target, current_joint_positions=arm_q,
                            n_iter=50, frame=GRIPPER_FRAME)

        sim.set_arm_commands(arm_q)
        sim.set_joint_commands(np.array([gripper_ctrl[0], gripper_ctrl[1]]),
                               joint_names=["proximal", "distal"])
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        arm_q = sim.get_arm_positions()
        if viewer is not None:
            viewer.sync()

    return arm_q


def phase_close(sim, arm_q, n_frames, kin, frames, viewer=None):
    """Ramp gripper ctrl from open to closed while holding the arm pose."""
    for i in range(n_frames):
        record_frame(sim, kin, arm_q, frames)

        t = (i + 1) / n_frames
        prox_cmd = (1.0 - t) * PROXIMAL_OPEN + t * PROXIMAL_CLOSED
        dist_cmd = (1.0 - t) * DISTAL_OPEN + t * DISTAL_CLOSED
        sim.set_arm_commands(arm_q)
        sim.set_joint_commands(np.array([prox_cmd, dist_cmd]),
                               joint_names=["proximal", "distal"])
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        if viewer is not None:
            viewer.sync()


def phase_hold(sim, arm_q, n_frames, gripper_ctrl, kin, frames, viewer=None):
    """Hold arm + gripper command for n_frames recorded frames."""
    for _ in range(n_frames):
        record_frame(sim, kin, arm_q, frames)
        sim.set_arm_commands(arm_q)
        sim.set_joint_commands(np.array([gripper_ctrl[0], gripper_ctrl[1]]),
                               joint_names=["proximal", "distal"])
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        if viewer is not None:
            viewer.sync()


# ----- Episode driver ---------------------------------------------------------

def run_episode(scene_xml, rng, use_viewer=False):
    """Run a single grasp-and-lift episode.

    Returns (frames, success, cube_start_xy, cube_final_z).
    Fresh Simulation per episode; ~10 s wall-clock per Stage 1.
    """
    sim = Simulation(scene_xml=scene_xml)
    kin = Kinematics()
    viewer = sim.launch_passive_viewer() if use_viewer else None

    # FK at SEED_JOINTS gives the canonical "gripper down" rotation that
    # IK can consistently solve for in this part of the workspace.
    T_seed = kin.forward(SEED_JOINTS, frame=GRIPPER_FRAME)
    R_down = T_seed[:3, :3].copy()

    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    cube_qadr = sim.model.jnt_qposadr[cube_jnt_id]

    # Random cube position
    cube_x = rng.uniform(*CUBE_X_RANGE)
    cube_y = rng.uniform(*CUBE_Y_RANGE)
    set_cube_pose(sim, cube_qadr, cube_x, cube_y, CUBE_START_Z)

    # IK to a pose 5 cm above the cube, gripper pointing down.
    pre_target = np.array([cube_x, cube_y, CUBE_START_Z + APPROACH_HEIGHT])
    T_pre = make_pose(pre_target, R_down)
    arm_q = kin.inverse(T_pre, current_joint_positions=SEED_JOINTS,
                        n_iter=200, frame=GRIPPER_FRAME)
    sim.reset_arm(arm_q)
    sim.set_joint_commands(np.array([PROXIMAL_OPEN, DISTAL_OPEN]),
                           joint_names=["proximal", "distal"])

    frames: list[dict] = []

    # 1. Approach: hold at the pre-grasp pose so the arm settles + we record
    #    diverse "above cube, gripper open" frames.
    arm_q = phase_to_xyz(sim, kin, arm_q, pre_target, R_down,
                         FRAMES_APPROACH,
                         (PROXIMAL_OPEN, DISTAL_OPEN), frames, viewer=viewer)

    # Reference cube position used for the lift target (after settling).
    cube_pos_start = sim.data.body("red_cube").xpos.copy()

    # 2. Descend to the cube center.
    descend_target = np.array([cube_x, cube_y, CUBE_START_Z])
    arm_q = phase_to_xyz(sim, kin, arm_q, descend_target, R_down,
                         FRAMES_DESCEND,
                         (PROXIMAL_OPEN, DISTAL_OPEN), frames, viewer=viewer)

    # 3. Close the gripper (ramp ctrl).
    phase_close(sim, arm_q, FRAMES_CLOSE, kin, frames, viewer=viewer)

    # 4. Hold closed (let contacts settle).
    phase_hold(sim, arm_q, FRAMES_HOLD, (PROXIMAL_CLOSED, DISTAL_CLOSED),
               kin, frames, viewer=viewer)

    # 5. Lift to 10 cm above the cube's original position.
    lift_target = np.array([cube_pos_start[0], cube_pos_start[1],
                            cube_pos_start[2] + LIFT_HEIGHT])
    arm_q = phase_to_xyz(sim, kin, arm_q, lift_target, R_down,
                         FRAMES_LIFT,
                         (PROXIMAL_CLOSED, DISTAL_CLOSED), frames, viewer=viewer)

    # 6. Final settle.
    phase_hold(sim, arm_q, FRAMES_SETTLE, (PROXIMAL_CLOSED, DISTAL_CLOSED),
               kin, frames, viewer=viewer)

    cube_final = sim.data.body("red_cube").xpos.copy()
    success = (cube_final[2] - cube_pos_start[2]) > LIFT_SUCCESS_THRESHOLD

    if viewer is not None:
        viewer.close()

    return frames, success, (cube_pos_start[0], cube_pos_start[1]), float(cube_final[2])


# ----- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect scripted grasp-and-lift demos as a LeRobotDataset "
                    "(arm-based variant; for the free-floating Grabette version "
                    "see manual_grasp_test.py / collect_grasp_dataset.py).")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--repo_id", type=str, required=True,
        help="Dataset repo id, e.g. 'sim_grasp_smoke'. Local-only; no Hub push.")
    parser.add_argument(
        "--output_root", type=str, default=None,
        help="Optional explicit local dataset root. If unset, LeRobot uses its "
             "standard cache (~/.cache/huggingface/lerobot/<repo_id>).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", action="store_true",
                        help="Open the MuJoCo passive viewer per episode (visual "
                             "verification mode; closes at end of each episode).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)

    output_root = Path(args.output_root) if args.output_root else None
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=build_features(),
        root=output_root,
        robot_type="openarm_gripette_sim",
        use_videos=True,
    )
    logger.info(f"Created LeRobotDataset at {dataset.root}")
    logger.info(f"Collecting {args.episodes} episodes -> repo_id={args.repo_id}")

    n_saved = 0
    n_success = 0
    print(f"{'ep':>4} {'cube_x':>8} {'cube_y':>8} {'final_z':>9} "
          f"{'frames':>7} {'dt_s':>6} {'result':>7}")
    for ep in range(args.episodes):
        t0 = time.perf_counter()
        frames, success, (cx, cy), final_z = run_episode(SCENE, rng, use_viewer=args.viewer)
        dt = time.perf_counter() - t0

        result = "OK" if success else "FAIL"
        print(f"{ep:4d} {cx:8.3f} {cy:8.3f} {final_z:9.4f} "
              f"{len(frames):7d} {dt:6.1f} {result:>7}")

        # Discard failures: never call add_frame for them.
        if not success:
            continue

        actions = frames_to_actions_8d(frames)
        for i, f in enumerate(frames):
            dataset.add_frame({
                "task": TASK,
                "observation.images.cam0": f["image"],         # (H, W, 3) uint8
                "action": actions[i].astype(np.float32),       # (8,) float32
            })
        dataset.save_episode()
        n_saved += 1
        n_success += 1

    dataset.finalize()

    rate = 100.0 * n_success / args.episodes if args.episodes else 0.0
    logger.info(
        f"Done. Saved {n_saved}/{args.episodes} episodes "
        f"({n_success} success, {rate:.1f}%). Dataset root: {dataset.root}")


if __name__ == "__main__":
    main()
