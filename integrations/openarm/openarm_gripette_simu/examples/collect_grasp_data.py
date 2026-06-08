"""Collect synthetic reach-and-touch trajectories in LeRobot dataset format.

The arm reaches forward to touch a red cube on the table. Cube position
and arm start are randomized per episode. Uses the camera frame for state
reporting and the gripper frame for targeting.

Scene: table_red_cube.xml

Output: a LeRobotDataset on disk (parquet + mp4 video) matching the same
input contract as real Grabette data, so that
`examples/openarm_gripette/convert_dataset.py` can post-process it without
modification.

Schema written here (pre-conversion):
  - observation.images.cam0: video frame (H, W, 3) uint8, 972x1296
  - action: float32 (8,) [x, y, z, ax, ay, az, proximal_rad, distal_rad]
            absolute pose in camera frame, axis-angle rotation, gripper in radians.
  - task: constant string "reach_cube"

Usage:
    uv run python examples/collect_grasp_data.py --episodes 100 --repo_id steve/sim_reach
    uv run python examples/collect_grasp_data.py --episodes 5 --viewer
"""

import argparse
import logging
import time
from pathlib import Path

import mujoco
import numpy as np

from openarm_gripette_simu import Kinematics, Simulation
from openarm_gripette_simu.kinematics import GRIPPER_FRAME
from openarm_gripette_simu.rotation import rotation_matrix_to_6d, rotation_6d_to_matrix

# LeRobot dataset API + rotation helper to convert rotation matrix -> axis-angle (rotvec).
# We re-use LeRobot's Rotation class (numpy-based, scipy-compatible API) instead of
# pulling in scipy directly, since lerobot is already a dependency for this script.
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.rotation import Rotation as LeRobotRotation

logger = logging.getLogger(__name__)

SCENE = Path(__file__).parent.parent / "scenes" / "table_red_cube.xml"

# Start config
START_JOINTS = np.array([-0.35, 0.0, 0.0, 1.81, 0.0, 0.0, 0.0])

# Nominal cube position (matches table_red_cube.xml)
CUBE_NOMINAL_X = 0.40
CUBE_NOMINAL_Y = -0.15

# Randomization
CUBE_X_NOISE = 0.06     # ±6cm in X (limited by arm reach)
CUBE_Y_NOISE = 0.2      # ±20cm in Y (limited by table edge)
CUBE_YAW_NOISE = np.pi  # full rotation around Z
ARM_JOINT_NOISE = 0.08  # ±0.08 rad arm start

# Table bounds (from scene XML: table pos=[0.45,0], size=[0.30,0.30])
TABLE_X_MIN = 0.165
TABLE_X_MAX = 0.735
TABLE_Y_MIN = -0.285
TABLE_Y_MAX = 0.285

# Timing — consistent with 50fps (20ms per frame)
STEPS_REACH = 80       # steps to reach the cube
STEPS_HOLD = 20        # steps holding at the cube
SIM_SUBSTEPS = 10      # 50fps -> 20ms per frame -> 10 steps at dt=0.002
SETTLE_STEPS = 200
FPS = 50

# Image dimensions (must match Simulation.render_camera output).
IMG_HEIGHT = 972
IMG_WIDTH = 1296

# Constant task label written for every frame.
TASK = "reach_cube"


def fk_state_8d(kin: Kinematics, arm_q: np.ndarray, gripper_rad: np.ndarray) -> np.ndarray:
    """Compute the 8D absolute pose [x, y, z, ax, ay, az, proximal_rad, distal_rad]
    from the camera frame.

    The rotation is encoded as an axis-angle (rotation vector). Gripper joints are
    kept in radians, matching the contract that downstream `convert_dataset.py` expects.
    """
    T = kin.forward(arm_q, "camera")
    pos = T[:3, 3]
    # axis-angle / rotation vector — 3 floats encoding axis*angle.
    rotvec = LeRobotRotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([pos, rotvec, gripper_rad]).astype(np.float32)


def has_table_collision(sim: Simulation) -> bool:
    """Check if any robot geom is in contact with the table."""
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        name1 = sim.model.geom(c.geom1).name
        name2 = sim.model.geom(c.geom2).name
        is_table = "table" in name1 or "leg" in name1 or "table" in name2 or "leg" in name2
        is_cube = "red_cube" in name1 or "red_cube" in name2
        if is_table and not is_cube:
            return True
    return False


def run_phase(sim, kin, arm_q, target_pos, target_r6d, n_steps, frames, viewer):
    """Interpolate the gripper frame to a target, recording at each step.

    `frames` is a mutable list of dicts; each dict captures the per-frame state and
    image, which becomes the absolute pose at time t (the action is reconstructed
    later as state[t+1]).

    Returns (arm_q, table_collision).
    """
    T_grip_now = kin.forward(arm_q, frame=GRIPPER_FRAME)
    start_pos = T_grip_now[:3, 3].copy()
    start_r6d = rotation_matrix_to_6d(T_grip_now[:3, :3])
    collision = False

    for i in range(n_steps):
        t = (i + 1) / n_steps

        # Record observation BEFORE applying action (matches original layout).
        grip_actual = np.array([
            sim.data.joint("proximal").qpos[0],
            sim.data.joint("distal").qpos[0],
        ])  # already in radians
        state_8d = fk_state_8d(kin, arm_q, grip_actual)

        # Render the camera image as RGB uint8 (H, W, 3).
        img_rgb = sim.render_camera()

        frames.append({"state_8d": state_8d, "image": img_rgb})

        # Interpolate gripper target in 6D space (smooth between two rotations).
        interp_pos = start_pos + t * (target_pos - start_pos)
        interp_r6d = start_r6d + t * (target_r6d - start_r6d)

        T_ik = np.eye(4)
        T_ik[:3, :3] = rotation_6d_to_matrix(interp_r6d)
        T_ik[:3, 3] = interp_pos
        arm_q = kin.inverse(T_ik, current_joint_positions=arm_q, n_iter=100,
                            frame=GRIPPER_FRAME)

        sim.set_arm_commands(arm_q)
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        if viewer:
            viewer.sync()

        if has_table_collision(sim):
            collision = True

        arm_q = sim.get_arm_positions()

    return arm_q, collision


def run_episode(sim, kin, rng, viewer=None):
    """Run one reach-and-touch episode. Returns per-frame data and metadata.

    Returns:
        frames: list of dicts with keys {"state_8d", "image"}.
        success: bool.
        table_collision: bool.
    """
    # --- Randomize cube position ---
    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    cube_qadr = sim.model.jnt_qposadr[cube_jnt_id]

    cube_x = CUBE_NOMINAL_X + rng.uniform(-CUBE_X_NOISE, CUBE_X_NOISE)
    cube_y = CUBE_NOMINAL_Y + rng.uniform(-CUBE_Y_NOISE, CUBE_Y_NOISE)
    cube_x = np.clip(cube_x, TABLE_X_MIN + 0.02, TABLE_X_MAX - 0.02)
    cube_y = np.clip(cube_y, TABLE_Y_MIN + 0.02, TABLE_Y_MAX - 0.02)
    cube_z = 0.415  # on the table

    yaw = rng.uniform(-CUBE_YAW_NOISE, CUBE_YAW_NOISE)
    qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)

    sim.data.qpos[cube_qadr:cube_qadr + 3] = [cube_x, cube_y, cube_z]
    sim.data.qpos[cube_qadr + 3:cube_qadr + 7] = [qw, 0, 0, qz]

    # --- Randomize arm start ---
    init_joints = START_JOINTS + rng.uniform(-ARM_JOINT_NOISE, ARM_JOINT_NOISE, size=7)
    sim.reset_arm(init_joints)
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(SETTLE_STEPS):
        sim.step()
    if viewer:
        viewer.sync()

    arm_q = sim.get_arm_positions()
    T_grip = kin.forward(arm_q, frame=GRIPPER_FRAME)
    grip_r6d = rotation_matrix_to_6d(T_grip[:3, :3])

    # Target: gripper at the cube position (same height as gripper).
    target_pos = np.array([cube_x, cube_y, T_grip[2, 3]])

    cube_pos_start = sim.data.body("red_cube").xpos.copy()

    frames: list[dict] = []
    table_collision = False

    arm_q, col = run_phase(sim, kin, arm_q, target_pos, grip_r6d, STEPS_REACH,
                           frames, viewer)
    table_collision |= col

    arm_q, col = run_phase(sim, kin, arm_q, target_pos, grip_r6d, STEPS_HOLD,
                           frames, viewer)
    table_collision |= col

    cube_pos_end = sim.data.body("red_cube").xpos.copy()
    cube_moved = np.linalg.norm(cube_pos_end[:2] - cube_pos_start[:2]) > 0.003
    success = cube_moved and not table_collision

    return frames, success, table_collision


def build_features() -> dict:
    """LeRobot feature spec — matches the input contract that convert_dataset.py expects.

    - action: 8D absolute pose [x,y,z, ax,ay,az, proximal_rad, distal_rad].
              convert_dataset.py recognizes the 8D form (axis-angle) and expands to 11D
              before computing deltas.
    - observation.images.cam0: stored as video; convert_dataset.py just passes it through.

    Note: no observation.state here. convert_dataset.py creates it (gripper-only 2D or
    pose-relative 11D).
    """
    return {
        "observation.images.cam0": {
            "dtype": "video",
            # CHW order is the convention in info.json features (height/width are
            # passed in here too — frames may be added in either CHW or HWC).
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


def frames_to_actions_8d(frames: list[dict]) -> np.ndarray:
    """Build per-frame absolute 8D actions: action[t] = state[t+1] (last frame repeats).

    This mirrors the original data layout where 'action' is the next-frame absolute
    pose. convert_dataset.py will subtract neighbouring frames to produce position +
    rotation deltas.
    """
    states = np.stack([f["state_8d"] for f in frames]).astype(np.float32)
    actions = np.empty_like(states)
    actions[:-1] = states[1:]
    actions[-1] = states[-1]
    return actions


def main():
    parser = argparse.ArgumentParser(description="Collect synthetic reach data into a LeRobotDataset")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--repo_id",
        type=str,
        default="local/sim_reach",
        help="Dataset repo id, e.g. 'user/sim_reach'. Local-only; no Hub push.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional explicit local dataset root. If unset, LeRobot uses its standard "
             "cache (~/.cache/huggingface/lerobot/<repo_id>).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)

    sim = Simulation(scene_xml=SCENE)
    kin = Kinematics()

    viewer = None
    if args.viewer:
        viewer = sim.launch_passive_viewer()

    # Create the dataset writer up-front. We add 1 episode worth of frames at a time,
    # then save_episode() flushes parquet + mp4 to disk.
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

    n_success = 0
    n_collision = 0
    n_saved = 0

    for ep in range(args.episodes):
        t0 = time.perf_counter()
        frames, success, table_collision = run_episode(sim, kin, rng, viewer)
        dt = time.perf_counter() - t0

        if table_collision:
            n_collision += 1
            logger.info(f"  episode {ep:4d}: {dt:.1f}s, TABLE COLLISION — discarded")
            # Drop the episode buffer entirely — never call add_frame for these.
            continue

        n_success += int(success)

        # Build absolute 8D actions; gripper is already in radians from MuJoCo qpos.
        actions = frames_to_actions_8d(frames)

        # Push frames to the dataset.
        for i, f in enumerate(frames):
            dataset.add_frame({
                "task": TASK,
                "observation.images.cam0": f["image"],          # (H, W, 3) uint8
                "action": actions[i].astype(np.float32),        # (8,) float32
            })

        # Flush this episode (writes parquet + encodes mp4).
        dataset.save_episode()

        status = "TOUCH" if success else "MISS"
        logger.info(f"  episode {ep:4d}: {len(frames)} steps, {dt:.1f}s, {status}")
        n_saved += 1

    dataset.finalize()

    logger.info(
        f"Done. {n_saved} episodes saved ({n_success} touch, "
        f"{n_saved - n_success} miss, {n_collision} discarded for collision). "
        f"Dataset root: {dataset.root}"
    )
    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()
