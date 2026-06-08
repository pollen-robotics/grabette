"""Collect a reach-and-touch dataset in LeRobot v3 format.

Generates diverse trajectories mixing three behaviors:
  1. Direct reach: cube visible at start, go straight to it
  2. Scan then reach: start looking away, sweep to find cube, then reach
  3. Offset reach: start displaced, re-center then reach

Features match the Grabette pipeline:
  - observation.state: [11] float32 — [x, y, z, r6d_0..5, proximal_deg, distal_deg]
  - observation.images.cam0: video — Gripette camera (fisheye, 972x1296)
  - action: [11] float32 — same 11D (absolute, next-step target)

Usage:
    # LeRobot format (requires: uv sync --extra dataset)
    uv run python examples/collect_reach_dataset.py \
        --repo-id user/simu_reach --episodes 500
    # Raw npy/jpg (no lerobot dependency)
    uv run python examples/collect_reach_dataset.py \
        --raw --root data/raw_reach --episodes 50
    # With viewer
    uv run python examples/collect_reach_dataset.py \
        --raw --episodes 5 --viewer
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from openarm_gripette_simu import Simulation, Kinematics
from openarm_gripette_simu.kinematics import GRIPPER_FRAME
from openarm_gripette_simu.rotation import rotation_matrix_to_6d, rotation_6d_to_matrix
from openarm_gripette_simu.camera import CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY, CAMERA_WIDTH, CAMERA_HEIGHT

logger = logging.getLogger(__name__)

SCENE = Path(__file__).parent.parent / "scenes" / "table_red_cube.xml"

# Start config
START_JOINTS = np.array([-0.35, 0.0, 0.0, 1.81, 0.0, 0.0, 0.0])

# Cube nominal position
CUBE_NOMINAL_X = 0.40
CUBE_NOMINAL_Y = -0.15

# Randomization
CUBE_X_NOISE = 0.06
CUBE_Y_NOISE = 0.2
CUBE_YAW_NOISE = np.pi
ARM_JOINT_NOISE = 0.08

# Extra yaw noise for scan trajectories (pushes cube out of view)
SCAN_YAW_RANGE = (0.5, 1.0)  # arm_yaw offset range for scan start

# Table bounds
TABLE_X_MIN = 0.165
TABLE_X_MAX = 0.735
TABLE_Y_MIN = -0.285
TABLE_Y_MAX = 0.285

# Timing — must match real data (50fps)
FPS = 50
STEPS_SCAN = 40       # steps for scan phase (sweep to find cube)
STEPS_REACH = 80      # steps for reach phase
STEPS_HOLD = 20       # steps holding at cube
SIM_SUBSTEPS = 10     # 50fps → 20ms per frame → 10 steps at dt=0.002
SETTLE_STEPS = 200

# How much to rotate toward the cube (0=none, 1=fully look at it)
LOOK_AT_STRENGTH = 0.3

# Trajectory type probabilities
P_DIRECT = 0.4   # cube visible, go straight
P_SCAN = 0.35    # start looking away, scan to find, then reach
P_OFFSET = 0.25  # start displaced, re-center then reach

# Feature names
STATE_NAMES = ["x", "y", "z", "r6d_0", "r6d_1", "r6d_2",
               "r6d_3", "r6d_4", "r6d_5", "proximal", "distal"]


def look_at_rotation(cam_pos, target_pos):
    """Compute rotation matrix that points camera Z (OpenCV forward) at target."""
    z = target_pos - cam_pos
    z = z / np.linalg.norm(z)
    world_up = np.array([0, 0, 1.0])
    x = np.cross(z, world_up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([0, 1.0, 0]))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def cube_visible_in_camera(cam_T, cube_pos, margin=50):
    """Check if the cube center projects inside the camera image."""
    R = cam_T[:3, :3]
    t = cam_T[:3, 3]
    p_cam = R.T @ (cube_pos - t)
    if p_cam[2] <= 0.01:
        return False
    u = CAMERA_FX * p_cam[0] / p_cam[2] + CAMERA_CX
    v = CAMERA_FY * p_cam[1] / p_cam[2] + CAMERA_CY
    return margin < u < CAMERA_WIDTH - margin and margin < v < CAMERA_HEIGHT - margin


def fk_state(kin, arm_q, gripper_rad):
    """Compute 11D state from camera frame."""
    T = kin.forward(arm_q, "camera")
    pos = T[:3, 3]
    r6d = rotation_matrix_to_6d(T[:3, :3])
    return np.concatenate([pos, r6d, np.degrees(gripper_rad)]).astype(np.float32)


def has_table_collision(sim):
    """Check if any robot geom touches the table."""
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        name1 = sim.model.geom(c.geom1).name
        name2 = sim.model.geom(c.geom2).name
        is_table = "table" in name1 or "leg" in name1 or "table" in name2 or "leg" in name2
        is_cube = "red_cube" in name1 or "red_cube" in name2
        if is_table and not is_cube:
            return True
    return False


def run_phase(sim, kin, arm_q, start_pos, start_r6d, target_pos, target_r6d,
              n_steps, frames, cube_pos, viewer):
    """Run one trajectory phase, recording frames. Returns (arm_q, collision)."""
    collision = False
    for i in range(n_steps):
        t = (i + 1) / n_steps

        # Record observation
        grip_actual = np.array([
            sim.data.joint("proximal").qpos[0],
            sim.data.joint("distal").qpos[0],
        ])
        state = fk_state(kin, arm_q, grip_actual)
        img_rgb = sim.render_camera()
        frames.append({"state": state, "image": Image.fromarray(img_rgb)})

        # Interpolate
        interp_pos = start_pos + t * (target_pos - start_pos)
        interp_r6d = start_r6d + t * (target_r6d - start_r6d)

        T_target = np.eye(4)
        T_target[:3, :3] = rotation_6d_to_matrix(interp_r6d)
        T_target[:3, 3] = interp_pos
        arm_q = kin.inverse(T_target, current_joint_positions=arm_q, n_iter=100, frame="camera")

        sim.set_arm_commands(arm_q)
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        if viewer:
            viewer.sync()
        if has_table_collision(sim):
            collision = True
        arm_q = sim.get_arm_positions()

    return arm_q, collision



def _cube_contacts_robot(sim):
    """Check if the cube is in contact with any robot geom (not table/floor)."""
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        n1 = sim.model.geom(c.geom1).name
        n2 = sim.model.geom(c.geom2).name
        is_cube = "red_cube" in n1 or "red_cube" in n2
        is_env = ("table" in n1 or "leg" in n1 or "floor" in n1 or
                  "table" in n2 or "leg" in n2 or "floor" in n2)
        if is_cube and not is_env:
            return True
    return False


def setup_episode(sim, kin, rng, start_joints):
    """Reset sim with randomized cube and arm. Uses MuJoCo collision
    detection to guarantee no cube-robot contact at spawn."""
    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    cube_qadr = sim.model.jnt_qposadr[cube_jnt_id]

    # Reset arm first
    sim.reset_arm(start_joints)
    sim.data.qvel[:] = 0

    # Sample cube positions until MuJoCo confirms no robot contact
    while True:
        cube_x = np.clip(
            CUBE_NOMINAL_X + rng.uniform(-CUBE_X_NOISE, CUBE_X_NOISE),
            TABLE_X_MIN + 0.02, TABLE_X_MAX - 0.02)
        cube_y = np.clip(
            CUBE_NOMINAL_Y + rng.uniform(-CUBE_Y_NOISE, CUBE_Y_NOISE),
            TABLE_Y_MIN + 0.02, TABLE_Y_MAX - 0.02)
        yaw = rng.uniform(-CUBE_YAW_NOISE, CUBE_YAW_NOISE)

        sim.data.qpos[cube_qadr:cube_qadr + 3] = [cube_x, cube_y, 0.415]
        sim.data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(sim.model, sim.data)

        if not _cube_contacts_robot(sim):
            break

    # Settle physics
    for _ in range(SETTLE_STEPS):
        sim.step()

    return sim.get_arm_positions(), np.array([cube_x, cube_y, 0.415])


def compute_reach_targets(kin, arm_q, cube_pos):
    """Compute camera target position and orientation for reaching the cube."""
    T_cam = kin.forward(arm_q, "camera")
    T_grip = kin.forward(arm_q, GRIPPER_FRAME)
    cam_pos = T_cam[:3, 3].copy()
    cam_r6d = rotation_matrix_to_6d(T_cam[:3, :3])

    # Position: move gripper to cube XY, keep Z
    grip_target = np.array([cube_pos[0], cube_pos[1], T_grip[2, 3]])
    cam_target_pos = cam_pos + (grip_target - T_grip[:3, 3])

    # Orientation: partially look at cube
    R_look = look_at_rotation(cam_pos, cube_pos)
    look_r6d = rotation_matrix_to_6d(R_look)
    cam_target_r6d = cam_r6d + LOOK_AT_STRENGTH * (look_r6d - cam_r6d)

    return cam_pos, cam_r6d, cam_target_pos, cam_target_r6d


def run_episode(sim, kin, rng, traj_type, viewer=None):
    """Run one episode. Returns (frames, success, discard_reason)."""

    if traj_type == "scan":
        # Start with extra yaw so cube is out of view
        yaw_offset = rng.choice([-1, 1]) * rng.uniform(*SCAN_YAW_RANGE)
        start_joints = START_JOINTS.copy()
        start_joints[2] += yaw_offset  # arm_yaw
        start_joints += rng.uniform(-ARM_JOINT_NOISE * 0.5, ARM_JOINT_NOISE * 0.5, size=7)
    elif traj_type == "offset":
        # Start with extra roll to displace sideways
        start_joints = START_JOINTS.copy()
        start_joints[1] += rng.uniform(-0.15, 0.15)  # arm_roll
        start_joints += rng.uniform(-ARM_JOINT_NOISE, ARM_JOINT_NOISE, size=7)
    else:  # direct
        start_joints = START_JOINTS + rng.uniform(-ARM_JOINT_NOISE, ARM_JOINT_NOISE, size=7)

    arm_q, cube_pos = setup_episode(sim, kin, rng, start_joints)
    if viewer:
        viewer.sync()

    cube_start_xy = cube_pos[:2].copy()
    frames = []
    collision = False

    if traj_type == "scan":
        # Phase 1: sweep back toward cube (reduce yaw to 0)
        T_cam = kin.forward(arm_q, "camera")
        scan_start_pos = T_cam[:3, 3].copy()
        scan_start_r6d = rotation_matrix_to_6d(T_cam[:3, :3])

        # Target: return to nominal yaw (arm_yaw ≈ 0) while orienting toward cube
        neutral_joints = START_JOINTS + rng.uniform(-ARM_JOINT_NOISE * 0.3, ARM_JOINT_NOISE * 0.3, size=7)
        # Compute where the camera would be at neutral
        T_cam_neutral = kin.forward(neutral_joints, "camera")
        scan_target_pos = T_cam_neutral[:3, 3].copy()
        # Orient toward cube during scan
        R_look = look_at_rotation(scan_target_pos, cube_pos)
        scan_target_r6d = rotation_matrix_to_6d(T_cam_neutral[:3, :3]) + \
            LOOK_AT_STRENGTH * (rotation_matrix_to_6d(R_look) - rotation_matrix_to_6d(T_cam_neutral[:3, :3]))

        arm_q, col = run_phase(sim, kin, arm_q, scan_start_pos, scan_start_r6d,
                               scan_target_pos, scan_target_r6d,
                               STEPS_SCAN, frames, cube_pos, viewer)
        collision |= col

    # Phase: reach toward cube
    cam_pos, cam_r6d, target_pos, target_r6d = compute_reach_targets(kin, arm_q, cube_pos)
    arm_q, col = run_phase(sim, kin, arm_q, cam_pos, cam_r6d,
                           target_pos, target_r6d,
                           STEPS_REACH, frames, cube_pos, viewer)
    collision |= col

    # Phase: hold at cube
    T_cam_hold = kin.forward(arm_q, "camera")
    hold_pos = T_cam_hold[:3, 3].copy()
    hold_r6d = rotation_matrix_to_6d(T_cam_hold[:3, :3])
    arm_q, col = run_phase(sim, kin, arm_q, hold_pos, hold_r6d,
                           hold_pos, hold_r6d,
                           STEPS_HOLD, frames, cube_pos, viewer)
    collision |= col

    if collision:
        return None, False, "table_collision"

    # Build actions: action[t] = state[t+1]
    for i in range(len(frames) - 1):
        frames[i]["action"] = frames[i + 1]["state"].copy()
    frames[-1]["action"] = frames[-1]["state"].copy()

    # Success: cube moved
    cube_end_xy = sim.data.body("red_cube").xpos[:2].copy()
    cube_moved = np.linalg.norm(cube_end_xy - cube_start_xy) > 0.003

    return frames, cube_moved, None


def main():
    parser = argparse.ArgumentParser(description="Collect reach dataset in LeRobot format")
    parser.add_argument("--repo-id", type=str, default="local/simu_reach")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--raw", action="store_true",
                        help="Save raw npy/jpg files instead of LeRobot format (no lerobot dependency)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rng = np.random.default_rng(args.seed)
    sim = Simulation(scene_xml=SCENE)
    kin = Kinematics()

    viewer = None
    if args.viewer:
        viewer = sim.launch_passive_viewer()

    # Set up output
    ds = None
    output_dir = None

    if args.raw:
        import json
        output_dir = Path(args.root) if args.root else Path("data/raw_reach")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Collecting {args.episodes} episodes (raw) -> {output_dir}")
    else:
        try:
            from lerobot.datasets import LeRobotDataset
        except ImportError:
            logger.error("lerobot not installed. Run: uv sync --extra dataset (or use --raw)")
            return

        features = {
            "observation.state": {
                "dtype": "float32", "shape": (11,), "names": STATE_NAMES,
            },
            "observation.images.cam0": {
                "dtype": "video", "shape": (3, 972, 1296), "names": ["channels", "height", "width"],
            },
            "action": {
                "dtype": "float32", "shape": (11,), "names": STATE_NAMES,
            },
        }
        root = Path(args.root) if args.root else None
        ds = LeRobotDataset.create(
            repo_id=args.repo_id, fps=FPS, features=features, root=root,
            robot_type="openarm_gripette", use_videos=True,
        )
        logger.info(f"Collecting {args.episodes} episodes (LeRobot) -> {ds.root}")

    n_success = 0
    n_attempted = 0
    n_collision = 0
    type_counts = {"direct": 0, "scan": 0, "offset": 0}
    t_start = time.perf_counter()

    while n_success < args.episodes:
        # Pick trajectory type
        r = rng.random()
        if r < P_DIRECT:
            traj_type = "direct"
        elif r < P_DIRECT + P_SCAN:
            traj_type = "scan"
        else:
            traj_type = "offset"

        frames, success, discard = run_episode(sim, kin, rng, traj_type, viewer)
        n_attempted += 1

        if discard:
            n_collision += 1
            continue

        if args.raw:
            # Save as npy + jpg
            ep_dir = output_dir / f"episode_{n_success:04d}"
            ep_dir.mkdir(exist_ok=True)

            states = np.array([f["state"] for f in frames])
            actions = np.array([f["action"] for f in frames])
            np.save(ep_dir / "states.npy", states)
            np.save(ep_dir / "actions.npy", actions)

            img_dir = ep_dir / "images"
            img_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(frames):
                frame["image"].save(img_dir / f"{i:04d}.jpg", quality=70)

            meta = {
                "n_steps": len(frames),
                "state_dim": 11,
                "action_dim": 11,
                "trajectory_type": traj_type,
                "success": bool(success),
                "state_names": STATE_NAMES,
            }
            (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        else:
            for frame in frames:
                ds.add_frame({
                    "observation.state": frame["state"],
                    "observation.images.cam0": frame["image"],
                    "action": frame["action"],
                    "task": "reach_red_cube",
                })
            ds.save_episode()

        n_success += 1
        type_counts[traj_type] += 1

        if n_success % 10 == 0:
            elapsed = time.perf_counter() - t_start
            eps = n_success / elapsed
            eta = (args.episodes - n_success) / eps if eps > 0 else 0
            logger.info(
                f"  {n_success}/{args.episodes} ({n_attempted} tried, {n_collision} collisions, "
                f"{eps:.1f} ep/s, ETA {eta:.0f}s) "
                f"types: {type_counts}"
            )

    if ds is not None:
        ds.finalize()

    elapsed = time.perf_counter() - t_start
    logger.info(f"Done. {n_success} episodes in {elapsed:.0f}s ({n_attempted} tried)")
    logger.info(f"Types: {type_counts}")

    if args.push and ds is not None:
        logger.info(f"Pushing to HuggingFace Hub as {args.repo_id}...")
        ds.push_to_hub()
        logger.info("Push complete.")

    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()
