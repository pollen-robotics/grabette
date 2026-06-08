"""Collect scripted grasp-and-lift demonstrations as a LeRobot-format dataset
(virtual Grabette variant — free-floating gripper anchored to a mocap body via
an <equality><weld>; no arm in the loop).

Stage 4: random DIAGONAL grasp orientation per episode (random tilt off
vertical and azimuth around world +Z), with a sentry waypoint that sits back
along the gripper's local approach axis instead of straight above the grasp.

Each episode runs the same scripted trajectory as `manual_grasp_test.py`
(approach -> descend -> pre-grip-settle -> close -> hold -> lift -> retract ->
final-settle). Successful episodes (cube lifted >= 5 cm above its starting z)
are saved as a single LeRobotDataset on disk; failures are discarded.

State recording is mocap-driven: the (x, y, z) is read from
`data.mocap_pos[mocap_id]` and the rotation is read from
`data.mocap_quat[mocap_id]` (MuJoCo (w,x,y,z) ordering, converted to scipy/lerobot
(x,y,z,w) before computing the axis-angle). The mocap pose is the *commanded*
gripper pose and is the source of truth for "where the gripper is" in this
no-arm setup. Gripper joint angles come straight from the proximal/distal
qpos (radians).

Schema written here matches `collect_grasp_data.py` (Stage 0):
    * observation.images.cam0: video, (3, 972, 1296) uint8
    * action: float32 (8,) [x, y, z, ax, ay, az, proximal_rad, distal_rad]
    * task: constant string
    * 50 fps. Per-frame action is the absolute pose at the next frame, so
      `convert_dataset.py` can compute deltas from neighbour frames.

Usage:
    uv run python examples/collect_grasp_dataset.py --episodes 10 --repo_id sim_grabette_grasp
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm

# Make sibling helper module importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grabette_trajectory import (  # noqa: E402
    CUBE_START_Z,
    DISTAL_CLOSED,
    DISTAL_OPEN,
    EVAL_CUBE_X_RANGE,
    EVAL_CUBE_Y_RANGE,
    EpisodeWaypoints,
    LIFT_SUCCESS_THRESHOLD,
    PROXIMAL_CLOSED,
    PROXIMAL_OPEN,
    RETRACT_EXTRA,
    TRAIN_CUBE_X_RANGE,
    TRAIN_CUBE_Y_RANGE,
    body_pose_to_camera_pose,
    episode_target_poses,
    sample_episode_waypoints,
    slerp_quat,
    smoothstep,
)
from openarm_gripette_simu import DRConfig, IKFeasibilityChecker, Simulation, randomize_scene
from openarm_gripette_simu.kinematics import CAMERA_FRAME, Kinematics

# LeRobot dataset writer + axis-angle helper (same imports as Stage 0).
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.rotation import Rotation as LeRobotRotation

logger = logging.getLogger(__name__)

SCENE = Path(__file__).parent.parent / "scenes" / "grabette_grasp.xml"

# --- Phase frame counts (recorded frames at 50 fps -> 20 ms / frame).
# These mirror the manual test's sim-step counts, divided by SIM_SUBSTEPS=20:
#   manual STEPS_INITIAL_SETTLE=100  -> 5 frames
#   manual STEPS_APPROACH=1600       -> 80 frames
#   manual STEPS_DESCEND=500         -> 25 frames
#   manual STEPS_PRE_GRIP_SETTLE=200 -> 10 frames
#   manual STEPS_CLOSE=200           -> 10 frames
#   manual STEPS_HOLD=500            -> 25 frames
#   manual STEPS_LIFT=600            -> 30 frames
#   manual STEPS_RETRACT=1000        -> 50 frames
#   manual STEPS_FINAL_SETTLE=300    -> 15 frames
# Total ~ 250 frames per episode (~5.0 s at 50 fps).
FPS = 50
SIM_SUBSTEPS = 20  # 20 ms per frame at sim dt=0.001
FRAMES_INITIAL_SETTLE = 5
FRAMES_OPEN_RAMP = 15            # release-episode: ramp closed -> open
FRAMES_HOVER = 50                # hover-episode: extra time at grasp pose with gripper still open
FRAMES_APPROACH = 80
# Stage 5e: slowed contact-critical phases. Arm replay shows the kinematic
# motion is correct but contact transients on close/lift kick the cube out
# of the V-pocket. Lengthening close/hold/lift gives the contact dynamics
# time to settle: cube is gripped firmly before lift starts, lift
# acceleration stays under the cube's static-friction budget.
FRAMES_DESCEND = 50          # was 25
FRAMES_PRE_GRIP_SETTLE = 25
FRAMES_CLOSE = 30            # was 10 — gentler close ramp
FRAMES_HOLD = 50             # was 25
FRAMES_LIFT = 60             # was 30 — gentler lift
FRAMES_RETRACT = 50
FRAMES_FINAL_SETTLE = 15

# --- Image dimensions (must match Simulation.render_camera) ---
IMG_HEIGHT = 972
IMG_WIDTH = 1296

# --- Constant task label ---
TASK = "grasp_and_lift_cube"


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


def mocap_state_8d(sim: Simulation, mocap_id: int) -> np.ndarray:
    """Build the 8D absolute pose [x, y, z, ax, ay, az, proximal, distal].

    Pipeline-wide convention: pose is the CAMERA SITE pose in world
    coordinates (Z-up, gravity-aligned). The real Grabette records the
    iPhone camera pose via SLAM; the deployed system applies the resulting
    delta actions in that same camera-site convention. We must match that
    convention here, otherwise sim demos and real demos live in
    different reference frames.

    Implementation: read the welded body pose from the mocap target, then
    apply the static body→camera transform (BODY_TO_CAMERA_POS / QUAT —
    the camera site offset inside gripper_base, taken from robot.xml).

    Quaternion ordering: MuJoCo stores (w, x, y, z); scipy /
    LeRobotRotation expects (x, y, z, w). The reordering below is
    essential — getting it wrong yields a silently-wrong axis-angle and
    convert_dataset.py would compute nonsense rotation deltas.
    """
    body_pos = sim.data.mocap_pos[mocap_id].copy()
    body_quat = sim.data.mocap_quat[mocap_id].copy()  # MuJoCo (w, x, y, z)
    cam_pos, cam_quat = body_pose_to_camera_pose(body_pos, body_quat)
    qw, qx, qy, qz = cam_quat
    rotvec = LeRobotRotation.from_quat([qx, qy, qz, qw]).as_rotvec()
    grip = np.array([
        sim.data.joint("proximal").qpos[0],
        sim.data.joint("distal").qpos[0],
    ])
    return np.concatenate([cam_pos, rotvec, grip]).astype(np.float32)


def frames_to_actions_8d(frames: list[dict]) -> np.ndarray:
    """action[t] = state[t+1] (last frame repeats). Matches Stage 0 layout."""
    states = np.stack([f["state_8d"] for f in frames]).astype(np.float32)
    actions = np.empty_like(states)
    actions[:-1] = states[1:]
    actions[-1] = states[-1]
    return actions


# ----- Sim helpers ------------------------------------------------------------

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

    Used at episode reset only — during runtime we move only the mocap and let
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


def record_frame(sim: Simulation, mocap_id: int, frames: list[dict]):
    """Snapshot current state + camera image into the per-episode buffer."""
    state_8d = mocap_state_8d(sim, mocap_id)
    img_rgb = sim.render_camera()
    frames.append({"state_8d": state_8d, "image": img_rgb})


# ----- Trajectory phases ------------------------------------------------------

def phase_smooth(sim: Simulation, mocap_id: int,
                 start_pos: np.ndarray, start_quat: np.ndarray,
                 end_pos: np.ndarray, end_quat: np.ndarray,
                 n_frames: int, gripper_ctrl: tuple,
                 prox_id: int, dist_id: int,
                 frames: list[dict], viewer=None):
    """Smoothstep+slerp drive of the mocap from (start_pos, start_quat) to
    (end_pos, end_quat) over n_frames recorded frames. Each recorded frame
    runs SIM_SUBSTEPS sim sub-steps. Returns end_pos.
    """
    for i in range(n_frames):
        # Snapshot BEFORE applying the next command at this frame, so the
        # recorded state is the pose-just-issued at the previous step.
        record_frame(sim, mocap_id, frames)

        t = (i + 1) / n_frames
        s = smoothstep(t)
        sim.data.mocap_pos[mocap_id] = start_pos + s * (end_pos - start_pos)
        sim.data.mocap_quat[mocap_id] = slerp_quat(start_quat, end_quat, s)
        sim.data.ctrl[prox_id] = gripper_ctrl[0]
        sim.data.ctrl[dist_id] = gripper_ctrl[1]
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        if viewer is not None:
            viewer.sync()
    return end_pos.copy()


def phase_multiwaypoint_smooth(sim: Simulation, mocap_id: int,
                                waypoints_pos: list[np.ndarray],
                                waypoints_quat: list[np.ndarray],
                                segment_frames: list[int],
                                gripper_ctrl: tuple,
                                prox_id: int, dist_id: int,
                                frames: list[dict], viewer=None):
    """Smoothstep+slerp through a list of waypoints, recording per-frame."""
    assert len(waypoints_pos) == len(waypoints_quat)
    assert len(segment_frames) == len(waypoints_pos) - 1
    for k, n in enumerate(segment_frames):
        phase_smooth(
            sim, mocap_id,
            waypoints_pos[k], waypoints_quat[k],
            waypoints_pos[k + 1], waypoints_quat[k + 1],
            n, gripper_ctrl, prox_id, dist_id, frames, viewer,
        )


def phase_hold(sim: Simulation, mocap_id: int,
               pos: np.ndarray, quat: np.ndarray,
               n_frames: int, gripper_ctrl: tuple,
               prox_id: int, dist_id: int,
               frames: list[dict], viewer=None):
    """Hold the mocap pose and gripper command for n_frames recorded frames."""
    for _ in range(n_frames):
        record_frame(sim, mocap_id, frames)
        sim.data.mocap_pos[mocap_id] = pos
        sim.data.mocap_quat[mocap_id] = quat
        sim.data.ctrl[prox_id] = gripper_ctrl[0]
        sim.data.ctrl[dist_id] = gripper_ctrl[1]
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        if viewer is not None:
            viewer.sync()


def phase_open(sim: Simulation, mocap_id: int,
               pos: np.ndarray, quat: np.ndarray,
               n_frames: int, prox_id: int, dist_id: int,
               frames: list[dict], viewer=None):
    """Ramp gripper ctrl from CLOSED to OPEN while holding the mocap fixed.

    Mirror of phase_close — used in 'release' episodes that start with the
    gripper closed and open it before approach. Gives the trained policy
    examples of closed→open transitions, which it otherwise never sees."""
    for i in range(n_frames):
        t = (i + 1) / n_frames
        record_frame(sim, mocap_id, frames)
        sim.data.mocap_pos[mocap_id] = pos
        sim.data.mocap_quat[mocap_id] = quat
        sim.data.ctrl[prox_id] = (1.0 - t) * PROXIMAL_CLOSED + t * PROXIMAL_OPEN
        sim.data.ctrl[dist_id] = (1.0 - t) * DISTAL_CLOSED + t * DISTAL_OPEN
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        if viewer is not None:
            viewer.sync()


def phase_close(sim: Simulation, mocap_id: int,
                pos: np.ndarray, quat: np.ndarray,
                n_frames: int, prox_id: int, dist_id: int,
                frames: list[dict], viewer=None):
    """Ramp gripper ctrl from open to closed while holding the mocap fixed."""
    for i in range(n_frames):
        record_frame(sim, mocap_id, frames)

        t = (i + 1) / n_frames
        sim.data.mocap_pos[mocap_id] = pos
        sim.data.mocap_quat[mocap_id] = quat
        sim.data.ctrl[prox_id] = (1.0 - t) * PROXIMAL_OPEN + t * PROXIMAL_CLOSED
        sim.data.ctrl[dist_id] = (1.0 - t) * DISTAL_OPEN + t * DISTAL_CLOSED
        for _ in range(SIM_SUBSTEPS):
            sim.step()
        if viewer is not None:
            viewer.sync()


# ----- Episode driver ---------------------------------------------------------

def plan_episode(
    rng: np.random.Generator,
    *,
    cube_x_range: tuple[float, float],
    cube_y_range: tuple[float, float],
    checker: IKFeasibilityChecker | None = None,
    max_attempts: int = 50,
):
    """Sample an episode plan; optionally reject IK-infeasible ones.

    Cube is sampled from the supplied ``cube_*_range`` (used to implement
    the train/eval split). With ``checker=None`` this is a single draw
    (legacy behaviour). With a checker, we rejection-sample until the
    planned per-frame trajectory is fully reachable on the target arm, or
    we exhaust ``max_attempts``.

    Returns ``(waypoints | None, stats | None)``. ``stats`` is None when no
    filter was applied; otherwise it carries the rejection-sampling outcome.
    """
    sample_kwargs = dict(cube_x_range=cube_x_range, cube_y_range=cube_y_range)

    if checker is None:
        return sample_episode_waypoints(rng, **sample_kwargs), None

    def builder():
        wp = sample_episode_waypoints(rng, **sample_kwargs)
        return episode_target_poses(wp), wp

    _poses, wp, stats = checker.sample_feasible_trajectory(
        builder, max_attempts=max_attempts,
    )
    return wp, stats


def run_episode(scene_xml: Path, wp: EpisodeWaypoints, use_viewer: bool = False,
                dr_cfg: DRConfig | None = None,
                dr_rng: np.random.Generator | None = None,
                episode_type: str = "normal"):
    """Run a single grasp episode using the supplied plan.

    Episode types control what the dataset teaches the policy:
      * "normal"  — full grasp-and-lift trajectory (the default).
      * "release" — gripper starts CLOSED at home; first phase ramps it open,
        then proceeds with the normal grasp. Gives the model closed→open
        transition examples it would otherwise never see in training.
      * "hover"   — approach + descend as usual, then HOLD at grasp pose
        with the gripper STILL OPEN for `FRAMES_HOVER` frames, then retract.
        No close, no lift. Negative example: "near cube + gripper open
        does NOT mean close yet".

    Success criteria differ:
      * normal/release: cube lifted >= LIFT_SUCCESS_THRESHOLD above start.
      * hover:          cube did NOT move significantly (no accidental nudge).

    Returns (frames, success, cube_start_xy, home_xyz, grasp_dbg, cube_final_z).
    """
    assert episode_type in ("normal", "release", "hover"), episode_type
    sim = Simulation(scene_xml=scene_xml)
    if dr_cfg is not None:
        randomize_scene(sim.model, dr_rng, dr_cfg)
    viewer = sim.launch_passive_viewer() if use_viewer else None

    # Resolve ids once.
    mocap_id = sim.model.body("grabette_mocap").mocapid[0]
    prox_id = sim.model.actuator("proximal").id
    dist_id = sim.model.actuator("distal").id
    cube_jnt_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")
    cube_qadr = sim.model.jnt_qposadr[cube_jnt_id]

    # 1. Place cube at the planned position.
    set_cube_pose(sim, cube_qadr, float(wp.cube_xyz[0]), float(wp.cube_xyz[1]), CUBE_START_Z)

    # 2-3. Pull keyframes out of the plan.
    home_xyz, home_quat = wp.home_xyz, wp.home_quat
    mid_xyz, mid_quat = wp.mid_xyz, wp.mid_quat
    sentry_xyz, sentry_quat = wp.sentry_xyz, wp.sentry_quat
    grasp_xyz, grasp_quat = wp.grasp_xyz, wp.grasp_quat
    grasp_dbg = wp.grasp_dbg

    # 4. Reset Grabette to home. Initial gripper state depends on episode type.
    set_grabette_pose(sim, mocap_id, home_xyz, home_quat)
    if episode_type == "release":
        # Release episodes start CLOSED so the policy sees closed→open.
        sim.data.ctrl[prox_id] = PROXIMAL_CLOSED
        sim.data.ctrl[dist_id] = DISTAL_CLOSED
        sim.data.joint("proximal").qpos[0] = PROXIMAL_CLOSED
        sim.data.joint("distal").qpos[0] = DISTAL_CLOSED
        mujoco.mj_forward(sim.model, sim.data)
        initial_ctrl = (PROXIMAL_CLOSED, DISTAL_CLOSED)
    else:
        sim.data.ctrl[prox_id] = PROXIMAL_OPEN
        sim.data.ctrl[dist_id] = DISTAL_OPEN
        initial_ctrl = (PROXIMAL_OPEN, DISTAL_OPEN)

    frames: list[dict] = []

    # 5. Initial settle so the weld stabilizes at home.
    phase_hold(sim, mocap_id, home_xyz, home_quat,
               FRAMES_INITIAL_SETTLE, initial_ctrl,
               prox_id, dist_id, frames, viewer)

    # 5b. Release-only: ramp gripper closed -> open before approach.
    if episode_type == "release":
        phase_open(sim, mocap_id, home_xyz, home_quat,
                   FRAMES_OPEN_RAMP, prox_id, dist_id, frames, viewer)

    # Cube starting position after the settle (for the success check).
    cube_pos_start = sim.data.body("red_cube").xpos.copy()

    # 6. Approach: home -> mid -> sentry. Two segments: seg1 carries the
    # orientation rotation from home_quat to grasp_quat (and arcs upward via
    # mid); seg2 arrives at the sentry with orientation already aligned.
    seg1 = int(FRAMES_APPROACH * 0.60)
    seg2 = FRAMES_APPROACH - seg1
    phase_multiwaypoint_smooth(
        sim, mocap_id,
        waypoints_pos=[home_xyz, mid_xyz, sentry_xyz],
        waypoints_quat=[home_quat, mid_quat, sentry_quat],
        segment_frames=[seg1, seg2],
        gripper_ctrl=(PROXIMAL_OPEN, DISTAL_OPEN),
        prox_id=prox_id, dist_id=dist_id,
        frames=frames, viewer=viewer,
    )

    # 7. Descend: sentry -> grasp_xyz (linear push along the gripper's
    # approach axis, orientation fixed at grasp_quat).
    phase_smooth(sim, mocap_id,
                 sentry_xyz, grasp_quat, grasp_xyz, grasp_quat,
                 FRAMES_DESCEND, (PROXIMAL_OPEN, DISTAL_OPEN),
                 prox_id, dist_id, frames, viewer)

    # 8. Pre-grip settle so the weld converges before closing/hovering.
    phase_hold(sim, mocap_id, grasp_xyz, grasp_quat,
               FRAMES_PRE_GRIP_SETTLE, (PROXIMAL_OPEN, DISTAL_OPEN),
               prox_id, dist_id, frames, viewer)

    if episode_type == "hover":
        # 8b. HOVER: stay at grasp pose with the gripper STILL OPEN.
        # Negative example to teach the policy that proximity to cube is
        # not the same as "close gripper now".
        phase_hold(sim, mocap_id, grasp_xyz, grasp_quat,
                   FRAMES_HOVER, (PROXIMAL_OPEN, DISTAL_OPEN),
                   prox_id, dist_id, frames, viewer)

        # 8c. Retract back to home, gripper still open.
        retract_target = home_xyz + np.array([0.0, 0.0, RETRACT_EXTRA])
        phase_smooth(sim, mocap_id,
                     grasp_xyz, grasp_quat, retract_target, home_quat,
                     FRAMES_RETRACT, (PROXIMAL_OPEN, DISTAL_OPEN),
                     prox_id, dist_id, frames, viewer)

        # 8d. Final settle (still open).
        phase_hold(sim, mocap_id, retract_target, home_quat,
                   FRAMES_FINAL_SETTLE, (PROXIMAL_OPEN, DISTAL_OPEN),
                   prox_id, dist_id, frames, viewer)

        cube_final = sim.data.body("red_cube").xpos.copy()
        # Hover success: cube was NOT disturbed. We don't want to teach the
        # model that hovering is OK if the gripper accidentally pushed the
        # cube during the approach.
        cube_drift_m = float(np.linalg.norm(cube_final - cube_pos_start))
        success = cube_drift_m < 0.02

        if viewer is not None:
            viewer.close()
        return (frames, success,
                (float(cube_pos_start[0]), float(cube_pos_start[1])),
                tuple(home_xyz.tolist()),
                grasp_dbg,
                float(cube_final[2]))

    # 9. Close the gripper (ramp ctrl, mocap fixed).
    phase_close(sim, mocap_id, grasp_xyz, grasp_quat,
                FRAMES_CLOSE, prox_id, dist_id, frames, viewer)

    # 10. Hold closed.
    phase_hold(sim, mocap_id, grasp_xyz, grasp_quat,
               FRAMES_HOLD, (PROXIMAL_CLOSED, DISTAL_CLOSED),
               prox_id, dist_id, frames, viewer)

    # 11. Lift: smoothstep up by LIFT_HEIGHT, orientation held.
    lift_target = wp.lift_xyz
    phase_smooth(sim, mocap_id,
                 grasp_xyz, grasp_quat, lift_target, grasp_quat,
                 FRAMES_LIFT, (PROXIMAL_CLOSED, DISTAL_CLOSED),
                 prox_id, dist_id, frames, viewer)

    # 12. Retract: lift -> home + extra (slerp orientation back to home_quat).
    retract_target = home_xyz + np.array([0.0, 0.0, RETRACT_EXTRA])
    phase_smooth(sim, mocap_id,
                 lift_target, grasp_quat, retract_target, home_quat,
                 FRAMES_RETRACT, (PROXIMAL_CLOSED, DISTAL_CLOSED),
                 prox_id, dist_id, frames, viewer)

    # 13. Final settle.
    phase_hold(sim, mocap_id, retract_target, home_quat,
               FRAMES_FINAL_SETTLE, (PROXIMAL_CLOSED, DISTAL_CLOSED),
               prox_id, dist_id, frames, viewer)

    cube_final = sim.data.body("red_cube").xpos.copy()
    success = (cube_final[2] - cube_pos_start[2]) > LIFT_SUCCESS_THRESHOLD

    if viewer is not None:
        viewer.close()

    return (frames, success,
            (float(cube_pos_start[0]), float(cube_pos_start[1])),
            tuple(home_xyz.tolist()),
            grasp_dbg,
            float(cube_final[2]))


# ----- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect scripted grasp-and-lift demos for the virtual "
                    "Grabette (Stage 4 diagonal grasp) as a LeRobotDataset.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--repo_id", type=str, required=True,
        help="Dataset repo id, e.g. 'sim_grabette_grasp'. Local-only; no Hub push.")
    parser.add_argument(
        "--output_root", type=str, default=None,
        help="Optional explicit local dataset root. If unset, LeRobot uses its "
             "standard cache (~/.cache/huggingface/lerobot/<repo_id>).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", action="store_true",
                        help="Open the MuJoCo passive viewer per episode (off by default).")

    # Train/eval split.
    parser.add_argument(
        "--split", choices=("train", "eval"), default="train",
        help="Cube-position split. 'train' samples from the main cube area "
             "minus the held-out strip; 'eval' samples only from the held-out "
             "strip (a 2cm band on the +y side). Used to test generalisation.")

    # IK feasibility filter (UMI-style data prefiltering).
    ik = parser.add_argument_group("IK feasibility filter")
    ik.add_argument("--ik_filter", dest="ik_filter", action="store_true", default=True,
                    help="Reject sampled plans whose per-frame poses are not all "
                         "reachable by the OpenArm. Default: ON.")
    ik.add_argument("--no_ik_filter", dest="ik_filter", action="store_false",
                    help="Disable the IK feasibility filter (record everything).")
    ik.add_argument("--max_ik_attempts", type=int, default=200,
                    help="Max rejection-sampling attempts per episode before "
                         "the episode is dropped. Default: 200. Each failed "
                         "attempt early-stops at the first infeasible frame, "
                         "so the cost is small (~5ms per fail).")
    ik.add_argument("--ik_pos_tol", type=float, default=0.015,
                    help="Per-frame position tolerance in metres (default 1.5cm).")
    ik.add_argument("--ik_rot_tol_deg", type=float, default=30.0,
                    help="Per-frame rotation tolerance in degrees. The IK is "
                         "weighted to prioritise position (Placo position task "
                         "100x orientation task), so we record whatever "
                         "orientation the arm achieves at the right position. "
                         "Default 30° is loose enough that 70% of trajectories "
                         "pass the filter while keeping pose realistic.")

    # Episode-type mix: standard grasp + a few release / hover episodes.
    et = parser.add_argument_group("Episode types (for diversity beyond grasp-only)")
    et.add_argument("--release_fraction", type=float, default=0.15,
                    help="Fraction of episodes that start with the gripper "
                         "CLOSED and ramp it open before approach. Teaches the "
                         "policy closed→open transitions it would otherwise "
                         "never see in a grasp-only dataset. Default 0.15.")
    et.add_argument("--hover_fraction", type=float, default=0.15,
                    help="Fraction of episodes that approach the cube and "
                         "HOVER with gripper still open instead of grasping. "
                         "Negative example: 'near cube + open gripper does NOT "
                         "mean close yet'. Default 0.15.")

    # Visual domain randomization (nuisance variation only — same red cube,
    # no distractors, no semantic change to the task).
    dr = parser.add_argument_group("Visual domain randomization")
    dr.add_argument("--dr", dest="dr", action="store_true", default=True,
                    help="Per-episode visual DR: light intensity/direction, "
                         "table colour jitter, camera position jitter. ON by default.")
    dr.add_argument("--no_dr", dest="dr", action="store_false",
                    help="Disable visual DR (every episode looks the same).")
    dr.add_argument("--dr_light_intensity_lo", type=float, default=0.7)
    dr.add_argument("--dr_light_intensity_hi", type=float, default=1.3)
    dr.add_argument("--dr_light_dir_jitter_deg", type=float, default=15.0,
                    help="Half-angle of cone for light direction jitter (deg).")
    dr.add_argument("--dr_material_rgb_jitter", type=float, default=0.10,
                    help="Per-channel RGB additive jitter on the table material.")
    dr.add_argument("--dr_camera_pos_jitter_m", type=float, default=0.01,
                    help="Per-axis position jitter on the gripper camera (m).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)
    # Distinct rng for DR so changing DR settings doesn't shuffle the cube/home
    # samples (deterministic ablation across DR settings at fixed --seed).
    dr_rng = np.random.default_rng(args.seed + 10_000)

    # Train / eval split picks the cube-position distribution.
    if args.split == "eval":
        cube_x_range = EVAL_CUBE_X_RANGE
        cube_y_range = EVAL_CUBE_Y_RANGE
    else:
        cube_x_range = TRAIN_CUBE_X_RANGE
        cube_y_range = TRAIN_CUBE_Y_RANGE
    logger.info(
        f"Split: {args.split}; cube_x∈{cube_x_range}, cube_y∈{cube_y_range}."
    )

    # Build the (optional) IK feasibility checker. Same arm seed and tolerances
    # as `check_grabette_reachable.py`, so this filter accepts exactly the
    # episodes that script would call "fully reachable".
    checker: IKFeasibilityChecker | None = None
    if args.ik_filter:
        from check_grabette_reachable import ARM_IK_SEED  # avoid duplicating the seed
        checker = IKFeasibilityChecker(
            Kinematics(),
            frame=CAMERA_FRAME,
            seed_joints=ARM_IK_SEED,
            pos_tol_m=args.ik_pos_tol,
            rot_tol_deg=args.ik_rot_tol_deg,
            n_iter=200,
        )
        logger.info(
            f"IK filter ENABLED (pos_tol={args.ik_pos_tol*1000:.1f}mm, "
            f"rot_tol={args.ik_rot_tol_deg:.1f}deg, max_attempts={args.max_ik_attempts}). "
            "Plans not reachable by the OpenArm will be resampled before physics."
        )
    else:
        logger.info("IK filter DISABLED — all sampled plans go to physics.")

    # Visual DR config (None disables the channel entirely).
    dr_cfg: DRConfig | None = None
    if args.dr:
        dr_cfg = DRConfig(
            headlight_intensity_range=(args.dr_light_intensity_lo, args.dr_light_intensity_hi),
            light_dir_jitter_rad=float(np.deg2rad(args.dr_light_dir_jitter_deg)),
            light_names=("main_light",),
            material_rgb_jitter=args.dr_material_rgb_jitter,
            material_names=("wood",),
            camera_pos_jitter_m=args.dr_camera_pos_jitter_m,
            camera_names=("gripette_cam",),
        )
        logger.info(
            f"Visual DR ENABLED (headlight×{args.dr_light_intensity_lo}-"
            f"{args.dr_light_intensity_hi}, light_dir±{args.dr_light_dir_jitter_deg}°, "
            f"wood RGB±{args.dr_material_rgb_jitter}, cam±{args.dr_camera_pos_jitter_m*1000:.0f}mm)."
        )
    else:
        logger.info("Visual DR DISABLED — every episode looks identical.")

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
    n_ik_dropped = 0
    ik_attempts_total = 0
    saved_frame_counts: list[int] = []

    print(f"{'ep':>4} {'cube_x':>8} {'cube_y':>8} "
          f"{'home_x':>7} {'home_y':>7} {'home_z':>7} "
          f"{'g_pitch':>7} {'g_azim':>7} "
          f"{'final_z':>9} {'frames':>7} {'ik_try':>6} {'dt_s':>6} {'result':>7}")
    # Episode-type sampler (deterministic given args.seed via a separate rng).
    type_rng = np.random.default_rng(args.seed + 20_000)
    p_release = float(args.release_fraction)
    p_hover = float(args.hover_fraction)
    p_normal = max(0.0, 1.0 - p_release - p_hover)
    type_counts = {"normal": 0, "release": 0, "hover": 0}
    type_success = {"normal": 0, "release": 0, "hover": 0}
    logger.info(
        f"Episode-type mix: normal={p_normal:.2f}, release={p_release:.2f}, hover={p_hover:.2f}"
    )

    pbar = tqdm(
        range(args.episodes),
        desc="episodes",
        unit="ep",
        smoothing=0.1,
        dynamic_ncols=True,
    )
    for ep in pbar:
        t0 = time.perf_counter()

        # Choose this episode's type from the mix.
        u = type_rng.random()
        if u < p_release:
            episode_type = "release"
        elif u < p_release + p_hover:
            episode_type = "hover"
        else:
            episode_type = "normal"
        type_counts[episode_type] += 1

        # Plan an episode (with optional IK rejection sampling).
        wp, ik_stats = plan_episode(
            rng,
            cube_x_range=cube_x_range,
            cube_y_range=cube_y_range,
            checker=checker,
            max_attempts=args.max_ik_attempts,
        )
        ik_try = ik_stats.n_attempts if ik_stats is not None else 0
        ik_attempts_total += ik_try

        if wp is None:
            # Filter exhausted — skip this episode.
            n_ik_dropped += 1
            dt = time.perf_counter() - t0
            tqdm.write(f"{ep:4d} {'-':>8} {'-':>8} {'-':>7} {'-':>7} {'-':>7} "
                       f"{'-':>7} {'-':>7} {'-':>9} {'-':>7} {ik_try:6d} {dt:6.1f} "
                       f"{'IK_FAIL':>7}")
            pbar.set_postfix(saved=n_saved, ik_drop=n_ik_dropped, refresh=False)
            continue

        frames, success, (cx, cy), home_xyz, grasp_dbg, final_z = run_episode(
            SCENE, wp, use_viewer=args.viewer, dr_cfg=dr_cfg, dr_rng=dr_rng,
            episode_type=episode_type,
        )
        dt = time.perf_counter() - t0

        if success:
            type_success[episode_type] += 1
        result = ("OK" if success else "FAIL") + f"[{episode_type[0]}]"
        tqdm.write(f"{ep:4d} {cx:8.3f} {cy:8.3f} "
                   f"{home_xyz[0]:7.3f} {home_xyz[1]:7.3f} {home_xyz[2]:7.3f} "
                   f"{grasp_dbg['grasp_pitch_deg']:7.1f} {grasp_dbg['grasp_azimuth_deg']:7.1f} "
                   f"{final_z:9.4f} {len(frames):7d} {ik_try:6d} {dt:6.1f} {result:>7}")

        # Discard failures: never call add_frame for them.
        if not success:
            pbar.set_postfix(saved=n_saved, ik_drop=n_ik_dropped, refresh=False)
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
        saved_frame_counts.append(len(frames))
        pbar.set_postfix(saved=n_saved, ik_drop=n_ik_dropped, refresh=False)

    pbar.close()
    dataset.finalize()

    rate = 100.0 * n_success / args.episodes if args.episodes else 0.0
    avg_frames = float(np.mean(saved_frame_counts)) if saved_frame_counts else 0.0
    logger.info(
        f"Done. Saved {n_saved}/{args.episodes} episodes "
        f"({n_success} success, {rate:.1f}%). "
        f"Avg frames per saved episode: {avg_frames:.1f}. "
        f"Dataset root: {dataset.root}")
    logger.info(
        "Per-type breakdown (success/total): "
        f"normal={type_success['normal']}/{type_counts['normal']}, "
        f"release={type_success['release']}/{type_counts['release']}, "
        f"hover={type_success['hover']}/{type_counts['hover']}"
    )
    if checker is not None:
        n_planned = args.episodes - n_ik_dropped
        # Avg attempts on the accepted episodes only (dropped episodes always
        # used the full budget, so mixing them in would obscure the rate).
        accepted_attempts = ik_attempts_total - n_ik_dropped * args.max_ik_attempts
        avg_accepted = (accepted_attempts / max(n_planned, 1)) if n_planned else 0.0
        drop_rate = 100.0 * n_ik_dropped / max(args.episodes, 1)
        logger.info(
            f"IK filter: {n_ik_dropped}/{args.episodes} episodes dropped "
            f"(no feasible plan within {args.max_ik_attempts} attempts); "
            f"avg attempts on accepted plans: {avg_accepted:.1f}."
        )
        if drop_rate > 30.0:
            logger.warning(
                f"IK drop rate is {drop_rate:.0f}%. Either bump "
                "--max_ik_attempts (each rejected attempt is cheap, ~5ms), or "
                "tighten HOME_*_RANGE / GRASP_TILT_RANGE_DEG / "
                "GRASP_AZIMUTH_RANGE_DEG in grabette_trajectory.py to bring "
                "the sampling distribution closer to the arm's workspace."
            )


if __name__ == "__main__":
    main()
