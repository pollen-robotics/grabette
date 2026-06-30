"""Replay a recorded LeRobot dataset to visually check it.

Plays each episode's RECORDED data back, in sync:
  * "recorded cam0" (OpenCV window): the actual frames stored in the dataset —
    the exact images the policy sees (camera framing, distortion, resolution).
  * "sim gripette_cam (fisheye)" (OpenCV window): the sim camera re-rendered at
    the replayed pose with the SAME KB8 fisheye distortion the data uses — an
    apples-to-apples view of the camera model (the cube is at the scene default,
    so only the cube position differs from the recorded frame).
  * MuJoCo passive viewer: the free-floating gripper driven to each frame's
    recorded ACTION pose (the oak_l/CONTROL_FRAME pose + gripper angles), so you
    see the recorded trajectory in 3D.

The dataset stores no cube position, so the free-floating 3D viewer's cube sits at
the scene default — judge the *grasp* from the recorded cam0 window, the
*trajectory* from the 3D viewer.

With `--arm`, instead of the free-floating gripper the recorded oak_l poses are
replayed on the FULL ARM (per-frame IK + physics, like the eval server): this
checks the saved episodes are actually arm-executable, and shows the grasp on the
arm. The cube isn't stored, so it's placed at the position the trajectory implies
(the gripper pocket at the most-closed frame), and per-episode grasp success
(cube lifted ≥ 5 cm) is reported. Requires the raw 8-D dataset.

Usage (needs a display — do NOT set MUJOCO_GL=egl):
    uv run python examples/replay_dataset.py --repo_id sim_grabette_grasp
    uv run python examples/replay_dataset.py --repo_id my/ds --output_root /tmp/my_ds --episodes 3
    uv run python examples/replay_dataset.py --repo_id my/ds --output_root /tmp/my_ds --arm
"""
import argparse
import time
import sys
from pathlib import Path

import numpy as np
import cv2
import mujoco
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from grabette_trajectory import CUBE_START_Z, GRASP_OFFSET_BODY  # noqa: E402
from check_grabette_reachable import ARM_IK_SEED  # noqa: E402
from openarm_gripette_simu import Simulation
from openarm_gripette_simu.kinematics import Kinematics, CONTROL_FRAME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SCENE = Path(__file__).parent.parent / "scenes" / "grabette_grasp.xml"
ARM_SCENE = Path(__file__).parent.parent / "scenes" / "table_grasp.xml"
ARM_SIM_SUBSTEPS = 33          # ~30 fps dynamic replay, matching the dataset rate
PROXIMAL_CMD_SIGN = -1.0       # arm proximal closes POSITIVE; recorded is free-floating (negative)
LIFT_SUCCESS_M = 0.05


def to_bgr(img):
    """LeRobot video frame -> HWC uint8 BGR for cv2.imshow."""
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):     # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:                          # float [0,1] -> uint8
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _pose_T(action):
    """4x4 oak_l target from an 8-D action [x,y,z, ax,ay,az, prox, dist]."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(action[3:6]).as_matrix()
    T[:3, 3] = action[:3]
    return T


def replay_on_arm(ds, n_show, speed):
    """Replay the recorded oak_l trajectory on the full arm (per-frame IK +
    physics), placing the cube at the trajectory-implied grasp point so the grasp
    actually happens, and reporting per-episode lift success."""
    sim = Simulation(scene_xml=str(ARM_SCENE))
    viewer = sim.launch_passive_viewer()
    kin = Kinematics(orientation_weight=10.0)
    prox_id = sim.model.actuator("proximal").id
    dist_id = sim.model.actuator("distal").id
    cube_qadr = sim.model.jnt_qposadr[
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_joint")]
    dt = 1.0 / (ds.fps * speed)
    # Frame range per episode from the episode_index column (cheap, no video
    # decode; robust across lerobot versions that lack episode_data_index).
    eidx = np.asarray(ds.hf_dataset["episode_index"])
    n_ok = 0
    for ep in range(n_show):
        w = np.where(eidx == ep)[0]
        f0, f1 = int(w[0]), int(w[-1]) + 1
        frames = [ds[i] for i in range(f0, f1)]
        acts = np.stack([np.asarray(f["action"], dtype=float) for f in frames])

        # Place the cube where the trajectory implies it: at the most-closed
        # frame, the gripper pocket (GRASP_OFFSET_BODY in the oak_l frame) is on
        # the cube. Keep it on the table (z = CUBE_START_Z).
        g = int(np.argmin(acts[:, 6]))            # most-negative proximal = most closed
        Rg = Rotation.from_rotvec(acts[g, 3:6]).as_matrix()
        cube_xy = (Rg @ GRASP_OFFSET_BODY + acts[g, :3])[:2]
        sim.data.qpos[cube_qadr:cube_qadr + 3] = (cube_xy[0], cube_xy[1], CUBE_START_Z)
        sim.data.qpos[cube_qadr + 3:cube_qadr + 7] = (1.0, 0.0, 0.0, 0.0)

        # Home the arm at the first recorded pose.
        arm_q = kin.inverse(_pose_T(acts[0]), current_joint_positions=ARM_IK_SEED.copy(),
                            n_iter=200, frame=CONTROL_FRAME)
        sim.reset_arm(arm_q)
        sim.data.qvel[:] = 0
        mujoco.mj_forward(sim.model, sim.data)
        cube_z0 = float(sim.data.body("red_cube").xpos[2])
        print(f"  episode {ep}: cube at ({cube_xy[0]:.3f}, {cube_xy[1]:.3f})")

        quit_now = False
        for fr, a in zip(frames, acts):
            arm_q = kin.inverse(_pose_T(a), current_joint_positions=arm_q,
                                n_iter=50, frame=CONTROL_FRAME)
            sim.set_arm_commands(arm_q)
            sim.data.ctrl[prox_id] = PROXIMAL_CMD_SIGN * a[6]   # arm proximal sign
            sim.data.ctrl[dist_id] = a[7]
            for _ in range(ARM_SIM_SUBSTEPS):
                sim.step()
            viewer.sync()
            cv2.imshow("recorded cam0", to_bgr(fr["observation.images.cam0"]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                quit_now = True
                break
            time.sleep(dt)
        lift = float(sim.data.body("red_cube").xpos[2]) - cube_z0
        ok = lift > LIFT_SUCCESS_M
        n_ok += ok
        print(f"    grasp {'OK' if ok else 'FAIL'} (cube lift {lift * 1000:+.0f} mm)")
        if quit_now:
            break
    print(f"arm replay: {n_ok}/{n_show} grasped (lift >= {LIFT_SUCCESS_M*1000:.0f} mm)")
    viewer.close()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--output_root", default=None,
                    help="Dataset root (omit to use the standard HF cache).")
    ap.add_argument("--episodes", type=int, default=5, help="How many episodes to replay.")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    ap.add_argument("--arm", action="store_true",
                    help="Replay the recorded oak_l poses on the FULL ARM (IK + physics) "
                         "instead of the free-floating gripper; reports grasp success.")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.output_root)
    dt = 1.0 / (ds.fps * args.speed)
    n_show = min(args.episodes, ds.num_episodes)
    adim = len(np.asarray(ds[0]["action"]))
    # The 3D pose replay needs the RAW 8-D absolute-pose action
    # [x,y,z, ax,ay,az, proximal, distal] that collect_grasp_dataset writes.
    # A convert_dataset.py output has 11-D DELTA actions, which can't be replayed
    # as absolute poses — fall back to image-only for those.
    do_3d = (adim == 8)
    print(f"{ds.num_episodes} episodes / {ds.num_frames} frames @ {ds.fps} fps "
          f"-> replaying {n_show} episode(s); action dim={adim} "
          f"({'8-D raw: 3D + image' if do_3d else 'not 8-D: recorded image only'})")
    if not do_3d:
        print("  (point at the RAW collected dataset for the 3D trajectory view)")

    if args.arm:
        if not do_3d:
            print(f"  --arm needs the raw 8-D dataset (got {adim}-D); aborting.")
            return
        replay_on_arm(ds, n_show, args.speed)
        return

    viewer = m = d = None
    if do_3d:
        sim = Simulation(scene_xml=str(SCENE))
        viewer = sim.launch_passive_viewer()
        m, d = sim.model, sim.data
        fj = m.joint("grabette_freejoint").qposadr[0]
        pj = m.joint("proximal").qposadr[0]
        dj = m.joint("distal").qposadr[0]

    cur_ep = -1
    sim_cam_ok = do_3d   # render the distorted gripette cam unless GL conflicts
    for i in range(ds.num_frames):
        s = ds[i]
        ep = int(s["episode_index"])
        if ep >= n_show:
            break
        if ep != cur_ep:
            cur_ep = ep
            print(f"  episode {ep}")
            time.sleep(0.4)

        rec = to_bgr(s["observation.images.cam0"])
        if do_3d:
            a = np.asarray(s["action"], dtype=float)
            pos, rotvec, prox, dist = a[:3], a[3:6], a[6], a[7]
            qx, qy, qz, qw = Rotation.from_rotvec(rotvec).as_quat()
            d.qpos[fj:fj + 3] = pos
            d.qpos[fj + 3:fj + 7] = [qw, qx, qy, qz]     # MuJoCo (w,x,y,z)
            d.qpos[pj], d.qpos[dj] = prox, dist
            mujoco.mj_forward(m, d)
            viewer.sync()
            if sim_cam_ok:
                # Re-render the gripette camera with the SAME KB8 fisheye
                # distortion the dataset uses, at the replayed pose + recorded
                # resolution. Table/gripper framing + distortion should match the
                # recorded cam0; only the cube differs (it's at the scene default,
                # not stored in the dataset).
                try:
                    sim_cam = sim.render_camera(out_size=(rec.shape[1], rec.shape[0]))
                    cv2.imshow("sim gripette_cam (fisheye)", to_bgr(sim_cam))
                except Exception as e:
                    print(f"  (sim camera render disabled: {e})")
                    sim_cam_ok = False

        cv2.imshow("recorded cam0", rec)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(dt)

    if viewer is not None:
        viewer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
