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

The dataset stores no cube position, so the 3D viewer's cube sits at the scene
default — judge the *grasp* from the recorded cam0 window, the *trajectory* from
the 3D viewer.

Usage (needs a display — do NOT set MUJOCO_GL=egl):
    uv run python examples/replay_dataset.py --repo_id sim_grabette_grasp
    uv run python examples/replay_dataset.py --repo_id my/ds --output_root /tmp/my_ds --episodes 3
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
from openarm_gripette_simu import Simulation
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SCENE = Path(__file__).parent.parent / "scenes" / "grabette_grasp.xml"


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--output_root", default=None,
                    help="Dataset root (omit to use the standard HF cache).")
    ap.add_argument("--episodes", type=int, default=5, help="How many episodes to replay.")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
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
