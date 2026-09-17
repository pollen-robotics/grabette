"""Where is the grasp point in the image? Geometry instead of colour.

Colour cannot find the sugar cube: its tan is not unique against a patterned
white cloth, and every mask either claimed a third of the frame or latched onto
a shadow. But the dataset knows exactly where the gripper is going, and the
camera is calibrated, so the answer can be computed.

  1. The 11-dim action gives, per step, a delta translation d_k in the camera
     frame at step k and a delta rotation R_k (Zhou 6D, verified: mean dr6d is
     the identity's first two columns, median per-step rotation 0.73 deg).
  2. Chain them to put the grasp-time camera position into the CURRENT camera
     frame:  p = d_t + R_t d_(t+1) + R_t R_(t+1) d_(t+2) + ...
  3. Project p through the calibrated fisheye model, rescaled from the
     1296x972 calibration to the dataset's 480x360 frames.

Self-checking by construction: the grasp point is a fixed point in the world,
so its projection must stay pinned to the same physical spot as the camera
moves through the approach. If the marker slides across the scene, the chain
is wrong and nothing downstream should be believed.

One honest offset: p is where the CAMERA will be at grasp time, and the camera
sits behind the fingertips, so the marker lands slightly beyond the cube along
the viewing direction rather than exactly on it. The camera-to-fingertip
transform is not in this dataset's metadata.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from grabette_attention.sources import DatasetSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G

CALIB = ("/home/steve/Project/Repo/GRABETTE/universal_manipulation_interface/"
         "rpi_bno080_calibration/rpi_camera_intrinsics.json")
DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
CAM = "observation.images.cam0"
EPISODES = (0, 1, 2)
CLOSURE = 10
ROWS, COLS = 12, 16


def six_d_to_matrix(six):
    """Zhou et al. 6D -> rotation matrix, falling back to identity."""
    a1, a2 = np.asarray(six[:3], float), np.asarray(six[3:], float)
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        return np.eye(3)
    b1 = a1 / n1
    a2p = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2p)
    if n2 < 1e-8:
        return np.eye(3)
    b2 = a2p / n2
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def load_camera(width, height):
    """Fisheye intrinsics rescaled to the dataset's frame size."""
    c = json.load(open(CALIB))
    i = c["intrinsics"]
    s = width / c["image_width"]
    assert abs(s - height / c["image_height"]) < 1e-3, "non-uniform rescale"
    f = i["focal_length"] * s
    return {
        "fx": f,
        "fy": f * i["aspect_ratio"],
        "cx": i["principal_pt_x"] * s,
        "cy": i["principal_pt_y"] * s,
        "k": (i["radial_distortion_1"], i["radial_distortion_2"],
              i["radial_distortion_3"], i["radial_distortion_4"]),
    }


def project(p, cam):
    """Equidistant fisheye projection. Returns (u, v) or None if behind."""
    x, y, z = p
    if z <= 1e-6:
        return None
    a, b = x / z, y / z
    r = np.hypot(a, b)
    theta = np.arctan(r)
    k1, k2, k3, k4 = cam["k"]
    t2 = theta * theta
    theta_d = theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
    scale = theta_d / r if r > 1e-9 else 1.0
    return cam["fx"] * a * scale + cam["cx"], cam["fy"] * b * scale + cam["cy"]


def grasp_points(rows, grasp):
    """Grasp-time camera position expressed in each earlier frame's camera frame."""
    out = {}
    for t in range(grasp):
        p = np.zeros(3)
        A = np.eye(3)
        for k in range(t, grasp):
            p = p + A @ rows[k, :3]
            A = A @ six_d_to_matrix(rows[k, 3:9])
        out[t] = p
    return out


def main() -> None:
    tbl = pq.read_table(G.ACTIONS, columns=["episode_index", "action"])
    ep = np.asarray(tbl["episode_index"])
    act = np.stack([np.asarray(r) for r in tbl["action"].to_pylist()]).astype(np.float64)

    figure, axes = plt.subplots(
        len(EPISODES), 5, figsize=(19, 3.6 * len(EPISODES)), dpi=100
    )
    cam = None

    for row, episode in enumerate(EPISODES):
        rows = act[ep == episode]
        closure = rows[:, CLOSURE]
        hot = np.nonzero(closure > 0.5 * max(closure.max(), 1e-6))[0]
        grasp = int(hot[0])
        points = grasp_points(rows, grasp)
        # Five frames spread across the approach.
        picks = [max(2, int(f * grasp)) for f in (0.0, 0.25, 0.5, 0.7, 0.85)]
        print(f"\nep{episode}: grasp at {grasp}, sampling {picks}")

        source = DatasetSource(
            G.REPO, episodes=[episode], camera_keys=(CAM,), task=G.TASK,
            root=G.ROOT, selection=sorted(set(picks)),
        )
        frames = {obs.frame: obs.images[CAM] for obs in source.frames()}

        for col, t in enumerate(picks):
            axis = axes[row, col]
            axis.set_axis_off()
            image = frames.get(t)
            if image is None or t not in points:
                continue
            h, w = image.shape[:2]
            if cam is None:
                cam = load_camera(w, h)
                print(f"  camera: fx {cam['fx']:.1f} fy {cam['fy']:.1f} "
                      f"cx {cam['cx']:.1f} cy {cam['cy']:.1f}")
            p = points[t]
            uv = project(p, cam)
            axis.imshow(image)
            if uv is None:
                title = f"ep{episode} f{t}: grasp behind camera"
            else:
                u, v = uv
                axis.plot(u, v, marker="+", markersize=22, markeredgewidth=2.5,
                          color="#00e5ff")
                axis.plot(u, v, marker="o", markersize=13, markerfacecolor="none",
                          markeredgewidth=2.0, color="#00e5ff")
                gr, gc = int(v // (h / ROWS)), int(u // (w / COLS))
                title = (f"ep{episode} f{t}  ({u:.0f},{v:.0f}) → r{gr} c{gc}\n"
                         f"range {np.linalg.norm(p)*1000:.0f} mm")
                print(f"  f{t:4d}: p = ({p[0]*1000:7.1f},{p[1]*1000:7.1f},"
                      f"{p[2]*1000:7.1f}) mm  -> pixel ({u:6.1f},{v:6.1f})  "
                      f"cell r{gr:2d} c{gc:2d}")
            axis.set_xlim(0, w)
            axis.set_ylim(h, 0)
            axis.set_title(title, fontsize=8)

    figure.suptitle(
        "Grasp point projected from the recorded trajectory and the calibrated "
        "fisheye model\n"
        "the marker must stay on the same physical spot as the camera "
        "approaches — that is the correctness check",
        fontsize=12,
    )
    figure.tight_layout()
    path = DST / "grasp_point_projection.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
