"""Cube, mug and gripper regions against the causal map.

The previous colour-only cube mask claimed 3.5% of the frame across 38 cells —
it was catching the table edge, the mat and the green cup, so its verdict on
the cube was worthless. The cube is small and COMPACT, so add shape: label
connected components of the warm mask and keep only blobs of a plausible cube
size and aspect ratio.

Also adds the gripper fingers (large dark wedges low in the frame), because the
question is precisely whether the finger region — which draws the most
attention — carries the causal effect the attention implies.

Writes a verification image with every accepted region outlined, so the
detector can be checked rather than trusted. No inference: the occlusion grids
are read from disk.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from grabette_attention.sources import DatasetSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G

DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
CAM = "observation.images.cam0"
EPISODE = 0
ROWS, COLS = 12, 16
FRAMES = {2: "approach 0%", 48: "GRASP"}


def blobs(mask, min_area, max_area, max_aspect, min_fill):
    """Connected components passing size, aspect and solidity constraints."""
    labels, count = ndimage.label(mask)
    out = []
    for i in range(1, count + 1):
        ys, xs = np.nonzero(labels == i)
        area = len(ys)
        if not (min_area <= area <= max_area):
            continue
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        aspect = max(h, w) / max(min(h, w), 1)
        fill = area / (h * w)
        if aspect > max_aspect or fill < min_fill:
            continue
        out.append({
            "area": area, "bbox": (xs.min(), xs.max(), ys.min(), ys.max()),
            "aspect": aspect, "fill": fill, "mask": labels == i,
        })
    return sorted(out, key=lambda b: -b["area"])


def to_cells(mask, h, w):
    ys, xs = np.nonzero(mask)
    return {
        (min(int(y // (h / ROWS)), ROWS - 1), min(int(x // (w / COLS)), COLS - 1))
        for y, x in zip(ys, xs)
    }


def main() -> None:
    source = DatasetSource(
        G.REPO, episodes=[EPISODE], camera_keys=(CAM,), task=G.TASK,
        root=G.ROOT, selection=sorted(FRAMES),
    )
    figure, axes = plt.subplots(1, len(FRAMES), figsize=(15, 5.6), dpi=110)
    panel = 0

    for obs in source.frames():
        rgb = obs.images[CAM]
        h, w = rgb.shape[:2]
        r, g, b = (rgb[..., i].astype(int) for i in range(3))
        label = FRAMES[obs.frame]
        print(f"\n=== ep{EPISODE} f{obs.frame} — {label} ({h}x{w}) ===")

        # Cube: warm, mid-bright, small and compact.
        warm = (r > b + 20) & (r > 120) & (r < 225) & (g > b + 8)
        cubes = blobs(warm, min_area=150, max_area=2500, max_aspect=2.0, min_fill=0.45)
        print(f"  cube candidates (compact warm blobs): {len(cubes)}")
        for c in cubes[:4]:
            x0, x1, y0, y1 = c["bbox"]
            print(f"    area {c['area']:5d} px  x {x0}-{x1} y {y0}-{y1}  "
                  f"aspect {c['aspect']:.2f} fill {c['fill']:.2f}")

        # Mug: strongly blue. Unique in this scene, so no shape needed.
        mug_mask = (b > r + 25) & (b > g + 15) & (b > 70)
        # Fingers: large dark wedges, lower half of the frame.
        dark = (np.maximum(np.maximum(r, g), b) < 105)
        dark[: int(0.45 * h), :] = False
        fingers = blobs(dark, min_area=600, max_area=40000, max_aspect=6.0, min_fill=0.25)
        print(f"  finger candidates (dark low blobs): {len(fingers)}")
        for f in fingers[:3]:
            x0, x1, y0, y1 = f["bbox"]
            print(f"    area {f['area']:5d} px  x {x0}-{x1} y {y0}-{y1}")

        causal = np.load(DST / f"occlusion_ep{EPISODE}_f{obs.frame}.npy")
        grid_cells = {(rr, cc) for rr in range(ROWS) for cc in range(COLS)}

        regions = {}
        if cubes:
            regions["cube"] = to_cells(cubes[0]["mask"], h, w)
        if mug_mask.sum() > 40:
            regions["mug"] = to_cells(mug_mask, h, w)
        if fingers:
            union = np.zeros_like(dark)
            for f in fingers[:2]:
                union |= f["mask"]
            regions["fingers"] = to_cells(union, h, w)

        print(f"  {'region':<9} {'cells':>5} {'mean causal':>12} {'elsewhere':>10} {'ratio':>7}")
        for name, cells in regions.items():
            inside = np.array([causal[c] for c in cells])
            outside = np.array([causal[c] for c in grid_cells - cells])
            print(f"  {name:<9} {len(cells):5d} {inside.mean():12.2f} "
                  f"{outside.mean():10.2f} {inside.mean()/max(outside.mean(),1e-9):7.2f}x")

        axis = axes[panel]
        axis.imshow(rgb)
        colours = {"cube": "yellow", "mug": "deepskyblue", "fingers": "red"}
        for name, cells in regions.items():
            for (rr, cc) in cells:
                axis.add_patch(mpatches.Rectangle(
                    (cc * w / COLS, rr * h / ROWS), w / COLS, h / ROWS,
                    fill=False, edgecolor=colours[name], linewidth=1.4,
                ))
        axis.set_title(
            f"ep{EPISODE} f{obs.frame} — {label}\n"
            "yellow=cube  blue=mug  red=fingers", fontsize=10,
        )
        axis.set_axis_off()
        panel += 1

    figure.suptitle(
        "Detector check: are the regions actually where the objects are?",
        fontsize=12,
    )
    figure.tight_layout()
    path = DST / "region_detector_check.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
