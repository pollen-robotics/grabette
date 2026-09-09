"""Does the attention follow the GRIPPER or the OBJECT?

The camera is mounted on the gripper, which makes this decidable rather than a
matter of taste: the fingers occupy the SAME image coordinates in every frame
of every episode, while the object's image position varies with where it was
placed and how far the approach has progressed.

So:

  attention on the fingers (or on a border/register artefact)
      -> nearly the same map every frame; a fixed template explains almost
         everything; the residual does not point anywhere in particular.

  attention on the object
      -> the map moves between frames, and the part that moves lands on the
         object's actual image position.

Measures both. Locates the object by its red cap, which is unambiguous in this
scene (the earlier "warm saturated" mask caught the table and a cardboard box
and was useless). Saves every grid to an .npz so the analysis can be redone
without re-running inference.

Ablation is off: this asks about the maps only, which halves the cost per
frame.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

from grabette_attention.adapters.pi05 import Pi05Adapter
from grabette_attention.analysis import analyse_frame
from grabette_attention.loader import load_pi05
from grabette_attention.sources import DatasetSource

CKPT = "SteveNguyen/pick3_graspproj_chunkrel_pi05"
REPO = "SteveNguyen/mustard_graspproj"
ROOT = "/home/steve/.cache/huggingface/lerobot/local-converted/mustard_graspproj"
PARQUET = f"{ROOT}/data/chunk-000/file-000.parquet"
TASK = "pick up the mustard bottle"
CLOSURE = 10
DST = Path("/home/steve/attention_out")
NPZ = DST / "attention_grids.npz"

EPISODES = (0, 2, 4, 6, 8, 13, 17, 21)
FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def cap_cell(rgb: np.ndarray, rows: int, cols: int):
    """Grid cell of the mustard bottle's red cap, or None if not visible.

    Red cap only: strongly red and clearly above both other channels. Nothing
    else in this scene is, and a looser 'warm' mask picks up the table.
    """
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    mask = (r > 110) & (r > g + 45) & (r > b + 45)
    if mask.sum() < 60:
        return None, 0.0
    ys, xs = np.nonzero(mask)
    h, w = rgb.shape[:2]
    row = int(np.median(ys) // (h / rows))
    col = int(np.median(xs) // (w / cols))
    return (min(row, rows - 1), min(col, cols - 1)), 100.0 * mask.mean()


def frame_plan():
    tbl = pq.read_table(PARQUET, columns=["episode_index", "action"])
    ep = np.asarray(tbl["episode_index"])
    act = np.stack([np.asarray(r) for r in tbl["action"].to_pylist()]).astype(np.float64)
    plan = {}
    for e in EPISODES:
        rows = act[ep == e]
        closure = rows[:, CLOSURE]
        hot = np.nonzero(closure > 0.5 * max(closure.max(), 1e-6))[0]
        if hot.size < 5:
            continue
        grasp = int(hot[0])
        plan[e] = sorted({max(2, int(round(f * grasp))) for f in FRACTIONS})
    return plan


def collect():
    plan = frame_plan()
    total = sum(len(v) for v in plan.values())
    print(f"episodes {sorted(plan)}, {total} frames (~{total*6/60:.0f} min, no ablation)")

    policy, pre, post = load_pi05(CKPT, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)
    camera = adapter.camera_keys[0]

    grids, caps, meta, masses = [], [], [], []
    done = 0
    for episode in sorted(plan):
        source = DatasetSource(
            REPO, episodes=[episode], camera_keys=adapter.camera_keys,
            task=TASK, root=ROOT, selection=plan[episode],
        )
        for obs in source.frames():
            result = analyse_frame(adapter, obs, denoise_step="first", ablate=False)
            attention = result.cameras[camera]
            grid = attention.grid
            cell, pct = cap_cell(obs.images[camera], *grid.shape)
            grids.append(grid)
            masses.append(attention.mass)
            caps.append((-1, -1) if cell is None else cell)
            meta.append((episode, obs.frame))
            done += 1
            print(f"[{done}/{total}] ep{episode} f{obs.frame} mass {attention.mass:.2f} "
                  f"cap {cell} ({pct:.2f}% of frame)", flush=True)

    DST.mkdir(parents=True, exist_ok=True)
    np.savez(
        NPZ, grids=np.stack(grids), caps=np.array(caps),
        meta=np.array(meta), masses=np.array(masses),
    )
    print(f"\nwrote {NPZ}")


def analyse_saved() -> None:
    data = np.load(NPZ)
    grids, caps, meta = data["grids"], data["caps"], data["meta"]
    n, rows, cols = grids.shape
    # Normalise each grid so frames with different camera mass are comparable.
    norm = grids.reshape(n, -1)
    norm = norm / norm.sum(axis=1, keepdims=True)

    template = norm.mean(axis=0)
    print(f"\n{n} frames, grid {rows}x{cols}")

    # 1. How much of each map IS the fixed template?
    cos = (norm @ template) / (
        np.linalg.norm(norm, axis=1) * np.linalg.norm(template)
    )
    print("\n1. similarity of each frame's map to the across-frame mean template")
    print(f"   cosine similarity: min {cos.min():.4f}  mean {cos.mean():.4f}  "
          f"max {cos.max():.4f}")
    print("   ~1.00 => every frame has essentially the SAME map, i.e. the")
    print("   attention is locked to the camera frame (fingers / border), not")
    print("   to the scene")

    # 2. Is the peak in the same place every time?
    peaks = np.argmax(norm, axis=1)
    uniq, counts = np.unique(peaks, return_counts=True)
    order = np.argsort(counts)[::-1]
    print("\n2. where the peak cell lands, over all frames")
    for i in order[:5]:
        cell = divmod(int(uniq[i]), cols)
        print(f"   row {cell[0]:2d} col {cell[1]:2d}: {counts[i]:3d}/{n} frames "
              f"({counts[i]/n:5.0%})")

    # 3. Does the object get more than background attention?
    have = caps[:, 0] >= 0
    print(f"\n3. attention ON the object's cap cell ({have.sum()}/{n} frames "
          "with the cap visible)")
    if have.any():
        at_cap = np.array([
            norm[i].reshape(rows, cols)[caps[i, 0], caps[i, 1]]
            for i in np.nonzero(have)[0]
        ])
        uniform = 1.0 / (rows * cols)
        print(f"   mean attention at the cap cell: {at_cap.mean():.5f}")
        print(f"   uniform share for one cell:     {uniform:.5f}")
        print(f"   ratio: {at_cap.mean()/uniform:.2f}x uniform")
        print("   <1 means the object's own cell gets LESS than an indifferent")
        print("   policy would give it")

        # 4. Does the moving part of the map point at the object?
        residual = norm - template
        hits = 0
        for i in np.nonzero(have)[0]:
            peak = np.unravel_index(int(np.argmax(residual[i])), (rows, cols))
            if abs(peak[0] - caps[i, 0]) <= 1 and abs(peak[1] - caps[i, 1]) <= 1:
                hits += 1
        print(f"\n4. residual (map minus template) peaks within one cell of the")
        print(f"   cap in {hits}/{int(have.sum())} frames ({hits/max(have.sum(),1):.0%})")
        print("   chance for a 3x3 window on a 12x16 grid is about 5%")

    # Render the template and the variability, so the fixed part is visible.
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=110)
    for axis, (field, label) in zip(
        axes,
        [(template.reshape(rows, cols), "mean attention across all frames\n"
          "(the part that does NOT move: gripper / border)"),
         (norm.std(axis=0).reshape(rows, cols), "per-cell standard deviation\n"
          "(the part that DOES move between frames)")],
    ):
        image = axis.imshow(field, cmap="inferno", interpolation="nearest")
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("grid col")
        axis.set_ylabel("grid row")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    path = DST / "attention_template_vs_variation.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    if not NPZ.exists():
        collect()
    analyse_saved()
