"""Does the finger region light up MORE when the cube is in front of it?

Steve's hypothesis: the model learned the grasp by watching the cube in front
of the fingertips. At the grasp the cube and the fingertips occupy nearly the
same pixels, so a single frame cannot separate "attends to the finger" from
"attends to what is in front of the finger". What separates them is TIME: the
finger is always there, the cube only arrives near the grasp and leaves at the
release.

So: average the attention map within each phase, then subtract the
start-of-approach map. Cells that GAIN attention as the cube arrives are the
relational cue; a purely fixed finger template cancels to zero in the
difference. No object detector needed, and no region hand-drawn by me -- the
difference map says where the change is, wherever that turns out to be.

Ablation is off (it does not enter this question), which halves the cost.
Grids are saved so the analysis can be redone without re-running inference.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grabette_attention.adapters.pi05 import Pi05Adapter
from grabette_attention.analysis import analyse_frame
from grabette_attention.loader import load_pi05
from grabette_attention.sources import DatasetSource

# Import the sibling module by location rather than by cwd, so the script runs
# from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G      # phase detection, paths and prompt live there

DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
NPZ = DST / "pnp_grids.npz"


def collect():
    plan = G.plans()
    total = sum(len(p["frames"]) for p in plan.values())
    print(f"episodes {sorted(plan)}, {total} frames (~{total*6/60:.0f} min)\n")

    policy, pre, post = load_pi05(G.CKPT, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)
    camera = adapter.camera_keys[0]

    grids, phases, meta, frames_rgb = [], [], [], []
    done = 0
    for episode in sorted(plan):
        p = plan[episode]
        source = DatasetSource(
            G.REPO, episodes=[episode], camera_keys=adapter.camera_keys,
            task=G.TASK, root=G.ROOT, selection=sorted(set(p["frames"])),
        )
        by_frame = {obs.frame: obs for obs in source.frames()}
        for want, label in zip(p["frames"], p["labels"]):
            obs = by_frame.get(want)
            if obs is None:
                continue
            result = analyse_frame(adapter, obs, denoise_step="first", ablate=False)
            grids.append(result.cameras[camera].grid)
            phases.append(label)
            meta.append((episode, obs.frame))
            # Keep one representative frame per phase for the overlay figure.
            frames_rgb.append(obs.images[camera][::4, ::4])
            done += 1
            print(f"[{done}/{total}] ep{episode} f{obs.frame} {label:14s} "
                  f"mass {result.cameras[camera].mass:.3f}", flush=True)

    DST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NPZ, grids=np.stack(grids), phases=np.array(phases),
        meta=np.array(meta), frames=np.stack(frames_rgb),
    )
    print(f"\nwrote {NPZ}")


def analyse() -> None:
    data = np.load(NPZ)
    grids, phases = data["grids"], data["phases"]
    n, rows, cols = grids.shape
    flat = grids.reshape(n, -1)
    flat = flat / flat.sum(axis=1, keepdims=True)     # comparable across frames

    order = [p for p in G_ORDER if (phases == p).any()]
    means = {p: flat[phases == p].mean(axis=0).reshape(rows, cols) for p in order}
    base = means[order[0]]

    print(f"{n} frames, grid {rows}x{cols}\n")
    print("per-phase mean attention, summed over the bottom 4 rows (the finger")
    print("band) and over the rest of the frame:")
    print(f"{'phase':<14} {'finger band':>12} {'rest':>8} {'ratio':>7}")
    for p in order:
        band = means[p][-4:, :].sum()
        rest = means[p][:-4, :].sum()
        print(f"{p:<14} {band:12.3f} {rest:8.3f} {band/max(rest,1e-9):7.2f}")

    print("\nlargest GAIN over start-of-approach, per phase (cell, delta):")
    for p in order[1:]:
        d = means[p] - base
        idx = int(np.argmax(d))
        r, c = divmod(idx, cols)
        lost = int(np.argmin(d))
        lr, lc = divmod(lost, cols)
        print(f"  {p:<14} gain r{r:2d} c{c:2d} {d[r,c]:+.4f}   "
              f"loss r{lr:2d} c{lc:2d} {d[lr,lc]:+.4f}")

    # Two strips: the phase means, and the change from the approach baseline.
    figure, axes = plt.subplots(2, len(order), figsize=(3.0 * len(order), 6.4), dpi=110)
    vmax = max(m.max() for m in means.values())
    dmax = max(np.abs(means[p] - base).max() for p in order)
    for i, p in enumerate(order):
        ax = axes[0, i]
        ax.imshow(means[p], cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(p, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax = axes[1, i]
        im = ax.imshow(
            means[p] - base, cmap="coolwarm", vmin=-dmax, vmax=dmax,
            interpolation="nearest",
        )
        ax.set_title(f"{p} − {order[0]}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("mean attention", fontsize=9)
    axes[1, 0].set_ylabel("change vs approach 0%", fontsize=9)
    figure.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02)
    figure.suptitle(
        "Top: attention by phase. Bottom: change from start of approach — RED "
        "gains, BLUE loses.\nA fixed finger template cancels to zero in the "
        "bottom row; only what actually moves survives.",
        fontsize=11,
    )
    path = DST / "pnp_phase_difference_maps.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {path}")


G_ORDER = ["approach 0%", "approach 35%", "approach 70%", "GRASP",
           "carry 30%", "carry 65%", "RELEASE"]


if __name__ == "__main__":
    if not NPZ.exists():
        collect()
    analyse()
