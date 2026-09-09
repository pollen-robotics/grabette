"""Segment the gripper by cross-EPISODE variance at a matched phase.

The within-episode version failed for a real reason: image motion scales with
proximity, so distant background is nearly as static as the gripper and a
variance threshold cannot tell them apart.

Comparing across episodes at the SAME phase removes that confound. At the
grasp the jaws are in the same pose every time, while the scene is different
wholesale — the table sits elsewhere, the objects are placed differently, the
background shifts. So anything with low variance ACROSS EPISODES is attached
to the camera, whatever its distance.

Uses the frames already stored in pnp_grids.npz (downsampled 4x, 90x120 —
ample for a 12x16 cell decision), so this costs no video decoding and no
inference.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
ROWS, COLS = 12, 16
EPISODE = 0
OCCLUSION_FRAMES = {2: "approach 0%", 48: "GRASP"}
QUANTILE = 12          # percent: the gripper is the most static thing there is


def main() -> None:
    data = np.load(DST / "pnp_grids.npz")
    frames, phases = data["frames"], data["phases"]
    print(f"{len(frames)} frames, {frames.shape[1:]} each, "
          f"phases {sorted(set(phases.tolist()))}\n")

    grey = frames.astype(np.float32).mean(axis=3)
    h, w = grey.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))
    interior = radius < 0.90

    # Cross-episode std within each phase, then averaged over phases: a pixel
    # must be static at EVERY phase to count as gripper.
    per_phase = []
    for phase in ("approach 0%", "GRASP", "carry 30%", "RELEASE"):
        sel = grey[phases == phase]
        if len(sel) < 3:
            continue
        per_phase.append(sel.std(axis=0))
        print(f"  {phase:<14} {len(sel)} episodes, "
              f"median cross-episode std {np.median(sel.std(axis=0)[interior]):.1f}")
    variance = np.mean(per_phase, axis=0)

    cut = np.percentile(variance[interior], QUANTILE)
    gripper = (variance < cut) & interior
    print(f"\ncutoff (p{QUANTILE}) = {cut:.1f}; "
          f"mask covers {100*gripper.mean():.1f}% of the frame")

    cells = set()
    ch, cw = h / ROWS, w / COLS
    for rr in range(ROWS):
        for cc in range(COLS):
            block = gripper[
                int(rr * ch):int((rr + 1) * ch), int(cc * cw):int((cc + 1) * cw)
            ]
            if block.mean() > 0.5:
                cells.add((rr, cc))
    print(f"majority-gripper cells: {len(cells)}")
    print("  " + ", ".join(f"r{r}c{c}" for r, c in sorted(cells)))

    # What does the attention do on those cells?
    grids = data["grids"]
    flat = grids.reshape(len(grids), -1)
    flat = flat / flat.sum(axis=1, keepdims=True)
    mean_attn = flat.mean(axis=0).reshape(ROWS, COLS)
    uniform = 1.0 / (ROWS * COLS)
    everything = {(r, c) for r in range(ROWS) for c in range(COLS)}
    if cells:
        att_in = np.mean([mean_attn[c] for c in cells]) / uniform
        att_out = np.mean([mean_attn[c] for c in everything - cells]) / uniform
        print(f"\nATTENTION on gripper cells: {att_in:.2f}x uniform  "
              f"vs elsewhere {att_out:.2f}x  ({att_in/att_out:.2f}x)")

    print(f"\n{'frame':<14} {'cells':>5} {'causal in':>10} {'causal out':>11} {'ratio':>7}")
    for frame, label in OCCLUSION_FRAMES.items():
        path = DST / f"occlusion_ep{EPISODE}_f{frame}.npy"
        if not path.exists() or not cells:
            continue
        causal = np.load(path)
        inside = np.array([causal[c] for c in cells])
        outside = np.array([causal[c] for c in everything - cells])
        print(f"{label:<14} {len(cells):5d} {inside.mean():10.2f} "
              f"{outside.mean():11.2f} "
              f"{inside.mean()/max(outside.mean(),1e-9):6.2f}x")

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.4), dpi=110)
    axes[0].imshow(frames[list(phases).index("GRASP")])
    axes[0].set_title("a grasp frame", fontsize=10)
    image = axes[1].imshow(variance, cmap="magma")
    axes[1].set_title("cross-episode std at matched phases\n"
                      "dark = same in every episode = attached to the camera",
                      fontsize=10)
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    axes[2].imshow(frames[list(phases).index("GRASP")])
    axes[2].imshow(
        np.ma.masked_where(~gripper, gripper.astype(float)),
        cmap="cool", alpha=0.6, vmin=0, vmax=1,
    )
    axes[2].set_title(f"derived gripper mask (p{QUANTILE} of variance)", fontsize=10)
    for ax in axes:
        ax.set_axis_off()
    figure.suptitle(
        "Gripper segmentation by cross-episode agreement — the scene differs "
        "every episode, the hardware does not", fontsize=12,
    )
    figure.tight_layout()
    out = DST / "gripper_mask_cross_episode.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
