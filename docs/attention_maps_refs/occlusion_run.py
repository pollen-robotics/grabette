"""Attention map vs CAUSAL map, on the same frames.

Answers the finger question by intervention rather than by attention. At the
grasp the cube sits between the jaws, so attention cannot separate "looks at
the finger" from "looks at what is in front of the finger" -- both are the same
pixels. Occlusion can: covering the fingertips and covering the cube are
different interventions with different consequences.

Two frames per episode, chosen to contrast:

  approach 0%  -- the cube is far from the jaws. If the cube carries the
                  motion, its own region should dominate here.
  GRASP        -- the cube is in the jaws. If the relevant thing is the cube
                  rather than the hardware, the hot region should sit on the
                  cube, and covering bare finger away from the cube should
                  matter less.

Grid is 12x16, matching the attention grid's shape after its letterbox rows are
cropped, so the two maps are comparable cell for cell. That costs 193 forward
passes per frame; there is no cheaper way to get a causal map, and a coarser
grid cannot resolve fingertip from cube.
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
from grabette_attention.occlusion import occlusion_saliency
from grabette_attention.sources import DatasetSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G

DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
ROWS, COLS = 12, 16
EPISODE = 0
WANT_PHASES = ("approach 0%", "GRASP")


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    plan = G.plans()[EPISODE]
    picks = [
        (f, lab) for f, lab in zip(plan["frames"], plan["labels"])
        if lab in WANT_PHASES
    ]
    print(f"ep{EPISODE}: {picks}")
    print(f"{len(picks)} frames x ({ROWS*COLS} occlusions + 1) passes\n")

    policy, pre, post = load_pi05(G.CKPT, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)
    camera = adapter.camera_keys[0]

    source = DatasetSource(
        G.REPO, episodes=[EPISODE], camera_keys=adapter.camera_keys,
        task=G.TASK, root=G.ROOT, selection=[f for f, _ in picks],
    )
    frames = {obs.frame: obs for obs in source.frames()}

    figure, axes = plt.subplots(len(picks), 3, figsize=(16.5, 4.6 * len(picks)), dpi=110)
    if len(picks) == 1:
        axes = axes[None, :]

    for row, (want, label) in enumerate(picks):
        obs = frames[want]
        attention = analyse_frame(
            adapter, obs, denoise_step="first", ablate=False
        ).cameras[camera]

        # One noise draw shared by the attention pass' sibling and every
        # occluded pass, so the two maps describe the same sample.
        occ = occlusion_saliency(
            adapter, obs, camera=camera, rows=ROWS, cols=COLS, fill="mean"
        )
        print(f"{label}: baseline {occ.baseline_mm:.1f} mm   "
              f"occlusion delta min {occ.grid.min():.1f} "
              f"median {np.median(occ.grid):.1f} max {occ.grid.max():.1f} mm",
              flush=True)
        top = np.argsort(occ.grid.ravel())[::-1][:6]
        for idx in top:
            r, c = divmod(int(idx), COLS)
            print(f"    causal peak r{r:2d} c{c:2d}  {occ.grid[r, c]:7.1f} mm", flush=True)
        atop = np.argsort(attention.grid.ravel())[::-1][:6]
        for idx in atop:
            r, c = divmod(int(idx), COLS)
            print(f"    attn   peak r{r:2d} c{c:2d}  "
                  f"{attention.grid[r, c] / attention.grid.mean():6.2f}x mean", flush=True)

        frame = obs.images[camera]
        extent = (0, frame.shape[1], frame.shape[0], 0)

        axes[row, 0].imshow(frame)
        axes[row, 0].set_title(f"ep{EPISODE} f{want} — {label}", fontsize=10)

        axes[row, 1].imshow(frame)
        axes[row, 1].imshow(
            attention.grid, extent=extent, interpolation="bilinear",
            alpha=0.55, cmap="inferno",
            vmin=attention.grid.min(),
            vmax=np.percentile(attention.grid, 99),
        )
        axes[row, 1].set_title(
            f"ATTENTION (mass {attention.mass:.2f}, p99 clip)", fontsize=10
        )

        axes[row, 2].imshow(frame)
        image = axes[row, 2].imshow(
            occ.grid, extent=extent, interpolation="bilinear",
            alpha=0.6, cmap="viridis",
        )
        axes[row, 2].set_title(
            f"CAUSAL: mm of chunk change when covered\n"
            f"(baseline motion {occ.baseline_mm:.0f} mm)", fontsize=10
        )
        figure.colorbar(image, ax=axes[row, 2], fraction=0.046)
        for ax in axes[row]:
            ax.set_axis_off()

        np.save(DST / f"occlusion_ep{EPISODE}_f{want}.npy", occ.grid)

    figure.suptitle(
        "Where the policy LOOKS vs where the motion COMES FROM\n"
        f"{G.CKPT.split('/')[-1]}, prompt {G.TASK!r}, {ROWS}x{COLS} occlusion, "
        "mean fill",
        fontsize=12,
    )
    figure.tight_layout()
    path = DST / "occlusion_vs_attention.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
