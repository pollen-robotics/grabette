"""Zoomed frames with a labelled pixel grid, so the cube can be read off by eye.

Locating a visible tan cube in a photograph is a different and far easier task
than interpreting a heatmap, and my record on the latter does not transfer.
Hand annotation here serves two purposes: it answers the question directly for
the two frames the occlusion sweeps used, and it gives ground truth to check
any automatic detector against.

Draws the 12x16 analysis grid with cell labels so a location can be named in
the same coordinates every other measurement uses.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grabette_attention.sources import DatasetSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G

DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
CAM = "observation.images.cam0"
EPISODE = 0
FRAMES = (2, 48)
ROWS, COLS = 12, 16


def main() -> None:
    source = DatasetSource(
        G.REPO, episodes=[EPISODE], camera_keys=(CAM,), task=G.TASK,
        root=G.ROOT, selection=list(FRAMES),
    )
    for obs in source.frames():
        image = obs.images[CAM]
        h, w = image.shape[:2]
        figure, ax = plt.subplots(figsize=(16, 12), dpi=110)
        ax.imshow(image, interpolation="nearest")
        ch, cw = h / ROWS, w / COLS
        for r in range(ROWS + 1):
            ax.axhline(r * ch, color="#00e5ff", linewidth=0.6, alpha=0.55)
        for c in range(COLS + 1):
            ax.axvline(c * cw, color="#00e5ff", linewidth=0.6, alpha=0.55)
        for r in range(ROWS):
            for c in range(COLS):
                ax.text(c * cw + 2, r * ch + 11, f"{r},{c}",
                        fontsize=6.5, color="#00e5ff", alpha=0.9)
        ax.set_xticks(range(0, w + 1, 40))
        ax.set_yticks(range(0, h + 1, 40))
        ax.tick_params(labelsize=8)
        ax.set_title(f"ep{EPISODE} f{obs.frame} — {w}x{h}, 12x16 grid "
                     "(cell labels are row,col)", fontsize=11)
        out = DST / f"zoom_ep{EPISODE}_f{obs.frame}.png"
        figure.savefig(out, bbox_inches="tight")
        plt.close(figure)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
