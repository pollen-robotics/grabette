"""The specific cells Steve asked about, plus why they sit where they do.

The 12-row grid is a CROP of the model's 16x16 patch grid: a 720x960 frame
letterboxes to 168x224 inside 224x224, so 2 patch rows of pure padding are
removed top and bottom and content_rows == (2, 14). Displayed row 0 is
therefore the FIRST content row and row 11 the LAST -- both immediately
adjacent to the padding. If sinks prefer the content boundary, they should
concentrate in those two rows.
"""

import numpy as np
from pathlib import Path

from grabette_attention.sources import DatasetSource

REPO = "SteveNguyen/mustard_graspproj"
ROOT = "/home/steve/.cache/huggingface/lerobot/local-converted/mustard_graspproj"
CAM = "observation.images.cam0"
NPZ = Path("/home/steve/attention_out/attention_grids.npz")
WATCH = [(0, 15), (0, 14), (0, 0), (10, 1), (11, 4), (11, 14), (9, 11), (5, 7)]


def main() -> None:
    data = np.load(NPZ)
    grids, meta = data["grids"], data["meta"]
    n, rows, cols = grids.shape
    norm = grids.reshape(n, -1)
    norm = (norm / norm.sum(axis=1, keepdims=True)).reshape(n, rows, cols)

    stds = np.zeros((rows, cols))
    count = 0
    for episode in sorted({int(e) for e, _ in meta}):
        wanted = [int(f) for e, f in meta if int(e) == episode]
        source = DatasetSource(
            REPO, episodes=[episode], camera_keys=(CAM,), task="t",
            root=ROOT, selection=wanted,
        )
        for obs in source.frames():
            grey = obs.images[CAM].astype(np.float32).mean(axis=2)
            ch, cw = grey.shape[0] / rows, grey.shape[1] / cols
            for rr in range(rows):
                for cc in range(cols):
                    stds[rr, cc] += grey[
                        int(rr * ch):int((rr + 1) * ch),
                        int(cc * cw):int((cc + 1) * cw),
                    ].std()
            count += 1
    stds /= count

    template = norm.mean(axis=0)
    relvar = norm.std(axis=0) / np.maximum(template, 1e-12)
    uniform = 1.0 / (rows * cols)

    print(f"{count} frames; uniform share per cell = {uniform:.5f}; "
          f"frame mean pixel std = {stds.mean():.1f}\n")
    print(f"{'cell':>10} {'xUniform':>9} {'pixel std':>10} {'rel.var':>8}  verdict")
    for rr, cc in WATCH:
        x = template[rr, cc] / uniform
        detail = stds[rr, cc]
        v = relvar[rr, cc]
        if x > 2 and detail < 0.6 * stds.mean() and v < 0.3:
            verdict = "SINK (high attn, no detail, unchanging)"
        elif x > 2:
            verdict = "high attention, content-responsive"
        else:
            verdict = "ordinary"
        print(f"    r{rr:2d} c{cc:2d} {x:9.2f} {detail:10.1f} {v:8.2f}  {verdict}")

    print("\nby grid row -- row 0 and row 11 are the content-boundary rows,")
    print("immediately adjacent to the letterbox padding that was cropped:")
    print(f"{'row':>5} {'mean xUniform':>14} {'mean pixel std':>15} {'mean rel.var':>13}")
    for rr in range(rows):
        edge = "  <-- boundary" if rr in (0, rows - 1) else ""
        print(f"{rr:5d} {template[rr].mean()/uniform:14.2f} "
              f"{stds[rr].mean():15.1f} {relvar[rr].mean():13.2f}{edge}")


if __name__ == "__main__":
    main()
