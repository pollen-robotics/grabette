"""Two questions, no new inference — reuses the 48 saved grids.

Q1. Does attention follow the OBJECT, once the fixed gripper template is
    properly cancelled?

    A permutation control does this without needing to model the template.
    For frame i with object footprint O_i, compare

        on-object:  mean attention in A_i[O_i]
        control:    mean attention in A_i[O_j] for j != i

    The control applies ANOTHER frame's object footprint to this frame. Any
    fixed spatial structure — fingers, borders, sinks — contributes equally to
    both, so a gap between them is object-following and nothing else. This
    fixes both flaws in the earlier attempt: it uses the object's whole
    footprint rather than the cap's single cell, and it never takes an argmax
    that the finger modulation can capture.

Q2. What are the bottom-left and top-right hot cells?

    Hypothesis: attention sinks / register tokens. ViTs park high-norm tokens
    carrying global state in the least informative patches, and this is a
    fisheye view whose corners are dark and featureless. Two predictions
    distinguish a sink from a scene response:
      - it sits in LOW local pixel variance (nothing there to look at)
      - it is CONTENT-INDEPENDENT, so its attention barely varies across
        frames and episodes relative to cells that respond to the scene.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from grabette_attention.sources import DatasetSource

REPO = "SteveNguyen/mustard_graspproj"
ROOT = "/home/steve/.cache/huggingface/lerobot/local-converted/mustard_graspproj"
CAM = "observation.images.cam0"
DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
NPZ = DST / "attention_grids.npz"


def object_cells(rgb, rows, cols):
    """Grid cells covered by the bottle: the red cap plus the body beneath it.

    The cap is unambiguous by colour. The body is pale against a pale table,
    so segmenting it by colour is unreliable; instead take the columns the cap
    occupies and extend downward by the cap's own height times three, which is
    roughly the bottle's aspect. Coarse, but it is a footprint rather than a
    single cell, and it is applied identically to the frame and to the control.
    """
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    cap = (r > 110) & (r > g + 45) & (r > b + 45)
    if cap.sum() < 60:
        return None
    ys, xs = np.nonzero(cap)
    h, w = rgb.shape[:2]
    ch = h / rows
    cw = w / cols
    r0 = int(np.percentile(ys, 5) // ch)
    r1 = int(np.percentile(ys, 95) // ch)
    c0 = int(np.percentile(xs, 5) // cw)
    c1 = int(np.percentile(xs, 95) // cw)
    body_rows = max(1, (r1 - r0 + 1)) * 3
    cells = {
        (rr, cc)
        for rr in range(r0, min(rows, r1 + body_rows + 1))
        for cc in range(c0, min(cols, c1 + 1))
    }
    return cells


def main() -> None:
    data = np.load(NPZ)
    grids, meta = data["grids"], data["meta"]
    n, rows, cols = grids.shape
    norm = grids.reshape(n, -1)
    norm = (norm / norm.sum(axis=1, keepdims=True)).reshape(n, rows, cols)

    # Reload the same frames (no inference) for the pixel statistics and the
    # object footprints.
    frames = {}
    for episode in sorted({int(e) for e, _ in meta}):
        wanted = [int(f) for e, f in meta if int(e) == episode]
        source = DatasetSource(
            REPO, episodes=[episode], camera_keys=(CAM,), task="t",
            root=ROOT, selection=wanted,
        )
        for obs in source.frames():
            frames[(episode, obs.frame)] = obs.images[CAM]
    print(f"{n} grids, {len(frames)} frames reloaded, grid {rows}x{cols}\n")

    footprints, keep = [], []
    for i, (e, f) in enumerate(meta):
        cells = object_cells(frames[(int(e), int(f))], rows, cols)
        if cells:
            footprints.append(cells)
            keep.append(i)
    print(f"object footprint recovered in {len(keep)}/{n} frames")
    sizes = [len(c) for c in footprints]
    print(f"footprint size: {min(sizes)}-{max(sizes)} cells "
          f"(mean {np.mean(sizes):.1f} of {rows*cols})\n")

    print("Q1. permutation control: this frame's attention on its OWN object")
    print("    footprint, versus on OTHER frames' footprints")
    own, ctrl = [], []
    rng = np.random.default_rng(0)
    for a, i in enumerate(keep):
        grid = norm[i]
        own.append(np.mean([grid[c] for c in footprints[a]]))
        others = [b for b in range(len(keep)) if b != a]
        for b in rng.choice(others, size=min(12, len(others)), replace=False):
            ctrl.append(np.mean([grid[c] for c in footprints[b]]))
    own = np.array(own)
    ctrl = np.array(ctrl)
    print(f"    on own object:    {own.mean():.5f}  (n={len(own)})")
    print(f"    on other objects: {ctrl.mean():.5f}  (n={len(ctrl)})")
    print(f"    ratio: {own.mean()/ctrl.mean():.3f}x")
    # Paired sign test: how often does a frame prefer its own footprint?
    wins = sum(
        1 for a, i in enumerate(keep)
        if own[a] > np.mean([
            np.mean([norm[i][c] for c in footprints[b]])
            for b in range(len(keep)) if b != a
        ])
    )
    print(f"    frames preferring their OWN footprint: {wins}/{len(keep)} "
          f"({wins/len(keep):.0%}; 50% is chance)")
    print("    >1.0 and well above 50% means attention genuinely follows the")
    print("    object, with all fixed structure cancelled\n")

    print("Q2. the corner cells: sink or scene response?")
    template = norm.mean(axis=0)
    variability = norm.std(axis=0) / np.maximum(template, 1e-12)
    # Local pixel variance per cell, averaged over frames.
    stds = np.zeros((rows, cols))
    for key, rgb in frames.items():
        grey = rgb.astype(np.float32).mean(axis=2)
        ch, cw = grey.shape[0] / rows, grey.shape[1] / cols
        for rr in range(rows):
            for cc in range(cols):
                patch = grey[
                    int(rr * ch):int((rr + 1) * ch), int(cc * cw):int((cc + 1) * cw)
                ]
                stds[rr, cc] += patch.std()
    stds /= len(frames)

    order = np.argsort(template.ravel())[::-1]
    print(f"    {'cell':>10} {'attention':>10} {'xUniform':>9} "
          f"{'pixel std':>10} {'rel.var':>8}")
    for idx in order[:8]:
        rr, cc = divmod(int(idx), cols)
        print(f"    r{rr:2d} c{cc:2d}  {template[rr,cc]:10.5f} "
              f"{template[rr,cc]*rows*cols:9.2f} {stds[rr,cc]:10.1f} "
              f"{variability[rr,cc]:8.2f}")
    print(f"    whole-frame mean pixel std for reference: {stds.mean():.1f}")
    print("    a sink shows HIGH attention, LOW pixel std, LOW relative")
    print("    variability; a scene response shows high pixel std or high")
    print("    variability\n")

    # Correlation across all cells: does attention prefer featureless patches?
    r = np.corrcoef(stds.ravel(), template.ravel())[0, 1]
    print(f"    corr(local pixel std, mean attention) over all {rows*cols} "
          f"cells: {r:+.3f}")
    print("    negative => attention systematically prefers FEATURELESS")
    print("    patches, which is the register/sink signature")

    figure, axes = plt.subplots(1, 3, figsize=(19, 4.2), dpi=110)
    for axis, (field, label) in zip(axes, [
        (template, "mean attention"),
        (stds, "mean local PIXEL std\n(image detail per cell)"),
        (variability, "attention variability\n(std / mean, across frames)"),
    ]):
        image = axis.imshow(field, cmap="inferno", interpolation="nearest")
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("grid col")
        axis.set_ylabel("grid row")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(
        "Cells that are bright in panel 1 but dark in panel 2 are attention "
        "sinks: high attention on featureless image", fontsize=11,
    )
    figure.tight_layout()
    path = DST / "attention_sinks_diagnosis.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
