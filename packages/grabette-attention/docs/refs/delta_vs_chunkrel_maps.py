"""Side-by-side maps: delta vs chunk-relative, same two frames.

Two rows (approach, grasp) x five columns:

    frame | delta ATTENTION | chunkrel ATTENTION | delta CAUSAL | chunkrel CAUSAL

Attention panels are each clipped at their own 99th percentile, because the two
models put different total mass on the image and the question here is WHERE it
goes, not how much. Stated in the caption so nobody reads brightness across
them.

Causal panels are normalised by each model's own baseline commanded motion and
share one scale, so they ARE directly comparable: a cell reads as "covering
this block changes the command by X% of what the model was about to do". That
is the same normalisation the fraction tables use, applied per cell.

Chunk-relative baselines for these two frames come from the prompt sweep's
per-frame output (61.4 mm at f2, 94.0 mm at f48); the noise draw is
deterministic per seed, so those transfer exactly.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grabette_attention.sources import DatasetSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G

DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
CAM = "observation.images.cam0"
EPISODE = 0
FRAMES = {2: "approach 0%", 48: "GRASP"}
CHUNKREL_BASELINE_MM = {2: 61.4, 48: 94.0}
ROWS, COLS = 12, 16
INK, INK_SOFT, SURFACE = "#12161c", "#4a5561", "#f7f8fa"


def chunkrel_attention():
    data = np.load(DST / "pnp_grids.npz")
    grids, meta = data["grids"], data["meta"]
    out = {}
    for i, (e, f) in enumerate(meta):
        if int(e) == EPISODE and int(f) in FRAMES:
            out[int(f)] = grids[i]
    return out


def main() -> None:
    ck_attn = chunkrel_attention()
    delta_stats = json.loads((DST / "delta_maps_stats.json").read_text())
    by_frame = {v["frame"]: v for v in delta_stats.values()}

    source = DatasetSource(
        G.REPO, episodes=[EPISODE], camera_keys=(CAM,), task=G.TASK,
        root=G.ROOT, selection=sorted(FRAMES),
    )
    frames = {obs.frame: obs.images[CAM] for obs in source.frames()}

    # Shared scale for the two causal columns, in % of own baseline motion.
    causal = {}
    for f in FRAMES:
        causal[("delta", f)] = (
            np.load(DST / f"delta_occl_ep{EPISODE}_f{f}.npy")
            / max(by_frame[f]["baseline_mm"], 1e-9) * 100.0)
        causal[("chunkrel", f)] = (
            np.load(DST / f"occlusion_ep{EPISODE}_f{f}.npy")
            / CHUNKREL_BASELINE_MM[f] * 100.0)
    cmax = max(v.max() for v in causal.values())
    print(f"shared causal scale: 0 to {cmax:.1f}% of baseline motion per cell")

    figure, axes = plt.subplots(2, 5, figsize=(21, 7.2), dpi=105,
                                facecolor=SURFACE)
    heads = ["frame", "ATTENTION · delta", "ATTENTION · chunk-relative",
             "CAUSAL · delta", "CAUSAL · chunk-relative"]

    for row, f in enumerate(sorted(FRAMES)):
        image = frames[f]
        extent = (0, image.shape[1], image.shape[0], 0)
        panels = [
            None,
            (np.load(DST / f"delta_attn_ep{EPISODE}_f{f}.npy"), "inferno", None),
            (ck_attn[f], "inferno", None),
            (causal[("delta", f)], "viridis", cmax),
            (causal[("chunkrel", f)], "viridis", cmax),
        ]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.set_axis_off()
            ax.imshow(image)
            if panel is None:
                if row == 0:
                    ax.set_title(heads[col], fontsize=10.5, color=INK,
                                 loc="left", pad=8)
                ax.set_ylabel(FRAMES[f])
                ax.text(0.02, 0.04, f"ep{EPISODE} f{f} — {FRAMES[f]}",
                        transform=ax.transAxes, fontsize=9, color="white",
                        bbox=dict(facecolor=INK, edgecolor="none",
                                  boxstyle="round,pad=0.32", alpha=0.85))
                continue
            grid, cmap, vmax = panel
            kw = {"vmin": 0.0, "vmax": vmax} if vmax else {
                "vmin": float(grid.min()),
                "vmax": float(np.percentile(grid, 99)),
            }
            handle = ax.imshow(grid, extent=extent, interpolation="bilinear",
                               alpha=0.6, cmap=cmap, **kw)
            if row == 0:
                ax.set_title(heads[col], fontsize=10.5, color=INK,
                             loc="left", pad=8)
            if vmax and col == 4:
                bar = figure.colorbar(handle, ax=axes[row, 3:5].tolist(),
                                      fraction=0.028, pad=0.012)
                bar.set_label("% of baseline motion per cell",
                              fontsize=8, color=INK_SOFT)
                bar.ax.tick_params(labelsize=7.5, colors=INK_SOFT)

    figure.suptitle(
        "Same episode, same frames, two action representations — where each "
        "model LOOKS and where its motion COMES FROM\n"
        "attention clipped per panel (compare position, not brightness); "
        "causal panels share one scale in % of each model's own commanded motion",
        fontsize=12, color=INK, y=1.02,
    )
    out = DST / "delta_vs_chunkrel_maps.png"
    figure.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {out}")

    print(f"\n{'frame':<14}{'model':<11}{'attn peak':>12}{'causal peak':>13}"
          f"{'corr':>8}{'top10':>7}")
    for f in sorted(FRAMES):
        for model, attn in (("delta",
                             np.load(DST / f"delta_attn_ep{EPISODE}_f{f}.npy")),
                            ("chunkrel", ck_attn[f])):
            c = causal[(model, f)]
            a = attn.ravel().astype(np.float64)
            cr = c.ravel().astype(np.float64)
            ap = divmod(int(np.argmax(a)), COLS)
            cp = divmod(int(np.argmax(cr)), COLS)
            corr = np.corrcoef(a, cr)[0, 1]
            overlap = len(set(np.argsort(a)[::-1][:10]) &
                          set(np.argsort(cr)[::-1][:10]))
            print(f"{FRAMES[f]:<14}{model:<11}"
                  f"{'r%dc%d' % ap:>12}{'r%dc%d' % cp:>13}"
                  f"{corr:+8.3f}{overlap:>5}/10")


if __name__ == "__main__":
    main()
