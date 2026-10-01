"""Figure for the delta vs chunk-relative comparison.

Three panels, each with exactly two series so identity never rests on colour
alone (legend plus direct end-labels):

  A  camera dependence as a fraction of each model's OWN commanded motion, by
     phase, against a 100% reference. The headline: the two representations
     agree, and both saturate near 100%, which is itself a limit of the
     measure.
  B  the ratio delta/chunk-relative for attention mass and for causal
     dependence. One axis, because both are ratios against a 1.0 reference --
     never two scales on one chart.
  C  depth's share of the ablation by phase. The phase migration the report
     rests on, shown to reproduce in both representations.

Reads the saved JSONs so no number is transcribed by hand.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
PHASES = ["approach 0%", "approach 35%", "approach 70%", "GRASP",
          "carry 30%", "carry 65%", "RELEASE"]
SHORT = ["appr\n0%", "appr\n35%", "appr\n70%", "GRASP", "carry\n30%",
         "carry\n65%", "RELEASE"]

# Validated categorical slots 1 and 2 (light surface #f7f8fa): all six checks
# pass, worst adjacent CVD dE 24.7 under protanopia.
CHUNK = "#2a78d6"
DELTA = "#eb6834"
SURFACE = "#f7f8fa"
INK = "#12161c"
INK_SOFT = "#4a5561"
INK_MUTED = "#8b95a1"
GRID = "#d7dde5"


def load(label):
    return json.loads((DST / f"repr_compare_{label}.json").read_text())["rows"]


def series(rows, key):
    out = []
    for phase in PHASES:
        rs = rows.get(phase, [])
        out.append(np.mean([r[key] for r in rs]) if rs else np.nan)
    return np.array(out, dtype=float)


def depth_share(rows):
    out = []
    for phase in PHASES:
        rs = rows.get(phase, [])
        if not rs:
            out.append(np.nan)
            continue
        axes = np.mean([r["axes"] for r in rs], axis=0)
        out.append(100.0 * axes[2] / max(axes.sum(), 1e-9))
    return np.array(out, dtype=float)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8.5, length=0)


def main() -> None:
    d, c = load("delta"), load("chunkrel")
    x = np.arange(len(PHASES))

    figure = plt.figure(figsize=(15.5, 4.5), dpi=115, facecolor=SURFACE)
    gs = figure.add_gridspec(1, 3, width_ratios=[1.35, 0.62, 1.15], wspace=0.28)

    # ---- A: dependence as a fraction of own motion -----------------------
    ax = figure.add_subplot(gs[0, 0])
    style(ax)
    ax.axhline(100, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("100% — output as different as the motion itself",
                xy=(0.985, 100), xycoords=("axes fraction", "data"),
                xytext=(0, -13), textcoords="offset points", ha="right",
                fontsize=7.5, color=INK_MUTED)
    for rows, colour, name in ((c, CHUNK, "chunk-relative"), (d, DELTA, "delta")):
        y = 100.0 * series(rows, "fraction")
        ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name, zorder=3)
    ax.set_xticks(x, SHORT)
    ax.set_xlim(-0.45, len(PHASES) - 0.55)
    ax.set_ylim(80, 118)
    ax.set_ylabel("camera ablation, % of own commanded motion",
                  color=INK_SOFT, fontsize=8.5)
    ax.set_title("A · Vision dependence is the same in both",
                 color=INK, fontsize=10.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT,
              loc="lower left", handlelength=1.6)

    # ---- B: attention vs causality, as ratios ---------------------------
    ax = figure.add_subplot(gs[0, 1])
    style(ax)
    mass = np.mean([r["mass"] for rs in d.values() for r in rs]) / \
        np.mean([r["mass"] for rs in c.values() for r in rs])
    causal = np.nanmean(series(d, "fraction")) / np.nanmean(series(c, "fraction"))
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    bars = ax.bar([0, 1], [mass, causal], width=0.5,
                  color=[DELTA, DELTA], zorder=3)
    bars[1].set_color(CHUNK)
    for xi, value in zip((0, 1), (mass, causal)):
        ax.annotate(f"{value:.2f}×", xy=(xi, value), xytext=(0, 6),
                    textcoords="offset points", ha="center",
                    color=INK, fontsize=10, fontweight="semibold")
    ax.set_xticks([0, 1], ["attention\nmass", "causal\ndependence"])
    ax.set_ylim(0, 1.55)
    ax.set_ylabel("delta ÷ chunk-relative", color=INK_SOFT, fontsize=8.5)
    ax.set_title("B · Attention moved 30%.\nCausality did not.",
                 color=INK, fontsize=10.5, loc="left", pad=10)

    # ---- C: depth share reproduces --------------------------------------
    ax = figure.add_subplot(gs[0, 2])
    style(ax)
    for rows, colour, name in ((c, CHUNK, "chunk-relative"), (d, DELTA, "delta")):
        y = depth_share(rows)
        ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name, zorder=3)
    ax.annotate("range matters on approach", xy=(0.6, 56), xytext=(0, 12),
                textcoords="offset points", fontsize=7.5, color=INK_MUTED)
    ax.annotate("alignment at contact", xy=(3, 29), xytext=(-4, -17),
                textcoords="offset points", ha="center",
                fontsize=7.5, color=INK_MUTED)
    ax.set_xticks(x, SHORT)
    ax.set_xlim(-0.45, len(PHASES) - 0.55)
    ax.set_ylim(18, 72)
    ax.set_ylabel("depth's share of the ablation (%)",
                  color=INK_SOFT, fontsize=8.5)
    ax.set_title("C · The phase migration reproduces",
                 color=INK, fontsize=10.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SOFT,
              loc="upper right", handlelength=1.6)

    figure.suptitle(
        "Action representation changes the attention, not the dependence — "
        "sugar-cube task, same recording, same 20 000 steps, same prompt",
        color=INK, fontsize=11.5, x=0.5, y=1.04,
    )
    out = DST / "repr_delta_vs_chunkrel.png"
    figure.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {out}")
    print(f"  attention mass ratio {mass:.3f}   causal ratio {causal:.3f}")
    print(f"  delta depth share:    {np.array2string(depth_share(d), precision=0)}")
    print(f"  chunkrel depth share: {np.array2string(depth_share(c), precision=0)}")


if __name__ == "__main__":
    main()
