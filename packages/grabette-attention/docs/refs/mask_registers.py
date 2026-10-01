"""Does attention agree with causality once the register cells are excluded?

The top-10 disjointness in the report is driven by register tokens: they take
attention's highest cells and carry no causal weight, which destroys any
correlation. But the cube REGION is elevated in both maps (2.3-3.1x attention,
5-12x causal), so attention is not devoid of object information -- it is
swamped.

Registers live at the content boundary: the 12-row grid is a crop of the
model's 16-row patch grid, so displayed rows 0 and 11 sit against the letterbox
padding, and cols 0 and 15 at the frame edge. Excluding that border is a
principled, content-independent exclusion -- not a search for cells that happen
to improve the answer.

Reports the correlation before and after, so the effect of the exclusion is
visible rather than asserted.
"""

from pathlib import Path

import numpy as np

DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
EPISODE = 0
FRAMES = {2: "approach 0%", 48: "GRASP"}
ROWS, COLS = 12, 16


def chunkrel_attention():
    data = np.load(DST / "pnp_grids.npz")
    out = {}
    for i, (e, f) in enumerate(data["meta"]):
        if int(e) == EPISODE and int(f) in FRAMES:
            out[int(f)] = data["grids"][i]
    return out


def main() -> None:
    ck = chunkrel_attention()
    interior = np.zeros((ROWS, COLS), bool)
    interior[1:ROWS - 1, 1:COLS - 1] = True
    print(f"border excluded: {(~interior).sum()} of {ROWS*COLS} cells "
          f"(rows 0 and {ROWS-1}, cols 0 and {COLS-1})\n")

    print(f"{'frame':<13}{'model':<10}{'map pair':<22}"
          f"{'all cells':>11}{'interior':>11}")
    print("-" * 68)
    for f in sorted(FRAMES):
        for model, attn in (
            ("delta", np.load(DST / f"delta_attn_ep{EPISODE}_f{f}.npy")),
            ("chunkrel", ck[f]),
        ):
            for kind, causal in (
                ("attn vs causal RMS", np.load(
                    DST / (f"delta_occl_ep{EPISODE}_f{f}.npy" if model == "delta"
                           else f"occlusion_ep{EPISODE}_f{f}.npy"))),
                ("attn vs endpoint", np.load(
                    DST / f"endpoint_{model}_ep{EPISODE}_f{f}.npy")),
            ):
                a, c = attn.astype(np.float64), causal.astype(np.float64)
                whole = np.corrcoef(a.ravel(), c.ravel())[0, 1]
                inner = np.corrcoef(a[interior], c[interior])[0, 1]
                print(f"{FRAMES[f]:<13}{model:<10}{kind:<22}"
                      f"{whole:+11.3f}{inner:+11.3f}")
        print()

    print("a large jump means attention DOES carry causal information that the")
    print("register cells were hiding; little change means it does not")


if __name__ == "__main__":
    main()
