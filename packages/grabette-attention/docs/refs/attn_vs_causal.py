"""Put a number on the attention-vs-causality agreement.

The figure shows the two maps disagreeing. This measures it: Pearson and
Spearman correlation between the attention grid and the occlusion grid over
all 192 cells, plus whether the causal peak is anywhere near the attention
peak, plus how concentrated the causal map is.

Recomputes the attention grids (one capture pass each) and reads the
occlusion grids from the .npy files the sweep saved, so no occlusion is
re-run.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from grabette_attention.adapters.pi05 import Pi05Adapter
from grabette_attention.analysis import analyse_frame
from grabette_attention.loader import load_pi05
from grabette_attention.sources import DatasetSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G

DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
EPISODE = 0
FRAMES = {2: "approach 0%", 48: "GRASP"}


def main() -> None:
    policy, pre, post = load_pi05(G.CKPT, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)
    camera = adapter.camera_keys[0]

    source = DatasetSource(
        G.REPO, episodes=[EPISODE], camera_keys=adapter.camera_keys,
        task=G.TASK, root=G.ROOT, selection=sorted(FRAMES),
    )
    for obs in source.frames():
        label = FRAMES[obs.frame]
        attn = analyse_frame(
            adapter, obs, denoise_step="first", ablate=False
        ).cameras[camera].grid
        causal = np.load(DST / f"occlusion_ep{EPISODE}_f{obs.frame}.npy")
        assert attn.shape == causal.shape, (attn.shape, causal.shape)
        rows, cols = attn.shape

        a, c = attn.ravel().astype(np.float64), causal.ravel().astype(np.float64)
        pear = np.corrcoef(a, c)[0, 1]
        spear = spearmanr(a, c).statistic

        print(f"\n=== ep{EPISODE} f{obs.frame} — {label} ===")
        print(f"grid {rows}x{cols} = {a.size} cells")
        print(f"  Pearson  corr(attention, causal) = {pear:+.3f}")
        print(f"  Spearman corr(attention, causal) = {spear:+.3f}")

        # Does the top of one map appear in the top of the other?
        for k in (5, 10, 20):
            ta = set(np.argsort(a)[::-1][:k])
            tc = set(np.argsort(c)[::-1][:k])
            overlap = len(ta & tc)
            expected = k * k / a.size
            print(f"  top-{k:<2} overlap: {overlap}/{k} cells "
                  f"(chance ~{expected:.1f})")

        # How concentrated is the causal map? A distributed representation
        # shows a small peak share; a critical region shows a large one.
        order = np.argsort(c)[::-1]
        print(f"  causal: max {c.max():.2f} mm, median {np.median(c):.2f} mm, "
              f"max/median {c.max()/max(np.median(c),1e-9):.1f}x")
        print(f"  causal mass in top 5 cells: {c[order[:5]].sum()/c.sum():.1%} "
              f"(uniform would be {5/c.size:.1%})")
        print(f"  attention mass in top 5 cells: "
              f"{a[np.argsort(a)[::-1][:5]].sum()/a.sum():.1%}")


if __name__ == "__main__":
    main()
