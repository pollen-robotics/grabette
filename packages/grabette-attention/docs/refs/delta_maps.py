"""Attention AND causal maps for the DELTA model, on the same two frames.

The tables said the two action representations depend on vision equally while
the delta model attends 30% harder. That invites the obvious question: does it
attend somewhere DIFFERENT, and does the attention-vs-causality dissociation
reproduce?

Only the delta side needs computing. The chunk-relative attention grids are in
pnp_grids.npz and its occlusion maps are saved as .npy, so this loads one
checkpoint, not two -- which also keeps it inside the memory the OOM reaper
allows.

Same frames as the chunk-relative run (ep0 f2 approach, ep0 f48 grasp), same
12x16 occlusion grid, same mean fill, same first denoising step for attention.
193 forward passes per frame.
"""

import json
import sys
from pathlib import Path

import numpy as np

from grabette_attention.adapters.pi05 import Pi05Adapter
from grabette_attention.analysis import analyse_frame
from grabette_attention.loader import load_pi05
from grabette_attention.occlusion import occlusion_saliency
from grabette_attention.sources import DatasetSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G

CKPT = "chouziel/sugar_cup_grasproj_pi05"
DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
EPISODE = 0
FRAMES = {2: "approach 0%", 48: "GRASP"}
ROWS, COLS = 12, 16


def main() -> None:
    policy, pre, post = load_pi05(CKPT, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)
    camera = adapter.camera_keys[0]
    print(f"delta model, action dims "
          f"{int(policy.config.output_features['action'].shape[0])}")

    source = DatasetSource(
        G.REPO, episodes=[EPISODE], camera_keys=adapter.camera_keys,
        task=G.TASK, root=G.ROOT, selection=sorted(FRAMES),
    )
    stats = {}
    for obs in source.frames():
        label = FRAMES[obs.frame]
        attention = analyse_frame(
            adapter, obs, denoise_step="first", ablate=False
        ).cameras[camera]
        occ = occlusion_saliency(
            adapter, obs, camera=camera, rows=ROWS, cols=COLS, fill="mean"
        )
        np.save(DST / f"delta_attn_ep{EPISODE}_f{obs.frame}.npy", attention.grid)
        np.save(DST / f"delta_occl_ep{EPISODE}_f{obs.frame}.npy", occ.grid)

        a = attention.grid.ravel().astype(np.float64)
        c = occ.grid.ravel().astype(np.float64)
        pear = float(np.corrcoef(a, c)[0, 1])
        overlaps = {}
        for k in (5, 10, 20):
            ta = set(np.argsort(a)[::-1][:k])
            tc = set(np.argsort(c)[::-1][:k])
            overlaps[k] = len(ta & tc)
        stats[label] = {
            "frame": int(obs.frame),
            "mass": float(attention.mass),
            "baseline_mm": occ.baseline_mm,
            "causal_max_mm": float(c.max()),
            "causal_median_mm": float(np.median(c)),
            "pearson": pear,
            "overlap": overlaps,
        }
        print(f"\n{label} (f{obs.frame}): mass {attention.mass:.3f}   "
              f"baseline {occ.baseline_mm:.2f} mm")
        print(f"  causal max {c.max():.2f} mm  median {np.median(c):.2f} mm  "
              f"= {c.max()/max(occ.baseline_mm,1e-9):.0%} of baseline motion")
        print(f"  corr(attention, causal) = {pear:+.3f}")
        print(f"  top-k overlap: " + "  ".join(
            f"{k}:{v}/{k}" for k, v in overlaps.items()))
        for name, arr in (("attn", a), ("causal", c)):
            top = np.argsort(arr)[::-1][:4]
            cells = ", ".join(f"r{r}c{cc}" for r, cc in
                              (divmod(int(i), COLS) for i in top))
            print(f"  {name:>6} peaks: {cells}")

    (DST / "delta_maps_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nwrote {DST / 'delta_maps_stats.json'}")


if __name__ == "__main__":
    main()
