"""Occlusion measured as ENDPOINT DIVERGENCE, for both representations.

Replaces the normalisation the previous comparison depended on. The 7x
"concentration" difference came out of dividing each model's per-block effect
by its own commanded motion -- and in raw millimetres the ranking reverses,
because the delta model's baseline is 3.5 mm and the chunk-relative model's is
61 mm. Both readings were defensible, which means neither was decisive.

Endpoint divergence embeds no such choice: it is how far apart the two
commanded trajectories END, in millimetres, after composing the delta model's
per-step rotations. One physical distance, directly comparable.

One model per process -- an fp32 pi0.5 is ~19 GB resident and two at once gets
killed. Run as:  endpoint_run.py delta   then   endpoint_run.py chunkrel
"""

import json
import sys
from pathlib import Path

import numpy as np

from grabette_attention.adapters.pi05 import Pi05Adapter
from grabette_attention.loader import load_pi05
from grabette_attention.metrics import OFFSETS, PER_STEP_DELTAS, chunk_endpoint_m
from grabette_attention.occlusion import occlusion_saliency
from grabette_attention.sources import DatasetSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G

MODELS = {
    "delta": ("chouziel/sugar_cup_grasproj_pi05", PER_STEP_DELTAS),
    "chunkrel": ("SteveNguyen/sugar_cup_chunkrel_pi05_step20000", OFFSETS),
}
DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
EPISODE = 0
FRAMES = {2: "approach 0%", 48: "GRASP"}
ROWS, COLS = 12, 16


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which not in MODELS:
        raise SystemExit(f"usage: {sys.argv[0]} [{'|'.join(MODELS)}]")
    checkpoint, representation = MODELS[which]
    print(f"{which}: {checkpoint}\nrepresentation: {representation}")

    policy, pre, post = load_pi05(checkpoint, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)
    camera = adapter.camera_keys[0]

    source = DatasetSource(
        G.REPO, episodes=[EPISODE], camera_keys=adapter.camera_keys,
        task=G.TASK, root=G.ROOT, selection=sorted(FRAMES),
    )
    out = {}
    for obs in source.frames():
        label = FRAMES[obs.frame]
        noise = adapter.draw_noise()
        baseline = adapter.run(obs, noise, capture=False)
        travel = chunk_endpoint_m(baseline.chunk, representation=representation)
        travel_mm = float(np.linalg.norm(travel)) * 1000.0

        occ = occlusion_saliency(
            adapter, obs, camera=camera, rows=ROWS, cols=COLS, fill="mean",
            metric="endpoint", representation=representation, noise=noise,
        )
        np.save(DST / f"endpoint_{which}_ep{EPISODE}_f{obs.frame}.npy", occ.grid)
        flat = occ.grid.ravel().astype(np.float64)
        peak = divmod(int(np.argmax(flat)), COLS)
        out[label] = {
            "frame": int(obs.frame),
            "travel_mm": travel_mm,
            "peak_mm": float(flat.max()),
            "median_mm": float(np.median(flat)),
            "peak_cell": list(peak),
            "peak_frac_of_travel": float(flat.max()) / max(travel_mm, 1e-9),
        }
        print(f"\n{label} (f{obs.frame})")
        print(f"  commanded trajectory travels {travel_mm:8.2f} mm to its endpoint")
        print(f"  worst single block moves that endpoint by {flat.max():8.2f} mm "
              f"(r{peak[0]}c{peak[1]})")
        print(f"  median block                              {np.median(flat):8.2f} mm")
        print(f"  worst block as a share of the travel      "
              f"{flat.max()/max(travel_mm,1e-9):8.1%}")

    path = DST / f"endpoint_{which}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
