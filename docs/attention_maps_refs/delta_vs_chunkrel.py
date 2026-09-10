"""Delta vs chunk-relative: does the action representation change what the
policy depends on?

Both checkpoints were trained for 20000 steps on the SAME recording with the
SAME prompt; only the action parameterisation differs:

  delta     chouziel/sugar_cup_grasproj_pi05          11-dim per-step deltas
  chunkrel  SteveNguyen/sugar_cup_chunkrel_pi05_...    8-dim cumulative offsets

Their raw millimetres are NOT comparable. The delta model's chunk holds
per-step motion (~3 mm scale); the chunk-relative model's holds cumulative
offsets over 50 steps (~60-100 mm). The same intervention therefore reads
20-50x larger on the latter purely from parameterisation. Verified from the
checkpoints' own processor configs: the delta model's relative_actions and
absolute_actions steps are both enabled=false, so its output is the dataset's
raw per-step representation.

So everything here is reported as a FRACTION of each model's own commanded
motion, which asks both the same question: what proportion of what you were
about to do changes when the camera goes away? Raw millimetres are printed
alongside, clearly separated, so nobody reads them across the two columns.

Frames come from the chunk-relative dataset copy for both models -- the two
copies are the same 150 episodes with identical per-episode lengths (verified),
so the images and the phase timing are identical.
"""

import json
import sys
from pathlib import Path

import numpy as np

from grabette_attention.adapters.pi05 import Pi05Adapter
from grabette_attention.analysis import analyse_frame
from grabette_attention.loader import load_pi05
from grabette_attention.sources import DatasetSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G

MODELS = {
    "delta": "chouziel/sugar_cup_grasproj_pi05",
    "chunkrel": "SteveNguyen/sugar_cup_chunkrel_pi05_step20000",
}
EPISODES = (0, 1, 2, 3, 4, 5)
DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")


def run(label, checkpoint, plan):
    print(f"\n########## {label}: {checkpoint}")
    policy, pre, post = load_pi05(checkpoint, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)
    camera = adapter.camera_keys[0]
    dims = int(policy.config.output_features["action"].shape[0])
    print(f"action dims: {dims}")

    rows = {}
    for episode in EPISODES:
        p = plan[episode]
        source = DatasetSource(
            G.REPO, episodes=[episode], camera_keys=adapter.camera_keys,
            task=G.TASK, root=G.ROOT, selection=sorted(set(p["frames"])),
        )
        by_frame = {obs.frame: obs for obs in source.frames()}
        for want, phase in zip(p["frames"], p["labels"]):
            obs = by_frame.get(want)
            if obs is None:
                continue
            result = analyse_frame(adapter, obs, denoise_step="first", ablate=True)
            abl = result.ablations[camera]
            frac = abl.delta_mm / max(result.baseline_mm, 1e-9)
            rows.setdefault(phase, []).append({
                "delta_mm": abl.delta_mm,
                "baseline_mm": result.baseline_mm,
                "fraction": frac,
                "mass": result.cameras[camera].mass,
                "axes": abl.per_axis_mm,
            })
            print(f"  ep{episode} f{obs.frame:4d} {phase:14s} "
                  f"baseline {result.baseline_mm:7.1f} mm  "
                  f"ablate {abl.delta_mm:7.1f} mm  "
                  f"= {frac:5.1%} of motion   mass {result.cameras[camera].mass:.2f}",
                  flush=True)
    return rows, dims


def main() -> None:
    plan = G.plans()
    labels = list(plan[EPISODES[0]]["labels"])
    DST.mkdir(parents=True, exist_ok=True)

    # ONE MODEL PER PROCESS. An fp32 pi0.5 is ~18 GB resident and this machine
    # has ~30 GB free, so loading both in one process is killed by the OOM
    # reaper partway through the second. Each run saves its own rows; a final
    # 'compare' invocation reads them back and prints the tables.
    which = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if which in MODELS:
        rows, dims = run(which, MODELS[which], plan)
        path = DST / f"repr_compare_{which}.json"
        path.write_text(json.dumps({"dims": dims, "rows": rows}))
        print(f"\nwrote {path}")
        return
    if which != "compare":
        raise SystemExit(f"usage: {sys.argv[0]} [{'|'.join(MODELS)}|compare]")

    out = {}
    for label in MODELS:
        path = DST / f"repr_compare_{label}.json"
        if not path.exists():
            raise SystemExit(f"missing {path} — run '{label}' first")
        doc = json.loads(path.read_text())
        out[label] = doc["rows"]
        out[label + "_dims"] = doc["dims"]

    print("\n\n" + "=" * 86)
    print("CAMERA ABLATION AS A FRACTION OF EACH MODEL'S OWN COMMANDED MOTION")
    print("(the only cross-representation comparison that means anything)")
    print("=" * 86)
    print(f"{'phase':<14}{'delta':>12}{'chunkrel':>12}{'ratio':>9}   "
          f"{'delta mm':>10}{'chunkrel mm':>13}")
    for phase in labels:
        d = out["delta"].get(phase, [])
        c = out["chunkrel"].get(phase, [])
        if not d or not c:
            continue
        df = np.mean([r["fraction"] for r in d])
        cf = np.mean([r["fraction"] for r in c])
        print(f"{phase:<14}{df:11.1%}{cf:12.1%}{df/max(cf,1e-9):8.2f}x   "
              f"{np.mean([r['delta_mm'] for r in d]):10.1f}"
              f"{np.mean([r['delta_mm'] for r in c]):13.1f}")

    print(f"\n{'':<14}{'delta':>12}{'chunkrel':>12}")
    for name, key in (("mean baseline motion (mm)", "baseline_mm"),
                      ("mean camera attention mass", "mass")):
        d = np.mean([r[key] for rs in out["delta"].values() for r in rs])
        c = np.mean([r[key] for rs in out["chunkrel"].values() for r in rs])
        fmt = "12.1f" if "mm" in name else "12.3f"
        print(f"{name:<14}" + f"{d:{fmt}}{c:{fmt}}")

    print("\nPER-AXIS SHARE OF THE ABLATION (fraction of the 3-axis total)")
    print(f"{'phase':<14}" + "".join(
        f"{m + ' ' + a:>14}" for m in ("delta", "chunkrel")
        for a in ("lat", "vert", "depth")))
    for phase in labels:
        cells = []
        for model in ("delta", "chunkrel"):
            rs = out[model].get(phase, [])
            if not rs:
                cells += ["-"] * 3
                continue
            axes = np.mean([r["axes"] for r in rs], axis=0)
            total = max(axes.sum(), 1e-9)
            cells += [f"{a/total:.0%}" for a in axes]
        print(f"{phase:<14}" + "".join(f"{c:>14}" for c in cells))

    print(f"\naction dims — delta {out['delta_dims']}, "
          f"chunkrel {out['chunkrel_dims']}")


if __name__ == "__main__":
    main()
