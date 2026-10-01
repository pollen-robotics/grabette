"""Attention across several episodes AND across time, in one grid.

Why this shape. At the grasp the gripper fingers surround the bottle, so
"attention on the object" and "attention on the fingers" are nearly the same
pixels and the map cannot distinguish them. Early in an episode the fingers are
far from the object, so whichever one the attention follows is unambiguous.

Rows are episodes, columns run from the start of the episode to the grasp at
fixed fractions of the approach, which makes episodes of different lengths and
speeds directly comparable.

Uses the FIRST denoising step, not the default last: the register/sink cell
grows monotonically with denoising, so step 0 is the least contaminated view of
the content. Colour limits come from the package's own clipping helper so these
panels match what the CLI writes.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

from grabette_attention.adapters.pi05 import Pi05Adapter
from grabette_attention.analysis import analyse_frame
from grabette_attention.frontends.png import _colour_limits
from grabette_attention.loader import load_pi05
from grabette_attention.sources import DatasetSource

CKPT = "SteveNguyen/pick3_graspproj_chunkrel_pi05"
REPO = "SteveNguyen/mustard_graspproj"
ROOT = "/home/steve/.cache/huggingface/lerobot/local-converted/mustard_graspproj"
PARQUET = f"{ROOT}/data/chunk-000/file-000.parquet"
TASK = "pick up the mustard bottle"
CLOSURE, DZ = 10, 2
DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")

# Fraction of the way from the start of the episode to the grasp.
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
EPISODES = (0, 2, 4, 6, 8, 13)


def plans():
    tbl = pq.read_table(PARQUET, columns=["episode_index", "action"])
    ep = np.asarray(tbl["episode_index"])
    act = np.stack([np.asarray(r) for r in tbl["action"].to_pylist()]).astype(np.float64)
    out = {}
    for e in EPISODES:
        rows = act[ep == e]
        closure = rows[:, CLOSURE]
        hot = np.nonzero(closure > 0.5 * max(closure.max(), 1e-6))[0]
        if hot.size < 5:
            print(f"episode {e}: no clear closure, skipped")
            continue
        grasp = int(hot[0])
        remaining = np.concatenate(
            [np.cumsum(rows[:grasp, DZ][::-1])[::-1], np.zeros(len(rows) - grasp)]
        )
        # Frame 0 can be a partial/settling frame; start a couple in.
        frames = [max(2, int(round(f * grasp))) for f in FRACTIONS]
        out[e] = {"grasp": grasp, "frames": frames, "remaining": remaining}
    return out


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    plan = plans()
    total = sum(len(p["frames"]) for p in plan.values())
    print(f"episodes {sorted(plan)}, {total} frames (~{total*11/60:.0f} min)")

    policy, pre, post = load_pi05(CKPT, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)
    camera = adapter.camera_keys[0]

    rows = sorted(plan)
    figure, axes = plt.subplots(
        len(rows), len(FRACTIONS), figsize=(4.2 * len(FRACTIONS), 3.3 * len(rows)), dpi=95
    )
    done = 0

    for r, episode in enumerate(rows):
        p = plan[episode]
        source = DatasetSource(
            REPO, episodes=[episode], camera_keys=adapter.camera_keys,
            task=TASK, root=ROOT, selection=p["frames"],
        )
        by_frame = {obs.frame: obs for obs in source.frames()}
        for c, want in enumerate(p["frames"]):
            axis = axes[r, c]
            axis.set_axis_off()
            obs = by_frame.get(want)
            if obs is None:
                continue
            result = analyse_frame(adapter, obs, denoise_step="first", ablate=True)
            attention = result.cameras[camera]
            frame = obs.images[camera]
            vmin, vmax = _colour_limits(attention.grid)
            axis.imshow(frame)
            axis.imshow(
                attention.grid,
                extent=(0, frame.shape[1], frame.shape[0], 0),
                interpolation="bilinear", alpha=0.55, cmap="inferno",
                vmin=vmin, vmax=vmax,
            )
            rem = p["remaining"][min(obs.frame, p["grasp"])] * 1000
            abl = result.ablations[camera]
            axis.set_title(
                f"ep{episode}  f{obs.frame}  {int(round(FRACTIONS[c]*100))}% to grasp\n"
                f"{rem:.0f} mm to go   mass {attention.mass:.2f}\n"
                f"ablate {abl.delta_mm:.0f} mm  "
                f"(lat {abl.per_axis_mm[0]:.0f}, vert {abl.per_axis_mm[1]:.0f}, "
                f"depth {abl.per_axis_mm[2]:.0f})",
                fontsize=7.5,
            )
            done += 1
            print(f"[{done}/{total}] ep{episode} f{obs.frame} rem {rem:6.1f} mm "
                  f"mass {attention.mass:.2f} delta {abl.delta_mm:5.1f}", flush=True)

    figure.suptitle(
        "Attention across episodes and across time — first denoising step, p99 clip\n"
        "left column = start of episode (fingers far from the object); "
        "right column = the grasp (fingers surrounding it)",
        fontsize=12,
    )
    figure.tight_layout()
    path = DST / "attention_episodes_over_time.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
