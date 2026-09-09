"""Attention across episodes and across time, for any of pick3's objects.

Generalises the mustard-only version so the object is a parameter: same
checkpoint, same timings, same reduction, only the dataset and the prompt
change. That is what makes a mustard-vs-sugar comparison meaningful.

Run as:  python object_grid.py sugar
         python object_grid.py mustard

Two details that matter and are easy to get wrong:

1. THE PROMPT MUST BE THE ONE THE CHECKPOINT TRAINED ON. pick3's three tasks
   are "pick up the red can", "pick up the mustard bottle", "pick up the cup".
   The sugar dataset's own task string is "pick up the sugar cup", which the
   checkpoint never saw; using it would put the language conditioning off
   distribution and confound the comparison.

2. For sugar, the 11-dim per-step actions and the videos live in two different
   copies of the same recording (identical 150 episodes, identical per-episode
   lengths, verified). Actions come from the graspproj copy so the grasp frame
   and the remaining-forward-travel annotation are computed exactly as they
   were for mustard; frames come from the chunkrel copy, which is the one whose
   videos are downloaded.

Uses the FIRST denoising step: the register/sink cell grows monotonically with
denoising, so step 0 is the least contaminated view of the content.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from grabette_attention.adapters.pi05 import Pi05Adapter
from grabette_attention.analysis import analyse_frame
from grabette_attention.frontends.png import _colour_limits
from grabette_attention.loader import load_pi05
from grabette_attention.sources import DatasetSource

CKPT = "SteveNguyen/pick3_graspproj_chunkrel_pi05"
DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
HUB = "/home/steve/.cache/huggingface/hub"

CONFIG = {
    "mustard": {
        "repo": "SteveNguyen/mustard_graspproj",
        "root": "/home/steve/.cache/huggingface/lerobot/local-converted/mustard_graspproj",
        "actions": (
            "/home/steve/.cache/huggingface/lerobot/local-converted/"
            "mustard_graspproj/data/chunk-000/file-000.parquet"
        ),
        "task": "pick up the mustard bottle",
        "label": "mustard bottle",
    },
    "sugar": {
        "repo": "SteveNguyen/sugar_cup_graspproj_chunkrel",
        "root": (
            f"{HUB}/datasets--SteveNguyen--sugar_cup_graspproj_chunkrel/snapshots/"
            "05e70fd531b3f2fd342dca427cc6baf7272dddb8"
        ),
        "actions": (
            f"{HUB}/datasets--chouziel--grabette-sugar-cup-2008_graspproj/snapshots/"
            "96f1794700bccb9c7088100caed3c974aac82ad9/data/chunk-000/file-000.parquet"
        ),
        # The prompt pick3 was trained with, NOT the dataset's own task string.
        "task": "pick up the cup",
        "label": "sugar cup",
    },
}

CLOSURE, DZ = 10, 2      # channels of the 11-dim per-step action
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
# Candidates, not a fixed list: an episode whose gripper is already closed at
# the start has no approach to sample, and every fraction rounds to the same
# frame. Take the first N that actually qualify.
CANDIDATES = tuple(range(0, 40))
N_EPISODES = 6
# The grasp must be late enough that the fractions give distinct frames.
MIN_GRASP = 4 * len(FRACTIONS)


def plans(actions_parquet: str):
    tbl = pq.read_table(actions_parquet, columns=["episode_index", "action"])
    ep = np.asarray(tbl["episode_index"])
    act = np.stack([np.asarray(r) for r in tbl["action"].to_pylist()]).astype(np.float64)
    if act.shape[1] <= CLOSURE:
        raise ValueError(
            f"expected an 11-dim per-step action, got {act.shape[1]} dims: "
            "this script needs the graspproj copy, not the chunk-relative one"
        )
    out = {}
    for e in CANDIDATES:
        if len(out) >= N_EPISODES:
            break
        rows = act[ep == e]
        if not len(rows):
            continue
        closure = rows[:, CLOSURE]
        hot = np.nonzero(closure > 0.5 * max(closure.max(), 1e-6))[0]
        if hot.size < 5:
            print(f"episode {e}: no clear gripper closure, skipped")
            continue
        grasp = int(hot[0])
        if grasp < MIN_GRASP:
            # Already closed at the start: there is no approach to sample, and
            # every fraction would round to the same frame, silently filling a
            # row of the figure with five copies of one panel.
            print(f"episode {e}: grasp at frame {grasp} < {MIN_GRASP}, "
                  "no approach to sample, skipped")
            continue
        frames = sorted({max(2, int(round(f * grasp))) for f in FRACTIONS})
        if len(frames) < len(FRACTIONS):
            print(f"episode {e}: grasp at {grasp} gives only {len(frames)} "
                  "distinct timings, skipped")
            continue
        remaining = np.concatenate(
            [np.cumsum(rows[:grasp, DZ][::-1])[::-1], np.zeros(len(rows) - grasp)]
        )
        out[e] = {"grasp": grasp, "frames": frames, "remaining": remaining}
    return out


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "sugar"
    if which not in CONFIG:
        raise SystemExit(f"unknown object {which!r}; pick one of {sorted(CONFIG)}")
    cfg = CONFIG[which]
    DST.mkdir(parents=True, exist_ok=True)

    plan = plans(cfg["actions"])
    total = sum(len(p["frames"]) for p in plan.values())
    print(f"{cfg['label']}: episodes {sorted(plan)}, {total} frames "
          f"(~{total*11/60:.0f} min)\nprompt: {cfg['task']!r}\n")

    policy, pre, post = load_pi05(CKPT, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)
    camera = adapter.camera_keys[0]

    rows = sorted(plan)
    figure, axes = plt.subplots(
        len(rows), len(FRACTIONS),
        figsize=(4.2 * len(FRACTIONS), 3.3 * len(rows)), dpi=95,
    )
    done = 0
    for r, episode in enumerate(rows):
        p = plan[episode]
        source = DatasetSource(
            cfg["repo"], episodes=[episode], camera_keys=adapter.camera_keys,
            task=cfg["task"], root=cfg["root"], selection=p["frames"],
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
                  f"mass {attention.mass:.2f} delta {abl.delta_mm:5.1f} "
                  f"(lat {abl.per_axis_mm[0]:5.1f} vert {abl.per_axis_mm[1]:5.1f} "
                  f"depth {abl.per_axis_mm[2]:5.1f})", flush=True)

    figure.suptitle(
        f"{cfg['label']} — attention across episodes and across time\n"
        f"pick3 checkpoint, prompt {cfg['task']!r}, first denoising step, p99 clip\n"
        "left column = start of episode; right column = the grasp",
        fontsize=12,
    )
    figure.tight_layout()
    path = DST / f"attention_{which}_over_time.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
