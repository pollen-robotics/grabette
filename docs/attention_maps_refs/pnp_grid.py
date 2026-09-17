"""Attention through a full pick-and-place: approach, grasp, transport, release.

The sugar task is not a pick. The gripper approaches the CUBE, grasps it,
carries it to the MUG, and releases. So the target of attention should change
partway through, and a timeline that stops at the grasp -- as the earlier grid
did -- cannot show that.

Phases are found from the gripper closure channel: it rises at the grasp,
stays high through transport, and falls at the release. Columns sample the
approach at fractions of the way to the grasp, then the carry at fractions of
the way from grasp to release.

Run with `--plan` to print the detected phases without running any inference.

Prompt: the checkpoint's own training string, verbatim. It reads wrongly for
this task -- it says "pick up the sugar cup" when the task is putting a cube
into a mug -- but it is the label the model learned, so it is the only
in-distribution choice. Using anything else measures the wrong thing, which is
exactly the mistake this script exists to correct.
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

CKPT = "SteveNguyen/sugar_cup_chunkrel_pi05_step20000"
REPO = "SteveNguyen/sugar_cup_graspproj_chunkrel"
HUB = "/home/steve/.cache/huggingface/hub"
ROOT = (
    f"{HUB}/datasets--SteveNguyen--sugar_cup_graspproj_chunkrel/snapshots/"
    "05e70fd531b3f2fd342dca427cc6baf7272dddb8"
)
# The 11-dim per-step actions live in a different copy of the same recording
# (identical 150 episodes, identical per-episode lengths, verified).
ACTIONS = (
    f"{HUB}/datasets--chouziel--grabette-sugar-cup-2008_graspproj/snapshots/"
    "96f1794700bccb9c7088100caed3c974aac82ad9/data/chunk-000/file-000.parquet"
)
TASK = "pick up the sugar cup"          # the checkpoint's training string
DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")

CLOSURE, DZ = 10, 2
RUN = 3                                  # consecutive frames needed to accept a transition
APPROACH_FRACTIONS = (0.0, 0.35, 0.7)
CARRY_FRACTIONS = (0.3, 0.65, 1.0)
CANDIDATES = tuple(range(0, 40))
N_EPISODES = 6
MIN_APPROACH = 12
MIN_CARRY = 12


def _first_run(flags: np.ndarray, value: bool, start: int = 0) -> int | None:
    """Index where `flags` first holds `value` for RUN consecutive frames."""
    run = 0
    for i in range(start, len(flags)):
        run = run + 1 if bool(flags[i]) == value else 0
        if run >= RUN:
            return i - RUN + 1
    return None


def plans():
    tbl = pq.read_table(ACTIONS, columns=["episode_index", "action"])
    ep = np.asarray(tbl["episode_index"])
    act = np.stack([np.asarray(r) for r in tbl["action"].to_pylist()]).astype(np.float64)

    out = {}
    for e in CANDIDATES:
        if len(out) >= N_EPISODES:
            break
        rows = act[ep == e]
        if not len(rows):
            continue
        closure = rows[:, CLOSURE]
        closed = closure > 0.5 * max(closure.max(), 1e-6)
        grasp = _first_run(closed, True)
        if grasp is None or grasp < MIN_APPROACH:
            print(f"ep{e}: no approach to sample (grasp={grasp}), skipped")
            continue
        release = _first_run(closed, False, start=grasp + RUN)
        if release is None:
            release = len(rows) - 1
        if release - grasp < MIN_CARRY:
            print(f"ep{e}: carry too short ({grasp}->{release}), skipped")
            continue

        frames, labels = [], []
        for f in APPROACH_FRACTIONS:
            frames.append(max(2, int(round(f * grasp))))
            labels.append(f"approach {int(f*100)}%")
        frames.append(grasp)
        labels.append("GRASP")
        for f in CARRY_FRACTIONS:
            frames.append(int(round(grasp + f * (release - grasp))))
            labels.append("RELEASE" if f == 1.0 else f"carry {int(f*100)}%")

        if len(set(frames)) < len(frames):
            print(f"ep{e}: phases collide ({frames}), skipped")
            continue

        remaining = np.concatenate(
            [np.cumsum(rows[:grasp, DZ][::-1])[::-1], np.zeros(len(rows) - grasp)]
        )
        out[e] = {
            "grasp": grasp, "release": release, "frames": frames,
            "labels": labels, "remaining": remaining, "n": len(rows),
        }
    return out


def main() -> None:
    plan = plans()
    print(f"\nepisodes: {sorted(plan)}")
    for e in sorted(plan):
        p = plan[e]
        print(f"  ep{e:3d}  len {p['n']:4d}  grasp {p['grasp']:4d}  "
              f"release {p['release']:4d}  carry {p['release']-p['grasp']:3d} frames")
        print(f"        frames {p['frames']}")
    total = sum(len(p["frames"]) for p in plan.values())
    print(f"\n{total} frames to analyse (~{total*11/60:.0f} min)")

    if "--plan" in sys.argv:
        return

    DST.mkdir(parents=True, exist_ok=True)
    policy, pre, post = load_pi05(CKPT, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)
    camera = adapter.camera_keys[0]

    rows_ = sorted(plan)
    ncol = len(APPROACH_FRACTIONS) + 1 + len(CARRY_FRACTIONS)
    figure, axes = plt.subplots(
        len(rows_), ncol, figsize=(4.0 * ncol, 3.2 * len(rows_)), dpi=95
    )
    done = 0
    for r, episode in enumerate(rows_):
        p = plan[episode]
        source = DatasetSource(
            REPO, episodes=[episode], camera_keys=adapter.camera_keys,
            task=TASK, root=ROOT, selection=sorted(set(p["frames"])),
        )
        by_frame = {obs.frame: obs for obs in source.frames()}
        for c, (want, label) in enumerate(zip(p["frames"], p["labels"])):
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
            abl = result.ablations[camera]
            axis.set_title(
                f"ep{episode}  f{obs.frame}  {label}\n"
                f"mass {attention.mass:.2f}   ablate {abl.delta_mm:.0f} mm\n"
                f"(lat {abl.per_axis_mm[0]:.0f}, vert {abl.per_axis_mm[1]:.0f}, "
                f"depth {abl.per_axis_mm[2]:.0f})",
                fontsize=7.5,
            )
            done += 1
            print(f"[{done}/{total}] ep{episode} f{obs.frame} {label:14s} "
                  f"mass {attention.mass:.2f} delta {abl.delta_mm:6.1f} "
                  f"(lat {abl.per_axis_mm[0]:5.1f} vert {abl.per_axis_mm[1]:5.1f} "
                  f"depth {abl.per_axis_mm[2]:5.1f})", flush=True)

    figure.suptitle(
        "sugar cube -> mug, full pick-and-place — attention by phase\n"
        f"checkpoint {CKPT.split('/')[-1]}, prompt {TASK!r} (the model's own "
        "training string), first denoising step, p99 clip",
        fontsize=12,
    )
    figure.tight_layout()
    path = DST / "attention_sugarcube_pnp.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
