"""Does the policy's camera dependence grow as the gripper closes in?

One frame tells us the camera is used. A sweep tells us HOW, and that is the
shape that answers the range-perception question:

  - dependence rising as the object gets closer  -> visual servoing
  - dependence flat                              -> camera sets a coarse target,
                                                    the rest is proprioceptive

Frames are aligned on each episode's grasp so curves from different episodes
are comparable, and the x-axis is the REMAINING FORWARD TRAVEL to the grasp
(the cumulative +z the gripper still has to cover), read from the recorded
actions. That is a physical distance in metres, not a frame count, so episodes
performed at different speeds still line up.

Verified axis convention for this action space (camera frame, OpenCV):
    channel 0 = x  lateral, +right
    channel 1 = y  vertical, +DOWN
    channel 2 = z  depth/range, +FORWARD

Writes one CSV row per analysed frame so the aggregation can be redone without
re-running inference.
"""

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from grabette_attention.adapters.pi05 import Pi05Adapter
from grabette_attention.analysis import analyse_frame
from grabette_attention.loader import load_pi05
from grabette_attention.sources import DatasetSource

CKPT = "SteveNguyen/pick3_graspproj_chunkrel_pi05"
REPO = "SteveNguyen/mustard_graspproj"
ROOT = "/home/steve/.cache/huggingface/lerobot/local-converted/mustard_graspproj"
PARQUET = f"{ROOT}/data/chunk-000/file-000.parquet"
TASK = "pick up the mustard bottle"
CLOSURE = 10  # action channel index, per meta/info.json
DZ = 2  # forward channel of the per-step dataset action

OUT = Path(
    "/tmp/claude-1000/-home-steve-Project-Repo-GRABETTE-GRABETTE-RELEASE"
    "/616197ea-3b0d-4750-8fc3-e6b33174b8f0/scratchpad/sweep_rows.csv"
)

# Relative to the grasp frame. Negative is approach, positive is after closure.
OFFSETS = (-50, -40, -30, -20, -12, -6, 0, 8)
N_EPISODES = 10


def episode_plan():
    """Per episode: the grasp frame, and remaining forward travel per frame."""
    tbl = pq.read_table(PARQUET, columns=["episode_index", "action"])
    ep = np.asarray(tbl["episode_index"])
    act = np.stack([np.asarray(r) for r in tbl["action"].to_pylist()]).astype(np.float64)

    plans = {}
    for e in np.unique(ep):
        rows = act[ep == e]
        closure = rows[:, CLOSURE]
        hot = np.nonzero(closure > 0.5 * max(closure.max(), 1e-6))[0]
        if hot.size < 5 or hot[0] < max(-min(OFFSETS), 10):
            continue
        grasp = int(hot[0])
        if grasp + max(OFFSETS) >= len(rows):
            continue
        # Remaining forward travel from frame f to the grasp: the +z the gripper
        # has yet to cover. Positive while approaching, ~0 at the grasp.
        forward = rows[:, DZ]
        remaining = np.concatenate([np.cumsum(forward[:grasp][::-1])[::-1], [0.0]])
        plans[int(e)] = {"grasp": grasp, "remaining": remaining, "n": len(rows)}
        if len(plans) >= N_EPISODES:
            break
    return plans


def main() -> None:
    plans = episode_plan()
    print(f"episodes planned: {sorted(plans)}")
    total = sum(len(OFFSETS) for _ in plans)
    print(f"frames to analyse: {total}  (~{total * 11 / 60:.0f} min at 11 s/frame)\n")

    policy, pre, post = load_pi05(CKPT, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)
    camera = adapter.camera_keys[0]

    rows = ["episode,frame,offset,remaining_m,mass,delta_mm,x_mm,y_mm,z_mm"]
    started = time.perf_counter()
    done = 0

    for episode in sorted(plans):
        plan = plans[episode]
        grasp = plan["grasp"]
        wanted = [grasp + o for o in OFFSETS]
        source = DatasetSource(
            REPO,
            episodes=[episode],
            camera_keys=adapter.camera_keys,
            task=TASK,
            root=ROOT,
            selection=wanted,
        )
        for obs in source.frames():
            result = analyse_frame(adapter, obs, denoise_step="last", ablate=True)
            abl = result.ablations[camera]
            offset = obs.frame - grasp
            rem = float(plan["remaining"][min(obs.frame, grasp)])
            x, y, z = abl.per_axis_mm
            rows.append(
                f"{episode},{obs.frame},{offset},{rem:.5f},"
                f"{result.cameras[camera].mass:.4f},{abl.delta_mm:.3f},"
                f"{x:.3f},{y:.3f},{z:.3f}"
            )
            done += 1
            elapsed = time.perf_counter() - started
            print(
                f"[{done:3d}/{total}] ep{episode:03d} f{obs.frame:4d} "
                f"off{offset:+4d} rem {rem*1000:6.1f} mm  "
                f"mass {result.cameras[camera].mass:.2f}  "
                f"delta {abl.delta_mm:6.1f} mm  "
                f"(x {x:5.1f}  y {y:5.1f}  z {z:5.1f})  "
                f"eta {(elapsed/done)*(total-done)/60:4.1f} min",
                flush=True,
            )
            OUT.write_text("\n".join(rows) + "\n")

    print(f"\nwrote {OUT} ({done} rows)")
    print(json.dumps({"offsets": list(OFFSETS), "episodes": sorted(plans)}))


if __name__ == "__main__":
    main()
