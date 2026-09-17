"""Which action channel is vertical, and is the frame body-relative or world?

The per-axis ablation millimetres are unlabelled until we know this. Two
questions, one test, run over many episodes so a single odd grasp cannot
decide it:

1. Frame type. If the deltas live in the gripper/camera frame, the approach
   direction is the SAME channel in every episode (the object is always in
   front of the camera). If they live in a gravity-aligned world frame, the
   approach direction scatters with where the object happened to sit, while
   the LIFT direction stays fixed.
2. Which channel, which sign. The lift after closure is unambiguous: the
   gripper goes up. Whichever channel carries it, with whatever sign, is
   vertical.

Reads the parquet directly — no video decode, so all 199 episodes are cheap.
"""

import numpy as np
import pyarrow.parquet as pq

ROOT = "/home/steve/.cache/huggingface/lerobot/local-converted/mustard_graspproj"
PARQUET = f"{ROOT}/data/chunk-000/file-000.parquet"
CLOSURE = 10  # action channel index, per meta/info.json names
NAMES = ("dx", "dy", "dz")


def phase_means(act: np.ndarray, closed: np.ndarray):
    """Mean translation per channel for the approach and the lift phases."""
    # First frame where the gripper is meaningfully closed = the grasp.
    hot = np.nonzero(closed > 0.5 * max(closed.max(), 1e-6))[0]
    if hot.size < 5 or hot[0] < 10:
        return None
    grasp = int(hot[0])
    approach = act[max(0, grasp - 40) : grasp, :3]
    lift = act[grasp : grasp + 40, :3]
    if len(approach) < 10 or len(lift) < 10:
        return None
    return approach.sum(axis=0), lift.sum(axis=0)


def main() -> None:
    tbl = pq.read_table(PARQUET, columns=["episode_index", "frame_index", "action"])
    ep = np.asarray(tbl["episode_index"])
    actions = np.stack([np.asarray(r) for r in tbl["action"].to_pylist()]).astype(np.float64)
    print(f"{len(actions)} frames, {len(np.unique(ep))} episodes, action dim {actions.shape[1]}")

    approaches, lifts = [], []
    for e in np.unique(ep):
        m = ep == e
        got = phase_means(actions[m], actions[m][:, CLOSURE])
        if got is not None:
            approaches.append(got[0])
            lifts.append(got[1])
    A = np.stack(approaches)
    L = np.stack(lifts)
    print(f"usable episodes: {len(A)}\n")

    for label, D in (("APPROACH (40 frames before grasp)", A), ("LIFT (40 frames after grasp)", L)):
        print(f"{label}, cumulative travel in metres:")
        for i, n in enumerate(NAMES):
            col = D[:, i]
            # Sign agreement: how consistent is the direction across episodes?
            agree = max((col > 0).mean(), (col < 0).mean())
            print(
                f"  {n}: mean {col.mean():+.4f}  sd {col.std():.4f}"
                f"  |mean|/sd {abs(col.mean())/max(col.std(),1e-9):5.2f}"
                f"  same-sign in {agree:5.0%} of episodes"
            )
        print()

    print("reading:")
    print("  a channel with high |mean|/sd AND ~100% sign agreement is a fixed")
    print("  physical direction; scattered channels are not.")


if __name__ == "__main__":
    main()
