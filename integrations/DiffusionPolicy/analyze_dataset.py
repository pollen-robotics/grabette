"""Quantify a converted Gripette dataset (11D delta action + 2D gripper state).

Prints the numbers that matter for diagnosing "not enough data / variability":
size, gripper-closure coverage, distal usage, action-delta magnitudes, and an
episode-type breakdown (normal / release-start / never-closes). Run it on the
real dataset AND the working sim dataset and compare — the gaps are the fixes.

Lightweight: reads only the `action`/`episode_index` columns (no video).

Usage:
  uv run python analyze_dataset.py --repo_id <id> [<id2> ...]
"""

import argparse
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# action layout: [dx,dy,dz, r6d_0..5, proximal(9), distal(10)]
PROX, DIST = 9, 10
CLOSED_THRESH = -1.0  # proximal below this = "closed" (full close ~ -1.5)


def analyze(repo_id: str, root: str | None = None):
    ds = LeRobotDataset(repo_id, root=root)
    hf = ds.hf_dataset.select_columns(["action", "episode_index"])
    A = np.stack([np.asarray(a, dtype=np.float64) for a in hf["action"]])  # [T, 11]
    ep = np.asarray(hf["episode_index"])
    eps = np.unique(ep)

    print(f"\n{'='*64}\n  {repo_id}\n{'='*64}")
    lens = np.array([(ep == e).sum() for e in eps])
    print(f"  episodes: {len(eps)}   frames: {len(A)}   "
          f"ep length min/mean/max: {lens.min()}/{lens.mean():.0f}/{lens.max()}")

    # --- Gripper (OBJECT-AGNOSTIC: use within-episode SWING, not absolute
    #     depth. Absolute closure depth depends on object width and, in sim,
    #     on non-physical full-closure — so it's not comparable across
    #     datasets/objects. Swing = how much the gripper actuated, which
    #     measures "did a grasp motion happen" independent of object size.) ---
    prox, dist = A[:, PROX], A[:, DIST]
    ep_swing_prox = np.array([prox[ep == e].max() - prox[ep == e].min() for e in eps])
    ep_swing_dist = np.array([dist[ep == e].max() - dist[ep == e].min() for e in eps])
    SWING_THRESH = 0.3  # rad of proximal travel ≈ "a close transition occurred"
    print(f"\n  GRIPPER (object-agnostic swing = within-episode max-min)")
    print(f"    proximal swing: mean {ep_swing_prox.mean():.2f}  "
          f"p10 {np.percentile(ep_swing_prox,10):.2f}  p90 {np.percentile(ep_swing_prox,90):.2f}")
    print(f"    distal   swing: mean {ep_swing_dist.mean():.2f}  "
          f"p10 {np.percentile(ep_swing_dist,10):.2f}  p90 {np.percentile(ep_swing_dist,90):.2f}")
    print(f"    episodes with a close transition (prox swing>{SWING_THRESH}): "
          f"{(ep_swing_prox>SWING_THRESH).sum()}/{len(eps)} "
          f"({100*(ep_swing_prox>SWING_THRESH).mean():.0f}%)")
    print(f"    distal swing / proximal swing (mean ratio): "
          f"{ep_swing_dist.mean()/max(ep_swing_prox.mean(),1e-6):.2f} "
          f"(≈1 = distal actuates with proximal; «1 = distal under-used)")

    # --- Action deltas (position) ---
    dmag = np.linalg.norm(A[:, :3], axis=1)
    print(f"\n  POSITION DELTAS (per-step, mm)")
    print(f"    |Δpos| mean {1000*dmag.mean():.2f}  p50 {1000*np.median(dmag):.2f}  "
          f"p95 {1000*np.percentile(dmag,95):.2f}  max {1000*dmag.max():.2f}")

    # --- Episode-type breakdown (object-agnostic: by gripper SWING direction) ---
    # A "close transition" = proximal ends meaningfully more closed than its
    # episode max (it actuated toward closed). Direction, not absolute depth.
    closes = starts_low = 0
    n = len(eps)
    for e in eps:
        p = prox[ep == e]
        swing = p.max() - p.min()
        if swing > 0.3:
            # did it move toward closed (min reached after some open phase)?
            closes += 1
        if (p[0] - p.min()) < 0.1 and swing > 0.3:
            # starts already near its most-closed → release-like
            starts_low += 1
    print(f"\n  EPISODE TYPES (object-agnostic, by swing)")
    print(f"    episodes that actuate the gripper (swing>0.3): {closes}/{n} ({100*closes/n:.0f}%)")
    print(f"    episodes starting already-closed (release-like): {starts_low}/{n} ({100*starts_low/n:.0f}%)")

    # --- Data-hygiene anomalies: glitchy (position-delta spike) + truncated ---
    SPIKE_MM = 15.0    # a per-step delta this large is almost certainly a SLAM/tracking glitch
    SHORT_FRAMES = 80  # episodes shorter than this are likely truncated/incomplete
    dmag_mm = np.linalg.norm(A[:, :3], axis=1) * 1000
    glitchy, short = [], []
    for e in eps:
        m = ep == e
        if dmag_mm[m].max() > SPIKE_MM:
            glitchy.append((int(e), round(float(dmag_mm[m].max()), 1)))
        if m.sum() < SHORT_FRAMES:
            short.append((int(e), int(m.sum())))
    print(f"\n  ANOMALIES (candidates to drop before training)")
    print(f"    glitchy (Δpos spike >{SPIKE_MM}mm): {len(glitchy)} eps  "
          f"{glitchy[:12]}{' ...' if len(glitchy)>12 else ''}")
    print(f"    truncated (<{SHORT_FRAMES} frames): {len(short)} eps  "
          f"{short[:12]}{' ...' if len(short)>12 else ''}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", nargs="+", required=True, help="One or more dataset repo ids to analyze/compare")
    p.add_argument("--root", default=None, help="Local dataset root (for a local-converted dataset, e.g. ~/.cache/huggingface/lerobot/local-converted/<repo--id>). Applies to all --repo_id.")
    args = p.parse_args()
    for rid in args.repo_id:
        analyze(rid, root=args.root)


if __name__ == "__main__":
    main()
