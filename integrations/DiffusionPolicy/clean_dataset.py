"""Reject glitch-ridden episodes from a Grabette raw dataset before conversion.

SLAM-tracked poses contain impulsive glitches — frames where the tracker jumps
by a physically-impossible amount (e.g. 100–400 mm in one 20 ms frame). Two kinds:
  * "return" spike — teleports out and back (2 bad per-step transitions);
  * "relocalization" step — teleports and STAYS (1 bad transition; the offset
    then cancels in every later camera-local delta).

Because training uses camera-local deltas, an ISOLATED glitch is best handled at
conversion: `convert_dataset.py` zeroes any delta above --despike_max_mm/deg
("hold for that frame"), which removes the bad action with no side effect. This
script handles the case zeroing CAN'T fix: an episode with a glitch segment too
long to absorb (a real tracking-loss — many consecutive dead frames), or a
glitch inside the grasp window. Those whole episodes are dropped.

Detection uses the SAME per-step cap as the converter, so "what gets zeroed" and
"what triggers a drop" share one definition. An episode is rejected if:
  * its longest run of consecutive bad transitions > --max_run (tracking loss), OR
  * its bad-transition fraction > --reject_fraction, OR
  * a bad transition falls within ± --grasp_window frames of the most-closed
    instant (the grasp phase, where fidelity matters most).

Non-destructive: writes a NEW dataset (via lerobot's delete_episodes); the raw
dataset is never touched.

Usage:
  # audit only (decide thresholds, change nothing):
  uv run python clean_dataset.py --repo_id SteveNguyen/test_pick_can_100 --dry_run

  # write the kept-episode dataset, then convert THAT (with matching despike caps):
  uv run python clean_dataset.py --repo_id SteveNguyen/test_pick_can_100 \\
      --output_repo_id SteveNguyen/test_pick_can_100_clean
  uv run python convert_dataset.py --repo_id SteveNguyen/test_pick_can_100_clean ...
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes
from rotation import rotation_6d_to_rotation_matrix_numpy


logger = logging.getLogger(__name__)


# --- Layout: the converter accepts a raw 8D or an (absolute) 11D action. The
#     position is always dims 0:3; rotation and gripper columns differ. ---
def layout_for(dim: int) -> dict:
    if dim == 8:  # [x,y,z, ax,ay,az, proximal(6), distal(7)]
        return {"pos": slice(0, 3), "rot": slice(3, 6), "rot_kind": "rotvec", "prox": 6}
    if dim == 11:  # [x,y,z, r6d_0..5, proximal(9), distal(10)]
        return {"pos": slice(0, 3), "rot": slice(3, 9), "rot_kind": "6d", "prox": 9}
    raise ValueError(f"Unexpected action dim {dim}; expected 8 (raw) or 11 (converted).")


def rotations_of(poses: np.ndarray, lay: dict) -> Rotation:
    """Build a (vectorised) scipy Rotation from the rotation columns."""
    r = poses[:, lay["rot"]]
    if lay["rot_kind"] == "rotvec":
        return Rotation.from_rotvec(r)
    return Rotation.from_matrix(rotation_6d_to_rotation_matrix_numpy(r))


def find_runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start, end) index pairs for each consecutive True run."""
    runs = []
    t, n = 0, len(flag)
    while t < n:
        if flag[t]:
            s = t
            while t + 1 < n and flag[t + 1]:
                t += 1
            runs.append((s, t))
        t += 1
    return runs


def detect_episode(sub: np.ndarray, lay: dict, cfg) -> dict:
    """Flag per-step transitions above the physical cap; return episode stats.

    A "transition" t is the step from frame t to t+1. Same definition the
    converter uses to zero deltas — so the runs here are exactly the segments
    the converter would zero, and we reject the ones too long to absorb.
    """
    pos = sub[:, lay["pos"]].astype(np.float64)
    rot = rotations_of(sub, lay)
    n = len(sub)
    if n < 2:
        return {"n": n, "n_bad": 0, "max_run": 0, "frac": 0.0, "grasp_hit": False, "bad_idx": []}
    dpos_mm = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 1000.0  # (n-1,)
    dang_deg = np.degrees((rot[:-1].inv() * rot[1:]).magnitude())    # (n-1,)
    bad = (dpos_mm > cfg.despike_max_mm) | (dang_deg > cfg.despike_max_deg)
    runs = find_runs(bad)
    max_run = max((b - a + 1 for a, b in runs), default=0)
    # Grasp window: transitions near the most-closed instant (min proximal —
    # full close is the most negative). A glitch here can't be safely zeroed.
    gc = int(np.argmin(sub[:, lay["prox"]]))
    gw = cfg.grasp_window
    grasp_hit = any(not (b < gc - gw or a > gc + gw) for a, b in runs)
    return {"n": n, "n_bad": int(bad.sum()), "max_run": max_run,
            "frac": int(bad.sum()) / max(n, 1), "grasp_hit": grasp_hit,
            "bad_idx": np.where(bad)[0].tolist()}


def decide(stats: dict, cfg) -> tuple[bool, str]:
    """Reject decision for an episode. Returns (reject?, reason)."""
    if stats["n_bad"] == 0:
        return False, "clean"
    if stats["max_run"] > cfg.max_run:
        return True, f"loss run {stats['max_run']}>{cfg.max_run} (tracking loss)"
    if stats["frac"] > cfg.reject_fraction:
        return True, f"frac {stats['frac']*100:.1f}%>{cfg.reject_fraction*100:.0f}%"
    if stats["grasp_hit"]:
        return True, "glitch in grasp window"
    return False, f"keep ({stats['n_bad']} isolated, zeroed at conversion)"


def audit(repo_id: str, root, cfg) -> list[int]:
    """Detect on the RAW dataset, decide rejects, print the audit log."""
    ds = LeRobotDataset(repo_id, root=root)
    hf = ds.hf_dataset.select_columns(["action", "episode_index"])
    A = np.stack([np.asarray(a, dtype=np.float64) for a in hf["action"]])
    ep = np.asarray(hf["episode_index"])
    eps = np.unique(ep)
    lay = layout_for(A.shape[1])

    print(f"\n{'='*72}\n  AUDIT  {repo_id}\n{'='*72}")
    print(f"  layout: action dim {A.shape[1]}   episodes: {len(eps)}   frames: {len(A)}")
    print(f"  glitch: >{cfg.despike_max_mm:.0f}mm or >{cfg.despike_max_deg:.0f}° per step   "
          f"reject if: run>{cfg.max_run}  frac>{cfg.reject_fraction*100:.0f}%  grasp±{cfg.grasp_window}")

    reject, rej_lines, keep_lines, n_zeroed = [], [], [], 0
    for e in eps:
        idx = np.where(ep == e)[0]
        stats = detect_episode(A[idx], lay, cfg)
        is_reject, reason = decide(stats, cfg)
        frames = ",".join(str(t) for t in stats["bad_idx"])
        if is_reject:
            reject.append(int(e))
            rej_lines.append(f"    ep {int(e):>4}  {stats['n_bad']:>2} glitch(es) @ {frames}  → {reason}")
        elif stats["n_bad"] > 0:
            n_zeroed += stats["n_bad"]
            keep_lines.append(f"    ep {int(e):>4}  {stats['n_bad']:>2} spike(s) @ {frames}")

    # Frame indices are local transition indices t: the jump is between frame t and t+1.
    print(f"\n  REJECT ({len(reject)}/{len(eps)}) — dropped entirely:")
    for line in rej_lines or ["    (none)"]:
        print(line)
    print(f"\n  KEEP, SPIKES ZEROED AT CONVERSION ({len(keep_lines)}) — {n_zeroed} isolated delta(s):")
    print("  (frame = local transition t; jump is between frame t and t+1)")
    for line in keep_lines or ["    (none)"]:
        print(line)
    print(f"\n  KEEP, CLEAN: {len(eps) - len(reject) - len(keep_lines)} episodes\n")
    return reject


def copy_dataset(src_root: Path, dst_root: Path, overwrite: bool) -> None:
    """Self-contained copy (deref HF-cache symlinks), used when nothing is rejected."""
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_root} exists; pass --overwrite_output or pick another output.")
        shutil.rmtree(dst_root)
    dst_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, dst_root)  # symlinks=False: dereference into a snapshot
    for req in ("meta/info.json", "meta/stats.json"):
        if not (dst_root / req).exists():
            raise FileNotFoundError(f"Copy incomplete: {dst_root/req} missing.")


def parse_args():
    p = argparse.ArgumentParser(description="Reject glitch-ridden episodes before conversion")
    p.add_argument("--repo_id", required=True, help="Raw dataset repo id")
    p.add_argument("--root", default=None, help="Local dataset root (else resolved from HF cache)")
    p.add_argument("--output_repo_id", default=None, help="Cleaned dataset repo id. Required unless --dry_run.")
    p.add_argument("--output_root", default=None,
                   help="Destination path (default: ~/.cache/huggingface/lerobot/local-converted/<id>)")
    p.add_argument("--overwrite_output", action="store_true", help="Delete destination if it exists")
    p.add_argument("--dry_run", action="store_true", help="Audit only — decide rejects, change nothing")
    # glitch definition — keep in sync with convert_dataset.py's --despike_max_*
    p.add_argument("--despike_max_mm", type=float, default=80.0, help="Per-step |Δpos| above this (mm) is a glitch")
    p.add_argument("--despike_max_deg", type=float, default=45.0, help="Per-step rotation delta above this (deg) is a glitch")
    # rejection thresholds
    p.add_argument("--max_run", type=int, default=3, help="Longest run of consecutive glitch transitions still kept (else reject as tracking loss)")
    p.add_argument("--reject_fraction", type=float, default=0.05, help="Glitch fraction above which the episode is rejected")
    p.add_argument("--grasp_window", type=int, default=10, help="± frames around the most-closed instant that trigger reject")
    p.add_argument("--push_to_hub", default=None, help="If set, push the cleaned dataset to this Hub repo id")
    p.add_argument("--hub_private", action="store_true", help="Push as a private Hub repo")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    reject = audit(args.repo_id, args.root, args)

    if args.dry_run:
        logger.info("Dry run — no dataset written.")
        return
    if not args.output_repo_id:
        raise SystemExit("--output_repo_id is required (or use --dry_run).")

    if args.output_root:
        dst_root = Path(args.output_root).expanduser().resolve()
    else:
        dst_root = Path.home() / ".cache/huggingface/lerobot/local-converted" / args.output_repo_id.replace("/", "--")

    if reject:
        src = LeRobotDataset(args.repo_id, root=args.root)
        if dst_root.exists():
            if not args.overwrite_output:
                raise FileExistsError(f"{dst_root} exists; pass --overwrite_output or pick another output.")
            shutil.rmtree(dst_root)
        logger.info(f"Removing {len(reject)} episode(s) → {dst_root}")
        delete_episodes(src, reject, output_dir=dst_root, repo_id=args.output_repo_id)
    else:
        logger.info(f"No rejects → copying to {dst_root}")
        src_root = Path(LeRobotDataset(args.repo_id, root=args.root).root)
        copy_dataset(src_root, dst_root, args.overwrite_output)

    logger.info(f"Cleaned dataset ready: {dst_root}")

    if args.push_to_hub:
        ds_push = LeRobotDataset(args.output_repo_id, root=dst_root)
        if args.push_to_hub != args.output_repo_id:
            ds_push.repo_id = args.push_to_hub
            ds_push.meta.repo_id = args.push_to_hub
        ds_push.push_to_hub(private=args.hub_private, push_videos=True)
        logger.info(f"Pushed: {args.push_to_hub}")


if __name__ == "__main__":
    main()
