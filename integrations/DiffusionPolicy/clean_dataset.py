"""Reject episodes with UNRECOVERABLE SLAM tracking loss, before conversion.

The wrist camera loses SLAM tracking when the grasped object occludes it. The
postprocess build flags those frames (`is_lost`) and holds the last pose, so a
SHORT lost gap is benign: "assume no motion" (held pose → delta ≈ 0) is a fine
approximation for one or a few frames, and any residual re-acquisition jump is
zeroed per-frame by `convert_dataset.py`'s --despike backstop. A LONG lost run,
though, means the arm really moved through the occlusion — that motion is gone
and can't be recovered, so the whole episode is dropped.

So rejection keys off `is_lost` directly (the SLAM's own signal, carried into the
dataset by the postprocess build), NOT a magnitude heuristic:
  * reject if the longest consecutive lost run > --max_lost_run, OR
  * reject if the lost-frame fraction > --max_lost_fraction.
Everything else is kept (held pose ≈ no motion); the 80 mm/45° per-frame despike
in convert_dataset.py mops up the few re-acquisition jumps / genuine glitches.

Requires the `is_lost` feature (rebuild the dataset with the postprocess
generate_dataset.py if it's missing). Non-destructive: writes a NEW dataset via
lerobot's delete_episodes; the source is untouched.

Usage:
  # audit only (decide thresholds, change nothing):
  uv run python clean_dataset.py --repo_id <user>/<dataset> --dry_run

  # write the kept-episode dataset, then convert THAT:
  uv run python clean_dataset.py --repo_id <user>/<dataset> \\
      --output_repo_id <user>/<dataset>_clean
  uv run python convert_dataset.py --repo_id <user>/<dataset>_clean ...
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes


logger = logging.getLogger(__name__)


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


def detect_episode(is_lost: np.ndarray) -> dict:
    """SLAM-lost stats for one episode: count, fraction, longest run, run spans."""
    n = len(is_lost)
    lost = is_lost.astype(bool)
    runs = find_runs(lost)
    longest = max((b - a + 1 for a, b in runs), default=0)
    n_lost = int(lost.sum())
    return {"n": n, "n_lost": n_lost, "longest_run": longest,
            "frac": n_lost / max(n, 1), "runs": runs}


def decide(stats: dict, cfg) -> tuple[bool, str]:
    """Reject decision for an episode. Returns (reject?, reason)."""
    if stats["n_lost"] == 0:
        return False, "clean"
    if stats["longest_run"] > cfg.max_lost_run:
        return True, f"lost run {stats['longest_run']}>{cfg.max_lost_run} (motion unrecoverable)"
    if stats["frac"] > cfg.max_lost_fraction:
        return True, f"lost {stats['frac']*100:.0f}%>{cfg.max_lost_fraction*100:.0f}%"
    return False, f"keep ({stats['n_lost']} lost frame(s), held pose ≈ no motion)"


def audit(repo_id: str, root, cfg) -> list[int]:
    """Read `is_lost`, decide rejects, print the audit log."""
    ds = LeRobotDataset(repo_id, root=root)
    if "is_lost" not in ds.hf_dataset.column_names:
        raise SystemExit(
            "Dataset has no 'is_lost' feature — it predates the SLAM-lost fix.\n"
            "Rebuild it with the postprocess scripts/pipeline/generate_dataset.py first."
        )
    hf = ds.hf_dataset.select_columns(["is_lost", "episode_index"])
    il = np.asarray(hf["is_lost"], dtype=np.float32).reshape(-1)
    ep = np.asarray(hf["episode_index"])
    eps = np.unique(ep)

    print(f"\n{'='*72}\n  AUDIT  {repo_id}\n{'='*72}")
    print(f"  episodes: {len(eps)}   frames: {len(il)}   "
          f"reject if: longest lost run > {cfg.max_lost_run}  OR  lost fraction > {cfg.max_lost_fraction*100:.0f}%")

    reject, rej_lines, keep_lines = [], [], []
    for e in eps:
        idx = np.where(ep == e)[0]
        stats = detect_episode(il[idx])
        is_reject, reason = decide(stats, cfg)
        spans = ",".join(f"{a}-{b}" if b > a else f"{a}" for a, b in stats["runs"])
        line = (f"    ep {int(e):>4}  {stats['n_lost']:>3} lost ({stats['frac']*100:4.0f}%)  "
                f"longest run {stats['longest_run']:>3}")
        if is_reject:
            reject.append(int(e))
            rej_lines.append(f"{line}  → {reason}")
        elif stats["n_lost"] > 0:
            keep_lines.append(f"{line}  @ {spans}")

    print(f"\n  REJECT ({len(reject)}/{len(eps)}) — motion lost through a long occlusion:")
    for line in rej_lines or ["    (none)"]:
        print(line)
    print(f"\n  KEEP, SHORT LOST GAPS ({len(keep_lines)}) — held pose ≈ no motion; despike backstop in convert:")
    print("  (spans = local lost-frame index ranges)")
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
    p = argparse.ArgumentParser(description="Reject episodes with unrecoverable SLAM tracking loss")
    p.add_argument("--repo_id", required=True, help="Dataset repo id (must carry the is_lost feature)")
    p.add_argument("--root", default=None, help="Local dataset root (else resolved from HF cache)")
    p.add_argument("--output_repo_id", default=None, help="Cleaned dataset repo id. Required unless --dry_run.")
    p.add_argument("--output_root", default=None,
                   help="Destination path (default: ~/.cache/huggingface/lerobot/local-converted/<id>)")
    p.add_argument("--overwrite_output", action="store_true", help="Delete destination if it exists")
    p.add_argument("--dry_run", action="store_true", help="Audit only — decide rejects, change nothing")
    # rejection thresholds (is_lost based)
    p.add_argument("--max_lost_run", type=int, default=10,
                   help="Reject if the longest consecutive SLAM-lost run exceeds this (frames). "
                        "Short runs are kept: held pose ≈ no motion.")
    p.add_argument("--max_lost_fraction", type=float, default=0.3,
                   help="Also reject if the lost-frame fraction exceeds this (catches many scattered losses).")
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
