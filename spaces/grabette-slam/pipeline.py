"""Episode → SLAM → LeRobot → Hub pipeline, run in-process inside the Space.

Thin glue over grabette_postprocess. The SLAM binary is bundled in the image
(no Docker), so run_oak_slam is called with binary=OAK_VSLAM_BINARY.
"""

import os
from pathlib import Path

from grabette_postprocess.convert import convert_episode
from grabette_postprocess.oak_slam import run_oak_slam
from grabette_postprocess.dataset import build_dataset

BINARY = os.environ.get("OAK_VSLAM_BINARY", "/usr/local/bin/offline_vslam")


def find_episode_dirs(root: Path) -> list[Path]:
    """Every directory containing a raw OAK-D recording, anywhere under root.

    Recursive so it works whether the dataset wraps episodes in a folder, puts
    them at the top level, or is a single episode at the root.
    """
    return sorted({p.parent for p in Path(root).rglob("oakd_left.mp4")})


def run_slam(dataset_dir, log=print) -> list[Path]:
    """Convert + SLAM every episode. Returns the dirs that produced a trajectory."""
    episodes = find_episode_dirs(Path(dataset_dir))
    if not episodes:
        raise ValueError(f"No episodes (oakd_left.mp4) found under {dataset_dir}")

    log(f"Found {len(episodes)} episode(s)")
    processed = []
    for ep in episodes:
        log(f"▶ {ep.name}: convert")
        convert_episode(ep)

        log(f"▶ {ep.name}: SLAM")
        r = run_oak_slam(ep, binary=BINARY, show_progress=False)
        if r.trajectory_path is None:
            log(f"  ✗ {ep.name}: SLAM failed (rc={r.returncode})")
            continue
        log(f"  ✓ {ep.name}: tracking {r.tracking_pct:.1f}% ({r.tracked_frames}/{r.total_frames})")
        processed.append(ep)
    return processed


def process_dataset(dataset_dir, target_repo, task, root, log=print) -> int:
    """Convert + SLAM + build a LeRobot dataset + push it to the Hub.

    The caller must set the HF_TOKEN env var (with write/manage-repos scope) first
    so push_to_hub uploads under that account. Returns the episode count.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    processed = run_slam(dataset_dir, log=log)
    if not processed:
        raise RuntimeError("No episode produced a trajectory; nothing to push.")

    log(f"Building LeRobot dataset from {len(processed)} episode(s)…")
    build_dataset(repo_id=target_repo, episode_dirs=processed, task=task, root=Path(root))

    log(f"Pushing to https://huggingface.co/datasets/{target_repo} …")
    ds = LeRobotDataset(target_repo, root=Path(root))
    ds.push_to_hub(tags=["lerobot", "grabette"])
    return len(processed)
