"""Episode discovery + trajectory-CSV resolution — the single source of truth for
"which directories are episodes" and "where is this episode's trajectory".

Before this module, each script (and the HF Space) re-implemented its own glob:
check_dataset globbed oakd_imu.json, generate_dataset globbed raw_video.mp4,
trajectory had find_trajectory_episodes, the Space had find_episode_dirs. They now
all call these helpers so the conventions live in exactly one place.
"""

from pathlib import Path

# The trajectory CSV a processed episode carries: the SLAM output, or the older
# mapping-based name. Resolved (in this order) by find_trajectory_csv().
TRAJECTORY_CSV_NAMES = ("camera_trajectory.csv", "mapping_camera_trajectory.csv")


def find_episodes(root: Path, *, anchor: str = "oakd_left.mp4") -> list[Path]:
    """Raw-recording episode directories under `root`, identified by containing
    `anchor` (default oakd_left.mp4, the SLAM input — pass e.g. "oakd_imu.json"
    to anchor on the IMU instead).

    Recursive. If `root` itself contains `anchor` (and nothing nested does), `root`
    is returned as a single episode. Sorted and de-duplicated.
    """
    root = Path(root).expanduser().absolute()
    eps = sorted({p.parent for p in root.rglob(anchor)})
    if not eps and (root / anchor).is_file():
        eps = [root]
    return eps


def find_trajectory_csv(ep_dir: Path) -> Path | None:
    """The episode's trajectory CSV (camera_trajectory.csv, else the mapping
    variant), or None if neither exists."""
    for name in TRAJECTORY_CSV_NAMES:
        p = Path(ep_dir) / name
        if p.is_file():
            return p
    return None


def find_processed_episodes(root: Path) -> list[Path]:
    """Episode directories under `root` that already carry a trajectory CSV
    (i.e. SLAM has run) — ready for dataset building or trajectory checks.

    If `root` itself is such an episode it is returned alone; otherwise every
    matching subdirectory is returned. Sorted and de-duplicated.
    """
    root = Path(root).expanduser().absolute()
    if find_trajectory_csv(root) is not None:
        return [root]
    eps: list[Path] = []
    for name in TRAJECTORY_CSV_NAMES:
        for traj in sorted(root.rglob(name)):
            if traj.parent not in eps:
                eps.append(traj.parent)
    return sorted(set(eps))


def find_trajectory_episodes(path: Path) -> list[Path]:
    """Alias for find_processed_episodes — episodes that carry a trajectory CSV.
    Kept under this name for the check_trajectory CLI's readability."""
    return find_processed_episodes(path)
