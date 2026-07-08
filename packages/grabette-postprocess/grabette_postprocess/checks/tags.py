"""Derive LeRobot-compatible per-episode tags from the pipeline's checks.

Tags are short machine-readable labels written into the built dataset's
per-episode metadata (a `tags` list-of-strings column in meta/episodes/*.parquet;
see dataset._write_episode_tags). LeRobot preserves extra episode-metadata columns
and exposes them as ``ds.meta.episodes[i]["tags"]``, so downstream training can
filter episodes by a known quality issue without re-running the checks.

Two sources, mirroring the two check phases:
  - recording (pre-SLAM)  → TAG_FIXED_GRIPPER   (a gripper joint never actuated)
  - trajectory (post-SLAM) → TAG_TRACKING_LOST   (SLAM lost the camera)
                          → TAG_TRAJ_WITH_JUMPS (path riddled with relocalization
                                                 jumps, even when tracking is high)

Auto-derived only — there is no manual tagging path.
"""

from pathlib import Path

from grabette_postprocess.checks.recording import static_gripper_joints

TAG_FIXED_GRIPPER = "fixed_gripper"
TAG_TRACKING_LOST = "tracking_lost"
TAG_TRAJ_WITH_JUMPS = "traj_with_jumps"

# Tracked-frame percentage below which SLAM effectively lost the camera over much
# of the episode (matches the "Low tracking" trajectory warning threshold).
TRACKING_LOST_PCT = 50.0

# A trajectory riddled with large relocalization jumps — mirrors the two
# jump-based trajectory checks (see checks.trajectory.check_trajectory):
#   - the "N jumps > threshold" warning: jumps over JUMP_FRACTION of tracked frames
#   - the "Zigzag pattern" error: > ZIGZAG_MIN_JUMPS jumps with a median
#     direction-change above ZIGZAG_MIN_ANGLE_DEG (repeated relocalization failures)
# Either firing means the SLAM path is unreliable even when tracking % stays high.
JUMP_FRACTION = 0.1
ZIGZAG_MIN_JUMPS = 5
ZIGZAG_MIN_ANGLE_DEG = 90.0


def recording_tags(ep_dir: Path) -> list[str]:
    """Tags derivable from the raw recording (pre-SLAM). Global `fixed_gripper`
    when either gripper joint stayed static over the whole episode."""
    return [TAG_FIXED_GRIPPER] if static_gripper_joints(ep_dir) else []


def trajectory_tags(report) -> list[str]:
    """Tags derivable from the SLAM trajectory report (post-SLAM), stable-ordered.
    None-safe.

    `tracking_lost` when tracking was essentially absent (< 2 tracked frames) or
    SLAM held the camera for less than TRACKING_LOST_PCT of the episode.

    `traj_with_jumps` when the path is broken by large relocalization jumps —
    either the jump rate exceeds JUMP_FRACTION of tracked frames, or the zigzag
    pattern (many jumps with reversing direction) fired. Independent of tracking %,
    so it catches episodes SLAM tracked well but reconstructed unreliably."""
    if report is None:
        return []
    tags: list[str] = []
    if report.n_tracked < 2 or report.tracking_pct < TRACKING_LOST_PCT:
        tags.append(TAG_TRACKING_LOST)
    jumpy = report.n_jumps > report.n_tracked * JUMP_FRACTION
    zigzag = (report.n_jumps > ZIGZAG_MIN_JUMPS
              and report.median_angle_deg > ZIGZAG_MIN_ANGLE_DEG)
    if jumpy or zigzag:
        tags.append(TAG_TRAJ_WITH_JUMPS)
    return tags


def episode_tags(ep_dir: Path, report=None) -> list[str]:
    """Combined, de-duplicated, stable-ordered tags for one built episode.

    ep_dir: the raw episode directory (for the recording-derived tags).
    report: the episode's TrajectoryReport, or None when unavailable.
    """
    seen: list[str] = []
    for tag in (*recording_tags(ep_dir), *trajectory_tags(report)):
        if tag not in seen:
            seen.append(tag)
    return seen
