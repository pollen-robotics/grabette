"""Tests for auto-derived episode quality tags (checks.tags).

These tags land in per-episode dataset metadata and let training filter out
bad episodes — so their logic is worth pinning.
"""
import json

from grabette_postprocess.checks.tags import (
    TAG_FIXED_GRIPPER,
    TAG_TRACKING_LOST,
    TAG_TRAJ_WITH_JUMPS,
    episode_tags,
    recording_tags,
    trajectory_tags,
)
from grabette_postprocess.checks.trajectory import TrajectoryReport


def test_trajectory_tags_none_is_empty():
    assert trajectory_tags(None) == []


def test_trajectory_tags_clean_is_empty():
    r = TrajectoryReport(name="x", n_tracked=100, tracking_pct=100.0,
                         n_jumps=0, median_angle_deg=10.0)
    assert trajectory_tags(r) == []


def test_trajectory_tags_tracking_lost_by_pct():
    r = TrajectoryReport(name="x", n_tracked=100, tracking_pct=40.0)
    assert TAG_TRACKING_LOST in trajectory_tags(r)


def test_trajectory_tags_tracking_lost_by_count():
    r = TrajectoryReport(name="x", n_tracked=1, tracking_pct=100.0)
    assert TAG_TRACKING_LOST in trajectory_tags(r)


def test_trajectory_tags_jumps_by_fraction():
    r = TrajectoryReport(name="x", n_tracked=100, tracking_pct=100.0,
                         n_jumps=20, median_angle_deg=10.0)
    assert TAG_TRAJ_WITH_JUMPS in trajectory_tags(r)


def test_trajectory_tags_zigzag():
    r = TrajectoryReport(name="x", n_tracked=100, tracking_pct=100.0,
                         n_jumps=6, median_angle_deg=120.0)
    assert TAG_TRAJ_WITH_JUMPS in trajectory_tags(r)


def test_recording_tags_fixed_gripper(tmp_path):
    (tmp_path / "angle_data.json").write_text(json.dumps({"samples": [
        {"cts": 0, "value": [0.0, 0.2]},
        {"cts": 100, "value": [0.5, 0.2]},   # proximal static
    ]}))
    assert recording_tags(tmp_path) == [TAG_FIXED_GRIPPER]


def test_episode_tags_combined_dedup_and_order(tmp_path):
    (tmp_path / "angle_data.json").write_text(json.dumps({"samples": [
        {"cts": 0, "value": [0.0, 0.2]},
        {"cts": 100, "value": [0.5, 0.2]},
    ]}))
    r = TrajectoryReport(name="x", n_tracked=1, tracking_pct=100.0)  # tracking_lost
    assert episode_tags(tmp_path, r) == [TAG_FIXED_GRIPPER, TAG_TRACKING_LOST]
