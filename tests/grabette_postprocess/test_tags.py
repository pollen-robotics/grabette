"""Unit tests for grabette_postprocess.checks.tags (pure tag-derivation logic)."""

from types import SimpleNamespace

from grabette_postprocess.checks.tags import (
    TAG_TRACKING_LOST,
    TAG_TRAJ_WITH_JUMPS,
    episode_tags,
    trajectory_tags,
)


def _report(*, n_tracked=1000, tracking_pct=99.0, n_jumps=0, median_angle_deg=0.0):
    return SimpleNamespace(n_tracked=n_tracked, tracking_pct=tracking_pct,
                           n_jumps=n_jumps, median_angle_deg=median_angle_deg)


def test_trajectory_tags_none_report_is_empty():
    """A missing report produces no tags (None-safe)."""
    assert trajectory_tags(None) == []


def test_trajectory_tags_clean_report_is_empty():
    """A healthy trajectory (high tracking, no jumps) produces no tags."""
    assert trajectory_tags(_report()) == []


def test_trajectory_tags_low_tracking_flags_lost():
    """Low tracking % or too few tracked frames flags tracking_lost."""
    assert trajectory_tags(_report(tracking_pct=40.0)) == [TAG_TRACKING_LOST]
    # Essentially no tracked frames also counts as lost.
    assert trajectory_tags(_report(n_tracked=1)) == [TAG_TRACKING_LOST]


def test_trajectory_tags_jump_rate_flags_jumps():
    """A jump count above 10% of tracked frames flags traj_with_jumps."""
    # n_jumps (200) > 10% of n_tracked (1000) -> jumpy.
    assert trajectory_tags(_report(n_jumps=200)) == [TAG_TRAJ_WITH_JUMPS]


def test_trajectory_tags_zigzag_flags_jumps_despite_good_tracking():
    """A reversing-direction zigzag flags traj_with_jumps even with good tracking."""
    # Few jumps relative to tracked frames, but reversing direction -> zigzag.
    tags = trajectory_tags(_report(n_jumps=6, median_angle_deg=120.0))
    assert tags == [TAG_TRAJ_WITH_JUMPS]


def test_trajectory_tags_stable_order():
    """Multiple issues emit tags in a stable order (tracking_lost before traj_with_jumps)."""
    tags = trajectory_tags(_report(tracking_pct=10.0, n_jumps=500))
    assert tags == [TAG_TRACKING_LOST, TAG_TRAJ_WITH_JUMPS]


def test_episode_tags_dedupes_and_is_none_safe(tmp_path):
    """episode_tags merges recording+trajectory tags, dedupes, and tolerates a None report."""
    # No angle_data.json in tmp_path -> no recording tags; report drives the rest.
    tags = episode_tags(tmp_path, _report(tracking_pct=10.0))
    assert tags == [TAG_TRACKING_LOST]
    assert episode_tags(tmp_path, None) == []
