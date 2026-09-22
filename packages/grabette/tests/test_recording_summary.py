"""Rules behind the Test Recording verdict.

The counters come back from the device; what counts as a usable episode is a
judgement made here, and it is the part worth pinning down.
"""

from grabette.ui.summary import recording_summary

GOOD = {
    "episode_id": "2026-09-21T14-32-05",
    "duration_seconds": 12.4,
    "frame_count": 372,
    "angle_sample_count": 610,
    "imu_sample_count": 2480,
    "metadata_ok": True,
    "has_video": True,
}
COMPLETE = {"missing": [], "complete": True}
NO_DEPTH = {"missing": ["dcam_depth.mkv", "dcam_left.mp4"], "complete": False}


def test_healthy_recording_passes():
    out = recording_summary(GOOD, COMPLETE)
    assert "✓ Recording looks good" in out
    assert "2026-09-21T14-32-05" in out
    assert "12.4s" in out
    assert "372" in out
    assert "⚠" not in out


def test_no_camera_frames_is_a_fault():
    out = recording_summary({**GOOD, "frame_count": 0}, COMPLETE)
    assert "✗ Something is wrong" in out
    assert "RGB camera recorded nothing" in out


def test_no_angle_samples_is_a_fault():
    out = recording_summary({**GOOD, "angle_sample_count": 0}, COMPLETE)
    assert "✗ Something is wrong" in out
    assert "angle sensors recorded nothing" in out


def test_missing_depth_files_are_a_fault():
    """The whole reason the check route exists: an episode can have a perfect
    RGB video, perfect angles, and no RGB-D data at all."""
    out = recording_summary(GOOD, NO_DEPTH)
    assert "✗ Something is wrong" in out
    assert "no RGB-D data" in out
    assert "dcam_left.mp4" in out


def test_a_long_missing_list_is_summarised_not_recited():
    """Every depth artifact is missing whenever the camera wrote nothing, and
    seven filenames in a row say no more than one of them does."""
    out = recording_summary(GOOD, {"missing": [f"dcam_{i}.json" for i in range(7)],
                                   "complete": False})
    assert "and 5 more" in out
    assert "dcam_6.json" not in out


def test_non_depth_gaps_do_not_claim_the_depth_camera_failed():
    out = recording_summary(GOOD, {"missing": ["angle_data.json"], "complete": False})
    assert "no RGB-D data" not in out


def test_unreadable_metadata_is_a_fault():
    out = recording_summary({**GOOD, "metadata_ok": False}, COMPLETE)
    assert "✗ Something is wrong" in out
    assert "metadata.json" in out


def test_missing_video_file_is_a_fault():
    out = recording_summary({**GOOD, "has_video": False}, COMPLETE)
    assert "✗ Something is wrong" in out


def test_too_short_warns_but_still_passes():
    out = recording_summary({**GOOD, "duration_seconds": 0.4}, COMPLETE)
    assert "✓ Recording looks good" in out
    assert "0.4s" in out
    assert "⚠" in out


def test_unchecked_episode_is_not_held_against_it():
    """None means 'could not check', which is not evidence of a bad episode."""
    out = recording_summary(GOOD, None)
    assert "✓ Recording looks good" in out
    assert "not checked" in out


def test_unreadable_episode_reports_that_and_nothing_else():
    out = recording_summary(None, None)
    assert "Could not read the episode" in out
    assert "Something is wrong" not in out


def test_missing_counters_do_not_crash():
    out = recording_summary({"episode_id": "x"}, COMPLETE)
    assert "✗ Something is wrong" in out  # zero frames and zero angles
