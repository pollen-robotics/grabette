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
}
GOOD_EPISODE = {"metadata_ok": True, "has_video": True}


def test_healthy_recording_passes():
    out = recording_summary(GOOD, GOOD_EPISODE, depth_on=True)
    assert out.startswith("### ✓")
    assert "2026-09-21T14-32-05" in out
    assert "12.4s" in out
    assert "372" in out
    assert "⚠" not in out


def test_transport_error_is_reported_verbatim():
    out = recording_summary({"error": "Not capturing"}, None, depth_on=True)
    assert out.startswith("### ✗")
    assert "Not capturing" in out


def test_no_episode_id_is_a_failure():
    out = recording_summary({"status": "cancelled"}, None, depth_on=True)
    assert out.startswith("### ✗")


def test_no_camera_frames_is_a_fault():
    out = recording_summary({**GOOD, "frame_count": 0}, GOOD_EPISODE, depth_on=True)
    assert out.startswith("### ✗")
    assert "No camera frames" in out


def test_no_angle_samples_is_a_fault():
    out = recording_summary(
        {**GOOD, "angle_sample_count": 0}, GOOD_EPISODE, depth_on=True
    )
    assert out.startswith("### ✗")
    assert "No angle samples" in out


def test_unreadable_metadata_is_a_fault():
    out = recording_summary(GOOD, {"metadata_ok": False, "has_video": True}, depth_on=True)
    assert out.startswith("### ✗")
    assert "metadata.json" in out


def test_depth_off_warns_but_still_passes():
    out = recording_summary(GOOD, GOOD_EPISODE, depth_on=False)
    assert out.startswith("### ✓")
    assert "depth camera was off" in out.lower()


def test_too_short_warns_but_still_passes():
    out = recording_summary({**GOOD, "duration_seconds": 0.4}, GOOD_EPISODE, depth_on=True)
    assert out.startswith("### ✓")
    assert "0.4s" in out


def test_unreadable_episode_is_not_held_against_it():
    """None means 'could not check', which is not evidence of a bad episode."""
    out = recording_summary(GOOD, None, depth_on=True)
    assert out.startswith("### ✓")


def test_missing_counters_do_not_crash():
    out = recording_summary({"episode_id": "x"}, None, depth_on=True)
    assert out.startswith("### ✗")  # zero frames and zero angles are both faults
