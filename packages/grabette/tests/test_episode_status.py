"""The Status column on the Episodes page.

Read off the listing the task API already returns, so a table of fifty
episodes costs no extra call. What the rules must never do is call an episode
fine when its own counters say otherwise.
"""

from grabette.ui.app import _ep_header_html, _episode_status

GOOD = {
    "episode_id": "2026-09-21T14-32-05",
    "duration_seconds": 12.4,
    "frame_count": 372,
    "angle_sample_count": 610,
    "has_video": True,
    "metadata_ok": True,
}


def test_healthy_episode_is_ok():
    assert _episode_status(GOOD) == "✓ ok"


def test_unreadable_metadata_wins_over_everything():
    # The counters are zeros from a missing metadata.json, not measurements —
    # reporting "no camera" here would be inventing a fault.
    assert _episode_status({**GOOD, "metadata_ok": False}) == "✗ metadata"


def test_no_frames_is_a_camera_fault():
    assert _episode_status({**GOOD, "frame_count": 0}) == "✗ camera"
    assert _episode_status({**GOOD, "has_video": False}) == "✗ camera"


def test_no_angles_is_a_fault():
    assert _episode_status({**GOOD, "angle_sample_count": 0}) == "✗ angles"


def test_a_stray_press_is_flagged_but_not_a_fault():
    assert _episode_status({**GOOD, "duration_seconds": 0.8}) == "⚠ very short"


def test_missing_fields_do_not_crash():
    assert _episode_status({}) in {"✗ camera", "✗ metadata"}


# ── the header above the table ───────────────────────────────────────

def test_header_shows_the_description():
    assert "Cups into the bin" in _ep_header_html("Cups into the bin")


def test_header_is_empty_without_a_description():
    assert _ep_header_html() == ""


def test_header_says_when_the_api_is_down():
    # An empty list because the call failed must never read as "nothing
    # recorded" — that is the reading that gets an episode re-recorded.
    out = _ep_header_html(api_down=True)
    assert "Could not reach" in out
