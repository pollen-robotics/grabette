"""A full SD card must cost one take, not the device.

Observed in the field: the card filled during a take, the RGB mux failed at
stop, and stop_capture raised half-way through its teardown. _stopping and
_capturing were never cleared, so the LED fast-blinked forever and only a second
press got the device out — leaving an episode with no metadata.json. Nothing
had refused the start either, although the card was all but full.
"""
import asyncio
import json
import types

import pytest

from grabette.backend import rpi
from grabette.output import write_json_atomic
from grabette.task import episode_issues


class _Speaker:
    def __init__(self):
        self.played = []

    def play_stop(self):
        self.played.append("stop")

    def play_error(self):
        self.played.append("error")

    def play_saved(self):
        self.played.append("saved")


class _FailingCamera:
    frame_count = 0

    def stop(self):
        raise RuntimeError("ffmpeg muxing failed: ... No space left on device")


class _Sync:
    def get_timestamp_ms(self):
        return 4200.0

    def reset(self):
        pass


def _capturing_backend(monkeypatch, tmp_path, camera=None, oakd=None):
    monkeypatch.setattr(rpi, "_free_bytes", lambda path: 50 * 1024**3)
    b = rpi.RpiBackend(enable_oakd=False)
    b._speaker = _Speaker()
    b._sync = _Sync()
    b._camera = camera or _FailingCamera()
    b._oakd = oakd
    b._reinit_hardware = lambda: None  # no picamera2 off the Pi
    ep = tmp_path / "ep"
    ep.mkdir()
    b._episode_dir = ep
    b._capturing = True
    return b, ep


def test_a_failed_teardown_still_ends_the_capture(monkeypatch, tmp_path):
    b, ep = _capturing_backend(monkeypatch, tmp_path)

    status = asyncio.run(b.stop_capture())

    # Back to idle: the LED stops fast-blinking, the next press starts a take.
    assert not b.is_capturing
    assert not b.is_stopping
    assert b._needs_reinit
    assert b._episode_dir is None
    # Returned, not raised: every caller files the episode like any other.
    assert status.episode_id == "ep"
    assert status.duration_seconds == 4.2
    assert b._speaker.played == ["stop", "error"]


def test_the_failed_episode_says_why(monkeypatch, tmp_path):
    b, ep = _capturing_backend(monkeypatch, tmp_path)

    asyncio.run(b.stop_capture())

    meta = json.loads((ep / "metadata.json").read_text())
    assert "No space left on device" in meta["stop_error"]
    assert episode_issues({"metadata_ok": True, **meta}) == ["stop_failed"]


def test_a_full_card_is_named_as_the_cause(monkeypatch, tmp_path):
    b, ep = _capturing_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(rpi, "_free_bytes", lambda path: 0)

    asyncio.run(b.stop_capture())

    meta = json.loads((ep / "metadata.json").read_text())
    assert meta["stop_error"].startswith("SD card almost full")
    # And the device now refuses the next take, rather than losing it too.
    assert "SD card almost full" in b.hardware_error


def test_a_failing_depth_mux_is_waited_for_and_survived(monkeypatch, tmp_path):
    class _Oakd:
        is_recording = True
        is_initialized = True

        def stop_recording(self):
            raise OSError(28, "No space left on device")

    class _OkCamera:
        frame_count = 10

        def stop(self):
            return [0.0, 21.7]

    b, ep = _capturing_backend(monkeypatch, tmp_path, camera=_OkCamera(), oakd=_Oakd())

    status = asyncio.run(b.stop_capture())

    assert not b.is_capturing and not b.is_stopping
    assert status.episode_id == "ep"
    assert "No space left" in json.loads((ep / "metadata.json").read_text())["stop_error"]


def test_an_unwritable_card_still_ends_the_capture(monkeypatch, tmp_path):
    b, ep = _capturing_backend(monkeypatch, tmp_path)

    def _full(path, payload):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(rpi, "write_json_atomic", _full)

    status = asyncio.run(b.stop_capture())

    assert not b.is_capturing and not b.is_stopping
    assert status.episode_id == "ep"
    assert b._speaker.played == ["stop", "error"]


# --- refusing to start on a nearly full card ----------------------------------

def test_a_nearly_full_card_blocks_capture(monkeypatch):
    monkeypatch.setattr(rpi, "_free_bytes", lambda path: int(0.5 * 1024**3))
    b = rpi.RpiBackend(enable_oakd=False)

    assert b.hardware_error.startswith("SD card almost full (0.5 GB free")
    with pytest.raises(RuntimeError, match="SD card almost full"):
        b.raise_if_capture_blocked()


def test_freeing_space_clears_the_fault(monkeypatch):
    free = {"bytes": 0}
    monkeypatch.setattr(rpi, "_free_bytes", lambda path: free["bytes"])
    b = rpi.RpiBackend(enable_oakd=False)
    assert b.hardware_error

    free["bytes"] = 10 * 1024**3
    b._disk_checked_at = None  # skip the cache delay

    assert b.hardware_error == ""


def test_the_disk_fault_comes_first_and_keeps_the_others(monkeypatch):
    monkeypatch.setattr(rpi, "_free_bytes", lambda path: 0)
    b = rpi.RpiBackend(enable_oakd=False)
    b._set_hw_error(rpi._HW_CAMERA, "RGB camera did not start")

    assert b.hardware_error.startswith("SD card almost full")
    assert b.hardware_error.endswith(" / RGB camera did not start")


def test_not_reported_mid_take(monkeypatch):
    # Crossing the threshold mid-take doesn't stop it from saving; the error
    # LED would only hide that it is still recording.
    monkeypatch.setattr(rpi, "_free_bytes", lambda path: 0)
    b = rpi.RpiBackend(enable_oakd=False)
    b._capturing = True

    assert b.hardware_error == ""


def test_an_unmeasurable_card_never_blocks(monkeypatch):
    monkeypatch.setattr(rpi, "_free_bytes", lambda path: None)
    b = rpi.RpiBackend(enable_oakd=False)

    assert b.hardware_error == ""


def test_free_bytes_measures_a_data_dir_not_created_yet(tmp_path):
    assert rpi._free_bytes(tmp_path / "not" / "yet") > 0


def test_threshold_comes_from_settings(monkeypatch):
    monkeypatch.setattr(rpi, "_free_bytes", lambda path: 4 * 1024**3)
    monkeypatch.setattr(rpi.settings, "min_free_disk_gb", 5.0)
    assert "5 GB needed" in rpi.RpiBackend(enable_oakd=False).hardware_error
    assert rpi._disk_space_message(4 * 1024**3, 5.0)
    assert rpi._disk_space_message(6 * 1024**3, 5.0) == ""


# --- the atomic write on a full card -----------------------------------------

def test_a_failed_atomic_write_leaves_no_temp_file(tmp_path):
    target = tmp_path / "metadata.json"

    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": object()})

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


# --- the dashboard -------------------------------------------------------------

from grabette.disk import storage_level  # noqa: E402
from grabette.ui.app import (  # noqa: E402
    _ov_checks, _ov_checks_html, _ov_health_card, _ov_problems_html,
    _storage_popup_card,
)

_FULL = "SD card almost full (0.3 GB free, 2 GB needed): free some space"


def test_storage_levels():
    assert storage_level(None, 2.0) == "unknown"
    assert storage_level(0.3, 2.0) == "full"
    assert storage_level(3.0, 2.0) == "low"
    assert storage_level(40.0, 2.0) == "ok"


def test_a_full_card_is_not_ready_to_record():
    cap = {"blocked_reason": _FULL, "storage_error": _FULL}
    cam = {"connected": True, "reinitializing": False}
    dcam = {"supported": True, "label": "Gemini 305", "connected": True, "error": ""}
    issues = _ov_checks(cap, cam, dcam, {"enabled": True, "error": ""})

    # A problem to fix, not "busy, wait": it does not clear on its own.
    assert issues == [("fail", None, _FULL)]
    out = _ov_checks_html(issues, blocked=True)
    assert "Cannot record" in out and "Ready to record" not in out
    assert out.count("✓") == 3  # every part is fine
    assert "1 problem to fix before recording" in _ov_problems_html(issues)
    assert "SD card almost full" in _ov_problems_html(issues)


def test_the_health_card_shows_space_left_and_flags_a_full_card():
    ok = _ov_health_card({"disk_free_gb": 40.0, "disk_total_gb": 58.0,
                          "disk_min_free_gb": 2.0})
    assert "Space left (card: 58 GB)" in ok and ">40.0 GB<" in ok
    assert "gb-health-ok" in ok

    low = _ov_health_card({"disk_free_gb": 3.0, "disk_total_gb": 58.0,
                           "disk_min_free_gb": 2.0})
    assert "gb-health-low" in low and "#f97316" in low

    full = _ov_health_card({"disk_free_gb": 0.0, "disk_total_gb": 58.0,
                            "disk_min_free_gb": 2.0})
    assert "⚠ 0.0 GB · full" in full
    assert "border:2px solid #ef4444" in full


def test_the_storage_popup_only_for_a_full_card():
    assert _storage_popup_card(None) == ""
    assert _storage_popup_card({"disk_free_gb": 3.0, "disk_min_free_gb": 2.0}) == ""
    card = _storage_popup_card({"disk_free_gb": 0.0, "disk_min_free_gb": 2.0})
    assert "SD card full" in card and "recording needs 2 GB" in card


def test_the_capture_status_carries_the_storage_fault(monkeypatch):
    monkeypatch.setattr(rpi, "_free_bytes", lambda path: 0)
    b = rpi.RpiBackend(enable_oakd=False)

    status = b.get_capture_status()

    assert status.storage_error.startswith("SD card almost full")
    assert status.blocked_reason.startswith("SD card almost full")
