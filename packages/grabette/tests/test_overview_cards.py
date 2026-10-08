"""The Overview page's two info cards.

Pure builders, so the judgements they encode — what a missing reading shows,
when the battery turns red — are pinned here rather than read off a screenshot.
"""

from grabette.config import settings
from grabette.ui.app import _ov_device_card, _ov_health_card

INFO = {
    "hostname": "grabette-01",
    "ip": "192.168.1.42",
    "cpu_temp_c": 47.8,
    "disk_free_gb": 42.3,
    "disk_total_gb": 118.0,
    "battery_pct": 76,
    "battery_charging": False,
}
WIFI = {"ssid": "pollen-lab", "ip": "192.168.1.42"}


def test_device_card_shows_identity():
    out = _ov_device_card(INFO, WIFI)
    assert "grabette-01" in out
    assert "192.168.1.42" in out
    assert "pollen-lab" in out


def test_device_card_survives_a_silent_device():
    # get_system_info() returns None when the API cannot be reached; the card
    # must say "we don't know", never crash or invent a hostname.
    out = _ov_device_card(None, None)
    assert "—" in out


def test_wifi_ip_wins_over_the_system_one():
    out = _ov_device_card({**INFO, "ip": "10.0.0.1"}, WIFI)
    assert "192.168.1.42" in out
    assert "10.0.0.1" not in out


def test_device_card_shows_the_configured_side(monkeypatch):
    monkeypatch.setattr(settings, "hand", "left")
    assert "Left" in _ov_device_card(INFO, WIFI)


def test_health_card_shows_readings():
    out = _ov_health_card(INFO)
    assert "76 %" in out
    assert "47.8 °C" in out
    assert "75.7 / 118.0 GB used" in out


def test_battery_colours_follow_the_level():
    assert "#22c55e" in _ov_health_card({**INFO, "battery_pct": 76})
    assert "#f97316" in _ov_health_card({**INFO, "battery_pct": 30})
    assert "#ef4444" in _ov_health_card({**INFO, "battery_pct": 12})
    # Plugged in is fine at any level.
    assert "#22c55e" in _ov_health_card(
        {**INFO, "battery_pct": 12, "battery_charging": True})


def test_charging_is_marked():
    assert "⚡" in _ov_health_card({**INFO, "battery_charging": True})


def test_health_card_survives_missing_readings():
    out = _ov_health_card({})
    assert out.count("—") == 3


# ── the account slab ─────────────────────────────────────────────────

from grabette.webauth import widget_page


def test_button_variant_is_one_slab():
    out = widget_page(variant="button")
    # The card chrome goes, the OAuth button becomes the whole widget, and the
    # token field folds away behind a disclosure.
    assert "button.oauth{width:100%" in out
    assert "or use a token" in out


def test_button_variant_keeps_the_one_login_implementation():
    # Same card and script as Settings — only the skin differs.
    from grabette.webauth import LOGIN_CARD

    assert LOGIN_CARD in widget_page(variant="button")
    assert LOGIN_CARD in widget_page()


def test_other_skins_are_untouched():
    assert "button.oauth{width:100%" not in widget_page()
    assert "button.oauth{width:100%" not in widget_page(compact=True)


# ── Pre-recording checks ──────────────────────────────────────────────

from grabette.ui.app import (  # noqa: E402
    _ov_checks, _ov_checks_html, _ts_calib_html, _ts_depth_html,
    _ts_first_section, _ts_tabs_css,
)

CAP = {"is_capturing": False, "blocked_reason": "", "needs_calibration": False}
CAM = {"connected": True, "reinitializing": False}
DCAM = {"supported": True, "enabled": False, "initialized": False,
        "initializing": False, "label": "Gemini 305", "connected": True,
        "error": ""}
ANGLE = {"enabled": True, "initialized": True, "error": ""}


def test_a_healthy_device_is_ready():
    issues = _ov_checks(CAP, CAM, DCAM, ANGLE)
    assert issues == []
    assert "Ready to record" in _ov_checks_html(issues)


def test_a_silent_device_is_unknown_not_healthy():
    assert _ov_checks(None, None, None, None) is None
    assert "gb-checks-unknown" in _ov_checks_html(None)


def test_every_part_at_fault_is_named_with_its_section():
    issues = _ov_checks(
        {**CAP, "blocked_reason": "x", "needs_calibration": True},
        {"connected": False, "reinitializing": False},
        {**DCAM, "connected": False},
        {**ANGLE, "error": "the gripper angle sensors stopped answering"})
    # The fault is the angle sensors; the calibration they cannot do waits.
    assert [(lvl, sec) for lvl, sec, _ in issues] == [
        ("fail", "rgb"), ("fail", "depth"), ("fail", "angle"),
        ("warn", "calib")]
    out = _ov_checks_html(issues)
    assert "3 problems to fix before recording" in out
    assert "gb-checks-fail" in out


def test_the_depth_cameras_own_fault_is_quoted():
    issues = _ov_checks(CAP, CAM, {**DCAM, "error": "calibration unreadable"},
                        ANGLE)
    assert issues == [("fail", "depth", "Gemini 305: calibration unreadable")]
    assert "calibration unreadable" in _ts_depth_html(
        {**DCAM, "error": "calibration unreadable"})


def test_a_busy_device_only_waits():
    # An upload refuses the recording, but no part is at fault.
    issues = _ov_checks({**CAP, "blocked_reason": "an upload is running"},
                        CAM, DCAM, ANGLE)
    assert issues == [("warn", None,
                       "Cannot record right now — an upload is running")]
    assert "gb-checks-warn" in _ov_checks_html(issues)


def test_troubleshooting_opens_on_the_first_part_at_fault():
    assert _ts_first_section([]) == "rgb"
    assert _ts_first_section(None) == "rgb"
    assert _ts_first_section([("warn", "rgb", ""), ("fail", "calib", "")]) == "calib"
    assert _ts_first_section([("warn", None, "busy")]) == "rgb"


def test_tabs_take_the_colour_of_their_part():
    css = _ts_tabs_css([("fail", "angle", ""), ("warn", "rgb", "")])
    assert '[data-tab-id="angle"]{--gb-tab-c:#ef4444;--gb-tab-m:"✗";}' in css
    assert '[data-tab-id="rgb"]{--gb-tab-c:#f59e0b;' in css
    assert '[data-tab-id="depth"]{--gb-tab-c:#10b981;--gb-tab-m:"✓";}' in css
    assert '[data-tab-id="depth"]{--gb-tab-c:#94a3b8;' in _ts_tabs_css(None)


def test_an_unplugged_depth_camera_is_not_ready():
    # Nothing past "Plugged in" can pass while it is unplugged.
    out = _ts_depth_html({**DCAM, "connected": False})
    assert "#10b981" not in out
    assert "depth-camera-cable.gif" in out
    assert "no need to restart" in out
    assert "depth-camera-cable.gif" not in _ts_depth_html(DCAM)


def test_calibration_waits_for_the_angle_sensors():
    out = _ts_calib_html({"needs_calibration": True},
                         {**ANGLE, "error": "down"})
    assert "Fix the angle sensors first" in out
    assert "not calibrated" in _ts_calib_html({"needs_calibration": True},
                                              ANGLE)
