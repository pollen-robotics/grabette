"""The Overview page's two info cards.

Pure builders, so the judgements they encode — what a missing reading shows,
when the battery turns red — are pinned here rather than read off a screenshot.
"""

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


def test_health_card_shows_readings():
    out = _ov_health_card(INFO)
    assert "76 %" in out
    assert "47.8 °C" in out
    assert "42.3 GB free of 118.0 GB" in out


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


# ── the account button ───────────────────────────────────────────────

from grabette.ui.app import _hf_button_html


def test_hf_button_names_the_account():
    out = _hf_button_html({"is_logged_in": True, "username": "pollen-robotics"})
    assert "pollen-robotics" in out
    assert 'href="/settings"' in out


def test_hf_button_invites_a_login_when_logged_out():
    out = _hf_button_html({"is_logged_in": False, "username": None})
    assert "Connect" in out


def test_hf_button_survives_no_answer():
    # hf_status() falls back to a logged-out dict, but None must not crash it.
    assert "Connect" in _hf_button_html(None)


def test_account_and_fleet_buttons_share_one_shape():
    from grabette.ui.app import _ERRAND_BUTTON, _FLEET_BUTTON_HTML

    # The two sit side by side; only their colour may differ.
    assert _ERRAND_BUTTON in _FLEET_BUTTON_HTML
    assert _ERRAND_BUTTON in _hf_button_html(None)
