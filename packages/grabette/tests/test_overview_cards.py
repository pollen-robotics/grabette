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
