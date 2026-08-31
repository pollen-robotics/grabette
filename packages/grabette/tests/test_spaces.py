"""Which fleet Space a device targets, and what it takes to move it.

The URL used to be a literal in three places, and the interesting failure was
never "wrong URL" — it was two of the three disagreeing: a device that registers
with one fleet while building its OAuth redirect_uri from another, which logs in
nowhere. So the pair is asserted together here, not just the value.

The other half is that prod must be what you get by DEFAULT and by accident: an
unset variable, a typo, a stray case. Only the exact word "test" moves a device
off production.
"""
from __future__ import annotations

import pytest

from grabette import spaces

PROD_FLEET = "https://pollen-robotics-grabette-fleet.hf.space"
TEST_FLEET = "https://pollen-robotics-grabette-fleet-test.hf.space"


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("GRABETTE_FLEET_ENV", raising=False)
    monkeypatch.delenv("GRABETTE_RELAY_URL", raising=False)
    yield


def test_the_default_is_production():
    assert spaces.fleet_env() == spaces.PROD
    assert spaces.fleet_url() == PROD_FLEET


def test_one_named_var_moves_the_device_to_test(monkeypatch):
    monkeypatch.setenv("GRABETTE_FLEET_ENV", "test")
    assert spaces.is_test()
    assert spaces.fleet_url() == TEST_FLEET


@pytest.mark.parametrize("value", ["TEST", " test ", "Test"])
def test_the_test_value_is_case_and_space_insensitive(monkeypatch, value):
    monkeypatch.setenv("GRABETTE_FLEET_ENV", value)
    assert spaces.fleet_url() == TEST_FLEET


@pytest.mark.parametrize("value", ["", "prod", "tset", "testing", "dev", "1", "true"])
def test_anything_that_is_not_test_is_production(monkeypatch, value):
    # A typo must not be what points a device at the development fleet — a day of
    # recordings would land where nobody looks for them, and the device would
    # give no sign of it.
    monkeypatch.setenv("GRABETTE_FLEET_ENV", value)
    assert spaces.fleet_url() == PROD_FLEET


def test_an_explicit_url_wins_over_the_env(monkeypatch):
    # How a duplicated Space is targeted (docs/source/spaces.md).
    monkeypatch.setenv("GRABETTE_FLEET_ENV", "test")
    monkeypatch.setenv("GRABETTE_RELAY_URL", "https://me-grabette-fleet.hf.space/")
    assert spaces.fleet_url() == "https://me-grabette-fleet.hf.space"


def test_an_empty_url_is_honoured_rather_than_falling_back_to_a_default(monkeypatch):
    # "" is a real setting (direct OAuth for local dev), not a missing one, so it
    # must NOT fall through to the derived Space URL. It does not by itself stop
    # the relay client — that is GRABETTE_RELAY_ENABLED, a different setting.
    monkeypatch.setenv("GRABETTE_RELAY_URL", "")
    assert spaces.fleet_url() == ""


def test_both_consumers_read_the_same_source():
    """THE bug the three duplicated literals invited: the relay the device polls
    and the Space its redirect_uri points at drifting apart, so a device
    registers with one fleet and logs in nowhere.

    Asserted by showing both derive from spaces.fleet_url() under the ambient
    environment — NOT by reloading the two modules under a patched env. auth.py
    resolves its URLs at import time, so a reload rebinds config.settings for
    every later test in the session, which is how this file used to break
    test_task.py. Since the other tests here pin how fleet_url() moves, one
    shared source is enough to prove the pair moves together."""
    from grabette import auth
    from grabette.config import Settings

    assert Settings().relay_url == spaces.fleet_url()
    assert auth.OAUTH_REDIRECT_URI == spaces.fleet_url() + "/oauth/grabette/callback"


def test_the_relay_default_is_computed_not_frozen(monkeypatch):
    """The field must keep a default_factory: a literal default would be baked in
    at class-creation time and GRABETTE_FLEET_ENV would silently stop working."""
    from grabette.config import Settings

    factory = Settings.model_fields["relay_url"].default_factory
    assert factory is not None
    monkeypatch.setenv("GRABETTE_FLEET_ENV", "test")
    assert factory() == TEST_FLEET


def test_no_shipped_default_points_at_a_test_space(monkeypatch):
    # The guarantee for main: with nothing set anywhere, no committed default can
    # reach a "-test" Space.
    assert "-test" not in spaces.fleet_url()
    assert "-test" not in spaces.space_url("fleet")
    assert "-test" not in spaces.space_url("slam")
