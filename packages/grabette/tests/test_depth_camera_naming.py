"""The dashboard must name whichever depth camera is configured.

Before this, "OAK-D" was hardcoded in the status badge, the toggle button and
the 501 messages, so a device running an Orbbec Gemini 305 told the operator it
was talking to an OAK-D. That is the kind of wrong-but-plausible label that
costs someone an hour when two cameras behave differently.

The label is served by the API (`/api/oakd/status` -> `label`) rather than mapped
in the UI, so there is one authoritative name. These tests pin both ends.
"""
import pytest

from grabette.hardware.depth_camera import (
    GENERIC_DISPLAY_NAME,
    display_name,
)


# ── the mapping itself ───────────────────────────────────────────────────────

def test_display_name_known_models():
    assert display_name("oakd") == "OAK-D"
    assert display_name("gemini305") == "Gemini 305"


@pytest.mark.parametrize("value", [None, "", "some-future-camera"])
def test_display_name_falls_back(value):
    # A model added to config but not to DISPLAY_NAMES, or a backend that cannot
    # say which camera it has, must still yield a usable label rather than
    # None/KeyError leaking into the UI.
    assert display_name(value) == GENERIC_DISPLAY_NAME


def test_every_config_choice_has_a_display_name():
    # Guards the real drift risk: adding a camera to the config Literal without
    # giving it a name, which would silently show "Depth camera" forever.
    from typing import get_args

    from grabette.config import Settings
    from grabette.hardware.depth_camera import DISPLAY_NAMES

    choices = get_args(Settings.model_fields["depth_camera"].annotation)
    missing = [c for c in choices if c not in DISPLAY_NAMES]
    assert not missing, f"depth_camera values with no display name: {missing}"


# ── what the API serves ──────────────────────────────────────────────────────

class _Backend:
    """Minimal stand-in for RpiBackend's depth-camera surface."""

    def __init__(self, model):
        self.depth_camera_model = model
        self.is_oakd_enabled = True
        self.is_oakd_initialized = True
        self.is_oakd_initializing = False

    async def set_oakd_enabled(self, on):  # presence => "supported"
        pass


@pytest.mark.parametrize("model,expected", [
    ("oakd", "OAK-D"),
    ("gemini305", "Gemini 305"),
])
def test_status_reports_model_and_label(model, expected):
    from grabette.app.routers.oakd import _status

    s = _status(_Backend(model))
    assert s["model"] == model
    assert s["label"] == expected
    # The pre-existing keys must survive — the UI polls all of them.
    for key in ("supported", "enabled", "initialized", "initializing"):
        assert key in s


def test_status_label_is_usable_without_a_model():
    # MockBackend has no depth-camera surface at all.
    from grabette.app.routers.oakd import _status

    class Bare:
        pass

    s = _status(Bare())
    assert s["supported"] is False
    assert s["model"] is None
    assert s["label"] == GENERIC_DISPLAY_NAME


# ── what the operator actually sees ──────────────────────────────────────────

@pytest.mark.parametrize("label", ["OAK-D", "Gemini 305"])
def test_status_bar_badge_uses_the_served_label(label):
    # gradio is an optional [ui] extra; ui.app imports it at module level.
    pytest.importorskip("gradio")
    from grabette.ui.app import _status_bar_html

    html = _status_bar_html(
        None,
        {"supported": True, "initialized": True, "label": label},
        None,
    )
    assert label in html
    assert "Connected" in html


def test_status_bar_badge_without_a_label_is_still_named():
    pytest.importorskip("gradio")
    from grabette.ui.app import _status_bar_html

    # An older daemon (or a failed call) returns no label; the badge must not
    # render "None" at the operator.
    html = _status_bar_html(None, {"supported": True, "initialized": True}, None)
    assert "None" not in html
    assert GENERIC_DISPLAY_NAME in html


def test_status_bar_does_not_hardcode_oakd_for_a_gemini():
    # The regression this whole file exists for.
    pytest.importorskip("gradio")
    from grabette.ui.app import _status_bar_html

    html = _status_bar_html(
        None,
        {"supported": True, "initialized": True, "label": "Gemini 305"},
        None,
    )
    assert "OAK-D" not in html
