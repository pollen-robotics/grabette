"""Pure rendering helpers of the dashboard — no Gradio app, no network.

Two things these hold in place. The battery reading has to appear in the SAME
place on every page (the shared header) and nowhere else, and every colour has
to be a theme variable — a literal hex is exactly what breaks when the operator
flips to the light theme.
"""

from __future__ import annotations

import re

import pytest

from grabette.ui import app as ui


def _hex_colors(markup: str) -> list[str]:
    """Hardcoded colours in generated markup — the light-theme hazard."""
    return re.findall(r"#[0-9a-fA-F]{3,8}\b", markup)


def _css() -> str:
    """APP_CSS with comments stripped, so prose about a selector is not a rule."""
    return re.sub(r"/\*.*?\*/", "", ui.APP_CSS, flags=re.S)


def _block(css: str, selector: str) -> dict[str, str]:
    """The --gb-* declarations of the first rule matching `selector`."""
    body = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css).group(1)
    return dict(re.findall(r"(--gb-[a-z-]+):\s*([^;]+);", body))


# ── The header: same device name + battery on every page ──────────────────

def test_header_shows_hostname_and_battery():
    out = ui.page_header_html({"hostname": "R-grabette", "battery_pct": 62})
    assert "R-grabette" in out
    assert "62 %" in out


def test_header_marks_charging():
    out = ui.page_header_html({"hostname": "h", "battery_pct": 9, "battery_charging": True})
    assert "⚡ 9 %" in out
    # Charging outranks the level: a plugged-in device must not read as critical.
    assert ui.C_BAD not in out


def test_header_survives_a_failed_system_info_call():
    """None is 'the daemon did not answer', not 'the battery is at 0'."""
    out = ui.page_header_html(None)
    assert "N/A" in out
    assert "%" not in out


def test_header_escapes_the_hostname():
    out = ui.page_header_html({"hostname": "<script>x</script>"})
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_header_uses_no_hardcoded_colors():
    assert _hex_colors(ui.page_header_html({"hostname": "h", "battery_pct": 50})) == []


# ── Battery colours ───────────────────────────────────────────────────────

@pytest.mark.parametrize("pct,charging,expected", [
    (95, False, "C_OK"),
    (41, False, "C_OK"),
    (40, False, "C_WARN"),
    (21, False, "C_WARN"),
    (20, False, "C_BAD"),
    (8, False, "C_BAD"),
    (8, True, "C_OK"),  # charging always reads OK, whatever the level
])
def test_battery_colors(pct, charging, expected):
    assert ui._battery_colors(pct, charging)[0] == getattr(ui, expected)


# ── The Episodes strip: cameras only, battery lives in the header ─────────

def test_episode_strip_has_no_battery():
    out = ui._status_bar_html(
        {"supported": True, "initialized": True},
        {"connected": True},
    )
    assert "Battery" not in out
    assert "RGB Camera" in out and "OAK-D" in out


def test_episode_strip_reports_a_hardware_fault_over_a_power_state():
    """The fault is the one state that blocks recording; "Off" must not mask it."""
    out = ui._status_bar_html(
        {"supported": True, "hardware_error": "USB link down", "enabled": False},
        {"connected": True},
    )
    assert "Cannot record" in out
    assert "USB link down" in out  # the full reason, as the hover tooltip


def test_episode_strip_says_na_when_the_calls_failed():
    out = ui._status_bar_html(None, None)
    assert out.count("N/A") == 2


def test_episode_strip_uses_no_hardcoded_colors():
    out = ui._status_bar_html({"supported": True, "initialized": True}, {"connected": True})
    assert _hex_colors(out) == []


# ── Cards ─────────────────────────────────────────────────────────────────

def test_info_card_escapes_its_tooltip():
    out = ui._info_card("OAK-D", "Cannot record", title='he said "boom" & <b>')
    assert "&quot;boom&quot;" in out
    assert "<b>" not in out


def test_card_row_reflows_via_the_stylesheet_not_inline_widths():
    """Layout has to live in APP_CSS so the phone media query can override it."""
    row = ui._card_row([ui._info_card("A", "1"), ui._info_card("B", "2")])
    assert "gb-cards" in row and row.count("gb-card'") == 2
    assert "flex:1" not in row.replace(" ", "")


# ── The palette itself ────────────────────────────────────────────────────

def test_every_palette_token_is_defined_in_both_themes():
    """A var() with no dark value would silently keep its light colour."""
    css = _css()
    # The palette pair specifically — a later :root block holds layout
    # constants (--gb-measure, --gb-gutter) that are theme-independent.
    light = _block(css, ":root")
    dark = _block(css, "body.dark")
    assert light and light.keys() == dark.keys()

    # Every C_* the module renders with must resolve to one of them.
    used = {v for k, v in vars(ui).items() if k.startswith("C_") and isinstance(v, str)}
    for ref in used:
        name = re.fullmatch(r"var\((--gb-[a-z-]+)\)", ref)
        assert name, f"{ref} is not a palette variable"
        assert name.group(1) in light, f"{ref} is not defined in APP_CSS"


# ── Layout shell ──────────────────────────────────────────────────────────

def test_nav_is_styled_through_the_real_element_not_the_hidden_stub():
    """gr.Navbar's elem_id lands on a display:none placeholder in Gradio 6.

    The visible bar is <nav> inside .nav-holder, a sibling of the content
    column. Styling #grabette-nav silently does nothing — which is how the
    Power Off tint and the bar's own background went missing for a release.
    """
    css = _css()
    assert ".nav-holder nav" in css
    for rule in re.findall(r"([^{}]*)\{", css):
        if "#grabette-nav" in rule:
            raise AssertionError(f"styles the hidden stub: {rule.strip()!r}")


def test_content_is_capped_while_the_shell_is_full_bleed():
    """The bar spans the window; the content does not, or nothing is readable."""
    css = _css()
    assert "--gb-measure" in css
    # Gradio's own responsive cap has to be lifted, or the bar stops short.
    assert ".fillable:not(.fill_width)" in css
    # ...and <main> needs a width, not just a max-width: its parent centres
    # children with align-items, so without one it is shrink-to-fit.
    main_rule = re.search(r"main\.contain\s*\{([^}]*)\}", css).group(1)
    assert "width: 100%" in main_rule
    assert "max-width: var(--gb-measure)" in main_rule


def test_phone_breakpoint_collapses_the_card_grid():
    phone = _css().split("@media (max-width: 700px)", 1)[1]
    assert "grid-template-columns: 1fr" in phone
