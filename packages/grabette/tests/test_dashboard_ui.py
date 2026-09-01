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


# ── Section headings ──────────────────────────────────────────────────────

def test_section_heading_is_plain_html_and_escaped():
    """Markdown's block chrome clips a heading into a pill; ours is bare HTML."""
    out = ui._section("Network & <b>more</b>")
    assert out.startswith("<h2 class='gb-section'>")
    assert "<b>" not in out and "&amp;" in out


def test_markdown_block_chrome_is_stripped():
    """Gradio's inline `overflow: auto` on the block is what does the clipping."""
    css = _css()
    rule = re.search(r"\.block:has\(\[data-testid=\"markdown-wrapper\"\]\)\s*\{([^}]*)\}", css)
    assert rule, "markdown blocks are no longer neutralised"
    # !important is required: it has to beat the component's inline style.
    assert "overflow: visible !important" in rule.group(1)


# ── Theme switch ──────────────────────────────────────────────────────────

def test_theme_segments_set_their_own_mode():
    """Two buttons, not one toggle: picking the active one must be a no-op."""
    for mode in ("light", "dark"):
        js = ui._theme_set_js(mode)
        assert f"'{mode}'" in js
        assert "localStorage.setItem('grabette-theme'" in js
        assert "__grabetteApplyTheme" in js
    assert ui._theme_set_js("light") != ui._theme_set_js("dark")


def test_theme_choice_survives_navigation_and_honours_the_url():
    lib = ui._THEME_JS_LIB
    assert "localStorage" in lib          # kept across the navbar's plain links
    assert "__theme" in lib               # an explicit URL request still wins
    assert "prefers-color-scheme" in lib  # OS setting is the last fallback


# ── Fleet call to action ──────────────────────────────────────────────────

def test_fleet_cta_text_is_the_higher_contrast_choice():
    """White on this emerald→blue gradient bottoms out at 2.5:1; black at 5.6:1.

    Recomputed here rather than trusted, so a change to the gradient that makes
    white viable (or black worse) fails instead of passing silently.
    """
    rule = re.search(r"\.gb-fleet\s*\{([^}]*)\}", _css()).group(1)
    stops = re.findall(r"#([0-9a-f]{6})", rule)
    assert len(stops) >= 2, "expected a two-stop gradient"

    def luminance(hexstr):
        parts = [int(hexstr[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in parts]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    def ratio(a, b):
        hi, lo = max(a, b), min(a, b)
        return (hi + 0.05) / (lo + 0.05)

    # Sample the sweep, not just the stops: the midpoint is its own worst case.
    a, b = [tuple(int(s[i:i + 2], 16) for i in (0, 2, 4)) for s in stops[:2]]
    samples = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        mix = "".join(f"{round(a[i] + (b[i] - a[i]) * t):02x}" for i in range(3))
        samples.append(luminance(mix))

    worst_black = min(ratio(lum, 0.0) for lum in samples)
    worst_white = min(ratio(lum, 1.0) for lum in samples)
    assert worst_black > worst_white
    assert worst_black >= 4.5, f"AA needs 4.5:1, got {worst_black:.2f}"
    assert "color: #000" in rule
