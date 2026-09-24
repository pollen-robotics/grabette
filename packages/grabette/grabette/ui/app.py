"""Gradio dashboard for Grabette — camera view, capture controls, task/session/episode management."""

from __future__ import annotations

import html
import io
import logging
import math
from urllib.parse import quote

import gradio as gr
from PIL import Image

from grabette.config import settings
from grabette.ui.api_client import GrabetteClient
from grabette.ui.summary import recording_summary

logger = logging.getLogger(__name__)


MODAL_CSS = """
#hf-auth-modal {
    position: fixed !important;
    inset: 0 !important;
    background: rgba(0, 0, 0, 0.78) !important;
    z-index: 9999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    padding: 1rem !important;
    border-radius: 0 !important;
    border: none !important;
    gap: 0 !important;
}
#hf-auth-card {
    max-width: 460px !important;
    width: 100% !important;
    background: #1f2937 !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6) !important;
    border: 1px solid #374151 !important;
}
/* Title bar: GRABETTE on the left, the power-off button on the right. */
.gb-titlebar {
    align-items: center !important;
    justify-content: space-between !important;
    flex-wrap: nowrap !important;
    gap: 1rem !important;
}
/* Black rather than red, so the ⛔ stands out on it. */
.gb-titlebar .gb-power {
    background: #000 !important;
    border-color: #000 !important;
    color: #fff !important;
    font-size: 1.15rem !important;
    padding: .7rem 1.4rem !important;
}
.gb-titlebar .gb-power:hover {
    background: #333 !important;
}
.gb-titlebar > * {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
}
/* Episodes: a table that reads as a table, and buttons the size of their labels instead of four stretched bars. */
/* The task menu is sized to a task name on a laptop, full width on a phone. */
#ep-page .ep-task-menu {
    width: 100% !important;
    max-width: 420px !important;
    margin: 0 auto 0 0 !important;
    align-self: flex-start !important;
}
#ep-page .ep-actions {
    justify-content: flex-start !important;
    align-items: center !important;
    gap: .5rem !important;
}
#ep-page .ep-actions button {
    width: auto !important;
    flex: 0 0 auto !important;
}
@media (max-width: 560px) {
    #ep-page .ep-actions button {
        width: 100% !important;
    }
}
/* Network: one errand, read top to bottom in a column rather than a form
   stretched across a laptop screen. */
#nw-page {
    max-width: 640px !important;
    width: 100% !important;
    margin: 0 auto !important;
}
/* Overview: the call to action is one button, so it is sized to its label and
   centred rather than stretched across the page. */
#ov-page .ov-cta {
    justify-content: center !important;
}
/* gr.Button(link=...) renders an <a>, not a <button> — both are named here so
   the rule survives whichever gradio picks. */
#ov-page .ov-cta button,
#ov-page .ov-cta a {
    width: auto !important;
    flex: 0 0 auto !important;
    background: #1a1a2e !important;
    border-color: #1a1a2e !important;
    color: #fff !important;
}
#ov-page .ov-cta button:hover,
#ov-page .ov-cta a:hover {
    background: #2a2a4a !important;
}
/* The camera toggle is a segmented control, not two loose radio dots: the
   native inputs are hidden and each label becomes half of one pill, centred
   under the preview it belongs to. :has() is what lights the chosen half. */
#ov-page .ov-toggle {
    display: flex !important;
    justify-content: center !important;
    gap: 0 !important;
    margin: .75rem 0 0 !important;
    border: none !important;
    background: none !important;
}
#ov-page .ov-toggle label {
    margin: 0 !important;
    padding: .35rem 1.1rem !important;
    border: 1px solid var(--border-color-primary) !important;
    background: var(--background-fill-primary) !important;
    color: var(--body-text-color-subdued) !important;
    font-size: var(--button-small-text-size) !important;
    font-weight: 600 !important;
    cursor: pointer;
    box-shadow: none !important;
}
#ov-page .ov-toggle label:first-child {
    border-radius: 999px 0 0 999px !important;
}
#ov-page .ov-toggle label:last-child {
    border-radius: 0 999px 999px 0 !important;
    border-left: none !important;
}
#ov-page .ov-toggle label input {
    display: none !important;
}
#ov-page .ov-toggle label:has(input:checked) {
    background: var(--button-primary-background-fill) !important;
    border-color: var(--button-primary-background-fill) !important;
    color: var(--button-primary-text-color) !important;
}
/* gradio pads an HTML block but not an Image, which started the cards and the
   viewer 10px below the camera and threw the whole row off its baseline. */
#ov-page .ov-tiles .html-container,
#ov-page .ov-errands .html-container {
    padding: 0 !important;
}
/* The button under a tile sits on the same line as the toggle under the
   camera — same top margin, centred the same way. */
#ov-page .ov-tile-btn {
    margin-top: .75rem !important;
    justify-content: center !important;
}
#ov-page .ov-tile-btn button,
#ov-page .ov-tile-btn a {
    width: auto !important;
    flex: 0 0 auto !important;
}
#ov-page .ov-note p {
    font-size: .8rem !important;
    text-align: center !important;
    margin: .35rem 0 0 !important;
}
@media (max-width: 560px) {
    /* A phone shows one column, so a full-width button is the easy target. */
    #ov-page .ov-cta button,
    #ov-page .ov-cta a {
        width: 100% !important;
    }
}
/* Live dot on the Test Recording status pill (see _tr_pill). */
@keyframes grabette-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.25; }
}
/* Test Recording is a procedure to read, not a dashboard to scan: a column
   narrow enough for the eye to fall down it, centred, with the steps as
   rounded cards. Full width put the two button animations metres apart. */
#tr-page {
    max-width: 560px !important;
    width: 100% !important;
    margin: 0 auto !important;
}
#tr-page .grabette-step {
    border-radius: 18px !important;
    padding: 1.15rem 1.3rem !important;
    overflow: hidden !important;
}
/* Buttons the width of their label, left-aligned, instead of stretching to the
   card edge — a full-width button reads as the step itself. */
#tr-page button {
    width: auto !important;
    min-width: 0 !important;
    flex: 0 0 auto !important;
    align-self: flex-start !important;
}
/* The secondary button next to it still has to look clickable on the grey
   card, where gradio's default is nearly the card's own colour. */
#tr-page button.secondary {
    border: 1px solid var(--button-primary-background-fill, #3b82f6) !important;
    color: var(--button-primary-background-fill, #3b82f6) !important;
    font-weight: 600 !important;
}
#tr-page .grabette-step .row {
    justify-content: flex-start !important;
}
/* Only the action row is centred: its two controls must sit on one line even
   when one of them wraps. Rows of figures stay top-aligned, so the animations
   line up whatever their captions do. */
#tr-page .tr-actions {
    align-items: center !important;
}
/* The download link is a gr.HTML standing next to a gr.Button; the html
   container's own padding is what set it 10px lower than the button. */
#tr-page .tr-dl .html-container {
    padding: 0 !important;
}
/* Phone: a label-width button is a thumb target too small, so give it the
   card back. */
@media (max-width: 560px) {
    #tr-page .grabette-step {
        padding: 1rem .9rem !important;
    }
    #tr-page button,
    #tr-page a[download] {
        width: 100% !important;
    }
}
"""

# The two errands at the bottom of the Overview are one shape in two colours:
# same height, same weight, same radius, so neither looks like the important
# one. Both are plain links out — grabette-fleet is OAuth-gated and this
# dashboard is served over plain HTTP, so its login cannot render in an iframe
# here, and the HF login card lives on Settings.
_ERRAND_BUTTON = (
    'display:flex;align-items:center;justify-content:center;gap:.5rem;'
    'height:56px;padding:0 1rem;border-radius:9px;box-sizing:border-box;'
    'text-align:center;text-decoration:none;color:#fff;font-weight:700;'
    'box-shadow:0 4px 14px rgba(0,0,0,.22);'
)

_FLEET_BUTTON_HTML = (
    '<a href="{url}" target="_blank" rel="noopener" '
    f'style="{_ERRAND_BUTTON}'
    'background:linear-gradient(135deg,#10b981,#3b82f6);">'
    'Open fleet dashboard ↗</a>'
)


# The account slab on the Overview: the login widget under its button skin, so
# the OAuth round trip, the token fallback and logout all happen right there
# rather than sending someone to Settings. The onload loop keeps the iframe as
# tall as its content, which grows when the token field is unfolded.
_HF_AUTH_BUTTON_IFRAME = (
    '<iframe src="/api/hf-auth/widget?variant=button" scrolling="no"'
    ' onload="var f=this;(function r(){'
    'if(!document.contains(f))return;'
    'try{f.style.height=f.contentDocument.body.scrollHeight+2+\'px\';}catch(e){}'
    'setTimeout(r,400);})()"'
    ' style="width:100%;border:none;min-height:56px;display:block;"></iframe>'
)


# Test Recording shows the grabette's own button being pressed instead of
# offering start/stop buttons of its own: the physical button is how a recording
# is made in the field, and this is the page whose job is to teach that.
# Served from grabette/ui/assets, mounted at /ui-assets (NOT /assets, which is
# Gradio's own bundle); onerror keeps the step readable without the file.
def _button_gif(filename: str, caption: str) -> str:
    return (
        '<figure style="margin:0 auto;width:100%;">'
        f'<img src="/ui-assets/{filename}" alt="{html.escape(caption)}"'
        ' style="width:100%;aspect-ratio:1;object-fit:cover;display:block;'
        'border-radius:14px;background:#0f172a;"'
        ' onerror="this.style.display=\'none\';'
        'this.nextElementSibling.style.display=\'flex\';">'
        '<div style="display:none;align-items:center;justify-content:center;'
        'aspect-ratio:1;border-radius:14px;color:#94a3b8;font-size:.78rem;'
        'text-align:center;padding:.5rem;'
        'border:1px dashed var(--border-color-primary,#cbd5e1);">'
        'Animation coming soon</div>'
        '<figcaption style="margin-top:.5rem;text-align:center;font-size:.85rem;'
        'font-weight:600;color:var(--body-text-color);">'
        f'{html.escape(caption)}</figcaption></figure>'
    )


# Step 2's download is a plain link to the API's own archive route: the click
# lands straight in the browser's downloads. Going through a gradio File meant
# copying the whole archive onto the device's disk first, then a second click
# on a box that was empty until then.
#
# Dressed in gradio's own button variables rather than fixed sizes, so it comes
# out the same height, font and radius as the button it stands next to —
# including inside a Group, which squares the corners and drops the border.
_DL_BASE = (
    'display:inline-flex;align-items:center;justify-content:center;'
    'box-sizing:border-box;text-decoration:none;border:none;'
    'padding:var(--button-small-padding);'
    'font-size:var(--button-small-text-size);'
    'font-weight:var(--button-small-text-weight);'
    'line-height:var(--line-md);'
    'border-radius:var(--button-small-radius);'
    # The same fill as the button it stands beside, so the two read as a pair
    # in both states; gradio dims a disabled button to 0.5 opacity, and the
    # greyed-out link copies that rather than inventing its own look.
    'background:var(--button-primary-background-fill);'
    'color:var(--button-primary-text-color);'
)


def _download_link_html(episode_id: str | None) -> str:
    if not episode_id:
        return (f'<span style="{_DL_BASE}opacity:.5;cursor:not-allowed;">'
                'Download (.tar.gz)</span>')
    return (f'<a href="/api/episodes/{quote(episode_id)}/download" download '
            f'style="{_DL_BASE}">Download (.tar.gz)</a>')


def _step_header(number: int, title: str, hint: str = "") -> str:
    """A numbered step's heading — one line, no gradio Markdown heading.

    gr.Markdown("### 1 · …") inside a Group renders with the margins of a
    document heading, which is what made the steps tall enough to scroll.
    """
    sub = (
        '<div style="font-size:.88rem;opacity:.75;margin-top:.15rem;">'
        f'{html.escape(hint)}</div>' if hint else ""
    )
    return (
        '<div style="display:flex;gap:.7rem;align-items:baseline;">'
        '<span style="flex:none;width:1.6rem;height:1.6rem;border-radius:50%;'
        'display:inline-flex;align-items:center;justify-content:center;'
        'font-size:.82rem;font-weight:700;color:#fff;background:#3b82f6;">'
        f'{number}</span><div><div style="font-weight:700;font-size:1.02rem;">'
        f'{html.escape(title)}</div>{sub}</div></div>'
    )


# The one live control on the page: what the device is doing right now, polled
# rather than set by a click, because the press happens on the grabette.
_TR_PILL_STYLES = {
    "idle": ("#94a3b8", ""),
    "recording": ("#ef4444", "animation:grabette-pulse 1.2s ease-in-out infinite;"),
    "waiting": ("#f59e0b", "animation:grabette-pulse 1.2s ease-in-out infinite;"),
    "done": ("#10b981", ""),
    "blocked": ("#ef4444", ""),
}


def _tr_pill(kind: str, text: str) -> str:
    color, anim = _TR_PILL_STYLES.get(kind, _TR_PILL_STYLES["idle"])
    return (
        '<div style="display:inline-flex;align-items:center;gap:.55rem;'
        'padding:.45rem .9rem;border-radius:999px;font-size:.9rem;'
        f'background:{color}1a;border:1px solid {color}55;color:{color};">'
        f'<span style="width:.55rem;height:.55rem;border-radius:50%;'
        f'background:{color};{anim}"></span>'
        f'<span style="font-weight:600;">{html.escape(text)}</span></div>'
    )


def _replay_video_iframe(episode_id: str, stream: str = "raw") -> str:
    """Player slaved to the replay clock. stream="dcam" plays the depth
    camera's own image stream rather than the head camera."""
    return (
        f'<iframe src="/api/replay/video?episode_id={episode_id}&stream={stream}" '
        'style="width:100%;height:240px;border:none;'
        'border-radius:8px;background:#000;"></iframe>'
    )


_VIEWER_IFRAME_HTML = (
    '<iframe id="urdf-viewer" src="/viewer" '
    'style="width:100%;height:28vh;border:none;'
    'border-radius:8px;background:#1a1a2e;"></iframe>'
)

# The Overview's two previews stand side by side, so the viewer is pinned to
# the same height as the camera image rather than to the viewport.
# 200px is the camera preview's height too: pinning every tile of the first row
# to it is what puts them on one baseline and their buttons on the next.
_OV_TILE_H = 200

_OV_VIEWER_IFRAME = (
    '<iframe id="urdf-viewer" src="/viewer?yaw=180" '
    f'style="width:100%;height:{_OV_TILE_H}px;border:none;'
    'border-radius:8px;background:#1a1a2e;"></iframe>'
)

# Row separator — a hairline in the theme's own border colour, not the dark
# slate <hr> the Live View uses, which is invisible on the light dashboard.
_OV_RULE = (
    '<div style="height:1px;background:var(--border-color-primary);'
    'margin:1.4rem 0 1.1rem;"></div>'
)

_OV_RGB = "RGB"
_OV_DEPTH = "Depth"

_GYRO_IFRAME_HTML = (
    '<iframe src="/charts/gyro" '
    'style="width:100%;height:28vh;border:none;'
    'border-radius:8px;background:transparent;"></iframe>'
)
_ACCEL_IFRAME_HTML = (
    '<iframe src="/charts/accel" '
    'style="width:100%;height:28vh;border:none;'
    'border-radius:8px;background:transparent;"></iframe>'
)
_ANGLE_IFRAME_HTML = (
    '<iframe src="/charts/angle" '
    'style="width:100%;height:28vh;border:none;'
    'border-radius:8px;background:transparent;"></iframe>'
)
_WIFI_SETTINGS_HTML = (
    '<iframe src="/api/wifi/setup" id="wifi-iframe" scrolling="no"'
    ' onload="var f=this;(function r(){'
    'if(!document.contains(f))return;'
    'try{f.style.height=f.contentDocument.body.scrollHeight+20+\'px\';}catch(e){}'
    'setTimeout(r,400);})()"'
    ' style="width:100%;border:none;border-radius:8px;min-height:200px;">'
    '</iframe>'
)

# Rendered as explicit HTML rather than Markdown so the title font is pinned to a
# complete system sans-serif stack. The Markdown <h1> inherited the theme's
# webfont (--font), which renders inconsistently — and falls back to serif — when
# it loads partially or fails (e.g. the robot runs offline).
# ── Overview cards ───────────────────────────────────────────────────
#
# Pure builders (no network) so the rules — what a missing reading looks like,
# when the battery turns red — can be unit-tested. Everything is expressed in
# theme variables: the same card has to read on the light dashboard and in dark
# mode, which the hardcoded slate cards of the Live View bar do not.
_OV_CARD = (
    "display:flex;flex-direction:column;justify-content:space-evenly;"
    "box-sizing:border-box;"
    f"height:{_OV_TILE_H}px;"
    "background:var(--background-fill-secondary);"
    "border:1px solid var(--border-color-primary);"
    "border-radius:14px;padding:.85rem 1rem;overflow:hidden;"
)


def _ov_row(label: str, value: str, color: str = "") -> str:
    """One label/value line inside an overview card."""
    tint = f"color:{color};" if color else "color:var(--body-text-color);"
    return (
        '<div>'
        '<div style="font-size:.68rem;text-transform:uppercase;'
        'letter-spacing:.09em;color:var(--body-text-color-subdued);">'
        f'{html.escape(label)}</div>'
        f'<div style="font-size:.95rem;font-weight:600;{tint}'
        'word-break:break-all;">'
        f'{value}</div></div>'
    )


def _ov_battery(info: dict) -> tuple[str, str]:
    """Battery reading and the colour that says how worried to be."""
    if "battery_pct" not in info:
        return "—", ""
    pct = info["battery_pct"]
    charging = info.get("battery_charging")
    if charging or pct > 40:
        color = "#22c55e"
    elif pct > _BATTERY_WARN_PCT:
        color = "#f97316"
    else:
        color = "#ef4444"
    return (f"⚡ {pct} %" if charging else f"{pct} %"), color


def _ov_device_card(info: dict | None, wifi: dict | None) -> str:
    """Hostname, address and network — who this device is on the network."""
    info, wifi = info or {}, wifi or {}
    ip = wifi.get("ip") or info.get("ip") or "—"
    ssid = wifi.get("ssid") or "—"
    return (
        f'<div style="{_OV_CARD}">'
        + _ov_row("Hostname", html.escape(str(info.get("hostname") or "—")))
        + _ov_row("Side", settings.hand.capitalize())
        + _ov_row("IP address", html.escape(str(ip)))
        + _ov_row("Network", html.escape(str(ssid)))
        + "</div>"
    )


def _ov_health_card(info: dict | None) -> str:
    """Battery, temperature and the storage a recording can still land in."""
    info = info or {}
    batt, color = _ov_battery(info)
    temp = f"{info['cpu_temp_c']} °C" if "cpu_temp_c" in info else "—"
    if "disk_free_gb" in info:
        total = info.get("disk_total_gb")
        storage = (f"{info['disk_free_gb']} GB free"
                   + (f" of {total} GB" if total else ""))
    else:
        storage = "—"
    return (
        f'<div style="{_OV_CARD}">'
        + _ov_row("Battery", batt, color)
        + _ov_row("Temperature", temp)
        + _ov_row("Storage", storage)
        + "</div>"
    )


_TITLE_HTML = (
    "<h1 style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',"
    "Roboto,Helvetica,Arial,sans-serif;font-weight:700;"
    "font-size:var(--text-xxl,2rem);color:var(--body-text-color);"
    "margin:var(--spacing-xxl) 0 var(--spacing-lg);\">GRABETTE</h1>"
)

# Battery percentage at/below which the low-battery warning popup + sound fire.
_BATTERY_WARN_PCT = 25

# Run once per page load via `<page>.load(js=...)`. Gradio executes `js` load
# handlers on the client (unlike the `head=` param, whose inline <script> is
# injected via innerHTML and never runs). Defines window.__grabetteBatteryBeep()
# — a two-tone Web Audio chime + system notification — and, because browser
# autoplay policy blocks audio until the user interacts with the page, resumes
# the AudioContext / requests Notification permission on the first user gesture.
# A hidden/background tab (screen asleep, tab not focused) can still beep and
# notify as long as the machine itself is not fully suspended — a real OS
# suspend halts all JS and no local page can work around that.
_BATTERY_INIT_JS = """
() => {
  if (window.__grabetteBatteryBeep) { return; }
  var ctx = null;
  var lastBeep = 0;

  function ensureCtx() {
    if (!ctx) {
      try { ctx = new (window.AudioContext || window.webkitAudioContext)(); }
      catch (e) { ctx = null; }
    }
    return ctx;
  }

  function unlock() {
    var c = ensureCtx();
    if (c && c.state === 'suspended') { c.resume(); }
    if ('Notification' in window && Notification.permission === 'default') {
      try { Notification.requestPermission(); } catch (e) {}
    }
  }
  ['pointerdown', 'keydown', 'touchstart'].forEach(function (ev) {
    window.addEventListener(ev, unlock, { passive: true });
  });

  function chime(c) {
    function tone(freq, start, dur) {
      var osc = c.createOscillator();
      var gain = c.createGain();
      osc.type = 'sine';
      osc.frequency.value = freq;
      var t = c.currentTime + start;
      gain.gain.setValueAtTime(0.0001, t);
      gain.gain.exponentialRampToValueAtTime(0.35, t + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      osc.connect(gain).connect(c.destination);
      osc.start(t);
      osc.stop(t + dur + 0.02);
    }
    tone(880, 0.0, 0.25);
    tone(660, 0.30, 0.35);
  }

  window.__grabetteBatteryBeep = function (pct) {
    // Throttle so a fast popup poll doesn't over-beep: at most once per 60 s.
    var now = Date.now();
    if (now - lastBeep < 60000) { return; }
    lastBeep = now;

    var c = ensureCtx();
    if (c) {
      if (c.state === 'suspended') { c.resume(); }
      try { chime(c); } catch (e) {}
    }
    if ('Notification' in window && Notification.permission === 'granted') {
      try {
        new Notification('Grabette — battery low', {
          body: 'Please charge soon.',
          tag: 'grabette-battery',
          renotify: true,
        });
      } catch (e) {}
    }
  };
}
"""

# Frontend handler bound to the (hidden) battery-beep signal's `change` event.
# Runs client-side — unlike HTML-component content, it is never sanitized and
# fires reliably. The signal carries "<pct>|<nonce>"; the nonce changes every
# poll so `change` keeps firing while the battery stays low (throttled to one
# chime per 60 s inside __grabetteBatteryBeep).
_BATTERY_BEEP_JS = (
    "(v) => { if (v && window.__grabetteBatteryBeep) "
    "{ window.__grabetteBatteryBeep(String(v).split('|')[0]); } }"
)


# Live View is off: five pages of dashboard for a device whose job is to record
# with its own button was four more than anyone opened. The page is kept whole —
# its handlers, its timers, its charts — behind this one flag, so bringing it
# back is flipping False to True.
_LIVE_VIEW_ENABLED = False


# ── Episode status ───────────────────────────────────────────────────
#
# Read off the listing the task API already returns (counters, has_video,
# metadata_ok), so a table of fifty episodes costs no extra call. What it
# cannot see from there is which files are on disk — that is the Check button,
# which asks the check endpoint for the selected episode and shows the same
# verdict card as Test Recording.
_EP_MIN_SECONDS = 2.0


def _episode_status(ep: dict) -> str:
    """One cell: the worst thing this episode's own counters admit to."""
    if not ep.get("metadata_ok", True):
        return "✗ metadata"
    if not ep.get("has_video", True) or not ep.get("frame_count"):
        return "✗ camera"
    if not ep.get("angle_sample_count"):
        return "✗ angles"
    if float(ep.get("duration_seconds") or 0) < _EP_MIN_SECONDS:
        return "⚠ very short"
    return "✓ ok"


def _ep_header_html(description: str = "", api_down: bool = False) -> str:
    """The note above the table: what the task is for, or that the API is down.

    The task's name and episode count are already in the menu above it.
    """
    if api_down:
        return (
            '<div style="border-radius:12px;padding:.85rem 1rem;'
            'background:var(--background-fill-secondary);'
            'border-left:4px solid #ef4444;">'
            '<strong>Could not reach the grabette API.</strong> '
            '<span style="opacity:.8;">The list below is empty because the '
            'call failed, not because nothing is recorded.</span></div>'
        )
    if not description:
        return ""
    return ('<div style="font-size:.88rem;opacity:.75;">'
            f'{html.escape(description)}</div>')


def _section_label(text: str) -> str:
    """Small uppercase gray column header used across the Live View page."""
    return (
        "<div style='font-size:0.72rem;text-transform:uppercase;"
        "letter-spacing:0.09em;color:#94a3b8;margin-bottom:0.3rem;'>"
        f"{text}</div>"
    )


def _status_bar_html(sys_info, oakd_status, cam_status):
    """Build the Episodes status strip (battery + RGB + OAK-D) from already-fetched dicts.

    Pure function (no network calls) so it can be unit-tested. Each argument
    may be None when the corresponding API call failed.
    """

    # (value color, border color). Neutral gray covers off / N/A / unknown.
    GRAY = ("#94a3b8", "#334155")
    GREEN = ("#22c55e", "#166534")
    ORANGE = ("#f97316", "#9a3412")
    RED = ("#ef4444", "#991b1b")

    def _badge(label, value, colors, title=""):
        value_color, border_color = colors
        # `title` carries the long form (a hardware fault's full explanation) as
        # a hover tooltip: the badge row has one line per badge and truncates,
        # so a message that matters can't live in the value itself.
        tip = f" title=\"{html.escape(title, quote=True)}\"" if title else ""
        return (
            f"<div{tip} style='background:#1e293b;border-radius:8px;padding:0.55rem 1rem;"
            f"border:2px solid {border_color};flex:1;min-width:0;'>"
            f"<div style='font-size:0.65rem;text-transform:uppercase;letter-spacing:0.09em;"
            f"color:#94a3b8;margin-bottom:0.2rem;'>{label}</div>"
            f"<div style='font-size:0.9rem;font-weight:700;color:{value_color};"
            f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{value}</div>"
            f"</div>"
        )

    # Battery (⚡ + green while charging, regardless of level)
    if sys_info and "battery_pct" in sys_info:
        pct = sys_info["battery_pct"]
        if sys_info.get("battery_charging"):
            batt_badge = _badge("Battery", f"⚡ {pct} %", GREEN)
        else:
            colors = GREEN if pct > 40 else ORANGE if pct > 20 else RED
            batt_badge = _badge("Battery", f"{pct} %", colors)
    else:
        batt_badge = _badge("Battery", "N/A", GRAY)

    # RGB camera (3-state: connected / reinitializing / disconnected; N/A if call failed)
    if cam_status is None:
        rgb_badge = _badge("RGB Camera", "N/A", GRAY)
    elif cam_status.get("connected"):
        rgb_badge = _badge("RGB Camera", "Connected", GREEN)
    elif cam_status.get("reinitializing"):
        rgb_badge = _badge("RGB Camera", "Unavailable", ORANGE)
    else:
        rgb_badge = _badge("RGB Camera", "Disconnected", RED)

    # Depth camera (5-state: fault / connected / starting / off / error; N/A
    # when unsupported). The fault comes FIRST: it is the one state where the
    # device refuses to record, so it must not be masked by "Off" after a
    # power-down. The label comes from the API so the badge names the hardware
    # actually configured — "OAK-D" or "Gemini 305" — rather than assuming.
    depth_label = (oakd_status or {}).get("label") or "Depth camera"
    if not oakd_status or not oakd_status.get("supported"):
        oakd_badge = _badge(depth_label, "N/A", GRAY)
    elif oakd_status.get("hardware_error"):
        oakd_badge = _badge(depth_label, "Cannot record", RED,
                            title=oakd_status["hardware_error"])
    elif oakd_status.get("initialized"):
        oakd_badge = _badge(depth_label, "Connected", GREEN)
    elif oakd_status.get("initializing"):
        oakd_badge = _badge(depth_label, "Starting…", ORANGE)
    elif oakd_status.get("enabled"):
        oakd_badge = _badge(depth_label, "Error", RED)
    else:
        oakd_badge = _badge(depth_label, "Off", GRAY)

    return (
        "<div style='display:flex;flex-direction:row;gap:0.5rem;flex-wrap:wrap;"
        "margin:0.25rem 0 0.75rem;'>"
        + batt_badge + rgb_badge + oakd_badge
        + "</div>"
    )


def create_ui(api_url: str | None = None) -> gr.Blocks:
    # Route downloaded episode archives to the SD-card-backed data_dir
    # instead of the OS /tmp (which on Pi OS is a small tmpfs). Same reason
    # as the SessionManager staging; the SessionManager's startup sweep
    # cleans this same directory across daemon restarts.
    client = GrabetteClient(
        base_url=api_url,
        download_dir=settings.data_dir / ".downloads",
    )

    # ── Camera ────────────────────────────────────────────────────────

    def get_camera_frame():
        data = client.get_snapshot()
        if data is None:
            return None
        try:
            return Image.open(io.BytesIO(data))
        except Exception:
            return None

    def get_depth_frame():
        data = client.get_depth_snapshot()
        if data is None:
            return None
        try:
            return Image.open(io.BytesIO(data))
        except Exception:
            return None

    # ── Sensor state (Live Streaming page) ────────────────────────────

    def _mono(inner: str) -> str:
        """Wrap colour-labelled sensor text in a monospace, pre-spaced span.

        ``white-space:pre`` keeps the numeric column alignment that markdown
        ``<code>`` would otherwise collapse.
        """
        return (
            "<span style='font-family:monospace;white-space:pre;"
            "font-size:0.95em'>" + inner + "</span>"
        )

    def get_sensor_state():
        """Returns (gyro_text, accel_text, angle_text)."""
        state = client.get_state()
        if state is None:
            return "*Disconnected*", "*Disconnected*", "*Disconnected*"

        imu = state.get("imu")
        if imu:
            a = imu["accel"]
            g = imu["gyro"]
            # Label colours match the uPlot curve strokes in charts.py so the
            # readout doubles as the chart legend: X=#e55, Y=#5b5, Z=#55e.
            gyro_text = _mono(
                f"<span style='color:#e55'>X:</span>{g[0]:+8.4f}  "
                f"<span style='color:#5b5'>Y:</span>{g[1]:+8.4f}  "
                f"<span style='color:#55e'>Z:</span>{g[2]:+8.4f}  rad/s"
            )
            accel_text = _mono(
                f"<span style='color:#e55'>X:</span>{a[0]:+8.3f}  "
                f"<span style='color:#5b5'>Y:</span>{a[1]:+8.3f}  "
                f"<span style='color:#55e'>Z:</span>{a[2]:+8.3f}  m/s²"
            )
        else:
            gyro_text = "*No IMU data*"
            accel_text = "*No IMU data*"

        angle = state.get("angle")
        if angle:
            p_deg = math.degrees(angle["proximal"])
            d_deg = math.degrees(angle["distal"])
            # Proximal=#4488cc, Distal=#cc8844 match the angle chart strokes.
            angle_text = _mono(
                f"<span style='color:#4488cc'>Proximal:</span> "
                f"{p_deg:+7.2f}°  ({angle['proximal']:+.4f} rad)\n"
                f"<span style='color:#cc8844'>Distal:  </span> "
                f"{d_deg:+7.2f}°  ({angle['distal']:+.4f} rad)"
            )
        else:
            angle_text = "*No data*"

        return gyro_text, accel_text, angle_text

    # ── Capture (Datasets page) ───────────────────────────────────────

    # ── Test Recording page ───────────────────────────────────────────
    #
    # The recording is started and stopped on the grabette's own button, so this
    # page never commands a capture: it POLLS the device and reacts. `episode_id`
    # has to be remembered while the capture still reports it — CaptureStatus
    # drops it the moment the recording ends.
    _TR_FLOW0 = {"capturing": False, "episode_id": None, "settling": 0}

    # stop_capture writes metadata.json from a deferred callback, so the episode
    # is briefly on disk with nothing readable in it. Re-read for a few ticks
    # before believing "metadata is missing".
    _TR_SETTLE_TICKS = 8

    def _tr_verdict(episode_id):
        """The verdict card for a finished recording, or None while it settles."""
        episode = client.get_episode(episode_id)
        if episode is None or not episode.get("metadata_ok", True):
            return None
        return recording_summary(episode, client.check_episode(episode_id))

    def _tr_controls(episode_id):
        """The three step controls, in the order the poll outputs them.

        Check and delete are gradio buttons that grey out; the download is a
        link, so it is swapped for a dead-looking span instead.
        """
        return (gr.update(interactive=bool(episode_id)),
                gr.update(value=_download_link_html(episode_id)),
                gr.update(interactive=bool(episode_id)))

    def poll_test_recording(flow):
        """One tick: mirror the device's capture state onto the page.

        Outputs: status pill, verdict card, the three episode controls (check,
        download, delete) and the flow dict.
        """
        flow = dict(flow or _TR_FLOW0)
        cap = (client.get_state() or {}).get("capture", {})
        capturing = bool(cap.get("is_capturing"))
        blocked = cap.get("blocked_reason") or ""

        if capturing:
            # A new recording supersedes whatever steps 2 and 3 were pointing at.
            flow.update(capturing=True, settling=0,
                        episode_id=cap.get("episode_id") or flow["episode_id"])
            return (
                _tr_pill("recording",
                         f"Recording — {float(cap.get('duration_seconds') or 0):.0f}s"),
                gr.update(value=""),
                *_tr_controls(None),
                flow,
            )

        if flow["capturing"] or flow["settling"]:
            # The recording just ended (or is still being written out).
            flow["capturing"] = False
            verdict = _tr_verdict(flow["episode_id"]) if flow["episode_id"] else None
            # No episode id at all means the capture never reported one — there
            # is nothing on disk to wait for, so don't spend the settle window.
            if (verdict is None and flow["episode_id"]
                    and flow["settling"] < _TR_SETTLE_TICKS):
                flow["settling"] += 1
                return (_tr_pill("waiting", "Saving the episode…"),
                        gr.update(value=""),
                        *_tr_controls(None), flow)
            flow["settling"] = 0
            if verdict is None:
                # Out of patience: report what the episode actually says now,
                # missing metadata included — that IS the finding.
                verdict = recording_summary(
                    client.get_episode(flow["episode_id"]),
                    client.check_episode(flow["episode_id"]),
                )
            return (
                _tr_pill("done", "Recorded — check it below"),
                gr.update(value=verdict),
                *_tr_controls(flow["episode_id"]),
                flow,
            )

        # Idle. Only the pill moves from here on: the verdict and the two
        # buttons keep whatever the tick that finished the recording set.
        if blocked:
            # Not capturing is not the same as ready: a device held back by an
            # upload or a hardware fault would read "waiting for the recording"
            # here, and the press it invites is the one that fails.
            pill = _tr_pill("blocked", f"Cannot record — {blocked}")
        elif cap.get("is_starting"):
            pill = _tr_pill("waiting", "Starting…")
        elif flow["episode_id"]:
            pill = _tr_pill("done", "Recorded — check it below")
        else:
            pill = _tr_pill("idle", "Waiting for the recording")
        return pill, *([gr.update()] * 4), flow

    def on_test_check(flow):
        """Replay the episode: the daemon feeds its recorded samples back into
        the same live state the charts read, so the angle chart moves with the
        video rather than showing the sensors' current values."""
        episode_id = (flow or {}).get("episode_id")
        if not episode_id:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        result = client.replay_start(episode_id)
        if "error" in result:
            return (
                gr.update(visible=False), gr.update(), gr.update(),
                gr.update(), f"⛔ {result['error']}",
            )
        return (
            gr.update(visible=True),
            gr.update(value=_replay_video_iframe(episode_id, "raw")),
            gr.update(value=_replay_video_iframe(episode_id, "dcam")),
            gr.update(active=True),
            f"Replaying `{episode_id}`",
        )

    def on_test_replay_stop():
        client.replay_stop()
        return gr.update(visible=False), gr.update(active=False), "Replay stopped"

    def poll_test_replay():
        """Follow the replay clock, and hand the sensors back at the end.

        The engine stays active once it reaches the last sample (playing goes
        false, time stops at duration), which leaves every live reading on the
        device pinned to replay data until something stops it. Stop it here.
        The panel stays up so the replay can be played again.
        """
        st = client.replay_status()
        if not st.get("active"):
            return gr.update(), gr.update(active=False), "Replay ended"
        t, dur = st.get("time_ms", 0), st.get("duration_ms", 0)
        if dur and t >= dur and not st.get("playing"):
            client.replay_stop()
            return gr.update(), gr.update(active=False), "Replay ended — ↻ to play it again"
        return gr.update(), gr.update(), f"{t / 1000:.1f}s / {dur / 1000:.1f}s"

    def on_test_delete(flow):
        flow = dict(flow or _TR_FLOW0)
        episode_id = flow.get("episode_id")
        if not episode_id:
            return (*([gr.update()] * 7), flow)
        # Stop first: deleting the files under a running replay leaves the
        # daemon reading a directory that is no longer there.
        client.replay_stop()
        result = client.delete_episode(episode_id)
        msg = (f"⛔ {result['error']}" if result.get("error")
               else f"Deleted `{episode_id}`.")
        flow.update(episode_id=None, settling=0)
        return (
            *_tr_controls(None),            # check, download, delete
            gr.update(visible=False),       # replay panel
            gr.update(active=False),        # replay timer
            msg,
            gr.update(value=""),            # verdict — it describes what is gone
            flow,
        )


    def _oakd_button_update():
        """Compute the depth-camera toggle's appearance + data row visibility.

        Returns (button_update, oak_row_visibility) so callers can keep the
        depth/IMU/accelerometer row hidden until the camera is enabled.

        The button names the configured camera using the label the API serves,
        so it reads "Gemini 305: ON" on a device running the Orbbec.
        """
        s = client.get_oakd_status() or {}
        name = s.get("label") or "Depth camera"
        if not s.get("supported"):
            return (
                gr.update(
                    value=f"{name} not available",
                    variant="secondary",
                    interactive=False,
                ),
                gr.update(visible=False),
            )
        enabled = bool(s.get("enabled"))
        # Greyed out while capture or teleop holds the OAK — toggling is
        # refused server-side anyway, but the visual cue prevents user
        # confusion.
        state = client.get_state() or {}
        capturing = bool(state.get("capture", {}).get("is_capturing"))
        tstatus = client.get_teleop_status() or {}
        teleop = bool(tstatus.get("active"))
        busy = capturing or teleop
        if enabled:
            label = f"{name}: ON" + ("  (busy)" if busy else "  — click to disable")
            variant = "primary"
        else:
            label = f"{name}: OFF" + ("  (busy)" if busy else "  — click to enable")
            variant = "secondary"
        return (
            gr.update(value=label, variant=variant, interactive=not busy),
            gr.update(visible=enabled),
        )

    def on_toggle_oakd():
        s = client.get_oakd_status() or {}
        enabled = bool(s.get("enabled"))
        result = client.set_oakd(not enabled)
        if "error" in result:
            logger.warning("Depth camera toggle failed: %s", result["error"])
        return _oakd_button_update()

    def poll_oakd():
        return _oakd_button_update()

    # ── Task helpers ──────────────────────────────────────────────────

    def _get_sessions():
        return client.list_tasks()

    def _task_menu_choices(sessions):
        """(label, id) for the task dropdown, each carrying its episode count."""
        def label(s):
            n = len(s.get("episodes") or [])
            return f"{s['name']} - {n} episode{'' if n == 1 else 's'}"
        return [(label(s), s["id"]) for s in sessions]

    def _refresh_episode_table(session_id, sessions=None):
        """(rows, header html) for one task."""
        if sessions is None:
            sessions = _get_sessions()
        rows = []
        task_description = ""
        # The device always has Unassigned, so an empty list means the API call
        # failed — never that there is nothing to show. Saying so beats a blank
        # page that looks exactly like "no episodes recorded yet".
        api_down = not sessions
        for task in sessions:
            if task["id"] == session_id:
                task_description = task.get("description", "")
                for ep in task.get("episodes", []):
                    rows.append([
                        False,
                        ep["episode_id"],
                        f"{ep['duration_seconds']:.1f}s",
                        ep["frame_count"],
                        ep.get("angle_sample_count", 0),
                        _episode_status(ep),
                    ])
                break
        rows.reverse()
        return rows, _ep_header_html(task_description, api_down)

    def refresh_tasks(stored_id: str = ""):
        sessions = _get_sessions()
        choices = _task_menu_choices(sessions)
        valid_ids = {c[1] for c in choices}
        # Pick which task to land on at (re)load time:
        #   1. during a capture session, always the session's task;
        #   2. otherwise the task this browser had selected (persisted
        #      client-side via BrowserState), so a refresh stays put;
        #   3. otherwise fall back to the first task.
        cap_session = client.get_session_status()
        cap_task = cap_session.get("task_id") if cap_session.get("active") else None
        if cap_task in valid_ids:
            value = cap_task
        elif stored_id in valid_ids:
            value = stored_id
        else:
            value = choices[0][1] if choices else None
        rows, header = _refresh_episode_table(value, sessions)
        return gr.update(choices=choices, value=value), header, rows

    def on_task_select(session_id):
        # Selecting a task also points the device at it, so a recording started
        # from the grabette's own button lands where the dashboard is looking.
        if session_id:
            client.set_active_task(session_id)
        rows, header = _refresh_episode_table(session_id)
        return header, rows

    def _get_selected_ids(table_data) -> list[str]:
        if table_data is None:
            return []
        try:
            if table_data.empty:
                return []
            selected = table_data[table_data.iloc[:, 0] == True]  # noqa: E712  pandas element-wise boolean mask, not a truthiness check
            return selected.iloc[:, 1].tolist()
        except Exception:
            return []

    # ── Episode actions ───────────────────────────────────────────────

    def on_download_episodes(table_data):
        """Build the archive and reveal the drop it lands in."""
        episode_ids = _get_selected_ids(table_data)
        if not episode_ids:
            return gr.update(), "Tick the episodes to download"
        path = client.download_episodes(episode_ids)
        if path is None:
            return gr.update(), "⛔ Could not build the archive."
        return (gr.update(value=path, visible=True),
                f"{len(episode_ids)} episode(s) ready to download")

    def on_delete_episode(table_data, session_id):
        episode_ids = _get_selected_ids(table_data)
        if not episode_ids:
            return "No episode selected", gr.update(), gr.update()
        errors = []
        for eid in episode_ids:
            result = client.delete_episode(eid)
            if "error" in result:
                errors.append(f"{eid}: {result['error']}")
        rows, header = _refresh_episode_table(session_id)
        # Force the interactive dataframe to re-render: after the user ticks
        # rows it holds "dirty" client-side state that a bare list won't
        # overwrite, so the deleted rows (and their checkboxes) would linger.
        table_upd = gr.update(value=rows)
        if errors:
            return "Errors: " + "; ".join(errors), table_upd, header
        return f"Deleted {len(episode_ids)} episode(s)", table_upd, header

    def on_delete_all(confirmed, stored_id):
        """Wipe every episode on the device, once the browser has confirmed."""
        if not confirmed:
            return (gr.update(),) * 6
        # Stop first: deleting the files under a running replay leaves the
        # daemon reading from a directory that is gone.
        client.replay_stop()
        result = client.delete_all_episodes()
        task_upd, header, rows = refresh_tasks(stored_id)
        msg = (f"⛔ {result['error']}" if "error" in result
               else f"Deleted all data ({result.get('deleted', 0)} episode(s))")
        return (msg, task_upd, header, gr.update(value=rows),
                gr.update(visible=False), gr.update(active=False))

    def on_check_episode(table_data):
        """Full verdict for one selected episode — the file check included.

        The table's own Status column only knows what the listing says; this is
        what asks the device which files are actually on disk.
        """
        episode_id = (_get_selected_ids(table_data) or [None])[0]
        if not episode_id:
            return gr.update(value=""), "Tick an episode to check it"
        verdict = recording_summary(
            client.get_episode(episode_id), client.check_episode(episode_id),
        )
        return gr.update(value=verdict), f"Checked {episode_id}"

    # ── SLAM ──────────────────────────────────────────────────────────

    # ── Replay ────────────────────────────────────────────────────────

    def _video_iframe(episode_id: str) -> str:
        return (
            f'<iframe src="/api/replay/video?episode_id={episode_id}" '
            'style="width:100%;height:320px;border:none;'
            'border-radius:8px;background:#000;"></iframe>'
        )

    def on_replay_start(table_data):
        episode_id = (_get_selected_ids(table_data) or [None])[0]
        if not episode_id:
            return "No episode selected", gr.update(visible=False), gr.update(), gr.update(), gr.update()
        result = client.replay_start(episode_id)
        if "error" in result:
            return f"Error: {result['error']}", gr.update(visible=False), gr.update(), gr.update(), gr.update()
        dur = result.get("duration_ms", 0)
        return (
            f"Replaying {episode_id}",
            gr.update(visible=True),
            gr.update(maximum=dur, value=0),
            gr.update(active=True),
            gr.update(value=_video_iframe(episode_id)),
        )

    def on_replay_stop():
        client.replay_stop()
        return "Replay stopped", gr.update(visible=False), gr.update(active=False), gr.update(value="")

    def on_replay_pause_play():
        st = client.replay_status()
        if st.get("playing"):
            client.replay_pause()
            return "Play"
        else:
            client.replay_resume()
            return "Pause"

    def on_replay_seek(time_ms):
        if time_ms is not None:
            client.replay_seek(float(time_ms))

    def poll_replay_status():
        st = client.replay_status()
        if not st.get("active"):
            return (
                gr.update(), gr.update(), gr.update(),
                gr.update(active=False),
                gr.update(visible=False),
                gr.update(value=""),
            )
        t = st.get("time_ms", 0)
        dur = st.get("duration_ms", 0)
        playing = st.get("playing", False)
        label = f"{t / 1000:.1f}s / {dur / 1000:.1f}s" + (" (paused)" if not playing else "")
        return (
            gr.update(value=t),
            label,
            "Pause" if playing else "Play",
            gr.update(),
            gr.update(),
            gr.update(),
        )

    # ── Battery warning popup ─────────────────────────────────────────

    # Monotonic nonce so the hidden beep signal changes value on every low poll,
    # which re-fires the signal's `change` handler (recurring chime reminder).
    _batt_beep = {"n": 0}

    def check_battery_warning():
        """(popup_update, beep_signal) — bound to the battery timers/loads."""
        return _battery_popup_html(client.get_system_info())

    # ── System bar ────────────────────────────────────────────────────

    def _battery_popup_html(info: dict | None):
        """Return (popup_update, beep_signal) from a system info dict.

        beep_signal is "<pct>|<nonce>" while the battery is low (nonce bumps each
        call so the frontend `change` handler keeps firing) and "" otherwise.
        The warning is suppressed while charging, so plugging Grabette back in
        clears the popup + chime even below the threshold.
        """
        if (
            info
            and "battery_pct" in info
            and info["battery_pct"] <= _BATTERY_WARN_PCT
            and not info.get("battery_charging")
        ):
            pct = info["battery_pct"]
            _batt_beep["n"] += 1
            html = (
                "<div style='position:fixed;bottom:24px;right:24px;z-index:9999;"
                "background:#fef2f2;border:1px solid #fca5a5;border-radius:12px;"
                "padding:16px 20px;max-width:260px;"
                "box-shadow:0 4px 20px rgba(0,0,0,0.15);'>"
                "<div style='font-weight:700;color:#dc2626;font-size:1rem;"
                "margin-bottom:4px;'>Battery low</div>"
                f"<div style='font-size:0.88rem;color:#7f1d1d;'>{pct} % — please charge soon.</div>"
                "</div>"
            )
            return gr.update(visible=True, value=html), f"{pct}|{_batt_beep['n']}"
        return gr.update(visible=False), ""

    def get_system_bar():
        """Returns (system_bar_html, battery_popup_update, beep_signal)."""
        info = client.get_system_info()
        if info is None:
            bar = "<p style='color:#64748b;font-size:0.85rem;margin:0.5rem 0;'>System disconnected</p>"
            return bar, gr.update(visible=False), ""

        def _card(label, value, extra_style=""):
            return (
                f"<div style='background:#1e293b;border-radius:8px;padding:0.55rem 1rem;"
                f"border:1px solid #334155;flex:1;min-width:0;{extra_style}'>"
                f"<div style='font-size:0.65rem;text-transform:uppercase;letter-spacing:0.09em;"
                f"color:#94a3b8;margin-bottom:0.2rem;'>{label}</div>"
                f"<div style='font-size:0.9rem;font-weight:600;color:#f1f5f9;white-space:nowrap;"
                f"overflow:hidden;text-overflow:ellipsis;'>{value}</div>"
                f"</div>"
            )

        parts = []

        if info.get("hostname"):
            parts.append(_card("Host", info["hostname"]))
        if "cpu_temp_c" in info:
            parts.append(_card("CPU Temp", f"{info['cpu_temp_c']} °C"))
        if "disk_free_gb" in info:
            parts.append(_card("Disk Free", f"{info['disk_free_gb']} GB"))

        if "battery_pct" in info:
            pct = info["battery_pct"]
            charging = info.get("battery_charging")
            if charging or pct > 40:
                batt_color = "#22c55e"
                batt_border = "#166534"
            elif pct > 20:
                batt_color = "#f97316"
                batt_border = "#9a3412"
            else:
                batt_color = "#ef4444"
                batt_border = "#991b1b"
            batt_value = f"⚡ {pct} %" if charging else f"{pct} %"
            parts.append(
                f"<div style='background:#1e293b;border-radius:8px;padding:0.55rem 1rem;"
                f"border:2px solid {batt_border};flex:1;min-width:0;'>"
                f"<div style='font-size:0.65rem;text-transform:uppercase;letter-spacing:0.09em;"
                f"color:#94a3b8;margin-bottom:0.2rem;'>Battery</div>"
                f"<div style='font-size:0.9rem;font-weight:700;color:{batt_color};'>{batt_value}</div>"
                f"</div>"
            )

        bar = (
            "<div style='display:flex;flex-direction:row;gap:0.5rem;flex-wrap:wrap;'>"
            + "".join(parts)
            + "</div>"
        )
        popup_update, beep_signal = _battery_popup_html(info)
        return bar, popup_update, beep_signal

    # ── Episodes status strip (battery + camera connections) ─────────

    # ── Power off ─────────────────────────────────────────────────────

    def _poweroff_notice(text: str, color: str = "#f97316") -> str:
        return (
            f"<div style='max-width:520px;margin-top:0.75rem;padding:0.85rem 1.1rem;"
            f"background:#1e293b;border-left:4px solid {color};border-radius:8px;"
            f"color:#e2e8f0;font-size:0.92rem;'>{text}</div>"
        )

    def on_poweroff():
        result = client.shutdown()
        if "error" in result:
            return (
                gr.update(value=_poweroff_notice(f"⚠ {result['error']}", "#ef4444"), visible=True),
                gr.update(),
            )
        return (
            gr.update(
                value=_poweroff_notice(
                    "Device is shutting down, it can take ~30 s. This page will stop responding shortly."
                    "#22c55e",
                ),
                visible=True,
            ),
            gr.update(interactive=False, variant="secondary"),
        )

    def on_title_poweroff(confirmed):
        if not confirmed:
            return gr.update(), gr.update()
        return on_poweroff()

    def _title_bar():
        """The GRABETTE title with a power-off button beside it, on every page."""
        with gr.Row(elem_classes="gb-titlebar"):
            gr.HTML(_TITLE_HTML)
            btn = gr.Button(
                "⛔ Power off", size="lg", scale=0, elem_classes="gb-power"
            )
        notice = gr.HTML(value="", visible=False)
        confirmed = gr.Checkbox(value=False, visible=False)
        btn.click(
            fn=on_title_poweroff,
            inputs=[confirmed],
            outputs=[notice, btn],
            js="() => confirm('Power off the grabette?')",
        )

    # ══════════════════════════════════════════════════════════════════
    # Page 1 — Overview (landing)
    # ══════════════════════════════════════════════════════════════════
    #
    # What the device is doing and who it is, in one screen: what it sees, how
    # it is posed, where it is on the network, how it is holding up. The one
    # action worth taking from here — make a test recording — sits alone under
    # that row; the two errands (account, fleet) go last.

    def refresh_overview():
        """The two info cards. One tick, one call to each endpoint."""
        info = client.get_system_info()
        wifi = client.wifi_status() if info is not None else None
        return _ov_device_card(info, wifi), _ov_health_card(info)

    def ov_frame(mode):
        """Whichever camera the toggle is showing."""
        return get_depth_frame() if mode == _OV_DEPTH else get_camera_frame()

    def on_ov_mode(mode):
        """Switching to depth turns the depth camera on if it is off.

        Without this the toggle would show a permanently black panel on a
        device whose depth camera is disabled, which reads as a broken camera
        rather than a switched-off one. Switching back leaves it running:
        turning someone's camera off behind their back is the worse surprise.
        """
        if mode != _OV_DEPTH:
            return mode, ""
        status = client.get_oakd_status() or {}
        name = status.get("label") or "Depth camera"
        if not status.get("supported"):
            return mode, f"*{name} is not available on this device.*"
        if status.get("enabled"):
            return mode, ""
        result = client.set_oakd(True)
        if "error" in result:
            return mode, f"⛔ {result['error']}"
        return mode, f"*Starting {name}…*"

    with gr.Blocks(title="Grabette", css=MODAL_CSS) as demo:
        gr.Navbar(main_page_name="Overview", elem_id="grabette-nav")
        _title_bar()

        with gr.Column(elem_id="ov-page"):

            # ── Camera | 3D model | Device | Health ───────────────────
            # min_width is what makes this responsive: four columns on a
            # laptop, two on a tablet, one on a phone, decided by gradio from
            # the width each column says it needs.
            with gr.Row(equal_height=False, elem_classes="ov-tiles"):
                with gr.Column(scale=1, min_width=230):
                    gr.HTML(_section_label("Cameras"))
                    ov_camera_img = gr.Image(
                        label=None, show_label=False, height=_OV_TILE_H,
                        container=False,
                    )
                    # The tile's control, on the tile's own centre line.
                    ov_cam_mode = gr.Radio(
                        [_OV_RGB, _OV_DEPTH], value=_OV_RGB,
                        show_label=False, container=False,
                        elem_classes="ov-toggle",
                    )
                    ov_cam_msg = gr.Markdown("", elem_classes="ov-note")
                with gr.Column(scale=1, min_width=230):
                    gr.HTML(_section_label("3D model"))
                    gr.HTML(_OV_VIEWER_IFRAME)
                with gr.Column(scale=1, min_width=230):
                    gr.HTML(_section_label("Device"))
                    ov_device_card = gr.HTML(_ov_device_card(None, None))
                    with gr.Row(elem_classes="ov-tile-btn"):
                        gr.Button("Change network", link="/network", size="sm")
                with gr.Column(scale=1, min_width=230):
                    gr.HTML(_section_label("Health"))
                    ov_health_card = gr.HTML(_ov_health_card(None))

            gr.HTML(_OV_RULE)

            # ── The one thing to do from here ─────────────────────────
            with gr.Row(elem_classes="ov-cta"):
                gr.Button(
                    "Make a test recording →",
                    link="/test-recording",
                    variant="primary",
                    size="lg",
                )

            gr.HTML(_OV_RULE)

            # ── Account | Fleet Space ─────────────────────────────────
            with gr.Row(equal_height=False, elem_classes="ov-errands"):
                with gr.Column(scale=1, min_width=260):
                    gr.HTML(_section_label("HuggingFace account"))
                    gr.HTML(_HF_AUTH_BUTTON_IFRAME)
                with gr.Column(scale=1, min_width=260):
                    gr.HTML(_section_label("Fleet Space"))
                    gr.HTML(_FLEET_BUTTON_HTML.format(url=settings.relay_url))

        ov_mode_state = gr.State(_OV_RGB)
        ov_cam_mode.change(fn=on_ov_mode, inputs=ov_cam_mode,
                           outputs=[ov_mode_state, ov_cam_msg])

        cn_camera_timer = gr.Timer(0.2)
        cn_camera_timer.tick(fn=ov_frame, inputs=ov_mode_state,
                             outputs=ov_camera_img)

        ov_info_timer = gr.Timer(10.0)
        ov_info_timer.tick(fn=refresh_overview,
                           outputs=[ov_device_card, ov_health_card])
        demo.load(fn=refresh_overview, outputs=[ov_device_card, ov_health_card])

        batt_popup_cn = gr.HTML(visible=False)
        batt_beep_cn = gr.Textbox(visible=False)
        batt_timer_cn = gr.Timer(60.0)
        batt_timer_cn.tick(fn=check_battery_warning, outputs=[batt_popup_cn, batt_beep_cn])
        batt_beep_cn.change(fn=None, inputs=batt_beep_cn, outputs=None, js=_BATTERY_BEEP_JS)
        demo.load(fn=check_battery_warning, outputs=[batt_popup_cn, batt_beep_cn])
        demo.load(fn=None, js=_BATTERY_INIT_JS)

    # ══════════════════════════════════════════════════════════════════
    # Page 2 — Test Recording
    # ══════════════════════════════════════════════════════════════════

    # The first recording a new owner makes, and the only place that answers
    # "did it work?" without them having to read a file listing. No task, no
    # session: the episode lands in Unassigned and can be moved or deleted from
    # Episodes afterwards.
    with demo.route("Test Recording") as test_demo:
        gr.Navbar(main_page_name="Overview", elem_id="grabette-nav")
        _title_bar()

        # Everything the page remembers between ticks — see poll_test_recording.
        tr_flow = gr.State(dict(_TR_FLOW0))

        with gr.Column(elem_id="tr-page"):
            # ── 1 — Record ────────────────────────────────────────────
            with gr.Group(elem_classes="grabette-step"):
                gr.HTML(_step_header(
                    1, "Record a few seconds",
                    "Press the button, pick an object up, and press again to stop.",
                ))
                # min_width low enough to keep the two animations side by
                # side in the narrow card, high enough that a phone wraps
                # them onto two rows instead of shrinking them to thumbnails.
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=200):
                        gr.HTML(_button_gif("start-recording.gif",
                                            "Press to start and wait\n for the LED to stop blinking"))
                    with gr.Column(scale=1, min_width=200):
                        gr.HTML(_button_gif("stop-recording.gif",
                                            "Press again to stop"))
                tr_state = gr.HTML(_tr_pill("idle", "Waiting for the button"))
                # 1 Hz against the daemon's cached state — the page has no other
                # way to learn about a press that happened on the device.
                tr_poll_timer = gr.Timer(1.0)

            # ── 2 — Check ─────────────────────────────────────────────
            with gr.Group(elem_classes="grabette-step"):
                gr.HTML(_step_header(
                    2, "Check what was recorded",
                    "Plays back the episode or download it, to see the different files.",
                ))
                # Emptied rather than hidden: a visible=False -> True update on
                # this component never reaches the browser, while the value
                # update in the very same tick does. An empty HTML block takes
                # no room, so there is nothing to hide.
                tr_summary = gr.HTML("")
                with gr.Row(elem_classes="tr-actions"):
                    tr_check_btn = gr.Button("Show the recorded data",
                                             size="sm", variant="primary",
                                             interactive=False)
                    tr_download_link = gr.HTML(_download_link_html(None),
                                              elem_classes="tr-dl")
                tr_replay_msg = gr.Markdown("")
                with gr.Group(visible=False) as tr_replay_panel:
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            gr.HTML(_section_label("RGB camera"))
                            tr_raw_video = gr.HTML(value="")
                        with gr.Column(scale=1):
                            gr.HTML(_section_label("Depth camera (RGB-D)"))
                            tr_dcam_video = gr.HTML(value="")
                    gr.HTML(_section_label("Angle sensors"))
                    gr.HTML(_ANGLE_IFRAME_HTML)
                    with gr.Row():
                        tr_replay_again_btn = gr.Button("↻ Replay again",
                                                        size="sm",
                                                        variant="primary")
                        tr_replay_stop_btn = gr.Button("Stop replay", size="sm")
                tr_replay_timer = gr.Timer(0.5, active=False)

            # ── 3 — Delete ────────────────────────────────────────────
            with gr.Group(elem_classes="grabette-step"):
                gr.HTML(_step_header(
                    3, "Delete it",
                    "A test recording is not training data. Keep it only if "
                    "you meant to — it is filed under Unassigned in Episodes.",
                ))
                tr_delete_btn = gr.Button("Delete this episode", variant="stop",
                                          size="sm", interactive=False)
                tr_delete_msg = gr.Markdown("")

        # ── Wire events ───────────────────────────────────────────────
        tr_poll_timer.tick(
            fn=poll_test_recording, inputs=tr_flow,
            outputs=[tr_state, tr_summary, tr_check_btn, tr_download_link,
                     tr_delete_btn, tr_flow],
        )
        # Same handler for both: a restart is a replay started over from zero,
        # and the players follow whatever the clock says.
        for _btn in (tr_check_btn, tr_replay_again_btn):
            _btn.click(
                fn=on_test_check, inputs=tr_flow,
                outputs=[tr_replay_panel, tr_raw_video, tr_dcam_video,
                         tr_replay_timer, tr_replay_msg],
            )
        tr_replay_stop_btn.click(
            fn=on_test_replay_stop,
            outputs=[tr_replay_panel, tr_replay_timer, tr_replay_msg],
        )
        tr_replay_timer.tick(
            fn=poll_test_replay,
            outputs=[tr_replay_panel, tr_replay_timer, tr_replay_msg],
        )
        tr_delete_btn.click(
            fn=on_test_delete, inputs=tr_flow,
            outputs=[tr_check_btn, tr_download_link, tr_delete_btn,
                     tr_replay_panel, tr_replay_timer, tr_delete_msg,
                     tr_summary, tr_flow],
        )

        batt_popup_tr = gr.HTML(visible=False)
        batt_beep_tr = gr.Textbox(visible=False)
        batt_timer_tr = gr.Timer(60.0)
        batt_timer_tr.tick(fn=check_battery_warning, outputs=[batt_popup_tr, batt_beep_tr])
        batt_beep_tr.change(fn=None, inputs=batt_beep_tr, outputs=None, js=_BATTERY_BEEP_JS)
        test_demo.load(fn=check_battery_warning, outputs=[batt_popup_tr, batt_beep_tr])
        test_demo.load(fn=None, js=_BATTERY_INIT_JS)

    # ══════════════════════════════════════════════════════════════════
    # Page 3 — Episodes
    # ══════════════════════════════════════════════════════════════════
    #
    # A reading page: pick a task, see what it holds, act on a selection.
    # Recording is not started here — the grabette's own button does that, and
    # the capture/session controls that used to sit on top only offered a
    # second, contradictory way in.

    with demo.route("Episodes") as episodes_demo:
        gr.Navbar(main_page_name="Overview", elem_id="grabette-nav")
        _title_bar()

        with gr.Column(elem_id="ep-page"):
            with gr.Column():
                # Tasks are created and edited on the fleet, not on the device;
                # here you only choose which one to look at.
                task_list = gr.Dropdown(choices=[], show_label=False,
                                        container=False, interactive=True,
                                        elem_classes="ep-task-menu")
                # Remembers, per browser, which task was selected so a page
                # refresh stays on it instead of snapping back to the first.
                selected_task_state = gr.BrowserState(
                    "", storage_key="grabette_selected_task",
                )

                episodes_header = gr.HTML("")

                episodes_table = gr.Dataframe(
                    headers=["✓", "Episode", "Duration", "Frames", "Angles",
                             "Status"],
                    datatype=["bool", "str", "str", "number", "number", "str"],
                    interactive=True,
                    static_columns=[1, 2, 3, 4, 5],
                    col_count=(6, "fixed"),
                    show_search="filter",
                    elem_classes="ep-table",
                )

                with gr.Row(elem_classes="ep-actions"):
                    replay_btn = gr.Button("▶ Replay", size="sm",
                                           variant="primary")
                    check_btn = gr.Button("Check", size="sm")
                    dl_btn = gr.Button("Download", size="sm")
                    del_episode_btn = gr.Button("Delete", size="sm",
                                                variant="stop")

                # Markdown, not a Textbox: an empty message should leave no
                # trace, and an empty Textbox is a box.
                episode_msg = gr.Markdown("")
                # The full verdict for one episode — empty until Check is used.
                episode_check = gr.HTML("")
                # Hidden until an archive exists: an empty file drop is a hole
                # in the page on every state but one.
                dl_file = gr.File(label="Episode archive", visible=False,
                                  height=90)

                # Replay panel (hidden until replay starts)
                with gr.Group(visible=False) as replay_panel:
                    gr.HTML(_section_label("Replay"))
                    replay_video = gr.HTML(value="")
                    gr.HTML(
                        '<iframe src="/charts/angle" '
                        'style="width:100%;height:180px;border:none;'
                        'border-radius:8px;background:transparent;"></iframe>'
                    )
                    replay_slider = gr.Slider(
                        minimum=0, maximum=1, step=1, value=0,
                        label="Timeline (ms)", interactive=True,
                    )
                    replay_time_label = gr.Textbox(
                        value="0.0s / 0.0s", show_label=False,
                        interactive=False, max_lines=1,
                    )
                    with gr.Row(elem_classes="ep-actions"):
                        replay_pause_btn = gr.Button("Pause", size="sm")
                        replay_stop_btn = gr.Button("Stop replay",
                                                    variant="stop", size="sm")
                replay_timer = gr.Timer(0.5, active=False)

                # Set apart at the foot of the page: the one action here that
                # cannot be taken back.
                gr.HTML(_OV_RULE)
                with gr.Row(elem_classes="ep-actions"):
                    del_all_btn = gr.Button("🗑 Delete all Grabette data",
                                            variant="stop", size="lg")
                del_all_confirmed = gr.Checkbox(value=False, visible=False)

        # ── Wire events ───────────────────────────────────────────────

        task_list.change(
            fn=on_task_select, inputs=task_list,
            outputs=[episodes_header, episodes_table],
        )
        # Persist the current selection in the browser so a refresh keeps it.
        task_list.change(fn=lambda v: v, inputs=task_list, outputs=selected_task_state)

        dl_btn.click(fn=on_download_episodes, inputs=episodes_table,
                     outputs=[dl_file, episode_msg])
        check_btn.click(fn=on_check_episode, inputs=episodes_table,
                        outputs=[episode_check, episode_msg])
        del_episode_btn.click(
            fn=on_delete_episode, inputs=[episodes_table, task_list],
            outputs=[episode_msg, episodes_table, episodes_header],
        )
        del_all_btn.click(
            fn=on_delete_all, inputs=[del_all_confirmed, selected_task_state],
            outputs=[episode_msg, task_list, episodes_header, episodes_table,
                     replay_panel, replay_timer],
            js="(_, task) => [confirm('Delete ALL episodes on this Grabette? "
               "This cannot be undone.'), task]",
        )

        replay_btn.click(
            fn=on_replay_start, inputs=episodes_table,
            outputs=[episode_msg, replay_panel, replay_slider, replay_timer, replay_video],
        )
        replay_stop_btn.click(
            fn=on_replay_stop,
            outputs=[episode_msg, replay_panel, replay_timer, replay_video],
        )
        replay_pause_btn.click(fn=on_replay_pause_play, outputs=replay_pause_btn)
        replay_slider.release(fn=on_replay_seek, inputs=replay_slider)
        replay_timer.tick(
            fn=poll_replay_status,
            outputs=[replay_slider, replay_time_label, replay_pause_btn,
                     replay_timer, replay_panel, replay_video],
        )

        # The table refreshes on a slow beat rather than every half second:
        # nothing on this page starts a recording, so the only changes to catch
        # are the ones the grabette's button makes on its own.
        episodes_timer = gr.Timer(10.0)
        episodes_timer.tick(
            fn=on_task_select, inputs=task_list,
            outputs=[episodes_header, episodes_table],
        )

        batt_popup_ep = gr.HTML(visible=False)
        batt_beep_ep = gr.Textbox(visible=False)
        batt_timer_ep = gr.Timer(60.0)
        batt_timer_ep.tick(fn=check_battery_warning, outputs=[batt_popup_ep, batt_beep_ep])
        batt_beep_ep.change(fn=None, inputs=batt_beep_ep, outputs=None, js=_BATTERY_BEEP_JS)
        episodes_demo.load(fn=check_battery_warning, outputs=[batt_popup_ep, batt_beep_ep])
        episodes_demo.load(fn=None, js=_BATTERY_INIT_JS)

        episodes_demo.load(
            fn=refresh_tasks, inputs=[selected_task_state],
            outputs=[task_list, episodes_header, episodes_table],
        )

    # (Datasets page removed — dataset generation is done on the fleet.)

    # ══════════════════════════════════════════════════════════════════
    # Page 4 — Live View
    # ══════════════════════════════════════════════════════════════════

    if _LIVE_VIEW_ENABLED:
        with demo.route("Live View") as live_demo:
            gr.Navbar(main_page_name="Overview", elem_id="grabette-nav")
            _title_bar()

            # ── System bar (full width) ────────────────────────────────────
            dv_system_bar = gr.HTML()

            gr.HTML("<hr style='margin:0.75rem 0;border:none;border-top:1px solid #1e293b;'>")

            # ── Camera | Angle sensors | 3D viewer ─────────────────────────
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.HTML(_section_label("Camera"))
                    camera_img = gr.Image(
                        label=None, show_label=False, height="28vh", container=False,
                    )
                with gr.Column(scale=1):
                    gr.HTML(_section_label("Angle Sensors"))
                    angle_box = gr.Markdown("*—*")
                    gr.HTML(value=_ANGLE_IFRAME_HTML)
                with gr.Column(scale=1):
                    gr.HTML(_section_label("3D Model"))
                    gr.HTML(
                        '<iframe id="urdf-viewer" src="/viewer" '
                        'style="width:100%;height:28vh;border:none;'
                        'border-radius:8px;background:#1a1a2e;"></iframe>'
                    )

            gr.HTML("<hr style='margin:0.75rem 0;border:none;border-top:1px solid #1e293b;'>")

            # ── OAK-D data: Depth | IMU (gyro) | Accelerometer ─────────────
            # The whole row is hidden until the OAK-D is enabled (its depth, IMU
            # and accelerometer streams only exist while the camera is running).
            # The toggle button stays outside the row so it's always reachable.
            oakd_btn = gr.Button("Depth camera: OFF  — click to enable", size="sm")
            with gr.Row(visible=False, equal_height=True) as oak_row:
                with gr.Column(scale=1):
                    gr.HTML(_section_label("Depth (OAK-D)"))
                    depth_img = gr.Image(
                        label=None, show_label=False, height="28vh", container=False,
                    )
                with gr.Column(scale=1):
                    gr.HTML(_section_label("Gyroscope"))
                    gyro_box = gr.Markdown("*—*")
                    gr.HTML(value=_GYRO_IFRAME_HTML)
                with gr.Column(scale=1):
                    gr.HTML(_section_label("Accelerometer"))
                    accel_box = gr.Markdown("*—*")
                    gr.HTML(value=_ACCEL_IFRAME_HTML)

            camera_timer = gr.Timer(0.2)
            camera_timer.tick(fn=get_camera_frame, outputs=camera_img)

            depth_timer = gr.Timer(0.2)
            depth_timer.tick(fn=get_depth_frame, outputs=depth_img)

            sensor_timer = gr.Timer(0.5)
            sensor_timer.tick(fn=get_sensor_state, outputs=[gyro_box, accel_box, angle_box])

            oakd_timer = gr.Timer(3.0)
            oakd_timer.tick(fn=poll_oakd, outputs=[oakd_btn, oak_row])
            oakd_btn.click(fn=on_toggle_oakd, outputs=[oakd_btn, oak_row])
            live_demo.load(fn=poll_oakd, outputs=[oakd_btn, oak_row])

            batt_popup_lv = gr.HTML(visible=False)
            batt_beep_lv = gr.Textbox(visible=False)

            dv_system_timer = gr.Timer(10)
            dv_system_timer.tick(fn=get_system_bar, outputs=[dv_system_bar, batt_popup_lv, batt_beep_lv])
            batt_beep_lv.change(fn=None, inputs=batt_beep_lv, outputs=None, js=_BATTERY_BEEP_JS)
            live_demo.load(fn=get_system_bar, outputs=[dv_system_bar, batt_popup_lv, batt_beep_lv])
            live_demo.load(fn=None, js=_BATTERY_INIT_JS)

    # ══════════════════════════════════════════════════════════════════
    # Page 5 — Network
    # ══════════════════════════════════════════════════════════════════
    #
    # One errand: put the device on another WiFi. The card above the form is
    # the same one the Overview shows, so "which network am I on" reads the
    # same wherever it is asked. The HuggingFace login used to share this page;
    # it is now the account slab on the Overview, which runs the whole flow.

    def refresh_network_card():
        info = client.get_system_info()
        wifi = client.wifi_status() if info is not None else None
        return _ov_device_card(info, wifi)

    with demo.route("Network") as settings_demo:
        gr.Navbar(main_page_name="Overview", elem_id="grabette-nav")
        _title_bar()

        with gr.Column(elem_id="nw-page"):
            gr.HTML(_section_label("Device"))
            nw_device_card = gr.HTML(_ov_device_card(None, None))
            gr.HTML(_OV_RULE)
            gr.HTML(_section_label("Change WiFi network"))
            gr.HTML(_WIFI_SETTINGS_HTML)

        nw_timer = gr.Timer(10.0)
        nw_timer.tick(fn=refresh_network_card, outputs=nw_device_card)
        settings_demo.load(fn=refresh_network_card, outputs=nw_device_card)

        batt_popup_st = gr.HTML(visible=False)
        batt_beep_st = gr.Textbox(visible=False)
        # No status strip on this page to piggyback on; poll gently on its own.
        batt_timer_st = gr.Timer(30.0)
        batt_timer_st.tick(fn=check_battery_warning, outputs=[batt_popup_st, batt_beep_st])
        batt_beep_st.change(fn=None, inputs=batt_beep_st, outputs=None, js=_BATTERY_BEEP_JS)
        settings_demo.load(fn=check_battery_warning, outputs=[batt_popup_st, batt_beep_st])
        settings_demo.load(fn=None, js=_BATTERY_INIT_JS)

    return demo
