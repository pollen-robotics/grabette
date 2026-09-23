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
/* "Power Off" is the last navbar entry — push it to the far right and tint it
   red so it reads as separate from the normal pages. Best-effort: relies on the
   navbar being a flex row (gradio 6.x); the 🔴 label is the guaranteed cue. */
#grabette-nav a:last-child {
    margin-left: auto !important;
    color: #f87171 !important;
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

# Same widget, denser skin (see webauth._COMPACT_CSS). The home page shows the
# account as a status line; the full card still lives on Settings.
_HF_AUTH_IFRAME_COMPACT = (
    '<iframe src="/api/hf-auth/widget?compact=1" scrolling="no"'
    ' onload="var f=this;(function r(){'
    'if(!document.contains(f))return;'
    'try{f.style.height=f.contentDocument.body.scrollHeight+4+\'px\';}catch(e){}'
    'setTimeout(r,400);})()"'
    ' style="width:100%;border:none;min-height:56px;display:block;"></iframe>'
)

# Sized to sit beside the login line rather than dominate the page. Still a
# plain link out: grabette-fleet is OAuth-gated and this dashboard is served
# over plain HTTP, so its login cannot render in an iframe here.
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
    '<iframe id="urdf-viewer" src="/viewer" '
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

_HF_AUTH_IFRAME = (
    '<iframe src="/api/hf-auth/widget" scrolling="no"'
    ' onload="var f=this;(function r(){'
    'if(!document.contains(f))return;'
    'try{f.style.height=f.contentDocument.body.scrollHeight+10+\'px\';}catch(e){}'
    'setTimeout(r,400);})()"'
    ' style="width:100%;border:none;min-height:160px;"></iframe>'
)


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

    def get_capture_status():
        state = client.get_state()
        if state is None:
            return "○ Idle"
        cap = state.get("capture", {})
        if cap.get("is_capturing", False):
            parts = [
                f"● RECORDING  {cap.get('episode_id', '')}",
                f"Duration: {cap.get('duration_seconds', 0):.1f}s",
                f"Frames: {cap.get('frame_count', 0)}  |  IMU: {cap.get('imu_sample_count', 0)}",
            ]
            if cap.get("angle_sample_count", 0):
                parts[-1] += f"  |  Angle: {cap['angle_sample_count']}"
            return "\n".join(parts)
        # Not capturing is not the same as free. A device tied up by an upload
        # (or held back by a hardware fault) used to read "○ Idle" here, which is
        # precisely the reading that gets a recording started on top of one.
        blocked = cap.get("blocked_reason") or ""
        if blocked:
            return f"⛔ Cannot record — {blocked}"
        return "○ Idle"

    def on_toggle_capture(session_id):
        state = client.get_state()
        capturing = state.get("capture", {}).get("is_capturing", False) if state else False
        if capturing:
            client.stop_capture()
            rows, move_dd, _task_header, desc, *_ = _refresh_episode_table(session_id)
            return gr.update(value="Start Capture", variant="primary"), rows, move_dd, desc
        else:
            client.start_capture(task_id=session_id or None)
            return gr.update(value="Stop Capture", variant="stop"), gr.update(), gr.update(), gr.update()

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


    def on_start_stop_session(current_task):
        cap_session = client.get_session_status()
        if cap_session.get("active"):
            client.stop_session()
            _, _, _, _, cap_title, _ = _refresh_episode_table(current_task)
            return (
                gr.update(value="▶ Start Session", variant="secondary"),
                gr.update(value=cap_title),
                gr.update(value=""),
            )
        else:
            result = client.start_session(task_id=current_task or None)
            if "error" in result:
                return gr.update(), gr.skip(), gr.skip()
            task_name = result.get("task_name", "")
            return (
                gr.update(value="■ Stop Session", variant="stop"),
                gr.update(value=f"### Capture a new episode for *{task_name}*"),
                gr.update(value=_session_banner_html(task_name, 0)),
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

    def _task_choices(sessions):
        return [(s["name"], s["id"]) for s in sessions]

    def _refresh_episode_table(session_id, sessions=None):
        if sessions is None:
            sessions = _get_sessions()
        rows = []
        task_name = ""
        task_description = ""
        # The device always has Unassigned, so an empty list means the API call
        # failed — never that there is nothing to show. Saying so beats a blank
        # page that looks exactly like "no episodes recorded yet".
        api_down = not sessions
        for s in sessions:
            if s["id"] == session_id:
                task_name = s.get("name", "")
                task_description = s.get("description", "")
                for ep in s.get("episodes", []):
                    rows.append([
                        False,
                        ep["episode_id"],
                        f"{ep['duration_seconds']:.1f}s",
                        ep["frame_count"],
                        ep["imu_sample_count"],
                        ep.get("angle_sample_count", 0),
                    ])
                break
        rows.reverse()
        move_choices = _task_choices(sessions)
        move_dd = gr.update(
            choices=move_choices,
            value=move_choices[0][1] if move_choices else None,
        )
        task_header = f"## Task: {task_name}" if task_name else ""
        cap_title = "### Capture" if not task_name else f"### Capture a new episode for *{task_name}*"
        count = len(rows)
        count_str = f"{count} episode" + ("s" if count != 1 else "")
        ep_title = f"## Episodes for *{task_name}*" if task_name else "## Episodes"
        desc_parts = []
        if api_down:
            desc_parts.append(
                "⚠️ **Could not reach the grabette API** — the task list below is "
                "empty because the call failed, not because there is nothing "
                "recorded. Check the daemon log for the error."
            )
        if task_description:
            desc_parts.append(f"**Task description:** {task_description}")
        if not api_down:
            desc_parts.append(f"*{count_str} recorded*")
        desc = "\n\n".join(desc_parts)
        return rows, move_dd, task_header, desc, cap_title, ep_title

    def _session_banner_html(task_name: str, count: int = 0) -> str:
        ep_str = f"{count} episode{'s' if count != 1 else ''}"
        return (
            '<div style="padding:0.85rem 1.2rem;background:#1c1710;border-radius:10px;'
            'border:1px solid #f97316;display:flex;align-items:center;gap:0.9rem;">'
            '<span style="font-size:1.6rem;line-height:1;filter:brightness(0) invert(1);">🔒</span>'
            '<div>'
            '<div style="font-weight:700;color:#fb923c;font-size:0.95rem;">Active session</div>'
            '<div style="color:#e2e8f0;font-size:0.88rem;margin-top:2px;">'
            f'All recordings are saved to: <strong style="color:#fff;">{task_name}</strong>'
            '</div>'
            '<div style="color:#e2e8f0;font-size:0.88rem;margin-top:3px;">'
            f'Session: <strong style="color:#fb923c;">{ep_str} recorded</strong>'
            '</div>'
            '</div>'
            '</div>'
        )

    def refresh_tasks(stored_id: str = ""):
        sessions = _get_sessions()
        choices = _task_choices(sessions)
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
        rows, move_dd, task_header, desc, cap_title, ep_title = _refresh_episode_table(value, sessions)
        return gr.update(choices=choices, value=value), task_header, cap_title, desc, ep_title, rows, move_dd

    def on_task_select(session_id):
        cap_session = client.get_session_status()
        session_active = cap_session.get("active", False)
        if not session_active and session_id:
            client.set_active_task(session_id)
        rows, move_dd, task_header, desc, cap_title, ep_title = _refresh_episode_table(session_id)
        if session_active:
            return task_header, gr.skip(), desc, ep_title, rows, move_dd
        return task_header, cap_title, desc, ep_title, rows, move_dd

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
        episode_ids = _get_selected_ids(table_data)
        if not episode_ids:
            return None
        return client.download_episodes(episode_ids)

    def on_delete_episode(table_data, session_id):
        episode_ids = _get_selected_ids(table_data)
        if not episode_ids:
            return "No episode selected", gr.update(), gr.update(), gr.update()
        errors = []
        for eid in episode_ids:
            result = client.delete_episode(eid)
            if "error" in result:
                errors.append(f"{eid}: {result['error']}")
        rows, move_dd, _th, desc, *_ = _refresh_episode_table(session_id)
        # Force the interactive dataframe to re-render: after the user ticks
        # rows it holds "dirty" client-side state that a bare list won't
        # overwrite, so the deleted rows (and their checkboxes) would linger.
        table_upd = gr.update(value=rows)
        if errors:
            return "Errors: " + "; ".join(errors), table_upd, move_dd, desc
        return f"Deleted {len(episode_ids)} episode(s)", table_upd, move_dd, desc

    def on_move_episodes(table_data, target_session_id, current_session_id):
        episode_ids = _get_selected_ids(table_data)
        if not episode_ids:
            return "No episode selected", gr.update(), gr.update(), gr.update()
        if not target_session_id:
            return "No target task", gr.update(), gr.update(), gr.update()
        result = client.move_episodes(episode_ids, target_session_id)
        if "error" in result:
            return f"Error: {result['error']}", gr.update(), gr.update(), gr.update()
        rows, move_dd, _th, desc, *_ = _refresh_episode_table(current_session_id)
        msg = f"Moved {len(result.get('moved', episode_ids))} episode(s)"
        if result.get("skipped"):
            msg += f" ({len(result['skipped'])} not found here)"
        # An episode recorded with another grabette only gets refiled HERE. Unless
        # the peer is refiled too, the two devices end up reporting the same
        # episode under different tasks, which the fleet flags as a split.
        shared = result.get("shared") or []
        if shared:
            peers = ", ".join(sorted({p for s in shared for p in s["peers"]}))
            msg += (f" — warning: {len(shared)} of them were recorded with {peers};"
                    " refile them there too, or the pair ends up split across tasks")
        # gr.update(value=...) forces the interactive dataframe to drop its
        # dirty checkbox state so the moved rows actually disappear.
        return msg, gr.update(value=rows), move_dd, desc

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

    def get_episode_status_bar():
        """(status_bar_html, battery_popup, beep_signal) from ONE system-info read.

        The battery warning piggybacks on this 3 s poll rather than a dedicated
        timer, so a single get_system_info() feeds both the strip and the popup
        (no redundant I2C read).
        """
        info = client.get_system_info()
        bar = _status_bar_html(
            info,
            client.get_oakd_status(),
            client.get_camera_status(),
        )
        popup_update, beep_signal = _battery_popup_html(info)
        return bar, popup_update, beep_signal

    # ── WiFi network info (Settings page) ────────────────────────────

    def get_wifi_network_info():
        status = client.wifi_status()
        info = client.get_system_info() or {}
        hostname = info.get("hostname", "—")
        ssid = status.get("ssid") or "—"
        ip = status.get("ip") or info.get("ip") or "—"
        return (
            f"**Hostname:** {hostname}  \n"
            f"**Current network:** {ssid}  \n"
            f"**IP address:** {ip}"
        )

    # ── Power off ─────────────────────────────────────────────────────

    def _poweroff_notice(text: str, color: str = "#f97316") -> str:
        return (
            f"<div style='max-width:520px;margin-top:0.75rem;padding:0.85rem 1.1rem;"
            f"background:#1e293b;border-left:4px solid {color};border-radius:8px;"
            f"color:#e2e8f0;font-size:0.92rem;'>{text}</div>"
        )

    def load_poweroff_page():
        """Arm the button when idle; disable + warn while a recording is active."""
        cap = (client.get_state() or {}).get("capture", {})
        if cap.get("is_capturing") or cap.get("is_starting"):
            return (
                gr.update(
                    value=_poweroff_notice(
                        "A recording is in progress — stop the capture before powering off."
                    ),
                    visible=True,
                ),
                gr.update(interactive=False, variant="secondary"),
            )
        return gr.update(value="", visible=False), gr.update(interactive=True, variant="stop")

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
                    "Device is shutting down. This page will stop responding shortly — "
                    "wait ~20 s, then it is safe to unplug.",
                    "#22c55e",
                ),
                visible=True,
            ),
            gr.update(interactive=False, variant="secondary"),
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
        gr.HTML(_TITLE_HTML)

        with gr.Column(elem_id="ov-page"):

            # ── Camera | 3D model | Device | Health ───────────────────
            # min_width is what makes this responsive: four columns on a
            # laptop, two on a tablet, one on a phone, decided by gradio from
            # the width each column says it needs.
            with gr.Row(equal_height=False, elem_classes="ov-tiles"):
                with gr.Column(scale=1, min_width=230):
                    gr.HTML(_section_label("Camera"))
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
                        gr.Button("Change network", link="/settings", size="sm")
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
        gr.HTML(_TITLE_HTML)

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

    with demo.route("Episodes") as episodes_demo:
        gr.Navbar(main_page_name="Overview", elem_id="grabette-nav")
        gr.HTML(_TITLE_HTML)
        episode_status_bar = gr.HTML("")

        # ── Main layout ───────────────────────────────────────────────
        with gr.Row():

            # ── LEFT: Tasks ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=200, elem_id="tasks-col"):
                gr.Markdown("## Tasks")
                task_list = gr.Radio(choices=[], label=None, container=False)
                # Remembers, per browser, which task was selected so a page
                # refresh stays on it instead of snapping back to the first
                # task. Independent of the (server-side) capture session.
                selected_task_state = gr.BrowserState(
                    "", storage_key="grabette_selected_task",
                )
                # Tasks are created/edited on the fleet (grabette-fleet), not on
                # the device — here we only pick a task to view its episodes.

            # ── RIGHT: Episodes ──────────────────────────────────────
            with gr.Column(scale=3):

                # Capture (always at top so the primary action is prominent)
                session_banner = gr.HTML("")
                capture_title = gr.Markdown("### Capture")
                with gr.Row():
                    capture_box = gr.Textbox(
                        label="Status", lines=2, interactive=False, scale=3,
                    )
                    with gr.Column(scale=1, min_width=150):
                        session_btn = gr.Button("▶ Start Session", variant="secondary")
                        toggle_btn = gr.Button("Start Capture", variant="primary")

                task_header_md = gr.Markdown("", visible=False)

                gr.HTML("<div style='margin-top:2rem;'></div>")
                episodes_title = gr.Markdown("## Episodes")
                task_desc_md = gr.Markdown("")

                episodes_table = gr.Dataframe(
                    headers=["✓", "Episode ID", "Duration", "Frames", "IMU", "Angle"],
                    datatype=["bool", "str", "str", "number", "number", "number"],
                    interactive=True,
                    static_columns=[1, 2, 3, 4, 5],
                    col_count=(6, "fixed"),
                    show_search="filter",
                )
                with gr.Row():
                    replay_btn = gr.Button("▶ Replay", size="md", scale=1)
                    with gr.Accordion("Download", open=False):
                        dl_btn = gr.Button("Download selected", size="sm")
                        dl_file = gr.File(label="Download")
                    with gr.Accordion("Move to Task", open=False):
                        move_target_dd = gr.Dropdown(label="Move to task", interactive=True)
                        move_btn = gr.Button("Move", size="sm")
                    with gr.Accordion("Delete", open=False):
                        del_episode_btn = gr.Button("Delete selected", variant="stop", size="sm")

                episode_msg = gr.Textbox(show_label=False, interactive=False, max_lines=1)

                # Replay panel (hidden until replay starts)
                with gr.Group(visible=False) as replay_panel:
                    gr.Markdown("#### Replay")
                    replay_video = gr.HTML(value="")
                    gr.HTML(
                        '<iframe src="/charts/imu" '
                        'style="width:100%;height:300px;border:none;'
                        'border-radius:8px;background:transparent;"></iframe>'
                    )
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
                    with gr.Row():
                        replay_pause_btn = gr.Button("Pause", size="sm")
                        replay_stop_btn = gr.Button("Stop Replay", variant="stop", size="sm")
                replay_timer = gr.Timer(0.5, active=False)

        # ── Wire events ───────────────────────────────────────────────

        task_list.change(
            fn=on_task_select, inputs=task_list,
            outputs=[task_header_md, capture_title, task_desc_md, episodes_title, episodes_table, move_target_dd],
        )
        # Persist the current selection in the browser so a refresh keeps it.
        task_list.change(fn=lambda v: v, inputs=task_list, outputs=selected_task_state)

        session_btn.click(
            fn=on_start_stop_session,
            inputs=[task_list],
            outputs=[session_btn, capture_title, session_banner],
        )

        toggle_btn.click(
            fn=on_toggle_capture,
            inputs=[task_list],
            outputs=[toggle_btn, episodes_table, move_target_dd, task_desc_md],
        )

        dl_btn.click(fn=on_download_episodes, inputs=episodes_table, outputs=dl_file)
        del_episode_btn.click(
            fn=on_delete_episode, inputs=[episodes_table, task_list],
            outputs=[episode_msg, episodes_table, move_target_dd, task_desc_md],
        )
        move_btn.click(
            fn=on_move_episodes, inputs=[episodes_table, move_target_dd, task_list],
            outputs=[episode_msg, episodes_table, move_target_dd, task_desc_md],
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

        _capture_state = {"was_active": False}

        def get_capture_status_and_active_task(current_task):
            state = client.get_state()
            cap_session = client.get_session_status()
            cap = (state or {}).get("capture", {})
            is_recording = cap.get("is_capturing", False)
            is_starting = cap.get("is_starting", False)

            # Detect recording stop to refresh episode table
            currently_active = is_recording or is_starting
            just_stopped = _capture_state["was_active"] and not currently_active
            _capture_state["was_active"] = currently_active
            if just_stopped and current_task:
                rows, move_dd_upd, _task_header, desc, *_ = _refresh_episode_table(current_task)
                table_update = rows
                move_dd_update = move_dd_upd
                desc_update = desc
            else:
                table_update = gr.skip()
                move_dd_update = gr.skip()
                desc_update = gr.skip()

            # Build status text and toggle button state
            if is_starting:
                status = "◌ Initializing depth camera…"
                toggle_btn_update = gr.update(interactive=False, value="Start Capture", variant="primary")
            elif is_recording:
                parts = [
                    f"● RECORDING  {cap.get('episode_id', '')}",
                    f"Duration: {cap.get('duration_seconds', 0):.1f}s",
                    f"Frames: {cap.get('frame_count', 0)}  |  IMU: {cap.get('imu_sample_count', 0)}",
                ]
                if cap.get("angle_sample_count", 0):
                    parts[-1] += f"  |  Angle: {cap['angle_sample_count']}"
                status = "\n".join(parts)
                toggle_btn_update = gr.update(interactive=True, value="Stop Capture", variant="stop")
            else:
                status = "○ Idle"
                toggle_btn_update = gr.update(interactive=True, value="Start Capture", variant="primary")

            # Session button + capture title + banner sync
            if cap_session.get("active"):
                task_name = cap_session.get("task_name", "")
                # The count already excludes any in-progress capture — episodes
                # are registered only once recording stops.
                display_count = cap_session.get("count", 0)
                sess_btn = gr.update(value="■ Stop Session", variant="stop")
                cap_title = gr.update(value=f"### Capture a new episode for *{task_name}*")
                banner = gr.update(value=_session_banner_html(task_name, display_count))
                task_update = gr.skip()
            else:
                active = client.get_active_task()
                sess_btn = gr.update(value="▶ Start Session", variant="secondary")
                cap_title = gr.skip()
                banner = gr.update(value="")
                # The active task can change out-of-band — e.g. a fleet-driven
                # (physical-button) recording creates a task locally via
                # get_or_create_task. Pointing the dropdown at it WITHOUT also
                # refreshing its choices makes Gradio raise "Value ... not in
                # the list of choices" every tick until reload. So when the
                # active task changed, refresh the choices in the same update
                # (and never set a value that isn't among them).
                if active and active != current_task:
                    choices = _task_choices(_get_sessions())
                    if any(cid == active for _n, cid in choices):
                        task_update = gr.update(choices=choices, value=active)
                    else:
                        task_update = gr.skip()
                else:
                    task_update = gr.skip()

            return status, task_update, sess_btn, cap_title, banner, toggle_btn_update, table_update, move_dd_update, desc_update

        capture_timer = gr.Timer(0.5)
        capture_timer.tick(
            fn=get_capture_status_and_active_task,
            inputs=[task_list],
            outputs=[capture_box, task_list, session_btn, capture_title, session_banner, toggle_btn,
                     episodes_table, move_target_dd, task_desc_md],
        )

        batt_popup_ep = gr.HTML(visible=False)
        batt_beep_ep = gr.Textbox(visible=False)
        batt_beep_ep.change(fn=None, inputs=batt_beep_ep, outputs=None, js=_BATTERY_BEEP_JS)
        demo.load(fn=None, js=_BATTERY_INIT_JS)

        # Battery warning rides on the status-bar poll (one system-info read
        # feeds both the strip and the popup) — no dedicated battery timer.
        status_bar_outputs = [episode_status_bar, batt_popup_ep, batt_beep_ep]
        status_bar_timer = gr.Timer(3.0)
        status_bar_timer.tick(fn=get_episode_status_bar, outputs=status_bar_outputs)

        episodes_demo.load(fn=refresh_tasks, inputs=[selected_task_state], outputs=[task_list, task_header_md, capture_title, task_desc_md, episodes_title, episodes_table, move_target_dd])
        # One load fills the strip AND the battery popup/beep — get_episode_status_bar
        # now returns all three from a single system-info read, so no separate
        # check_battery_warning load is needed here.
        episodes_demo.load(fn=get_episode_status_bar, outputs=status_bar_outputs)

    # (Datasets page removed — dataset generation is done on the fleet.)

    # ══════════════════════════════════════════════════════════════════
    # Page 4 — Live View
    # ══════════════════════════════════════════════════════════════════

    with demo.route("Live View") as live_demo:
        gr.Navbar(main_page_name="Overview", elem_id="grabette-nav")
        gr.HTML(_TITLE_HTML)

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
    # Page 5 — Settings
    # ══════════════════════════════════════════════════════════════════

    with demo.route("Settings") as settings_demo:
        gr.Navbar(main_page_name="Overview", elem_id="grabette-nav")
        gr.HTML(_TITLE_HTML)

        with gr.Row(equal_height=False):

            # ── HuggingFace Account ───────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("## HuggingFace Account")
                gr.HTML(_HF_AUTH_IFRAME)

            # ── WiFi ─────────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("## WiFi")
                wifi_network_info = gr.Markdown("*Loading…*")
                gr.HTML(_WIFI_SETTINGS_HTML)

        settings_demo.load(fn=get_wifi_network_info, outputs=wifi_network_info)

        batt_popup_st = gr.HTML(visible=False)
        batt_beep_st = gr.Textbox(visible=False)
        # No status strip on this page to piggyback on; poll gently on its own.
        batt_timer_st = gr.Timer(30.0)
        batt_timer_st.tick(fn=check_battery_warning, outputs=[batt_popup_st, batt_beep_st])
        batt_beep_st.change(fn=None, inputs=batt_beep_st, outputs=None, js=_BATTERY_BEEP_JS)
        settings_demo.load(fn=check_battery_warning, outputs=[batt_popup_st, batt_beep_st])
        settings_demo.load(fn=None, js=_BATTERY_INIT_JS)

    # ══════════════════════════════════════════════════════════════════
    # Page 6 — Power Off
    # ══════════════════════════════════════════════════════════════════

    with demo.route("🔴 Power Off") as poweroff_demo:
        gr.Navbar(main_page_name="Overview", elem_id="grabette-nav")
        gr.HTML(_TITLE_HTML)

        gr.HTML(
            "<div style='max-width:520px;margin-top:1rem;padding:1.5rem;"
            "background:#1c1310;border:1px solid #991b1b;border-radius:12px;'>"
            "<h2 style='margin:0 0 0.5rem;color:#f87171;'>Power off the device</h2>"
            "<p style='color:#e2e8f0;margin:0;font-size:0.95rem;'>"
            "This performs a clean shutdown of the Raspberry Pi. Once it has halted "
            "you can safely disconnect power.</p></div>"
        )

        poweroff_msg = gr.HTML(value="", visible=False)
        with gr.Row():
            poweroff_btn = gr.Button("Power off now", variant="stop", scale=0)

        poweroff_btn.click(fn=on_poweroff, outputs=[poweroff_msg, poweroff_btn])
        poweroff_demo.load(fn=load_poweroff_page, outputs=[poweroff_msg, poweroff_btn])

    return demo
