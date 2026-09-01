"""Gradio dashboard for Grabette — camera view, capture controls, task/session/episode management."""

from __future__ import annotations

import html
import io
import logging
import math

import gradio as gr
from PIL import Image

from grabette.config import settings
from grabette.ui.api_client import GrabetteClient

logger = logging.getLogger(__name__)


# ── Visual charter ────────────────────────────────────────────────────────
# Aligned with the grabette-fleet dashboard (and the Bluetooth tool, which
# already mirrors it): the same #1a1a2e→#16213e gradient, translucent white
# cards, #c3cbe0 / #a0aec0 text ramp and #ffcc4d primary accent. Operators move
# between the three all day; they should look like one product.
FLEET_BG = "linear-gradient(135deg,#1a1a2e,#16213e)"
FLEET_SURFACE = "#16213e"
FLEET_CARD = "rgba(255,255,255,.06)"
FLEET_BORDER = "rgba(255,255,255,.12)"
FLEET_TEXT = "#ffffff"
FLEET_TEXT_SOFT = "#c3cbe0"
FLEET_MUTED = "#8b98ad"
FLEET_ACCENT = "#ffcc4d"

# Both the light and the dark value of every token is set to the same colour:
# the charter is a dark one, and the dashboard must not half-flip when the
# browser is set to light (which is what the hardcoded slate cards used to do).
def fleet_theme():
    # System fonts only, no GoogleFont: the robot regularly runs offline, and a
    # webfont that half-loads there falls back to serif (the same trap that made
    # _TITLE_HTML pin its own stack).
    return gr.themes.Base(
        font=["-apple-system", "system-ui", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
    ).set(
        body_background_fill=FLEET_BG,
        body_background_fill_dark=FLEET_BG,
        background_fill_primary=FLEET_BG,
        background_fill_primary_dark=FLEET_BG,
        background_fill_secondary=FLEET_CARD,
        background_fill_secondary_dark=FLEET_CARD,
        block_background_fill=FLEET_CARD,
        block_background_fill_dark=FLEET_CARD,
        block_border_color=FLEET_BORDER,
        block_border_color_dark=FLEET_BORDER,
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_text_color=FLEET_MUTED,
        block_label_text_color_dark=FLEET_MUTED,
        block_title_text_color=FLEET_TEXT_SOFT,
        block_title_text_color_dark=FLEET_TEXT_SOFT,
        body_text_color=FLEET_TEXT,
        body_text_color_dark=FLEET_TEXT,
        body_text_color_subdued=FLEET_MUTED,
        body_text_color_subdued_dark=FLEET_MUTED,
        border_color_primary=FLEET_BORDER,
        border_color_primary_dark=FLEET_BORDER,
        panel_background_fill=FLEET_CARD,
        panel_background_fill_dark=FLEET_CARD,
        input_background_fill=FLEET_SURFACE,
        input_background_fill_dark=FLEET_SURFACE,
        button_primary_background_fill=FLEET_ACCENT,
        button_primary_background_fill_dark=FLEET_ACCENT,
        button_primary_text_color="#1a1a2e",
        button_primary_text_color_dark="#1a1a2e",
        button_secondary_background_fill="rgba(255,255,255,.08)",
        button_secondary_background_fill_dark="rgba(255,255,255,.08)",
        button_secondary_text_color=FLEET_TEXT_SOFT,
        button_secondary_text_color_dark=FLEET_TEXT_SOFT,
        # variant="stop" (Stop Capture, Power off now) maps to the "cancel"
        # family — unset, it renders as an ordinary grey button, which is the
        # last thing a shutdown control should look like.
        button_cancel_background_fill="#ef4444",
        button_cancel_background_fill_dark="#ef4444",
        button_cancel_background_fill_hover="#dc2626",
        button_cancel_background_fill_hover_dark="#dc2626",
        button_cancel_text_color="#ffffff",
        button_cancel_text_color_dark="#ffffff",
        button_cancel_border_color="#ef4444",
        button_cancel_border_color_dark="#ef4444",
        block_radius="14px",
        button_large_radius="8px",
        button_small_radius="8px",
        # The episode table is its own token family — left at the defaults it
        # renders a white sheet in the middle of the dark page.
        table_even_background_fill="rgba(255,255,255,.03)",
        table_even_background_fill_dark="rgba(255,255,255,.03)",
        table_odd_background_fill="rgba(255,255,255,.07)",
        table_odd_background_fill_dark="rgba(255,255,255,.07)",
        table_border_color=FLEET_BORDER,
        table_border_color_dark=FLEET_BORDER,
        table_text_color=FLEET_TEXT,
        table_text_color_dark=FLEET_TEXT,
        table_row_focus="rgba(59,130,246,.22)",
        table_row_focus_dark="rgba(59,130,246,.22)",
        table_radius="12px",
    )


# Applied by app/main.py at mount time — see create_ui. (This CSS silently did
# nothing for as long as it was passed to gr.Blocks(css=...) under Gradio 6.)
APP_CSS = """
/* The gradient has to be painted on the app shell too: Gradio's container sits
   on top of <body> with its own opaque fill, which would hide it. */
.gradio-container, .app, .main, body {
    background: linear-gradient(135deg,#1a1a2e,#16213e) !important;
}
.gradio-container { max-width: 1200px !important; }
/* Blocks that only group other blocks (rows, columns) must stay transparent —
   otherwise every nesting level stacks another translucent white veil. */
.gradio-container .form, .gradio-container .gap,
.gradio-container div.block:not(.padded) { background: transparent; }
/* Gradio dims disabled buttons by lightening them, which on a dark ground
   produces a near-white bar with unreadable text. Darken instead. */
.gradio-container button:disabled, .gradio-container button[disabled] {
    background: rgba(255,255,255,.05) !important;
    color: #8b98ad !important;
    opacity: 1 !important;
}
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
#grabette-nav {
    background: rgba(255,255,255,.04) !important;
    border-bottom: 1px solid rgba(255,255,255,.12) !important;
}
#grabette-nav a { color: #c3cbe0 !important; }
#grabette-nav a:hover { color: #ffffff !important; }
"""

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
# The Bluetooth provisioning tool (docs/index.html, published to GitHub Pages).
# It has to be served from an https origin — Web Bluetooth is secure-context
# only — which is why it lives there and not on the device.
_BT_TOOL_URL = "https://pollen-robotics.github.io/grabette/"

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
        f"letter-spacing:0.09em;color:{FLEET_MUTED};margin-bottom:0.3rem;'>"
        f"{text}</div>"
    )


def _battery_colors(pct: float, charging: bool | None) -> tuple[str, str]:
    """(value colour, border colour) for a battery level — one source of truth.

    Charging is green whatever the level: the number is on its way up, and a red
    badge on a device that is plugged in only teaches operators to ignore it.
    """
    if charging or pct > 40:
        return "#6ee7b7", "rgba(16,185,129,.45)"
    if pct > 20:
        return "#fcd34d", "rgba(245,158,11,.5)"
    return "#fca5a5", "rgba(239,68,68,.55)"


def _info_card(
    label: str,
    value: str,
    *,
    value_color: str = FLEET_TEXT,
    border_color: str = FLEET_BORDER,
    extra_style: str = "",
    title: str = "",
) -> str:
    """One translucent fleet-charter card: dim uppercase label over a bold value.

    The single card shape behind every status strip in the dashboard (Home,
    Episodes, Live View), so a badge means the same thing on every page.

    ``title`` carries the long form (a hardware fault's full explanation) as a
    hover tooltip: a card is one line and truncates, so a message that matters
    cannot live in the value itself.
    """
    tip = f" title=\"{html.escape(title, quote=True)}\"" if title else ""
    return (
        f"<div{tip} style='background:{FLEET_CARD};border-radius:14px;padding:0.6rem 1rem;"
        f"border:1px solid {border_color};flex:1;min-width:0;{extra_style}'>"
        f"<div style='font-size:0.65rem;text-transform:uppercase;letter-spacing:0.09em;"
        f"color:{FLEET_MUTED};margin-bottom:0.2rem;'>{label}</div>"
        f"<div style='font-size:0.9rem;font-weight:700;color:{value_color};"
        f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>{value}</div>"
        f"</div>"
    )


def _status_bar_html(sys_info, oakd_status, cam_status):
    """Build the Episodes status strip (battery + RGB + OAK-D) from already-fetched dicts.

    Pure function (no network calls) so it can be unit-tested. Each argument
    may be None when the corresponding API call failed.
    """

    # (value color, border color). Neutral gray covers off / N/A / unknown.
    # Palette matches the fleet's pill colours: soft foregrounds on translucent
    # cards rather than the saturated hue on near-black it used to be.
    GRAY = ("#a0aec0", "rgba(255,255,255,.14)")
    GREEN = ("#6ee7b7", "rgba(16,185,129,.45)")
    ORANGE = ("#fcd34d", "rgba(245,158,11,.5)")
    RED = ("#fca5a5", "rgba(239,68,68,.55)")

    def _badge(label, value, colors, title=""):
        value_color, border_color = colors
        return _info_card(label, value, value_color=value_color,
                          border_color=border_color, title=title)

    # Battery (⚡ + green while charging, regardless of level)
    if sys_info and "battery_pct" in sys_info:
        pct = sys_info["battery_pct"]
        charging = sys_info.get("battery_charging")
        value = f"⚡ {pct} %" if charging else f"{pct} %"
        batt_badge = _badge("Battery", value, _battery_colors(pct, charging))
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

    # OAK-D (5-state: fault / connected / starting / off / error; N/A when
    # unsupported). The fault comes FIRST: it is the one state where the device
    # refuses to record, so it must not be masked by "Off" after a power-down.
    if not oakd_status or not oakd_status.get("supported"):
        oakd_badge = _badge("OAK-D", "N/A", GRAY)
    elif oakd_status.get("hardware_error"):
        oakd_badge = _badge("OAK-D", "Cannot record", RED,
                            title=oakd_status["hardware_error"])
    elif oakd_status.get("initialized"):
        oakd_badge = _badge("OAK-D", "Connected", GREEN)
    elif oakd_status.get("initializing"):
        oakd_badge = _badge("OAK-D", "Starting…", ORANGE)
    elif oakd_status.get("enabled"):
        oakd_badge = _badge("OAK-D", "Error", RED)
    else:
        oakd_badge = _badge("OAK-D", "Off", GRAY)

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
        """Compute the OAK-D toggle button's appearance + OAK data row visibility.

        Returns (button_update, oak_row_visibility) so callers can keep the
        depth/IMU/accelerometer row hidden until the camera is enabled.
        """
        s = client.get_oakd_status() or {}
        if not s.get("supported"):
            return (
                gr.update(
                    value="OAK-D not available",
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
            label = "OAK-D: ON" + ("  (busy)" if busy else "  — click to disable")
            variant = "primary"
        else:
            label = "OAK-D: OFF" + ("  (busy)" if busy else "  — click to enable")
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
            logger.warning("OAK-D toggle failed: %s", result["error"])
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
                "background:#16213e;border:1px solid rgba(248,113,113,.55);"
                "border-radius:14px;padding:16px 20px;max-width:260px;"
                "box-shadow:0 10px 30px rgba(0,0,0,.45);'>"
                "<div style='font-weight:700;color:#fca5a5;font-size:1rem;"
                "margin-bottom:4px;'>Battery low</div>"
                f"<div style='font-size:0.88rem;color:{FLEET_TEXT_SOFT};'>"
                f"{pct} % — please charge soon.</div>"
                "</div>"
            )
            return gr.update(visible=True, value=html), f"{pct}|{_batt_beep['n']}"
        return gr.update(visible=False), ""

    def get_system_bar():
        """Returns (system_bar_html, battery_popup_update, beep_signal)."""
        info = client.get_system_info()
        if info is None:
            bar = (f"<p style='color:{FLEET_MUTED};font-size:0.85rem;margin:0.5rem 0;'>"
                   "System disconnected</p>")
            return bar, gr.update(visible=False), ""

        def _card(label, value, extra_style=""):
            return _info_card(label, value, extra_style=extra_style)

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
            batt_color, batt_border = _battery_colors(pct, charging)
            batt_value = f"⚡ {pct} %" if charging else f"{pct} %"
            parts.append(_info_card(
                "Battery", batt_value, value_color=batt_color, border_color=batt_border,
            ))

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

    # ── Home page essentials (battery + identity + network) ──────────

    def get_home_status():
        """(cards_html, battery_popup, beep_signal) — the four things you land on.

        Battery, hostname, IP and network, in that order: the state you check
        before touching anything, and — since there is no Settings page any more
        — the only place the device's identity is written down. The battery
        popup piggybacks on the same system-info read, as on the other pages.
        """
        info = client.get_system_info() or {}
        status = client.wifi_status()

        parts = []

        if "battery_pct" in info:
            pct = info["battery_pct"]
            charging = info.get("battery_charging")
            color, border = _battery_colors(pct, charging)
            parts.append(_info_card(
                "Battery", f"⚡ {pct} %" if charging else f"{pct} %",
                value_color=color, border_color=border,
            ))
        else:
            parts.append(_info_card("Battery", "N/A", value_color=FLEET_MUTED))

        parts.append(_info_card("Hostname", html.escape(info.get("hostname") or "—")))
        parts.append(_info_card("IP address",
                                html.escape(status.get("ip") or info.get("ip") or "—")))

        # The network card doubles as the reachability verdict: on the hotspot
        # the dashboard is only reachable from whoever is joined to grabette
        # itself, which is worth seeing before wondering why nothing syncs.
        mode = status.get("mode")
        ssid = status.get("ssid")
        if mode == "connected" and ssid:
            net_value, net_color = html.escape(ssid), "#6ee7b7"
        elif mode == "hotspot":
            net_value, net_color = "Hotspot — not on a network", "#fcd34d"
        else:
            net_value, net_color = "Offline", "#fca5a5"
        parts.append(_info_card("Network", net_value, value_color=net_color))

        bar = (
            "<div style='display:flex;flex-direction:row;gap:0.5rem;flex-wrap:wrap;"
            "margin:0.25rem 0 1rem;'>" + "".join(parts) + "</div>"
        )
        popup_update, beep_signal = _battery_popup_html(info)
        return bar, popup_update, beep_signal

    # ── Power off ─────────────────────────────────────────────────────

    def _poweroff_notice(text: str, color: str = "#f97316") -> str:
        return (
            f"<div style='max-width:520px;margin-top:0.75rem;padding:0.85rem 1.1rem;"
            f"background:{FLEET_CARD};border-left:4px solid {color};border-radius:12px;"
            f"color:{FLEET_TEXT_SOFT};font-size:0.92rem;'>{text}</div>"
        )

    def _poweroff_header(hostname: str) -> str:
        """The shutdown card, with the device it will shut down named in it.

        Operators keep several grabettes open in several tabs; "Power off the
        device" alone does not say WHICH, and the wrong tab costs a session.
        """
        who = (
            f"<div style='font-size:1.05rem;font-weight:700;color:#fff;"
            f"margin:0.15rem 0 0.6rem;'>{html.escape(hostname)}</div>"
            if hostname else ""
        )
        return (
            "<div style='max-width:520px;margin-top:1rem;padding:1.5rem;"
            "background:rgba(239,68,68,.1);border:1px solid rgba(248,113,113,.45);"
            "border-radius:14px;'>"
            "<h2 style='margin:0;color:#fca5a5;font-size:1rem;'>Power off the device</h2>"
            f"{who}"
            f"<p style='color:{FLEET_TEXT_SOFT};margin:0;font-size:0.95rem;'>"
            "This performs a clean shutdown of the Raspberry Pi. Once it has halted "
            "you can safely disconnect power.</p></div>"
        )

    def load_poweroff_page():
        """Name the device, then arm the button — unless a recording is running."""
        hostname = (client.get_system_info() or {}).get("hostname", "")
        header = _poweroff_header(hostname)
        cap = (client.get_state() or {}).get("capture", {})
        if cap.get("is_capturing") or cap.get("is_starting"):
            return (
                header,
                gr.update(
                    value=_poweroff_notice(
                        "A recording is in progress — stop the capture before powering off."
                    ),
                    visible=True,
                ),
                gr.update(interactive=False, variant="secondary"),
            )
        return (
            header,
            gr.update(value="", visible=False),
            gr.update(interactive=True, variant="stop"),
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
                    "Device is shutting down. This page will stop responding shortly — "
                    "wait ~20 s, then it is safe to unplug.",
                    "#22c55e",
                ),
                visible=True,
            ),
            gr.update(interactive=False, variant="secondary"),
        )

    # ══════════════════════════════════════════════════════════════════
    # Page 1 — Home (landing): the device's essentials, then network + account
    # ══════════════════════════════════════════════════════════════════

    # NB: `theme` and `css` are NOT passed here. Gradio 6 moved both off the
    # Blocks constructor — passing them raises a UserWarning and is otherwise
    # ignored — so they are handed to mount_gradio_app in app/main.py instead.
    with gr.Blocks(title="Grabette") as demo:
        gr.Navbar(main_page_name="Home", elem_id="grabette-nav")
        gr.HTML(_TITLE_HTML)

        # Battery, hostname, IP, network — above everything else, because they
        # are what you open the dashboard to check.
        home_status_bar = gr.HTML("")

        with gr.Row(equal_height=False):

            # ── Network ──────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("## Network")
                with gr.Accordion("Switch network", open=False):
                    gr.HTML(_WIFI_SETTINGS_HTML)
                # The Bluetooth tool is the way in when the device is on no
                # network this browser can reach — which is exactly when this
                # page cannot be loaded, so the link has to be somewhere an
                # operator has already seen it. It is a normal https page on
                # GitHub Pages (Web Bluetooth needs a secure context; the
                # dashboard's own plain-HTTP origin cannot host it).
                gr.HTML(
                    f'<a href="{_BT_TOOL_URL}" target="_blank" rel="noopener" '
                    'style="display:block;margin-top:.8rem;padding:1rem 1.2rem;'
                    'border-radius:14px;text-decoration:none;color:#fff;text-align:left;'
                    'background:rgba(255,255,255,.06);'
                    'border:1px solid rgba(255,255,255,.18);">'
                    '<div style="font-size:1rem;font-weight:700;">'
                    'Bluetooth tool ↗</div>'
                    f'<div style="font-size:.85rem;color:{FLEET_TEXT_SOFT};'
                    'margin-top:.3rem;font-weight:400;line-height:1.45;">'
                    'Move grabette to a brand-new network over Bluetooth — '
                    'no network needed to reach it. Chrome or Edge.</div></a>'
                )

            # ── HuggingFace Account ──────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("## HuggingFace Account")
                gr.HTML(_HF_AUTH_IFRAME)

        # Big, obvious way to reach the fleet. We can't embed grabette-fleet
        # here — it's OAuth-gated and this dashboard is served over plain HTTP,
        # so its login can't render in an iframe — so we link out in a new tab,
        # where the HF session and OAuth work normally.
        gr.HTML(
            f'<a href="{settings.relay_url}" target="_blank" rel="noopener" '
            'style="display:block;margin-top:1.2rem;padding:2.6rem 1.5rem;border-radius:16px;'
            'text-align:center;text-decoration:none;color:#fff;'
            'background:linear-gradient(135deg,#10b981,#3b82f6);'
            'box-shadow:0 6px 22px rgba(0,0,0,.28);">'
            '<div style="font-size:1.7rem;font-weight:800;">Open fleet dashboard ↗</div>'
            '<div style="font-size:.95rem;opacity:.85;margin-top:.5rem;font-weight:500;">'
            'Manage tasks, sessions and datasets on grabette-fleet</div></a>'
        )

        batt_popup_cn = gr.HTML(visible=False)
        batt_beep_cn = gr.Textbox(visible=False)
        # 30 s, not 60: this page is the battery read-out now, so the number on
        # it has to be current, not just fresh enough to warn on.
        batt_timer_cn = gr.Timer(30.0)
        batt_timer_cn.tick(
            fn=get_home_status,
            outputs=[home_status_bar, batt_popup_cn, batt_beep_cn],
        )
        batt_beep_cn.change(fn=None, inputs=batt_beep_cn, outputs=None, js=_BATTERY_BEEP_JS)
        demo.load(
            fn=get_home_status,
            outputs=[home_status_bar, batt_popup_cn, batt_beep_cn],
        )
        demo.load(fn=None, js=_BATTERY_INIT_JS)

    # ══════════════════════════════════════════════════════════════════
    # Page 2 — Episodes
    # ══════════════════════════════════════════════════════════════════

    with demo.route("Episodes") as episodes_demo:
        gr.Navbar(main_page_name="Home", elem_id="grabette-nav")
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
                status = "◌ Initializing OAK camera…"
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
        # returns all three from a single system-info read, so this page needs no
        # separate battery poll.
        episodes_demo.load(fn=get_episode_status_bar, outputs=status_bar_outputs)

    # (Datasets page removed — dataset generation is done on the fleet.)

    # ══════════════════════════════════════════════════════════════════
    # Page 3 — Live View
    # ══════════════════════════════════════════════════════════════════

    with demo.route("Live View") as live_demo:
        gr.Navbar(main_page_name="Home", elem_id="grabette-nav")
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
        oakd_btn = gr.Button("OAK-D: OFF  — click to enable", size="sm")
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
    # Page 4 — Power Off
    #
    # (There is no Settings page: its two panels — HuggingFace account and
    # WiFi — are the Home page now, next to the device info that used to be
    # split across the two.)
    # ══════════════════════════════════════════════════════════════════

    with demo.route("🔴 Power Off") as poweroff_demo:
        gr.Navbar(main_page_name="Home", elem_id="grabette-nav")
        gr.HTML(_TITLE_HTML)

        # Filled in on load with this device's hostname — see _poweroff_header.
        poweroff_header = gr.HTML(_poweroff_header(""))

        poweroff_msg = gr.HTML(value="", visible=False)
        with gr.Row():
            poweroff_btn = gr.Button("Power off now", variant="stop", scale=0)

        poweroff_btn.click(fn=on_poweroff, outputs=[poweroff_msg, poweroff_btn])
        poweroff_demo.load(
            fn=load_poweroff_page,
            outputs=[poweroff_header, poweroff_msg, poweroff_btn],
        )

    return demo
