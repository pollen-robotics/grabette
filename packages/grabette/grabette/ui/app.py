"""Gradio dashboard for Grabette — camera view, capture controls, session/episode management."""

from __future__ import annotations

import io
import logging
import math
import time

import gradio as gr
from PIL import Image

from grabette.ui.api_client import GrabetteClient

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
# Replacement HTML used while teleop is active. gr.update(value="") doesn't
# seem to force a DOM swap (Gradio may treat empty as no-op), so we use an
# explicit non-empty placeholder. Same height as the real iframes to avoid
# layout shift; src=about:blank guarantees no /api/state/history polling.
_GYRO_IFRAME_PAUSED = (
    '<iframe src="about:blank" '
    'style="width:100%;height:28vh;border:none;'
    'border-radius:8px;background:#1a1a1a;"></iframe>'
)
_ACCEL_IFRAME_PAUSED = (
    '<iframe src="about:blank" '
    'style="width:100%;height:28vh;border:none;'
    'border-radius:8px;background:#1a1a1a;"></iframe>'
)
_ANGLE_IFRAME_PAUSED = (
    '<iframe src="about:blank" '
    'style="width:100%;height:28vh;border:none;'
    'border-radius:8px;background:#1a1a1a;"></iframe>'
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

    def _badge(label, value, colors):
        value_color, border_color = colors
        return (
            f"<div style='background:#1e293b;border-radius:8px;padding:0.55rem 1rem;"
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
            colors = GREEN if pct > 60 else ORANGE if pct > 20 else RED
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

    # OAK-D (4-state: connected / starting / off / error; N/A when unsupported)
    if not oakd_status or not oakd_status.get("supported"):
        oakd_badge = _badge("OAK-D", "N/A", GRAY)
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


def _text_bar(pct: float, width: int = 22) -> str:
    """Render a fixed-width text progress bar like ██████░░░░ for markdown."""
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 100.0 * width))
    return "`" + "█" * filled + "░" * (width - filled) + f"` {pct:.0f}%"


def _upload_progress_md(repo_id: str, rows: list[dict], finished: bool) -> str:
    """Build the markdown shown while/after pushing episodes to HF Hub.

    rows: list of {episode_id, status, progress, message}. A pure function so
    it's easy to reason about — the polling loop just feeds it fresh job state.
    """
    total = len(rows)
    done = sum(1 for r in rows if r["status"] in ("completed", "failed"))
    ok = sum(1 for r in rows if r["status"] == "completed")
    # Overall %: finished jobs count as 100, in-flight contribute their own %.
    overall = (
        sum(100.0 if r["status"] in ("completed", "failed") else r.get("progress", 0.0)
            for r in rows) / total
        if total else 0.0
    )

    icon = {"completed": "✅", "failed": "❌", "running": "⏳", "pending": "⏳"}
    lines: list[str] = []
    if finished:
        lines.append(f"### {'✅' if ok == total else '⚠️'} Pushed {ok}/{total} episode(s) to HuggingFace")
        if ok:
            lines.append(
                f"\n**[Open dataset → {repo_id}](https://huggingface.co/datasets/{repo_id})**\n"
            )
    else:
        lines.append(f"### Pushing {total} episode(s) to `{repo_id}` …")
        lines.append(f"\n{_text_bar(overall)} — {done}/{total} done\n")

    for r in rows:
        mark = icon.get(r["status"], "⏳")
        eid = r["episode_id"]
        if r["status"] == "failed":
            detail = r.get("message") or r.get("error") or "failed"
        elif r["status"] == "completed":
            detail = "uploaded"
        else:
            detail = f"{r.get('message', '')} ({r.get('progress', 0):.0f}%)"
        lines.append(f"- {mark} `{eid}` — {detail}")
    return "\n".join(lines)


def create_ui(api_url: str | None = None) -> gr.Blocks:
    # Route downloaded episode archives to the SD-card-backed data_dir
    # instead of the OS /tmp (which on Pi OS is a small tmpfs). Same reason
    # as the SessionManager staging; the SessionManager's startup sweep
    # cleans this same directory across daemon restarts.
    from grabette.config import settings
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
                f"● RECORDING  {cap.get('session_id', '')}",
                f"Duration: {cap.get('duration_seconds', 0):.1f}s",
                f"Frames: {cap.get('frame_count', 0)}  |  IMU: {cap.get('imu_sample_count', 0)}",
            ]
            if cap.get("angle_sample_count", 0):
                parts[-1] += f"  |  Angle: {cap['angle_sample_count']}"
            return "\n".join(parts)
        return "○ Idle"

    def on_toggle_capture(session_id):
        state = client.get_state()
        capturing = state.get("capture", {}).get("is_capturing", False) if state else False
        if capturing:
            client.stop_capture()
            rows, move_dd, _task_header, desc, *_ = _refresh_episode_table(session_id)
            return gr.update(value="Start Capture", variant="primary"), rows, move_dd, desc
        else:
            client.start_capture(session_id=session_id or None)
            return gr.update(value="Stop Capture", variant="stop"), gr.update(), gr.update(), gr.update()

    def on_start_stop_session(current_task):
        cap_session = client.get_capture_session_status()
        if cap_session.get("active"):
            client.stop_capture_session()
            _, _, _, _, cap_title, _ = _refresh_episode_table(current_task)
            return (
                gr.update(value="▶ Start Session", variant="secondary"),
                gr.update(value=cap_title),
                gr.update(value=""),
            )
        else:
            result = client.start_capture_session(task_id=current_task or None)
            if "error" in result:
                return gr.update(), gr.skip(), gr.skip()
            task_name = result.get("task_name", "")
            return (
                gr.update(value="■ Stop Session", variant="stop"),
                gr.update(value=f"### Capture a new episode for *{task_name}*"),
                gr.update(value=_session_banner_html(task_name, 0)),
            )

    def get_teleop_display():
        """Polled on a slow (~1 Hz) timer, separately from get_sensor_state.

        Returns the teleop_msg text. When teleop is off, the textbox is
        cleared so it doesn't visually compete with the capture box.
        Doing this on the main state_timer caused HTTP backpressure that
        made the IMU / Angle markdown flicker and bursted the WS stream.
        """
        tstatus = client.get_teleop_status() or {}
        if not tstatus.get("active"):
            return ""
        sending = "YES" if tstatus.get("sending") else "no"
        stats = tstatus.get("stats", {}) or {}
        hz = stats.get("mean_hz", 0)
        n = stats.get("n_poses", 0)
        return f"● TELEOP ON   sending: {sending}   VIO: {hz:.1f} Hz   {n} poses"

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

    def on_toggle_teleop():
        """Single-button toggle: enter teleop mode if off, exit if on.

        Entering teleop pauses ALL UI live-view sources so uvicorn's event
        loop is free for /api/teleop/stream:
          - Gradio Timers (camera, depth, sensor, teleop) → interval set to
            a huge value (Gradio's active=False propagation is unreliable for
            gr.Timer at runtime; bumping the interval is a deterministic kill)
          - gyro/accel/angle chart iframes → swapped to about:blank
            placeholders so their JS stops polling /api/state/history

        Returns: (teleop_msg, teleop_btn, camera_timer, depth_timer,
        sensor_timer, teleop_timer, gyro_iframe, accel_iframe, angle_iframe).
        """
        status = client.get_teleop_status() or {}
        active = bool(status.get("active"))
        daemon = client.get_daemon_status() or {}
        if daemon.get("backend") != "RpiBackend":
            return ("Teleop not available (mock backend)",
                    gr.update(value="Enter Teleop Mode", variant="secondary", interactive=False),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update())
        if active:
            result = client.stop_teleop()
            if "error" in result:
                return (f"Stop error: {result['error']}",
                        gr.update(value="Exit Teleop Mode", variant="stop"),
                        gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update())
            # Exiting teleop — resume live-view timers and restore iframes.
            return ("Teleop OFF",
                    gr.update(value="Enter Teleop Mode", variant="secondary"),
                    gr.update(value=0.2),    # camera_timer
                    gr.update(value=0.2),    # depth_timer
                    gr.update(value=0.5),    # sensor_timer
                    gr.update(value=1.0),    # teleop_timer
                    gr.update(value=_GYRO_IFRAME_HTML),
                    gr.update(value=_ACCEL_IFRAME_HTML),
                    gr.update(value=_ANGLE_IFRAME_HTML))
        else:
            result = client.start_teleop()
            if "error" in result:
                return (f"Start error: {result['error']}",
                        gr.update(value="Enter Teleop Mode", variant="secondary"),
                        gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update())
            # Entering teleop — disable ALL live-view timers via huge intervals.
            return ("Teleop ON (press button to send deltas)",
                    gr.update(value="Exit Teleop Mode", variant="stop"),
                    gr.update(value=86400),  # camera_timer
                    gr.update(value=86400),  # depth_timer
                    gr.update(value=86400),  # sensor_timer
                    gr.update(value=86400),  # teleop_timer
                    gr.update(value=_GYRO_IFRAME_PAUSED),
                    gr.update(value=_ACCEL_IFRAME_PAUSED),
                    gr.update(value=_ANGLE_IFRAME_PAUSED))

    # ── Task (Session) helpers ────────────────────────────────────────

    def _get_sessions():
        return client.list_sessions()

    def _task_choices(sessions):
        return [(s["name"], s["id"]) for s in sessions]

    def _refresh_episode_table(session_id, sessions=None):
        if sessions is None:
            sessions = _get_sessions()
        rows = []
        task_name = ""
        task_description = ""
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
        if task_description:
            desc_parts.append(f"**Task description:** {task_description}")
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
        cap_session = client.get_capture_session_status()
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
        cap_session = client.get_capture_session_status()
        session_active = cap_session.get("active", False)
        if not session_active and session_id:
            client.set_active_session(session_id)
        rows, move_dd, task_header, desc, cap_title, ep_title = _refresh_episode_table(session_id)
        if session_active:
            return task_header, gr.skip(), desc, ep_title, rows, move_dd
        return task_header, cap_title, desc, ep_title, rows, move_dd

    def on_create_task(name, description):
        if not name:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(visible=True)
        result = client.create_session(name, description or "")
        if "error" in result:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(visible=True)
        new_id = result["id"]
        # Point the server-side active session at the new task (unless a capture
        # session owns it). Otherwise the 0.5s capture timer reads the stale
        # active session and snaps the radio back to the previously-selected
        # task, so the episode list would keep showing the old task's episodes.
        if not client.get_capture_session_status().get("active", False):
            client.set_active_session(new_id)
        sessions = _get_sessions()
        choices = _task_choices(sessions)
        rows, move_dd, task_header, desc, cap_title, ep_title = _refresh_episode_table(new_id, sessions)
        return gr.update(choices=choices, value=new_id), task_header, cap_title, desc, ep_title, rows, move_dd, gr.update(visible=False)

    # ── Edit Task helpers ─────────────────────────────────────────────

    def on_open_edit_form(session_id):
        # Always collapse the delete-confirm box: it may have been left open
        # from a previous edit, and re-opening Edit should start clean.
        if not session_id:
            return gr.update(visible=False), "", "", gr.update(visible=False)
        sessions = _get_sessions()
        for s in sessions:
            if s["id"] == session_id:
                return gr.update(visible=True), s.get("name", ""), s.get("description", ""), gr.update(visible=False)
        return gr.update(visible=True), "", "", gr.update(visible=False)

    def on_save_task(session_id, new_name, new_desc):
        if not session_id or not new_name.strip():
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(visible=True)
        client.update_session(session_id, name=new_name.strip(), description=new_desc)
        sessions = _get_sessions()
        choices = _task_choices(sessions)
        _, _, task_header, desc, cap_title, ep_title = _refresh_episode_table(session_id, sessions)
        return (
            gr.update(choices=choices, value=session_id),
            task_header, cap_title, desc, ep_title,
            gr.update(visible=False),
        )

    def _delete_task(session_id, delete_episodes):
        if not session_id:
            return (gr.update(),) * 9
        client.delete_session(session_id, delete_episodes=delete_episodes)
        sessions = _get_sessions()
        choices = _task_choices(sessions)
        value = choices[0][1] if choices else None
        # Sync the server-side active session to the task we land on, so the
        # 0.5s capture timer doesn't try to restore the just-deleted task.
        if value and not client.get_capture_session_status().get("active", False):
            client.set_active_session(value)
        rows, move_dd, task_header, desc, cap_title, ep_title = _refresh_episode_table(value, sessions)
        return (
            gr.update(choices=choices, value=value),
            task_header, cap_title, desc, ep_title, rows, move_dd,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def on_delete_task_keep(session_id):
        # Delete the task but keep its episodes (they fall back to Unassigned).
        return _delete_task(session_id, delete_episodes=False)

    def on_delete_task_purge(session_id):
        # Delete the task AND permanently remove all of its episodes from disk.
        return _delete_task(session_id, delete_episodes=True)

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
        # gr.update(value=...) forces the interactive dataframe to drop its
        # dirty checkbox state so the moved rows actually disappear.
        return f"Moved {len(episode_ids)} episode(s)", gr.update(value=rows), move_dd, desc

    # ── SLAM ──────────────────────────────────────────────────────────

    def on_slam_run(table_data, repo_id: str):
        episode_ids = _get_selected_ids(table_data)
        episode_id = episode_ids[0] if episode_ids else None
        if not episode_id:
            return "Select an episode first"
        if not repo_id:
            return "Enter a HuggingFace repo ID first"
        result = client.slam_run(episode_id, repo_id)
        if "error" in result:
            return f"Error: {result['error']}"
        return f"SLAM started (job: {result.get('job_id', '?')})"

    def get_slam_status():
        jobs = client.hf_list_jobs()
        slam_jobs = [j for j in jobs if j.get("name", "").startswith("slam:")]
        if not slam_jobs:
            return "No SLAM jobs"
        latest = slam_jobs[-1]
        status = latest["status"]
        if status == "completed":
            return f"Complete: {latest.get('result', '')}"
        if status == "failed":
            return f"Failed: {latest.get('error', '')}"
        if status == "running":
            return f"Running ({latest.get('progress', 0):.0f}%): {latest.get('message', '')}"
        return f"Pending: {latest.get('message', '')}"

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
            if charging or pct > 60:
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

    # ── HuggingFace ───────────────────────────────────────────────────

    def _ds_upload_btn_update(authenticated: bool):
        if authenticated:
            return gr.update(value="Push to HuggingFace Hub", interactive=True, variant="huggingface")
        return gr.update(
            value="You need to be authenticated to push to HuggingFace Hub",
            interactive=False,
            variant="secondary",
        )

    def check_hf_auth_on_load(current_ns: str | None = None):
        result = client.hf_check_auth()
        authenticated = result.get("authenticated", False)
        if authenticated:
            namespaces = client.hf_get_namespaces()
            ns_choices = [f"{ns}/" for ns in namespaces]
            # Preserve the user's selection if it's still valid; only reset on
            # first load (current_ns is None) or if the value disappeared.
            if current_ns and current_ns in ns_choices:
                value = current_ns
            else:
                value = ns_choices[0] if ns_choices else None
            ns_update = gr.update(choices=ns_choices, value=value)
        else:
            ns_update = gr.update(choices=[], value=None)
        return gr.update(visible=not authenticated), _ds_upload_btn_update(authenticated), ns_update

    def load_datasets_page():
        sessions = _get_sessions()
        task_choices = [(s["name"], s["id"]) for s in sessions]
        namespaces = client.hf_get_namespaces()
        ns_choices = [f"{ns}/" for ns in namespaces]
        ns_update = gr.update(
            choices=ns_choices,
            value=ns_choices[0] if ns_choices else None,
        )
        return gr.update(choices=task_choices, value=[]), ns_update

    def on_task_selection_count(task_ids):
        if not task_ids:
            return ""
        sessions = _get_sessions()
        session_map = {s["id"]: s for s in sessions}
        total = sum(
            len(s.get("episodes", []))
            for tid in task_ids
            if (s := session_map.get(tid))
        )
        n_tasks = len(task_ids)
        return (
            f'<div style="background:#1e3a5f;border:1px solid #2563eb;border-radius:6px;'
            f'padding:0.4rem 0.75rem;margin-top:0.5rem;display:inline-block;">'
            f'<span style="color:#93c5fd;font-size:0.85rem;">'
            f'{n_tasks} task{"s" if n_tasks != 1 else ""} selected</span>'
            f'<span style="color:#bfdbfe;font-size:0.85rem;margin:0 0.4rem;">·</span>'
            f'<span style="color:#ffffff;font-size:0.9rem;font-weight:700;">{total}</span>'
            f'<span style="color:#93c5fd;font-size:0.85rem;"> episode{"s" if total != 1 else ""}</span>'
            f'</div>'
        )

    def on_ds_upload(task_ids, namespace, repo_name, private):
        """Start one upload job per episode, then poll until all finish,
        streaming progress + per-episode status, and end on a dataset link.

        A generator: each yield updates the status markdown live."""
        if not task_ids:
            yield "Select at least one task"
            return
        if not namespace or not repo_name.strip():
            yield "Enter an owner and a repository name"
            return
        name = repo_name.strip()
        target_repo = f"{namespace}{name}"
        raw_repo = f"{namespace}{name}-raw"

        # Build task description from all selected tasks
        sessions = _get_sessions()
        session_map = {s["id"]: s for s in sessions}
        descriptions = [
            s["description"]
            for tid in task_ids
            if (s := session_map.get(tid)) and s.get("description")
        ]
        task_description = ", ".join(descriptions) if descriptions else name

        yield f"Starting… uploading to {raw_repo}, then processing to {target_repo}"

        result = client.hf_push_and_process(
            task_ids=list(task_ids),
            target_repo=target_repo,
            raw_repo=raw_repo,
            task_description=task_description,
            private=bool(private),
        )
        if "error" in result:
            yield f"Error: {result['error']}"
            return

        job_id = result["job_id"]
        while True:
            time.sleep(3)
            job = client.hf_get_job(job_id)
            if job is None:
                yield "Error: job lost"
                return
            status = job.get("status", "running")
            error = job.get("error") or ""
            msg = (error if status == "failed" else None) or job.get("message") or error
            pct = job.get("progress", 0)
            if status == "completed":
                link = job.get("result") or f"https://huggingface.co/datasets/{target_repo}"
                yield f"✅ Done! Dataset: [{link}]({link})"
                return
            elif status == "failed":
                yield f"❌ Failed: {msg}"
                return
            else:
                yield f"[{pct:.0f}%] {msg}"

    def on_hf_upload(table_data, repo_id):
        episode_ids = _get_selected_ids(table_data)
        episode_id = episode_ids[0] if episode_ids else None
        if not episode_id:
            return "Select an episode first"
        if not repo_id:
            return "Enter a repo ID (e.g. username/grabette-data)"
        result = client.hf_upload_episode(episode_id, repo_id)
        if "error" in result:
            return f"Error: {result['error']}"
        return f"Upload started (job: {result.get('job_id', '?')})"

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
    # Page 1 — Episodes
    # ══════════════════════════════════════════════════════════════════

    with gr.Blocks(title="Grabette", css=MODAL_CSS) as demo:
        gr.Navbar(main_page_name="Episodes", elem_id="grabette-nav")
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
                new_task_btn = gr.Button("+ New Task", size="sm", variant="primary")
                with gr.Group(visible=False) as new_task_form:
                    new_task_name = gr.Textbox(label="Name", placeholder="e.g. Kitchen Pick & Place")
                    new_task_desc = gr.Textbox(label="Description", placeholder="Optional")
                    with gr.Row():
                        create_task_btn = gr.Button("Create", variant="primary", size="sm")
                        cancel_task_btn = gr.Button("Cancel", size="sm")
                edit_task_btn = gr.Button("✏ Edit task", size="sm")
                with gr.Group(visible=False) as edit_task_form:
                    gr.Markdown("#### Edit Task")
                    rename_input = gr.Textbox(label="Name", placeholder="Task name…")
                    desc_edit_input = gr.Textbox(label="Description", placeholder="Description…")
                    with gr.Row():
                        delete_task_btn = gr.Button("Delete Task", variant="stop", size="sm")
                        cancel_edit_btn = gr.Button("Cancel", size="sm")
                        save_task_btn = gr.Button("Save changes", variant="primary", size="sm")
                    with gr.Group(visible=False) as delete_confirm:
                        gr.Markdown(
                            "⚠ **Delete this task — what about its episodes?**\n\n"
                            "- **Keep episodes**: the task is removed and its episodes "
                            "move to *Unassigned*.\n"
                            "- **Delete everything**: the task *and* all its episodes "
                            "are permanently deleted. This cannot be undone."
                        )
                        with gr.Row():
                            delete_keep_btn = gr.Button(
                                "Delete task, keep episodes", variant="secondary", size="sm",
                            )
                            confirm_delete_btn = gr.Button(
                                "Delete everything", variant="stop", size="sm",
                            )
                            cancel_delete_btn = gr.Button("Cancel", size="sm")

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

        gr.HTML("""
            <div style="margin-top:2rem;padding:1.25rem 1.5rem;
                        background:#1e3a5f;border-radius:10px;
                        border:1px solid #2563eb;display:flex;
                        align-items:center;justify-content:space-between;gap:1rem;">
                <span style="color:#e2e8f0;font-size:0.95rem;">
                    Ready to push your episodes to HuggingFace?
                </span>
                <a href="/datasets" style="background:#2563eb;color:#fff;
                           padding:8px 18px;border-radius:6px;font-weight:600;
                           font-size:0.9rem;text-decoration:none;white-space:nowrap;">
                    Create a dataset &#8594;
                </a>
            </div>
        """)

        # ── Wire events ───────────────────────────────────────────────

        new_task_btn.click(fn=lambda: gr.update(visible=True), outputs=new_task_form)
        cancel_task_btn.click(fn=lambda: gr.update(visible=False), outputs=new_task_form)
        create_task_btn.click(
            fn=on_create_task,
            inputs=[new_task_name, new_task_desc],
            outputs=[task_list, task_header_md, capture_title, task_desc_md, episodes_title, episodes_table, move_target_dd, new_task_form],
        )
        task_list.change(
            fn=on_task_select, inputs=task_list,
            outputs=[task_header_md, capture_title, task_desc_md, episodes_title, episodes_table, move_target_dd],
        )
        # Persist the current selection in the browser so a refresh keeps it.
        task_list.change(fn=lambda v: v, inputs=task_list, outputs=selected_task_state)

        # Edit Task
        edit_task_btn.click(
            fn=on_open_edit_form, inputs=task_list,
            outputs=[edit_task_form, rename_input, desc_edit_input, delete_confirm],
        )
        cancel_edit_btn.click(
            fn=lambda: gr.update(visible=False), outputs=edit_task_form,
        )
        save_task_btn.click(
            fn=on_save_task,
            inputs=[task_list, rename_input, desc_edit_input],
            outputs=[task_list, task_header_md, capture_title, task_desc_md, episodes_title, edit_task_form],
        )
        delete_task_btn.click(
            fn=lambda: gr.update(visible=True), outputs=delete_confirm,
        )
        cancel_delete_btn.click(
            fn=lambda: gr.update(visible=False), outputs=delete_confirm,
        )
        _delete_outputs = [task_list, task_header_md, capture_title, task_desc_md, episodes_title,
                           episodes_table, move_target_dd, edit_task_form, delete_confirm]
        delete_keep_btn.click(
            fn=on_delete_task_keep, inputs=task_list, outputs=_delete_outputs,
        )
        confirm_delete_btn.click(
            fn=on_delete_task_purge, inputs=task_list, outputs=_delete_outputs,
        )

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
            cap_session = client.get_capture_session_status()
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
                    f"● RECORDING  {cap.get('session_id', '')}",
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
                active = client.get_active_session()
                sess_btn = gr.update(value="▶ Start Session", variant="secondary")
                cap_title = gr.skip()
                banner = gr.update(value="")
                task_update = gr.skip() if (active is None or active == current_task) else gr.update(value=active)

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

        demo.load(fn=refresh_tasks, inputs=[selected_task_state], outputs=[task_list, task_header_md, capture_title, task_desc_md, episodes_title, episodes_table, move_target_dd])
        demo.load(fn=get_episode_status_bar, outputs=status_bar_outputs)

    # ══════════════════════════════════════════════════════════════════
    # Page 2 — Datasets (HF auth popup + upload)
    # ══════════════════════════════════════════════════════════════════

    with demo.route("Datasets") as datasets_demo:
        gr.Navbar(main_page_name="Episodes", elem_id="grabette-nav")

        # HF Auth popup
        with gr.Group(visible=False, elem_id="hf-auth-modal") as ds_auth_modal:
            with gr.Group(elem_id="hf-auth-card"):
                gr.HTML(_HF_AUTH_IFRAME)

        # ── Page header ───────────────────────────────────────────────
        gr.HTML("""
        <div style="padding:2rem 0 1.5rem;">
          <h1 style="margin:0 0 0.4rem;font-size:1.8rem;">Create a Dataset</h1>
          <p style="margin:0;color:#94a3b8;font-size:0.95rem;">
            Package your recorded tasks and push them to HuggingFace Hub.
          </p>
        </div>
        """)

        # ── Step 1 ────────────────────────────────────────────────────
        gr.HTML("""
        <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;">
          <span style="background:#f97316;color:#fff;font-weight:700;
                       border-radius:50%;width:28px;height:28px;display:flex;
                       align-items:center;justify-content:center;flex-shrink:0;">1</span>
          <div>
            <div style="font-weight:600;font-size:1rem;">Select tasks to include</div>
            <div style="color:#94a3b8;font-size:0.85rem;">
              All episodes within each selected task will be uploaded.
            </div>
          </div>
        </div>
        """)
        ds_task_cbg = gr.CheckboxGroup(choices=[], label=None, container=False)
        ds_episode_count = gr.HTML("")

        # ── Step 2 ────────────────────────────────────────────────────
        gr.HTML("""
        <div style="display:flex;align-items:center;gap:0.75rem;
                    margin-top:1.5rem;margin-bottom:0.5rem;">
          <span style="background:#f97316;color:#fff;font-weight:700;
                       border-radius:50%;width:28px;height:28px;display:flex;
                       align-items:center;justify-content:center;flex-shrink:0;">2</span>
          <div>
            <div style="font-weight:600;font-size:1rem;">Name your destination repository</div>
            <div style="color:#94a3b8;font-size:0.85rem;">
              Choose an owner and give a name to the dataset.
            </div>
          </div>
        </div>
        """)
        with gr.Row():
            ds_namespace = gr.Dropdown(
                label="Owner", choices=[], interactive=True, scale=1,
            )
            ds_repo_name = gr.Textbox(
                label="Repository name", placeholder="grabette-data",
                scale=2,
            )
        ds_private = gr.Checkbox(
            label="Private repository",
            value=False,
        )

        # ── Upload ────────────────────────────────────────────────────
        gr.HTML("<div style='margin-top:1.5rem;max-width:260px;'>")
        ds_upload_btn = gr.Button(
            "You need to be authenticated to push to HuggingFace Hub",
            variant="secondary",
            interactive=False,
        )
        gr.HTML("</div>")
        ds_upload_msg = gr.Markdown(container=False)
        gr.HTML("</div>")

        ds_task_cbg.change(
            fn=on_task_selection_count,
            inputs=ds_task_cbg,
            outputs=ds_episode_count,
        )
        ds_upload_btn.click(
            fn=on_ds_upload,
            inputs=[ds_task_cbg, ds_namespace, ds_repo_name, ds_private],
            outputs=ds_upload_msg,
        )
        datasets_demo.load(fn=load_datasets_page, outputs=[ds_task_cbg, ds_namespace])
        datasets_demo.load(fn=check_hf_auth_on_load, outputs=[ds_auth_modal, ds_upload_btn, ds_namespace])

        # HF auth state changes rarely — poll gently (was 3 s, unnecessarily hot).
        ds_auth_timer = gr.Timer(15.0)
        ds_auth_timer.tick(fn=check_hf_auth_on_load, inputs=[ds_namespace], outputs=[ds_auth_modal, ds_upload_btn, ds_namespace])

        batt_popup_ds = gr.HTML(visible=False)
        batt_beep_ds = gr.Textbox(visible=False)
        # No status strip on this page to piggyback on; poll gently on its own.
        batt_timer_ds = gr.Timer(30.0)
        batt_timer_ds.tick(fn=check_battery_warning, outputs=[batt_popup_ds, batt_beep_ds])
        batt_beep_ds.change(fn=None, inputs=batt_beep_ds, outputs=None, js=_BATTERY_BEEP_JS)
        datasets_demo.load(fn=check_battery_warning, outputs=[batt_popup_ds, batt_beep_ds])
        datasets_demo.load(fn=None, js=_BATTERY_INIT_JS)

    # ══════════════════════════════════════════════════════════════════
    # Page 3 — Live View
    # ══════════════════════════════════════════════════════════════════

    with demo.route("Live View") as live_demo:
        gr.Navbar(main_page_name="Episodes", elem_id="grabette-nav")
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
                angle_iframe = gr.HTML(value=_ANGLE_IFRAME_HTML)
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
                gyro_iframe = gr.HTML(value=_GYRO_IFRAME_HTML)
            with gr.Column(scale=1):
                gr.HTML(_section_label("Accelerometer"))
                accel_box = gr.Markdown("*—*")
                accel_iframe = gr.HTML(value=_ACCEL_IFRAME_HTML)

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
    # Page 4 — Settings
    # ══════════════════════════════════════════════════════════════════

    with demo.route("Settings") as settings_demo:
        gr.Navbar(main_page_name="Episodes", elem_id="grabette-nav")
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
    # Page 5 — Power Off
    # ══════════════════════════════════════════════════════════════════

    with demo.route("🔴 Power Off") as poweroff_demo:
        gr.Navbar(main_page_name="Episodes", elem_id="grabette-nav")
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
