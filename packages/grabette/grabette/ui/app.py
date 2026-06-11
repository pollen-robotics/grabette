"""Gradio dashboard for Grabette — camera view, capture controls, session/episode management."""

from __future__ import annotations

import io
import logging
import math

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
"""


_IMU_IFRAME_HTML = (
    '<iframe src="/charts/imu" '
    'style="width:100%;height:38vh;border:none;'
    'border-radius:8px;background:transparent;"></iframe>'
)
_ANGLE_IFRAME_HTML = (
    '<iframe src="/charts/angle" '
    'style="width:100%;height:18vh;border:none;'
    'border-radius:8px;background:transparent;"></iframe>'
)
# Replacement HTML used while teleop is active. gr.update(value="") doesn't
# seem to force a DOM swap (Gradio may treat empty as no-op), so we use an
# explicit non-empty placeholder. Same height as the real iframes to avoid
# layout shift; src=about:blank guarantees no /api/state/history polling.
_IMU_IFRAME_PAUSED = (
    '<iframe src="about:blank" '
    'style="width:100%;height:38vh;border:none;'
    'border-radius:8px;background:#1a1a1a;"></iframe>'
)
_ANGLE_IFRAME_PAUSED = (
    '<iframe src="about:blank" '
    'style="width:100%;height:18vh;border:none;'
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


def create_ui(api_url: str | None = None) -> gr.Blocks:
    client = GrabetteClient(base_url=api_url)

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

    def get_sensor_state():
        """Returns (gyro_text, accel_text, angle_text)."""
        state = client.get_state()
        if state is None:
            return "*Disconnected*", "*Disconnected*", "*Disconnected*"

        imu = state.get("imu")
        if imu:
            a = imu["accel"]
            g = imu["gyro"]
            gyro_text = f"`X: {g[0]:+8.4f}  Y: {g[1]:+8.4f}  Z: {g[2]:+8.4f}  rad/s`"
            accel_text = f"`X: {a[0]:+8.3f}  Y: {a[1]:+8.3f}  Z: {a[2]:+8.3f}  m/s²`"
        else:
            gyro_text = "*No IMU data*"
            accel_text = "*No IMU data*"

        angle = state.get("angle")
        if angle:
            p_deg = math.degrees(angle["proximal"])
            d_deg = math.degrees(angle["distal"])
            angle_text = (
                f"`Proximal: {p_deg:+7.2f}°  ({angle['proximal']:+.4f} rad)`\n\n"
                f"`Distal:   {d_deg:+7.2f}°  ({angle['distal']:+.4f} rad)`"
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
            rows, move_dd, *_ = _refresh_episode_table(session_id)
            return gr.update(value="Start Capture", variant="primary"), rows, move_dd
        else:
            client.start_capture(session_id=session_id or None)
            return gr.update(value="Stop Capture", variant="stop"), gr.update(), gr.update()

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
            result = client.start_capture_session()
            if "error" in result:
                return gr.update(), gr.skip(), gr.skip()
            task_name = result.get("task_name", "")
            return (
                gr.update(value="■ Stop Session", variant="stop"),
                gr.update(value=f"### Capture a new episode for *{task_name}*"),
                gr.update(value=_session_banner_html(task_name)),
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
        """Compute the OAK-D toggle button's appearance from current state."""
        s = client.get_oakd_status() or {}
        if not s.get("supported"):
            return gr.update(
                value="OAK-D not available",
                variant="secondary",
                interactive=False,
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
        return gr.update(value=label, variant=variant, interactive=not busy)

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
          - IMU/angle chart iframes → swapped to about:blank placeholders
            so their JS stops polling /api/state/history

        Returns: (teleop_msg, teleop_btn, camera_timer, depth_timer,
        sensor_timer, teleop_timer, imu_iframe, angle_iframe).
        """
        status = client.get_teleop_status() or {}
        active = bool(status.get("active"))
        daemon = client.get_daemon_status() or {}
        if daemon.get("backend") != "RpiBackend":
            return ("Teleop not available (mock backend)",
                    gr.update(value="Enter Teleop Mode", variant="secondary", interactive=False),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update())
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
                    gr.update(value=_IMU_IFRAME_HTML),
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
                    gr.update(value=_IMU_IFRAME_PAUSED),
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
        desc = f"**Description:** {task_description}" if task_description else ""
        cap_title = f"### Capture" if not task_name else f"### Capture a new episode for *{task_name}*"
        count = len(rows)
        count_str = f"{count} episode" + ("s" if count != 1 else "")
        ep_title = (
            f"## Episodes for *{task_name}*\n\n*{count_str} recorded*"
            if task_name else "## Episodes"
        )
        return rows, move_dd, task_header, desc, cap_title, ep_title

    def _session_banner_html(task_name: str) -> str:
        return (
            '<div style="background:#14532d;border:2px solid #16a34a;border-radius:8px;'
            'padding:12px 18px;display:flex;align-items:center;gap:14px;margin-bottom:4px;">'
            '<span style="font-size:1.8rem;line-height:1;">🔒</span>'
            '<div>'
            '<div style="font-weight:700;color:#4ade80;font-size:1.05rem;'
            'letter-spacing:0.04em;text-transform:uppercase;">Session en cours</div>'
            '<div style="color:#bbf7d0;font-size:0.9rem;margin-top:2px;">'
            f'Tous les enregistrements sont assignés à : <strong>{task_name}</strong>'
            '</div>'
            '</div>'
            '</div>'
        )

    def refresh_tasks():
        sessions = _get_sessions()
        choices = _task_choices(sessions)
        value = choices[0][1] if choices else None
        if value:
            client.set_active_session(value)
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
        sessions = _get_sessions()
        choices = _task_choices(sessions)
        new_id = result["id"]
        rows, move_dd, task_header, desc, cap_title, ep_title = _refresh_episode_table(new_id, sessions)
        return gr.update(choices=choices, value=new_id), task_header, cap_title, desc, ep_title, rows, move_dd, gr.update(visible=False)

    # ── Edit Task helpers ─────────────────────────────────────────────

    def on_open_edit_form(session_id):
        if not session_id:
            return gr.update(visible=False), "", ""
        sessions = _get_sessions()
        for s in sessions:
            if s["id"] == session_id:
                return gr.update(visible=True), s.get("name", ""), s.get("description", "")
        return gr.update(visible=True), "", ""

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

    def on_delete_task(session_id):
        if not session_id:
            return (gr.update(),) * 9
        client.delete_session(session_id)
        sessions = _get_sessions()
        choices = _task_choices(sessions)
        value = choices[0][1] if choices else None
        rows, move_dd, task_header, desc, cap_title, ep_title = _refresh_episode_table(value, sessions)
        return (
            gr.update(choices=choices, value=value),
            task_header, cap_title, desc, ep_title, rows, move_dd,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def _get_selected_ids(table_data) -> list[str]:
        if table_data is None:
            return []
        try:
            if table_data.empty:
                return []
            selected = table_data[table_data.iloc[:, 0] == True]
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
            return "No episode selected", gr.update(), gr.update()
        errors = []
        for eid in episode_ids:
            result = client.delete_episode(eid)
            if "error" in result:
                errors.append(f"{eid}: {result['error']}")
        rows, move_dd, *_ = _refresh_episode_table(session_id)
        if errors:
            return "Errors: " + "; ".join(errors), rows, move_dd
        return f"Deleted {len(episode_ids)} episode(s)", rows, move_dd

    def on_move_episodes(table_data, target_session_id, current_session_id):
        episode_ids = _get_selected_ids(table_data)
        if not episode_ids:
            return "No episode selected", gr.update(), gr.update()
        if not target_session_id:
            return "No target task", gr.update(), gr.update()
        result = client.move_episodes(episode_ids, target_session_id)
        if "error" in result:
            return f"Error: {result['error']}", gr.update(), gr.update()
        rows, move_dd, *_ = _refresh_episode_table(current_session_id)
        return f"Moved {len(episode_ids)} episode(s)", rows, move_dd

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

    def check_battery_warning():
        return _battery_popup_html(client.get_system_info())

    # ── System bar ────────────────────────────────────────────────────

    def _battery_popup_html(info: dict | None):
        """Return (visible, html) for the battery popup from a system info dict."""
        if info and "battery_pct" in info and info["battery_pct"] <= 30:
            pct = info["battery_pct"]
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
            return gr.update(visible=True, value=html)
        return gr.update(visible=False)

    def get_system_bar():
        """Returns (system_bar_html, battery_popup_update) from a single API call."""
        info = client.get_system_info()
        if info is None:
            bar = "<p style='color:#64748b;font-size:0.85rem;margin:0.5rem 0;'>System disconnected</p>"
            return bar, gr.update(visible=False)

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
            if pct > 60:
                batt_color = "#22c55e"
                batt_border = "#166534"
            elif pct > 20:
                batt_color = "#f97316"
                batt_border = "#9a3412"
            else:
                batt_color = "#ef4444"
                batt_border = "#991b1b"
            parts.append(
                f"<div style='background:#1e293b;border-radius:8px;padding:0.55rem 1rem;"
                f"border:2px solid {batt_border};flex:1;min-width:0;'>"
                f"<div style='font-size:0.65rem;text-transform:uppercase;letter-spacing:0.09em;"
                f"color:#94a3b8;margin-bottom:0.2rem;'>Battery</div>"
                f"<div style='font-size:0.9rem;font-weight:700;color:{batt_color};'>{pct} %</div>"
                f"</div>"
            )

        bar = (
            "<div style='display:flex;flex-direction:row;gap:0.5rem;flex-wrap:wrap;'>"
            + "".join(parts)
            + "</div>"
        )
        return bar, _battery_popup_html(info)

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

    def _hf_status_text(result):
        if result.get("authenticated"):
            return f"Authenticated as {result.get('user', {}).get('username', '?')}"
        return "Not authenticated"

    def _ds_upload_btn_update(authenticated: bool):
        if authenticated:
            return gr.update(value="Push to HuggingFace Hub", interactive=True, variant="huggingface")
        return gr.update(
            value="You need to be authenticated to push to HuggingFace Hub",
            interactive=False,
            variant="secondary",
        )

    def check_hf_auth_on_load():
        result = client.hf_check_auth()
        authenticated = result.get("authenticated", False)
        return gr.update(visible=not authenticated), _ds_upload_btn_update(authenticated)

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

    def on_ds_upload(task_ids, namespace, repo_name):
        if not task_ids:
            return "Select at least one task"
        if not namespace or not repo_name.strip():
            return "Enter a namespace and a repository name"
        repo_id = f"{namespace}{repo_name.strip()}"
        sessions = _get_sessions()
        session_map = {s["id"]: s for s in sessions}
        jobs, errors = [], []
        for tid in task_ids:
            s = session_map.get(tid)
            if not s:
                continue
            for ep in s.get("episodes", []):
                result = client.hf_upload_episode(ep["episode_id"], repo_id)
                if "error" in result:
                    errors.append(f"{ep['episode_id']}: {result['error']}")
                else:
                    jobs.append(result.get("job_id", "?"))
        if errors:
            return f"Errors: {'; '.join(errors)}"
        if not jobs:
            return "No episodes found in selected tasks"
        return f"Started {len(jobs)} upload job(s)"

    def on_modal_auth(token):
        if not token:
            return gr.update(visible=True, value="Please enter a token"), gr.update(), gr.update()
        result = client.hf_set_auth(token)
        if result.get("authenticated"):
            return gr.update(visible=False), gr.update(visible=False), _ds_upload_btn_update(True)
        return (
            gr.update(visible=True, value=f"Auth failed: {result.get('error', 'unknown')}"),
            gr.update(),
            gr.update(),
        )

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

    def check_hf_account():
        return _hf_status_text(client.hf_check_auth())

    def on_hf_update_token(token):
        if not token:
            return "No token provided", gr.update()
        result = client.hf_set_auth(token)
        return _hf_status_text(result), gr.update(value="")

    def on_hf_remove_token():
        client.hf_set_auth("")
        return "Not authenticated"

    # ══════════════════════════════════════════════════════════════════
    # Page 1 — Episodes
    # ══════════════════════════════════════════════════════════════════

    with gr.Blocks(title="Grabette", css=MODAL_CSS) as demo:
        gr.Navbar(main_page_name="Episodes")
        gr.Markdown("# GRABETTE")

        # ── Main layout ───────────────────────────────────────────────
        with gr.Row():

            # ── LEFT: Tasks ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=200, elem_id="tasks-col"):
                gr.Markdown("## Tasks")
                task_list = gr.Radio(choices=[], label=None, container=False)
                new_task_btn = gr.Button("+ New Task", size="sm", variant="primary")
                with gr.Group(visible=False) as new_task_form:
                    new_task_name = gr.Textbox(label="Name", placeholder="e.g. Kitchen Pick & Place")
                    new_task_desc = gr.Textbox(label="Description", placeholder="Optional")
                    with gr.Row():
                        create_task_btn = gr.Button("Create", variant="primary", size="sm")
                        cancel_task_btn = gr.Button("Cancel", size="sm")

            # ── RIGHT: Episodes ──────────────────────────────────────
            with gr.Column(scale=3):

                # Task header: "## Task: X" + edit button
                with gr.Row():
                    with gr.Column(scale=5):
                        task_header_md = gr.Markdown("")
                    edit_task_btn = gr.Button("✏ Edit", size="sm", scale=1)
                task_desc_md = gr.Markdown("")

                # Edit Task panel (appears below description)
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
                            "⚠ **This will permanently delete the task and ALL its episodes. "
                            "This action cannot be undone.**"
                        )
                        with gr.Row():
                            confirm_delete_btn = gr.Button(
                                "Yes, delete everything", variant="stop", size="sm",
                            )
                            cancel_delete_btn = gr.Button("Cancel", size="sm")

                # Capture
                session_banner = gr.HTML("")
                capture_title = gr.Markdown("### Capture")
                with gr.Row():
                    capture_box = gr.Textbox(
                        label="Status", lines=2, interactive=False, scale=3,
                    )
                    with gr.Column(scale=1, min_width=150):
                        session_btn = gr.Button("▶ Start Session", variant="secondary", size="sm")
                        toggle_btn = gr.Button("Start Capture", variant="primary")

                episodes_title = gr.Markdown("## Episodes")

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
                        'style="width:100%;height:160px;border:none;'
                        'border-radius:8px;background:transparent;"></iframe>'
                    )
                    gr.HTML(
                        '<iframe src="/charts/angle" '
                        'style="width:100%;height:100px;border:none;'
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

        # Edit Task
        edit_task_btn.click(
            fn=on_open_edit_form, inputs=task_list,
            outputs=[edit_task_form, rename_input, desc_edit_input],
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
        confirm_delete_btn.click(
            fn=on_delete_task, inputs=task_list,
            outputs=[task_list, task_header_md, capture_title, task_desc_md, episodes_title,
                     episodes_table, move_target_dd, edit_task_form, delete_confirm],
        )

        session_btn.click(
            fn=on_start_stop_session,
            inputs=[task_list],
            outputs=[session_btn, capture_title, session_banner],
        )

        toggle_btn.click(
            fn=on_toggle_capture,
            inputs=[task_list],
            outputs=[toggle_btn, episodes_table, move_target_dd],
        )

        dl_btn.click(fn=on_download_episodes, inputs=episodes_table, outputs=dl_file)
        del_episode_btn.click(
            fn=on_delete_episode, inputs=[episodes_table, task_list],
            outputs=[episode_msg, episodes_table, move_target_dd],
        )
        move_btn.click(
            fn=on_move_episodes, inputs=[episodes_table, move_target_dd, task_list],
            outputs=[episode_msg, episodes_table, move_target_dd],
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

        def get_capture_status_and_active_task(current_task):
            state = client.get_state()
            cap_session = client.get_capture_session_status()
            cap = (state or {}).get("capture", {})
            is_recording = cap.get("is_capturing", False)

            # Build status text
            if is_recording:
                parts = [
                    f"● RECORDING  {cap.get('session_id', '')}",
                    f"Duration: {cap.get('duration_seconds', 0):.1f}s",
                    f"Frames: {cap.get('frame_count', 0)}  |  IMU: {cap.get('imu_sample_count', 0)}",
                ]
                if cap.get("angle_sample_count", 0):
                    parts[-1] += f"  |  Angle: {cap['angle_sample_count']}"
                if cap_session.get("active"):
                    count = cap_session.get("count", 0)
                    parts.insert(0, f"● SESSION: {count} episode{'s' if count != 1 else ''} recorded")
                status = "\n".join(parts)
            elif cap_session.get("active"):
                count = cap_session.get("count", 0)
                status = f"● SESSION: {count} episode{'s' if count != 1 else ''} recorded\n○ Idle"
            else:
                status = "○ Idle"

            # Session button + capture title + banner sync
            if cap_session.get("active"):
                task_name = cap_session.get("task_name", "")
                sess_btn = gr.update(value="■ Stop Session", variant="stop")
                cap_title = gr.update(value=f"### Capture a new episode for *{task_name}*")
                banner = gr.update(value=_session_banner_html(task_name))
                task_update = gr.skip()
            else:
                active = client.get_active_session()
                sess_btn = gr.update(value="▶ Start Session", variant="secondary")
                cap_title = gr.skip()
                banner = gr.update(value="")
                task_update = gr.skip() if (active is None or active == current_task) else gr.update(value=active)

            return status, task_update, sess_btn, cap_title, banner

        capture_timer = gr.Timer(0.5)
        capture_timer.tick(
            fn=get_capture_status_and_active_task,
            inputs=[task_list],
            outputs=[capture_box, task_list, session_btn, capture_title, session_banner],
        )

        batt_popup_ep = gr.HTML(visible=False)
        batt_timer_ep = gr.Timer(60.0)
        batt_timer_ep.tick(fn=check_battery_warning, outputs=batt_popup_ep)

        demo.load(fn=refresh_tasks, outputs=[task_list, task_header_md, capture_title, task_desc_md, episodes_title, episodes_table, move_target_dd])
        demo.load(fn=check_battery_warning, outputs=batt_popup_ep)

    # ══════════════════════════════════════════════════════════════════
    # Page 2 — Datasets (HF auth popup + upload)
    # ══════════════════════════════════════════════════════════════════

    with demo.route("Datasets") as datasets_demo:
        gr.Navbar(main_page_name="Episodes")

        # HF Auth popup
        with gr.Group(visible=False, elem_id="hf-auth-modal") as ds_auth_modal:
            with gr.Group(elem_id="hf-auth-card"):
                gr.HTML(
                    "<h2 style='margin:0 0 0.4rem;'>HuggingFace Authentication</h2>"
                    "<p style='color:#9ca3af;margin:0 0 1.2rem;font-size:0.9rem;'>"
                    "A HuggingFace token is required to push datasets.</p>"
                )
                ds_modal_token = gr.Textbox(label="HF Token", type="password", placeholder="hf_...")
                ds_modal_msg = gr.Textbox(show_label=False, interactive=False, max_lines=1, visible=False)
                ds_modal_auth_btn = gr.Button("Authenticate", variant="primary", size="sm")

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
              Choose a namespace and give a name to the dataset.
            </div>
          </div>
        </div>
        """)
        with gr.Row():
            ds_namespace = gr.Dropdown(
                label="Namespace", choices=[], interactive=True, scale=1,
            )
            ds_repo_name = gr.Textbox(
                label="Repository name", placeholder="grabette-data",
                scale=2,
            )

        # ── Upload ────────────────────────────────────────────────────
        gr.HTML("<div style='margin-top:1.5rem;max-width:260px;'>")
        ds_upload_btn = gr.Button(
            "You need to be authenticated to push to HuggingFace Hub",
            variant="secondary",
            interactive=False,
        )
        gr.HTML("</div>")
        ds_upload_msg = gr.Textbox(
            show_label=False, interactive=False, max_lines=2, container=False,
        )
        gr.HTML("</div>")

        ds_modal_auth_btn.click(
            fn=on_modal_auth, inputs=ds_modal_token,
            outputs=[ds_modal_msg, ds_auth_modal, ds_upload_btn],
        )
        ds_upload_btn.click(
            fn=on_ds_upload,
            inputs=[ds_task_cbg, ds_namespace, ds_repo_name],
            outputs=ds_upload_msg,
        )
        datasets_demo.load(fn=load_datasets_page, outputs=[ds_task_cbg, ds_namespace])
        datasets_demo.load(fn=check_hf_auth_on_load, outputs=[ds_auth_modal, ds_upload_btn])

        batt_popup_ds = gr.HTML(visible=False)
        batt_timer_ds = gr.Timer(60.0)
        batt_timer_ds.tick(fn=check_battery_warning, outputs=batt_popup_ds)
        datasets_demo.load(fn=check_battery_warning, outputs=batt_popup_ds)

    # ══════════════════════════════════════════════════════════════════
    # Page 3 — Live View
    # ══════════════════════════════════════════════════════════════════

    with demo.route("Live View") as live_demo:
        gr.Navbar(main_page_name="Episodes")
        gr.Markdown("# GRABETTE")

        # ── System bar (full width) ────────────────────────────────────
        dv_system_bar = gr.HTML()

        gr.HTML("<hr style='margin:0.75rem 0;border:none;border-top:1px solid #1e293b;'>")

        # ── Camera | Depth | 3D viewer ────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                gr.HTML("<div style='font-size:0.72rem;text-transform:uppercase;"
                        "letter-spacing:0.09em;color:#94a3b8;margin-bottom:0.3rem;'>"
                        "Camera</div>")
                camera_img = gr.Image(
                    label=None, show_label=False, height="28vh", container=False,
                )
            with gr.Column(scale=1):
                gr.HTML("<div style='font-size:0.72rem;text-transform:uppercase;"
                        "letter-spacing:0.09em;color:#94a3b8;margin-bottom:0.3rem;'>"
                        "Depth (OAK-D)</div>")
                depth_img = gr.Image(
                    label=None, show_label=False, height="28vh", container=False,
                )
                oakd_btn = gr.Button("OAK-D: OFF  — click to enable", size="sm")
            with gr.Column(scale=1):
                gr.HTML("<div style='font-size:0.72rem;text-transform:uppercase;"
                        "letter-spacing:0.09em;color:#94a3b8;margin-bottom:0.3rem;'>"
                        "3D Model</div>")
                gr.HTML(
                    '<iframe id="urdf-viewer" src="/viewer" '
                    'style="width:100%;height:28vh;border:none;'
                    'border-radius:8px;background:#1a1a2e;"></iframe>'
                )

        gr.HTML("<hr style='margin:0.75rem 0;border:none;border-top:1px solid #1e293b;'>")

        # ── IMU (gyro) | Accelerometer | Angle sensors ────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### IMU")
                gyro_box = gr.Markdown("*—*")
                imu_iframe = gr.HTML(value=_IMU_IFRAME_HTML)
            with gr.Column(scale=1):
                gr.Markdown("### Accelerometer")
                accel_box = gr.Markdown("*—*")
            with gr.Column(scale=1):
                gr.Markdown("### Angle Sensors")
                angle_box = gr.Markdown("*—*")
                angle_iframe = gr.HTML(value=_ANGLE_IFRAME_HTML)

        gr.HTML("<hr style='margin:0.75rem 0;border:none;border-top:1px solid #1e293b;'>")

        # ── Teleop ────────────────────────────────────────────────────
        gr.Markdown("### Teleop")
        with gr.Row():
            teleop_btn = gr.Button("Enter Teleop Mode", variant="secondary", scale=1)
            teleop_msg = gr.Textbox(
                show_label=False, interactive=False, max_lines=1, scale=3,
            )

        camera_timer = gr.Timer(0.2)
        camera_timer.tick(fn=get_camera_frame, outputs=camera_img)

        depth_timer = gr.Timer(0.2)
        depth_timer.tick(fn=get_depth_frame, outputs=depth_img)

        sensor_timer = gr.Timer(0.5)
        sensor_timer.tick(fn=get_sensor_state, outputs=[gyro_box, accel_box, angle_box])

        teleop_timer = gr.Timer(1.0)
        teleop_timer.tick(fn=get_teleop_display, outputs=teleop_msg)

        oakd_timer = gr.Timer(3.0)
        oakd_timer.tick(fn=poll_oakd, outputs=oakd_btn)
        oakd_btn.click(fn=on_toggle_oakd, outputs=oakd_btn)
        live_demo.load(fn=poll_oakd, outputs=oakd_btn)

        teleop_btn.click(
            fn=on_toggle_teleop,
            outputs=[teleop_msg, teleop_btn,
                     camera_timer, depth_timer, sensor_timer, teleop_timer,
                     imu_iframe, angle_iframe],
        )

        batt_popup_lv = gr.HTML(visible=False)

        dv_system_timer = gr.Timer(10)
        dv_system_timer.tick(fn=get_system_bar, outputs=[dv_system_bar, batt_popup_lv])
        live_demo.load(fn=get_system_bar, outputs=[dv_system_bar, batt_popup_lv])

    # ══════════════════════════════════════════════════════════════════
    # Page 4 — Settings
    # ══════════════════════════════════════════════════════════════════

    with demo.route("Settings") as settings_demo:
        gr.Navbar(main_page_name="Episodes")
        gr.Markdown("# GRABETTE")

        with gr.Row(equal_height=False):

            # ── HuggingFace Account ───────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("## HuggingFace Account")
                gr.Markdown("### Current status")
                hf_account_status = gr.Textbox(
                    label=None, container=False, interactive=False,
                )
                gr.Markdown("### Update Token")
                new_token_input = gr.Textbox(
                    label=None, container=False,
                    type="password", placeholder="hf_...",
                )
                with gr.Row():
                    update_token_btn = gr.Button("Save token", variant="primary", size="sm")
                    remove_token_btn = gr.Button("Remove current token", variant="stop", size="sm")
                account_msg = gr.Textbox(show_label=False, interactive=False, max_lines=1)

            # ── WiFi ─────────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("## WiFi")
                wifi_network_info = gr.Markdown("*Loading…*")
                gr.HTML(_WIFI_SETTINGS_HTML)

        update_token_btn.click(
            fn=on_hf_update_token, inputs=new_token_input,
            outputs=[hf_account_status, new_token_input],
        )
        remove_token_btn.click(fn=on_hf_remove_token, outputs=hf_account_status)

        settings_demo.load(fn=check_hf_account, outputs=hf_account_status)
        settings_demo.load(fn=get_wifi_network_info, outputs=wifi_network_info)

        batt_popup_st = gr.HTML(visible=False)
        batt_timer_st = gr.Timer(60.0)
        batt_timer_st.tick(fn=check_battery_warning, outputs=batt_popup_st)
        settings_demo.load(fn=check_battery_warning, outputs=batt_popup_st)

    return demo
