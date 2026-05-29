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

PAGE_JS = """
() => {
    const style = document.createElement('style');
    style.textContent = `
        .nav-holder {
            background: #111827 !important;
            border-bottom: 2px solid #f97316 !important;
            padding: 0 1rem !important;
        }
        nav a {
            color: #9ca3af !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 12px 20px !important;
            border-radius: 0 !important;
            border: none !important;
            border-bottom: 3px solid transparent !important;
        }
        nav a.active {
            color: #ffffff !important;
            background-color: transparent !important;
            border-bottom: 3px solid #f97316 !important;
        }
        nav a:hover {
            color: #e5e7eb !important;
            background-color: rgba(255, 255, 255, 0.07) !important;
        }
        #tasks-col {
            background: #1e293b !important;
            border-radius: 8px !important;
            padding: 8px !important;
        }
    `;
    document.head.appendChild(style);
}
"""


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

    # ── Sensor state (Live Streaming page) ────────────────────────────

    def get_sensor_state():
        state = client.get_state()
        if state is None:
            return "## IMU Live\n*Disconnected*", "## Angle Sensors\n*Disconnected*"

        imu = state.get("imu")
        if imu:
            a = imu["accel"]
            g = imu["gyro"]
            imu_text = (
                f"## IMU Live\n"
                f"`Accel: [{a[0]:+8.3f}, {a[1]:+8.3f}, {a[2]:+8.3f}] m/s²`\n\n"
                f"`Gyro:  [{g[0]:+8.4f}, {g[1]:+8.4f}, {g[2]:+8.4f}] rad/s`"
            )
        else:
            imu_text = "## IMU Live\n*No IMU data*"

        angle = state.get("angle")
        if angle:
            p_deg = math.degrees(angle["proximal"])
            d_deg = math.degrees(angle["distal"])
            angle_text = (
                f"## Angle Sensors\n"
                f"`Proximal: {p_deg:+7.2f}°  ({angle['proximal']:+.4f} rad)`\n\n"
                f"`Distal:   {d_deg:+7.2f}°  ({angle['distal']:+.4f} rad)`"
            )
        else:
            angle_text = "## Angle Sensors\n*No angle data*"

        return imu_text, angle_text

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
        ep_title = f"## Episodes for *{task_name}*" if task_name else "## Episodes"
        return rows, move_dd, task_header, desc, cap_title, ep_title

    def refresh_tasks():
        sessions = _get_sessions()
        choices = _task_choices(sessions)
        value = choices[0][1] if choices else None
        rows, move_dd, task_header, desc, cap_title, ep_title = _refresh_episode_table(value, sessions)
        return gr.update(choices=choices, value=value), task_header, cap_title, desc, ep_title, rows, move_dd

    def on_task_select(session_id):
        rows, move_dd, task_header, desc, cap_title, ep_title = _refresh_episode_table(session_id)
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

    # ── System bar ────────────────────────────────────────────────────

    def get_system_bar():
        info = client.get_system_info()
        if info is None:
            return "System: disconnected"
        parts = [info.get("hostname", "?")]
        if "cpu_temp_c" in info:
            parts.append(f"{info['cpu_temp_c']}°C")
        if "disk_free_gb" in info:
            parts.append(f"{info['disk_free_gb']}GB free")
        if "ip" in info:
            parts.append(info["ip"])
        return " | ".join(parts)

    # ── HuggingFace ───────────────────────────────────────────────────

    def _hf_status_text(result):
        if result.get("authenticated"):
            return f"Authenticated as {result.get('user', {}).get('username', '?')}"
        return "Not authenticated"

    def check_hf_auth_on_load():
        result = client.hf_check_auth()
        return gr.update(visible=not result.get("authenticated", False))

    def load_datasets_page():
        sessions = _get_sessions()
        choices = [(s["name"], s["id"]) for s in sessions]
        return gr.update(choices=choices, value=[])

    def on_ds_upload(task_ids, repo_id):
        if not task_ids:
            return "Select at least one task"
        if not repo_id.strip():
            return "Enter a repository name (e.g. username/grabette-data)"
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
            return gr.update(visible=True, value="Please enter a token"), gr.update()
        result = client.hf_set_auth(token)
        if result.get("authenticated"):
            return gr.update(visible=False), gr.update(visible=False)
        return (
            gr.update(visible=True, value=f"Auth failed: {result.get('error', 'unknown')}"),
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
    # Page 1 — Datasets
    # ══════════════════════════════════════════════════════════════════

    with gr.Blocks(title="Grabette", css=MODAL_CSS, js=PAGE_JS) as demo:
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
                capture_title = gr.Markdown("### Capture")
                with gr.Row():
                    capture_box = gr.Textbox(
                        label="Status", lines=2, interactive=False, scale=3,
                    )
                    toggle_btn = gr.Button("Start Capture", variant="primary", scale=1)

                episodes_title = gr.Markdown("## Episodes")

                episodes_table = gr.Dataframe(
                    headers=["✓", "Episode ID", "Duration", "Frames", "IMU", "Angle"],
                    datatype=["bool", "str", "str", "number", "number", "number"],
                    interactive=True,
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

        capture_timer = gr.Timer(0.5)
        capture_timer.tick(fn=get_capture_status, outputs=capture_box)

        demo.load(fn=refresh_tasks, outputs=[task_list, task_header_md, capture_title, task_desc_md, episodes_title, episodes_table, move_target_dd])

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
              Format: <code style="background:#1e293b;padding:2px 6px;border-radius:4px;">
              username/my-dataset</code>
            </div>
          </div>
        </div>
        """)
        ds_repo = gr.Textbox(
            label=None, container=False,
            placeholder="username/grabette-data",
        )

        # ── Upload ────────────────────────────────────────────────────
        gr.HTML("<div style='margin-top:1.5rem;'>")
        ds_upload_btn = gr.Button(
            "Push to HuggingFace Hub",
            variant="huggingface",
        )
        gr.HTML("</div>")
        ds_upload_msg = gr.Textbox(
            show_label=False, interactive=False, max_lines=2, container=False,
        )

        ds_modal_auth_btn.click(
            fn=on_modal_auth, inputs=ds_modal_token,
            outputs=[ds_modal_msg, ds_auth_modal],
        )
        ds_upload_btn.click(
            fn=on_ds_upload, inputs=[ds_task_cbg, ds_repo], outputs=ds_upload_msg,
        )
        datasets_demo.load(fn=load_datasets_page, outputs=ds_task_cbg)
        datasets_demo.load(fn=check_hf_auth_on_load, outputs=ds_auth_modal)

    # ══════════════════════════════════════════════════════════════════
    # Page 3 — Data View
    # ══════════════════════════════════════════════════════════════════

    with demo.route("Data View") as live_demo:
        gr.Navbar(main_page_name="Episodes")
        gr.Markdown("# GRABETTE")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                camera_img = gr.Image(label="Camera Live View", height="30vh")
            with gr.Column(scale=1):
                gr.HTML(
                    '<iframe id="urdf-viewer" src="/viewer" '
                    'style="width:100%;height:30vh;border:none;'
                    'border-radius:8px;background:#1a1a2e;"></iframe>'
                )

        with gr.Row():
            with gr.Column(scale=1):
                imu_box = gr.Markdown("## IMU Live")
                gr.HTML(
                    '<iframe src="/charts/imu" '
                    'style="width:100%;height:40vh;border:none;'
                    'border-radius:8px;background:transparent;"></iframe>'
                )
            with gr.Column(scale=1):
                angle_box = gr.Markdown("## Angle Sensors")
                gr.HTML(
                    '<iframe src="/charts/angle" '
                    'style="width:100%;height:20vh;border:none;'
                    'border-radius:8px;background:transparent;"></iframe>'
                )

        dv_system_bar = gr.Textbox(show_label=False, interactive=False, max_lines=1)

        camera_timer = gr.Timer(0.2)
        camera_timer.tick(fn=get_camera_frame, outputs=camera_img)

        sensor_timer = gr.Timer(0.5)
        sensor_timer.tick(fn=get_sensor_state, outputs=[imu_box, angle_box])

        dv_system_timer = gr.Timer(10)
        dv_system_timer.tick(fn=get_system_bar, outputs=dv_system_bar)

    # ══════════════════════════════════════════════════════════════════
    # Page 4 — HF Account
    # ══════════════════════════════════════════════════════════════════

    with demo.route("HF Account") as hf_demo:
        gr.Navbar(main_page_name="Episodes")
        gr.Markdown("# GRABETTE")
        gr.Markdown("## HuggingFace Account")

        hf_account_status = gr.Textbox(label="Current status", interactive=False)

        gr.HTML("<hr style='margin:24px 0;border:none;border-top:1px solid #333;'>")
        gr.Markdown("### Update Token")
        new_token_input = gr.Textbox(
            label="New HF Token", type="password", placeholder="hf_...",
        )
        with gr.Row():
            update_token_btn = gr.Button("Save token", variant="primary", size="sm")
            remove_token_btn = gr.Button("Remove current token", variant="stop", size="sm")
        account_msg = gr.Textbox(show_label=False, interactive=False, max_lines=1)

        update_token_btn.click(
            fn=on_hf_update_token, inputs=new_token_input,
            outputs=[hf_account_status, new_token_input],
        )
        remove_token_btn.click(fn=on_hf_remove_token, outputs=hf_account_status)

        hf_demo.load(fn=check_hf_account, outputs=hf_account_status)

    return demo
