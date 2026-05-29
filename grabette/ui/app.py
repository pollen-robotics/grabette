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

    def on_toggle_capture():
        state = client.get_state()
        capturing = state.get("capture", {}).get("is_capturing", False) if state else False
        if capturing:
            client.stop_capture()
            return gr.update(value="Start Capture", variant="primary")
        else:
            client.start_capture()
            return gr.update(value="Stop Capture", variant="stop")

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
        title = f"## Episodes for *{task_name}*" if task_name else "## Episodes"
        desc = f"**Description:** {task_description}" if task_description else ""
        cap_title = f"### Capture an episode for *{task_name}*" if task_name else "### Capture"
        return rows, move_dd, title, desc, cap_title

    def refresh_tasks():
        sessions = _get_sessions()
        choices = _task_choices(sessions)
        value = choices[0][1] if choices else None
        rows, move_dd, title, desc, cap_title = _refresh_episode_table(value, sessions)
        return gr.update(choices=choices, value=value), cap_title, title, desc, rows, move_dd

    def on_task_select(session_id):
        rows, move_dd, title, desc, cap_title = _refresh_episode_table(session_id)
        return cap_title, title, desc, rows, move_dd

    def on_create_task(name, description):
        if not name:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(visible=True)
        result = client.create_session(name, description or "")
        if "error" in result:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(visible=True)
        sessions = _get_sessions()
        choices = _task_choices(sessions)
        new_id = result["id"]
        rows, move_dd, title, desc, cap_title = _refresh_episode_table(new_id, sessions)
        return gr.update(choices=choices, value=new_id), cap_title, title, desc, rows, move_dd, gr.update(visible=False)

    # ── Edit Task helpers ─────────────────────────────────────────────

    def on_open_edit_form(session_id):
        if not session_id:
            return gr.update(visible=False), "", ""
        sessions = _get_sessions()
        for s in sessions:
            if s["id"] == session_id:
                return gr.update(visible=True), s.get("name", ""), s.get("description", "")
        return gr.update(visible=True), "", ""

    def on_rename_task(session_id, new_name):
        if not session_id or not new_name.strip():
            return "Enter a name", gr.update(), gr.update(), gr.update()
        client.update_session(session_id, name=new_name.strip())
        sessions = _get_sessions()
        choices = _task_choices(sessions)
        _, _, title, _, cap_title = _refresh_episode_table(session_id, sessions)
        return "Renamed", gr.update(choices=choices, value=session_id), cap_title, title

    def on_update_task_desc(session_id, new_desc):
        if not session_id:
            return "No task selected", gr.update()
        client.update_session(session_id, description=new_desc)
        desc = f"**Description:** {new_desc}" if new_desc.strip() else ""
        return "Updated", desc

    def on_delete_task(session_id):
        if not session_id:
            return (gr.update(),) * 8
        client.delete_session(session_id)
        sessions = _get_sessions()
        choices = _task_choices(sessions)
        value = choices[0][1] if choices else None
        rows, move_dd, title, desc, cap_title = _refresh_episode_table(value, sessions)
        return (
            gr.update(choices=choices, value=value),
            cap_title, title, desc, rows, move_dd,
            gr.update(visible=False),  # close edit form
            gr.update(visible=False),  # close delete confirm
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
        rows, move_dd = _refresh_episode_table(session_id)
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
        rows, move_dd = _refresh_episode_table(current_session_id)
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
        return _hf_status_text(result), gr.update(visible=not result.get("authenticated", False))

    def on_modal_auth(token):
        if not token:
            return gr.update(visible=True, value="Please enter a token"), gr.update(), gr.update()
        result = client.hf_set_auth(token)
        if result.get("authenticated"):
            msg = _hf_status_text(result)
            return gr.update(visible=False), gr.update(visible=False), gr.update(value=msg)
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
    # Page 1 — Datasets
    # ══════════════════════════════════════════════════════════════════

    with gr.Blocks(title="Grabette", css=MODAL_CSS, js=PAGE_JS) as demo:
        gr.Navbar(main_page_name="Datasets")
        gr.Markdown("# GRABETTE")

        # ── HF Auth popup ─────────────────────────────────────────────
        with gr.Group(visible=False, elem_id="hf-auth-modal") as auth_modal:
            with gr.Group(elem_id="hf-auth-card"):
                gr.HTML(
                    "<h2 style='margin:0 0 0.4rem;'>HuggingFace Authentication</h2>"
                    "<p style='color:#9ca3af;margin:0 0 1.2rem;font-size:0.9rem;'>"
                    "Enter your HF token to enable episode uploads.</p>"
                )
                modal_token = gr.Textbox(label="HF Token", type="password", placeholder="hf_...")
                modal_msg = gr.Textbox(show_label=False, interactive=False, max_lines=1, visible=False)
                with gr.Row():
                    modal_auth_btn = gr.Button("Authenticate", variant="primary", size="sm")
                    modal_skip_btn = gr.Button("Skip for now", variant="secondary", size="sm")

        # ── Main layout ───────────────────────────────────────────────
        with gr.Row():

            # ── LEFT: Tasks ──────────────────────────────────────────
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("## Tasks")
                task_list = gr.Radio(choices=[], label=None, container=False)
                new_task_btn = gr.Button("+ New Task", size="sm", variant="primary")
                with gr.Group(visible=False) as new_task_form:
                    new_task_name = gr.Textbox(label="Name", placeholder="e.g. Kitchen Pick & Place")
                    new_task_desc = gr.Textbox(label="Description", placeholder="Optional")
                    with gr.Row():
                        create_task_btn = gr.Button("Create", variant="primary", size="sm")
                        cancel_task_btn = gr.Button("Cancel", size="sm")

                with gr.Group(visible=False) as edit_task_form:
                    with gr.Row():
                        gr.Markdown("#### Edit Task")
                        close_edit_btn = gr.Button("✕ Close", size="sm")
                    edit_task_msg = gr.Textbox(
                        show_label=False, interactive=False, max_lines=1,
                    )
                    with gr.Row():
                        rename_input = gr.Textbox(
                            label="New name", placeholder="Task name…", scale=3,
                        )
                        rename_btn = gr.Button("Rename", size="sm", scale=1)
                    with gr.Row():
                        desc_edit_input = gr.Textbox(
                            label="Description", placeholder="Description…", scale=3,
                        )
                        update_desc_btn = gr.Button("Update", size="sm", scale=1)
                    gr.HTML("<hr style='margin:12px 0;border:none;border-top:1px solid #555;'>")
                    delete_task_btn = gr.Button("Delete Task", variant="stop", size="sm")
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

            # ── RIGHT: Episodes ──────────────────────────────────────
            with gr.Column(scale=3):

                # Capture (above episodes, title shows selected task)
                capture_title = gr.Markdown("### Capture")
                with gr.Row():
                    capture_box = gr.Textbox(
                        label="Status", lines=2, interactive=False, scale=3,
                    )
                    toggle_btn = gr.Button("Start Capture", variant="primary", scale=1)

                gr.HTML("<hr style='margin:16px 0;border:none;border-top:1px solid #333;'>")

                # Episodes header + task info
                episodes_title = gr.Markdown("## Episodes")
                with gr.Row():
                    with gr.Column(scale=5):
                        task_desc_md = gr.Markdown("")
                    edit_task_btn = gr.Button("✏ Edit", size="sm", scale=1)

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

                # HF upload section
                gr.HTML("<hr style='margin:24px 0;border:none;border-top:1px solid #333;'>")
                gr.Markdown("### HuggingFace Upload")
                hf_status = gr.Textbox(label="HF Status", interactive=False, max_lines=1)
                with gr.Row():
                    hf_repo = gr.Textbox(
                        label="Dataset Repo ID", placeholder="username/grabette-data", scale=3,
                    )
                    hf_upload_btn = gr.Button("Upload Episode", size="sm", variant="huggingface", scale=1)
                hf_upload_msg = gr.Textbox(label="Upload Status", interactive=False, max_lines=1)

        system_bar = gr.Textbox(show_label=False, interactive=False, max_lines=1)

        # ── Wire events ───────────────────────────────────────────────

        modal_auth_btn.click(
            fn=on_modal_auth, inputs=modal_token,
            outputs=[modal_msg, auth_modal, hf_status],
        )
        modal_skip_btn.click(fn=lambda: gr.update(visible=False), outputs=auth_modal)

        new_task_btn.click(fn=lambda: gr.update(visible=True), outputs=new_task_form)
        cancel_task_btn.click(fn=lambda: gr.update(visible=False), outputs=new_task_form)
        create_task_btn.click(
            fn=on_create_task,
            inputs=[new_task_name, new_task_desc],
            outputs=[task_list, capture_title, episodes_title, task_desc_md, episodes_table, move_target_dd, new_task_form],
        )
        task_list.change(
            fn=on_task_select, inputs=task_list,
            outputs=[capture_title, episodes_title, task_desc_md, episodes_table, move_target_dd],
        )

        # Edit Task
        edit_task_btn.click(
            fn=on_open_edit_form, inputs=task_list,
            outputs=[edit_task_form, rename_input, desc_edit_input],
        )
        close_edit_btn.click(
            fn=lambda: gr.update(visible=False), outputs=edit_task_form,
        )
        rename_btn.click(
            fn=on_rename_task, inputs=[task_list, rename_input],
            outputs=[edit_task_msg, task_list, capture_title, episodes_title],
        )
        update_desc_btn.click(
            fn=on_update_task_desc, inputs=[task_list, desc_edit_input],
            outputs=[edit_task_msg, task_desc_md],
        )
        delete_task_btn.click(
            fn=lambda: gr.update(visible=True), outputs=delete_confirm,
        )
        cancel_delete_btn.click(
            fn=lambda: gr.update(visible=False), outputs=delete_confirm,
        )
        confirm_delete_btn.click(
            fn=on_delete_task, inputs=task_list,
            outputs=[task_list, capture_title, episodes_title, task_desc_md,
                     episodes_table, move_target_dd, edit_task_form, delete_confirm],
        )

        toggle_btn.click(
            fn=on_toggle_capture,
            outputs=[toggle_btn],
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

        hf_upload_btn.click(fn=on_hf_upload, inputs=[episodes_table, hf_repo], outputs=hf_upload_msg)

        capture_timer = gr.Timer(0.5)
        capture_timer.tick(fn=get_capture_status, outputs=capture_box)

        system_timer = gr.Timer(10)
        system_timer.tick(fn=get_system_bar, outputs=system_bar)

        demo.load(fn=refresh_tasks, outputs=[task_list, capture_title, episodes_title, task_desc_md, episodes_table, move_target_dd])
        demo.load(fn=check_hf_auth_on_load, outputs=[hf_status, auth_modal])

    # ══════════════════════════════════════════════════════════════════
    # Page 2 — Data View
    # ══════════════════════════════════════════════════════════════════

    with demo.route("Data View") as live_demo:
        gr.Navbar(main_page_name="Datasets")
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

        camera_timer = gr.Timer(0.2)
        camera_timer.tick(fn=get_camera_frame, outputs=camera_img)

        sensor_timer = gr.Timer(0.5)
        sensor_timer.tick(fn=get_sensor_state, outputs=[imu_box, angle_box])

    # ══════════════════════════════════════════════════════════════════
    # Page 3 — HF Account
    # ══════════════════════════════════════════════════════════════════

    with demo.route("HF Account") as hf_demo:
        gr.Navbar(main_page_name="Datasets")
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
            fn=on_hf_update_token,
            inputs=new_token_input,
            outputs=[hf_account_status, new_token_input],
        )
        remove_token_btn.click(fn=on_hf_remove_token, outputs=hf_account_status)

        hf_demo.load(fn=check_hf_account, outputs=hf_account_status)

    return demo
