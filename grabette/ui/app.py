"""Gradio dashboard for Grabette — camera view, capture controls, session/episode management."""

from __future__ import annotations

import io
import logging
import math

import gradio as gr
from PIL import Image

from grabette.ui.api_client import GrabetteClient

logger = logging.getLogger(__name__)

def create_ui(api_url: str | None = None) -> gr.Blocks:
    client = GrabetteClient(base_url=api_url)

    # ── Callback helpers ──────────────────────────────────────────────

    def get_camera_frame():
        data = client.get_snapshot()
        if data is None:
            return None
        try:
            return Image.open(io.BytesIO(data))
        except Exception:
            return None

    def get_sensor_state():
        state = client.get_state()
        if state is None:
            return ("Disconnected", "Disconnected", "Disconnected",
                    gr.update(active=True))

        # IMU
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

        # Angle sensors
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
            angle_text = "### Angle Sensors\n*No angle data*"

        # Capture
        cap = state.get("capture", {})
        capturing = cap.get("is_capturing", False)
        if capturing:
            parts = [
                f"\u25cf RECORDING  {cap.get('session_id', '')}",
                f"Duration: {cap.get('duration_seconds', 0):.1f}s",
                f"Frames: {cap.get('frame_count', 0)}  |  "
                f"IMU: {cap.get('imu_sample_count', 0)}",
            ]
            angle_cnt = cap.get("angle_sample_count", 0)
            if angle_cnt:
                parts[-1] += f"  |  Angle: {angle_cnt}"
            cap_text = "\n".join(parts)
        else:
            cap_text = "\u25cb Idle"

        camera_active = not capturing
        return (imu_text, angle_text, cap_text,
                gr.update(active=camera_active))

    def on_toggle_capture():
        state = client.get_state()
        capturing = state.get("capture", {}).get("is_capturing", False) if state else False
        if capturing:
            result = client.stop_capture()
            if "error" in result:
                return f"Error: {result['error']}", gr.update(value="Start Capture", variant="primary")
            dur = result.get("duration_seconds", 0)
            frames = result.get("frame_count", 0)
            return f"Stopped \u2014 {dur:.1f}s, {frames} frames", gr.update(value="Start Capture", variant="primary")
        else:
            result = client.start_capture()
            if "error" in result:
                return f"Error: {result['error']}", gr.update(value="Start Capture", variant="primary")
            return f"Started: {result.get('episode_id', '?')}", gr.update(value="Stop Capture", variant="stop")

    def get_grpc_status():
        st = client.get_grpc_status()
        if st is None or not st.get("enabled"):
            return "\u25cb gRPC disabled"

        def _stream(info, label):
            if info.get("active"):
                return f"{label} \u2713"
            last = info.get("last_seen_s")
            if last is None:
                return f"{label} \u2013"
            return f"{label} ({last:.0f}s)"

        cam = _stream(st.get("camera", {}), "Camera")
        rh = _stream(st.get("r_hand", {}), "R.Hand")
        lh = _stream(st.get("l_hand", {}), "L.Hand")

        if st.get("connected"):
            return f"\u25cf Connected  |  {cam}  |  {rh}  |  {lh}"
        return f"\u25cb No client  |  {cam}  |  {rh}  |  {lh}"

    # ── Session helpers ───────────────────────────────────────────────

    def _get_sessions():
        return client.list_sessions()

    def _session_choices(sessions):
        """Return list of (label, id) tuples for dropdown."""
        return [(s["name"], s["id"]) for s in sessions]

    def _target_session_choices(sessions):
        """Session choices for the move-to dropdown."""
        return [(s["name"], s["id"]) for s in sessions]

    def refresh_sessions():
        """Refresh session dropdown + episode table for current selection."""
        sessions = _get_sessions()
        choices = _session_choices(sessions)
        value = choices[0][1] if choices else None
        rows, move_dd = _refresh_episode_table(value, sessions)
        modifiable = [(name, sid) for name, sid in choices if sid != "unassigned"]
        return (
            gr.update(choices=choices, value=value),
            rows,
            move_dd,
            gr.update(choices=modifiable, value=[]),
        )

    def _get_selected_ids(table_data) -> list[str]:
        """Extract episode IDs from checked rows in the table."""
        if table_data is None:
            return []
        try:
            if table_data.empty:
                return []
            selected = table_data[table_data.iloc[:, 0] == True]
            return selected.iloc[:, 1].tolist()
        except Exception:
            return []

    def _refresh_episode_table(session_id, sessions=None):
        """Return (episode_rows, move_target_dropdown)."""
        if sessions is None:
            sessions = _get_sessions()

        rows = []
        for s in sessions:
            if s["id"] == session_id:
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
        move_choices = _target_session_choices(sessions)
        move_dd = gr.update(choices=move_choices, value=move_choices[0][1] if move_choices else None)
        return rows, move_dd

    def on_session_change(session_id):
        rows, move_dd = _refresh_episode_table(session_id)
        return rows, move_dd

    def on_create_session(name, description):
        if not name:
            return gr.update(), gr.update(), gr.update(), gr.update()
        result = client.create_session(name, description or "")
        if "error" in result:
            return gr.update(), gr.update(), gr.update(), gr.update()
        sessions = _get_sessions()
        choices = _session_choices(sessions)
        new_id = result["id"]
        rows, move_dd = _refresh_episode_table(new_id, sessions)
        modifiable = [(n, sid) for n, sid in choices if sid != "unassigned"]
        return (
            gr.update(choices=choices, value=new_id),
            rows,
            move_dd,
            gr.update(choices=modifiable, value=[]),
        )

    def on_rename_session(session_id, new_name):
        if not session_id or not new_name:
            return
        client.update_session(session_id, name=new_name)

    def on_delete_sessions(session_ids):
        if not session_ids:
            return gr.update(), gr.update(), gr.update(), gr.update()
        for sid in session_ids:
            client.delete_session(sid)
        sessions = _get_sessions()
        choices = _session_choices(sessions)
        value = choices[0][1] if choices else None
        rows, move_dd = _refresh_episode_table(value, sessions)
        modifiable = [(n, sid) for n, sid in choices if sid != "unassigned"]
        return (
            gr.update(choices=choices, value=value),
            rows,
            move_dd,
            gr.update(choices=modifiable, value=[]),
        )

    # ── Episode helpers ───────────────────────────────────────────────

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
        for episode_id in episode_ids:
            result = client.delete_episode(episode_id)
            if "error" in result:
                errors.append(f"{episode_id}: {result['error']}")
        rows, move_dd = _refresh_episode_table(session_id)
        if errors:
            return "Errors: " + "; ".join(errors), rows, move_dd
        return f"Deleted {len(episode_ids)} episode(s)", rows, move_dd

    def on_move_episodes(table_data, target_session_id, current_session_id):
        episode_ids = _get_selected_ids(table_data)
        if not episode_ids:
            return "No episode selected", gr.update(), gr.update()
        if not target_session_id:
            return "No target session", gr.update(), gr.update()
        result = client.move_episodes(episode_ids, target_session_id)
        if "error" in result:
            return f"Error: {result['error']}", gr.update(), gr.update()
        rows, move_dd = _refresh_episode_table(current_session_id)
        return f"Moved {len(episode_ids)} episode(s)", rows, move_dd

    # ── System ────────────────────────────────────────────────────────

    def get_system_bar():
        info = client.get_system_info()
        if info is None:
            return "System: disconnected"
        parts = [info.get("hostname", "?")]
        if "cpu_temp_c" in info:
            parts.append(f"{info['cpu_temp_c']}\u00b0C")
        if "disk_free_gb" in info:
            parts.append(f"{info['disk_free_gb']}GB free")
        if "ip" in info:
            parts.append(info["ip"])
        return " | ".join(parts)

    # ── HuggingFace ───────────────────────────────────────────────────

    def on_hf_auth(token: str):
        if not token:
            return "No token provided"
        result = client.hf_set_auth(token)
        if result.get("authenticated"):
            user = result.get("user", {})
            return f"Authenticated as {user.get('username', '?')}"
        return f"Auth failed: {result.get('error', 'unknown')}"

    def check_hf_auth():
        result = client.hf_check_auth()
        if result.get("authenticated"):
            user = result.get("user", {})
            return f"Authenticated as {user.get('username', '?')}"
        return "Not authenticated"

    def on_hf_upload(table_data, repo_id: str):
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
            'style="width:100%;height:350px;border:none;'
            'border-radius:8px;background:#000;"></iframe>'
        )

    def on_replay_start(table_data):
        episode_id = (_get_selected_ids(table_data) or [None])[0]
        if not episode_id:
            return ("No episode selected", gr.update(visible=False),
                    gr.update(), gr.update(),
                    gr.update(), gr.update())
        result = client.replay_start(episode_id)
        if "error" in result:
            return (f"Error: {result['error']}", gr.update(visible=False),
                    gr.update(), gr.update(),
                    gr.update(), gr.update())
        dur = result.get("duration_ms", 0)
        return (
            f"Replaying {episode_id}",
            gr.update(visible=True),
            gr.update(maximum=dur, value=0),
            gr.update(active=True),
            gr.update(visible=False),
            gr.update(visible=True, value=_video_iframe(episode_id)),
        )

    def on_replay_stop():
        client.replay_stop()
        return (
            "Replay stopped",
            gr.update(visible=False),
            gr.update(active=False),
            gr.update(visible=True),
            gr.update(visible=False, value=""),
        )

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
                gr.update(active=False), gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=False, value=""),
            )
        t = st.get("time_ms", 0)
        dur = st.get("duration_ms", 0)
        playing = st.get("playing", False)
        label = f"{t / 1000:.1f}s / {dur / 1000:.1f}s" + (" (paused)" if not playing else "")
        btn_label = "Pause" if playing else "Play"
        return (
            gr.update(value=t),
            label,
            btn_label,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    # ── Build layout ──────────────────────────────────────────────────

    with gr.Blocks(title="Grabette") as demo:
        gr.Markdown("# GRABETTE")

        # ── Live view ─────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                camera_img = gr.Image(
                    label="Camera Live View",
                    height="25vh",
                )
                replay_video = gr.HTML(visible=False)
            with gr.Column(scale=1):
                viewer_iframe = gr.HTML(
                    value=(
                        '<iframe id="urdf-viewer" src="/viewer" '
                        'style="width:100%;height:30vh;border:none;'
                        'border-radius:8px;background:#1a1a2e;"></iframe>'
                    ),
                    label="3D Model",
                )
            with gr.Column(scale=0.5):
                capture_box = gr.Textbox(
                    label="Capture Status",
                    lines=3,
                    interactive=False,
                )
                toggle_btn = gr.Button("Start Capture", variant="primary")
                capture_msg = gr.Textbox(
                    show_label=False, interactive=False, max_lines=1,
                )
                grpc_status_box = gr.Textbox(
                    label="gRPC Client",
                    interactive=False,
                    max_lines=1,
                )

        with gr.Row():
            with gr.Column(scale=1):
                imu_box = gr.Markdown("## IMU Live")
                gr.HTML(
                    value=(
                        '<iframe src="/charts/imu" '
                        'style="width:100%;height:42vh;border:none;'
                        'border-radius:8px;background:transparent;"></iframe>'
                    ),
                )
            with gr.Column(scale=1):
                angle_box = gr.Markdown("## Angle Sensors")
                gr.HTML(
                    value=(
                        '<iframe src="/charts/angle" '
                        'style="width:100%;height:20vh;border:none;'
                        'border-radius:8px;background:transparent;"></iframe>'
                    ),
                )

        # ── Sessions + Episodes ───────────────────────────────────────
        gr.HTML("<hr style='margin:24px 0;border:none;border-top:2px solid #333;'>")
        gr.Markdown("# Sessions")
        with gr.Row():
            session_dd = gr.Dropdown(
                label="Select a session", interactive=True, scale=4,
            )
            refresh_btn = gr.Button("↺ Refresh", size="sm", scale=1)
        with gr.Row():
            with gr.Accordion("Rename Session", open=False):
                rename_input = gr.Textbox(
                    label="New name", placeholder="New name", scale=3,
                )
                rename_btn = gr.Button("Rename", variant= "huggingface", size="sm")

            with gr.Accordion("New Session", open=False):
                new_session_name = gr.Textbox(
                    label="Name", placeholder="e.g. Kitchen Pick & Place",
                )
                new_session_desc = gr.Textbox(
                    label="Description", placeholder="Optional",
                )
                create_session_btn = gr.Button("Create Session", variant="primary", size="sm")

            with gr.Accordion("Delete Sessions", open=False):
                # choice in the sessions list, with checkboxes to select multiple for deletion
                sessions_cbg = gr.CheckboxGroup(
                    label="Select sessions to delete", choices=[],
                )
                delete_session_btn = gr.Button("Delete selected", variant="stop", size="sm")

        gr.HTML("<hr style='margin:24px 0;border:none;border-top:1px solid #222;'>")
        gr.Markdown("## Episodes")
        episodes_table = gr.Dataframe(
            headers=["✓", "Episode ID", "Duration", "Frames", "IMU", "Angle"],
            datatype=["bool", "str", "str", "number", "number", "number"],
            interactive=True,
            col_count=(6, "fixed"),
            show_search='filter'
        )
        with gr.Row():
            replay_btn = gr.Button("▶ Replay", size="md", scale=1)
            with gr.Accordion("Download", open=False):
                dl_btn = gr.Button("Download selected", size="sm", scale=1)
                dl_file = gr.File(label="Download")
            with gr.Accordion("Move to Session", open=False):
                move_target_dd = gr.Dropdown(
                    label="Move to session", interactive=True, scale=3,
                )
                move_btn = gr.Button("Move", size="sm", scale=1)
            with gr.Accordion("Delete", open=False):
                del_episode_btn = gr.Button("Delete selected", variant="stop", size="sm", scale=1)

        episode_msg = gr.Textbox(show_label=False, interactive=False, max_lines=1)

        # ── Replay panel (hidden until replay starts) ────────────────
        with gr.Group(visible=False) as replay_panel:
            gr.Markdown("#### Episode Replay")
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

        # ── HuggingFace ───────────────────────────────────────────────
        gr.HTML("<hr style='margin:24px 0;border:none;border-top:2px solid #333;'>")
        gr.Markdown("# HuggingFace")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                hf_token = gr.Textbox(
                    label="HF Token", type="password",
                    placeholder="hf_...", scale=2,
                )
                hf_auth_btn = gr.Button("Authenticate", size="sm", scale=1, variant="secondary")

            with gr.Column(scale=1):
                hf_repo = gr.Textbox(
                    label="Dataset Repo ID",
                    placeholder="username/grabette-data",
                    scale=2,
                )
                hf_upload_btn = gr.Button("Upload Episode", size="sm", scale=1, variant = "huggingface")
            

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                hf_status = gr.Textbox(label="HF Status", interactive=False, max_lines=1)
            with gr.Column(scale=1):
                hf_upload_msg = gr.Textbox(
                    label="Upload Status", interactive=False, max_lines=1,
                )

        # ── SLAM ──────────────────────────────────────────────────────
        gr.Markdown("### SLAM Processing")
        with gr.Row():
            slam_btn = gr.Button("Upload & Run SLAM", variant="primary", size="sm")
            slam_status = gr.Textbox(
                label="SLAM Status", interactive=False, scale=2,
            )

        # ── System bar ────────────────────────────────────────────────
        system_bar = gr.Textbox(
            show_label=False, interactive=False, max_lines=1,
        )

        # ── Wire events ───────────────────────────────────────────────

        # Capture
        toggle_btn.click(
            fn=on_toggle_capture,
            outputs=[capture_msg, toggle_btn],
        ).then(
            fn=refresh_sessions,
            outputs=[session_dd, episodes_table, move_target_dd, sessions_cbg],
        )

        # Session selection
        session_dd.change(
            fn=on_session_change, inputs=session_dd,
            outputs=[episodes_table, move_target_dd],
        )
        refresh_btn.click(
            fn=refresh_sessions,
            outputs=[session_dd, episodes_table, move_target_dd, sessions_cbg],
        )

        # Session CRUD
        create_session_btn.click(
            fn=on_create_session,
            inputs=[new_session_name, new_session_desc],
            outputs=[session_dd, episodes_table, move_target_dd, sessions_cbg],
        )
        rename_btn.click(
            fn=on_rename_session,
            inputs=[session_dd, rename_input],
        ).then(
            fn=refresh_sessions,
            outputs=[session_dd, episodes_table, move_target_dd, sessions_cbg],
        )
        delete_session_btn.click(
            fn=on_delete_sessions, inputs=sessions_cbg,
            outputs=[session_dd, episodes_table, move_target_dd, sessions_cbg],
        )

        # Episode actions
        dl_btn.click(fn=on_download_episodes, inputs=episodes_table, outputs=dl_file)
        del_episode_btn.click(
            fn=on_delete_episode, inputs=[episodes_table, session_dd],
            outputs=[episode_msg, episodes_table, move_target_dd],
        )
        move_btn.click(
            fn=on_move_episodes, inputs=[episodes_table, move_target_dd, session_dd],
            outputs=[episode_msg, episodes_table, move_target_dd],
        )

        # Replay
        replay_btn.click(
            fn=on_replay_start, inputs=episodes_table,
            outputs=[episode_msg, replay_panel, replay_slider, replay_timer,
                     camera_img, replay_video],
        )
        replay_stop_btn.click(
            fn=on_replay_stop,
            outputs=[episode_msg, replay_panel, replay_timer,
                     camera_img, replay_video],
        )
        replay_pause_btn.click(
            fn=on_replay_pause_play,
            outputs=replay_pause_btn,
        )
        replay_slider.release(fn=on_replay_seek, inputs=[replay_slider])
        replay_timer.tick(
            fn=poll_replay_status,
            outputs=[replay_slider, replay_time_label, replay_pause_btn,
                     replay_timer, replay_panel, camera_img, replay_video],
        )

        # HuggingFace
        hf_auth_btn.click(fn=on_hf_auth, inputs=hf_token, outputs=hf_status)
        hf_upload_btn.click(
            fn=on_hf_upload, inputs=[episodes_table, hf_repo],
            outputs=hf_upload_msg,
        )

        # SLAM
        slam_btn.click(
            fn=on_slam_run, inputs=[episodes_table, hf_repo],
            outputs=slam_status,
        )

        # ── Periodic updates (Gradio 6 Timer) ─────────────────────────
        camera_timer = gr.Timer(0.2)
        camera_timer.tick(fn=get_camera_frame, outputs=camera_img)

        state_timer = gr.Timer(0.5)
        state_timer.tick(
            fn=get_sensor_state,
            outputs=[imu_box, angle_box, capture_box, camera_timer],
        )

        grpc_timer = gr.Timer(1.0)
        grpc_timer.tick(fn=get_grpc_status, outputs=grpc_status_box)

        system_timer = gr.Timer(10)
        system_timer.tick(fn=get_system_bar, outputs=system_bar)

        # One-shot loads on page open
        demo.load(
            fn=refresh_sessions,
            outputs=[session_dd, episodes_table, move_target_dd, sessions_cbg],
        )
        demo.load(fn=check_hf_auth, outputs=hf_status)

    return demo
