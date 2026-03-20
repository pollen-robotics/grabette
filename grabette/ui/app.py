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
                f"Accel: [{a[0]:+8.3f}, {a[1]:+8.3f}, {a[2]:+8.3f}] m/s\u00b2\n"
                f"Gyro:  [{g[0]:+8.4f}, {g[1]:+8.4f}, {g[2]:+8.4f}] rad/s"
            )
        else:
            imu_text = "No IMU data"

        # Angle sensors
        angle = state.get("angle")
        if angle:
            p_deg = math.degrees(angle["proximal"])
            d_deg = math.degrees(angle["distal"])
            angle_text = (
                f"Proximal: {p_deg:+7.2f}\u00b0  ({angle['proximal']:+.4f} rad)\n"
                f"Distal:   {d_deg:+7.2f}\u00b0  ({angle['distal']:+.4f} rad)"
            )
        else:
            angle_text = "No angle data"

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

    def on_start_capture():
        result = client.start_capture()
        if "error" in result:
            return f"Error: {result['error']}"
        return f"Started: {result.get('episode_id', '?')}"

    def on_stop_capture():
        result = client.stop_capture()
        if "error" in result:
            return f"Error: {result['error']}"
        dur = result.get("duration_seconds", 0)
        frames = result.get("frame_count", 0)
        return f"Stopped \u2014 {dur:.1f}s, {frames} frames"

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
        return (
            gr.update(choices=choices, value=value),
            *_refresh_episode_table(value, sessions),
        )

    def _refresh_episode_table(session_id, sessions=None):
        """Return (episode_rows, episode_dropdown, move_target_dropdown, msg)."""
        if sessions is None:
            sessions = _get_sessions()

        rows = []
        ep_ids = []
        for s in sessions:
            if s["id"] == session_id:
                for ep in s.get("episodes", []):
                    rows.append([
                        ep["episode_id"],
                        f"{ep['duration_seconds']:.1f}s",
                        ep["frame_count"],
                        ep["imu_sample_count"],
                        ep.get("angle_sample_count", 0),
                    ])
                    ep_ids.append(ep["episode_id"])
                break

        ep_dd = gr.update(choices=ep_ids, value=ep_ids[0] if ep_ids else None)
        move_choices = _target_session_choices(sessions)
        move_dd = gr.update(choices=move_choices, value=move_choices[0][1] if move_choices else None)
        return rows, ep_dd, move_dd, ""

    def on_session_change(session_id):
        rows, ep_dd, move_dd, msg = _refresh_episode_table(session_id)
        return rows, ep_dd, move_dd

    def on_create_session(name, description):
        if not name:
            return "Enter a session name", gr.update(), gr.update(), gr.update(), gr.update()
        result = client.create_session(name, description or "")
        if "error" in result:
            return f"Error: {result['error']}", gr.update(), gr.update(), gr.update(), gr.update()
        sessions = _get_sessions()
        choices = _session_choices(sessions)
        new_id = result["id"]
        rows, ep_dd, move_dd, _ = _refresh_episode_table(new_id, sessions)
        return (
            f"Created: {result['name']}",
            gr.update(choices=choices, value=new_id),
            rows,
            ep_dd,
            move_dd,
        )

    def on_rename_session(session_id, new_name):
        if not session_id:
            return "No session selected"
        if not new_name:
            return "Enter a new name"
        result = client.update_session(session_id, name=new_name)
        if "error" in result:
            return f"Error: {result['error']}"
        return f"Renamed to: {result.get('name', new_name)}"

    def on_delete_session(session_id):
        if not session_id:
            return "No session selected", gr.update(), gr.update(), gr.update(), gr.update()
        result = client.delete_session(session_id)
        if "error" in result:
            return f"Error: {result['error']}", gr.update(), gr.update(), gr.update(), gr.update()
        sessions = _get_sessions()
        choices = _session_choices(sessions)
        value = choices[0][1] if choices else None
        rows, ep_dd, move_dd, _ = _refresh_episode_table(value, sessions)
        return (
            f"Deleted session {session_id}",
            gr.update(choices=choices, value=value),
            rows,
            ep_dd,
            move_dd,
        )

    # ── Episode helpers ───────────────────────────────────────────────

    def on_download_episode(episode_id):
        if not episode_id:
            return None
        return client.download_episode(episode_id)

    def on_delete_episode(episode_id, session_id):
        if not episode_id:
            return "No episode selected", gr.update(), gr.update(), gr.update()
        result = client.delete_episode(episode_id)
        if "error" in result:
            return f"Error: {result['error']}", gr.update(), gr.update(), gr.update()
        rows, ep_dd, move_dd, _ = _refresh_episode_table(session_id)
        return f"Deleted {episode_id}", rows, ep_dd, move_dd

    def on_move_episodes(episode_id, target_session_id, current_session_id):
        if not episode_id:
            return "No episode selected", gr.update(), gr.update(), gr.update()
        if not target_session_id:
            return "No target session", gr.update(), gr.update(), gr.update()
        result = client.move_episodes([episode_id], target_session_id)
        if "error" in result:
            return f"Error: {result['error']}", gr.update(), gr.update(), gr.update()
        rows, ep_dd, move_dd, _ = _refresh_episode_table(current_session_id)
        return f"Moved {episode_id}", rows, ep_dd, move_dd

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

    def on_hf_upload(episode_id: str | None, repo_id: str):
        if not episode_id:
            return "Select an episode first"
        if not repo_id:
            return "Enter a repo ID (e.g. username/grabette-data)"
        result = client.hf_upload_episode(episode_id, repo_id)
        if "error" in result:
            return f"Error: {result['error']}"
        return f"Upload started (job: {result.get('job_id', '?')})"

    # ── SLAM ──────────────────────────────────────────────────────────

    def on_slam_run(episode_id: str | None, repo_id: str):
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

    def on_replay_start(episode_id: str | None):
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
                    height=350,
                )
                replay_video = gr.HTML(visible=False)
            with gr.Column(scale=1):
                viewer_iframe = gr.HTML(
                    value=(
                        '<iframe id="urdf-viewer" src="/viewer" '
                        'style="width:100%;height:350px;border:none;'
                        'border-radius:8px;background:#1a1a2e;"></iframe>'
                    ),
                    label="3D Model",
                )
            with gr.Column(scale=1):
                imu_box = gr.Textbox(
                    label="IMU Live",
                    lines=2,
                    interactive=False,
                )
                gr.HTML(
                    value=(
                        '<iframe src="/charts/imu" '
                        'style="width:100%;height:420px;border:none;'
                        'border-radius:8px;background:transparent;"></iframe>'
                    ),
                )
                angle_box = gr.Textbox(
                    label="Angle Sensors",
                    lines=2,
                    interactive=False,
                )
                gr.HTML(
                    value=(
                        '<iframe src="/charts/angle" '
                        'style="width:100%;height:220px;border:none;'
                        'border-radius:8px;background:transparent;"></iframe>'
                    ),
                )
                capture_box = gr.Textbox(
                    label="Capture Status",
                    lines=4,
                    interactive=False,
                )
                with gr.Row():
                    start_btn = gr.Button("Start Capture", variant="primary")
                    stop_btn = gr.Button("Stop Capture", variant="stop")
                capture_msg = gr.Textbox(
                    show_label=False, interactive=False, max_lines=1,
                )
                grpc_status_box = gr.Textbox(
                    label="gRPC Client",
                    interactive=False,
                    max_lines=1,
                )

        # ── Sessions + Episodes ───────────────────────────────────────
        gr.Markdown("### Sessions")
        with gr.Row():
            session_dd = gr.Dropdown(
                label="Session", interactive=True,
            )
            refresh_btn = gr.Button("Refresh", size="sm")
        with gr.Row():
            new_session_name = gr.Textbox(
                label="New session name", scale=2,
                placeholder="e.g. Kitchen Pick & Place",
            )
            new_session_desc = gr.Textbox(
                label="Description", scale=2,
                placeholder="Optional description",
            )
            create_session_btn = gr.Button("Create Session", size="sm", scale=1)
        with gr.Row():
            rename_input = gr.Textbox(
                label="Rename to", scale=2,
                placeholder="New name",
            )
            rename_btn = gr.Button("Rename", size="sm", scale=1)
            delete_session_btn = gr.Button("Delete Session", variant="stop", size="sm", scale=1)
        session_msg = gr.Textbox(show_label=False, interactive=False, max_lines=1)

        gr.Markdown("### Episodes")
        episodes_table = gr.Dataframe(
            headers=["Episode ID", "Duration", "Frames", "IMU", "Angle"],
            interactive=False,
        )
        with gr.Row():
            episode_dd = gr.Dropdown(
                label="Selected Episode", interactive=True,
            )
        with gr.Row():
            dl_btn = gr.Button("Download .tar.gz", size="sm")
            del_episode_btn = gr.Button("Delete Episode", variant="stop", size="sm")
            replay_btn = gr.Button("Replay", size="sm")
        with gr.Row():
            move_target_dd = gr.Dropdown(
                label="Move to session", interactive=True,
            )
            move_btn = gr.Button("Move", size="sm")
        dl_file = gr.File(label="Download")
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
        gr.Markdown("### HuggingFace")
        with gr.Row():
            hf_token = gr.Textbox(
                label="HF Token", type="password",
                placeholder="hf_...", scale=2,
            )
            hf_auth_btn = gr.Button("Authenticate", size="sm", scale=1)
        hf_status = gr.Textbox(label="HF Status", interactive=False, max_lines=1)
        with gr.Row():
            hf_repo = gr.Textbox(
                label="Dataset Repo ID",
                placeholder="username/grabette-data",
                scale=2,
            )
            hf_upload_btn = gr.Button("Upload Episode", size="sm", scale=1)
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
        start_btn.click(fn=on_start_capture, outputs=capture_msg)
        stop_btn.click(fn=on_stop_capture, outputs=capture_msg).then(
            fn=refresh_sessions,
            outputs=[session_dd, episodes_table, episode_dd, move_target_dd],
        )

        # Session selection
        session_dd.change(
            fn=on_session_change, inputs=session_dd,
            outputs=[episodes_table, episode_dd, move_target_dd],
        )
        refresh_btn.click(
            fn=refresh_sessions,
            outputs=[session_dd, episodes_table, episode_dd, move_target_dd],
        )

        # Session CRUD
        create_session_btn.click(
            fn=on_create_session,
            inputs=[new_session_name, new_session_desc],
            outputs=[session_msg, session_dd, episodes_table, episode_dd, move_target_dd],
        )
        rename_btn.click(
            fn=on_rename_session,
            inputs=[session_dd, rename_input],
            outputs=session_msg,
        ).then(
            fn=refresh_sessions,
            outputs=[session_dd, episodes_table, episode_dd, move_target_dd],
        )
        delete_session_btn.click(
            fn=on_delete_session, inputs=session_dd,
            outputs=[session_msg, session_dd, episodes_table, episode_dd, move_target_dd],
        )

        # Episode actions
        dl_btn.click(fn=on_download_episode, inputs=episode_dd, outputs=dl_file)
        del_episode_btn.click(
            fn=on_delete_episode, inputs=[episode_dd, session_dd],
            outputs=[episode_msg, episodes_table, episode_dd, move_target_dd],
        )
        move_btn.click(
            fn=on_move_episodes, inputs=[episode_dd, move_target_dd, session_dd],
            outputs=[episode_msg, episodes_table, episode_dd, move_target_dd],
        )

        # Replay
        replay_btn.click(
            fn=on_replay_start, inputs=episode_dd,
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
            fn=on_hf_upload, inputs=[episode_dd, hf_repo],
            outputs=hf_upload_msg,
        )

        # SLAM
        slam_btn.click(
            fn=on_slam_run, inputs=[episode_dd, hf_repo],
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
            outputs=[session_dd, episodes_table, episode_dd, move_target_dd],
        )
        demo.load(fn=check_hf_auth, outputs=hf_status)

    return demo
