"""Grabette HuggingFace Spaces app.

Standalone Gradio interface that connects to a grabette robot API
via tunnel URL (set GRABETTE_API_URL as a Space secret).
"""

from __future__ import annotations

import io
import os

import gradio as gr
from PIL import Image

from api_client import GrabetteClient

# Default client — can be overridden via the URL input in the UI
_default_url = os.environ.get("GRABETTE_API_URL", "")
_client: GrabetteClient | None = None


def get_client() -> GrabetteClient | None:
    return _client


def connect(url: str):
    """Connect to a grabette robot API."""
    global _client
    url = url.strip()
    if not url:
        _client = None
        return "Disconnected"
    _client = GrabetteClient(base_url=url)
    info = _client.get_system_info()
    if info is None:
        _client = None
        return f"Failed to connect to {url}"
    host = info.get("hostname", "?")
    return f"Connected to {host} ({url})"


# ── Callbacks ─────────────────────────────────────────────────────────

def get_camera_frame():
    c = get_client()
    if c is None:
        return None
    data = c.get_snapshot()
    if data is None:
        return None
    try:
        return Image.open(io.BytesIO(data))
    except Exception:
        return None


def get_sensor_state():
    c = get_client()
    if c is None:
        return "Not connected", "Not connected", gr.update(active=True)
    state = c.get_state()
    if state is None:
        return "Disconnected", "Disconnected", gr.update(active=True)
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
    cap = state.get("capture", {})
    capturing = cap.get("is_capturing", False)
    if capturing:
        cap_text = (
            f"\u25cf RECORDING  {cap.get('session_id', '')}\n"
            f"Duration: {cap.get('duration_seconds', 0):.1f}s\n"
            f"Frames: {cap.get('frame_count', 0)}  |  "
            f"IMU: {cap.get('imu_sample_count', 0)}"
        )
    else:
        cap_text = "\u25cb Idle"
    # Pause camera polling during capture to protect sync
    camera_active = not capturing
    return imu_text, cap_text, gr.update(active=camera_active)


def on_start_capture():
    c = get_client()
    if c is None:
        return "Not connected"
    result = c.start_capture()
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Started: {result.get('session_id', '?')}"


def on_stop_capture():
    c = get_client()
    if c is None:
        return "Not connected"
    result = c.stop_capture()
    if "error" in result:
        return f"Error: {result['error']}"
    dur = result.get("duration_seconds", 0)
    frames = result.get("frame_count", 0)
    return f"Stopped \u2014 {dur:.1f}s, {frames} frames"


def refresh_sessions():
    c = get_client()
    if c is None:
        return [], gr.update(choices=[], value=None)
    sessions = c.list_sessions()
    rows = []
    ids = []
    for s in sessions:
        rows.append([
            s["session_id"],
            f"{s['duration_seconds']:.1f}s",
            s["frame_count"],
            s["imu_sample_count"],
        ])
        ids.append(s["session_id"])
    return rows, gr.update(choices=ids, value=ids[0] if ids else None)


def on_download(session_id: str | None):
    c = get_client()
    if c is None or not session_id:
        return None
    return c.download_session(session_id)


def on_delete(session_id: str | None):
    c = get_client()
    if c is None or not session_id:
        return "No session selected", [], gr.update(choices=[], value=None)
    result = c.delete_session(session_id)
    rows, dropdown = refresh_sessions()
    if "error" in result:
        return f"Error: {result['error']}", rows, dropdown
    return f"Deleted {session_id}", rows, dropdown


def get_system_bar():
    c = get_client()
    if c is None:
        return "Not connected"
    info = c.get_system_info()
    if info is None:
        return "Disconnected"
    parts = [info.get("hostname", "?")]
    if "cpu_temp_c" in info:
        parts.append(f"{info['cpu_temp_c']}\u00b0C")
    if "disk_free_gb" in info:
        parts.append(f"{info['disk_free_gb']}GB free")
    if "ip" in info:
        parts.append(info["ip"])
    return " | ".join(parts)


def on_hf_auth(token: str):
    c = get_client()
    if c is None:
        return "Not connected"
    if not token:
        return "No token provided"
    result = c.hf_set_auth(token)
    if result.get("authenticated"):
        user = result.get("user", {})
        return f"Authenticated as {user.get('username', '?')}"
    return f"Auth failed: {result.get('error', 'unknown')}"


def check_hf_auth():
    c = get_client()
    if c is None:
        return "Not connected"
    result = c.hf_check_auth()
    if result.get("authenticated"):
        user = result.get("user", {})
        return f"Authenticated as {user.get('username', '?')}"
    return "Not authenticated"


def on_hf_upload(session_id: str | None, repo_id: str):
    c = get_client()
    if c is None:
        return "Not connected"
    if not session_id:
        return "Select a session first"
    if not repo_id:
        return "Enter a repo ID"
    result = c.hf_upload_session(session_id, repo_id)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Upload started (job: {result.get('job_id', '?')})"


def on_slam_run(session_id: str | None, repo_id: str):
    c = get_client()
    if c is None:
        return "Not connected"
    if not session_id:
        return "Select a session first"
    if not repo_id:
        return "Enter a HuggingFace repo ID first"
    result = c.slam_run(session_id, repo_id)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"SLAM started (job: {result.get('job_id', '?')})"


def get_slam_status():
    c = get_client()
    if c is None:
        return "Not connected"
    jobs = c.hf_list_jobs()
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


# ── Build UI ──────────────────────────────────────────────────────────

with gr.Blocks(title="Grabette") as demo:
    gr.Markdown("# GRABETTE")

    # ── Connection ────────────────────────────────────────────────
    with gr.Row():
        url_input = gr.Textbox(
            label="Robot API URL",
            value=_default_url,
            placeholder="https://xxx.trycloudflare.com",
            scale=3,
        )
        connect_btn = gr.Button("Connect", variant="primary", scale=1)
    connection_status = gr.Textbox(
        show_label=False, interactive=False, max_lines=1,
    )

    # ── Live view ─────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            camera_img = gr.Image(
                label="Camera Live View",
                height=480,
            )
        with gr.Column(scale=1):
            imu_box = gr.Textbox(
                label="IMU Live", lines=2, interactive=False,
            )
            capture_box = gr.Textbox(
                label="Capture Status", lines=3, interactive=False,
            )
            with gr.Row():
                start_btn = gr.Button("Start Capture", variant="primary")
                stop_btn = gr.Button("Stop Capture", variant="stop")
            capture_msg = gr.Textbox(
                show_label=False, interactive=False, max_lines=1,
            )

    # ── Sessions ──────────────────────────────────────────────────
    gr.Markdown("### Sessions")
    with gr.Row():
        refresh_btn = gr.Button("Refresh", size="sm")
        session_dd = gr.Dropdown(label="Selected Session", interactive=True)
    sessions_table = gr.Dataframe(
        headers=["Session ID", "Duration", "Frames", "IMU Samples"],
        interactive=False,
    )
    with gr.Row():
        dl_btn = gr.Button("Download .tar.gz", size="sm")
        del_btn = gr.Button("Delete", variant="stop", size="sm")
    dl_file = gr.File(label="Download")
    del_msg = gr.Textbox(show_label=False, interactive=False, max_lines=1)

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
        hf_upload_btn = gr.Button("Upload Session", size="sm", scale=1)
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

    # Connection
    connect_btn.click(fn=connect, inputs=url_input, outputs=connection_status)

    # Capture
    start_btn.click(fn=on_start_capture, outputs=capture_msg)
    stop_btn.click(fn=on_stop_capture, outputs=capture_msg).then(
        fn=refresh_sessions, outputs=[sessions_table, session_dd],
    )

    # Sessions
    refresh_btn.click(fn=refresh_sessions, outputs=[sessions_table, session_dd])
    dl_btn.click(fn=on_download, inputs=session_dd, outputs=dl_file)
    del_btn.click(
        fn=on_delete, inputs=session_dd,
        outputs=[del_msg, sessions_table, session_dd],
    )

    # HuggingFace
    hf_auth_btn.click(fn=on_hf_auth, inputs=hf_token, outputs=hf_status)
    hf_upload_btn.click(
        fn=on_hf_upload, inputs=[session_dd, hf_repo], outputs=hf_upload_msg,
    )

    # SLAM
    slam_btn.click(
        fn=on_slam_run, inputs=[session_dd, hf_repo], outputs=slam_status,
    )

    # Periodic updates (Gradio 6 Timer)
    # Camera timer is paused during capture to protect sync
    camera_timer = gr.Timer(0.2)
    camera_timer.tick(fn=get_camera_frame, outputs=camera_img)

    state_timer = gr.Timer(0.5)
    state_timer.tick(
        fn=get_sensor_state, outputs=[imu_box, capture_box, camera_timer],
    )

    system_timer = gr.Timer(10)
    system_timer.tick(fn=get_system_bar, outputs=system_bar)

    # Auto-connect if URL was provided via env var
    if _default_url:
        demo.load(fn=lambda: connect(_default_url), outputs=connection_status)
        demo.load(fn=refresh_sessions, outputs=[sessions_table, session_dd])
        demo.load(fn=check_hf_auth, outputs=hf_status)

if __name__ == "__main__":
    demo.launch()
