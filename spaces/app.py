"""Grabette HuggingFace Spaces app.

Standalone Gradio interface that connects to a grabette robot API
via tunnel URL (set GRABETTE_API_URL as a Space secret).
"""

from __future__ import annotations

import io
import math
import os
import time
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    url = url.strip().rstrip("/")
    if not url:
        _client = None
        return "Disconnected", _viewer_placeholder()
    _client = GrabetteClient(base_url=url)
    info = _client.get_system_info()
    if info is None:
        _client = None
        return f"Failed to connect to {url}", _viewer_placeholder()
    host = info.get("hostname", "?")
    viewer_html = (
        f'<iframe src="{url}/viewer" '
        'style="width:100%;height:350px;border:none;'
        'border-radius:8px;background:#1a1a2e;"></iframe>'
    )
    return f"Connected to {host} ({url})", viewer_html


def _viewer_placeholder():
    return (
        '<div style="width:100%;height:350px;border-radius:8px;'
        'background:#1a1a2e;display:flex;align-items:center;'
        'justify-content:center;color:#556688;font:14px monospace;">'
        'Connect to robot to load 3D model</div>'
    )


# ── Rolling data buffers for plots ────────────────────────────────────
_plot_maxlen = 30  # ~15 s at 0.5 s poll
_imu_t: deque[float] = deque(maxlen=_plot_maxlen)
_imu_accel: deque[list[float]] = deque(maxlen=_plot_maxlen)
_imu_gyro: deque[list[float]] = deque(maxlen=_plot_maxlen)
_ang_t: deque[float] = deque(maxlen=_plot_maxlen)
_ang_vals: deque[list[float]] = deque(maxlen=_plot_maxlen)


def _make_imu_plot():
    if not _imu_t:
        return None
    now = time.monotonic()
    t = np.array([x - now for x in _imu_t])
    accel = np.array(list(_imu_accel))
    gyro = np.array(list(_imu_gyro))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 3.5), tight_layout=True)
    for i, (c, label) in enumerate(zip("rgb", "XYZ")):
        ax1.plot(t, accel[:, i], color=c, linewidth=1, label=label)
    ax1.set_ylabel("m/s\u00b2")
    ax1.set_title("Accelerometer", fontsize=9)
    ax1.legend(loc="upper left", fontsize=7, ncol=3)
    ax1.grid(True, alpha=0.3)
    for i, (c, label) in enumerate(zip("rgb", "XYZ")):
        ax2.plot(t, gyro[:, i], color=c, linewidth=1, label=label)
    ax2.set_ylabel("rad/s")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Gyroscope", fontsize=9)
    ax2.legend(loc="upper left", fontsize=7, ncol=3)
    ax2.grid(True, alpha=0.3)
    plt.close(fig)
    return fig


def _make_angle_plot():
    if not _ang_t:
        return None
    now = time.monotonic()
    t = np.array([x - now for x in _ang_t])
    vals = np.array(list(_ang_vals))
    fig, ax = plt.subplots(figsize=(5, 3.5), tight_layout=True)
    ax.plot(t, vals[:, 0], color="#4488cc", linewidth=1.5, label="Proximal")
    ax.plot(t, vals[:, 1], color="#cc8844", linewidth=1.5, label="Distal")
    ax.set_ylabel("Degrees")
    ax.set_xlabel("Time (s)")
    ax.set_title("Angle Sensors", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.close(fig)
    return fig


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
        return ("Not connected", "Not connected", "Not connected",
                None, None, gr.update(active=True))
    state = c.get_state()
    if state is None:
        return ("Disconnected", "Disconnected", "Disconnected",
                None, None, gr.update(active=True))
    imu = state.get("imu")
    if imu:
        a = imu["accel"]
        g = imu["gyro"]
        imu_text = (
            f"Accel: [{a[0]:+8.3f}, {a[1]:+8.3f}, {a[2]:+8.3f}] m/s\u00b2\n"
            f"Gyro:  [{g[0]:+8.4f}, {g[1]:+8.4f}, {g[2]:+8.4f}] rad/s"
        )
        _imu_t.append(time.monotonic())
        _imu_accel.append(a)
        _imu_gyro.append(g)
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
        _ang_t.append(time.monotonic())
        _ang_vals.append([p_deg, d_deg])
    else:
        angle_text = "No angle data"
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
    # Plots
    imu_fig = _make_imu_plot()
    angle_fig = _make_angle_plot()
    # Pause camera polling during capture to protect sync
    camera_active = not capturing
    return (imu_text, angle_text, cap_text,
            imu_fig, angle_fig, gr.update(active=camera_active))


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
            s.get("angle_sample_count", 0),
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
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            camera_img = gr.Image(
                label="Camera Live View",
                height=350,
            )
        with gr.Column(scale=1):
            viewer_iframe = gr.HTML(
                value=_viewer_placeholder(),
                label="3D Model",
            )
        with gr.Column(scale=1):
            imu_box = gr.Textbox(
                label="IMU Live", lines=2, interactive=False,
            )
            angle_box = gr.Textbox(
                label="Angle Sensors", lines=2, interactive=False,
            )
            capture_box = gr.Textbox(
                label="Capture Status", lines=4, interactive=False,
            )
            with gr.Row():
                start_btn = gr.Button("Start Capture", variant="primary")
                stop_btn = gr.Button("Stop Capture", variant="stop")
            capture_msg = gr.Textbox(
                show_label=False, interactive=False, max_lines=1,
            )

    # ── Sensor plots ──────────────────────────────────────────────
    with gr.Row(equal_height=True):
        imu_plot_out = gr.Plot(label="IMU")
        angle_plot_out = gr.Plot(label="Angle Sensors")

    # ── Sessions ──────────────────────────────────────────────────
    gr.Markdown("### Sessions")
    with gr.Row():
        refresh_btn = gr.Button("Refresh", size="sm")
        session_dd = gr.Dropdown(label="Selected Session", interactive=True)
    sessions_table = gr.Dataframe(
        headers=["Session ID", "Duration", "Frames", "IMU", "Angle"],
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
    connect_btn.click(
        fn=connect, inputs=url_input,
        outputs=[connection_status, viewer_iframe],
    )

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
        fn=get_sensor_state,
        outputs=[imu_box, angle_box, capture_box,
                 imu_plot_out, angle_plot_out, camera_timer],
    )

    system_timer = gr.Timer(10)
    system_timer.tick(fn=get_system_bar, outputs=system_bar)

    # Auto-connect if URL was provided via env var
    if _default_url:
        demo.load(
            fn=lambda: connect(_default_url),
            outputs=[connection_status, viewer_iframe],
        )
        demo.load(fn=refresh_sessions, outputs=[sessions_table, session_dd])
        demo.load(fn=check_hf_auth, outputs=hf_status)

if __name__ == "__main__":
    demo.launch()
