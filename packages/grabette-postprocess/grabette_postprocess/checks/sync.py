"""Temporal-alignment checks for the OAK + Arducam rig.

All checks cross-correlate two motion signals resampled to a common grid; the
lag of the correlation peak is the timing offset between the two streams. Three
checks, each answering a different alignment question:

  1. check_oak_imu          — OAK left camera ↔ OAK gyro (SLAM VIO health).
     Same device, shared OAK clock (device_us). Confirms the visual-inertial
     SLAM *inputs* are time-aligned. A prerequisite for trusting the trajectory.

  2. check_image_trajectory — Arducam optical flow ↔ SLAM trajectory angular
     velocity (the end-to-end image↔pose alignment the policy trains on). This
     goes through the *actual* SLAM output, so it catches any timestamp handling
     SLAM introduces — unlike correlating against the raw gyro. POST-SLAM:
     needs camera_trajectory.csv.

  3. check_gripper          — Arducam optical flow ↔ gripper joint-angle speed.
     Validates that the angle stream is time-aligned with the images. Joint
     motion is NOT correlated with camera egomotion, so this is only meaningful
     on a deliberate gripper gesture filmed with the camera roughly still (open/
     close the gripper in view). A low correlation means egomotion-dominated or
     no gesture — inconclusive, not a failure.

The reusable core (optical flow, cross-correlation, classification, the data
loaders) lives here in the package so the CLI (scripts/checks/check_sync.py) and
any future caller share it. The expensive Arducam Farneback flow is computed once
and passed into both the trajectory and gripper checks via the `flow` argument.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from grabette_postprocess.trajectory import load_trajectory_csv

# Lag thresholds (seconds): below GOOD is fine, between is marginal, above breaks
# the temporal alignment between the two streams.
_GOOD_LAG_S = 0.020
_MARGINAL_LAG_S = 0.050
_LOW_CORR = 0.3  # below this the signals barely move together (little motion / desync)


def _samples(path: Path) -> list:
    with open(path) as f:
        return json.load(f).get("samples", [])


def compute_optical_flow_magnitude(
    video_path: Path,
    frame_ts_s: np.ndarray | None = None,
    max_frames: int = 300,
    resize: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame dense optical-flow magnitude from a video.

    Args:
        video_path: video file.
        frame_ts_s: per-frame timestamps in seconds (one per decoded frame, same
            order as the stream). If None, falls back to frame_index / fps.
        max_frames: cap on frames processed (optical flow is the slow part).
        resize: longest side the frames are scaled to before flow.

    Returns (timestamps_s, flow_magnitude). The flow between frame i-1 and i
    measures motion over [t_{i-1}, t_i], so it is stamped at the interval
    midpoint — the reference signals are instantaneous, and midpoint stamping
    removes the systematic half-frame bias an endpoint stamp would introduce.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total, max_frames)
    if frame_ts_s is not None:
        n_frames = min(n_frames, len(frame_ts_s))

    def ts(i):
        return frame_ts_s[i] if frame_ts_s is not None else i / fps

    timestamps, flow_mags = [], []
    prev_gray = None
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        scale = resize / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_mags.append(float(np.mean(mag)))
            timestamps.append(0.5 * (ts(i - 1) + ts(i)))
        prev_gray = gray

    cap.release()
    return np.array(timestamps), np.array(flow_mags)


def load_oak_gyro_norm(imu_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the OAK gyroscope and return (timestamps_s, angular_velocity_norm).

    Reads oakd_imu.json (flat schema, kind == "gyro"). Timestamps use the OAK
    device clock (device_us), which the cameras and IMU share on the OAK
    hardware, so the camera↔gyro comparison reflects true capture timing rather
    than USB-arrival (host_ms) jitter. Falls back to host_ms for legacy
    recordings without device_us.
    """
    gyro = [s for s in _samples(imu_path) if s.get("kind") == "gyro"]
    if not gyro:
        raise ValueError(f"No gyro samples in {imu_path}")
    if all("device_us" in s for s in gyro):
        ts = np.array([s["device_us"] for s in gyro], dtype=float) * 1e-6
    else:
        ts = np.array([s["host_ms"] for s in gyro], dtype=float) * 1e-3
    norms = np.linalg.norm([s["value"] for s in gyro], axis=1)
    return ts, norms


def oak_left_frame_ts(episode_dir: Path) -> np.ndarray | None:
    """Per-frame device_us timestamps (seconds) for oakd_left.mp4 — the OAK
    hardware clock, matching the gyro from load_oak_gyro_norm. Falls back to
    host_ms when device_us is absent. None when the file is missing."""
    ts_path = episode_dir / "oakd_left_timestamps.json"
    if not ts_path.is_file():
        return None
    samples = _samples(ts_path)
    if all("device_us" in s for s in samples):
        return np.array([s["device_us"] for s in samples], dtype=float) * 1e-6
    return np.array([s["host_ms"] for s in samples], dtype=float) * 1e-3


def arducam_frame_ts(episode_dir: Path) -> np.ndarray | None:
    """Per-frame timestamps (seconds) for raw_video.mp4 from frame_timestamps.json,
    or None when absent/empty (caller falls back to uniform fps)."""
    ft = episode_dir / "frame_timestamps.json"
    if not ft.is_file():
        return None
    with open(ft) as f:
        ts_ms = json.load(f)
    if not ts_ms:
        return None
    return np.array(ts_ms, dtype=float) * 1e-3


def trajectory_angular_velocity(
    episode_dir: Path, max_gap_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Angular speed (rad/s) of the SLAM trajectory vs time (seconds).

    Reads camera_trajectory.csv (columns include timestamp, q_x..q_w, is_lost),
    drops lost frames, and takes the geodesic angle between consecutive pose
    orientations divided by dt. Stamped at the interval midpoint to match
    compute_optical_flow_magnitude. The Arducam is rigid with the OAK, so this
    angular velocity tracks the same egomotion the Arducam sees — correlating the
    two measures the image↔pose timing offset end-to-end through SLAM.

    Pairs spanning a gap > max_gap_s (e.g. across a lost-tracking stretch) are
    dropped to avoid spurious low-rate velocities. Returns None when the
    trajectory is missing or has < 2 tracked poses.
    """
    path = episode_dir / "camera_trajectory.csv"
    if not path.is_file():
        return None
    df = load_trajectory_csv(path)
    if "is_lost" in df.columns:
        df = df[~df["is_lost"].astype(bool)]
    if len(df) < 2:
        return None

    t = df["timestamp"].to_numpy(dtype=float)
    quats = df[["q_x", "q_y", "q_z", "q_w"]].to_numpy(dtype=float)
    rots = Rotation.from_quat(quats)  # scalar-last [x, y, z, w]
    rel = rots[:-1].inv() * rots[1:]
    ang = rel.magnitude()  # geodesic rotation angle (rad) between consecutive poses
    dt = np.diff(t)
    ok = (dt > 1e-6) & (dt < max_gap_s)
    if ok.sum() < 2:
        return None
    omega = ang[ok] / dt[ok]
    ts_mid = (0.5 * (t[:-1] + t[1:]))[ok]
    return ts_mid, omega


def angle_velocity(episode_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Gripper joint-angle speed vs time (seconds) from angle_data.json.

    Schema {"samples": [{"cts": <ms>, "value": [distal, proximal]}]}. Returns the
    norm of the per-step change across both joints divided by dt, stamped at the
    interval midpoint. Spikes when the gripper opens/closes. Returns None when the
    file is missing or has < 2 samples.
    """
    path = episode_dir / "angle_data.json"
    if not path.is_file():
        return None
    samples = _samples(path)
    if len(samples) < 2:
        return None
    cts = np.array([s["cts"] for s in samples], dtype=float) * 1e-3
    vals = np.array([s["value"] for s in samples], dtype=float)  # (N, 2)
    dt = np.diff(cts)
    ok = dt > 1e-6
    if ok.sum() < 2:
        return None
    speed = np.linalg.norm(np.diff(vals, axis=0), axis=1)[ok] / dt[ok]
    ts_mid = (0.5 * (cts[:-1] + cts[1:]))[ok]
    return ts_mid, speed


def cross_correlate_signals(
    t1: np.ndarray, s1: np.ndarray,
    t2: np.ndarray, s2: np.ndarray,
    max_lag_s: float = 0.5,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Cross-correlate two irregularly sampled signals.

    Resamples both to a uniform grid, normalizes, and computes cross-correlation.
    Returns (best_lag_s, correlation_at_best_lag, lags_array, correlation_array).
    A positive lag means signal 1 (the camera/image) leads signal 2 (the
    reference: gyro, trajectory or angle stream).
    """
    dt = 0.005  # 5ms grid (~200Hz)
    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])
    if t_end <= t_start:
        return 0.0, 0.0, np.array([0.0]), np.array([0.0])

    t_uniform = np.arange(t_start, t_end, dt)
    s1u = np.interp(t_uniform, t1, s1)
    s2u = np.interp(t_uniform, t2, s2)

    s1u = s1u - np.mean(s1u)
    if np.std(s1u) > 0:
        s1u /= np.std(s1u)
    s2u = s2u - np.mean(s2u)
    if np.std(s2u) > 0:
        s2u /= np.std(s2u)

    max_lag_samples = int(max_lag_s / dt)
    n = len(t_uniform)
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    corr = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag >= 0:
            a, b = s1u[lag:], s2u[:n - lag]
        else:
            a, b = s1u[:n + lag], s2u[-lag:]
        if len(a) > 0:
            corr[i] = np.mean(a * b)

    lag_times = lags * dt
    best_idx = int(np.argmax(corr))
    return float(lag_times[best_idx]), float(corr[best_idx]), lag_times, corr


def classify_lag(best_lag: float, best_corr: float) -> tuple[str, str]:
    """Map a (lag, correlation) pair to a (verdict, note).

    verdict ∈ {GOOD, MARGINAL, BAD}; note is a one-line human explanation (empty
    for a clean GOOD). A low correlation is appended to the note as a caveat."""
    if abs(best_lag) < _GOOD_LAG_S:
        verdict, note = "GOOD", ""
    elif abs(best_lag) < _MARGINAL_LAG_S:
        verdict, note = "MARGINAL", "20–50 ms offset — may degrade temporal alignment."
    else:
        verdict, note = "BAD", ">50 ms offset — breaks temporal alignment."
    if best_corr < _LOW_CORR:
        caveat = f"low correlation ({best_corr:.2f}): little motion, or broken/desynced data."
        note = f"{note} {caveat}".strip()
    return verdict, note


def _arducam_flow(episode_dir: Path, max_frames: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Arducam optical-flow magnitude (shared by the trajectory & gripper checks)."""
    video = episode_dir / "raw_video.mp4"
    if not video.is_file():
        return None, None
    flow_ts, flow_mag = compute_optical_flow_magnitude(
        video, arducam_frame_ts(episode_dir), max_frames)
    if len(flow_mag) < 2:
        return None, None
    return flow_ts, flow_mag


def _result(label: str, ref_label: str, cam_ts, cam_sig, ref_ts, ref_sig,
            *, approx: bool = False) -> dict | None:
    """Cross-correlate camera vs reference and package a verdict dict (or None
    when either signal is too short to correlate)."""
    if cam_ts is None or len(cam_sig) < 2 or ref_ts is None or len(ref_sig) < 2:
        return None
    best_lag, best_corr, lag_times, corr = cross_correlate_signals(
        cam_ts, cam_sig, ref_ts, ref_sig)
    verdict, note = classify_lag(best_lag, best_corr)
    return {
        "label": label, "ref_label": ref_label,
        "verdict": verdict, "lag": best_lag, "corr": best_corr,
        "note": note, "approx": approx,
        "cam_ts": cam_ts, "cam_sig": cam_sig, "ref_ts": ref_ts, "ref_sig": ref_sig,
        "lag_times": lag_times, "corr_curve": corr,
    }


def check_oak_imu(episode_dir: Path, max_frames: int = 300) -> dict | None:
    """OAK left camera ↔ OAK gyro (SLAM VIO health). None when inputs missing."""
    episode_dir = Path(episode_dir)
    video = episode_dir / "oakd_left.mp4"
    imu = episode_dir / "oakd_imu.json"
    if not video.is_file() or not imu.is_file():
        return None
    try:
        gyro_ts, gyro_norm = load_oak_gyro_norm(imu)
    except (ValueError, KeyError):
        return None
    frame_ts = oak_left_frame_ts(episode_dir)
    flow_ts, flow_mag = compute_optical_flow_magnitude(video, frame_ts, max_frames)
    return _result("OAK-left ↔ OAK-gyro (SLAM VIO health)", "OAK gyro",
                   flow_ts, flow_mag, gyro_ts, gyro_norm, approx=frame_ts is None)


def check_image_trajectory(
    episode_dir: Path, max_frames: int = 300, flow: tuple | None = None,
) -> dict | None:
    """Arducam image ↔ SLAM trajectory (image↔pose, end-to-end). POST-SLAM.

    Pass `flow=(flow_ts, flow_mag)` to reuse a precomputed Arducam flow. None when
    the trajectory or the Arducam video is missing.
    """
    episode_dir = Path(episode_dir)
    traj = trajectory_angular_velocity(episode_dir)
    if traj is None:
        return None
    flow_ts, flow_mag = flow if flow is not None else _arducam_flow(episode_dir, max_frames)
    return _result("Arducam ↔ trajectory (image↔pose)", "trajectory |ω|",
                   flow_ts, flow_mag, traj[0], traj[1],
                   approx=arducam_frame_ts(episode_dir) is None)


def check_gripper(
    episode_dir: Path, max_frames: int = 300, flow: tuple | None = None,
) -> dict | None:
    """Arducam image ↔ gripper joint-angle speed (angle-stream sync).

    Only meaningful on a deliberate open/close gesture filmed with low egomotion;
    a low correlation means no gesture / egomotion-dominated (inconclusive). Pass
    `flow=(flow_ts, flow_mag)` to reuse a precomputed Arducam flow. None when the
    angle data or the Arducam video is missing.
    """
    episode_dir = Path(episode_dir)
    av = angle_velocity(episode_dir)
    if av is None:
        return None
    flow_ts, flow_mag = flow if flow is not None else _arducam_flow(episode_dir, max_frames)
    return _result("Arducam ↔ gripper angle (angle sync)", "angle speed",
                   flow_ts, flow_mag, av[0], av[1],
                   approx=arducam_frame_ts(episode_dir) is None)
