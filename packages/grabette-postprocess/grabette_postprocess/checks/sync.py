"""Camera ↔ OAK-IMU synchronization check for the OAK + Arducam rig.

Correlates frame-to-frame optical-flow magnitude (a proxy for the angular
velocity a camera sees) with the OAK gyroscope norm. A timing offset shifts the
cross-correlation peak away from zero lag; > 20 ms typically degrades SLAM and
the camera↔pose alignment in the LeRobot dataset.

The reusable core lives here (in the package) so both the CLI
(scripts/checks/check_sync.py) and the HF Space pipeline can call it — the CLI
reports both camera↔gyro pairs, while `check_sync` below returns only the
cross-device arducam↔gyro verdict (the OAK left camera and the gyro share the
OAK's hardware clock, so that pair is low-risk; the arducam is a separate device
whose clock alignment with the OAK-derived trajectory/action stream is what the
policy actually trains on).
"""

import json
from pathlib import Path

import cv2
import numpy as np

from grabette_postprocess.convert import fit_device_to_host_s

# Lag thresholds (seconds): below GOOD is fine, between is marginal, above breaks
# visual-inertial SLAM and the camera↔pose alignment.
_GOOD_LAG_S = 0.020
_MARGINAL_LAG_S = 0.050
_LOW_CORR = 0.3  # below this the signals barely move together (little motion / desync)


def _samples(path: Path) -> list:
    with open(path) as f:
        return json.load(f).get("samples", [])


def compute_optical_flow_magnitude(
    video_path: Path,
    frame_ts_s: np.ndarray | None = None,
    max_frames: int = 500,
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
    midpoint — the gyro is instantaneous, and midpoint stamping removes the
    systematic half-frame bias an endpoint stamp would introduce.
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


def gyro_on_host_timeline(device_ts_s: np.ndarray, episode_dir: Path) -> np.ndarray:
    """Map gyro device-clock timestamps onto the OAK *frame* host timeline — the
    clock the SLAM trajectory and the Arducam alignment live on — using the
    affine fit from oakd_left_timestamps.json (the same fit convert.py applies to
    the IMU). Needed for the cross-device Arducam↔gyro comparison, since the
    Arducam only carries host timestamps. Returns the input unchanged when no
    device↔host fit is available (legacy: ts are already host_ms)."""
    ts_path = episode_dir / "oakd_left_timestamps.json"
    if not ts_path.is_file():
        return device_ts_s
    fit = fit_device_to_host_s(_samples(ts_path))
    if fit is None:
        return device_ts_s
    slope, intercept = fit
    return slope * device_ts_s + intercept


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


def cross_correlate_signals(
    t1: np.ndarray, s1: np.ndarray,
    t2: np.ndarray, s2: np.ndarray,
    max_lag_s: float = 0.5,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Cross-correlate two irregularly sampled signals.

    Resamples both to a uniform grid, normalizes, and computes cross-correlation.
    Returns (best_lag_s, correlation_at_best_lag, lags_array, correlation_array).
    A positive lag means signal 1 (camera) leads signal 2 (gyro).
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
    """Map a (lag, correlation) pair to a (verdict, note) — the shared rule behind
    both the CLI report and the Space's pre-SLAM sync gate.

    verdict ∈ {GOOD, MARGINAL, BAD}; note is a one-line human explanation (empty
    for a clean GOOD). A low correlation is appended to the note as a caveat."""
    if abs(best_lag) < _GOOD_LAG_S:
        verdict, note = "GOOD", ""
    elif abs(best_lag) < _MARGINAL_LAG_S:
        verdict, note = "MARGINAL", "20–50 ms offset — may degrade SLAM / camera↔pose alignment."
    else:
        verdict, note = "BAD", ">50 ms offset — breaks visual-inertial SLAM / camera↔pose alignment."
    if best_corr < _LOW_CORR:
        caveat = f"low correlation ({best_corr:.2f}): little motion, or broken/desynced data."
        note = f"{note} {caveat}".strip()
    return verdict, note


def check_sync(ep_dir: Path, max_frames: int = 500) -> dict | None:
    """Arducam ↔ OAK-gyro synchronization verdict for one raw episode (pre-SLAM).

    Returns {verdict, lag, corr, approx, message} where verdict ∈ {GOOD, MARGINAL,
    BAD}, lag is in seconds (camera leads gyro when positive), and message is a
    human-readable one-liner. Returns None when the check can't run (no IMU/gyro,
    missing/short arducam video) — the caller treats that as "not checked", not a
    failure.
    """
    ep_dir = Path(ep_dir)
    imu_path = ep_dir / "oakd_imu.json"
    video = ep_dir / "raw_video.mp4"
    if not imu_path.is_file() or not video.is_file():
        return None

    try:
        gyro_ts, gyro_norm = load_oak_gyro_norm(imu_path)
    except (ValueError, KeyError):
        return None

    frame_ts = arducam_frame_ts(ep_dir)
    flow_ts, flow_mag = compute_optical_flow_magnitude(video, frame_ts, max_frames)
    if len(flow_mag) < 2:
        return None

    # Arducam carries only host timestamps, so bring the device-clock gyro onto
    # the OAK frame host timeline (the trajectory's clock) before correlating.
    gyro_host_ts = gyro_on_host_timeline(gyro_ts, ep_dir)
    best_lag, best_corr, _, _ = cross_correlate_signals(flow_ts, flow_mag, gyro_host_ts, gyro_norm)
    verdict, note = classify_lag(best_lag, best_corr)
    approx = frame_ts is None
    message = (f"arducam↔gyro lag {best_lag * 1000:+.0f} ms (corr {best_corr:.2f})"
               + (" [approx: no frame timestamps]" if approx else "")
               + (f" — {note}" if note else ""))
    return {"verdict": verdict, "lag": best_lag, "corr": best_corr,
            "approx": approx, "message": message}
