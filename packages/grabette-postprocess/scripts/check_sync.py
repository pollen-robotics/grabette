#!/usr/bin/env python3
"""
Check camera-IMU synchronization for the OAK + Arducam rig.

Correlates frame-to-frame optical flow magnitude (a proxy for the angular
velocity seen by a camera) with the OAK gyroscope norm. A timing offset shifts
the cross-correlation peak away from zero lag; > 20ms typically degrades SLAM
and the camera↔pose alignment in the LeRobot dataset.

Two pairs are checked (the IMU lives on the OAK, so it is the common reference):

  1. OAK left camera (oakd_left.mp4)  ↔  OAK gyro (oakd_imu.json)
     Same device, hardware-stamped — validates the visual-inertial SLAM inputs.
     Expect a near-zero lag.

  2. Arducam (raw_video.mp4)          ↔  OAK gyro (oakd_imu.json)
     Cross-device — validates that the observation camera shares a clock with
     the trajectory/action stream (which is derived from the OAK). This is the
     alignment the policy actually trains on.

All timestamps are taken on the host_ms clock so the two pairs are comparable.

Usage:
    uv run python scripts/check_sync.py /path/to/episode
    uv run python scripts/check_sync.py /path/to/episode --plot sync.png
"""

import json
from pathlib import Path

import click
import cv2
import numpy as np


def _samples(path: Path) -> list:
    with open(path) as f:
        return json.load(f).get("samples", [])


def compute_optical_flow_magnitude(
    video_path: Path,
    frame_ts_s: np.ndarray | None = None,
    max_frames: int = 500,
    resize: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame dense optical flow magnitude from a video.

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

    Reads oakd_imu.json (flat schema, kind == "gyro"). Timestamps use the host_ms
    clock (seconds), matching the camera frame timestamps.
    """
    gyro = [s for s in _samples(imu_path) if s.get("kind") == "gyro"]
    if not gyro:
        raise ValueError(f"No gyro samples in {imu_path}")
    ts = np.array([s["host_ms"] for s in gyro]) * 1e-3
    norms = np.linalg.norm([s["value"] for s in gyro], axis=1)
    return ts, norms


def _oak_left_frame_ts(episode_dir: Path) -> np.ndarray | None:
    """Per-frame host_ms timestamps (seconds) for oakd_left.mp4, or None."""
    ts_path = episode_dir / "oakd_left_timestamps.json"
    if not ts_path.is_file():
        return None
    return np.array([s["host_ms"] for s in _samples(ts_path)]) * 1e-3


def _arducam_frame_ts(episode_dir: Path) -> np.ndarray | None:
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


def _verdict(best_lag: float, best_corr: float, label: str, approx: bool) -> dict:
    """Print + return a verdict dict for one camera↔gyro correlation."""
    note = " (approx: no frame timestamps)" if approx else ""
    print(f"\n  {label}{note}")
    print(f"    best lag:    {best_lag*1000:+.1f} ms")
    print(f"    correlation: {best_corr:.3f}")
    if abs(best_lag) < 0.020:
        verdict = "GOOD"
        print(f"    → GOOD (< 20ms)")
    elif abs(best_lag) < 0.050:
        verdict = "MARGINAL"
        print(f"    → MARGINAL (20-50ms) — may degrade SLAM / alignment")
    else:
        verdict = "BAD"
        print(f"    → BAD (> 50ms) — breaks visual-inertial SLAM / camera-pose alignment")
    if best_corr < 0.3:
        print(f"    ! low correlation ({best_corr:.3f}): little motion, or broken/desynced data")
    return {"label": label, "lag": best_lag, "corr": best_corr, "verdict": verdict}


@click.command()
@click.argument("episode_dir", type=click.Path(exists=True))
@click.option("--max_frames", type=int, default=500, help="Max video frames per camera (default: 500)")
@click.option("--plot", "-p", type=click.Path(), default=None, help="Save a correlation plot (PNG)")
def main(episode_dir, max_frames, plot):
    """Check camera-IMU sync via optical-flow / OAK-gyro cross-correlation."""
    episode_dir = Path(episode_dir)
    imu_path = episode_dir / "oakd_imu.json"
    if not imu_path.is_file():
        raise click.ClickException(f"No oakd_imu.json in {episode_dir}")

    print("Loading OAK gyro...")
    gyro_ts, gyro_norm = load_oak_gyro_norm(imu_path)
    print(f"  {len(gyro_norm)} samples, {gyro_ts[-1]-gyro_ts[0]:.1f}s")

    pairs = [
        ("OAK left  ↔ OAK gyro", episode_dir / "oakd_left.mp4", _oak_left_frame_ts(episode_dir)),
        ("Arducam   ↔ OAK gyro", episode_dir / "raw_video.mp4", _arducam_frame_ts(episode_dir)),
    ]

    results = []
    plot_data = []
    for label, video, frame_ts in pairs:
        if not video.is_file():
            print(f"\n  {label}: skip ({video.name} missing)")
            continue
        print(f"\nComputing optical flow: {video.name}...")
        flow_ts, flow_mag = compute_optical_flow_magnitude(video, frame_ts, max_frames)
        if len(flow_mag) < 2:
            print(f"  {label}: skip (too few frames)")
            continue
        print(f"  {len(flow_mag)} frames, {flow_ts[-1]-flow_ts[0]:.1f}s")
        best_lag, best_corr, lag_times, corr = cross_correlate_signals(
            flow_ts, flow_mag, gyro_ts, gyro_norm)
        results.append(_verdict(best_lag, best_corr, label, approx=frame_ts is None))
        plot_data.append((label, flow_ts, flow_mag, lag_times, corr, best_lag))

    print(f"\n{'='*52}")
    for r in results:
        print(f"  {r['verdict']:9s} {r['label']}  ({r['lag']*1000:+.1f}ms, corr={r['corr']:.2f})")
    print(f"{'='*52}")

    if plot and plot_data:
        _save_plot(plot, plot_data, gyro_ts, gyro_norm)


def _save_plot(path, plot_data, gyro_ts, gyro_norm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed, skipping plot")
        return

    n = len(plot_data)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)
    gnorm = gyro_norm / np.max(gyro_norm) if np.max(gyro_norm) > 0 else gyro_norm
    for row, (label, flow_ts, flow_mag, lag_times, corr, best_lag) in enumerate(plot_data):
        fnorm = flow_mag / np.max(flow_mag) if np.max(flow_mag) > 0 else flow_mag
        ax = axes[row][0]
        ax.plot(flow_ts, fnorm, label="optical flow", alpha=0.8)
        ax.plot(gyro_ts, gnorm, label="OAK gyro", alpha=0.8)
        ax.set_title(f"{label} — signals")
        ax.set_xlabel("time (s)"); ax.legend()
        ax = axes[row][1]
        ax.plot(lag_times * 1000, corr)
        ax.axvline(best_lag * 1000, color="r", ls="--", label=f"{best_lag*1000:+.1f}ms")
        ax.axvline(0, color="gray", ls=":", alpha=0.5)
        ax.set_title(f"{label} — cross-correlation")
        ax.set_xlabel("lag (ms)"); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nPlot saved to: {path}")


if __name__ == "__main__":
    main()
