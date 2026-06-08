#!/usr/bin/env python3
"""
Check camera-IMU synchronization by correlating optical flow with gyroscope data.

Computes frame-to-frame optical flow magnitude (proxy for angular velocity as
seen by the camera) and compares it with the gyroscope norm. If the sensors are
synchronized, the cross-correlation peak should be near zero lag.

A timing offset > 20ms typically causes SLAM degradation.

Usage:
    uv run python scripts/check_sync.py test_data/test_HF/mapping
    uv run python scripts/check_sync.py test_data/test_HF/episodes/20260331_095957
    uv run python scripts/check_sync.py test_data/test_HF/mapping --plot sync.png
"""

import json
from pathlib import Path

import click
import cv2
import numpy as np


def compute_optical_flow_magnitude(video_path: Path, max_frames: int = 500,
                                   resize: int = 320) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frame optical flow magnitude from video.

    Returns (timestamps_s, flow_magnitude) arrays.
    Timestamps are frame_index / fps, starting at 0.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total, max_frames)

    timestamps = []
    flow_mags = []
    prev_gray = None

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed
        h, w = frame.shape[:2]
        scale = resize / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Dense optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_mags.append(float(np.mean(mag)))
            timestamps.append(i / fps)

        prev_gray = gray

    cap.release()
    return np.array(timestamps), np.array(flow_mags)


def load_gyro_norm(imu_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load gyroscope data and compute angular velocity norm.

    Returns (timestamps_s, gyro_norm) arrays. Timestamps are zero-based
    (offset by first ACCL sample, matching the video clock convention).
    """
    with open(imu_path) as f:
        data = json.load(f)

    streams = data["1"]["streams"]
    gyro_samples = streams["GYRO"]["samples"]
    accl_samples = streams["ACCL"]["samples"]

    # Zero-base timestamps (same convention as grabette_slam LoadTelemetry)
    t0 = accl_samples[0]["cts"] * 1e-3

    timestamps = np.array([s["cts"] * 1e-3 - t0 for s in gyro_samples])
    values = np.array([s["value"] for s in gyro_samples])
    norms = np.linalg.norm(values, axis=1)

    return timestamps, norms


def cross_correlate_signals(
    t1: np.ndarray, s1: np.ndarray,
    t2: np.ndarray, s2: np.ndarray,
    max_lag_s: float = 0.5,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Cross-correlate two irregularly sampled signals.

    Resamples both to a uniform grid, normalizes, and computes cross-correlation.

    Returns (best_lag_s, correlation_at_best_lag, lags_array, correlation_array).
    """
    # Resample both signals to uniform grid at ~200Hz
    dt = 0.005  # 5ms
    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])
    if t_end <= t_start:
        return 0.0, 0.0, np.array([0.0]), np.array([0.0])

    t_uniform = np.arange(t_start, t_end, dt)
    s1_uniform = np.interp(t_uniform, t1, s1)
    s2_uniform = np.interp(t_uniform, t2, s2)

    # Normalize (zero mean, unit variance)
    s1_uniform = (s1_uniform - np.mean(s1_uniform))
    s1_std = np.std(s1_uniform)
    if s1_std > 0:
        s1_uniform /= s1_std

    s2_uniform = (s2_uniform - np.mean(s2_uniform))
    s2_std = np.std(s2_uniform)
    if s2_std > 0:
        s2_uniform /= s2_std

    # Cross-correlation
    max_lag_samples = int(max_lag_s / dt)
    n = len(t_uniform)
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    corr = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag >= 0:
            a = s1_uniform[lag:]
            b = s2_uniform[:n - lag]
        else:
            a = s1_uniform[:n + lag]
            b = s2_uniform[-lag:]
        if len(a) > 0:
            corr[i] = np.mean(a * b)

    lag_times = lags * dt
    best_idx = np.argmax(corr)
    best_lag = lag_times[best_idx]
    best_corr = corr[best_idx]

    return best_lag, best_corr, lag_times, corr


@click.command()
@click.argument("episode_dir", type=click.Path(exists=True))
@click.option("--max_frames", type=int, default=500,
              help="Max video frames to process (default: 500)")
@click.option("--plot", "-p", type=click.Path(), default=None,
              help="Save correlation plot to file (PNG)")
def main(episode_dir, max_frames, plot):
    """Check camera-IMU synchronization via optical flow / gyro correlation."""

    episode_dir = Path(episode_dir)
    video_path = episode_dir / "raw_video.mp4"
    imu_path = episode_dir / "imu_data.json"

    if not video_path.is_file():
        raise click.ClickException(f"No raw_video.mp4 in {episode_dir}")
    if not imu_path.is_file():
        raise click.ClickException(f"No imu_data.json in {episode_dir}")

    # Compute optical flow
    print("Computing optical flow from video...")
    flow_ts, flow_mag = compute_optical_flow_magnitude(video_path, max_frames)
    print(f"  {len(flow_mag)} frames, duration {flow_ts[-1]:.2f}s")

    # Load gyro
    print("Loading gyroscope data...")
    gyro_ts, gyro_norm = load_gyro_norm(imu_path)
    print(f"  {len(gyro_norm)} samples, duration {gyro_ts[-1]:.2f}s")

    # Cross-correlate
    print("Cross-correlating...")
    best_lag, best_corr, lag_times, corr = cross_correlate_signals(
        flow_ts, flow_mag, gyro_ts, gyro_norm,
    )

    print(f"\n{'='*50}")
    print(f"  Best lag:     {best_lag*1000:+.1f} ms (camera vs IMU)")
    print(f"  Correlation:  {best_corr:.3f}")
    print(f"{'='*50}")

    if abs(best_lag) < 0.020:
        print(f"\n  RESULT: GOOD — sync offset {best_lag*1000:+.1f}ms (< 20ms)")
    elif abs(best_lag) < 0.050:
        print(f"\n  RESULT: MARGINAL — sync offset {best_lag*1000:+.1f}ms (20-50ms)")
        print(f"  May cause SLAM degradation. Check capture timing.")
    else:
        print(f"\n  RESULT: BAD — sync offset {best_lag*1000:+.1f}ms (> 50ms)")
        print(f"  This will break visual-inertial SLAM. Fix capture synchronization.")

    if best_corr < 0.3:
        print(f"\n  WARNING: Low correlation ({best_corr:.3f}). Possible causes:")
        print(f"  - Very little motion in the video (need dynamic sequence)")
        print(f"  - IMU data is broken or from a different recording")
        print(f"  - Camera and IMU are completely desynchronized")

    # Optional plot
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(12, 8))

            # Signals
            axes[0].plot(flow_ts, flow_mag / np.max(flow_mag) if np.max(flow_mag) > 0 else flow_mag,
                        label="Optical flow (normalized)", alpha=0.8)
            axes[0].plot(gyro_ts, gyro_norm / np.max(gyro_norm) if np.max(gyro_norm) > 0 else gyro_norm,
                        label="Gyro norm (normalized)", alpha=0.8)
            axes[0].set_xlabel("Time (s)")
            axes[0].set_ylabel("Magnitude (normalized)")
            axes[0].set_title("Camera vs IMU signals")
            axes[0].legend()

            # Cross-correlation
            axes[1].plot(lag_times * 1000, corr)
            axes[1].axvline(x=best_lag * 1000, color="r", linestyle="--",
                          label=f"Best lag: {best_lag*1000:+.1f}ms")
            axes[1].axvline(x=0, color="gray", linestyle=":", alpha=0.5)
            axes[1].set_xlabel("Lag (ms)")
            axes[1].set_ylabel("Correlation")
            axes[1].set_title("Cross-correlation")
            axes[1].legend()

            # Zoomed cross-correlation
            mask = np.abs(lag_times) < 0.1
            axes[2].plot(lag_times[mask] * 1000, corr[mask])
            axes[2].axvline(x=best_lag * 1000, color="r", linestyle="--",
                          label=f"Best lag: {best_lag*1000:+.1f}ms")
            axes[2].axvline(x=0, color="gray", linestyle=":", alpha=0.5)
            axes[2].set_xlabel("Lag (ms)")
            axes[2].set_ylabel("Correlation")
            axes[2].set_title("Cross-correlation (zoomed ±100ms)")
            axes[2].legend()

            plt.tight_layout()
            plt.savefig(plot, dpi=150)
            print(f"\nPlot saved to: {plot}")
        except ImportError:
            print(f"\nWarning: matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()
