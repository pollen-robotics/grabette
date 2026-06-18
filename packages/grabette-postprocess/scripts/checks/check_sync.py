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
The correlation core lives in grabette_postprocess.checks.sync (shared with the
HF Space pre-SLAM gate, which checks pair 2 only); this CLI reports both pairs.

Usage:
    uv run python scripts/checks/check_sync.py /path/to/episode
    uv run python scripts/checks/check_sync.py /path/to/episode --plot sync.png
"""

from pathlib import Path

import click
import numpy as np

from grabette_postprocess.checks.sync import (
    arducam_frame_ts,
    classify_lag,
    compute_optical_flow_magnitude,
    cross_correlate_signals,
    load_oak_gyro_norm,
    oak_left_frame_ts,
)


def _verdict(best_lag: float, best_corr: float, label: str, approx: bool) -> dict:
    """Print + return a verdict dict for one camera↔gyro correlation."""
    note = " (approx: no frame timestamps)" if approx else ""
    print(f"\n  {label}{note}")
    print(f"    best lag:    {best_lag*1000:+.1f} ms")
    print(f"    correlation: {best_corr:.3f}")
    verdict, memo = classify_lag(best_lag, best_corr)
    arrow = {"GOOD": "GOOD (< 20ms)",
             "MARGINAL": "MARGINAL (20-50ms) — may degrade SLAM / alignment",
             "BAD": "BAD (> 50ms) — breaks visual-inertial SLAM / camera-pose alignment"}[verdict]
    print(f"    → {arrow}")
    if memo and best_corr < 0.3:
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
        ("OAK left  ↔ OAK gyro", episode_dir / "oakd_left.mp4", oak_left_frame_ts(episode_dir)),
        ("Arducam   ↔ OAK gyro", episode_dir / "raw_video.mp4", arducam_frame_ts(episode_dir)),
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
