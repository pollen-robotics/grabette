#!/usr/bin/env python3
"""
Check temporal alignment for the OAK + Arducam rig.

Each check cross-correlates two motion signals; the lag of the correlation peak
is the timing offset. Two checks run by default, both off a single (shared)
Arducam optical-flow pass:

  1. Arducam ↔ trajectory     (image↔pose, the alignment the policy trains on)
     Arducam optical flow vs the SLAM trajectory's angular velocity. Goes through
     the actual SLAM output, so it measures the real image↔pose offset (not just
     image↔gyro). POST-SLAM: needs camera_trajectory.csv.

  2. Arducam ↔ gripper angle  (angle-stream sync)
     Arducam optical flow vs gripper joint-angle speed. Only meaningful on a
     deliberate open/close gesture filmed with the camera roughly still; a low
     correlation means no gesture / egomotion-dominated (inconclusive).

Opt-in (--vio-health), a second/slower optical-flow pass:

  3. OAK-left ↔ OAK-gyro      (SLAM VIO health)
     Same device, shared OAK clock. Confirms the visual-inertial SLAM inputs are
     time-aligned. Largely subsumed by check 1 (a bad trajectory shows up there),
     so it is a debugging aid rather than a default.

The correlation core and data loaders live in grabette_postprocess.checks.sync.

Usage:
    uv run python scripts/checks/check_sync.py -i /path/to/episode
    uv run python scripts/checks/check_sync.py -i /path/to/episode --vio-health --plot sync.png
"""

from pathlib import Path

import click
import numpy as np

from grabette_postprocess.checks.sync import (
    _arducam_flow,
    check_gripper,
    check_image_trajectory,
    check_oak_imu,
)


def _report(res: dict) -> None:
    """Print a verdict block for one check."""
    approx = " (approx: no frame timestamps)" if res["approx"] else ""
    print(f"\n  {res['label']}{approx}")
    print(f"    best lag:    {res['lag'] * 1000:+.1f} ms")
    print(f"    correlation: {res['corr']:.3f}")
    arrow = {"GOOD": "GOOD (< 20ms)",
             "MARGINAL": "MARGINAL (20-50ms)",
             "BAD": "BAD (> 50ms)"}[res["verdict"]]
    print(f"    → {arrow}" + (f" — {res['note']}" if res["note"] else ""))


@click.command()
@click.option("-i", "--input_dir", "episode_dir", required=True,
              type=click.Path(exists=True), help="Episode directory to check")
@click.option("--max_frames", type=int, default=300,
              help="Max consecutive video frames per camera (default: 300 ≈ first ~6s). "
                   "Raise it for the gripper check on long episodes so later gestures aren't cut off.")
@click.option("--vio-health", is_flag=True, default=False,
              help="Also check OAK-left↔gyro (SLAM-input health) — a second, slower optical-flow pass")
@click.option("--plot", "-p", type=click.Path(), default=None, help="Save a correlation plot (PNG)")
def main(episode_dir, max_frames, vio_health, plot):
    """Check temporal alignment via optical-flow cross-correlation."""
    episode_dir = Path(episode_dir)

    # The two checks that matter for the dataset (image↔pose, angle sync) share a
    # single Arducam optical-flow pass — the expensive Farneback part. The
    # OAK-left↔gyro SLAM-input check needs its own second flow pass and is largely
    # subsumed by image↔trajectory, so it is opt-in (--vio-health).
    print("Computing Arducam optical flow (raw_video.mp4)...")
    arducam_flow = _arducam_flow(episode_dir, max_frames)
    if arducam_flow[0] is None:
        print("  (no usable raw_video.mp4 — image↔trajectory and gripper checks skipped)")
    else:
        print(f"  {len(arducam_flow[1])} frames, {arducam_flow[0][-1] - arducam_flow[0][0]:.1f}s")

    checks = [
        ("Arducam ↔ trajectory", lambda: check_image_trajectory(episode_dir, max_frames, flow=arducam_flow)),
        ("Arducam ↔ gripper", lambda: check_gripper(episode_dir, max_frames, flow=arducam_flow)),
    ]
    if vio_health:
        print("Computing OAK-left optical flow (oakd_left.mp4)...")
        checks.append(("OAK-left ↔ OAK-gyro", lambda: check_oak_imu(episode_dir, max_frames)))

    results = []
    for name, run in checks:
        res = run()
        if res is None:
            print(f"\n  {name}: skip (missing inputs)")
            continue
        _report(res)
        results.append(res)

    print(f"\n{'=' * 60}")
    for r in results:
        print(f"  {r['verdict']:9s} {r['label']}  ({r['lag'] * 1000:+.1f}ms, corr={r['corr']:.2f})")
    print(f"{'=' * 60}")

    if plot and results:
        _save_plot(plot, results)


def _save_plot(path, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed, skipping plot")
        return

    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)
    for row, r in enumerate(results):
        def norm(x):
            m = np.max(x)
            return x / m if m > 0 else x
        ax = axes[row][0]
        ax.plot(r["cam_ts"], norm(r["cam_sig"]), label="Arducam/OAK flow", alpha=0.8)
        ax.plot(r["ref_ts"], norm(r["ref_sig"]), label=r["ref_label"], alpha=0.8)
        ax.set_title(f"{r['label']} — signals")
        ax.set_xlabel("time (s)"); ax.legend()
        ax = axes[row][1]
        ax.plot(r["lag_times"] * 1000, r["corr_curve"])
        ax.axvline(r["lag"] * 1000, color="r", ls="--", label=f"{r['lag'] * 1000:+.1f}ms")
        ax.axvline(0, color="gray", ls=":", alpha=0.5)
        ax.set_title(f"{r['label']} — cross-correlation")
        ax.set_xlabel("lag (ms)"); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nPlot saved to: {path}")


if __name__ == "__main__":
    main()
