#!/usr/bin/env python3
"""
Compare two SLAM trajectories — with a noise floor, because this pipeline is
not deterministic.

RTAB-Map's F2M odometry is RANSAC-based, so re-running `run_oak_slam.py` on the
SAME episode with the SAME settings produces a measurably different trajectory.
Measured on OAK-D episodes: ~2.3 mm local-delta RMSE against a ~3.7 mm mean
step. Any A/B that ignores this will happily "discover" an effect that is just
the pipeline talking to itself.

So a bare A-vs-B number is not interpretable. Pass --noise with a second run of
one of the two conditions and the effect gets reported as a MULTIPLE of the
noise floor. A ratio near 1.0 means "no detectable difference", however large
the millimetre figure looks.

The metric is the camera-local translation delta, R[t]^T (p[t+1] - p[t]).
It is invariant to any rigid rotation of the world frame, so it does not move
when gravity alignment is present in one run and absent in the other — which is
exactly the situation when comparing an OAK-D (has IMU) against a Gemini 305
(has none).

Usage:
    # IMU ablation, with a same-config rerun as the noise floor
    compare_trajectories.py -a ep/camera_trajectory.csv \\
                            -b ep/camera_trajectory_noimu.csv \\
                            --noise ep/camera_trajectory_run2.csv

    # Bare comparison (prints a warning that it is uncalibrated)
    compare_trajectories.py -a oak/camera_trajectory.csv -b g305/camera_trajectory.csv
"""
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Below this multiple of the noise floor, a difference is not distinguishable
# from re-running the pipeline. 3x is deliberately conservative.
SIGNIFICANCE_RATIO = 3.0


def _local_deltas(df: pd.DataFrame) -> np.ndarray:
    """Camera-local translation deltas, R[t]^T (p[t+1] - p[t]), in metres."""
    p = df[["x", "y", "z"]].to_numpy()
    R = Rotation.from_quat(df[["q_x", "q_y", "q_z", "q_w"]].to_numpy())
    return R[:-1].inv().apply(p[1:] - p[:-1])


def _rmse_mm(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Local-delta RMSE between two trajectories of the same episode, in mm."""
    n = min(len(a), len(b))
    da, db = _local_deltas(a.iloc[:n]), _local_deltas(b.iloc[:n])
    return float(np.sqrt(((da - db) ** 2).sum(axis=1).mean()) * 1000)


def _summarise(name: str, df: pd.DataFrame) -> dict:
    d = _local_deltas(df)
    p = df[["x", "y", "z"]].to_numpy()
    return {
        "name": name,
        "frames": len(df),
        "lost": int(df["is_lost"].astype(bool).sum()) if "is_lost" in df else 0,
        "path_m": float(np.linalg.norm(p[1:] - p[:-1], axis=1).sum()),
        "step_mm": float(np.linalg.norm(d, axis=1).mean() * 1000),
    }


@click.command()
@click.option("-a", "--traj_a", required=True, type=click.Path(exists=True),
              help="Baseline trajectory CSV")
@click.option("-b", "--traj_b", required=True, type=click.Path(exists=True),
              help="Trajectory CSV to compare against the baseline")
@click.option("--noise", type=click.Path(exists=True),
              help="A SAME-CONFIG rerun of A (or B). Establishes the noise floor; "
                   "without it the comparison cannot be interpreted.")
def main(traj_a, traj_b, noise):
    """Compare two trajectories using a world-frame-invariant metric."""
    a = pd.read_csv(traj_a)
    b = pd.read_csv(traj_b)

    print(f"{'':<10}{'frames':>8}{'lost':>7}{'path m':>9}{'mean step mm':>14}")
    for label, df, path in (("A", a, traj_a), ("B", b, traj_b)):
        s = _summarise(label, df)
        print(f"{label:<10}{s['frames']:>8}{s['lost']:>7}{s['path_m']:>9.3f}"
              f"{s['step_mm']:>14.3f}")
        print(f"{'':<10}{Path(path).name}")

    effect = _rmse_mm(a, b)
    print(f"\nA vs B local-delta RMSE : {effect:.3f} mm")

    if not noise:
        print("\n  WARNING: no --noise baseline given, so this number is")
        print("  uncalibrated. This pipeline is non-deterministic; re-run one")
        print("  condition with identical settings and pass it as --noise")
        print("  before concluding anything from the value above.")
        return

    floor = _rmse_mm(a, pd.read_csv(noise))
    print(f"A vs rerun (noise floor): {floor:.3f} mm")

    if floor < 1e-9:
        print("\n  Pipeline appears deterministic — the effect above is real.")
        return

    ratio = effect / floor
    print(f"effect / noise          : {ratio:.2f}x")
    if ratio < SIGNIFICANCE_RATIO:
        print(f"\n  => NOT DISTINGUISHABLE. A differs from B by no more than "
              f"{SIGNIFICANCE_RATIO:.0f}x")
        print("     what re-running the same pipeline produces.")
    else:
        print(f"\n  => REAL DIFFERENCE: {ratio:.1f}x the noise floor.")


if __name__ == "__main__":
    main()
