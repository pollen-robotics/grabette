#!/usr/bin/env python3
"""Convert grabette-data's camera_trajectory.csv to TUM format for evo.

CSV columns (offline rtabmap output):
    frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w

TUM trajectory format (one row per pose, space-separated):
    timestamp tx ty tz qx qy qz qw

Lost frames are skipped.

Usage:
    python scripts/csv_to_tum.py /path/to/camera_trajectory.csv -o offline.tum
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="camera_trajectory.csv input")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="TUM output path (default: <csv>.tum next to input)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[~df["is_lost"].astype(bool)]
    out = args.output or args.csv.with_suffix(".tum")
    with out.open("w") as f:
        for _, r in df.iterrows():
            f.write(f"{r.timestamp:.6f} {r.x:.6f} {r.y:.6f} {r.z:.6f} "
                    f"{r.q_x:.6f} {r.q_y:.6f} {r.q_z:.6f} {r.q_w:.6f}\n")
    print(f"Wrote {len(df)} poses to {out}")


if __name__ == "__main__":
    main()
