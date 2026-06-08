#!/usr/bin/env python3
"""Normalize a TUM trajectory so its first timestamp is 0.

Useful when comparing two trajectories captured with different clock
references (e.g. OAK device time vs host monotonic time) but the
recordings overlap in real time. After normalization, both files start
at t=0 and `evo_ape`/`evo_rpe` can match by closest timestamp.

Usage:
    python scripts/normalize_tum.py input.tum [-o output.tum]
    python scripts/normalize_tum.py *.tum --in-place
"""

import argparse
from pathlib import Path


def normalize(in_path: Path, out_path: Path) -> tuple[float, int]:
    """Subtract first timestamp from all rows. Returns (t0, n_rows)."""
    lines = in_path.read_text().splitlines()
    if not lines:
        return 0.0, 0
    # First non-empty, non-comment line gives t0
    t0 = None
    out_lines: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            out_lines.append(line)
            continue
        parts = s.split()
        if len(parts) != 8:
            raise ValueError(f"expected 8 columns (t tx ty tz qx qy qz qw), got {len(parts)}: {s!r}")
        t = float(parts[0])
        if t0 is None:
            t0 = t
        out_lines.append(f"{t - t0:.6f} " + " ".join(parts[1:]))
    out_path.write_text("\n".join(out_lines) + "\n")
    return t0 or 0.0, sum(1 for line in out_lines if line.strip() and not line.startswith("#"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", type=Path, nargs="+", help="TUM file(s) to normalize")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output path (only when one input is given; "
                         "default: <input>.normalized.tum)")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite each input file in place (mutually exclusive with -o)")
    args = ap.parse_args()

    if args.output and len(args.paths) != 1:
        raise SystemExit("-o requires exactly one input file")
    if args.output and args.in_place:
        raise SystemExit("--in-place and -o are mutually exclusive")

    for p in args.paths:
        if args.in_place:
            out = p
        elif args.output:
            out = args.output
        else:
            out = p.with_suffix(".normalized.tum")
        t0, n = normalize(p, out)
        print(f"{p}  →  {out}  (t0={t0:.6f}, {n} rows)")


if __name__ == "__main__":
    main()
