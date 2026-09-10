"""Join the chunk-relative half from two earlier runs instead of a third load.

An fp32 pi0.5 is ~19 GB resident plus a transient double while the dtype is
converted, and this machine has ~17 GB free with swap already in use, so the
chunk-relative pass kept being killed by the OOM reaper.

It does not need re-running. Two earlier runs already covered these exact
frames with this exact checkpoint:

  the phase sweep    ablation delta and per-axis split, episodes 0-5
  the prompt sweep   baseline commanded motion per frame, episodes 0-2

Joining them is EXACT, not an approximation, because Pi05Adapter.draw_noise
builds a fresh generator seeded from the adapter's seed on every call — so
every draw returns the identical tensor, within a run and across runs. Same
checkpoint, same frame, same noise, deterministic CPU inference: the baseline
chunk is bit-identical, so its magnitude transfers.

Restricted to episodes 0-2, where both runs overlap, rather than mixing
episode sets.
"""

import json
import re
from pathlib import Path

import numpy as np

TASKS = Path("/tmp/claude-1000/-home-steve-Project-Repo-GRABETTE-GRABETTE-RELEASE"
             "/616197ea-3b0d-4750-8fc3-e6b33174b8f0/tasks")
DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
PHASE_LOG = TASKS / "blmak5rdo.output"     # phase sweep, chunkrel, eps 0-5
PROMPT_LOG = TASKS / "bc4n9mzmz.output"    # prompt sweep, chunkrel, eps 0-2
EPISODES = (0, 1, 2)

PHASE_RE = re.compile(
    r"^\[\d+/\d+\] ep(\d+) f(\d+) (.+?)\s+mass ([\d.]+) delta\s+([\d.]+) "
    r"\(lat\s+([\d.]+) vert\s+([\d.]+) depth\s+([\d.]+)\)")
PROMPT_RE = re.compile(
    r"^ep(\d+) f(\d+) (.+?)\s+baseline\s+([\d.]+) mm \|")


def main() -> None:
    ablations, baselines = {}, {}
    for line in PHASE_LOG.read_text().splitlines():
        m = PHASE_RE.match(line.strip())
        if m and int(m.group(1)) in EPISODES:
            key = (int(m.group(1)), int(m.group(2)))
            ablations[key] = {
                "phase": m.group(3).strip(),
                "mass": float(m.group(4)),
                "delta_mm": float(m.group(5)),
                "axes": [float(m.group(i)) for i in (6, 7, 8)],
            }
    for line in PROMPT_LOG.read_text().splitlines():
        m = PROMPT_RE.match(line.strip())
        if m and int(m.group(1)) in EPISODES:
            baselines[(int(m.group(1)), int(m.group(2)))] = float(m.group(4))

    print(f"phase-sweep frames (eps 0-2): {len(ablations)}")
    print(f"prompt-sweep baselines:       {len(baselines)}")
    joined = sorted(set(ablations) & set(baselines))
    print(f"joined on (episode, frame):   {len(joined)}")
    missing = sorted(set(ablations) ^ set(baselines))
    if missing:
        print(f"  unmatched keys: {missing}")

    rows = {}
    for key in joined:
        a = ablations[key]
        rows.setdefault(a["phase"], []).append({
            "delta_mm": a["delta_mm"],
            "baseline_mm": baselines[key],
            "fraction": a["delta_mm"] / max(baselines[key], 1e-9),
            "mass": a["mass"],
            "axes": a["axes"],
        })

    out = DST / "repr_compare_chunkrel.json"
    out.write_text(json.dumps({"dims": 8, "rows": rows}))
    print(f"\nwrote {out}")
    for phase, rs in rows.items():
        print(f"  {phase:<14} n={len(rs)}  "
              f"baseline {np.mean([r['baseline_mm'] for r in rs]):6.1f} mm  "
              f"delta {np.mean([r['delta_mm'] for r in rs]):6.1f} mm  "
              f"= {np.mean([r['fraction'] for r in rs]):5.1%}")


if __name__ == "__main__":
    main()
