"""Aggregate the pick-and-place run by phase, straight from the run log."""

import re
from collections import defaultdict

LOG = (
    "/tmp/claude-1000/-home-steve-Project-Repo-GRABETTE-GRABETTE-RELEASE"
    "/616197ea-3b0d-4750-8fc3-e6b33174b8f0/tasks/blmak5rdo.output"
)
ORDER = ["approach 0%", "approach 35%", "approach 70%", "GRASP",
         "carry 30%", "carry 65%", "RELEASE"]

PATTERN = re.compile(
    r"^\[\d+/\d+\] ep(\d+) f(\d+) (.+?)\s+mass ([\d.]+) delta\s+([\d.]+) "
    r"\(lat\s+([\d.]+) vert\s+([\d.]+) depth\s+([\d.]+)\)"
)


def main() -> None:
    rows = defaultdict(list)
    for line in open(LOG):
        m = PATTERN.match(line.strip())
        if m:
            phase = m.group(3).strip()
            rows[phase].append(tuple(float(m.group(i)) for i in (4, 5, 6, 7, 8)))

    print(f"{'phase':<14} {'n':>2} {'mass':>6} {'delta':>7} "
          f"{'lat':>7} {'vert':>7} {'depth':>7}   dominant axis")
    for phase in ORDER:
        vals = rows.get(phase)
        if not vals:
            continue
        n = len(vals)
        mean = [sum(v[i] for v in vals) / n for i in range(5)]
        axes = {"lat": mean[2], "vert": mean[3], "depth": mean[4]}
        dom = max(axes, key=axes.get)
        print(f"{phase:<14} {n:>2} {mean[0]:6.2f} {mean[1]:7.1f} "
              f"{mean[2]:7.1f} {mean[3]:7.1f} {mean[4]:7.1f}   {dom}")

    print("\ncamera attention mass, first phase vs last, per episode:")
    first = {int(e): None for e in range(6)}
    last = dict(first)
    for line in open(LOG):
        m = PATTERN.match(line.strip())
        if not m:
            continue
        e, phase, mass = int(m.group(1)), m.group(3).strip(), float(m.group(4))
        if phase == "approach 0%":
            first[e] = mass
        if phase == "RELEASE":
            last[e] = mass
    for e in sorted(first):
        if first[e] is not None and last[e] is not None:
            print(f"  ep{e}: {first[e]:.2f} -> {last[e]:.2f}  "
                  f"({last[e]-first[e]:+.2f})")


if __name__ == "__main__":
    main()
