"""Aggregate the ablation sweep into the shape that answers the question.

Two readings, both needed:

  BY OFFSET  -- camera dependence as a function of how far the gripper still
                has to travel. Rising means visual servoing; flat means the
                camera sets a coarse target and proprioception does the rest.
  BY AXIS    -- whether the dependence is concentrated in the axis that would
                require judging RANGE (z, forward) or in the axes that only
                require judging ALIGNMENT (x lateral, y vertical).

Reports each axis both in absolute millimetres and relative to how far the
demonstrations actually move on that axis, because 4 mm on an axis that
travels 70 mm and 13 mm on an axis that travels 10 mm are opposite findings.
"""

import numpy as np

CSV = (
    "/tmp/claude-1000/-home-steve-Project-Repo-GRABETTE-GRABETTE-RELEASE"
    "/616197ea-3b0d-4750-8fc3-e6b33174b8f0/scratchpad/sweep_rows.csv"
)
# Demonstrated cumulative travel over the 40 frames before the grasp, from the
# 172-episode axis-convention measurement. Used only to put each axis's
# ablation delta on the scale of that axis's own motion.
DEMO_TRAVEL_MM = {"x/lat": 8.5, "y/vert": 10.2, "z/depth": 70.1}
AXES = ("x/lat", "y/vert", "z/depth")


def main() -> None:
    raw = np.genfromtxt(CSV, delimiter=",", names=True)
    print(f"{len(raw)} analysed frames, {len(np.unique(raw['episode']))} episodes\n")

    print("BY OFFSET FROM GRASP (mean over episodes; rem = forward travel still to go)")
    print(f"{'offset':>7} {'rem mm':>7} {'mass':>6} {'delta mm':>9} "
          f"{'x/lat':>7} {'y/vert':>7} {'z/depth':>8} {'n':>3}")
    offsets = np.unique(raw["offset"])
    for off in offsets:
        m = raw["offset"] == off
        print(
            f"{int(off):+7d} {raw['remaining_m'][m].mean()*1000:7.1f} "
            f"{raw['mass'][m].mean():6.2f} {raw['delta_mm'][m].mean():9.1f} "
            f"{raw['x_mm'][m].mean():7.1f} {raw['y_mm'][m].mean():7.1f} "
            f"{raw['z_mm'][m].mean():8.1f} {int(m.sum()):3d}"
        )

    # Does dependence track remaining distance? Use only the approach frames:
    # after the grasp the object is held, so "distance to go" is meaningless.
    app = raw[raw["offset"] < 0]
    r = np.corrcoef(app["remaining_m"], app["delta_mm"])[0, 1]
    print(f"\napproach frames only (n={len(app)}):")
    print(f"  corr(remaining distance, ablation delta) = {r:+.3f}")
    print("  negative => dependence GROWS as the object gets closer (servoing)")
    print("  ~zero    => dependence is independent of range (coarse targeting)")

    far = app[app["remaining_m"] > np.median(app["remaining_m"])]
    near = app[app["remaining_m"] <= np.median(app["remaining_m"])]
    print(f"  far  half: delta {far['delta_mm'].mean():5.1f} mm  mass {far['mass'].mean():.2f}")
    print(f"  near half: delta {near['delta_mm'].mean():5.1f} mm  mass {near['mass'].mean():.2f}")

    print("\nBY AXIS, over all analysed frames:")
    print(f"{'axis':>8} {'mean mm':>8} {'sd':>6} {'max':>6} "
          f"{'demo travel':>12} {'delta/travel':>13}")
    for name, col in zip(AXES, ("x_mm", "y_mm", "z_mm")):
        v = raw[col]
        travel = DEMO_TRAVEL_MM[name]
        print(
            f"{name:>8} {v.mean():8.1f} {v.std():6.1f} {v.max():6.1f} "
            f"{travel:12.1f} {v.mean()/travel:12.0%}"
        )

    # How often is depth the least affected axis? A per-frame count is stronger
    # than a comparison of means, which one outlier frame could carry.
    stack = np.stack([raw["x_mm"], raw["y_mm"], raw["z_mm"]], axis=1)
    smallest = np.argmin(stack, axis=1)
    largest = np.argmax(stack, axis=1)
    print("\nper-frame ranking (how consistent is the pattern?):")
    for i, name in enumerate(AXES):
        print(
            f"  {name:>8}: smallest in {(smallest==i).mean():5.0%} of frames,"
            f"  largest in {(largest==i).mean():5.0%}"
        )


if __name__ == "__main__":
    main()
