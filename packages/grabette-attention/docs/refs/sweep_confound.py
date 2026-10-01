"""Is the depth trend a finding, or an artefact of there being less left to do?

The sweep showed absolute z-dependence falling from 30 mm to 6 mm as the
gripper closes in, correlating +0.894 with remaining distance. But the chunk
near the grasp contains almost no forward motion, so there is almost no forward
distance available to get wrong. A shrinking absolute error there could be pure
mechanics rather than a change in how the policy uses the camera.

The test: divide each axis's ablation delta by the motion that axis actually
has left to make. If the RATIO is flat or rising while the absolute value
falls, the dependence never weakened -- only the opportunity to err did.

Also correlates each axis separately against remaining distance, since a single
pooled correlation cannot say which axis carried it.
"""

import numpy as np

CSV = (
    "/tmp/claude-1000/-home-steve-Project-Repo-GRABETTE-GRABETTE-RELEASE"
    "/616197ea-3b0d-4750-8fc3-e6b33174b8f0/scratchpad/sweep_rows.csv"
)


def main() -> None:
    raw = np.genfromtxt(CSV, delimiter=",", names=True)
    app = raw[raw["offset"] < 0]

    print("per-axis correlation with remaining forward distance (approach frames only)")
    for name, col in (("x/lat", "x_mm"), ("y/vert", "y_mm"), ("z/depth", "z_mm")):
        r = np.corrcoef(app["remaining_m"], app[col])[0, 1]
        print(f"  {name:>8}: r = {r:+.3f}")
    print("  a trend carried by ONE axis is a claim about that axis, not about")
    print("  'camera dependence' in general\n")

    print("depth dependence RELATIVE to the forward distance still to travel")
    print(f"{'offset':>7} {'rem mm':>7} {'z mm':>7} {'z / rem':>9}")
    for off in np.unique(app["offset"]):
        m = app["offset"] == off
        rem = app["remaining_m"][m].mean() * 1000
        z = app["z_mm"][m].mean()
        ratio = f"{z/rem:8.2f}" if rem > 0.5 else "     n/a"
        print(f"{int(off):+7d} {rem:7.1f} {z:7.1f} {ratio}")
    print("  falling absolute z with a FLAT-OR-RISING ratio means the policy's")
    print("  reliance on the camera for range never weakened\n")

    print("attention mass vs causal dependence, over the whole sweep")
    r_mass = np.corrcoef(raw["mass"], raw["delta_mm"])[0, 1]
    print(f"  mass range over 80 frames: {raw['mass'].min():.3f} to {raw['mass'].max():.3f}"
          f"  (spread {raw['mass'].max()-raw['mass'].min():.3f})")
    print(f"  delta range:               {raw['delta_mm'].min():.1f} to "
          f"{raw['delta_mm'].max():.1f} mm")
    print(f"  corr(mass, delta) = {r_mass:+.3f}")
    print("  a near-constant mass across a 6x spread in causal effect is the")
    print("  'attention is not explanation' result, measured on this policy")


if __name__ == "__main__":
    main()
