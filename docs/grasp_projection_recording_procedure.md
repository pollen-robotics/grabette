# Recording for the grasp projection

**Status:** procedure, revised 2026-08-03. Companion to
`packages/gripette/gripette/grasp_projection.py`.

## What the projection does

The Gripette is not an open/close gripper: two joints, whose *ratio* is the shape
of the grasp. Asking a policy to regress those angles directly gives it a problem
it cannot solve — the demonstrated angle is where the human's fingers sat *while
pressing the object*, and a position servo replaying it stops just short and
touches nothing. Measured on the first three datasets, the demonstrated grasp uses
only **38–60% of the proximal range**.

So the two angles are re-expressed as

| | meaning |
|---|---|
| `s` strategy | the SHAPE of the grasp: 0 = pure proximal (flat finger), 1 = pure distal (curled fingertip) |
| `c` closure | how far along that shape: 0 = open, 1 = as closed as the shape can mechanically get |

and the policy commands `c = 1` — "close all the way along this shape" — letting
the **object** decide where the fingers stop, with the servo's torque cap doing
the stopping. The precise, object-dependent angle stops being something the model
has to predict.

## The key convention: rest is CLOSED

**Hold the gripper closed when idle**, not at a comfortable half-open posture.

This is not a detail; it is what makes automatic labelling possible. With rest
closed, **rest and grasp are the same command** — both are "drive to full close",
differing only in whether an object is in the way. Consequently:

- The one genuinely unresolvable ambiguity becomes harmless. Relaxing back to rest
  after a release and closing onto a wide object are *kinematically identical* —
  same direction, same shape, same duration — so no threshold, hold test or
  amplitude test can separate them. Under this convention they are also the same
  command, so confusing them costs nothing.
- Only **open** events need detecting, and that is the easy direction: opening to
  clear an object is large and deliberate.
- The gripper action becomes genuinely binary, which is what a pretrained VLA's
  gripper channel expects.
- A closed gripper has a thinner profile while approaching than a wide-open one,
  so less chance of fouling the table or the object edge.

Why the alternative was rejected: with an intermediate rest posture, approach and
grasp-holding sit at *similar angles and similar (near-zero) velocity*, so nothing
in the joint signal separates them. Measured on real data, the closure histogram
is flat — there is no sharp "rest mode" to key off. For the can, the tallest bin
landed on the *grasp* value.

Note this means `s` varies **within** an episode: your habitual closed rest
posture, then whatever the object demands. That is a feature — it gives the
strategy channel real signal, tied to something observable.

## Pilot recording

Small, sized to validate the labelling end to end rather than to train anything.

**2 objects of clearly different size, ~15 episodes each (~30 total).** One that
fits inside your resting aperture, one wider so you must open past it and close
differently — that pair is what exercises the whole strategy range.

Per episode: idle **closed** → open (only as wide as needed) → grasp → move →
release → back to **closed**.

Acceptance, checkable the same day and before any training:

- exactly one open event detected per grasp and one per release
- `s` at the grasp separates from `s` at rest by clearly more than its
  within-cell spread
- no open event detected during transport (that would be a dropped object)

## Guidance

Natural use is the goal; these only make the events cleaner.

- **Return to the same closed rest posture each time.** It is a free stylistic
  choice, and a policy averages variation it cannot condition on. It may not be
  constant across operators, so it is worth stating explicitly rather than assumed.
- **Pause briefly before closing.** Already known to help the model, and it makes
  the ramp onset crisp.
- **Close in one deliberate motion** rather than creeping in stages. Small
  hesitations are merged automatically; a slow crawl looks like drift.
- **Open only as wide as needed** — natural anyway, and it means the opening
  posture carries information about the coming grasp.
- **Don't idly adjust the fingers while transporting.** A held grasp should stay
  held; fiddling mid-transport is what creates a false release.
- **Release by opening clearly wider than the object**, so the event has real
  amplitude.

## Calibration

Position limits and the torque ceiling are **standard values** in
`gripette.config`, not per-device settings — the torque cap is the protection, so
the limits can be the real collision angles:

| | value | provenance |
|---|---|---|
| `motor1_max` | 93.5° | measured collision, moved by hand with the distal open |
| `motor2_max` | 116° | left loose on purpose; a torque-capped stall reached 102°, which is a lower bound on real travel, so tightening could reject reachable commands |
| `torque_ceiling` | 0.5 | enforced server-side on every command; an unset per-command limit resolves to it rather than to full torque |

Verified on hardware: a full close reached **92.55°**, held steady, load capped at
exactly 500/1000 during motion and relaxing to 88 once settled — it reaches the
stop without grinding into it.

Zero offsets *are* per-device (`/etc/gripette/env`, written by
`scripts/calibrate_zero_local.py`). Assume they are good to ~1°; propagated
through the encode, 1° shifts `s` by ~0.003 against a real spread of 0.11–0.44, so
it is negligible.

## What is measured, and what is not

Measured on mustard (199 eps), can (156) and cup (198):

- encode/decode round-trip is exact — 2×10⁻¹⁶ rad on real data
- the event detector finds exactly one close in **90–97%** of episodes, and the
  recovered rest → grasp levels match the independently measured grasp closures
- detected opens cluster at episode *starts* (pre-shape adjustments), not ends, so
  they are not false releases

Not established:

- these three datasets use the **intermediate-rest** convention, so they are not a
  test of the rest-closed scheme; they also never release
- multi-grasp tasks (key → unlock → open door) are out of scope here. Repeated
  grasp/release cycles with different strategies need the smart-labelling work
  (segmentation/detection models), not this heuristic
- the `p` (boundary curvature) parameter is still unfitted. It cannot come from
  the current CAD model: `proximal_bend_r` has **no collision geometry**, so the
  sim cannot see finger-to-palm contact at all. A hardware sweep found the coupled
  boundary is a box with one linear chamfer rather than a superellipse, but that
  measurement was on a prototype with a known mid-motion snag at ~71° that later
  revisions fix, so it should be redone on a current unit before being trusted
