# Recording procedure for the grasp projection

**Status:** procedure, drafted 2026-07-30. Companion to
`packages/gripette/gripette/grasp_projection.py`.

## What this is for

The projection maps the gripper's two joint angles to `(strategy s, closure c)`
so a policy can command *"close all the way along this shape"* and let the object
stop the fingers, instead of regressing a precise object-dependent angle it has
no way to get right. The geometry needs no data. Two things do:

| what | why data is needed | which session |
|---|---|---|
| `p` — the full-close boundary | where the finger fouls the thumb, i.e. what "fully closed" means per shape | **A (mimic)** |
| `a`, `b` — the path exponents | whether the proximal joint leads and the distal curls late | **A (mimic)** |
| strategy *recognition* | so a policy can pick the shape from what it sees | **B (objects)** |

Session A is short and fits the mapping. Session B is longer and is only needed
once `s` should become a live policy output rather than a fixed per-task value.

**Why object recordings cannot replace session A.** With an object in the jaws
the closing motion is *truncated* at the object — the path past that point, and
the mechanical boundary itself, are never observed. Both `p` and the late-path
exponent `b` live exactly in the region the object hides. Free-air mimicry is the
only way to see them.

---

## Session A — free-air mimic (fits `p`, `a`, `b`)

No object, no arm motion needed. Hold the Grabette still; only the fingers move.
Roughly 15 minutes of recording.

### A1 · Path sweep — 5 strategies × 10 reps = 50 episodes

For each of five hand shapes, spanning your flat → rounded range:

1. Start **fully open** (both joints at zero). Hold ~0.5 s so the start is
   unambiguous.
2. Close **slowly and continuously** to the hard mechanical stop — aim for
   **2–3 seconds** of travel.
3. Hold at the stop ~0.5 s.
4. Re-open, and repeat.

The five shapes, by joint emphasis:

| # | shape | what to do |
|---|---|---|
| 1 | flat / straight finger | close with the fingertip joint kept as straight as you can |
| 2 | mostly proximal | mostly the base joint, slight tip curl |
| 3 | balanced | both joints together |
| 4 | mostly distal | mostly tip curl, base joint trailing |
| 5 | fully rounded | curl the tip as hard as possible, base joint last |

**Slowly matters.** The fit reads the *shape* of the path. At 50 fps a 0.3 s snap
close gives ~15 samples over the whole trajectory, which aliases the curvature
we're trying to measure. 2–3 s gives 100–150 samples.

### A2 · Boundary poses — 10 episodes

Short episodes, one pose each: close **as hard as the gripper physically allows**
at varied hand shapes, including *index fingertip touching the palm*. These pin
`p` directly — they are the only direct observation of the boundary.

### What A gives us

- the untruncated `(proximal, distal)` path per shape → fits `a`, `b`
- the reachable boundary → fits `p`
- a global fit, **one parameter set for all datasets**, not one per dataset

### Acceptance criteria

- Reconstruction error on **held-out** mimic paths within sensor noise.
- The single global `(p, a, b)` fits the approach phase of mustard, can and cup
  equally well. **If one dataset needs different parameters, the parametric class
  is wrong** — that is a real result and it should be reported, not fitted around.
- `p` from A2 agrees with `p` from A1's endpoints. Disagreement means the slow
  closes weren't reaching the true stop.

### Do NOT train a policy on session A

The image contains no object while the action contains a grasp. Mixed into
training, a policy learns to close on empty air and to decorrelate closing from
any visual cue. Keep it a **separate dataset**, or tag it and exclude it. It is
calibration data for a coordinate transform, not demonstrations.

---

## Session B — object recordings (teaches strategy *recognition*)

Only needed to make `s` a live policy output. Until then `s` is fixed per task
and the gripper action is a single boolean, which is also the best match for
π0.5's pretrained prior.

### B1 · Same object, two strategies — the decisive block

2 objects × 2 strategies × 20 = **80 episodes**. Grasp the *same* object two
ways, e.g. a mug wrapped around the body vs pinched at the rim.

This is the only block that can show `s` is **commandable** rather than merely a
function of the object. Name the strategy in the task string
(`"wrap the mug"` / `"pinch the mug rim"`) so it is available as a conditioning
channel.

**Interleave the two strategies, alternating rather than 20-then-20.** The
`test_grabette_blue_cylinder` set has `corr(t, episode) = -0.92` — pure session
drift. Recorded in blocks, drift is indistinguishable from strategy and the
session answers nothing.

### B2 · Strategy anchors — 3 objects × 20 = 60 episodes

Objects whose geometry makes one shape the natural one, extending the range past
the 0.11–0.44 the current data covers:

| object | approx width | shape it forces |
|---|---|---|
| wide tube (chips can, 1 L bottle) | 75–90 mm | widest opening, contact toward the finger base |
| small block (die, eraser, wood cube) | 25–35 mm | curled tip |
| marker or AA battery | 15–18 mm | fully curled fingertip |

Sim found the contact region tracks object width: ~15 mm contacts near the tip,
50 mm on the distal link, 70 mm reaches the palm side — so width is the lever
that forces a shape.

### B3 · Compliance pair — 2 × 10 = 20 episodes

A sponge and a wooden block of similar size. Checks that a *full* close plus the
torque cap holds the rigid one without crushing the soft one — the premise of
"always fully close", on the axis sim cannot answer.

---

## Rules that decide whether the data is usable

1. **Within a cell, be deliberately consistent.** Same shape, same approach
   direction, same wrist angle, same closing speed. Mustard's within-object
   strategy spread is currently sd 0.177 over a 0.00–0.65 range; shrinking that
   *is* the deliverable. A policy averages variation it cannot condition on.
2. **Be diverse only about what the camera can see** — object position and
   orientation on the table.
3. **Per-episode task strings.** Every existing dataset carries a single task
   string, so strategy cannot currently be read per episode. B1 is pointless
   without this; confirm the recorder supports it before starting.
4. **One session per block where possible**, and interleave within a block.

## Acceptance criteria for session B

Re-run the step-0 measurement:

- within-cell **sd(t) < 0.06** (the can's current best) — down from 0.177
- **between/within variance of t > 2** — up from 0.69 today

Both are measurable the same day you record, before any training.

## Known data-quality issues to avoid repeating

- **Angles beyond the servo limits.** One set reaches **101% of the proximal
  limit** — a human hand outranges the servo, and the encode has to clamp. Worth
  a calibration check before recording.
- **Frames below the open limit.** ~18% of mustard frames sit slightly *below*
  zero, suggesting the recorded zero isn't the configured zero. Re-zero the
  gripper (`scripts/calibrate_zero.py`) before a session.
- **Mixed closing sign.** Older sets close negative. The converter detects this
  per channel, but stating the convention explicitly is better than detecting it.
