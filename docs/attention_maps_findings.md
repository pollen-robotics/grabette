# Attention maps and view ablation: first measurements

What the tool in `packages/grabette-attention` actually found when pointed at
`SteveNguyen/pick3_graspproj_chunkrel_pi05`, analysed on
`SteveNguyen/mustard_graspproj`. Companion to
`docs/attention_saliency_review.md`, which is the literature review and the
argument for the tool's design; this file is the data.

All runs on CPU, fp32, seed 0, `--denoise-step last` unless stated, 11 s per
analysed frame (one baseline pass plus one ablation pass).

**Standing caveat on the pairing:** the checkpoint is trained on three objects
and is analysed here against a mustard-only dataset. Defensible — mustard is
one of the three — but not an identical distribution.

## 1. The action space and its frame

The checkpoint emits 8 channels: `x, y, z, ax, ay, az, strategy, closure`
(from `datasets--SteveNguyen--pick3_graspproj_chunkrel/meta/info.json`). The
first three are translation, chunk-relative — offsets from the current pose,
not per-step deltas.

Measured over 172 usable episodes (`docs/attention_maps_refs/axis_convention.py`), splitting
each episode at first gripper closure:

| phase | dx | dy | dz |
| --- | --- | --- | --- |
| approach, 40 frames pre-grasp | −8.5 mm, 75% same sign | −10.2 mm, 81% | **+70.1 mm, 99%** |
| lift, 40 frames post-grasp | +10.9 mm, 90% | **−44.1 mm, 91%** | −2.3 mm, **51%** |

Two conclusions:

- **The frame is camera-relative, not world.** The approach is pinned to `+z`
  in 99% of episodes regardless of where the object sat. A gravity-aligned
  world frame would scatter that direction across channels as placement
  varied.
- **It is the standard OpenCV camera frame**: x lateral +right, y vertical
  **+down**, z depth **+forward** along the optical axis. During the lift `dz`
  falls to 51% sign agreement — indistinguishable from noise — while `dy`
  takes over.

This agrees independently with the calibration in the project `CLAUDE.md`:
`T_b_c1` is a 180° rotation about x, so camera y = −imu y, and imu y is up.

## 2. The millimetres are millimetres

The units fix (apply the postprocessor before scaling) was pinned only by a
unit test with a stub postprocessor. Checked against real data
(`docs/attention_maps_refs/check_units.py`), frame 192 of episode 0:

| quantity | value |
| --- | --- |
| demonstrated travel over the 50-step chunk | 21.3 mm |
| model's predicted final offset | 17.1 mm |

Same order of magnitude, ~80%. Had the postprocessor still been skipped the
prediction would have come back in normalized quantile space near 1.0 rather
than 0.017 — roughly sixty-fold off. The fix holds on real data.

The 18× ratio of median magnitudes is expected and benign: a chunk-relative
offset accumulates along the chunk while a demonstrated action is a per-step
delta.

## 3. The ablation sweep

80 frames, 10 episodes, aligned on each episode's grasp, with **remaining
forward travel to the grasp** on the x-axis — a physical distance read from
the recorded actions, so episodes performed at different speeds line up
(`docs/attention_maps_refs/ablation_sweep.py`, rows in `docs/attention_maps_refs/sweep_rows.csv`).

| offset | remaining | mass | delta | x/lat | y/vert | z/depth |
| --- | --- | --- | --- | --- | --- | --- |
| −50 | 54.5 mm | 0.30 | 38.2 mm | 12.6 | 14.9 | **30.3** |
| −40 | 31.7 mm | 0.29 | 28.1 mm | 12.5 | 14.3 | 18.0 |
| −30 | 16.5 mm | 0.29 | 21.2 mm | 11.6 | 14.3 | 8.6 |
| −20 | 6.1 mm | 0.28 | 18.7 mm | 9.1 | 14.2 | 6.3 |
| −12 | 2.2 mm | 0.28 | 19.5 mm | 8.7 | 15.0 | 7.5 |
| −6 | 1.0 mm | 0.28 | 19.8 mm | 8.1 | 15.3 | 7.8 |
| 0 | 0.0 mm | 0.28 | 22.9 mm | 12.6 | 16.1 | 8.1 |
| +8 | — | 0.29 | 23.1 mm | 7.7 | **19.1** | 8.4 |

### The camera supplies range information, proportionally and throughout

Absolute z-dependence falls 30.3 → 6.3 mm as the gripper closes in, and the
pooled correlation with remaining distance is +0.894. Read naively that says
the policy stops using the camera for range. It does not — the trend is
carried almost entirely by one axis (per-axis correlations: z **+0.928**,
x +0.532, y **−0.084**), and near the grasp there is barely any forward
distance left to get wrong.

Dividing out that opportunity (`docs/attention_maps_refs/sweep_confound.py`):

| remaining | z delta | z / remaining |
| --- | --- | --- |
| 54.5 mm | 30.3 mm | 0.56 |
| 31.7 mm | 18.0 mm | 0.57 |
| 16.5 mm | 8.6 mm | 0.52 |
| 6.1 mm | 6.3 mm | 1.04 |
| 2.2 mm | 7.5 mm | 3.39 |

Flat at ~0.55 across the genuine approach: **removing the camera changes the
forward command by a constant ~55% of the distance that remains, at any
range.** That is scale-invariant range perception, not coarse targeting. The
last two rows are an artefact of dividing by a vanishing denominator, not a
spike in dependence.

### Which axis the camera controls migrates over the approach

Within the swept window (≤54 mm remaining) `y/vert` sits at 14–15 mm at every
offset, r = −0.084 against remaining distance, against ~10.2 mm of
demonstrated vertical travel — over 100% of the signal.

**That flatness is an artefact of the window, not a property of the policy.**
A later run reaching the true start of each episode (~200 mm out, 6 episodes ×
5 timings, `docs/attention_maps_refs/episode_grid.py`) shows the dominant axis
crossing over:

| phase | depth | vertical |
| --- | --- | --- |
| start of episode, ~200 mm out | 50–84 mm | 4–12 mm |
| at the grasp | 3–12 mm | 13–24 mm |

Far from the object the camera almost entirely determines **range**; by the
grasp it almost entirely determines **height**. The sweep in the table above
never went far enough out to see it. Read the two together: §3's ~0.55
proportional range dependence is the near-field tail of a much larger
far-field range dependence.

## 4. Attention mass is not causal importance

Over the same 80 frames:

| | range |
| --- | --- |
| camera attention mass | 0.262 – 0.311 (spread 0.049) |
| ablation delta | 9.1 – 61.4 mm (6.7×) |
| correlation | +0.353 |

A nearly constant attention share across a 6.7× spread in measured causal
effect. This is the "attention is not explanation" result reproduced on our
own policy, and it is the empirical case for the tool's central design
choice: **the ablation millimetres are the evidence, the map is a
hypothesis.**

For scale, the uniform-share baseline for a 456-token prefix is 0.561 image /
0.439 language. Measured masses are ~0.29 image (0.52× its indifferent share)
and ~0.71 language (1.62×) — yet removing the image moves the trajectory by
more than the trajectory's own typical magnitude.

## 5. Sinks grow with denoising: prefer early steps for reading content

`--denoise-step all` on frame 192 (one forward pass, ten reductions) shows the
camera's mass drifting 0.33 → 0.29 from first step to last, and the peak
sharpening from 1.1× to 1.7× above the p99 clip. The bottom-left register cell
— featureless table, local pixel std 4.2 against a frame std of 63.2 — is a
faint dot at step 0 and a hard diamond at step 9.

**Practical consequence: the last step is the most sink-contaminated one, and
it is the tool's current default.** Early steps read cleaner. The default is
left at `last` pending a decision, since changing it changes every map
produced so far.

## 6. What the map is actually made of

48 frames, 8 episodes, every timing from episode start to grasp, first
denoising step. Three components, separated by three different tests.

### A dominant component fixed to the gripper

The camera rides on the gripper, so the fingers occupy the same image
coordinates in every frame of every episode while the object's position
varies. Anything fixed in image space is therefore fixed on the hardware.

Each frame's map has cosine similarity **0.928** (min 0.831) to the
across-frame mean, and the peak cell sits in one three-cell cluster
(rows 8–10, cols 11–12) in **83%** of frames. Attention is also strongly
bottom-weighted — mean share by grid row climbs monotonically from 0.25–0.45×
uniform in rows 0–5 to 1.67 / 1.85 / 2.32 / 2.48× in rows 8–11 — so roughly
the bottom third of the image, the fingers and the near table, carries most of
it.

### A real but modest component that follows the object

This one needs care to see, and two earlier attempts of mine got it wrong.
Measuring attention at the red **cap's single cell** gave 0.84× uniform and
suggested the object was below baseline — but the cap is the top of the bottle
and the warm region sits on the body. Taking the argmax of (map − template)
gave 8% hits against 5% chance — but the variance is largest at the finger
cluster, so that argmax is captured by the fingers brightening and dimming.
Both errors bias against detecting object attention.

The test that works needs no template model at all. For each frame, compare
its attention on **its own** object footprint against its attention on **other
frames'** footprints. Every fixed structure — fingers, borders, sinks —
contributes equally to both, so any gap is object-following and nothing else
(`docs/attention_maps_refs/object_and_corners.py`):

| | |
| --- | --- |
| attention on own object footprint | 0.00927 |
| attention on other frames' footprints | 0.00791 |
| ratio | **1.17×** |
| frames preferring their own footprint | **38/48 (79%)**, chance 50% |

79% of 48 against a fair-coin null is p ≈ 10⁻⁴. **Attention does follow the
object.** It is simply modest — 17% above control — so the fixed template
dominates the magnitude while the object-following part is the informative
residue. During the approach, when the object is far from the fingers, it is
visible by eye; at the grasp the two collapse onto the same pixels.

### Attention sinks, in specific low-information cells

| cell | attention | local pixel detail | variability |
| --- | --- | --- | --- |
| r10 c1 | 5.1× uniform | 11.3 (frame mean 25.8) | **0.10** |
| r11 c4 | 5.2× uniform | 7.6 | 0.19 |
| r11 c14 | 5.1× uniform | 8.7 | 0.20 |

Five times their share of attention, on patches with a third of the frame's
average detail, and the least variable cells in the whole map. High attention,
nothing to look at, unchanging regardless of scene: the register-token
signature. These are scratch space for global state, not statements about the
scene, and they are what the p99 clip in §5 exists to demote.

Not everything at an edge is a sink. The top-right corner cell (r0 c15) draws
3.7× uniform with **above**-average detail (32.9) and the second-highest
variability in the map (0.52) — it fails both sink predictions, and its
neighbour r0 c14 draws 17× less. It is a single corner cell responding to
content that genuinely changes: that corner holds cluttered background at the
extreme edge of the fisheye, where distortion is worst. Distractor response or
positional edge effect is unresolved.

### How to read a map here

The map is legible after clipping, and it does carry object information — but
the object-following signal is ~17% on top of a template three to five times
larger. §4 already showed mass is not importance. So a map is worth generating
and worth looking at, and it is not evidence on its own.

The measurement that would answer "where in the image matters" directly is the
interventional saliency the review recommended and this version deferred:
occlude a patch, re-run, measure the chunk delta — the view ablation,
spatially resolved. The machinery is already here (shared noise, the delta
metric); only the patch loop is missing.

## 7. Open

- Multi-camera genericity is verified only by unit tests against synthetic
  layouts. No real multi-view data exists yet to point it at.
- The top-right corner cell (§6): distractor response to background clutter,
  or a positional edge effect of the fisheye? Unresolved, and distinct from
  the bottom-edge sinks.
- Whether `--denoise-step` should default to an early step.
- Interventional (occlusion) saliency, to answer "where in the image matters"
  by measurement rather than by attention.
- One dataset, one object, one checkpoint.
