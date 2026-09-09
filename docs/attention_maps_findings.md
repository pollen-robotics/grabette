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

### Vertical is phase-invariant and entirely vision-determined

`y/vert` sits at 14–15 mm at every offset (r = −0.084 against remaining
distance), against ~10.2 mm of demonstrated vertical travel — over 100% of
the signal. Per frame, y is the largest component in 60% of frames and the
smallest in 10%; z is the smallest in 56%.

So the camera continuously determines height, and supplies a fixed proportion
of range. Both matter; they just have different shapes.

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

With p99 clipping in place, the earlier impression that the policy "looks at
everything but the object" was substantially a display artefact: one cell at
36× the median, holding 6% of the mass, owned the whole colour ramp. Clipped,
the attention is visibly structured around the bottle's base and the near
table edge rather than its textured interior — consistent with attending to
contact geometry — plus the usual border activation. That reading is a
hypothesis; §4 is why it must stay one.

## 6. Open

- Multi-camera genericity is verified only by unit tests against synthetic
  layouts. No real multi-view data exists yet to point it at.
- Frame-edge activation is unexplained and could be a border artefact of the
  letterbox crop rather than anything about the scene.
- Whether `--denoise-step` should default to an early step.
- One dataset, one object, one checkpoint.
