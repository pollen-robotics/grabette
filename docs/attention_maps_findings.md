# Attention maps and view ablation: first measurements

What the tool in `packages/grabette-attention` actually found. Companion to
`docs/attention_saliency_review.md`, which is the literature review and the
argument for the tool's design; this file is the data. Every script and CSV
cited here is in `docs/attention_maps_refs/`.

Checkpoints analysed, all pi0.5 fine-tunes:

| checkpoint | actions | analysed on |
| --- | --- | --- |
| `pick3_graspproj_chunkrel_pi05` | 8-dim chunk-relative | `mustard_graspproj` |
| `sugar_cup_chunkrel_pi05_step20000` | 8-dim chunk-relative | `sugar_cup_graspproj_chunkrel` |
| `sugar_cup_grasproj_pi05` | 11-dim per-step deltas | same recording |
| `sugarcube_in_mug_chunkrel_pi05_step20000` | 8-dim chunk-relative | same recording, retrained prompt |

All runs on CPU, fp32, seed 0. About 11 s per analysed frame with ablation,
5.5 s without; an occlusion sweep is 193 passes per frame, ~15 min.

**Standing caveat on the first pairing:** `pick3` is trained on three objects
and analysed against a mustard-only dataset. Defensible — mustard is one of
the three — but not an identical distribution. The sugar work has no such
gap: checkpoint and data are the same recording.

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

The measurement that answers "where in the image matters" directly is the
interventional saliency the review recommended — occlude a patch, re-run,
measure the chunk delta, i.e. the view ablation spatially resolved. It was
deferred when this section was written and has since been built: see §7 for
what it found, and §10 for the object-level reading that corrects the
"attention ignores the object" wording above.

## 7. Occlusion: where the motion comes from, and it is not where attention goes

Covering one 30×30-pixel block of the image and re-running gives a causal map
on exactly the grid the attention map uses, so the two are comparable cell for
cell (`docs/attention_maps_refs/occlusion_run.py`,
`attn_vs_causal.py`). Sugar-cube task, episode 0, 12×16 grid, mean fill,
193 forward passes per frame.

| statistic | approach frame | grasp frame |
| --- | --- | --- |
| Pearson corr(attention, causal) | +0.197 | +0.040 |
| Spearman | +0.387 | +0.181 |
| top-5 cells in common | **0 / 5** | **0 / 5** |
| top-10 in common | **0 / 10** | **0 / 10** |
| top-20 in common (chance ≈ 2.1) | 3 / 20 | 2 / 20 |

Not one cell shared in the top ten, at either phase. This is not one map being
vague: both concentrate ~14% of their mass in five cells. They are equally
structured and point elsewhere.

**No single patch is critical either.** The largest effect any block has is
3.4 mm at approach (5.6% of that frame's 61 mm RMS baseline) and 6.3 mm at the
grasp (6.7% of 94 mm), while removing the whole camera moves the trajectory
65–96 mm. The visual information is redundant, spread across many patches.

## 8. The prompt is a lexical key, and it collided with a sibling task

Language plus state carries roughly **0.7 of the prefix attention mass**, about
twice the camera's, and had never been intervened on. Same method as the view
ablation: substitute the prompt, re-run with the same noise, measure the change
in commanded translation (`docs/attention_maps_refs/prompt_run.py`). 21 frames
over 3 episodes, `sugar_cup_chunkrel_pi05_step20000`, mean baseline motion
100.5 mm.

| prompt substituted | mean | at grasp | worst phase |
| --- | --- | --- | --- |
| its own prompt (control) | **0.0** | 0.0 | 0.0 |
| `"pick up the cup"` | **5.9** | 4.3 | 8.6 |
| its own words, scrambled | 6.7 | 5.9 | 8.1 |
| `"pick up the mustard bottle"` | 9.8 | 7.2 | 11.4 |
| `"pick up the red can"` | 10.5 | 7.9 | 15.0 |
| `"put the sugar cube in the mug"` | 14.8 | 9.7 | 17.3 |
| `""` (empty) | 17.1 | 9.5 | 26.7 |
| `"close the drawer"` | **34.8** | 13.5 | **62.0** |

Three findings:

- **The prompt is load-bearing**, at roughly a third of the camera's weight
  (34.8 mm mean against whole-camera ablations of 80–155 mm at the same
  phases).
- **The conditioning is lexical, not semantic.** A correct description of the
  actual task sat 2.5× further from the learned behaviour than a semantically
  wrong string sharing vocabulary, and scrambling word order cost almost
  nothing.
- **The collision.** pick3's `"pick up the cup"` — its instruction for a paper
  cup — perturbed this model *less than shuffling its own prompt into
  nonsense*. Against pick3's other two strings it was 0.58× their mean: among
  three siblings, that one sat distinctly closer to the sugar model's key.
  Merging the datasets would have left a single-object pick and a two-stage
  place indistinguishable by prompt.

Language also matters least exactly where the camera matters most: every
variant dips at the grasp and peaks mid-approach. The prompt selects *what* to
do and is spent by contact; vision makes the contact.

### The retrain closed it

The model was retrained on `"put the sugar cube in the mug"` — which describes
the real task and shares only "the" with any pick3 string — and the sweep
repeated on `sugarcube_in_mug_chunkrel_pi05_step20000`
(`docs/attention_maps_refs/prompt_run_retrained.py`):

| prompt substituted | original | retrained |
| --- | --- | --- |
| `"pick up the cup"` | **5.9** | 15.2 |
| `"pick up the mustard bottle"` | 9.8 | 15.7 |
| `"pick up the red can"` | 10.5 | 18.4 |
| **collision ÷ sibling mean** | **0.58×** | **0.89×** |

"cup" is now indistinguishable from its siblings — the hazard is gone. And the
lexical-keying claim survived a test it was not derived from: scrambling the
NEW model's own words costs 7.9 mm, still less than any real alternative
including its predecessor's prompt (12.2 mm).

One change remains unexplained. Every delta rose about a third on the retrained
model (unrelated 34.8 → 46.4 mm) at essentially identical baseline motion
(100.5 → 100.7). Better grounding from a correct prompt is the flattering
reading; the new prompt also being longer, seven words against five, is the
deflationary one. Confounded; a length-matched nonsense prompt would separate
them and has not been run.

## 9. Delta vs chunk-relative: the representation changes attention, not dependence

Same recording, same 20 000 steps, same prompt; only the action
parameterisation differs (`docs/attention_maps_refs/delta_vs_chunkrel.py`,
`endpoint_run.py`).

**These cannot be compared in millimetres.** Read from the checkpoints' own
processor configs: the delta model's `relative_actions` and `absolute_actions`
steps are both `enabled: false`, so its chunk is raw per-step motion, mean
magnitude 6.3 mm. The chunk-relative model emits offsets accumulated over 50
steps, mean 100.5 mm. The same intervention reads ~20× larger on the latter for
reasons that have nothing to do with the policy.

Two normalisations were tried and **both were wrong**:

- *Fraction of own commanded motion* — gave a 7× difference in local
  sensitivity.
- *Raw millimetres* — reversed the ranking, because the delta baseline is small.

**Endpoint divergence embeds no such choice.** Compose each chunk into a
trajectory (chaining the delta model's per-step rotations) and measure how far
apart the endpoints land. Both models then command a trajectory of the same
length, which is what makes the columns comparable:

| frame | model | travel | worst block | median block |
| --- | --- | --- | --- | --- |
| approach | delta | 124.9 mm | **12.9 mm** | 1.43 mm |
| approach | chunk-relative | 127.3 mm | 5.2 mm | 0.52 mm |
| grasp | delta | 132.2 mm | 6.9 mm | 1.45 mm |
| grasp | chunk-relative | 150.5 mm | **8.4 mm** | 0.68 mm |

The real gap is **2.5×** on the worst block and 2.75× on the median, confined
to the approach; at the grasp it vanishes and chunk-relative is slightly more
sensitive. What carries the difference is the median rather than the peak: the
delta model is uniformly more sensitive across the image, not dependent on one
spot. Peak *location* is metric-dependent and is not claimed.

Independent check on the composition: the delta model's approach trajectory
travels 124.9 mm against 121.4 mm of forward travel the demonstration actually
had left to the grasp — within 3%.

**Vision dependence itself is indistinguishable** between the two (every phase
within ±9%, both near 100% of own commanded motion), but that measure saturates
there, so it is evidence of no *detectable* difference rather than none.

**Attention, however, differs by 30%** — camera mass 0.453 for delta against
0.349 for chunk-relative, at 0.99× the causal dependence. A third independent
demonstration that mass is not importance, this time across two models rather
than across space or frames.

The depth→alignment→depth migration of §3 reproduces in both, so it is the
task's geometry rather than an artefact of chunk-relative actions.

## 10. The cube, localised — and an argmax mistake

Colour could not find the sugar cube: four masks either claimed a third of the
frame or latched onto a shadow. Two things fixed it
(`docs/attention_maps_refs/zoom_frame.py`, `detect_cube.py`,
`cube_vs_rest.py`). A grid-labelled zoom read by eye put the cube at
`x 248–286, y 224–248` — cells r7c8, r7c9, r8c8, r8c9. Then **Grounding DINO
for the box plus SAM2 for the mask**, run locally, reproduced exactly those
four cells at 0.73 confidence and 4.7 px of centre error. The division of
labour matters: SAM2 is promptable and cannot find an object from a
description, so text→box has to come from elsewhere.

Mean map value on the segmented object against every cell in neither object,
approach frame:

| map | model | cube / elsewhere | mug / elsewhere |
| --- | --- | --- | --- |
| attention | delta | **3.13×** | 0.51× |
| attention | chunk-relative | **2.32×** | 0.77× |
| causal, RMS | delta | **11.81×** | 2.03× |
| causal, RMS | chunk-relative | 6.74× | 2.31× |
| causal, endpoint | delta | 4.94× | 3.02× |
| causal, endpoint | chunk-relative | 5.40× | 2.07× |

Both models attend the cube at two to three times the average cell and
under-attend the mug; the cube is also the dominant causal region, 5–12×. At
the grasp everything flattens (cube 1.6–2.5×, mug comparable or higher),
matching a decision about closing rather than locating.

**This corrected an earlier conclusion in this document's own history.** A
first reading held that neither model's attention was on the cube — judged by
the **argmax**, and the register cells take attention's top slots, so the cube
never enters a top-ten list. Among actual scene content it is the most attended
region. Scoring a region by whether it wins the peak was the mistake.

So §6's dissociation needs stating at the right level. Excluding the register
border roughly doubles the delta model's attention-vs-causal correlation on
approach (+0.176 → +0.354) and barely moves chunk-relative's (+0.197 → +0.207);
at the grasp both stay near zero (`mask_registers.py`). **At the region level
the two maps agree the cube matters; at the cell level across the whole map
they still barely correlate.** An attention map is not a saliency map — but
"attention ignores the object" is withdrawn.

## 11. Claims this investigation retracted

Reading these maps by eye produced repeated confident errors. Recorded because
the failure mode is systematic, not careless: every wrong conclusion came from
reading a spatial pattern and inferring a mechanism, and every correction came
from an intervention or a control.

| claimed | measurement said |
| --- | --- |
| Attention is structured around the object's base and the near table edge | That was the fixed gripper template, cosine 0.928 to every frame's map |
| The attention map is not a localiser; the object draws 0.84× uniform | That probe measured one cell on the bottle's *cap* while the warm region sat on its *body*. §10 settled it: 2.3–3.1× |
| Vertical camera dependence is phase-invariant | True only inside the swept window; at ~200 mm out depth dominates and the two cross over |
| Depth is barely camera-dependent | An artefact of sampling only near-grasp frames. Normalised by distance remaining, it is flat at ~0.55 |
| The attended "finger band" loses share through the task | The band was the bottom four rows while the hot region spans rows 6–9 — it measured where the line was drawn |
| The smaller sugar object costs more vision | The object is a mug, and that run used the wrong checkpoint on the wrong task entirely |
| A single patch matters 7× more to the delta model | A normalisation artefact; endpoint divergence gives 2.5×, approach only |
| A smudged or taped lens would hurt the delta model badly | Wrong twice: single-block occlusion says nothing about many blocks, and a lens obstruction blocks the entrance pupil — dimming and defocusing the whole frame, not blacking out a region |

## 12. Method limits

- **Nothing here is validated against behaviour.** Every figure measures how
  far the commanded output moves, never whether the policy fails more often.
  The experiment that would settle it applies the same mask *in software* on
  the policy's input during a closed loop and compares success rates — not
  obstructing the optics, which produces global blur rather than the regional
  occlusion measured here. Until that is run, every number is a proxy.
- **The occlusion fill is off-distribution.** A covered block must be painted
  with something and the policy has never seen a grey rectangle mid-scene.
  Comparisons *between* cells are sound; an absolute cell value is not a
  prediction about a genuinely empty region.
- **Occlusion cost.** 193 forward passes per frame at 12×16, ~15 min on CPU.
  Two frames of one episode per model; the disjointness and concentration
  results rest on those.
- **Single view.** Multi-camera support is implemented and unit-tested against
  synthetic layouts, but no real multi-view data exists yet to exercise it.
  Note that `sugar_cup_graspproj_chunkrel`'s episode metadata carries
  `observation.images.right_cam1` columns while `info.json` lists only `cam0` —
  a second view may be recoverable from the source recording.
- **The gripper is still unsegmented.** Motion-based attempts failed twice
  (image motion scales with proximity, so distant background looks as static as
  the jaws) and Grounding DINO matched a green jug's handles. The jaws are
  fixed in image space, so one hand-drawn polygon would serve every frame.
- **Attention reduction choices.** Mean over heads, over action-token queries,
  then over layers. Other reductions are defensible and were not compared.

## 13. Open

- Does any of this predict behaviour? (see §12 — the one falsifying test)
- The top-right corner cell (§6): distractor response to background clutter, or
  a positional edge effect of the fisheye? Distinct from the bottom-edge sinks.
- Whether `--denoise-step` should default to an early step.
- Whether the retrained model's 33% higher prompt sensitivity is grounding or
  prompt length.
- The finger-versus-cube question: consistent with the axis evidence at the
  grasp (lateral and vertical dominant, depth at its minimum) and unproven.
