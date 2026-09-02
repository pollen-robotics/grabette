# Chunk-relative actions: can we use LeRobot's mechanism?

**Status:** design + plan, 2026-08-26. Verified against the pinned **lerobot
0.6.0** — no version bump, no fork.

The proposal (from the [robot-folding
writeup](https://lerobot-robot-folding.hf.space)) is to stop baking per-step
deltas into the dataset and let LeRobot compute relative actions at training
time, UMI-style: every action in a chunk is an offset from the state at
prediction time.

## Verdict

| | |
|---|---|
| **the idea** | sound, and likely better than our per-step deltas — see [Why it might win](#why-it-might-win) |
| **LeRobot's built-in implementation** | **not usable for us** — 2 measured blockers below |
| **a custom `ProcessorStep`** | doable, ~150–200 lines, no fork |

## What LeRobot provides

```bash
lerobot-edit-dataset --repo_id <ds> --operation.type recompute_stats \
    --operation.relative_action true --operation.chunk_size 50
lerobot-train ... --policy.use_relative_actions=true
```

`lerobot/processor/relative_action_processor.py`
(`RelativeActionsProcessorStep` / `AbsoluteActionsProcessorStep`, mirroring
OpenPI's `DeltaActions`), on `pi0`, `pi0_fast`, `pi05`, configured by
`relative_exclude_joints`. The postprocessing step reverses the conversion at
inference. The operation is:

```python
actions[..., :dims] -= state[..., :dims]     # elementwise
```

## Why the built-in does not fit

### 1. Elementwise subtraction is not rotation composition

UMI composes: `R_rel = R_ref⁻¹ · R_i`. Subtraction is a different operation.
Measured error against the true relative rotation:

| true rotation | axis-angle (`ax,ay,az`) | 6D (`dr6d_0..5`) |
|---|---|---|
| 1° | 0.8° | 134.7° |
| 15° | 11.7° | 136.6° |
| 90° | 58.4° | 136.5° |

For 6D it is meaningless: subtracting two rotation matrices' columns does not
produce a rotation (measured column norms 1.19 / 1.68, dot +1.73 — a rotation
needs 1, 1, 0). For axis-angle the error is ~78% of the rotation, and it is
driven by *where the hand already is*, not by the size of the delta.

No representation fixes this — see the [appendix](#appendix-why-no-representation-can-work).
LeRobot is not wrong here: in **joint space** each dim is a 1-DOF rotation about
a fixed axis, which is abelian, and subtraction is exactly right. Hence
`exclude_joints`. Our action space is Cartesian pose + rotation; that is the
mismatch.

### 2. It produces world-frame offsets, and our origin is arbitrary

`p_i − p_ref` is a world-frame displacement. SLAM gives every episode an
arbitrary origin, so the same motion yields different targets. Trajectories are
gravity-aligned (`oak_slam._gravity_align_trajectory`), so roll/pitch are pinned
and the free parameters are translation plus a single **yaw**:

| re-origin | built-in `p_i − p_ref` | proposed `R_ref⁻¹(p_i − p_ref)` |
|---|---|---|
| arbitrary rotation + translation | up to **78 mm** | invariant |
| yaw 10° | **9.8 mm** | invariant (1e-13) |
| yaw 90° | **77.8 mm** | invariant |
| yaw 180° | **116.6 mm** | invariant |

This is the same lesson as before: world-frame additive deltas worked in sim only
because MuJoCo's yaw is constant.

### 3. Secondary blockers

- **Positional alignment.** `state[..., :dims]` matches dims by position, not
  name. Our `observation.state` is gripper-only, so it would subtract
  `proximal` from `x`, silently.
- **`relative_exclude_joints` defaults to `["gripper"]`**, matched by lowercase
  substring — it matches none of `proximal`/`distal`/`strategy`/`closure`, so the
  gripper would become relative. Meaningless for a projected `closure`, where
  `1.0` means "fully closed" by construction.
- **No absolute-pose RPC.** `ArmService` exposes only `SendCartesianDelta`.

### 4. And it cannot be precomputed in a dataset

Chunk-relative means action *t+i* is an offset from the pose at *t*, so the same
frame's action differs in every chunk it belongs to. A LeRobot dataset stores one
action per frame, so the representation is not expressible there. That is why
LeRobot implements it as a processor and why "just do it properly in
`convert_dataset.py`" is not an option.

## The design that does work

Compose into the reference frame instead of subtracting:

```
translation:  a_i = R_ref⁻¹ · (p_i − p_ref)
rotation:     A_i = R_ref⁻¹ · R_i            (re-encoded as 6D)
gripper:      passthrough — absolute, never relative
reference:    the chunk's first frame
```

Properties, all verified:

- **Origin-invariant.** For any rigid `(R_g, t_g)`:
  `(R_g R_ref)⁻¹ R_g (p_i − p_ref) = R_ref⁻¹ (p_i − p_ref)`. `R_g` cancels
  algebraically; confirmed to machine precision.
- **`observation.state` stays gripper-only** — no absolute pose enters the policy
  input, so the standing rule holds and blocker 3 disappears.
- **Camera-local by construction** — the side of the multiplication *is* the frame
  choice, and we take it deliberately rather than inheriting world frame.
- **Trains on the raw 8D dataset** that `generate_dataset.py` already emits
  (absolute pose in `action`). No `run_pipeline.sh` change; `convert_dataset.py`
  is not involved.
- **No proto change**: `evaluate.py` differences within the chunk to feed
  `SendCartesianDelta`, keeping the integrator and the contact guard.

### The one new risk: relocalisation jumps

A 5 cm mid-chunk pose jump:

```
per-step deltas corrupted : 1 of 5      <- despike zeroes it
chunk offsets  corrupted  : 3 of 6      <- every action after the jump
```

Per-step deltas localise a jump to a single outlier, which is exactly why the
despike strategy works in delta space. Chunk offsets spread it across the rest of
the chunk, where it looks like a plausible large motion rather than a spike. **The
step must split or reject chunks spanning a jump** — `is_lost` is in the dataset
and the >8 cm jump detection already exists for the smoothing segmentation.

## How the current pipeline handles jumps, and why that does not transfer

This is the strongest argument for the status quo that is not about effort, so it
belongs here rather than in a footnote.

Per-step deltas are defended in four layers:

| layer | where | what |
|---|---|---|
| 1. episode rejection | `clean_dataset.py` | `--max_lost_run` / `--max_lost_fraction` drop episodes with unrecoverable tracking loss |
| 2. tagging | `checks/trajectory.py`, `tags.py` | `tracking_lost`, `traj_with_jumps` in per-episode metadata, so training can filter |
| 3. **despike** | `convert_dataset.py` | per-step `|Δpos| > 80 mm` or rotation `> 5°` → that delta is **zeroed** |
| 4. smoothing segmentation | `smooth_poses_11d` | SG splits at jumps > 8 cm, so smoothing cannot smear a jump into a ramp that "sneaks under the despike cap" |

**Layer 3 is where the elegance is: in delta space the repair is itself a legal
action.** Zeroing a delta means "do not move this step" — the policy learns a
one-frame pause, which is harmless and in distribution, and the damage is
confined to exactly one action. A glitch costs one frame of supervision.

Chunk-relative has no equivalent. A jump does not corrupt one action, it corrupts
the *reference relationship* for every action after it, and no legal action
expresses "ignore the discontinuity" — the trajectory has to be reconstructed
(see `chunk_relative.splice_jumps`). So:

> The representation with 13–32x better SNR has strictly worse glitch locality.

That is a structural trade-off, not an implementation wart — **but measurement
says it is a small one.** Over `test_pick_mustard_200` (200 episodes, 39 042
frames), with the same caps the delta pipeline uses:

| | |
|---|---|
| per-step translation glitches | 5 (**0.013%** of steps) |
| per-step rotation glitches | 5 (**0.013%**) |
| K=15 chunks containing a glitch | **0.20%** |
| K=50 chunks containing a glitch | **0.59%** |

So under 1% of chunks are touched even at K=50. The repair is cheap insurance
rather than a load-bearing part of the design, and the locality argument — which I
had weighted as one of the main objections — should not carry much weight against
the SNR gain on data this clean. It would matter on a noisier capture session, so
the survey is worth re-running per dataset (`n_jumps_spliced` /
`n_rot_jumps_spliced` are in the step's report for exactly that).

Note rotation glitches occur at the same rate as translation ones, which is why
handling rotation was necessary rather than optional — but also why it is not
urgent.

What does carry over: layers 1, 2 and 4 are representation-agnostic. Episode
rejection and tagging happen upstream of any action encoding, and layer 4
disappears entirely if the better SNR removes the need to pre-smooth — which is
worth measuring, since `--smooth-poses` is still an unvalidated knob.

**One number worth respecting.** `--despike_max_deg` is 5°, and its help records
why: it used to be 45°, which "let 5–45° glitches into training; the policy
reproduced them at eval, amplified through the widened r6d normalization ranges,
and tripped the arm server's IK-jump watchdog." Rotation glitches demonstrably
occur in this data. Any chunk-space repair that handles translation only is
therefore a **regression** against the existing pipeline, not merely an
incomplete feature.

## Why it might win

Supervision SNR, and it is already measured in our own code. Per-step deltas at
50 fps are 2–3 mm against 1–2 mm of SLAM jitter — `convert_dataset.py` puts the
grasp phase at **~1.4:1**, which is what `--smooth-poses` exists to paper over. A
50-frame chunk offset is ~10 cm against the same noise. Two orders of magnitude
better conditioned, without smoothing away hand dynamics.

Note this is a *training* argument. Execution-time delta accumulation is a
separate, probably smaller effect, and is not the reason to expect a win.

## Measured so far (P1/P2 complete)

| | |
|---|---|
| supervision SNR | per-step deltas 3.83 mm vs chunk offsets 49.7 mm (K=15) / 121.8 mm (K=50), same 1–2 mm jitter |
| origin invariance | exact to machine precision under arbitrary rigid re-origin; built-in shifts 9.8 mm at 10° yaw, 117 mm at 180° |
| round trip | exact to <2 mm on real poses |
| glitch rate | 0.013% of steps; 0.20% of K=15 chunks, 0.59% of K=50 chunks |
| normalised range | [-1.27, +1.20] with relative stats; **[-20.6, +18.7]** with the dataset's absolute stats |
| step cost | 3.55 ms per batch at B=32/K=50 (0.111 ms/sample, ~9000 chunks/s) — noise against a pi05 step |

Two defects were found by writing the tests rather than by training:
`meta/stats.json` described the wrong distribution (rotation channels landed
*outside* [-1, 1]), and the chunk-start enumeration never sampled the last
`chunk_size - 1` frames of an episode — ~23% of a real episode, and the tail,
where the grasp completes. Both fixed.

## Implementation plan

All on a branch (`relative-actions`), in a worktree. Each phase has a pass
criterion; a phase that fails stops the plan rather than being worked around.

**P0 — baseline on record.** Write down the numbers the A/B is against: current
π0.5 checkpoint (`pick3_graspproj_pi05`, 2/2 real grasps), Diffusion (5/5), and
`check_trajectory` jump counts per candidate dataset — the last one predicts how
much data the jump filter will remove.
*Pass:* the comparison is quantified before anything is built.

**P1 — the step, offline, unit-tested.** `ChunkRelativeActionsStep` and its
inverse in `gripette` (lightest installable dep — it must import in the Jobs
container). Tests:
- round-trip exactness: `inverse(forward(x)) == x` on random chunks
- **origin invariance as a property test**: random rigid re-origin must not change
  the output (this is the whole design, so it gets a test)
- rotation composition matches `R_ref⁻¹ R_i`, not subtraction
- gripper channels untouched
- chunks spanning a synthetic jump are split or rejected
*Pass:* all green, plus the invariance test failing when the composition is
deliberately replaced by subtraction (proving the test has teeth).

**P2 — offline on real data.** Build the pipeline with the step inserted, run over
a real dataset. Check: round-trip on real trajectories, resulting action
distribution (should be ~cm-scale, not mm), fraction of chunks dropped by the jump
filter, normalisation stats sane.
*Pass:* round-trip exact; dropped-chunk fraction acceptable (decide the threshold
here, before it can be rationalised).

**P3 — the coupling test, cheap and early.** 100-step HF Jobs smoke. This is not
about learning: it verifies our registered step **survives `save_pretrained` /
`from_pretrained`** inside the Jobs container, which is the one failure mode that
would sink the whole approach late.
*Pass:* checkpoint reloads and the pipeline rebuilds with our step present.

**P4 — full training + offline gates.** 20k steps, then `smoke_generation.py` and
`probe_task_sensitivity.py`.
*Pass:* observation-conditioned and task-sensitive, same bar as the current
checkpoint.

**P5 — robot A/B.** Same objects, same start pose, same `--fps`/`--n_action_steps`,
one variable changed: the action representation. Steve runs it, one episode at a
time.
*Pass:* grasp success at least matching the current checkpoint. Anything less and
the SNR argument did not translate.

## Cost and what we give up

| | |
|---|---|
| code | ~150–200 lines + tests |
| compute | one HF Jobs run, ~$30 / ~12 h |
| robot | one session |
| **coupling** | the step must be importable wherever a checkpoint loads: Jobs container, Ficelle server, gates, `evaluate.py` |

That last row is the real price. Today the integrations run on **stock upstream
lerobot**, which is why the cloud recipe is one script plus a pinned version. A
custom step means a checkpoint plain lerobot cannot load — forget the import on
the Ficelle box and it fails at load time.

## Open decisions

1. **Reference = chunk's first action, or measured state?** First action is
   self-contained and keeps `observation.state` clean; measured state is what UMI
   does and matches deploy-time reality. Leaning first-action, accepting that
   training references a commanded pose and inference a measured one.
2. **Jump policy:** split chunks at a jump, or drop them? Dropping is simpler and
   loses data; splitting keeps data but yields short chunks.
3. **Scope:** π0.5 first. The step is policy-agnostic, so Diffusion/ACT could
   follow — unlike the built-in flag, which is π-family only.

## Appendix: why no representation can work

A representation `f: SO(3) → ℝⁿ` with `f(A·B) = f(A) + f(B)` maps into an abelian
group, which forces `f(A·B) = f(B·A)` for all rotations — so `f` must kill every
commutator. SO(3) is simple and non-abelian (it equals its own commutator
subgroup), so the only such `f` is `f ≡ 0`. There is no faithful additive encoding
of rotations, in any parameterisation.

Concretely: 90° about x then 90° about y gives rotvec `[69.3, 69.3, 69.3]`; the
reverse order gives `[69.3, 69.3, −69.3]` — 120° apart. But `a + b = b + a`
always, so no additive scheme can tell them apart.

Two exceptions, both narrow and both explanatory:

- **One fixed axis** is SO(2), which is abelian — subtraction is exact (measured
  error 8e-14° over 500 random coaxial pairs). This is why joint angles work.
- **Small angles**, where the error is the Lie bracket:
  `log(exp(a)exp(b)) = a + b + ½[a,b] + …`. Measured error tracks `½|a×b|` to
  three decimals (0.035° at 2°, 3.49° at 20°). It needs **both** operands small;
  in a chunk-relative setup the reference is a large arbitrary hand orientation,
  so it does not apply.

Note 6D/9D representations are chosen for *continuity* (no gimbal lock, no
antipodal ambiguity) so a network can regress them. That is a different property
from additivity, and the 6D form is simultaneously the right choice for the policy
output and the worst behaved under subtraction.

## Separately: the prediction window

- `--n_action_steps` (eval) — how much of a chunk runs before replanning.
  Currently **15**, inherited from ACT tuning. Free to change.
- `chunk_size` (training) — π0.5 trains at **50**. Changing it means retraining.

See `integrations/Pi05/README.md` for the measured trade-off.
