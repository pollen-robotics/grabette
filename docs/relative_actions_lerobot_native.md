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
| **the idea** | **works** — grasped first try at `--n_action_steps 50` once a wire-encoding bug was fixed. n=1; see [P5 result](#p5-result-it-grasps-2026-09-03). |
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

## The P4 dataset (built 2026-08-27, local only)

There is **no raw 8D `pick3`** on the Hub — every 8D dataset is single-task — so
the A/B dataset had to be assembled from the baseline's three ancestors. Built at
`~/grabette-work/pick3_raw8d_chunkrel/merged`, id `local/pick3_raw8d_chunkrel`,
nothing existing modified:

| | |
|---|---|
| episodes / frames | **554 / 91 123** — identical to `pick3_graspproj_480` |
| per task | can 166, mustard 199, cup 189 — identical to the baseline |
| action | 8D absolute `[x,y,z,ax,ay,az,proximal,distal]` |
| observation.state | 2D gripper only, `== action[6:8]` (verified) |
| video | 1 camera, 480x360 |
| stats | relative (chunk_size 50), absolute archived under `action_absolute` |

Steps, in order: `clean_dataset.py --keep_cameras cam0` per source (the can
cleaned 200 -> **166**, matching the baseline exactly; mustard and cup `clean/`
dirs were already cached at 199 and 189) -> `resize_dataset_videos.py` to
480x360 -> `merge_datasets` -> relabel `tasks.parquet` to the language strings
-> `write_relative_action_stats(chunk_size=50)`. Deliberately **no**
`convert_dataset.py`: the delta conversion is what is being replaced.

Two things worth recording:

- The sources carry **identifier task strings** (`test_pick_can_200`,
  `Cup Grasping`), which pi05 would be conditioned on verbatim and which the
  publish gate errors on. The relabel is keyed on the source identifier and
  raises rather than guesses, because mapping by position would silently
  mislabel which object an episode belongs to.
- These older `clean` datasets predate `generate_dataset.py` emitting
  `observation.state`, so it had to be added — as the gripper pair from the same
  frame, following `convert_dataset.py`'s convention, with q01/q99 stats since
  pi05 normalises STATE with QUANTILES too.

Jump repair on the real merged data: **25 translation and 45 rotation** splices
over 2093 chunks (~1.2% / ~2.1%). Note rotation glitches outnumber translation
ones here, which retro-justifies handling them — the first version of the
splicer did translation only.

## Executing it on the robot (built 2026-09-02)

`evaluate.py --chunk_relative {auto,on,off}`. The conversion is
`ChunkRelativeDeltas` in `grabette-chunkrel`.

**The reference cancels, so the arm never learns it.** The arm service composes
its delta against the integrator target, body-locally:

```
delta_pos_world = R_target @ delta_pos      target_pos  += delta_pos_world
                                            R_target_new = R_target @ R_delta
```

So feeding it `delta_pos_i = A_{i-1}⁻¹ (a_i − a_{i-1})` and
`R_delta_i = A_{i-1}⁻¹ A_i` reproduces the chunk exactly, wherever the arm
happens to be. No absolute-pose RPC is needed and no proto changed. Verified by
replaying the server's own algebra in the tests.

**Each chunk is a fresh segment anchored at the arm's current pose.** Chunk
*n+1* has its own reference, so the executed path is not continuous with the
training trajectory across a boundary — that is the semantics, the same as any
replanning scheme, not a defect. The consequence is that the reference must be
reset at **every** replan; a missed reset differences two unrelated frames and
commands a centimetre-scale jump. `policy._action_queue` emptying is the signal.

**Relative action 0 is identically zero.** The reference is the chunk's first
action, so `a_0 = R_ref⁻¹(p_ref − p_ref) = 0` and `A_0 = I`, for every sample in
training. Three consequences:

- one of every 50 supervised actions is a constant — wasteful but harmless;
- the arm does not move on the first tick after a replan — benign;
- **`select_action()` alone cannot gate the checkpoint.** Comparing two
  episodes' first actions reads "input-independent" no matter how good the
  policy is. `smoke_generation.py --chunk_relative` therefore uses
  `predict_action_chunk` and drops action 0 before differencing.

**The inverse step is removed at inference.** `AbsoluteFromChunkRelativeStep`
rebuilds absolute poses from a reference only the robot has; it would refuse.
Eval and the offline gates strip it from the postprocessor and consume the
offsets directly.

**Refused rather than guessed.** `auto` reads the registered step names from
`policy_preprocessor.json` — exact, unlike the projection's normaliser-range
heuristic, and not fooled by lerobot's own (disabled) `relative_actions_processor`
that every existing checkpoint carries. The action width is cross-checked (8D vs
11D) so a wrong flag fails at load rather than on the arm. `--async_exec` is
refused: that path runs chunks on a background thread and drops head actions
there, and the reference tracking is not wired through it.

### Coupling, revisited

The design's cost table flagged that the step must be importable wherever a
checkpoint loads. Status:

| | |
|---|---|
| HF Jobs container | `--with grabette-chunkrel @ git+…` — working |
| eval loop / local gates | declared in `openarm-gripette-simu[eval]`; `uv sync` |
| **ficelle server** | **not done** — `probe_task_sensitivity.py` needs `--policy_addr`, so that box needs the package installed and the inverse step stripped |

To avoid blocking on the last row, `smoke_generation.py` gained `--task2`: it
re-probes one frame with a second task string and scales the result against the
scene effect, which is the task-conditioning check ficelle was needed for.

## P4 results (2026-09-03)

Trained 20k steps on `SteveNguyen/pick3_graspproj_chunkrel` (554 eps), final
loss **0.013**, 12h37 on one GPU. Checkpoint
`SteveNguyen/pick3_graspproj_chunkrel_pi05` (4.14B params, 9.35 GB).

**The coupling risk is closed.** Our step survived `save_pretrained` at
preprocessor position 3 — before the normaliser, as the training guard requires
— and `AbsoluteFromChunkRelativeStep` at postprocessor position 1.

**The SNR argument, measured in the checkpoint's own baked-in stats:**

| | position q01…q99 | rotation q01…q99 |
|---|---|---|
| chunk-relative (8D) | x ±0.10 m, y ±0.14 m, z −0.10…0.21 m | up to 0.60 rad (34°) |
| baseline (11D deltas) | x ±0.002 m, y −0.007…0.004 m, z ±0.006 m | dims 3,7 pinned 0.951→1.0 |

~244 mm of supervised span against the baseline's ~8 mm, on the same 1–2 mm of
SLAM jitter. Roughly 30× better conditioned, as predicted.

**Gate 3 — generation health: PASS.**

| | offset span | mean err | final-action err |
|---|---|---|---|
| ep0 (red can) | 89.2 mm | 7.7 mm (8.6%) | 11.9 mm |
| ep100 (red can) | 100.2 mm | 5.3 mm (5.3%) | 8.5 mm |

Over a 50-frame chunk, i.e. ~1.0 s of motion at 50 fps. Scene effect 0.0676
(bar: 0.02–0.06; higher here because chunk-relative actions are simply larger).
Predicted `action 0` pose dims came back at 1e-4…3e-3 against a structural
zero — the encode/normalise/unnormalise chain is consistent end to end. The
gripper is near-exact: closure 0.9956 vs GT 1.0 on ep0, 0.2053 vs 0.2073 on
ep100, so the projection is learned precisely.

**Gate 4 — task sensitivity: WEAK.** 0.0098 against a 0.0051 same-task
re-sampling floor = **1.93×**. The baseline pi05 measured 0.0047/0.0036 =
1.31× on the same measure, so this is ~1.5× better but still under the 3× bar.

This is a **dataset** property, not a representation one: every training scene
contains exactly one object, so the task is fully predictable from pixels and
the language channel gets no gradient. Fixing it needs multi-object scenes, not
a different action encoding. Practical consequence for P5: put one object on the
table at a time and do not try to steer by instruction.

**P4 verdict: pass.** The plan's bar was "observation-conditioned and
task-sensitive, same bar as the current checkpoint" — observation-conditioning
passes outright, task-sensitivity is better than the current checkpoint.

## P5, attempt 1: INVALID — inverted rotation on the wire (2026-09-03)

**The first three robot runs do not measure the representation.** `rotvec_to_r6d`
encoded the first two **columns** of the rotation matrix; the convention the arm
decodes with — `rotation_matrix_to_rotation_6d_numpy`, which the working
per-step-delta pipeline emits — is the first two **rows**. Columns-transposed is
`R^T = R^-1`, so **every commanded rotation delta was inverted**.

At 0.5-0.85 deg per step that is 12-21 deg of accumulated orientation error over
a 25-50 action window, and since the arm applies `target_pos += R_target @
delta_pos`, the position integration drifts with it. "Smooth but completely
misses the target" is exactly the expected symptom.

Found by comparing my conversion against `convert_dataset.compute_delta_actions`
on real trajectories — the known-good path, since it drives the policies that
work. Position agreed to 0.00000 mm; rotation disagreed at **every** step by
**exactly twice** the raw step rotation (1.6930 vs 0.847 deg, 1.5841 vs 0.792,
1.5319 vs 0.766 ...), with zero splices. A 2x angle is the signature of an
inverse: `R^-1` compared against `R` gives `R^2`. After the fix: 0.00000 mm and
0.00001 deg.

**Why the tests missed it.** The decoder in the tests was hand-written to the
same column convention as the encoder, so the two agreed with each other and
disagreed with the arm. Every round-trip and every end-to-end test passed. I had
written the decoder twice — in `test_chunk_relative_deltas.py` and in
`test_chunk_relative_eval.py` — and both copies carried the same error, so the
"end to end against the arm server's own algebra" test was checking my algebra
against my mirror of it. `test_grip_assist.py` in this repo warns about exactly
this ("an earlier mirrored-logic test grew a bug of its own and passed while the
real code was wrong"); I cited the principle and then broke it.

Now guarded by two tests that reference the real implementation rather than a
local copy: `test_the_wire_encoding_matches_the_repo_convention` compares
`rotvec_to_r6d` against `rotation_matrix_to_rotation_6d_numpy`, and
`test_deltas_match_the_known_good_delta_pipeline` compares the whole conversion
against `compute_delta_actions`. Restoring the old encoder fails 12 tests.

### What survives from attempt 1

The offline measurements do not involve the wire encoding and still stand:

- offsets are accurate (2.9 / 5.8 mm mean; 50-action endpoint within a few mm);
- grasp timing is learned (predicted close index 40 vs GT 39, 10 vs 8);
- per-step SNR from differencing is 0.98 in the grasp phase;
- endpoint error is constant with horizon, so err/net falls 0.27 -> 0.05 from
  n_exec 5 -> 50;
- `--n_action_steps 15` discards the close entirely when it lands at index 40.

What is **no longer established** is that any of this is fatal. The horizon
trade-off below was inferred from runs with inverted rotations, so the claim that
"no setting of `--n_action_steps` fixes it" is unsupported. The `n=50` run was
smooth, which is consistent with the jitter analysis being right about the
translation; whether it can grasp with correct rotations is untested.

## P5 result: it grasps (2026-09-03)

With the wire encoding fixed, **`--n_action_steps 50` grasped the mustard bottle
first try**. Same session, same object, same start pose as the baseline control
that grasped first try immediately before it.

| run | representation | `--n_action_steps` | result |
|---|---|---|---|
| 1 | chunk-relative | 15 | jitter, watchdog tripped — **invalid** (inverted rotation) |
| 2 | chunk-relative | 50 | smooth, missed — **invalid** |
| 3 | chunk-relative | 25 | missed — **invalid** |
| 4 | baseline deltas | 15 | **grasped first try** (control, today's scene) |
| 5 | chunk-relative | 50 | **grasped first try** |

So the representation works end to end: training on chunk offsets, execution by
differencing them into body-local deltas, with the grasp projection decoding the
gripper. Runs 1-3 measured a bug, not the idea.

**This is n=1 and does not yet clear the plan's P5 bar** ("grasp success at
least matching the current checkpoint"). The baseline's record is 2/2 (red can,
earlier session) plus 1/1 today. To call P5 passed the comparison needs several
episodes per object, run one at a time.

### What the offline measurements got right, and wrong

Right, and confirmed on hardware:

- the representation is learnable and the grasp timing is learned (predicted
  close at chunk index 40 vs GT 39);
- offsets are accurate (2.9 / 5.8 mm mean; 50-action endpoint within a few mm);
- the training-time conditioning win is real (244 mm of supervised span against
  the baseline's 8 mm, ~30x);
- `--n_action_steps 15` discards the close when it lands at index 40, and the
  robot showed exactly that — closure reaching ~1.0 only at chunk index 14,
  twice out of two, then reopening.

Wrong, or at least unsupported:

- **"No setting of `--n_action_steps` fixes it."** Withdrawn. It was inferred
  from runs with inverted rotations. 50 works.
- **"The two failure modes sit at opposite ends of one dial."** The long-horizon
  failure was the rotation bug. Only the short-horizon differencing jitter is
  established, and that measurement is independent of the wire encoding:
  per-step SNR 0.98 in the grasp phase, hardware lag-1 autocorrelation
  -0.10 / -0.32 / -0.28, x travelling 40 mm of path for 0.5 mm of net.
- **The whole "what would have to change" analysis.** It assumed the ceiling was
  the offset accuracy. The ceiling was a transpose.

The endpoint-error-is-constant-with-horizon measurement still stands and now has
a positive reading: err/net falls 0.27 -> 0.05 from n_exec 5 -> 50, which is why
50 is the right setting rather than an act of desperation.

### The lesson worth keeping

Three robot sessions were spent measuring a one-character convention error, and
the offline analysis built an entire coherent theory on top of it — SNR
arithmetic, a horizon trade-off, a "fundamental tension", a recommendation to
stop. All of it internally consistent, all of it downstream of a bug the tests
could not see because the tests contained the same bug twice.

The specific failure was verifying a wire format against a decoder written by
the same hand, in the same idiom, at the same time. The fix that worked was
comparing against **the code that already works in production** —
`compute_delta_actions` — which is a different kind of test entirely: not "is my
maths self-consistent" but "does my output match what the robot already accepts".
Reach for that first when a new path replaces a working one.

### What is worth keeping regardless of the representation

- `--skip_stale` read pi05's dead `_queues[ACTION]` instead of `_action_queue`,
  so it dropped chunk-head actions on *every* tick rather than once per replan.
- `smoke_generation.py` gained execution-quality diagnostics (per-axis
  straightness, lag-1 autocorrelation, per-step SNR) and a gripper-schedule
  check against `--n_action_steps`. Check 3 passed on a checkpoint that could
  not grasp, so offset accuracy alone was never a sufficient gate.
- `--task2` with a same-task re-sampling noise floor: 1.93-2.23x versus the
  baseline's 1.31x. Still WEAK — a dataset property (one object per scene).
- ficelle serves checkpoints with external processor steps, and drops
  postprocessor steps that cannot run one action at a time.
- `uv lock` was broken workspace-wide by an unguarded `lerobot==0.6.0` pin.

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
