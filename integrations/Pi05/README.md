# π0.5 for Grabette

Fine-tune [π0.5](https://www.physicalintelligence.company/blog/pi05)
(`lerobot/pi05_base`, a ~2.3B flow-matching vision-language-action model) on
Grabette demonstrations, gate it offline, and run it on the robot through a
remote GPU. This is the **verified VLA recipe** for Grabette data: the
reference run (554 episodes, 3 tasks) grasped on its first real-arm episodes,
with the grasp trigger firing from raw policy output — none of the
inference-side aids the Diffusion baseline needs.

Companion to [`../DiffusionPolicy`](../DiffusionPolicy) — the **dataset
preparation is shared** (same converted 11D camera-local-delta datasets, 2D
gripper state, per-episode task strings). This integration adds training,
gates, and remote deployment.

> **Why π0.5 and not Pi0-FAST?** We tried Pi0-FAST first: its autoregressive
> action-token head **collapsed to a constant, input-independent action** at
> our data scale while its training loss looked perfectly healthy ($105 to
> learn that teacher-forced loss cannot see this failure). π0.5's continuous
> flow head was observation-conditioned by the first checkpoint (step 4000)
> on the same data. The pi0fast material is kept in [`pi0fast/`](pi0fast/)
> for reference; use π0.5.

## Requirements

- **Dataset**: produced by the shared pipeline
  (`../DiffusionPolicy/run_pipeline.sh`): 11D Cartesian deltas + 2D gripper
  state + natural-language task strings, 480×360 wrist camera.
- **Training GPU**: A100-80GB class for batch 32 (bf16 + gradient
  checkpointing). An HF Jobs `a100-large` run costs ~$30 and ~12 h.
- **Inference GPU**: ~10 GB in fp32 (RTX 3090/4090/5090) — either on the
  robot machine itself, or anywhere else via
  [Ficelle](https://github.com/SteveNguyen/Ficelle) remote serving (in which
  case the robot machine needs no GPU at all).
- **One lerobot revision everywhere**: training, gates, and the Ficelle
  server must share the rev pinned in `pyproject.toml`. Bump deliberately.

## Workflow — cheapest test first, robot last

```
0. smoke_pi05_reference.py      free      is the pi05 port itself healthy?
1. 100-step training smoke      ~$2       does training run + checkpoint persist?
2. full training                ~$30      20k steps, inline eval split
3. smoke_generation.py          free      is the fine-tune observation-conditioned?
4. probe_task_sensitivity.py    free      does it read the task string? (multi-task only)
5. robot session                robot     evaluate.py — local GPU or Ficelle remote
```

### 0. Port smoke (before spending anything)

Verifies that the pinned lerobot revision generates sane, input-dependent
actions with lerobot's own known-good libero π0.5 checkpoint (~7 GB GPU):

```bash
uv run python smoke_pi05_reference.py
```

### 1–2. Training

No action-tokenizer stage: π0.5 is flow-matching, so there is nothing to fit
or verify before training (the FAST tokenizer workflow applies only to
Pi0-FAST — see [`pi0fast/README.md`](pi0fast/README.md)). Dataset in,
training out.

The flags are identical locally and in the cloud; only the launcher wrapper
differs.

**Local GPU** (A100-80GB class for batch 32; a 24–32 GB card needs a much
smaller batch and proportionally more steps — untested by us):

```bash
uv run python train.py \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.empty_cameras=0 \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 \
  --policy.compile_model=false \
  --policy.scheduler_warmup_steps=4000 \
  --policy.scheduler_decay_steps=100000 \
  --policy.scheduler_decay_lr=1e-5 \
  --dataset.repo_id=<user>/<dataset>_cartesian \
  --dataset.eval_split=0.05 --eval_steps=1000 \
  --steps=20000 --batch_size=32 --num_workers=4 \
  --output_dir=outputs/<task>_pi05 \
  --policy.push_to_hub=true --policy.repo_id=<user>/<task>_pi05
```

**HF Jobs (cloud)** — same account/token/credit prerequisites as the
DiffusionPolicy integration (see its README's cloud section). Two cloud
specifics: `--dataset.video_backend=pyav` (the Jobs image has no system
FFmpeg) and, if you want the intermediate checkpoints (you do — the
mid-training gates read them), mount a storage bucket and point
`--output_dir` INTO it. A relative or container-local output_dir **silently
loses every checkpoint** when the container exits; only the final Hub push
would survive.

```bash
# $2 smoke first: verify training runs AND the checkpoint lands in the bucket
hf jobs uv run --flavor a100-large --timeout 1h -s HF_TOKEN \
    -v hf://buckets/<namespace>/<bucket>:/data \
    train.py -- \
    --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
    --policy.empty_cameras=0 \
    --policy.gradient_checkpointing=true --policy.dtype=bfloat16 \
    --policy.compile_model=false \
    --dataset.repo_id=<user>/<dataset>_cartesian \
    --dataset.video_backend=pyav \
    --steps=100 --batch_size=32 --num_workers=4 \
    --save_freq=100 --output_dir=/data/outputs/<task>_pi05_smoke
# then check: hf jobs logs, and that the bucket contains
# outputs/<task>_pi05_smoke/checkpoints/000100/pretrained_model/

# real run (~12 h on a100-large ≈ $30)
hf jobs uv run --flavor a100-large --timeout 24h -s HF_TOKEN \
    -v hf://buckets/<namespace>/<bucket>:/data \
    train.py -- \
    --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
    --policy.empty_cameras=0 \
    --policy.gradient_checkpointing=true --policy.dtype=bfloat16 \
    --policy.compile_model=false \
    --policy.scheduler_warmup_steps=4000 \
    --policy.scheduler_decay_steps=100000 \
    --policy.scheduler_decay_lr=1e-5 \
    --dataset.repo_id=<user>/<dataset>_cartesian \
    --dataset.video_backend=pyav \
    --dataset.eval_split=0.05 --eval_steps=1000 \
    --steps=20000 --batch_size=32 --num_workers=4 \
    --output_dir=/data/outputs/<task>_pi05 \
    --policy.push_to_hub=true --policy.repo_id=<user>/<task>_pi05
```

Recipe rationale (matched to the verified `lerobot/pi05-libero` fine-tune):

| Ingredient | Value | Note |
|---|---|---|
| chunk_size / n_action_steps | 50 / 50 (defaults) | π0.5's native flow horizon — don't shrink at training; the eval replans on a prefix instead |
| cameras | your 1 real camera, `empty_cameras=0` | do NOT zero-pad camera slots (prime suspect in the pi0fast collapse) |
| LR schedule | 2.5e-5, warmup 4000, decay horizon 100k → ≈constant over 20k | **the three scheduler flags are mandatory**: lerobot's auto-scaled default decays to 2.5e-6 by 20k, 10× below recipe |
| eval split | `--dataset.eval_split=0.05 --eval_steps=1000` | held-out CE loss each 1000 steps; expect it descending (reference: 0.755 → 0.447), no overfit gap |
| compile | **off** | `compile_model=true` + inline eval crashes (inductor layout conflict → illegal memory access at the first eval step) |
| image transforms | off | |
| action tokenizer | none | flow matching — no FAST stage, nothing to fit or verify |

`train.py` is stock `lerobot-train` with one surgical fix (see its
docstring): it rebuilds the processing pipeline fresh from your policy config
and dataset stats instead of deserializing the base checkpoint's.

### 3. Generation gate (before ANY robot time)

Training loss — even a clean held-out eval loss — **cannot detect a policy
that ignores its observations** (measured the hard way on pi0fast). Gate the
fine-tune on real dataset frames, at least two episodes per task:

```bash
uv run python smoke_generation.py \
    --checkpoint <user>/<task>_pi05 --policy_type pi05 --fp32 \
    --dataset_repo_id <user>/<dataset>_cartesian \
    --episodes 0 80 --frame 60 --task "<your task string>"
```

PASS = finite, sane-scale chunks that **differ across observations** (mean
|diff| ~0.02–0.06 on our data) and roughly track each frame's ground truth.
The collapsed pi0fast reference measured 0.000000. `--fp32` matters: the
pi05 port has a bf16 dtype clash in its flow path — fp32 for all inference.

### 4. Language gate (only if you rely on task strings)

A multi-task fine-tune where **every training scene contains exactly one
object teaches the model to ignore the instruction** — the task is 100%
predictable from pixels, so the language channel gets no gradient. Measured
on our 3-task model: swapping the task string moved actions by 0.0047 vs a
0.0036 same-task sampling-noise floor (i.e. nothing). It will grab its
favorite object regardless of what you ask.

```bash
# needs a Ficelle server running (step 5). All-local setup: start one on the
# same machine — uv run python serve.py --checkpoint <user>/<task>_pi05 \
#   --dtype float32     (websocket on :8000) — and pass localhost:8000.
uv run python probe_task_sensitivity.py \
    --policy_addr <ticket-or-host:port> \
    --dataset_repo_id <user>/<dataset>_cartesian \
    --episodes 80 300 --frame 60 \
    --tasks "pick up the red can" "pick up the mustard bottle"
```

If it fails and you need instruction-following, the fix is data: episodes
with **multiple objects in the scene** where the commanded one is grasped.

### 5. Run on the robot (local GPU or remote server)

Both modes use the same eval loop
([`openarm_gripette_simu/examples/evaluate.py`](../openarm/openarm_gripette_simu/README.md)
— see its README for the full flag reference and the start-pose calibration
rules, which apply to π0.5 exactly as to Diffusion). Pick by hardware:

**A. Robot machine has a ~10 GB GPU** — load the checkpoint locally, no
server needed:

```bash
uv run python examples/evaluate.py \
    --checkpoint <user>/<task>_pi05 \
    --task "<training task string>" \
    --n_action_steps 15 \
    --home_joints <calibrated start pose> --start_gripper <demo first-frame> \
    --num_episodes 10 --ask_success session.jsonl --dump_obs /tmp/dump_pi05
```

**B. GPU is elsewhere** — serve with
[Ficelle](https://github.com/SteveNguyen/Ficelle) and point the eval at it.
On the GPU machine (clone Ficelle; `--transport iroh` works through NAT with
no VPN and prints a connection ticket; on a LAN/VPN you can use the default
websocket transport and a plain `host:port` instead):

```bash
uv run python serve.py --checkpoint <user>/<task>_pi05 \
    --dtype float32 --transport iroh
#   -> iroh ticket: endpointv1...
```

On the robot machine — same command as A with `--policy_addr` replacing
`--checkpoint` (needs the ficelle client:
`uv pip install -e '<ficelle>/client[iroh]'`):

```bash
uv run python examples/evaluate.py \
    --policy_addr endpointv1... --jpeg_quality 90 \
    --task "<training task string>" \
    --n_action_steps 15 \
    --home_joints <calibrated start pose> --start_gripper <demo first-frame> \
    --num_episodes 10 --ask_success session.jsonl --dump_obs /tmp/dump_pi05
```

Deployment settings that matter (each traced to a measured failure):

- `--n_action_steps 15` (both modes) — replan cadence over the native
  50-chunk: long enough to amortize inference, short enough to stay
  closed-loop.
- `--grip_gain 1.3` (both modes) — if grasps slip: demo closes are recorded
  on the Grabette trigger linkage; a position-controlled servo chasing the
  same numbers squeezes less. Scales close depth around `--start_gripper`.
- The task string must be **exactly** a training task string.
- `--jpeg_quality 90` (remote only) — raw 480×360 frames are ~0.5 MB; over
  an iroh relay that was ~4 s per replan (burst-pause motion). JPEG →
  ~180 ms replans.
- Expect post-lift improvisation: end-at-lift demos define nothing after the
  hold, so behavior past that point is extrapolation.

## What's here

| File | What it does |
|---|---|
| `train.py` | `lerobot-train` + fresh-pipeline fix — the training entry point |
| `smoke_pi05_reference.py` | Port health check on lerobot's own libero π0.5 (step 0) |
| `smoke_generation.py` | Observation-conditioning gate on YOUR fine-tune (step 3) |
| `probe_task_sensitivity.py` | Language-channel gate via the Ficelle server (step 4) |
| `pi0fast/` | The Pi0-FAST attempt: tokenizer tooling + recipe + why it failed |
