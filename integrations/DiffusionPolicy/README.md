# Diffusion Policy — training & data prep

Robot-independent pipeline to train a **Diffusion Policy** on Grabette-recorded
demonstrations, depending only on **stock upstream lerobot** (no fork). It
mirrors the certified Pollen training recipe. Managed with **uv**.

| File | Purpose |
|---|---|
| `analyze_dataset.py` | Dataset QA (raw 8D **or** converted 11D): gripper-swing coverage, action-delta magnitudes, episode-type breakdown, anomaly detection (SLAM spikes / truncated episodes). Run first. |
| `clean_dataset.py` | Reject glitch-ridden episodes (tracking-loss segments too long to absorb, or glitches in the grasp window). Non-destructive; writes a new dataset. |
| `convert_dataset.py` | Convert a raw Grabette dataset (absolute camera poses) → **camera-local delta actions (11D) + 2D gripper state**, zeroing per-step outlier deltas (isolated SLAM glitches) along the way. |
| `train.py` | Train a `DiffusionPolicy` with the certified recipe: lean loop, best-by-val-loss checkpoint, periodic val-loss eval, UMI augmentations. |
| `offline_eval.py` | Open-loop sanity check of a trained checkpoint against the held-out val episodes (deployment inference path fed recorded observations). Run before a robot session. |
| `ood_check.py` | Is the robot seeing what the policy was trained on? Scores deployment frames (from `evaluate.py --dump_obs`) against the training distribution in the policy's own encoder features, + state-range parity. Run when the robot behaves "stereotyped"/ignores the scene. |
| `check_dataset_videos.py` | Decode-check every episode's video segments through the exact training path; prints the failing episodes (`train.py --exclude_episodes` takes the list). Run when training crashes on video decode. |
| `rotation.py` | Vendored 6D rotation helpers used by `convert_dataset.py` / `clean_dataset.py` (see [Notes](#notes)). |

**Not here:** deployment / evaluation is **robot-specific** (gRPC to the arm or
sim) and lives in the robot integration (e.g. `integrations/openarm`).

---

## Setup (uv)

```bash
cd integrations/DiffusionPolicy
uv sync                     # creates the env from pyproject.toml (lerobot + scipy)
uv sync --extra wandb       # add this if you want --wandb_project logging
```

Everything runs through `uv run`. Pin `lerobot` in `pyproject.toml` to the exact
version you validate against (the recipe was validated on lerobot 0.5.x).

---

## Workflow

**One-shot data prep:** `run_pipeline.sh` chains the filtering + conversion (steps
2–3 below, plus QA) so you don't run them by hand. Training stays a separate,
deliberate command — the script prints the exact `train.py` invocation at the end.

```bash
./run_pipeline.sh <raw_repo_id> [--raw-root DIR] [--proprioception none|relative] \
                  [--cameras "cam0"|all] [--no-qa]
```

By default the pipeline keeps **only `cam0`** (the camera the policy trains on) and
removes any extra recorded streams: an unused stream doubles video-decode cost at
every training step, and if its encoding is corrupt it crashes training even
though the policy never reads it. `--cameras all` keeps everything.

The steps below document each stage the script runs (and how to run them manually).

### 1. Inspect the raw dataset (recommended)

```bash
uv run python analyze_dataset.py --repo_id <user>/<raw_dataset>
# compare two datasets (e.g. real vs sim):
uv run python analyze_dataset.py --repo_id <user>/<real> <user>/<sim>
# local-converted dataset not on the Hub:
uv run python analyze_dataset.py --repo_id <user>/<ds> --root <local-converted path>
```

Flags episodes that never actuate the gripper, SLAM position spikes, and
truncated episodes.

### 2. Clean: reject episodes with unrecoverable tracking loss

When the grasped object occludes the wrist camera, SLAM loses tracking and the
build flags those frames (`is_lost`) while holding the last pose. A **short**
lost gap is fine — "assume no motion" (held pose → delta ≈ 0) is a good
approximation for one or a few frames. A **long** lost run means the arm really
moved through the occlusion; that motion is gone and unrecoverable, so the whole
episode is dropped.

```bash
# audit only — decide thresholds, change nothing:
uv run python clean_dataset.py --repo_id <user>/<dataset> --dry_run
# write the kept-episode dataset:
uv run python clean_dataset.py --repo_id <user>/<dataset> \
    --output_repo_id <user>/<dataset>_clean
```

Rejection keys off the SLAM's own `is_lost` flag (carried into the dataset by the
postprocess build — **rebuild an old dataset if it lacks the feature**): an
episode is dropped if its **longest consecutive lost run** exceeds `--max_lost_run`
(10) or its **lost fraction** exceeds `--max_lost_fraction` (30%). Longest-run is
the primary signal — a low-% but sustained occlusion is still unrecoverable.
Everything kept has only short lost gaps; the per-frame re-acquisition jumps are
mopped up by step 3's despike. Non-destructive (uses lerobot's `delete_episodes`).

### 3. Convert the dataset

The policy trains on **camera-local delta actions** (`[dx,dy,dz, dr6d_0..5,
proximal, distal]`, 11D) and **2D gripper state**, derived from the raw recorded
camera poses. Any single-step delta above the physical cap (an isolated SLAM
glitch) is zeroed — "hold for that frame" — which removes the bad action without
disturbing the rest of the trajectory:

```bash
uv run python convert_dataset.py \
    --repo_id <user>/<raw_dataset>_clean \
    --proprioception none \
    --output_repo_id <user>/<dataset>_cartesian
# reading a clean_dataset.py output that isn't on the Hub? add its local root:
#   --root ~/.cache/huggingface/lerobot/local-converted/<user>--<raw_dataset>_clean
```

- `--proprioception none` → `observation.state = [proximal, distal]` (gripper
  only, **no absolute position** — absolute SLAM position is meaningless to the
  policy and must never be fed in). This is the certified setup.
- `--output_repo_id` → writes a **non-destructive copy** under
  `~/.cache/huggingface/lerobot/local-converted/<repo--id>` and converts the
  copy; the raw dataset is untouched. Add `--push_to_hub <repo>` to publish.
- The converter is **frame-agnostic**: it expresses each delta in the local
  frame of the recorded pose (`Rᵀ·Δ`); whatever frame the data was recorded in
  (e.g. `oak_l`) is what the deltas live in. It bakes in no transform.

### 4. Train

```bash
uv run python train.py \
    --dataset_repo_id <user>/<dataset>_cartesian \
    --dataset_root <path printed by convert> \
    --output_dir outputs/diffusion \
    --training_steps 50000 --batch_size 64 --bf16 \
    --num_workers 4 --prefetch_factor 2 \
    --color_jitter --state_noise_std 0.01 \
    --eval_freq 500 --save_freq 5000 \
    --push_to_hub <user>/<model>
```

`train.py` **bakes in** the certified UMI-derived `DiffusionConfig` (resnet18 +
SpatialSoftmax, resize 236 / random-crop 0.95, DDIM 50/16, down_dims
(256,512,1024), lr 3e-4, betas (0.95,0.999), warmup 2000). You only pass dataset
+ run knobs. Key flags:

- `--color_jitter --state_noise_std 0.01` — the certified augmentations. **Keep them.**
- `--no_random_crop` — ablation only; default is random crop **ON** (UMI). Eval always uses a center crop.
- `--dataset_root <path>` + `HF_HUB_OFFLINE=1` — for a local-converted dataset not on the Hub.
- Best-by-val-loss checkpoint → `<output_dir>/best`, pushed to `<model>-best`. Periodic checkpoints → `<output_dir>/checkpoint_<step>`.
- `--resume_from <ckpt_dir>` — resume model + optimizer + step + rng.

---

### 5. Offline sanity check (before a robot session)

```bash
uv run python offline_eval.py \
    --checkpoint <user>/<model>-best \
    --dataset_repo_id <user>/<dataset>_cartesian [--dataset_root DIR]
```

Replays the **held-out val episodes** (same deterministic split as train.py)
through the exact deployment inference path (`select_action` queueing, eval-time
center crop), feeding recorded observations, and compares predicted vs
ground-truth actions. Catches normalization/frame bugs (`mag_ratio` far from 1),
averaging / mode collapse (`std_ratio` « 1), and gripper timing errors
(`grip_corr`, `grip_lag`), and writes per-episode overlay plots (integrated
path, |Δpos| profile, gripper channels).

**Open-loop agreement is necessary, not sufficient** — the policy sees
ground-truth observations, so compounding-error failures are invisible. A pass
means "worth a robot session", not "it works". Note `cos_dpos` ~0.4–0.5 is
normal (per-step direction of noisy SLAM-derivative deltas + 8-step replan
cadence); judge direction by the integrated-path overlay instead.

### 6. If the robot ignores the scene: OOD check

Offline pass + "stereotyped" robot behavior (same motion regardless of the
scene) usually means the deployment **observation** is out-of-distribution for
the policy's encoder. Dump one episode of the exact observations the robot
pipeline feeds the policy (`evaluate.py --dump_obs /tmp/deploy_obs
--num_episodes 1 ...`), then:

```bash
uv run python ood_check.py \
    --checkpoint <user>/<model>-best \
    --dataset_repo_id <user>/<dataset>_cartesian \
    --images /tmp/deploy_obs/ep000 [--self_test]
```

Fits the training-frame distribution in the policy's own encoder features
(64-D, Mahalanobis), calibrates on the val episodes, scores the deployment
frames, and checks `observation.state` ranges against the dataset (units /
sign / measured-vs-command). `--self_test` demonstrates sensitivity on
synthetic bugs: 180° rotation → strongly OOD; BGR swap → suspect; mild
exposure shifts → in-distribution (inside the color-jitter augmentation
envelope, so genuinely harmless). Sharpest for geometric/view mismatches —
the class that produces "ignores the scene" failures.

## `train.py` parameters

| Flag | Default | Notes |
|---|---|---|
| `--dataset_repo_id` | *(required-ish)* | Converted dataset repo id (`…_cartesian`). |
| `--dataset_root` | `None` | Local dataset dir; pair with `HF_HUB_OFFLINE=1` for a local-converted dataset. |
| `--output_dir` | `outputs/gripette/diffusion` | Checkpoints: `best/`, `checkpoint_<step>/`, and final at the root. |
| `--device` | `cuda` | Compute device. |
| `--batch_size` | `64` | Certified value. |
| `--training_steps` | `200000` | **Use `50000`** for the certified recipe. |
| `--n_action_steps` | `8` | Actions executed per re-plan (inference horizon). 8 = committed grasp; lower = more reactive but can hesitate on the trigger. |
| `--bf16` | off | bfloat16 autocast — ~1.5–2× on Ampere+/Blackwell, no GradScaler. **Use on 5090.** |
| `--compile` | off | `torch.compile` (experimental for diffusion; 1–5 min warmup). |
| `--num_workers` | `8` | DataLoader workers. `train.py` sets the file-system sharing strategy, so `/dev/shm` size won't cap this. |
| `--prefetch_factor` | `4` | Batches prefetched per worker. |
| `--color_jitter` | off | UMI color jitter. **On** for the certified recipe. |
| `--no_random_crop` | off | Disable random crop (ablation). Default = random crop **ON** (UMI); eval always center-crops. |
| `--state_noise_std` | `0.0` | Gaussian noise on the 2D gripper state. **`0.01`** for the certified recipe. |
| `--eval_freq` | `200` | Val-loss eval every N steps; updates `best/`. |
| `--save_freq` | `10000` | Periodic checkpoint every N steps. |
| `--val_ratio` | `0.1` | Fraction of episodes held out for validation (last N, deterministic). |
| `--exclude_episodes` | `None` | Episode indices to drop before the split. |
| `--cameras` | `observation.images.cam0` | Camera feature keys to use (others excluded). |
| `--push_to_hub` | `None` | Push final + `<repo>-best` to the Hub. ⚠️ conflicts with `HF_HUB_OFFLINE=1`. |
| `--hub_private` | off | Make the Hub repo private. |
| `--wandb_project` / `--wandb_run_name` | `None` | wandb logging (needs `uv sync --extra wandb`). |
| `--resume_from` | `None` | Resume model + optimizer + step + best-val + rng from a checkpoint dir. |
| `--wandb_resume_id` | `None` | Resume into an existing wandb run. |

Everything else (resnet18, resize 236 / crop 0.95, DDIM 50/16, down_dims, lr 3e-4,
betas, warmup 2000) is **baked into the `DiffusionConfig`** in `train.py` and not
exposed — that's the certified recipe.

## Example: RTX 5090 (validated)

Full 50k-step certified run on a single RTX 5090 (32 GB), on a local-converted
dataset:

```bash
HF_HUB_OFFLINE=1 uv run python train.py \
    --dataset_repo_id <user>/<dataset>_cartesian \
    --dataset_root ~/.cache/huggingface/lerobot/local-converted/<repo--id> \
    --output_dir outputs/diffusion \
    --training_steps 50000 --batch_size 64 --bf16 \
    --num_workers 4 --prefetch_factor 2 \
    --color_jitter --state_noise_std 0.01 \
    --eval_freq 500 --save_freq 5000 \
    --wandb_project gripette --wandb_run_name diffusion_5090
```

- **`--bf16`** — Blackwell runs bf16 natively (~1.5–2× vs fp32). Keep it on.
- **batch 64 / ~76 M params** uses only ~25 % of the 5090's 32 GB. You *can* raise
  `--batch_size` (128/256) for better GPU utilisation, but that deviates from the
  certified recipe — scale `lr` if you do.
- **`--num_workers 8`** is safe here because `train.py` uses the file-system
  sharing strategy. On a box with a tiny `/dev/shm` running an *older* copy of
  `train.py`, drop to `--num_workers 4 --prefetch_factor 2`.
- **Dataset must be h264, ≈960×720, 30 fps** — AV1 starves the dataloader (lerobot
  decodes video on CPU only), turning a ~few-hour run into a ~day.
- **Sanity check the first ~50 steps:** loss decreasing + steady step/s ⇒ let it
  run. Best-by-val-loss model lands in `outputs/diffusion/best`.

## Alternative: stock `lerobot-train` CLI

You can train with upstream's official CLI instead of `train.py`:

```bash
uv run lerobot-train \
    --dataset.repo_id=<user>/<dataset>_cartesian \
    --policy.type=diffusion --policy.device=cuda \
    --policy.resize_shape="[236,236]" --policy.crop_ratio=0.95 \
    --policy.down_dims="[256,512,1024]" --policy.noise_scheduler_type=DDIM \
    --policy.num_train_timesteps=50 --policy.num_inference_steps=16 \
    --policy.optimizer_lr=3e-4 --policy.optimizer_betas="[0.95,0.999]" \
    --policy.scheduler_warmup_steps=2000 --policy.use_amp=true \
    --batch_size=64 --steps=50000 --num_workers=8
```

Trade-offs vs `train.py`:

- ✅ Official, less code to maintain.
- ❌ No best-by-val-loss checkpoint, no periodic val-loss eval, no `state_noise`.
- ⚠️ **Do not** use `--dataset.image_transforms.enable=true` on synthetic data — its full-resolution jitter runs in the dataloader workers and is a severe throughput bottleneck (a different, far heavier path than `train.py`'s resize-first `--color_jitter`).
- ⚠️ Runs through `accelerate` + the processor pipeline, i.e. more per-step overhead than `train.py`'s lean loop.

Use `train.py` for the certified recipe + best-checkpoint; use `lerobot-train`
for a quick official baseline.

---

## Gotchas (hard-won — read before recording/training)

- **Data quality dominates everything.** See `recording_demonstrations_guide.md`:
  be **consistent** in what the policy can't condition on (grasp angle ≈ ±10°
  around one easy angle, decisive reach that *seats* the object, firm close),
  **diverse** in what it can (object position). The policy averages
  unconditioned variation into a single, often-bad mode.
- **Codec / resolution:** record & convert as **h264, ≈960×720, 30 fps** (match
  the real rig). AV1 software-decodes ~5–10× slower and training crawls —
  lerobot decodes video on **CPU only** (no GPU/NVDEC), so codec is the lever.
- **Dataloader / shared memory:** `train.py` sets the file-system sharing
  strategy at startup, so `--num_workers 8 --prefetch_factor 4` works even on a
  small `/dev/shm`. If you still hit `RuntimeError: unable to allocate shared
  memory` (old `train.py` copy, or a tiny `/tmp`), drop to `--num_workers 4
  --prefetch_factor 2` or raise `/dev/shm`.
- **Local dataset:** lerobot mirrors the Hub even when fully cached — for a
  local-converted dataset pass `--dataset_root <path>` and prepend
  `HF_HUB_OFFLINE=1`, or `--push_to_hub` it during conversion.
- **Gripper-state parity:** whatever you feed as `observation.state` in training
  (gripper *position* vs *command*) must be fed identically at deployment, or
  the policy won't recognise the grasped state (the lift won't trigger). On real
  hardware you only have *position* — record and deploy with position.
- **Reference-frame parity:** the deployment integrator/IK must control the
  **same frame** the data was recorded in, end-to-end. A frame mismatch shows up
  as a systematic Cartesian offset.
- **At eval:** `n_action_steps=8` gives a committed grasp; lower values make the
  approach more reactive but can make the policy hesitate on the grasp trigger.

---

## Troubleshooting (symptom → cause → fix)

Run trainings inside `screen`/`tmux`, log with `... 2>&1 | tee train.log`, and
read the true exit code with `${pipestatus[1]}` (zsh) / `${PIPESTATUS[0]}` (bash).
`137` = killed by the OS (usually out of RAM) · `139` = segfault in a native
library · `0`/`1` = read the log.

| Symptom | Cause | Fix |
|---|---|---|
| Training stops silently mid-run; `dmesg -T` shows `oom-kill … pt_data_worker` | DataLoader workers exhausted system RAM | Lower `--num_workers` / `--prefetch_factor` (defaults are safe); after any crash, `pkill -9 -f train.py` — orphaned workers keep eating RAM |
| `unable to allocate shared memory(shm) for file </torch_…>` mid-training | `$TMPDIR` is RAM-backed tmpfs; worker shm files filled it (train.py warns about this at startup) | `mkdir -p ~/tmp && TMPDIR=~/tmp uv run python train.py …` |
| `Could not push packet to decoder: Invalid data …` | A corrupt video segment in the dataset (often written to a full tmpfs) | `check_dataset_videos.py` names the episodes → `--exclude_episodes <list>`, or re-run the pipeline with `--work` on real disk |
| Instant exit, empty log, or import-time segfault | Broken venv (interrupted sync) or Python ≠ 3.12 | `rm -rf .venv && uv sync` (the pyproject pins Python 3.12 and lerobot 0.5.x) |
| `AttributeError: 'NoneType' … shape` at policy init | Pointed at the RAW dataset instead of the converted one (train.py now explains this itself) | Train on the `*_cartesian` output; the exact command is in `<work>/train_command.txt` |
| `HFValidationError: Repo id must be in the form…` on `--resume_from` | Checkpoint path didn't exist so it was treated as a Hub id (now guarded) | Pass the **absolute** path; dirs are zero-padded (`checkpoint_015000`) |
| Datasets vanished after a reboot | They were in `/tmp` (tmpfs) | Keep work dirs on disk (`run_pipeline.sh` now defaults to `~/.cache/grabette_pipeline` and warns on tmpfs); raw datasets belong on the Hub |

## Notes

- **Vendored `rotation.py`:** the 6D-rotation helpers `convert_dataset.py` needs
  (`rotvec_to_rotation_6d`, `rotation_*_6d_*_numpy`) are fork additions in the
  Pollen lerobot fork and are **not** in stock lerobot, so they're vendored here.
  Verified numerically identical to the fork (~6e-7).
- **lerobot API surface used:** `make_pre_post_processors`,
  `DiffusionConfig`/`DiffusionPolicy`, `LeRobotDataset(Metadata)`,
  `dataset_to_policy_features`, `DiffusionConfig.get_optimizer_preset`, and (in
  `convert_dataset.py`) `lerobot.datasets.dataset_tools.recompute_stats`. Stable
  across lerobot 0.5.x; if you bump lerobot and a call moves, that's where to look.
- **Reproducing the certified model:** keep `--color_jitter --state_noise_std
  0.01`, batch 64, 50k steps, and the default (random) crop.
