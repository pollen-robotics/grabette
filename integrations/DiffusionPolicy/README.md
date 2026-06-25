# Diffusion Policy — training & data prep

Robot-independent pipeline to train a **Diffusion Policy** on Grabette-recorded
demonstrations, depending only on **stock upstream lerobot** (no fork). It
mirrors the certified Pollen training recipe. Managed with **uv**.

| File | Purpose |
|---|---|
| `convert_dataset.py` | Convert a raw Grabette dataset (absolute camera poses) → **camera-local delta actions (11D) + 2D gripper state**. |
| `train.py` | Train a `DiffusionPolicy` with the certified recipe: lean loop, best-by-val-loss checkpoint, periodic val-loss eval, UMI augmentations. |
| `analyze_dataset.py` | Dataset QA: gripper-swing coverage, action-delta magnitudes, episode-type breakdown, anomaly detection (glitchy/truncated episodes). Run before training. |
| `rotation.py` | Vendored 6D rotation helpers used by `convert_dataset.py` (see [Notes](#notes)). |

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

### 1. Inspect the raw dataset (optional but recommended)

```bash
uv run python analyze_dataset.py --repo_id <user>/<raw_dataset>
# compare two datasets (e.g. real vs sim):
uv run python analyze_dataset.py --repo_id <user>/<real> <user>/<sim>
# local-converted dataset not on the Hub:
uv run python analyze_dataset.py --repo_id <user>/<ds> --root <local-converted path>
```

Flags episodes that never actuate the gripper, glitchy SLAM spikes, and
truncated episodes — drop those before training.

### 2. Convert the dataset

The policy trains on **camera-local delta actions** (`[dx,dy,dz, dr6d_0..5,
proximal, distal]`, 11D) and **2D gripper state**, derived from the raw recorded
camera poses:

```bash
uv run python convert_dataset.py \
    --repo_id <user>/<raw_dataset> \
    --proprioception none \
    --output_repo_id <user>/<dataset>_cartesian
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

### 3. Train

```bash
uv run python train.py \
    --dataset_repo_id <user>/<dataset>_cartesian \
    --dataset_root <path printed by convert> \
    --output_dir outputs/diffusion \
    --training_steps 50000 --batch_size 64 --bf16 \
    --num_workers 8 --prefetch_factor 4 \
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
- **Dataloader:** `--num_workers 8 --prefetch_factor 4`. If you hit
  `RuntimeError: unable to allocate shared memory`, lower `--num_workers` (4) or
  raise `/dev/shm`.
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
