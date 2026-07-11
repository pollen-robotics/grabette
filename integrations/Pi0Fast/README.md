# Pi0-FAST for Grabette

Fine-tune [Pi0-FAST](https://www.physicalintelligence.company/research/fast)
(`lerobot/pi0fast-base`, a 3B PaliGemma VLA with FAST action tokens) on
Grabette demonstrations. Companion to `integrations/DiffusionPolicy` — the
**dataset preparation is shared** (same converted 11D camera-local-delta
datasets); this integration adds the Pi0-FAST-specific pieces: action
tokenizer fitting, training recipe, and generation gates.

## Read this first: the base-checkpoint quirk that cost us a month

The stock `lerobot/pi0fast-base` checkpoint produces **degenerate generation
out of the box** (unicode garbage / `<bos>` loops) on every released lerobot
version — reproduced with public assets only (see the standalone repro
project, `pi0fast-minimal-repro`). Root cause per the LeRobot team: the
checkpoint itself bakes in a double-`<bos>` prompt quirk. The resolution is
NOT a code fix but a **consistent fine-tune**: train from `pi0fast-base` with
lerobot **main**, whose training and inference pipelines share the quirk, and
the fine-tune learns to work through it. Existence proof:
`lerobot/pi0fast-libero` generates well-formed, input-dependent actions on
main (verified 2026-07-10, real libero observations, predictions track GT).

Consequences:
- **Train and infer on the SAME lerobot main revision** (pinned in
  `pyproject.toml`). Do not mix a fine-tune with another lerobot series, and
  do not use the `fix/pi0fast-double-bos-degenerate-generation` branch — it
  breaks the consistency the fine-tune relies on.
- Teacher-forced training loss CANNOT detect a broken inference pipeline —
  our June 2026 fine-tunes had healthy losses and garbage generation
  (they were also stopped at ~5k steps, inside the 4k-step LR warmup).
  **Always run `smoke_generation.py` before a robot session.**

## What's here

| File | What it does |
|---|---|
| `fit_fast_tokenizer.py` | Fit a FAST action tokenizer on YOUR converted dataset's actions (MEAN_STD-normalized chunks). The working libero fine-tune used a dataset-specific tokenizer — do the same. |
| `verify_fast_tokenizer.py` | Round-trip encode→decode gate for the fitted tokenizer. Run before burning a training run. |
| `smoke_generation.py` | Post-training generation health gate on real dataset frames: well-formed decode, input-dependent, tracks GT. Run before ANY robot session. |

## Workflow

```bash
# 0. Prepare the dataset with the shared pipeline (see ../DiffusionPolicy):
#    ../DiffusionPolicy/run_pipeline.sh <user>/<raw_dataset> [--smooth-poses 9]

# 1. Fit the action tokenizer on the converted dataset (chunk_size = policy's)
uv run python fit_fast_tokenizer.py \
    --dataset_repo_id local/<name>_cartesian --dataset_root <pipeline work dir>/cartesian \
    --chunk_size 10 --output_dir ./fast_tokenizer_<task> \
    --push_repo <user>/fast_tokenizer_<task>

# 2. Verify the tokenizer round-trip (gate: mean abs err < 0.05)
uv run python verify_fast_tokenizer.py \
    --tokenizer <user>/fast_tokenizer_<task> \
    --dataset_repo_id local/<name>_cartesian --dataset_root ... \
    --action_horizon 10 --action_dim 11

# 3. Train (see recipe below — needs a big GPU, A100/H100 class for batch 32)

# 4. Generation gate on the fine-tune (BEFORE the robot)
uv run python smoke_generation.py \
    --checkpoint <user>/<model> \
    --dataset_repo_id local/<name>_cartesian --dataset_root ... --task pick

# 5. Deploy with the robot integration's evaluate.py (sync path), as for
#    Diffusion. Same start-pose rules apply (openarm_gripette README).
```

## Training recipe (copied from the verified `lerobot/pi0fast-libero` run)

Extracted from that checkpoint's `config.json` / `train_config.json` — the
only pi0fast fine-tune verified to generate correctly:

| Ingredient | Value | Note |
|---|---|---|
| pretrained | `lerobot/pi0fast-base` | |
| action tokenizer | your fitted one (step 1) | libero used its own (`jadechoghari/tokenizer-lib-mean`) |
| chunk_size / n_action_steps | 10 / 10 | n_obs_steps 1 |
| cameras | 1 real + `empty_cameras=2` | base expects 3 camera slots |
| normalization | VISUAL=IDENTITY, STATE/ACTION=MEAN_STD | |
| decoding | temperature 0.0, kv-cache, `validate_action_token_prefix=true`, max_action_tokens 256 | |
| optimizer | AdamW lr 2.5e-5, betas (0.9, 0.95), wd 0.01, clip 1.0 | |
| schedule | warmup 4000, decay to 1e-5 over 100k | do NOT stop inside warmup |
| steps / batch | **20 000** / 32 | bf16, gradient_checkpointing, compile |
| image transforms | off | |

Training goes through stock `lerobot-train` (draccus CLI) on the pinned main
revision. Template (validate flag names against `lerobot-train --help` on the
training box — main moves fast):

```bash
lerobot-train \
  --policy.type=pi0_fast \
  --policy.pretrained_path=lerobot/pi0fast-base \
  --policy.chunk_size=10 --policy.n_action_steps=10 \
  --policy.empty_cameras=2 \
  --policy.action_tokenizer_name=<user>/fast_tokenizer_<task> \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 \
  --dataset.repo_id=local/<name>_cartesian --dataset.root=<...>/cartesian \
  --steps=20000 --batch_size=32 --num_workers=4 \
  --output_dir=outputs/<task>_pi0fast --policy.push_to_hub=true \
  --policy.repo_id=<user>/<task>_pi0fast
```

Hardware: 3B params in bf16 + batch 32 with gradient checkpointing is
A100/H100 territory. A 32 GB consumer GPU works with a small batch +
gradient accumulation, slowly.

## Why Pi0-FAST at all (vs the shipped Diffusion baseline)

Same reason as in `docs/policy_comparison_gripette.md`: Grabette data is
inherently Cartesian-delta (hand-held, no robot joints at demo time), which
favors distributional/token-based policies over L1 regressors. Pi0-FAST adds
a pretrained vision-language prior — the bet is better generalization
(positions, distractors, lighting) than the from-scratch Diffusion baseline,
at the price of a 3B model and slower autoregressive inference.
