"""Train a Diffusion Policy for the Gripette project.

This script trains a DiffusionPolicy on a dataset prepared by convert_dataset.py:
  - observation.state = [proximal, distal]  (2D gripper joints — no absolute position)
  - action = [dx, dy, dz, dr6d_0..5, proximal, distal]  (11D: deltas + gripper)
  - observation.images.cam0 = camera image

The model sees camera + gripper state as input, and predicts delta actions.
No absolute position is fed to the model (it's meaningless in the SLAM reference frame).
Delta actions are pre-computed in the dataset (following the UMI approach).

See README.md in this directory for the full setup guide.

Prerequisites:
  - Dataset converted with convert_dataset.py:
      uv run python convert_dataset.py

Usage:
  uv run python train.py --dataset_repo_id <user>/<dataset>_cartesian
  uv run python train.py --dataset_repo_id <user>/<dataset>_cartesian --batch_size 64

Cloud training (no local GPU) — the PEP-723 header below makes this file a
self-contained uv script, so it runs on HF Jobs as-is (see README → "Cloud
training with HF Jobs"):
  hf jobs uv run --flavor a100-large --timeout 8h -s HF_TOKEN train.py -- \\
      --dataset_repo_id <user>/<dataset>_cartesian --push_to_hub <user>/<model> ...
"""

# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["lerobot>=0.5.1,<0.6"]
# ///

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.datasets.feature_utils import dataset_to_policy_features

# DataLoader-worker tensor sharing: default to $TMPDIR-file-backed shm, so a
# small /dev/shm (common on servers/containers) doesn't cause "RuntimeError:
# unable to allocate shared memory". Cost: every batch crosses the worker→main
# boundary through the filesystem — slow on container overlay fs; the
# --shm_strategy flag switches to /dev/shm ('file_descriptor') where it's large
# enough. Must be set before any DataLoader worker spawns; argparse runs later,
# so we peek at argv directly here.
_shm = "file_descriptor" if any("file_descriptor" in a for a in sys.argv) else "file_system"
torch.multiprocessing.set_sharing_strategy(_shm)

# Line-buffer stdout even when piped (e.g. `... | tee train.log`). Python
# block-buffers piped stdout, so a hard kill (OOM SIGKILL, segfault) discards
# everything still in the buffer — leaving an EMPTY log for a run that printed
# for minutes. Line buffering makes every step line land in the log immediately.
sys.stdout.reconfigure(line_buffering=True)


def _warn_if_shm_dir_on_tmpfs() -> None:
    """The file_system sharing strategy backs worker batches with files in
    $TMPDIR. If that is a RAM-backed tmpfs (common for /tmp), the files eat RAM
    and eventually exhaust the mount — the run dies mid-training with
    "unable to allocate shared memory(shm) for file </torch_...>".
    Warn up-front with the fix instead of failing at step 16000."""
    import tempfile
    tmp = Path(tempfile.gettempdir()).resolve()
    try:
        best, fstype = Path("/"), ""
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            mnt = Path(parts[1])
            if mnt == tmp or mnt in tmp.parents:
                if len(str(mnt)) >= len(str(best)):
                    best, fstype = mnt, parts[2]
        if fstype in ("tmpfs", "ramfs"):
            print(f"WARNING: TMPDIR ({tmp}) is on {fstype} (RAM-backed). DataLoader shared-\n"
                  f"         memory files will consume RAM and can exhaust the mount mid-\n"
                  f"         training ('unable to allocate shared memory ... </torch_...>').\n"
                  f"         Fix: run with TMPDIR set to a real-disk dir, e.g.\n"
                  f"             mkdir -p ~/tmp && TMPDIR=~/tmp uv run python train.py ...",
                  file=sys.stderr)
    except OSError:
        pass  # non-Linux or unreadable /proc — nothing to check


_warn_if_shm_dir_on_tmpfs()


def save_train_state(ckpt_dir: Path, *, optimizer, step: int, best_val_loss: float):
    """Save optimizer + bookkeeping next to the model checkpoint, so the run
    can be resumed exactly later."""
    torch.save(
        {
            "step": int(step),
            "best_val_loss": float(best_val_loss),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            ),
        },
        ckpt_dir / "train_state.pt",
    )


def load_train_state(ckpt_dir: Path, optimizer, device: torch.device):
    """Load optimizer + bookkeeping. Returns (step, best_val_loss). If
    train_state.pt is missing (older checkpoint), returns (0, inf) and the
    user gets a fresh-start training but with the loaded model weights."""
    p = ckpt_dir / "train_state.pt"
    if not p.exists():
        print(f"  WARNING: {p} not found; resuming model only (step=0, best_val=inf).")
        return 0, float("inf")
    state = torch.load(p, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
        try:
            torch.cuda.set_rng_state(state["cuda_rng_state"].cpu())
        except Exception:
            pass
    step = int(state["step"])
    best = float(state["best_val_loss"])
    print(f"  Resumed: step={step}, best_val_loss={best:.4f}")
    return step, best


def apply_state_noise(batch: dict, std: float, device: torch.device) -> dict:
    """Add zero-mean Gaussian noise to observation.state (training-only).

    The proprioception channel here is the 2D gripper joint state. With a
    small visible scripted-demo distribution the policy can memorise the
    state→action mapping and ignore the camera; jitter on the state input
    forces it to rely on visual features. Standard regulariser used in
    UMI / diffusion-policy work.

    Called BEFORE the preprocessor so normalization stats apply to the
    noised value. Gradients flow through the noise (it is just an additive
    perturbation, not a stochastic node we backprop through).
    """
    if std <= 0.0:
        return batch
    state = batch.get("observation.state")
    if state is None:
        return batch
    state = state.to(device, non_blocking=True)
    noise = torch.randn_like(state) * std
    batch["observation.state"] = state + noise
    return batch


def apply_color_jitter(
    batch: dict,
    image_keys: list[str],
    jitter: T.ColorJitter,
    device: torch.device,
    resize_shape: tuple[int, int] | None = None,
) -> dict:
    """Apply color jitter to image tensors in the batch (training-only augmentation).

    Images are moved to ``device`` first so the jitter runs on GPU — otherwise
    the CPU jitter path dominates iteration time on a GPU-bound training loop.

    If ``resize_shape`` is provided, images are resized *before* jitter. The
    jitter's hue conversion internally allocates ~6 intermediate tensors of
    the image size; at native 720×960 + batch 128 this peaks at ~12 GB, which
    is enough to OOM a 32 GB GPU once the model + compile workspace is in.
    Resizing first (e.g. to 236×236 — matching DiffusionConfig.resize_shape)
    cuts that ~12×. The model internally resizes anyway, so this only moves
    the resize upstream.

    Random per-batch brightness / contrast / saturation / hue perturbations.
    Applied in-place on the image tensors so the batch dict is returned unchanged
    except for the jittered images.

    Called BEFORE the preprocessor so normalization stats still apply correctly.
    The jittered image is what reaches the vision encoder during training.
    At inference (in eval scripts) this function is never called — checkpoint
    is unchanged. Matches UMI's approach (brightness=0.3, contrast=0.4,
    saturation=0.5, hue=0.08).
    """
    import torch.nn.functional as F

    for key in image_keys:
        if key not in batch:
            continue
        # Move to GPU first: ColorJitter on a CUDA tensor runs on the GPU and
        # takes ~microseconds; on CPU it's ~tens of ms per batch (dominating).
        img = batch[key].to(device, non_blocking=True)
        # Image shape from the dataloader: (B, T, C, H, W) because of n_obs_steps>1.
        if img.ndim == 5:  # (B, T, C, H, W)
            b, t = img.shape[:2]
            flat = img.reshape(b * t, *img.shape[2:])
            if resize_shape is not None and flat.shape[-2:] != tuple(resize_shape):
                flat = F.interpolate(
                    flat, size=resize_shape, mode="bilinear", align_corners=False
                )
            flat = jitter(flat)
            batch[key] = flat.reshape(b, t, *flat.shape[1:])
        else:
            if resize_shape is not None and img.shape[-2:] != tuple(resize_shape):
                img = F.interpolate(
                    img, size=resize_shape, mode="bilinear", align_corners=False
                )
            batch[key] = jitter(img)
    return batch


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """Convert frame-offset indices to seconds for the dataset's delta_timestamps."""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


@torch.no_grad()
def compute_val_loss(policy, preprocessor, val_dataloader, device, max_batches=50, bf16=False):
    """Compute average loss on the validation set."""
    policy.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in val_dataloader:
        batch = preprocessor(batch)
        if bf16 and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(batch)
        else:
            loss, _ = policy.forward(batch)
        total_loss += loss.item()
        num_batches += 1
        if num_batches >= max_batches:
            break
    policy.train()
    return total_loss / max(num_batches, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy for Gripette")
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="pollen/gripette_demo",
        help="LeRobot dataset repo ID",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Local directory holding the dataset (mirrors upstream's "
             "--dataset.root). Use this to read a local-converted dataset that "
             "was never pushed to the Hub, e.g. "
             "~/.cache/huggingface/lerobot/local-converted/<repo--id>. "
             "When set, prepend HF_HUB_OFFLINE=1 to skip the Hub round-trip.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gripette/diffusion",
        help="Directory for checkpoints",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--training_steps", type=int, default=200_000, help="Total training steps")
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=8,
        help="Actions executed before re-planning (4=reactive, 8=default, 16=smooth)",
    )
    parser.add_argument("--log_freq", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_freq", type=int, default=10_000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_freq", type=int, default=200, help="Evaluate on validation set every N steps")
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Fraction of episodes used for validation"
    )
    parser.add_argument(
        "--exclude_episodes", type=int, nargs="+", default=None,
        help="Episode indices to drop before the train/val split (e.g. the IK-flip "
             "episodes from convert_to_jointspace.py). Pass the SAME list as the ACT "
             "runs to keep the Diffusion-vs-ACT comparison on one episode set.",
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="Wandb project name (None = disabled)"
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["observation.images.cam0"],
        help="Camera feature keys to use as input (others are excluded)",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="HuggingFace Hub repo ID to push final + best checkpoints to (e.g. 'user/gripette_v1')",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Make the HuggingFace Hub repo private (default: public)",
    )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="Enable color jitter augmentation during training (UMI values)",
    )
    parser.add_argument(
        "--no_random_crop",
        action="store_true",
        help="DISABLE random-crop augmentation (ablation only). Default: random "
             "crop ON (UMI 95%% random crop of the 236px resize) — the certified "
             "recipe. Eval always uses a center crop regardless; this flag makes "
             "training use center crop too.",
    )
    parser.add_argument(
        "--state_noise_std",
        type=float,
        default=0.0,
        help="Per-step Gaussian noise std added to observation.state during "
             "training (radians, since observation.state is gripper joints). "
             "Discourages the policy from memorising state→action and forces "
             "the visual encoder to carry information. 0.0 disables.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint directory (e.g. outputs/.../checkpoint_010000 "
             "or .../best) to resume training from. Loads model weights, "
             "optimizer state, step counter, best val_loss tracker and rng. "
             "Continues until --training_steps. The model config still comes "
             "from --dataset_repo_id and the policy code, so make sure they "
             "match the saved checkpoint.",
    )
    parser.add_argument(
        "--wandb_resume_id",
        type=str,
        default=None,
        help="Optional wandb run id to resume into. Without this, --resume_from "
             "starts a fresh wandb run (the original run becomes orphaned). "
             "Find the id in the original run URL: wandb.ai/<entity>/<proj>/runs/<ID>.",
    )
    # -- GPU throughput --
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers. The default is sized to survive on ordinary RAM: "
             "each worker holds prefetched batches of DECODED video frames, and "
             "8+ workers have been OOM-killed (silent crash, exit 137, "
             "'pt_data_worker' in dmesg) on 32GB machines. Raise to 8-16 only on "
             "big-RAM boxes if the GPU is starved.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Batches pre-loaded per worker. Multiplies worker memory; raise only "
             "together with --num_workers on big-RAM machines.",
    )
    parser.add_argument(
        "--shm_strategy",
        choices=["file_system", "file_descriptor"],
        default="file_system",
        help="How DataLoader workers hand tensors to the main process. "
             "'file_system' (default) writes through $TMPDIR files — survives tiny "
             "/dev/shm but every batch pays a filesystem round-trip (slow on container "
             "overlay fs). 'file_descriptor' uses /dev/shm — faster where /dev/shm is "
             "large enough (watch for 'unable to allocate shared memory' if not).",
    )
    parser.add_argument(
        "--video_backend",
        choices=["torchcodec", "pyav"],
        default=None,
        help="Video decoding backend (default: lerobot's choice, usually torchcodec). "
             "torchcodec needs SYSTEM FFmpeg libraries; pyav bundles its own FFmpeg "
             "inside the wheel. Use 'pyav' in containers/cloud jobs without system "
             "FFmpeg (symptom: 'Could not load libtorchcodec / libavutil.so not found').",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 autocast for forward/backward. ~1.5-2x speedup on Ampere+/Blackwell "
        "(RTX 30xx/40xx/50xx). No GradScaler needed; bf16 has fp32-range exponent.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the policy. Adds 1-5 min warm-up at start but typically yields "
        "20-40%% throughput on recent GPUs. Experimental for diffusion — disable if training "
        "errors out during warmup.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- GPU throughput knobs ----
    # cudnn.benchmark picks the fastest conv kernel per-shape (stable shape = big win).
    # TF32 on Ampere+/Blackwell: ~2x faster matmul than fp32 with negligible accuracy hit.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- Dataset metadata ----
    # Load metadata without downloading the full dataset. This gives us the feature
    # definitions and statistics needed to configure the policy.
    dataset_metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    # Filter input features: keep only selected cameras + non-visual features.
    # This excludes cameras not listed in --cameras (e.g., cam1 when only cam0 is used).
    input_features = {
        key: ft
        for key, ft in features.items()
        if key not in output_features and (ft.type is not FeatureType.VISUAL or key in args.cameras)
    }

    # Action feature names, for logging only.
    action_feature_names = dataset_metadata.features.get("action", {}).get("names")

    print(f"Dataset:          {args.dataset_repo_id}")
    print(f"FPS:              {dataset_metadata.fps}")
    print(f"Input features:   {list(input_features.keys())}")
    print(f"Output features:  {list(output_features.keys())}")
    print(f"Action names:     {action_feature_names}")

    # ---- Guard: this recipe trains on the CONVERTED dataset only ----
    # A raw build has an 8D absolute-pose action and no observation.state;
    # pointing train.py at it crashes deep in the policy init with an opaque
    # AttributeError. Fail here with the actual diagnosis instead.
    action_dim = dataset_metadata.features["action"]["shape"][0]
    if action_dim != 11 or "observation.state" not in dataset_metadata.features:
        raise SystemExit(
            f"\nERROR: this looks like a RAW dataset (action dim {action_dim}, "
            f"state {'present' if 'observation.state' in dataset_metadata.features else 'MISSING'}).\n"
            f"train.py needs the CONVERTED dataset (11D delta actions + 2D gripper state).\n"
            f"Run the prep pipeline first:  ./run_pipeline.sh {args.dataset_repo_id}\n"
            f"then train on its printed output (repo_id local/<name>_cartesian + --dataset_root)."
        )

    # ---- Policy configuration ----
    # Parameters are aligned with the UMI (Universal Manipulation Interface) project,
    # which is a known-working diffusion policy for SLAM-recorded Cartesian datasets.
    # See docs/umi_analysis.md for the full comparison.
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        # -- Temporal structure (same as UMI) --
        n_obs_steps=2,
        horizon=16,
        n_action_steps=args.n_action_steps,
        # -- Vision encoder --
        # ResNet18 with GroupNorm + SpatialSoftmax (32 keypoints).
        # UMI uses ViT-base (CLIP pretrained), but ResNet18 is lighter and faster.
        # Resize to a larger shape than the final crop so the random crop has headroom
        # to pick different framings during training (matches UMI: crop_ratio=0.95).
        # At inference the crop is always centered — deterministic behavior.
        vision_backbone="resnet18",
        resize_shape=(236, 236),  # slightly larger so 95% crop = 224x224
        crop_ratio=0.95,
        crop_is_random=not args.no_random_crop,  # default True (UMI); pass --no_random_crop to disable
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=32,
        # -- U-Net (same as UMI) --
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=128,
        use_film_scale_modulation=True,
        # -- Diffusion scheduler --
        # DDIM with 50 training steps and 16 inference steps (matching UMI).
        # DDIM is ~6x faster than DDPM at inference with minimal quality loss.
        noise_scheduler_type="DDIM",
        num_train_timesteps=50,
        num_inference_steps=16,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        # -- Normalization --
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        # Delta actions are pre-computed in the dataset by convert_dataset.py;
        # observation.state is gripper only (2D), no absolute position, so the
        # policy predicts the deltas directly (no relative-action processing).
        # -- Optimizer (matching UMI: lr=3e-4, warmup=2000) --
        optimizer_lr=3e-4,
        optimizer_betas=(0.95, 0.999),
        optimizer_weight_decay=1e-6,
        scheduler_name="cosine",
        scheduler_warmup_steps=2000,
    )

    # ---- Instantiate policy ----
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        # Guard: if the directory doesn't exist, lerobot's from_pretrained
        # falls through to interpreting the path as a HUB REPO ID and dies
        # with a cryptic HFValidationError. Fail with the real diagnosis.
        if not (ckpt_path / "config.json").is_file():
            raise SystemExit(
                f"\nERROR: --resume_from checkpoint not found: {ckpt_path.resolve()}\n"
                f"(no config.json there). Relative paths resolve from the directory you\n"
                f"run in — pass the ABSOLUTE path to the checkpoint dir, e.g.\n"
                f"  --resume_from <output_dir>/checkpoint_15000  (must contain config.json)"
            )
        print(f"\nResuming from checkpoint: {ckpt_path}")
        policy = DiffusionPolicy.from_pretrained(ckpt_path)
    else:
        policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)

    param_count = sum(p.numel() for p in policy.parameters())
    print(f"\nModel parameters: {param_count:,}")

    # ---- Optional torch.compile ----
    # Compile triggers lazy tracing; first forward takes 1-5 min. Subsequent iters
    # run ~20-40% faster on recent GPUs. Disable via --compile if it errors out.
    compiled_policy = None
    if args.compile:
        print("  torch.compile: compiling policy (first iteration will be slow)...")
        compiled_policy = torch.compile(policy, mode="reduce-overhead", dynamic=False)

    # ---- Pre/post processors ----
    # The preprocessor converts raw data to model input:
    #   rename -> add batch dim -> move to device -> relative actions -> normalize
    # The postprocessor reverses the action transforms:
    #   unnormalize -> absolute actions -> move to CPU
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # ---- Dataset with temporal windowing ----
    # delta_timestamps tells the dataset to stack multiple frames per sample.
    # For observations: current frame + 1 previous frame (n_obs_steps=2).
    # For actions: a window of `horizon` frames starting from the previous observation.
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # Image features use the same observation timestamps as the state.
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    # ---- Train/val split by episodes ----
    all_episodes = sorted(dataset_metadata.episodes["episode_index"])
    if args.exclude_episodes:
        # Drop flagged episodes (e.g. the IK-flip episodes from
        # convert_to_jointspace.py). Pass the SAME list as the ACT runs so the
        # Diffusion-vs-ACT comparison is on an identical episode set.
        excl = set(args.exclude_episodes)
        all_episodes = [e for e in all_episodes if e not in excl]
        print(f"  Excluding {len(excl)} episodes; {len(all_episodes)} remain.")
    # STRIDED validation split (deterministic, no randomness): every Nth episode.
    # Consecutive episodes are recorded minutes apart — same lighting, operator
    # rhythm, object placements — so a tail split (old behavior) validates on a
    # correlated blob and flatters the metrics. Striding spreads the val set
    # across the whole recording session(s).
    stride = max(2, round(1 / args.val_ratio))
    val_episodes = all_episodes[::stride]
    val_set = set(val_episodes)
    train_episodes = [e for e in all_episodes if e not in val_set]
    # Persist the exact split next to the checkpoints: offline_eval.py reads it
    # from there, so eval can never silently disagree with training (exclusions
    # and split changes included).
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "val_episodes.json").write_text(json.dumps(
        {"val_episodes": val_episodes, "split": "stride", "val_ratio": args.val_ratio}))
    print(f"  Val episodes (strided, saved to val_episodes.json): {val_episodes}")

    train_dataset = LeRobotDataset(
        args.dataset_repo_id, root=args.dataset_root,
        delta_timestamps=delta_timestamps, episodes=train_episodes,
        video_backend=args.video_backend,
    )
    val_dataset = LeRobotDataset(
        args.dataset_repo_id, root=args.dataset_root,
        delta_timestamps=delta_timestamps, episodes=val_episodes,
        video_backend=args.video_backend,
    )

    print(f"  Train episodes:   {len(train_episodes)} ({len(train_dataset)} frames)")
    print(f"  Val episodes:     {len(val_episodes)} ({len(val_dataset)} frames)")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type != "cpu",
        drop_last=False,
        num_workers=2,
    )

    # ---- Optimizer ----
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    # ---- Resume bookkeeping ----
    resumed_step = 0
    resumed_best_val = float("inf")
    if args.resume_from:
        resumed_step, resumed_best_val = load_train_state(
            Path(args.resume_from), optimizer, device,
        )

    # ---- Color jitter augmentation (training only, not saved to checkpoint) ----
    # UMI defaults: brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08.
    color_jitter = None
    if args.color_jitter:
        color_jitter = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)
        image_keys = list(cfg.image_features.keys())
        print(f"  Color jitter:     enabled on {image_keys}")

    # ---- Wandb ----
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=args.wandb_resume_id,
            resume="allow" if args.wandb_resume_id else None,
            config={
                "dataset": args.dataset_repo_id,
                "batch_size": args.batch_size,
                "training_steps": args.training_steps,
                "lr": cfg.optimizer_lr,
                "warmup_steps": cfg.scheduler_warmup_steps,
                "n_obs_steps": cfg.n_obs_steps,
                "horizon": cfg.horizon,
                "n_action_steps": cfg.n_action_steps,
                "noise_scheduler": cfg.noise_scheduler_type,
                "num_train_timesteps": cfg.num_train_timesteps,
                "num_inference_steps": cfg.num_inference_steps,
                "vision_backbone": cfg.vision_backbone,
                "resize_shape": cfg.resize_shape,
                "down_dims": cfg.down_dims,
                "color_jitter": args.color_jitter,
                "state_noise_std": args.state_noise_std,
                "crop_ratio": cfg.crop_ratio,
                "action_dim": cfg.action_feature.shape[0],
                "state_dim": cfg.robot_state_feature.shape[0],
                "cameras": args.cameras,
                "model_params": param_count,
            },
        )

    # ---- Training loop ----
    print(f"\nStarting training for {args.training_steps} steps on {device}")
    print(f"  Batch size:       {args.batch_size}")
    print("  Actions:          pre-computed deltas (11D)")
    print(f"  Eval every:       {args.eval_freq} steps")
    print(f"  Checkpoints:      {output_dir}")
    if use_wandb:
        print(f"  Wandb:            {args.wandb_project}")
    print()

    best_val_loss = resumed_best_val
    step = resumed_step
    done = False
    t_log = time.perf_counter()  # throughput clock (it/s over each log window)
    speed = ""
    while not done:
        for batch in train_dataloader:
            # Training-only image augmentation (BEFORE normalization in preprocessor)
            if color_jitter is not None:
                batch = apply_color_jitter(
                    batch, image_keys, color_jitter, device, resize_shape=cfg.resize_shape
                )
            # Training-only state-noise regulariser (BEFORE normalization)
            if args.state_noise_std > 0.0:
                batch = apply_state_noise(batch, args.state_noise_std, device)

            # Forward pass (optionally in bf16 for ~1.5-2x speedup on Ampere+/Blackwell)
            batch = preprocessor(batch)
            fwd_policy = compiled_policy if compiled_policy is not None else policy
            if args.bf16 and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = fwd_policy.forward(batch)
            else:
                loss, _ = fwd_policy.forward(batch)

            # Backward pass (bf16 doesn't need GradScaler; fp32-range exponent handles it)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss = loss.item()

            # Throughput over the last log window (train steps only — the
            # val pause is excluded by resetting the clock after eval below).
            if step % args.log_freq == 0:
                now = time.perf_counter()
                its = args.log_freq / max(now - t_log, 1e-9) if step > 0 else 0.0
                eta_h = (args.training_steps - step) / its / 3600 if its > 0 else 0.0
                speed = f"  {its:5.2f} it/s  eta {eta_h:4.1f}h" if step > 0 else ""
                t_log = now

            # ---- Validation ----
            val_loss = None
            if step > 0 and step % args.eval_freq == 0:
                val_loss = compute_val_loss(
                    policy, preprocessor, val_dataloader, device, bf16=args.bf16
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save best checkpoint
                    best_dir = output_dir / "best"
                    policy.save_pretrained(best_dir)
                    preprocessor.save_pretrained(best_dir)
                    postprocessor.save_pretrained(best_dir)
                    save_train_state(best_dir, optimizer=optimizer,
                                     step=step, best_val_loss=best_val_loss)

                print(
                    f"step: {step:>7d} / {args.training_steps}  "
                    f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                    f"best_val: {best_val_loss:.4f}" + (" *" if val_loss <= best_val_loss else "")
                    + speed
                )
                t_log = time.perf_counter()  # don't bill the val pause to the next window

            if use_wandb:
                log_dict = {"train_loss": train_loss, "step": step}
                if step % args.log_freq == 0 and step > 0:
                    log_dict["it_per_s"] = its
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                    log_dict["best_val_loss"] = best_val_loss
                wandb.log(log_dict)

            if step % args.log_freq == 0 and val_loss is None:
                print(f"step: {step:>7d} / {args.training_steps}  train_loss: {train_loss:.4f}{speed}")

            # Periodic checkpoint
            if step > 0 and step % args.save_freq == 0:
                ckpt_dir = output_dir / f"checkpoint_{step:06d}"
                policy.save_pretrained(ckpt_dir)
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)
                save_train_state(ckpt_dir, optimizer=optimizer,
                                 step=step, best_val_loss=best_val_loss)
                print(f"  -> saved checkpoint to {ckpt_dir}")

            step += 1
            if step >= args.training_steps:
                done = True
                break

    # ---- Save final checkpoint ----
    # This saves the model weights, config, and processor pipelines.
    # The processors are self-contained (include normalization stats).
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    save_train_state(output_dir, optimizer=optimizer,
                     step=step, best_val_loss=best_val_loss)
    print(f"\nTraining complete. Model saved to {output_dir}")

    # ---- Push to HuggingFace Hub ----
    if args.push_to_hub is not None:
        final_repo = args.push_to_hub
        best_repo = f"{args.push_to_hub}-best"
        print("\nPushing to HuggingFace Hub:")
        print(f"  final checkpoint  -> {final_repo}")
        print(f"  best checkpoint   -> {best_repo}")

        # Push the final checkpoint
        policy.push_to_hub(final_repo, private=args.hub_private)
        preprocessor.push_to_hub(final_repo, private=args.hub_private)
        postprocessor.push_to_hub(final_repo, private=args.hub_private)

        # Push the best checkpoint (if it exists — training may have stopped before any eval)
        best_dir = output_dir / "best"
        if best_dir.exists():
            best_policy = DiffusionPolicy.from_pretrained(best_dir)
            best_pre, best_post = make_pre_post_processors(best_policy.config, pretrained_path=best_dir)
            best_policy.push_to_hub(best_repo, private=args.hub_private)
            best_pre.push_to_hub(best_repo, private=args.hub_private)
            best_post.push_to_hub(best_repo, private=args.hub_private)
            print(f"  best_val_loss:    {best_val_loss:.4f}")

        print("\nTo use on another machine:")
        print(f"  --checkpoint {best_repo}   (recommended, lowest val_loss)")
        print(f"  --checkpoint {final_repo}  (final step)")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
