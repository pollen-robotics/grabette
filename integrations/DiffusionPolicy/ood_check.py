"""Is the robot seeing what the policy was trained on? Encoder-feature OOD check.

A diffusion policy models p(action | observation) — it has no observation
likelihood, so "stereotyped behavior on the robot" (executing a dataset-average
motion regardless of the scene) usually means the DEPLOYMENT observation is
out-of-distribution for the policy's own vision encoder: the features carry no
scene information and the denoiser falls back on its prior. This tool tests that
directly, using the policy's own encoder:

  1. FIT      — encode training-episode frames (ResNet18+SpatialSoftmax, 64-D),
                fit a Gaussian → Mahalanobis distance.
  2. CALIBRATE— score the held-out val episodes: their distances define what
                "in-distribution" looks like (p50/p95/p99).
  3. SCORE    — score deployment frames (a --images dir dumped by
                evaluate.py --dump_obs) and compare against the val calibration.
                Also compares deployment observation.state ranges (state.jsonl)
                to the dataset's — catching unit/sign/offset mismatches.

`--self_test` corrupts val frames with the classic deployment bugs (BGR swap,
180° rotation, brightness shift) and scores them — verifying the detector
catches exactly the failure classes it exists for.

Usage:
  # on the robot: dump one episode's observations
  #   evaluate.py --dump_obs /tmp/deploy_obs --num_episodes 1 ...
  uv run python ood_check.py \\
      --checkpoint <user>/<model>-best \\
      --dataset_repo_id <user>/<dataset>_cartesian \\
      --images /tmp/deploy_obs/ep000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

from offline_eval import load_policy, val_episodes_from_checkpoint, val_episodes_like_train


# ---------------------------------------------------------------------------
# Feature extraction (the policy's own eyes)
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Images → the policy's rgb_encoder features, through the SAME
    preprocessing as inference (normalization pipeline + eval-time center crop,
    which the encoder applies internally in eval mode)."""

    def __init__(self, policy, pre, device: str):
        self.encoder = policy.diffusion.rgb_encoder
        self.pre = pre
        self.device = device
        self.mid_state = torch.zeros(1, 2)  # state doesn't affect image features

    @torch.no_grad()
    def __call__(self, images: list[torch.Tensor]) -> np.ndarray:
        """images: list of (C,H,W) float32 [0,1] RGB tensors → (N, D) features."""
        feats = []
        for i in range(0, len(images), 32):
            chunk = torch.stack(images[i:i + 32])
            batch = {
                "observation.images.cam0": chunk,
                "observation.state": self.mid_state.expand(len(chunk), -1),
                "task": "",
            }
            batch = self.pre(batch)
            feats.append(self.encoder(batch["observation.images.cam0"]).cpu().numpy())
        return np.concatenate(feats)


class Mahalanobis:
    """Gaussian fit on training features; distance in the encoder's own metric."""

    def __init__(self, feats: np.ndarray, ridge: float = 1e-3):
        self.mu = feats.mean(axis=0)
        cov = np.cov(feats.T) + ridge * np.eye(feats.shape[1])
        self.prec = np.linalg.inv(cov)

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        d = feats - self.mu
        return np.sqrt(np.einsum("ni,ij,nj->n", d, self.prec, d))


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------
def dataset_frames(ds: LeRobotDataset, stride: int) -> tuple[list[torch.Tensor], np.ndarray]:
    """Every `stride`-th frame (C,H,W float [0,1]) + all observation.state rows."""
    frames = [ds[i]["observation.images.cam0"] for i in range(0, len(ds), stride)]
    states = np.stack([np.asarray(ds[i]["observation.state"]) for i in range(0, len(ds), stride)])
    return frames, states


def image_dir_frames(dir_: Path) -> tuple[list[torch.Tensor], np.ndarray | None]:
    """PNGs from an evaluate.py --dump_obs dir (+ state.jsonl if present)."""
    import cv2
    paths = sorted(dir_.glob("*.png")) + sorted(dir_.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"No images found in {dir_}")
    frames = []
    for p in paths:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(img).float().permute(2, 0, 1) / 255.0)
    states = None
    sj = dir_ / "state.jsonl"
    if sj.is_file():
        states = np.array([json.loads(line)["state"] for line in sj.read_text().splitlines()])
    return frames, states


def corrupt(frames: list[torch.Tensor], kind: str) -> list[torch.Tensor]:
    """The classic deployment-bug corruptions, for --self_test."""
    if kind == "bgr_swap":
        return [f.flip(0) for f in frames]                     # RGB<->BGR
    if kind == "rot180":
        return [f.flip(1).flip(2) for f in frames]             # upside-down mount
    if kind == "dark":
        return [(f * 0.4) for f in frames]                     # exposure shift
    raise ValueError(kind)


# ---------------------------------------------------------------------------
def report(name: str, d: np.ndarray, val_p95: float, val_p99: float) -> None:
    frac = float((d > val_p95).mean())
    verdict = ("IN-DISTRIBUTION" if frac < 0.25
               else "SUSPECT" if frac < 0.75 else "OUT-OF-DISTRIBUTION")
    print(f"  {name:<22} p50 {np.percentile(d, 50):7.1f}   p95 {np.percentile(d, 95):7.1f}   "
          f">val_p95: {100 * frac:3.0f}%   → {verdict}")


def main():
    p = argparse.ArgumentParser(description="Encoder-feature OOD check for deployment observations")
    p.add_argument("--checkpoint", required=True, help="Trained checkpoint (local dir or Hub id)")
    p.add_argument("--dataset_repo_id", required=True, help="The dataset the checkpoint was trained on")
    p.add_argument("--dataset_root", default=None, help="Local dataset root (else HF cache/Hub)")
    p.add_argument("--images", default=None,
                   help="Dir of deployment frames to score (from evaluate.py --dump_obs)")
    p.add_argument("--self_test", action="store_true",
                   help="Score BGR-swapped / rotated / darkened val frames (detector sanity check)")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Reproduces train.py's split")
    p.add_argument("--val_split", choices=["stride", "tail"], default="stride",
                   help="Fallback split rule when the checkpoint has no val_episodes.json "
                        "('tail' for checkpoints trained before the strided split).")
    p.add_argument("--fit_stride", type=int, default=8, help="Fit on every Nth training frame")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if not args.images and not args.self_test:
        p.error("nothing to score: pass --images DIR and/or --self_test")

    policy = load_policy(args.checkpoint).to(args.device)
    policy.eval()
    from lerobot.policies.factory import make_pre_post_processors
    pre, _ = make_pre_post_processors(policy.config, args.checkpoint)
    extract = FeatureExtractor(policy, pre, args.device)

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    # Guard: fitting on a RAW dataset (8D actions, no state, extra cameras,
    # pre-rejection episodes) would score against the WRONG distribution — and
    # crash later with an opaque KeyError. Fail with the diagnosis.
    if "observation.state" not in meta.features or meta.features["action"]["shape"][0] != 11:
        raise SystemExit(
            f"\nERROR: '{args.dataset_repo_id}' looks like a RAW dataset "
            f"(action dim {meta.features['action']['shape'][0]}, "
            f"state {'present' if 'observation.state' in meta.features else 'MISSING'}).\n"
            f"This check must fit on the dataset the checkpoint TRAINED on — the converted\n"
            f"*_cartesian output of run_pipeline.sh. If it only exists on the training\n"
            f"machine, push it to the Hub from there, or run this check on that machine\n"
            f"with --dataset_root."
        )
    val_eps = val_episodes_from_checkpoint(args.checkpoint) \
        or val_episodes_like_train(meta.total_episodes, args.val_ratio, args.val_split)
    train_eps = [e for e in range(meta.total_episodes) if e not in val_eps]

    print(f"Fitting on {len(train_eps)} training episodes (stride {args.fit_stride})...")
    ds_train = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=train_eps)
    fit_frames, fit_states = dataset_frames(ds_train, args.fit_stride)
    dist = Mahalanobis(extract(fit_frames))
    print(f"  {len(fit_frames)} frames → 64-D features → Gaussian fit")

    print(f"Calibrating on val episodes {val_eps}...")
    ds_val = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=val_eps)
    val_frames, _ = dataset_frames(ds_val, max(1, args.fit_stride // 2))
    d_val = dist(extract(val_frames))
    val_p95, val_p99 = np.percentile(d_val, 95), np.percentile(d_val, 99)

    print(f"\n{'source':<24} {'p50':>10} {'p95':>12} {'frames > val_p95':>16}")
    print("-" * 72)
    report("val (calibration)", d_val, val_p95, val_p99)

    if args.self_test:
        base = val_frames[:: max(1, len(val_frames) // 120)]
        for kind in ("bgr_swap", "rot180", "dark"):
            report(f"self-test: {kind}", dist(extract(corrupt(base, kind))), val_p95, val_p99)

    if args.images:
        frames, states = image_dir_frames(Path(args.images))
        print(f"\nScoring {len(frames)} deployment frames from {args.images}")
        d_dep = dist(extract(frames))
        report("DEPLOYMENT images", d_dep, val_p95, val_p99)
        # Time course: a run that starts in-distribution and CLIMBS is live
        # covariate shift (the policy drifting off the demo manifold), not a
        # static camera/setup problem — very different fixes. 8 bins suffice.
        nb = min(8, len(d_dep))
        bins = np.array_split(d_dep, nb)
        print("  time course (episode split in 8): "
              + "  ".join(f"{np.mean(b):.1f}" for b in bins))
        if len(d_dep) >= 20 and np.mean(bins[-1]) > 1.5 * np.mean(bins[0]) \
                and np.mean(bins[0]) < 1.5 * val_p95:
            print("  ⚠ starts in-distribution and CLIMBS → covariate shift (the robot's own "
                  "motion drifts into views absent from the demos), not a camera/setup issue. "
                  "Fixes: recovery demos (guide Part C), tighter re-planning (--n_action_steps), "
                  "matched --fps.")
        # --- state parity (distribution-aware: a units/offset mismatch can sit
        #     entirely INSIDE the dataset range and still be extreme-tail OOD,
        #     e.g. the robot's "open" reading below 96% of training frames) ---
        if states is not None:
            print("\nSTATE PARITY (observation.state = [proximal, distal])")
            print(f"  {'':<10} {'p5':>16} {'p50':>16} {'p95':>16}   (dataset / deployment)")
            suspect = False
            for j, name in enumerate(["proximal", "distal"]):
                cells = [f"{np.percentile(fit_states[:, j], p):7.3f}/{np.percentile(states[:, j], p):7.3f}"
                         for p in (5, 50, 95)]
                print(f"  {name:<10} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")
                # where does the deployment MEDIAN sit inside the dataset distribution?
                tail = float((fit_states[:, j] <= np.median(states[:, j])).mean())
                if tail < 0.05 or tail > 0.95:
                    suspect = True
                    print(f"  ⚠ deployment {name} median sits at the {100 * tail:.0f}th percentile "
                          f"of the dataset — units / zero-calibration / scale mismatch likely "
                          f"(also check the ACTION-side gripper units).")
            if not suspect:
                print("  state distributions compatible ✓")

    print("\nReading: IN-DISTRIBUTION → the camera view is not your problem; look at "
          "control frame / fps / state. OUT-OF-DISTRIBUTION → fix the observation "
          "(view, orientation, color order, crop, exposure) before anything else.")


if __name__ == "__main__":
    main()
