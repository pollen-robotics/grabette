"""Offline open-loop sanity check of a trained policy against recorded episodes.

Replays held-out dataset episodes through the policy EXACTLY like deployment
(same `select_action` queueing / re-planning cadence, same pre/post processors,
eval-time center crop) but feeds the RECORDED observations instead of live ones,
and compares the predicted actions to the dataset's ground-truth deltas.

What it catches (the gross failures that waste a robot session):
  * normalization / frame bugs        → predictions wildly off-scale
  * mode collapse / averaging         → prediction std << data std
  * gripper never closing / mistimed  → close-initiation offset per episode
  * early trajectory divergence       → integrated-path overlay plots

What it CANNOT tell you: closed-loop success. The policy sees ground-truth
observations at every step, so compounding-error / covariate-shift failures are
invisible here. Treat a pass as "worth a robot session", not "it works".

By default it evaluates the SAME validation episodes train.py held out
(deterministic split: the last `val_ratio` fraction of episodes).

Usage:
  uv run python offline_eval.py \\
      --checkpoint <user>/<model>-best \\
      --dataset_repo_id <user>/<dataset>_cartesian [--dataset_root DIR]
  # specific episodes instead of the val split:
  uv run python offline_eval.py --checkpoint ... --dataset_repo_id ... --episodes 60 63 66
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from rotation import rotation_6d_to_rotation_matrix_numpy


def load_policy(checkpoint: str):
    """Load any LeRobot policy from a local dir or Hub id, dispatching on the
    `type` field of its config.json (same pattern as the deployment client)."""
    cfg_path = Path(checkpoint) / "config.json"
    if not cfg_path.is_file():
        from huggingface_hub import hf_hub_download
        cfg_path = Path(hf_hub_download(checkpoint, "config.json"))
    policy_type = json.loads(cfg_path.read_text())["type"]
    return get_policy_class(policy_type).from_pretrained(checkpoint)


def val_episodes_like_train(total: int, val_ratio: float, mode: str = "stride") -> list[int]:
    """Reproduce train.py's deterministic split.

    mode="stride" (current train.py): every Nth episode — val spread across the
    whole recording session, avoiding the correlated-consecutive-episodes trap.
    mode="tail" (legacy): the LAST max(1, ratio*N) episodes — for checkpoints
    trained before the strided split.
    """
    if mode == "tail":
        num_val = max(1, int(total * val_ratio))
        return list(range(total - num_val, total))
    stride = max(2, round(1 / val_ratio))
    return list(range(0, total, stride))


def val_episodes_from_checkpoint(checkpoint: str) -> list[int] | None:
    """The authoritative split: train.py saves val_episodes.json next to its
    checkpoints. If the checkpoint is local, read it (checking the dir and its
    parent, since --checkpoint may point at <output_dir>/best). Returns None for
    Hub checkpoints / older runs — caller falls back to the deterministic rule."""
    ckpt = Path(checkpoint)
    for candidate in (ckpt / "val_episodes.json", ckpt.parent / "val_episodes.json"):
        if candidate.is_file():
            return json.loads(candidate.read_text())["val_episodes"]
    return None


def integrate_deltas(deltas: np.ndarray) -> np.ndarray:
    """Integrate camera-local deltas [dx,dy,dz, r6d(6)] into an absolute path.

    Same composition the deployment integrator applies (pos += R @ dpos,
    R = R @ R_delta), starting from the identity pose — so predicted and
    ground-truth deltas are integrated identically and comparable in the same
    (arbitrary) frame.
    """
    pos = np.zeros(3)
    R = np.eye(3)
    path = np.zeros((len(deltas) + 1, 3))
    for t, d in enumerate(deltas):
        pos = pos + R @ d[:3]
        R = R @ rotation_6d_to_rotation_matrix_numpy(d[3:9].reshape(1, 6))[0]
        path[t + 1] = pos
    return path


def gripper_lag(pred: np.ndarray, gt: np.ndarray, max_lag: int = 60) -> tuple[int, float]:
    """Timing offset (steps) between predicted and GT gripper waveforms, via the
    peak of their normalized cross-correlation, plus the zero-lag correlation.

    Direction-agnostic (works whichever sign 'close' is on this gripper) and
    robust to transient dips, unlike a threshold-crossing test. lag > 0 means
    the prediction is LATE; corr ≈ 1 at lag 0 means the close/open transitions
    are reproduced on time.
    """
    p = pred - pred.mean()
    g = gt - gt.mean()
    denom = np.sqrt((p ** 2).sum() * (g ** 2).sum()) + 1e-9
    lags = range(-max_lag, max_lag + 1)
    xc = [np.sum(p[max(0, -l):len(p) - max(0, l)] * g[max(0, l):len(g) - max(0, -l)]) / denom
          for l in lags]
    best = int(np.argmax(xc))
    corr0 = float(np.sum(p * g) / denom)
    return list(lags)[best], corr0


def replay_episode(policy, pre, post, ds: LeRobotDataset, start: int, end: int,
                   device: str) -> tuple[np.ndarray, np.ndarray]:
    """Feed one episode's recorded observations through the deployment inference
    path; return (predicted, ground-truth) action arrays of shape (T, 11)."""
    policy.reset()  # fresh obs/action queues, like an episode start on the robot
    preds, gts = [], []
    for i in range(start, end):
        sample = ds[i]
        batch = {
            # dataset frames are already float32 CHW in [0,1]
            "observation.images.cam0": sample["observation.images.cam0"].unsqueeze(0).to(device),
            "observation.state": sample["observation.state"].unsqueeze(0).to(device),
            "task": sample.get("task", ""),
        }
        batch = pre(batch)
        with torch.no_grad():
            action = policy.select_action(batch)
        action = post(action)
        preds.append(action.squeeze(0).cpu().numpy())
        gts.append(np.asarray(sample["action"], dtype=np.float32))
    return np.stack(preds), np.stack(gts)


def episode_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Open-loop agreement metrics for one episode (11D delta actions)."""
    pos_p, pos_g = pred[:, :3], gt[:, :3]
    # cosine similarity of the position delta DIRECTION, where GT actually moves
    moving = np.linalg.norm(pos_g, axis=1) > 1e-4  # > 0.1 mm
    cos = np.sum(pos_p[moving] * pos_g[moving], axis=1) / (
        np.linalg.norm(pos_p[moving], axis=1) * np.linalg.norm(pos_g[moving], axis=1) + 1e-9
    )
    # gripper timing: cross-correlation lag of the proximal waveform
    lag, corr = gripper_lag(pred[:, 9], gt[:, 9])
    return {
        "mse_pos": float(np.mean((pos_p - pos_g) ** 2)),
        "mse_rot": float(np.mean((pred[:, 3:9] - gt[:, 3:9]) ** 2)),
        "mse_grip": float(np.mean((pred[:, 9:] - gt[:, 9:]) ** 2)),
        "cos_dpos": float(np.mean(cos)) if moving.any() else float("nan"),
        "mag_ratio": float(np.linalg.norm(pos_p, axis=1).mean()
                           / max(np.linalg.norm(pos_g, axis=1).mean(), 1e-9)),
        # averaging detector: predicted spread vs data spread (per-dim std, averaged)
        "std_ratio": float(np.mean(pred.std(axis=0) / np.maximum(gt.std(axis=0), 1e-6))),
        "grip_lag": lag,
        "grip_corr": corr,
    }


def plot_episode(pred: np.ndarray, gt: np.ndarray, ep: int, out_dir: Path) -> None:
    """3-panel overlay PNG: integrated path, per-step |Δpos|, gripper channels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping plots)")
        return
    path_p, path_g = integrate_deltas(pred), integrate_deltas(gt)
    fig = plt.figure(figsize=(15, 4.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(*path_g.T, label="GT", lw=2)
    ax.plot(*path_p.T, label="pred", lw=1.5, alpha=0.8)
    ax.scatter(*path_g[0], c="k", s=30)
    ax.set_title(f"ep {ep} — integrated path (m)")
    ax.legend()
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(np.linalg.norm(gt[:, :3], axis=1) * 1000, label="GT", lw=2)
    ax2.plot(np.linalg.norm(pred[:, :3], axis=1) * 1000, label="pred", lw=1, alpha=0.8)
    ax2.set_title("per-step |Δpos| (mm)")
    ax2.set_xlabel("step")
    ax2.legend()
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(gt[:, 9], label="prox GT", lw=2)
    ax3.plot(pred[:, 9], label="prox pred", lw=1, alpha=0.8)
    ax3.plot(gt[:, 10], label="dist GT", lw=2, ls="--")
    ax3.plot(pred[:, 10], label="dist pred", lw=1, ls="--", alpha=0.8)
    ax3.set_title("gripper (rad)")
    ax3.set_xlabel("step")
    ax3.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / f"offline_eval_ep{ep:03d}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  plot: {out}")


def main():
    p = argparse.ArgumentParser(description="Offline open-loop policy check on recorded episodes")
    p.add_argument("--checkpoint", required=True, help="Trained checkpoint (local dir or Hub id)")
    p.add_argument("--dataset_repo_id", required=True, help="Converted (11D delta) dataset")
    p.add_argument("--dataset_root", default=None, help="Local dataset root (else HF cache/Hub)")
    p.add_argument("--episodes", type=int, nargs="+", default=None,
                   help="Explicit episode indices. Default: the split saved next to the "
                        "checkpoint (val_episodes.json), else train.py's deterministic rule.")
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Val fraction used to reproduce train.py's split (default 0.1)")
    p.add_argument("--val_split", choices=["stride", "tail"], default="stride",
                   help="Split rule fallback when the checkpoint has no val_episodes.json: "
                        "'stride' = current train.py (every Nth episode); 'tail' = legacy "
                        "(last N) for checkpoints trained before the strided split.")
    p.add_argument("--max_episodes", type=int, default=None, help="Cap the number of episodes")
    p.add_argument("--device", default="cuda", help="Compute device")
    p.add_argument("--out_dir", default="offline_eval_out", help="Where the overlay PNGs go")
    p.add_argument("--no_plots", action="store_true", help="Metrics only, no PNGs")
    args = p.parse_args()

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    # Guard: a RAW dataset (8D actions, no observation.state) is not what the
    # checkpoint trained on — comparing against it is meaningless and crashes
    # later. Same guard as train.py / ood_check.py.
    if "observation.state" not in meta.features or meta.features["action"]["shape"][0] != 11:
        raise SystemExit(
            f"\nERROR: '{args.dataset_repo_id}' looks like a RAW dataset "
            f"(action dim {meta.features['action']['shape'][0]}, "
            f"state {'present' if 'observation.state' in meta.features else 'MISSING'}).\n"
            f"Point --dataset_repo_id at the converted *_cartesian dataset the checkpoint\n"
            f"was trained on (run_pipeline.sh output)."
        )
    split_src = "explicit --episodes"
    episodes = args.episodes
    if episodes is None:
        episodes = val_episodes_from_checkpoint(args.checkpoint)
        split_src = "val_episodes.json saved by train.py"
    if episodes is None:
        episodes = val_episodes_like_train(meta.total_episodes, args.val_ratio, args.val_split)
        split_src = f"deterministic rule ({args.val_split}, ratio {args.val_ratio})"
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    print(f"Dataset: {args.dataset_repo_id} ({meta.total_episodes} episodes)")
    print(f"Evaluating episodes: {episodes}  [{split_src}]")

    policy = load_policy(args.checkpoint).to(args.device)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, args.checkpoint)
    print(f"Policy: {type(policy).__name__} from {args.checkpoint}")

    ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=episodes)
    # frame ranges per episode within this (possibly re-indexed) dataset view
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_m = []
    hdr = (f"{'ep':>4} {'frames':>6} {'mse_pos':>9} {'cos_dpos':>8} {'mag_ratio':>9} "
           f"{'std_ratio':>9} {'grip_corr':>9} {'grip_lag':>8}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for k, ep in enumerate(episodes):
        idx = np.where(ep_col == ep)[0]
        if len(idx) == 0:  # datasets re-index episodes 0..N-1 after selection
            idx = np.where(ep_col == k)[0]
        pred, gt = replay_episode(policy, pre, post, ds, int(idx[0]), int(idx[-1]) + 1,
                                  args.device)
        m = episode_metrics(pred, gt)
        all_m.append(m)
        print(f"{ep:>4} {len(idx):>6} {m['mse_pos']:>9.2e} {m['cos_dpos']:>8.3f} "
              f"{m['mag_ratio']:>9.2f} {m['std_ratio']:>9.2f} {m['grip_corr']:>9.3f} "
              f"{m['grip_lag']:>+8d}")
        if not args.no_plots:
            plot_episode(pred, gt, ep, out_dir)

    print("\nSUMMARY (mean over episodes)")
    for key, good in [("cos_dpos", "→ 1 (direction agreement)"),
                      ("mag_ratio", "≈ 1 (speed calibration)"),
                      ("std_ratio", "≈ 1 («1 = averaging / mode collapse)"),
                      ("grip_corr", "→ 1 (gripper waveform reproduced)"),
                      ("grip_lag", "≈ 0 steps (>0 = closes LATE)")]:
        vals = [m[key] for m in all_m if np.isfinite(m[key])]
        print(f"  {key:>9}: {np.mean(vals):6.3f}   {good}")
    print("\nReminder: open-loop agreement is necessary, not sufficient — it can't see "
          "compounding errors. A pass here means 'worth a robot session'.")


if __name__ == "__main__":
    main()
