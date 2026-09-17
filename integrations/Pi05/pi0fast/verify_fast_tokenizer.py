"""Round-trip verification for a custom FAST action tokenizer.

Gate to run AFTER fitting a custom FAST tokenizer and BEFORE the (expensive)
Pi0-FAST training. Confirms that the tokenizer can faithfully encode→decode
the dataset's actions under the SAME normalization the policy uses (MEAN_STD),
so a correctly-predicted token sequence will detokenize back to the right
action.

If reconstruction error is tiny, the tokenizer is sound and the training is
worth it. If it's large, the tokenizer/normalization is still wrong and you'd
just waste a training run.

Usage:
  uv run python verify_fast_tokenizer.py \\
      --tokenizer <user>/fast_tokenizer_<task> \\
      --dataset_repo_id <user>/<dataset>_cartesian [--dataset_root DIR] \\
      --action_horizon 10 --action_dim 11
"""

import argparse

import numpy as np
from transformers import AutoProcessor

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="Custom FAST tokenizer repo/dir")
    p.add_argument("--dataset_repo_id", required=True, help="Dataset repo id")
    p.add_argument("--dataset_root", default=None, help="Local dataset root (else HF cache/Hub)")
    p.add_argument("--action_horizon", type=int, default=10,
                   help="= chunk_size used for the tokenizer/policy")
    p.add_argument("--action_dim", type=int, default=11,
                   help="= output ACTION dim the policy detokenizes")
    p.add_argument("--n_chunks", type=int, default=20, help="How many chunks to test")
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoProcessor.from_pretrained(args.tokenizer, trust_remote_code=True)
    ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)

    # MEAN_STD stats from the dataset (must match the policy's ACTION normalization).
    stats = ds.meta.stats["action"]
    mean = np.asarray(stats["mean"], dtype=np.float64)[: args.action_dim]
    std = np.asarray(stats["std"], dtype=np.float64)[: args.action_dim]

    # Pull contiguous action chunks straight from the dataset's parquet so we
    # get the raw (un-normalized) action sequence, then normalize it ourselves
    # the same way the policy's NormalizerProcessorStep (MEAN_STD) would.
    frames = ds.hf_dataset.select_columns("action")
    A = np.stack([np.asarray(a, dtype=np.float64) for a in frames["action"]])
    H = args.action_horizon

    errs = []
    starts = np.linspace(0, len(A) - H - 1, args.n_chunks).astype(int)
    for s in starts:
        chunk = A[s: s + H, : args.action_dim]                     # [H, D] raw
        norm = (chunk - mean) / (std + 1e-8)                       # MEAN_STD normalized
        tokens = tok(norm[None])[0]                                # encode
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        recon = tok.decode([tokens], time_horizon=H, action_dim=args.action_dim)[0]
        recon = np.asarray(recon, dtype=np.float64)
        err = np.abs(recon - norm)
        errs.append(err)
        if s == starts[0]:
            print("first chunk, step0 — normalized GT :", np.round(norm[0], 3))
            print("first chunk, step0 — reconstructed  :", np.round(recon[0], 3))

    errs = np.stack(errs)  # [n_chunks, H, D]
    mae = errs.mean()
    p95 = np.percentile(errs, 95)
    print("\nRound-trip reconstruction error (normalized space):")
    print(f"  mean abs err : {mae:.5f}")
    print(f"  p95  abs err : {p95:.5f}")
    print(f"  per-dim MAE  : {np.round(errs.mean(axis=(0, 1)), 5)}")
    verdict = "GOOD — tokenizer faithful, proceed to training" if mae < 0.05 else (
        "MARGINAL — bump vocab_size or check normalization" if mae < 0.15 else
        "BAD — tokenizer/normalization still wrong, do NOT train yet")
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
