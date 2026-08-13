"""Fit a custom FAST action tokenizer on a converted Grabette dataset.

Pi0-FAST predicts actions as discrete tokens produced by a FAST tokenizer
(DCT + BPE over action chunks). The stock tokenizers were fit on other
embodiments' action distributions; the LeRobot libero fine-tune — the one
verified working — used a tokenizer FIT ON ITS OWN DATASET
(`jadechoghari/tokenizer-lib-mean`). Do the same for Grabette actions.

Fits on MEAN_STD-normalized chunks (the same normalization the policy applies
before tokenizing), sampled as sliding windows over every episode.

Usage:
  uv run python fit_fast_tokenizer.py \\
      --dataset_repo_id <user>/<dataset>_cartesian [--dataset_root DIR] \\
      --output_dir ./fast_tokenizer_<task> \\
      [--chunk_size 10] [--push_repo <user>/fast_tokenizer_<task>]

After fitting, ALWAYS run verify_fast_tokenizer.py before training.
"""

import argparse

import numpy as np
from transformers import AutoProcessor

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# The lerobot mirror loads cleanly through AutoProcessor on transformers v5;
# the original `physical-intelligence/fast` repo does NOT (Hub packaging bug).
BASE_TOKENIZER = "lerobot/fast-action-tokenizer"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_repo_id", required=True, help="Converted (11D delta) dataset")
    p.add_argument("--dataset_root", default=None, help="Local dataset root (else HF cache/Hub)")
    p.add_argument("--chunk_size", type=int, default=10,
                   help="Action horizon the tokenizer is fit for — MUST equal the policy's chunk_size")
    p.add_argument("--stride", type=int, default=5, help="Sliding-window stride between training chunks")
    p.add_argument("--output_dir", required=True, help="Where to save the fitted tokenizer")
    p.add_argument("--push_repo", default=None, help="Optional Hub repo id to push the tokenizer to")
    args = p.parse_args()

    ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    stats = ds.meta.stats["action"]
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)

    frames = ds.hf_dataset.select_columns(["action", "episode_index"])
    A = np.stack([np.asarray(a, dtype=np.float64) for a in frames["action"]])
    ep = np.asarray(frames["episode_index"])
    print(f"{args.dataset_repo_id}: {len(np.unique(ep))} episodes, {len(A)} frames, "
          f"action dim {A.shape[1]}")

    # Sliding windows WITHIN episodes (never across an episode boundary),
    # MEAN_STD-normalized like the policy's NormalizerProcessorStep.
    chunks = []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        seq = (A[idx] - mean) / (std + 1e-8)
        for s in range(0, len(seq) - args.chunk_size + 1, args.stride):
            chunks.append(seq[s: s + args.chunk_size])
    chunks = np.stack(chunks)
    print(f"fitting on {len(chunks)} chunks of shape {chunks.shape[1:]}")

    tok = AutoProcessor.from_pretrained(BASE_TOKENIZER, trust_remote_code=True)
    tok = tok.fit(chunks)
    tok.save_pretrained(args.output_dir)
    print(f"saved fitted tokenizer to {args.output_dir}")

    if args.push_repo:
        tok.push_to_hub(args.push_repo)
        print(f"pushed to https://huggingface.co/{args.push_repo}")

    print("\nNext: verify the round-trip BEFORE training —")
    print(f"  uv run python verify_fast_tokenizer.py --tokenizer {args.push_repo or args.output_dir} \\")
    print(f"      --dataset_repo_id {args.dataset_repo_id} --action_horizon {args.chunk_size} "
          f"--action_dim {A.shape[1]}")


if __name__ == "__main__":
    main()
