"""Generation health smoke for a VLA checkpoint (pi05 / pi0_fast / pi0) —
run BEFORE any robot session.

Training loss — even a clean held-out eval loss — cannot detect a policy
whose generation ignores its observations: a pi0fast fine-tune of ours had
healthy losses while emitting ONE constant action for every input, and
autoregressive decoders can additionally produce degenerate text (unicode
garbage, <bos> loops). This gate feeds the checkpoint REAL observations from
its own training dataset and checks that:

  1. generation is WELL-FORMED (parses/decodes; finite, sane-scale actions),
  2. outputs are INPUT-DEPENDENT (different frames → different actions;
     the collapsed pi0fast reference measured a 0.000000 pairwise diff),
  3. predictions roughly track the dataset's ground-truth actions.

Usage (pi05 — always use --fp32: the port has a bf16 flow-path dtype clash):
  uv run python smoke_generation.py \\
      --checkpoint <user>/<model> --policy_type pi05 --fp32 \\
      --dataset_repo_id <user>/<dataset>_cartesian [--dataset_root DIR] \\
      [--episodes 0 80] [--frame 60] [--task "pick up the red can"]
"""

import argparse

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_repo_id", required=True)
    p.add_argument("--dataset_root", default=None)
    p.add_argument("--episodes", type=int, nargs=2, default=[0, 5],
                   help="Two episodes to probe (different scenes)")
    p.add_argument("--frame", type=int, default=10, help="Frame index within each episode")
    p.add_argument("--task", default="pick", help="Task string (must match training)")
    p.add_argument("--policy_type", default="pi0_fast", choices=["pi0_fast", "pi05", "pi0"],
                   help="Policy class of the checkpoint")
    p.add_argument("--fp32", action="store_true",
                   help="Load in float32 (the pi05 port has a bf16 dtype clash in "
                        "its flow path — use this on cards with >=16GB)")
    args = p.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = "cpu"  # load on CPU first, then cast + move
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = False  # deployment optimization; irrelevant to the gate
    policy = get_policy_class(args.policy_type).from_pretrained(args.checkpoint, config=cfg)
    policy = policy.to(dtype=torch.float32 if args.fp32 else torch.bfloat16).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)
    policy.config.device = device
    pre, post = make_pre_post_processors(policy.config, args.checkpoint)
    cams = [k for k in policy.config.input_features if "image" in k]
    print(f"loaded {args.checkpoint} on {device} | cameras {cams} | "
          f"chunk {policy.config.chunk_size}")

    outs = []
    for ep in args.episodes:
        ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=[ep])
        item = ds[min(args.frame, len(ds) - 1)]
        batch = {"task": args.task}
        for k in cams:
            # our datasets have ONE camera (cam0); empty_cameras slots are
            # zero-filled, matching training's empty-camera padding.
            batch[k] = (item[k] if k in item else torch.zeros_like(item[cams[0]])
                        ).unsqueeze(0).to(device)
        batch["observation.state"] = torch.as_tensor(
            np.asarray(item["observation.state"], dtype=np.float32)).unsqueeze(0).to(device)
        gt = np.asarray(item["action"], dtype=np.float32)
        policy.reset()
        try:
            with torch.no_grad():
                a = post(policy.select_action(pre(batch))).squeeze(0).float().cpu().numpy()
            outs.append(a)
            print(f"ep {ep} frame {args.frame}:")
            print(f"   pred = {np.round(a, 4)}")
            print(f"   GT   = {np.round(gt[: len(a)], 4)}")
        except AssertionError as e:
            print(f"ep {ep}: DEGENERATE GENERATION — {str(e)[:200]}")
            outs.append(None)
        except Exception as e:
            print(f"ep {ep}: ERROR — {type(e).__name__}: {str(e)[:250]}")
            outs.append(None)

    print()
    if all(o is not None for o in outs):
        diff = float(np.abs(np.asarray(outs[0]) - np.asarray(outs[1])).mean())
        print(f"mean |a(ep{args.episodes[0]}) - a(ep{args.episodes[1]})| = {diff:.6f}")
        if diff > 1e-6:
            print("VERDICT: PASS — well-formed, input-dependent generation. "
                  "Worth a robot session (open-loop only; run the DiffusionPolicy "
                  "gates offline_eval/ood_check for the rest).")
        else:
            print("VERDICT: SUSPICIOUS — well-formed but input-INDEPENDENT; "
                  "the model may be ignoring observations.")
    else:
        print("VERDICT: FAIL — degenerate generation; do NOT deploy. "
              "Check train/infer lerobot versions match (see README).")


if __name__ == "__main__":
    main()
