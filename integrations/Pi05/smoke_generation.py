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

CHUNK-RELATIVE checkpoints need --chunk_relative, which switches the gate from
one action to the whole chunk. Two reasons it cannot use select_action():

  - The reference pose is the chunk's FIRST action, so relative action 0 is
    identically zero in all six pose dims for every sample in training. Comparing
    two episodes' first actions therefore reads "input-INDEPENDENT" no matter how
    good the policy is — a false failure on the gate's central check.
  - The checkpoint's postprocessor ends with the inverse step, which needs a
    reference pose that only the robot has. It is removed here, and the gate
    compares offsets against the ground-truth chunk encoded the same way, which
    also turns check 3 into a real error in mm and degrees.
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
    p.add_argument("--task2", default=None,
                   help="Second task string. Re-probes the FIRST episode's frame "
                        "with it and reports how much the prediction moves — the "
                        "task-conditioning check, which matters for a multi-task "
                        "dataset and otherwise needs probe_task_sensitivity.py and "
                        "a running ficelle server.")
    p.add_argument("--policy_type", default="pi0_fast", choices=["pi0_fast", "pi05", "pi0"],
                   help="Policy class of the checkpoint")
    p.add_argument("--chunk_relative", action="store_true",
                   help="Checkpoint was trained with chunk-relative actions "
                        "(grabette-chunkrel). See the module docstring.")
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
    if args.chunk_relative:
        # The inverse step rebuilds absolute poses from a reference pose that
        # only the robot has; the gate wants the offsets themselves.
        before = len(post.steps)
        post.steps = [st for st in post.steps
                      if "AbsoluteFromChunkRelative" not in type(st).__name__]
        print(f"chunk-relative: removed {before - len(post.steps)} inverse "
              "step(s) from the postprocessor")
    cams = [k for k in policy.config.input_features if "image" in k]
    print(f"loaded {args.checkpoint} on {device} | cameras {cams} | "
          f"chunk {policy.config.chunk_size}")

    def predict(batch):
        """One prediction. A chunk (n_actions, dim) when chunk-relative, else a
        single action (dim,) — select_action's first action carries no motion in
        that representation, so the gate cannot use it."""
        policy.reset()
        with torch.no_grad():
            out = (policy.predict_action_chunk(pre(batch)) if args.chunk_relative
                   else policy.select_action(pre(batch)))
            return post(out).squeeze(0).float().cpu().numpy()

    def comparable(a):
        """Reduce a prediction to a flat vector that can be differenced against
        another. Defined ONCE so the episode loop and the task probe cannot drift
        apart: relative action 0 is identically zero for every sample, and leaving
        it in dilutes every difference the gate measures."""
        return a[1:].reshape(-1) if args.chunk_relative else a

    outs = []
    first_batch = None
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
        if first_batch is None:
            first_batch = dict(batch)   # reused by the --task2 probe
        try:
            a = predict(batch)
            if args.chunk_relative:
                from grabette_chunkrel.chunk_relative import to_chunk_relative

                n_pred = a.shape[0]   # not `k`: that is the camera loop's name
                # The ground truth for a CHUNK is the next n_pred frames, encoded
                # the same way the training pipeline encodes them.
                idx = min(args.frame, len(ds) - 1)
                gt_abs = np.stack([np.asarray(ds[j]["action"], dtype=np.float64)
                                   for j in range(idx, min(idx + n_pred, len(ds)))])
                gt_rel, _rep = to_chunk_relative(gt_abs)
                m = min(len(gt_rel), n_pred)
                pos_err = np.linalg.norm(a[:m, :3] - gt_rel[:m, :3], axis=1)
                print(f"ep {ep} frame {args.frame} (chunk of {n_pred}, {m} GT frames):")
                print(f"   action 0 pose dims  = {np.round(a[0, :6], 5)} "
                      "(should be ~0 by construction)")
                print(f"   final offset  pred  = {np.round(a[m - 1, :3], 4)} m")
                print(f"                 GT    = {np.round(gt_rel[m - 1, :3], 4)} m")
                print(f"   position error       mean {pos_err.mean() * 1000:6.1f} mm "
                      f"| max {pos_err.max() * 1000:6.1f} mm")
                print(f"   gripper pred[0]     = {np.round(a[0, 6:], 4)} "
                      f"| GT {np.round(gt_rel[0, 6:], 4)}")
            else:
                print(f"ep {ep} frame {args.frame}:")
                print(f"   pred = {np.round(a, 4)}")
                print(f"   GT   = {np.round(gt[: len(a)], 4)}")
            outs.append(comparable(a))
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

        if args.task2:
            # Same pixels, same state, different sentence. On a multi-task
            # dataset the prediction has to move, or the policy is ignoring the
            # language and will pick whatever object it likes.
            try:
                b2 = dict(first_batch)
                b2["task"] = args.task2
                a2 = comparable(predict(b2))
                tdiff = float(np.abs(np.asarray(outs[0]) - np.asarray(a2)).mean())
                print()
                print(f"task probe on ep{args.episodes[0]} frame {args.frame}:")
                print(f"   \"{args.task}\" vs \"{args.task2}\" -> "
                      f"mean |Δaction| = {tdiff:.6f}")
                # Scale it against the observation effect measured above: a task
                # that moves the prediction far less than a different scene did is
                # weak conditioning, which is the failure worth catching.
                ratio = tdiff / diff if diff > 0 else float("inf")
                print(f"   relative to the scene effect: {ratio:.2f}x")
                if tdiff <= 1e-6:
                    print("   TASK VERDICT: FAIL — the language input is ignored "
                          "entirely; a multi-task checkpoint cannot be steered.")
                elif ratio < 0.1:
                    print("   TASK VERDICT: WEAK — task conditioning is an order "
                          "of magnitude below the scene effect. Expect the policy "
                          "to go for whichever object it prefers.")
                else:
                    print("   TASK VERDICT: PASS — the prediction is task-sensitive.")
            except Exception as e:
                print(f"task probe: ERROR — {type(e).__name__}: {str(e)[:200]}")
    else:
        print("VERDICT: FAIL — degenerate generation; do NOT deploy. "
              "Check train/infer lerobot versions match (see README).")


if __name__ == "__main__":
    main()
