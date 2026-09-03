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


def report_execution_quality(pred, gt_rel, n_exec):
    """What the ROBOT gets, which is not what check 3 measures.

    Execution needs per-step motion, obtained by differencing consecutive chunk
    offsets. Differencing is not free: d_i = a_i - a_{i-1} and d_{i+1} =
    a_{i+1} - a_i share the term a_i with OPPOSITE sign, so white error on the
    offsets drives the lag-1 autocorrelation of the executed deltas to -0.5 and
    shows up as lateral jitter. A chunk can therefore have a small mean offset
    error (check 3 passes) while the motion derived from it is noise.

    Measured on hardware for reference: x travelled 40 mm of path for 0.5 mm of
    net displacement (straightness 0.01), lag-1 autocorr -0.10/-0.32/-0.28 on
    x/y/z. This tells us offline whether that is expected.
    """
    def lag1(x):
        """Lag-1 autocorrelation, or None when the series is too flat to have one.

        A near-constant axis has no meaningful autocorrelation: after subtracting
        the mean only float rounding is left, which returns a spurious value
        (a perfect constant-velocity axis measured -0.48 in testing, exactly the
        differencing-noise signature). Report nothing rather than a false alarm.
        """
        x = np.asarray(x, float)
        c = x - x.mean()
        denom = float((c * c).sum())
        if denom <= 0 or c.std() < 1e-4 * max(abs(x.mean()), 1e-9) + 1e-12:
            return None
        return float((c[:-1] * c[1:]).sum() / denom)

    m = min(len(pred), len(gt_rel))
    dp, dg = np.diff(pred[:m, :3], axis=0), np.diff(gt_rel[:m, :3], axis=0)
    print("   -- executed deltas (what the arm actually receives), mm --")
    print(f"   {'axis':>6} {'demo/step':>10} {'pred/step':>10} {'scatter':>9} "
          f"{'straightness':>13} {'lag-1 ac':>9}")
    for i, ax in enumerate("xyz"):
        net, path = abs(dp[:, i].sum()), np.abs(dp[:, i]).sum()
        ac = lag1(dp[:, i])
        print(f"   {ax:>6} {np.abs(dg[:, i]).mean() * 1000:10.2f} "
              f"{np.abs(dp[:, i]).mean() * 1000:10.2f} {dp[:, i].std() * 1000:9.2f} "
              f"{(net / path if path else 0):13.2f} "
              f"{'     flat' if ac is None else f'{ac:9.2f}'}")
    err = np.linalg.norm(dp - dg, axis=1) * 1000
    true = np.linalg.norm(dg, axis=1) * 1000
    snr = (true.mean() / err.mean()) if err.mean() > 1e-9 else float("inf")
    print(f"   per-step error {err.mean():5.2f} mm against {true.mean():5.2f} mm "
          f"of true motion -> SNR "
          + ("exact" if snr == float("inf") else f"{snr:.2f}"))
    print("   (straightness ~0, or lag-1 ac toward -0.5, means differencing "
          "noise rather than motion)")

    if n_exec < len(pred):
        de = np.diff(pred[:n_exec, :3], axis=0)
        print(f"   over the {n_exec} actions --n_action_steps would execute: "
              f"net {np.linalg.norm(de.sum(axis=0)) * 1000:.1f} mm of "
              f"{np.abs(de).sum() * 1000:.1f} mm travelled")


def report_gripper_schedule(pred, gt_rel, n_exec):
    """WHEN in the chunk does the policy close? Decides --n_action_steps.

    On the robot, closure reached ~1.0 only at chunk index 14 with
    --n_action_steps 15 — the last action executed, twice out of two — so the
    close command was issued once and then discarded at every replan. If the
    ramp starts after n_exec, executing more of the chunk is the fix; if it
    starts inside it, something else is wrong.
    """
    close = pred[:, -1]
    gt_close = gt_rel[:min(len(gt_rel), len(pred)), -1]

    def first_above(v, thr=0.9):
        idx = np.where(np.asarray(v) >= thr)[0]
        return int(idx[0]) if len(idx) else None

    fp, fg = first_above(close), first_above(gt_close)
    print("   -- gripper closure across the chunk --")
    step = max(1, len(close) // 10)
    print("   idx :  " + " ".join(f"{i:5d}" for i in range(0, len(close), step)))
    print("   pred:  " + " ".join(f"{close[i]:5.2f}" for i in range(0, len(close), step)))
    print("   GT  :  " + " ".join(
        f"{gt_close[i]:5.2f}" if i < len(gt_close) else "    -"
        for i in range(0, len(close), step)))
    print(f"   first closure >= 0.9 : pred index {fp}, GT index {fg}")
    if fp is None:
        print("   -> no close anywhere in this chunk (frame is before the grasp?)")
    elif fp >= n_exec:
        print(f"   -> !! the close is at index {fp} but --n_action_steps only "
              f"executes 0..{n_exec - 1}: the grasp command is DISCARDED every "
              f"replan. Raise --n_action_steps above {fp}.")
    else:
        print(f"   -> the close falls inside the executed window (0..{n_exec - 1}).")


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
                        "with it and compares the movement against a same-task "
                        "re-sampling NOISE FLOOR (pi05 is a flow model, so the "
                        "same input does not give the same output twice). The "
                        "task-conditioning check, which otherwise needs "
                        "probe_task_sensitivity.py and a running ficelle server.")
    p.add_argument("--policy_type", default="pi0_fast", choices=["pi0_fast", "pi05", "pi0"],
                   help="Policy class of the checkpoint")
    p.add_argument("--chunk_relative", action="store_true",
                   help="Checkpoint was trained with chunk-relative actions "
                        "(grabette-chunkrel). See the module docstring.")
    p.add_argument("--n_action_steps", type=int, default=15,
                   help="How many of the chunk the EVAL loop would execute before "
                        "replanning (evaluate.py's --n_action_steps). Only used to "
                        "judge the diagnostics: whether the policy's close command "
                        "falls inside the executed window, and how much real "
                        "displacement those actions produce.")
    p.add_argument("--task_samples", type=int, default=3,
                   help="Draws per condition for --task2 (default 3). pi05 is a "
                        "flow model, so both the noise floor and the task effect "
                        "are random variables; one draw each gives a ratio too "
                        "noisy to act on. Costs 2N forward passes.")
    p.add_argument("--fp32", action="store_true",
                   help="Load in float32 (the pi05 port has a bf16 dtype clash "
                        "in its flow path). Our pi05 checkpoints are 4.14B "
                        "params, so fp32 is 16.6GB of WEIGHTS before "
                        "activations: use a >=24GB card. A 16GB card OOMs.")
    args = p.parse_args()

    if args.chunk_relative:
        # Importing this module is what REGISTERS the steps: the
        # @ProcessorStepRegistry.register decorators run at import time, and
        # make_pre_post_processors() below resolves the checkpoint's step NAMES
        # against that registry. Merely depending on the package is not enough,
        # and importing grabette_chunkrel.chunk_relative (the maths) is not
        # either — the decorators live here. Without it lerobot raises a bare
        # KeyError naming every step except ours.
        try:
            import grabette_chunkrel.chunk_relative_processor  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "This checkpoint's processor pipeline needs the grabette-chunkrel "
                f"package ({e}).\\n"
                "Install it where the policy loads:\\n"
                "  uv pip install -e packages/grabette-chunkrel\\n"
                "or, standalone:\\n"
                "  uv run --with 'grabette-chunkrel @ git+https://github.com/"
                "pollen-robotics/grabette#subdirectory=packages/grabette-chunkrel' ..."
            ) from e

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
                report_execution_quality(a, gt_rel, args.n_action_steps)
                report_gripper_schedule(a, gt_rel, args.n_action_steps)
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
            # Same pixels, same state, different sentence. On a multi-task dataset
            # the prediction has to move, or the policy is ignoring the language
            # and will pick whatever object it likes.
            #
            # It must be measured against a NOISE FLOOR, not against zero: pi05 is
            # a flow model, so re-sampling the SAME input already moves the output.
            # Our 3-task pi05 measured a task swap of 0.0047 against a 0.0036
            # same-task floor — 1.3x, i.e. nothing — and reporting that 0.0047 as
            # "task-sensitive" is exactly the false pass this floor prevents.
            try:
                base = np.asarray(outs[0])

                def mean_abs_diff(batch):
                    return float(np.abs(base - np.asarray(comparable(predict(batch)))).mean())

                # Averaged over several draws: a ratio of two single samples is
                # itself noisy, and this number decides whether the instruction
                # can be trusted.
                n = max(1, args.task_samples)
                b2 = dict(first_batch)
                b2["task"] = args.task2
                floors = [mean_abs_diff(dict(first_batch)) for _ in range(n)]
                tdiffs = [mean_abs_diff(dict(b2)) for _ in range(n)]
                floor, tdiff = float(np.mean(floors)), float(np.mean(tdiffs))

                print()
                print(f"task probe on ep{args.episodes[0]} frame {args.frame} "
                      f"({n} draw{'s' if n > 1 else ''} each):")
                print(f"   same task, re-sampled  -> {floor:.6f}   (noise floor)"
                      f"   [{min(floors):.6f}..{max(floors):.6f}]")
                print(f"   \"{args.task}\"")
                print(f"     vs \"{args.task2}\" -> {tdiff:.6f}"
                      f"   [{min(tdiffs):.6f}..{max(tdiffs):.6f}]")
                print(f"   scene effect (ep{args.episodes[0]} vs "
                      f"ep{args.episodes[1]}) -> {diff:.6f}")
                snr = tdiff / floor if floor > 0 else float("inf")
                print(f"   task effect / noise floor: {snr:.2f}x")
                if min(tdiffs) < max(floors):
                    print("   (ranges overlap — treat the ratio as indicative only)")
                if snr < 1.5:
                    print("   TASK VERDICT: FAIL — the task swap is within sampling "
                          "noise. The language channel is not read; the policy will "
                          "grab its favourite object whatever you ask.")
                elif snr < 3.0:
                    print("   TASK VERDICT: WEAK — task effect is only just above "
                          "the noise floor. Do not rely on the instruction to "
                          "select the object.")
                else:
                    print("   TASK VERDICT: PASS — the prediction is task-sensitive "
                          "well beyond sampling noise.")
            except Exception as e:
                print(f"task probe: ERROR — {type(e).__name__}: {str(e)[:200]}")
    else:
        print("VERDICT: FAIL — degenerate generation; do NOT deploy. "
              "Check train/infer lerobot versions match (see README).")


if __name__ == "__main__":
    main()
