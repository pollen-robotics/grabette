"""Does the policy actually USE the image? Channel-specific vision probes.

A policy can score perfectly on every offline metric (val loss, grip_corr,
OOD checks) while ignoring the camera: on smooth demos, "continue the current
velocity and close on schedule" predicts the training actions almost as well
as looking. This tool measures image sensitivity DIRECTLY, on the channels
that matter, by feeding the trained policy controlled image swaps and
comparing the predicted action chunks.

IMPORTANT LESSON (mustard campaign, 2026-07): a naive swap test can lie.
Swapping in another episode's mid-approach frame, or mirroring the image,
showed ~zero action change — "the model is blind!" — but if the recording
protocol aims the camera at the object before approaching, the object is
centered in EVERY mid-approach frame and the correct answer for any of them
IS the same. A seeing model also shows no change. Probe channels where the
correct answer genuinely depends on the image:

  1. STOP-SWAP  — feed a PRE-GRASP image pair while at a mid-approach state,
                  and vice versa. A seeing model brakes and starts closing on
                  the pre-grasp image, and resumes approaching on the
                  mid-approach image. A phase/velocity-driven model ignores
                  the swap. This is the close-trigger channel.
  2. PIXEL-SHIFT — roll the image horizontally (moves the object in frame);
                  the predicted lateral motion (dx) and pan rotation should
                  follow the shift monotonically, with a gain comparable to
                  the forward speed. A weak gain (e.g. 10x slower lateral
                  than forward) means the model sees the offset but cannot
                  correct it before flying past the object — a data problem:
                  demos never demanded lateral correction.

Usage:
  uv run python vision_check.py \\
      --checkpoint <user>/<model>-best \\
      --dataset_repo_id <user>/<dataset>_cartesian \\
      [--episode 1] [--n_samples 3]

Verdict guide:
  stop-swap: swapped-in pre-grasp image should cut forward speed by >2x AND
             raise the gripper command; the reverse swap should restore
             approach speed. If neither budges -> close trigger is blind.
  pixel-shift: |dx| response at +/-60 px should be a substantial fraction of
             the forward speed. Below ~20% -> lateral servoing too weak to
             recover from aim error; record correction-rich demos.
"""

import argparse

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

from offline_eval import load_policy


def get_chunk(policy, pre, post, device, img_prev, img_now, state, n_samples):
    """Predicted action chunk for an observation pair, averaged over a few
    diffusion samples to suppress sampling noise."""
    chunks = []
    for _ in range(n_samples):
        policy.reset()
        with torch.no_grad():
            for img in (img_prev, img_now):
                batch = {
                    "observation.state": torch.as_tensor(state).float().unsqueeze(0).to(device),
                    "observation.images.cam0": img.unsqueeze(0).to(device),
                    "task": "pick",
                }
                a0 = policy.select_action(pre(batch))
        acts = [post(a0)]
        q = policy._queues["action"]
        while len(q):
            acts.append(post(q.popleft()))
        chunks.append(np.stack([a.squeeze(0).float().cpu().numpy() for a in acts]))
    return np.mean(np.stack(chunks), axis=0)


def describe(tag, chunk):
    dz = chunk[:, 2].mean() * 1000
    speed = np.linalg.norm(chunk[:, :3], axis=1).mean() * 1000
    grip = np.maximum(chunk[:, 9], chunk[:, 10])
    print(f"  {tag:36s} fwd dz {dz:6.2f} mm/step  |dp| {speed:5.2f}  "
          f"grip start {grip[0]:.3f} end {grip[-1]:.3f}")
    return dz, grip[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_repo_id", required=True)
    parser.add_argument("--dataset_root", default=None,
                        help="Local dataset root (for a locally-converted dataset)")
    parser.add_argument("--episode", type=int, default=1, help="episode to take probe frames from")
    parser.add_argument("--n_samples", type=int, default=3, help="diffusion samples averaged per probe")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy(args.checkpoint).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, args.checkpoint)

    def chunk(img_prev, img_now, state):
        return get_chunk(policy, pre, post, device, img_prev, img_now, state, args.n_samples)

    ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=[args.episode])
    n = len(ds)
    acts = np.stack([np.asarray(ds[i]["action"]) for i in range(n)])

    # locate the close from the gripper command signal (adaptive threshold,
    # same recipe as trim_release.py)
    sig = np.maximum(acts[:, 9], acts[:, 10])
    p10, p98 = np.percentile(sig, 10), np.percentile(sig, 98)
    if p98 - p10 < 0.08:
        raise SystemExit(f"episode {args.episode} shows no close (gripper range "
                         f"{p98 - p10:.3f}) — pick another with --episode")
    thr = p10 + 0.5 * (p98 - p10)
    t_close = int(np.where(sig > thr)[0][0])
    t_mid = t_close // 2
    t_pre = max(t_close - 6, 2)
    print(f"episode {args.episode}: {n} frames, close starts t={t_close} "
          f"(probing mid={t_mid}, pre-grasp={t_pre})")

    frame = lambda t: ds[t]["observation.images.cam0"]
    state = lambda t: np.asarray(ds[t]["observation.state"])

    print("\n== 1. STOP-SWAP (close-trigger channel) ==")
    print("at MID-APPROACH state:")
    dz_base, grip_base = describe("mid image (baseline)", chunk(frame(t_mid), frame(t_mid + 1), state(t_mid)))
    dz_swap, grip_swap = describe("PRE-GRASP image swapped in", chunk(frame(t_pre), frame(t_pre + 1), state(t_mid)))
    print("at PRE-GRASP state:")
    dz_pre, _ = describe("pre-grasp image (baseline)", chunk(frame(t_pre), frame(t_pre + 1), state(t_pre)))
    dz_rev, _ = describe("MID-APPROACH image swapped in", chunk(frame(t_mid), frame(t_mid + 1), state(t_pre)))

    brakes = abs(dz_base) > 2 * abs(dz_swap) and grip_swap > grip_base + 0.02
    resumes = abs(dz_rev) > 2 * abs(dz_pre)
    print(f"  -> pre-grasp image brakes+closes: {'YES' if brakes else 'NO'};"
          f" mid image resumes approach: {'YES' if resumes else 'NO'}")

    print("\n== 2. PIXEL-SHIFT (lateral-servoing channel) ==")
    responses, fwd = [], []
    for shift in (-60, -30, 0, 30, 60):
        f0 = torch.roll(frame(t_mid), shifts=shift, dims=-1)
        f1 = torch.roll(frame(t_mid + 1), shifts=shift, dims=-1)
        c = chunk(f0, f1, state(t_mid))
        dx, dz = c[:8, 0].mean() * 1000, c[:8, 2].mean() * 1000
        responses.append(dx)
        fwd.append(abs(dz))
        print(f"  shift {shift:+4d}px: dx {dx:+6.2f} mm/step   dz {dz:+6.2f}")
    lateral_span = max(responses) - min(responses)
    fwd_speed = float(np.median(fwd))
    gain_pct = 100 * lateral_span / max(fwd_speed, 1e-6)
    monotonic = all(b >= a - 0.05 for a, b in zip(responses, responses[1:])) or \
        all(b <= a + 0.05 for a, b in zip(responses, responses[1:]))
    print(f"  -> lateral response span {lateral_span:.2f} mm/step over 120px "
          f"({gain_pct:.0f}% of forward speed {fwd_speed:.2f}); "
          f"monotonic: {'YES' if monotonic else 'NO'}")

    print("\n== VERDICT ==")
    print(f"  close trigger sees the scene : {'PASS' if brakes and resumes else 'FAIL'}")
    lateral_ok = monotonic and gain_pct >= 20
    print(f"  lateral servoing gain        : "
          f"{'PASS' if lateral_ok else 'WEAK — record correction-rich demos (imperfect aim, corrected mid-approach)'}")


if __name__ == "__main__":
    main()
