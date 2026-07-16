# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "lerobot[dataset] @ git+https://github.com/huggingface/lerobot@e40b58a8dfa9e7b86918c374791599d070518d11",
#   "av",
#   "ficelle-client[iroh] @ git+https://github.com/SteveNguyen/Ficelle#subdirectory=client",
# ]
# ///
"""Language-channel gate: does the fine-tune actually READ the task string?

A multi-task fine-tune whose training scenes each contain exactly ONE object
never needs the instruction — the task is 100% predictable from pixels, so
the language channel gets no gradient and the deployed model grabs whatever
object it prefers, regardless of the command. This probe measures it in two
minutes, through the same Ficelle server you deploy with:

  1. same image + same task, twice      -> sampling-noise floor
  2. same image + task string SWAPPED   -> language sensitivity
  3. different images (reference ~0.05) -> vision sensitivity

PASS = task-swap diff well above the noise floor (the instruction conditions
the actions). FAIL = task-swap ~ noise floor (language channel unused; if you
need instruction-following, record scenes with multiple objects present).

Reference measurement (grabette_pick3_pi05, single-object scenes):
task-swap 0.0047 vs noise floor 0.0036 vs cross-image 0.05 -> FAIL, and the
robot confirmed it (asked for the can, went for the mustard).

Usage (server must be running — see README step 5):
  uv run python probe_task_sensitivity.py \\
      --policy_addr <iroh-ticket-or-host:port> \\
      --dataset_repo_id <user>/<dataset>_cartesian \\
      --episodes 80 300 --frame 60 \\
      --tasks "pick up the red can" "pick up the mustard bottle"
"""
import argparse

import numpy as np
import torch
from ficelle_client import open_client
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def get_obs(repo_id, root, episode, frame):
    ds = LeRobotDataset(repo_id, root=root, episodes=[episode], video_backend="pyav")
    item = ds[min(frame, len(ds) - 1)]
    img = (item["observation.images.cam0"].permute(1, 2, 0) * 255.0).round().clamp(0, 255)
    return (np.ascontiguousarray(img.to(torch.uint8).numpy()),
            np.asarray(item["observation.state"], dtype=np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_addr", required=True,
                   help="Ficelle server: iroh ticket or host:port")
    p.add_argument("--dataset_repo_id", required=True)
    p.add_argument("--dataset_root", default=None)
    p.add_argument("--episodes", type=int, nargs=2, required=True,
                   help="Two episodes with DIFFERENT tasks/scenes (one per task)")
    p.add_argument("--frame", type=int, default=60)
    p.add_argument("--tasks", nargs=2, required=True,
                   help="The two training task strings to swap")
    args = p.parse_args()

    kw = {"jpeg_quality": 90}
    if args.policy_addr.startswith("endpoint") and ":" not in args.policy_addr:
        kw["infer_timeout"] = 60.0  # first infer includes server warm-up
    client = open_client(args.policy_addr, **kw)
    print(f"connected: {client.metadata['policy_type']} "
          f"from {client.metadata['checkpoint']}")

    def infer(img, state, task):
        return client.infer({"observation.images.cam0": img,
                             "observation.state": state, "task": task})["actions"]

    obs = [get_obs(args.dataset_repo_id, args.dataset_root, ep, args.frame)
           for ep in args.episodes]

    # 1. sampling-noise floor: identical request twice (flow samples noise)
    a, b = infer(*obs[0], args.tasks[0]), infer(*obs[0], args.tasks[0])
    floor = float(np.abs(a - b).mean())

    # 2. language sensitivity: same image, task swapped (both scenes)
    swaps = [float(np.abs(infer(img, st, args.tasks[0])
                          - infer(img, st, args.tasks[1])).mean())
             for img, st in obs]

    # 3. vision sensitivity: different images, same task
    cross = float(np.abs(infer(*obs[0], args.tasks[0])
                         - infer(*obs[1], args.tasks[0])).mean())
    client.close()

    print(f"\nsampling-noise floor (same img, same task): {floor:.5f}")
    for (ep, s) in zip(args.episodes, swaps):
        print(f"task-swap |diff| (ep{ep}):                    {s:.5f}")
    print(f"cross-image |diff| (vision reference):      {cross:.5f}")
    verdict = ("PASS — the instruction conditions the actions"
               if min(swaps) > 3.0 * floor else
               "FAIL — task string ignored (language channel unused); "
               "instruction-following needs multi-object training scenes")
    print(f"\nGATE: {verdict}")


if __name__ == "__main__":
    main()
