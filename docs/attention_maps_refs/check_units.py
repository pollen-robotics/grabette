"""Are the tool's millimetres really millimetres, on real data?

The Critical fix (apply the postprocessor before scaling) is pinned by a unit
test with a stub postprocessor scaling by a known factor. That pins the
PLUMBING. It does not show the real unnormalizer yields physically sensible
metres.

Decisive check available here: the dataset's own recorded actions are already in
the units the postprocessor outputs. So the model's postprocessed chunk should
sit in the same order of magnitude as the demonstrated actions at the same
frame. If the postprocessor were still being skipped, the chunk would be in
normalized quantile space and land orders of magnitude off.

Also prints the uniform-share baseline for the attention masses, so "camera mass
0.29" can be read against what an indifferent policy would give.
"""

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from grabette_attention.adapters.pi05 import Pi05Adapter
from grabette_attention.loader import load_pi05
from grabette_attention.sources import DatasetSource

CKPT = "SteveNguyen/pick3_graspproj_chunkrel_pi05"
ROOT = "/home/steve/.cache/huggingface/lerobot/local-converted/mustard_graspproj"
CAM = "observation.images.cam0"
FRAME = 192


def main() -> None:
    ds = LeRobotDataset("SteveNguyen/mustard_graspproj", root=ROOT, episodes=[0])
    demo = np.stack(
        [np.asarray(ds[i]["action"], dtype=np.float64) for i in range(FRAME, FRAME + 50)]
    )
    dt = np.abs(demo[:, :3])
    print("DEMONSTRATED actions, frames 192-241, translation channels (metres):")
    print(f"  per-step |delta| median {np.median(dt):.3e}  max {dt.max():.3e}")
    print(f"  50-step cumulative travel {np.linalg.norm(demo[:, :3].sum(axis=0)):.4f} m")

    policy, pre, post = load_pi05(CKPT, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)
    source = DatasetSource(
        "SteveNguyen/mustard_graspproj",
        episodes=[0],
        camera_keys=adapter.camera_keys,
        task="pick up the mustard bottle",
        root=ROOT,
        selection=[FRAME],
    )
    obs = next(iter(source.frames()))
    chunk = adapter.run(obs, adapter.draw_noise(), capture=False).chunk
    ct = np.abs(chunk[:, :3].astype(np.float64))
    print(f"\nPREDICTED chunk shape {chunk.shape} (chunk-relative: offsets, not per-step)")
    print(f"  |translation| median {np.median(ct):.3e}  max {ct.max():.3e}")
    print(f"  final offset magnitude {np.linalg.norm(chunk[-1, :3]):.4f} m")

    ratio = np.median(ct) / max(np.median(dt), 1e-12)
    print(f"\nratio of predicted to demonstrated magnitude: {ratio:.1f}x")
    print("  a chunk-relative offset grows along the chunk, so single-digit-to-tens")
    print("  is expected; ~1e3x or ~1e-3x would mean the units are wrong")

    n_img, n_lang = 256, int(policy.config.tokenizer_max_length)
    total = n_img + n_lang
    print(f"\nuniform-share baseline for a {total}-token prefix:")
    print(f"  image  {n_img}/{total} = {n_img/total:.3f}   (measured 0.29 -> {0.29/(n_img/total):.2f}x uniform)")
    print(f"  language {n_lang}/{total} = {n_lang/total:.3f} (measured 0.71 -> {0.71/(n_lang/total):.2f}x uniform)")


if __name__ == "__main__":
    main()
