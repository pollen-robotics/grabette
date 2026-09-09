"""Does the sugar model's prompt do anything?

Variants chosen to separate distinct questions rather than to sample randomly:

  the real string      control -- must return exactly 0.0 mm
  ""                   does the prompt matter AT ALL?
  "pick up the cup"    THE COLLISION: pick3's string for picking up a paper
                       cup, one word away from this model's own prompt
  mustard / red can    pick3's other two strings: sibling tasks this model
                       never saw
  a correct description of the real task -- never seen in training, so this
                       asks whether anything semantic transferred
  word-scrambled       same tokens, destroyed syntax: separates "which words"
                       from "in what order"
  unrelated command    a different manipulation task entirely

Run at every task phase, because a prompt could matter only while choosing
what to approach and be irrelevant once the motion is underway.
"""

import sys
from pathlib import Path

import numpy as np

from grabette_attention.adapters.pi05 import Pi05Adapter
from grabette_attention.language import prompt_sensitivity
from grabette_attention.loader import load_pi05
from grabette_attention.sources import DatasetSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G

REAL = "pick up the sugar cup"
VARIANTS = [
    (REAL, "control (its own prompt)"),
    ("", "empty"),
    ("pick up the cup", "COLLISION with pick3"),
    ("pick up the mustard bottle", "pick3 sibling"),
    ("pick up the red can", "pick3 sibling"),
    ("put the sugar cube in the mug", "correct description, unseen"),
    ("cup the sugar up pick", "scrambled, same words"),
    ("close the drawer", "unrelated task"),
]
EPISODES = (0, 1, 2)


def main() -> None:
    plan = G.plans()
    print(f"prompt: {REAL!r}\nepisodes: {EPISODES}\n")

    policy, pre, post = load_pi05(G.CKPT, device="cpu", fp32=True)
    adapter = Pi05Adapter(policy, pre, post, device="cpu", seed=0)

    rows = {}
    for episode in EPISODES:
        p = plan[episode]
        source = DatasetSource(
            G.REPO, episodes=[episode], camera_keys=adapter.camera_keys,
            task=REAL, root=G.ROOT, selection=sorted(set(p["frames"])),
        )
        by_frame = {obs.frame: obs for obs in source.frames()}
        for want, label in zip(p["frames"], p["labels"]):
            obs = by_frame.get(want)
            if obs is None:
                continue
            result = prompt_sensitivity(
                adapter, obs, [v for v, _ in VARIANTS]
            )
            for variant in result.variants:
                rows.setdefault((variant.prompt, label), []).append(
                    (variant.delta_mm, result.baseline_mm)
                )
            print(f"ep{episode} f{obs.frame} {label:14s} "
                  f"baseline {result.baseline_mm:6.1f} mm | "
                  + "  ".join(f"{v.delta_mm:6.1f}" for v in result.variants),
                  flush=True)

    labels = [lab for _, lab in zip(plan[EPISODES[0]]["frames"],
                                    plan[EPISODES[0]]["labels"])]
    print("\n\nMEAN PROMPT DELTA (mm), averaged over "
          f"{len(EPISODES)} episodes")
    width = max(len(v) for v, _ in VARIANTS) + 2
    header = f"{'variant':<{width}}" + "".join(f"{lab[:11]:>12}" for lab in labels)
    print(header)
    print("-" * len(header))
    for prompt, note in VARIANTS:
        cells = []
        for lab in labels:
            vals = rows.get((prompt, lab))
            cells.append(f"{np.mean([v for v, _ in vals]):12.1f}" if vals else f"{'-':>12}")
        shown = repr(prompt) if prompt else "''"
        print(f"{shown:<{width}}" + "".join(cells))

    base = [b for vals in rows.values() for _, b in vals]
    print(f"\nmean baseline motion across all frames: {np.mean(base):.1f} mm")
    print("\nvariant notes:")
    for prompt, note in VARIANTS:
        print(f"  {prompt!r:<32} {note}")
    print("\nread: control must be 0.0. Any variant near 0 means the policy")
    print("ignores that difference. Compare each against the baseline motion")
    print("and against the whole-camera ablation for the same phase.")


if __name__ == "__main__":
    main()
