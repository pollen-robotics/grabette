"""Did the retrain remove the prompt collision?

The original sweep found that pick3's "pick up the cup" perturbed the sugar
model LESS than scrambling that model's own prompt into nonsense -- 5.9 mm
against 6.7 mm. Two near-identical strings, two unrelated behaviours: merging
those datasets would have left the tasks indistinguishable by prompt.

The model was retrained on "put the sugar cube in the mug", which shares only
the word "the" with any pick3 string. This re-runs the same measurement on the
new checkpoint, same frames, same seed, same protocol.

Two questions, and they can come apart:

  DID THE COLLISION GO?  "pick up the cup" should now sit up with the
      unrelated strings instead of near zero. This is the fix working.

  IS THE KEYING STILL LEXICAL?  Scrambling the model's own words should still
      cost little, and its OLD prompt ("pick up the sugar cup" -- which this
      checkpoint never saw) should now be far away. If instead the scrambled
      variant becomes expensive, the retrain changed how language is used, not
      just which string it keys on.

Frames come from the old dataset snapshot: the recording is identical and the
sweep overrides the task string anyway, so nothing about the prompt under test
comes from the dataset.
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

CKPT = "SteveNguyen/sugarcube_in_mug_chunkrel_pi05_step20000"
REAL = "put the sugar cube in the mug"

VARIANTS = [
    (REAL, "control (its own prompt)"),
    ("", "empty"),
    ("pick up the cup", "THE OLD COLLISION (pick3)"),
    ("pick up the sugar cup", "the OLD prompt, never seen by this model"),
    ("pick up the mustard bottle", "pick3 sibling"),
    ("pick up the red can", "pick3 sibling"),
    ("mug the in cube sugar the put", "scrambled, same words"),
    ("close the drawer", "unrelated task"),
]
EPISODES = (0, 1, 2)

# Measured on the OLD checkpoint, for side-by-side reading. Means over the
# same 21 frames; baseline motion then was 100.5 mm.
OLD = {
    "control (its own prompt)": 0.0,
    "empty": 17.1,
    "THE OLD COLLISION (pick3)": 5.9,
    "the OLD prompt, never seen by this model": 0.0,   # it WAS the old prompt
    "pick3 sibling": None,                             # 9.8 / 10.5, see notes
    "scrambled, same words": 6.7,
    "unrelated task": 34.8,
}


def main() -> None:
    plan = G.plans()
    print(f"checkpoint: {CKPT}\nprompt: {REAL!r}\nepisodes: {EPISODES}\n")

    policy, pre, post = load_pi05(CKPT, device="cpu", fp32=True)
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
            result = prompt_sensitivity(adapter, obs, [v for v, _ in VARIANTS])
            for variant in result.variants:
                rows.setdefault(variant.prompt, []).append(
                    (variant.delta_mm, result.baseline_mm)
                )
            print(f"ep{episode} f{obs.frame:4d} {label:14s} "
                  f"baseline {result.baseline_mm:6.1f} mm | "
                  + "  ".join(f"{v.delta_mm:6.1f}" for v in result.variants),
                  flush=True)

    print("\n\nMEAN PROMPT DELTA (mm) — RETRAINED model")
    width = max(len(v) for v, _ in VARIANTS) + 2
    print(f"{'variant':<{width}}{'new':>8}{'old':>8}   note")
    print("-" * (width + 20 + 44))
    for prompt, note in VARIANTS:
        vals = rows.get(prompt)
        new = np.mean([v for v, _ in vals]) if vals else float("nan")
        old = OLD.get(note)
        old_s = f"{old:8.1f}" if isinstance(old, (int, float)) else f"{'—':>8}"
        shown = repr(prompt) if prompt else "''"
        print(f"{shown:<{width}}{new:8.1f}{old_s}   {note}")

    base = np.mean([b for vals in rows.values() for _, b in vals])
    print(f"\nmean baseline motion: {base:.1f} mm   (old model: 100.5 mm)")

    collision = np.mean([v for v, _ in rows["pick up the cup"]])
    scrambled = np.mean([v for v, _ in rows["mug the in cube sugar the put"]])
    unrelated = np.mean([v for v, _ in rows["close the drawer"]])
    print("\nreading:")
    print(f"  collision  {collision:6.1f} mm   was 5.9 on the old model")
    print(f"  scrambled  {scrambled:6.1f} mm   was 6.7")
    print(f"  unrelated  {unrelated:6.1f} mm   was 34.8")
    print(f"  collision / unrelated = {collision/max(unrelated,1e-9):.2f}"
          "   (old: 0.17 — that ratio near zero WAS the collision)")
    print(f"  scrambled / unrelated = {scrambled/max(unrelated,1e-9):.2f}"
          "   (old: 0.19 — low means keying on words, not order)")


if __name__ == "__main__":
    main()
