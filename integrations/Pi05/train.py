# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "lerobot[pi,dataset] @ git+https://github.com/huggingface/lerobot@e40b58a8dfa9e7b86918c374791599d070518d11",
#   "scipy", "sentencepiece", "num2words", "accelerate", "protobuf", "wandb",
#   "av",  # pyav video backend — HF-jobs images ship no FFmpeg shared libs,
#          # so torchcodec cannot load there; pass --dataset.video_backend=pyav
# ]
# ///
"""VLA fine-tuning launcher (pi05 / pi0_fast): stock `lerobot-train` with ONE
surgical fix.

Why this exists: when fine-tuning FROM a base checkpoint
(`--policy.pretrained_path=lerobot/pi05_base` or `lerobot/pi0fast-base`),
lerobot-train deserializes the processor pipeline SAVED ALONGSIDE that base
checkpoint instead of building one for YOUR run — so the pipeline carries the
base checkpoint's serialized settings rather than your policy-config
overrides and your dataset's normalization stats. (For pi0fast specifically
it also pins `action_tokenizer_name='physical-intelligence/fast'`, which is
unloadable on transformers v5 and the wrong tokenizer anyway — see
pi0fast/README.md.)

The fix: build the pipeline FRESH from the policy config (which carries your
CLI overrides) and the training dataset's stats, instead of deserializing the
base checkpoint's. Model weights still load from the base checkpoint; only
the data-processing pipeline is rebuilt. Everything else is stock
lerobot-train — all lerobot-train flags work unchanged.

Usage (full recipe in README.md):
  uv run python train.py --policy.type=pi05 \\
      --policy.pretrained_path=lerobot/pi05_base ...
"""

import dataclasses
import os

import lerobot.scripts.lerobot_train as lerobot_train
from lerobot.policies import factory as policy_factory

_original = policy_factory.make_pre_post_processors

# Opt-in chunk-relative actions (docs/relative_actions_lerobot_native.md).
# An env var, not a CLI flag: lerobot-train parses argv itself and rejects
# unknown flags. Pass it with `hf jobs uv run --env GRABETTE_CHUNK_RELATIVE=1`.
_CHUNK_RELATIVE = os.environ.get("GRABETTE_CHUNK_RELATIVE", "").lower() in ("1", "true", "yes")


def _install_chunk_relative(pre, post):
    """Swap LeRobot's elementwise relative steps for the composed ones.

    LeRobot's built-in subtracts `observation.state` componentwise, which is not
    rotation composition and yields world-frame offsets — unusable for a
    Cartesian action space recorded against an arbitrary SLAM origin. Ours
    composes into the chunk's reference frame instead.

    Raises if the expected steps are absent. Silently training on absolute
    actions while believing otherwise is the expensive failure here: the loss
    would look healthy and the policy would command garbage.
    """
    from grabette_chunkrel.chunk_relative_processor import (
        AbsoluteFromChunkRelativeStep,
        ChunkRelativeActionsStep,
    )

    pre_names = [type(s).__name__ for s in pre.steps]
    post_names = [type(s).__name__ for s in post.steps]
    if "RelativeActionsProcessorStep" not in pre_names:
        raise RuntimeError(
            f"no RelativeActionsProcessorStep to replace in {pre_names}; this "
            "policy's pipeline has no relative-action slot, so chunk-relative "
            "actions cannot be installed"
        )
    if "AbsoluteActionsProcessorStep" not in post_names:
        raise RuntimeError(
            f"no AbsoluteActionsProcessorStep to replace in {post_names}; "
            "without the inverse the policy's output stays relative"
        )
    if "NormalizerProcessorStep" in pre_names and (
        pre_names.index("RelativeActionsProcessorStep")
        > pre_names.index("NormalizerProcessorStep")
    ):
        raise RuntimeError(
            "the relative slot sits AFTER the normaliser; the dataset stats "
            "describe the relative distribution, so this ordering would "
            "normalise with the wrong scale"
        )

    forward = ChunkRelativeActionsStep()
    pre_steps = list(pre.steps)
    pre_steps[pre_names.index("RelativeActionsProcessorStep")] = forward
    post_steps = list(post.steps)
    post_steps[post_names.index("AbsoluteActionsProcessorStep")] = (
        AbsoluteFromChunkRelativeStep(relative_step=forward)
    )
    print(
        "[chunk-relative] installed: actions are offsets from each chunk's "
        "reference pose, composed into its frame. Dataset action stats MUST "
        "describe the relative distribution (write_relative_action_stats)."
    )
    return (dataclasses.replace(pre, steps=pre_steps),
            dataclasses.replace(post, steps=post_steps))


def _fresh_pipeline(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs):
    # Drop the pretrained pipeline source and the overrides that only apply
    # to the deserialization path; the fresh build takes everything it needs
    # (normalization stats, tokenizer names, device) from policy_cfg +
    # dataset_stats.
    kwargs.pop("preprocessor_overrides", None)
    kwargs.pop("postprocessor_overrides", None)
    pre, post = _original(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs)
    if _CHUNK_RELATIVE:
        pre, post = _install_chunk_relative(pre, post)
    return pre, post


# Patch both the factory and the reference lerobot_train imported.
policy_factory.make_pre_post_processors = _fresh_pipeline
lerobot_train.make_pre_post_processors = _fresh_pipeline

if __name__ == "__main__":
    lerobot_train.main()
