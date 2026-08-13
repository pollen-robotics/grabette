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

import lerobot.scripts.lerobot_train as lerobot_train
from lerobot.policies import factory as policy_factory

_original = policy_factory.make_pre_post_processors


def _fresh_pipeline(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs):
    # Drop the pretrained pipeline source and the overrides that only apply
    # to the deserialization path; the fresh build takes everything it needs
    # (normalization stats, tokenizer names, device) from policy_cfg +
    # dataset_stats.
    kwargs.pop("preprocessor_overrides", None)
    kwargs.pop("postprocessor_overrides", None)
    return _original(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs)


# Patch both the factory and the reference lerobot_train imported.
policy_factory.make_pre_post_processors = _fresh_pipeline
lerobot_train.make_pre_post_processors = _fresh_pipeline

if __name__ == "__main__":
    lerobot_train.main()
