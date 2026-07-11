"""Pi0-FAST fine-tuning launcher: stock `lerobot-train` with ONE surgical fix.

Why this exists: when fine-tuning FROM `lerobot/pi0fast-base`
(`--policy.pretrained_path=...`), lerobot-train loads the SERIALIZED processor
pipeline saved alongside the base checkpoint. That pipeline pins
`action_tokenizer_name='physical-intelligence/fast'`, which

  1. cannot be loaded on any transformers v5 (the PI repo's Hub packaging is
     broken under the v5 AutoProcessor path — "Couldn't instantiate the
     backend tokenizer"), and
  2. is NOT the tokenizer we fitted on our own dataset anyway — training
     would tokenize actions with the wrong codec even if it loaded.

The fix: build the pipeline FRESH from the policy config (which carries our
`--policy.action_tokenizer_name`) instead of deserializing the base
checkpoint's. Model weights still load from the base checkpoint; only the
data-processing pipeline is rebuilt. Everything else is stock lerobot-train.

Usage: identical to lerobot-train —
  uv run python train_pi0fast.py --policy.type=pi0_fast \\
      --policy.pretrained_path=lerobot/pi0fast-base \\
      --policy.action_tokenizer_name=<user>/fast_tokenizer_<task> ...
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
