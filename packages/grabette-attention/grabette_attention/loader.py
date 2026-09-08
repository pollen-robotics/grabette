"""Load a pi0.5 checkpoint the way the rest of the repo does.

Copied deliberately from `integrations/Pi05/smoke_generation.py` rather than
invented: CPU config first, `compile_model` off so forward hooks are not
swallowed by a graph, fp32 because the pi05 port has a bf16 clash in its flow
path, and camera keys taken from the CHECKPOINT rather than the dataset.
"""

from typing import Any, Callable


def load_pi05(
    checkpoint: str, *, device: str = "cuda", fp32: bool = True
) -> tuple[Any, Callable[[dict], dict], Callable[[Any], Any]]:
    """Return (policy, preprocessor, postprocessor) ready for the adapter.

    The postprocessor matters as much as the preprocessor here: pi0.5 normalizes
    actions with per-channel quantile normalization, and it is the
    postprocessor's `UnnormalizerProcessorStep` (followed by
    `AbsoluteActionsProcessorStep`) that converts the model's raw output back to
    real units. Skipping it, as an earlier version of this loader did, leaves
    every downstream millimetre figure in normalized quantile space instead.
    """
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    # Chunk-relative checkpoints register their processor steps on import. Import
    # it if present so `make_pre_post_processors` can resolve them; a plain delta
    # checkpoint is unaffected.
    try:
        import grabette_chunkrel.chunk_relative_processor  # noqa: F401
    except ImportError:
        pass

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cpu"
    if hasattr(config, "compile_model"):
        config.compile_model = False

    policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
    policy = policy.to(dtype=torch.float32 if fp32 else torch.bfloat16).eval()
    policy = policy.to(device)
    policy.config.device = device

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    # Chunk-relative checkpoints' postprocessor ends with an
    # AbsoluteFromChunkRelativeStep, which needs a reference pose that only the
    # robot has. `smoke_generation.py --chunk_relative` drops it for the same
    # reason: offline, we want the offsets themselves, not an absolute pose we
    # cannot reconstruct. A plain delta checkpoint's postprocessor never has
    # this step, so the filter is a no-op for it.
    postprocessor.steps = [
        step for step in postprocessor.steps
        if "AbsoluteFromChunkRelative" not in type(step).__name__
    ]
    return policy, preprocessor, postprocessor
