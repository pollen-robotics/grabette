"""Load a pi0.5 checkpoint the way the rest of the repo does.

Copied deliberately from `integrations/Pi05/smoke_generation.py` rather than
invented: CPU config first, `compile_model` off so forward hooks are not
swallowed by a graph, fp32 because the pi05 port has a bf16 clash in its flow
path, and camera keys taken from the CHECKPOINT rather than the dataset.
"""

from typing import Any, Callable


def load_pi05(
    checkpoint: str, *, device: str = "cuda", fp32: bool = True
) -> tuple[Any, Callable[[dict], dict]]:
    """Return (policy, preprocessor) ready for the adapter."""
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

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor
