"""Capture attention probabilities from a policy without modifying lerobot.

Why a plain forward hook is enough: at inference `sample_actions` sets
`_attn_implementation = "eager"` on both towers, and the stock Gemma attention
module returns `(attn_output, attn_weights)`. The caller discards the second
element, but a forward hook sees the full return value. The fused
`compute_layer_complete` path that drops the weights is TRAINING-ONLY and is not
on this code path.
"""

from typing import Any, Sequence

import numpy as np


class AttentionCapture:
    """Context manager that records attention weights per (denoise step, layer).

    Bookkeeping rule: each layer's hook fires exactly once per denoising step, so
    a layer's own invocation count IS the step index. Do not derive the layer
    from a global call counter — that breaks the moment a layer is skipped or the
    step count changes.

    Captures are numpy arrays of shape (heads, queries, keys) with the batch
    dimension dropped; batch size is 1 for this tool.
    """

    def __init__(self, modules: Sequence[Any]):
        self._modules = list(modules)
        self._handles: list[Any] = []
        self._calls: dict[int, int] = {}
        self.captures: dict[tuple[int, int], np.ndarray] = {}

    def reset(self) -> None:
        """Forget everything, so the next frame starts again at step 0."""
        self._calls.clear()
        self.captures.clear()

    def _make_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            # GemmaAttention returns (attn_output, attn_weights). SDPA returns
            # None for the weights, which means eager was not in force.
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if weights is None:
                raise RuntimeError(
                    "attention weights are None: the policy is not running eager "
                    "attention. pi0.5 sets this itself in sample_actions; a "
                    "vision tower needs _attn_implementation='eager' set first."
                )
            step = self._calls.get(layer_index, 0)
            self._calls[layer_index] = step + 1
            self.captures[(step, layer_index)] = (
                weights.detach().to("cpu", copy=True).float().numpy()[0]
            )

        return hook

    def __enter__(self) -> "AttentionCapture":
        for index, module in enumerate(self._modules):
            self._handles.append(module.register_forward_hook(self._make_hook(index)))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
