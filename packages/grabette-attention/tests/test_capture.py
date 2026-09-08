"""Forward-hook capture and its (step, layer) bookkeeping.

pi0.5 runs 18 expert layers per denoising step and 10 steps per chunk, so a hook
on every layer fires 180 times. Getting the (step, layer) assignment wrong
silently mixes early and late denoising, which is exactly the axis nobody has
published — so it is pinned here.
"""
import numpy as np
import torch
from torch import nn

from grabette_attention.capture import AttentionCapture


class FakeAttention(nn.Module):
    """Stands in for GemmaAttention: returns (output, attention_weights)."""

    def __init__(self, layer: int, heads: int = 8, queries: int = 50, keys: int = 506):
        super().__init__()
        self.layer = layer
        self.heads, self.queries, self.keys = heads, queries, keys
        self.calls = 0

    def forward(self, x):
        # Weights encode (layer, call) so the test can assert the assignment.
        w = torch.full((1, self.heads, self.queries, self.keys), float(self.layer))
        w[0, 0, 0, 0] = float(self.calls)
        self.calls += 1
        return x, w


def test_it_captures_one_tensor_per_layer_per_step():
    layers = [FakeAttention(i) for i in range(18)]
    with AttentionCapture(layers) as cap:
        for _ in range(10):                       # 10 denoising steps
            for layer in layers:
                layer(torch.zeros(1))
    assert len(cap.captures) == 180
    assert set(cap.captures) == {(s, l) for s in range(10) for l in range(18)}


def test_the_step_is_each_layers_own_invocation_count():
    layers = [FakeAttention(i) for i in range(3)]
    with AttentionCapture(layers) as cap:
        for _ in range(4):
            for layer in layers:
                layer(torch.zeros(1))
    for (step, layer_idx), w in cap.captures.items():
        assert w[0, 0, 0] == step        # the call counter we stamped in
        assert w[0, 1, 1] == layer_idx   # the layer id we stamped in


def test_captures_are_numpy_with_the_batch_dimension_dropped():
    layers = [FakeAttention(0)]
    with AttentionCapture(layers) as cap:
        layers[0](torch.zeros(1))
    w = cap.captures[(0, 0)]
    assert isinstance(w, np.ndarray)
    assert w.shape == (8, 50, 506)       # (heads, queries, keys)


def test_hooks_are_removed_on_exit_so_later_passes_are_not_captured():
    layers = [FakeAttention(0)]
    with AttentionCapture(layers) as cap:
        layers[0](torch.zeros(1))
    layers[0](torch.zeros(1))            # an ablation pass, outside the block
    assert len(cap.captures) == 1


def test_hooks_are_removed_even_when_the_body_raises():
    layers = [FakeAttention(0)]
    try:
        with AttentionCapture(layers):
            raise RuntimeError("cuda oom")
    except RuntimeError:
        pass
    layers[0](torch.zeros(1))
    assert layers[0]._forward_hooks == {}


def test_reset_clears_the_counters_so_the_next_frame_starts_at_step_zero():
    layers = [FakeAttention(0)]
    with AttentionCapture(layers) as cap:
        layers[0](torch.zeros(1))
        cap.reset()
        layers[0](torch.zeros(1))
    assert set(cap.captures) == {(0, 0)}
