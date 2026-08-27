"""Tests for the chunk-relative processor steps.

The maths is covered in test_chunk_relative.py. These cover the lerobot shell:
batching, the forward/inverse pairing, the reference at inference, and
serialisation — the last one being what P3 verifies inside the Jobs container,
so it is worth catching here first.
"""

import numpy as np
import pytest
import torch
from lerobot.processor import DataProcessorPipeline, ProcessorStepRegistry
from lerobot.types import TransitionKey

from grabette_postprocess.chunk_relative_processor import (
    INVERSE_STEP_NAME,
    STEP_NAME,
    AbsoluteFromChunkRelativeStep,
    ChunkRelativeActionsStep,
)


def make_batch(b=4, t=12, seed=0):
    """Absolute-pose action chunks from an arbitrary origin, as the dataset gives."""
    r = np.random.default_rng(seed)
    pos = np.cumsum(r.normal(scale=0.01, size=(b, t, 3)), axis=1) + np.array([2.0, -1.0, 0.5])
    rot = np.cumsum(r.normal(scale=0.02, size=(b, t, 3)), axis=1) + np.array([0.3, -1.5, 0.8])
    grip = np.abs(r.normal(scale=0.2, size=(b, t, 2)))
    return torch.tensor(np.concatenate([pos, rot, grip], axis=2), dtype=torch.float32)


def transition(action):
    return {TransitionKey.ACTION: action, TransitionKey.OBSERVATION: {}}


# ── forward / inverse pairing ───────────────────────────────────────────

def test_forward_then_inverse_recovers_the_absolute_actions():
    """The pair must be exact: any error is a systematic offset on every
    commanded pose."""
    act = make_batch()
    fwd = ChunkRelativeActionsStep()
    inv = AbsoluteFromChunkRelativeStep(relative_step=fwd)

    rel = fwd(transition(act))[TransitionKey.ACTION]
    back = inv(transition(rel))[TransitionKey.ACTION]

    assert rel.shape == act.shape
    assert torch.abs(back[..., :3] - act[..., :3]).max() < 2e-3
    assert torch.abs(back[..., 6:] - act[..., 6:]).max() < 1e-4


def test_the_forward_step_actually_changes_the_actions():
    """Guard against a no-op passing the round-trip test."""
    act = make_batch()
    rel = ChunkRelativeActionsStep()(transition(act))[TransitionKey.ACTION]
    assert torch.abs(rel[..., :3] - act[..., :3]).max() > 0.1, "suspiciously unchanged"
    assert torch.abs(rel[:, 0, :6]).max() < 1e-5, "first action should be the identity"


def test_disabled_is_a_passthrough():
    act = make_batch()
    out = ChunkRelativeActionsStep(enabled=False)(transition(act))[TransitionKey.ACTION]
    assert torch.equal(out, act)


def test_dtype_and_device_are_preserved():
    act = make_batch()
    out = ChunkRelativeActionsStep()(transition(act))[TransitionKey.ACTION]
    assert out.dtype == act.dtype and out.device == act.device


# ── shapes seen at inference ────────────────────────────────────────────

def test_a_single_step_action_passes_through_the_forward_step():
    """At inference the action is (B, D) with no chunk to reference; the inverse
    step is what runs there, not the forward one."""
    act = torch.zeros(4, 8)
    out = ChunkRelativeActionsStep()(transition(act))[TransitionKey.ACTION]
    assert torch.equal(out, act)


def test_the_inverse_accepts_an_unbatched_chunk():
    """evaluate.py holds one sample, so (T, D) must work and stay (T, D)."""
    act = make_batch(b=1)[0]
    inv = AbsoluteFromChunkRelativeStep()
    inv.set_reference(np.zeros(6, dtype=np.float32))
    out = inv(transition(act))[TransitionKey.ACTION]
    assert out.shape == act.shape


def test_the_inverse_uses_an_explicitly_set_reference():
    """The deploy-time path: the reference is the robot's MEASURED pose, which no
    forward step has seen."""
    rel = torch.zeros(1, 5, 8)
    inv = AbsoluteFromChunkRelativeStep()
    inv.set_reference(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float32))
    out = inv(transition(rel))[TransitionKey.ACTION]
    # zero offsets from that reference must reproduce the reference itself
    assert torch.allclose(out[0, :, :3], torch.tensor([1.0, 2.0, 3.0]), atol=1e-5)


def test_the_inverse_refuses_to_guess_a_reference():
    """Silently assuming the origin would offset every commanded pose."""
    inv = AbsoluteFromChunkRelativeStep()
    with pytest.raises(ValueError, match="no reference pose"):
        inv(transition(torch.zeros(2, 5, 8)))


# ── jumps are repaired inside the batch ─────────────────────────────────

def test_jumps_are_repaired_and_counted():
    act = make_batch(b=2, t=14).clone()
    act[0, 7:, 0] += 0.4                                   # translation jump
    rel = ChunkRelativeActionsStep()
    rel(transition(act))
    assert rel._last_report["n_jumps"] == 1
    assert rel._last_report["n_rot_jumps"] == 0


# ── serialisation: what P3 checks in the Jobs container ─────────────────

def test_both_steps_are_registered_under_stable_names():
    """The saved pipeline stores NAMES; renaming a step breaks every checkpoint
    that used it."""
    assert ProcessorStepRegistry.get(STEP_NAME) is ChunkRelativeActionsStep
    assert ProcessorStepRegistry.get(INVERSE_STEP_NAME) is AbsoluteFromChunkRelativeStep


def test_config_round_trips_through_the_registry():
    step = ChunkRelativeActionsStep(rot="rotvec", jump_cap_m=0.05, jump_cap_deg=4.0)
    cfg = step.get_config()
    rebuilt = ProcessorStepRegistry.get(STEP_NAME)(**cfg)
    assert rebuilt.get_config() == cfg


def test_a_pipeline_containing_the_step_saves_and_loads(tmp_path):
    """The one failure mode that would otherwise surface after a paid training
    run: the checkpoint's pipeline cannot be rebuilt."""
    pipe = DataProcessorPipeline(steps=[ChunkRelativeActionsStep()], name="test")
    pipe.save_pretrained(str(tmp_path))
    loaded = DataProcessorPipeline.from_pretrained(str(tmp_path), config_filename="test.json")
    assert any(isinstance(s, ChunkRelativeActionsStep) for s in loaded.steps)

    act = make_batch()
    a = pipe(transition(act))[TransitionKey.ACTION]
    b = loaded(transition(act))[TransitionKey.ACTION]
    assert torch.abs(a - b).max() < 1e-6


def test_r6d_is_refused_until_the_feature_spec_is_rewritten():
    """It widens the action 8 -> 11; emitting a spec that disagrees with the
    tensors would fail deep inside the policy instead of here."""
    with pytest.raises(NotImplementedError, match="action dimension"):
        ChunkRelativeActionsStep(rot="r6d").transform_features({})
