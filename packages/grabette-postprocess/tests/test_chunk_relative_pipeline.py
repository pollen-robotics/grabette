"""Integration: our step inside the REAL pi05 processor pipeline.

Two things the unit tests cannot cover, both of which would fail silently:

  1. POSITION. Our step must run before NormalizerProcessorStep, because the
     normaliser's stats describe the relative distribution. Asserted against the
     shipped factory SOURCE, so a lerobot bump that reorders the pipeline breaks
     this test rather than a training run.
  2. RANGE. With relative stats the normalised action must land in [-1, 1];
     absolute actions through the same stats land ~20x outside it.

The tokenizer step is skipped: it needs `transformers` and sits after the
normaliser, so it cannot affect either property.
"""

import inspect
import re

import numpy as np
import pytest
import torch

from grabette_postprocess.chunk_relative import to_chunk_relative
from grabette_postprocess.chunk_relative_processor import ChunkRelativeActionsStep

pytest.importorskip("lerobot", reason="needs lerobot")

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import processor_pi05  # noqa: E402
from lerobot.processor import (  # noqa: E402
    AddBatchDimensionProcessorStep,
    NormalizerProcessorStep,
    RenameObservationsProcessorStep,
)
from lerobot.types import TransitionKey  # noqa: E402

K, B, ADIM, SDIM = 50, 2, 8, 2


def absolute_batch(seed=0):
    r = np.random.default_rng(seed)
    pos = np.cumsum(r.normal(scale=0.01, size=(B, K, 3)), axis=1) + np.array([3.0, -1.0, 0.5])
    rot = np.cumsum(r.normal(scale=0.02, size=(B, K, 3)), axis=1) + np.array([0.4, -2.0, 1.1])
    grip = np.abs(r.normal(scale=0.2, size=(B, K, 2)))
    return np.concatenate([pos, rot, grip], axis=2)


def stats_of(a):
    return {"q01": torch.tensor(np.quantile(a, 0.01, axis=0), dtype=torch.float32),
            "q99": torch.tensor(np.quantile(a, 0.99, axis=0), dtype=torch.float32),
            "min": torch.tensor(a.min(0), dtype=torch.float32),
            "max": torch.tensor(a.max(0), dtype=torch.float32),
            "mean": torch.tensor(a.mean(0), dtype=torch.float32),
            "std": torch.tensor(a.std(0) + 1e-8, dtype=torch.float32)}


def prefix(with_our_step, action_stats):
    features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(SDIM,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(ADIM,)),
    }
    norm = NormalizerProcessorStep(
        features=features,
        norm_map={"VISUAL": NormalizationMode.IDENTITY,
                  "STATE": NormalizationMode.QUANTILES,
                  "ACTION": NormalizationMode.QUANTILES},
        stats={"action": action_stats,
               "observation.state": stats_of(np.zeros((4, SDIM)) + 0.5)},
    )
    steps = [RenameObservationsProcessorStep(rename_map={}), AddBatchDimensionProcessorStep()]
    if with_our_step:
        steps.append(ChunkRelativeActionsStep())
    steps.append(norm)
    return steps


def run(steps, absolute):
    tr = {
        TransitionKey.OBSERVATION: {"observation.state": torch.full((B, SDIM), 0.5)},
        TransitionKey.ACTION: torch.tensor(absolute, dtype=torch.float32),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick up the mustard bottle"] * B},
    }
    for s in steps:
        tr = s(tr)
    return tr[TransitionKey.ACTION]


def test_the_relative_slot_precedes_the_normaliser_in_the_shipped_factory():
    """Read from lerobot's source: if a bump reorders the pipeline, our whole
    stats argument changes and we want to hear about it here."""
    src = inspect.getsource(processor_pi05.make_pi05_pre_post_processors)
    order = re.findall(
        r"(relative_step|NormalizerProcessorStep|UnnormalizerProcessorStep"
        r"|AbsoluteActionsProcessorStep)", src)
    assert order.index("relative_step") < order.index("NormalizerProcessorStep")
    assert order.index("UnnormalizerProcessorStep") < order.index("AbsoluteActionsProcessorStep")


def test_relative_actions_normalise_into_the_trained_range():
    absolute = absolute_batch()
    rel = np.concatenate([to_chunk_relative(c)[0] for c in absolute]).astype(np.float64)
    out = run(prefix(True, stats_of(rel)), absolute)
    assert out.shape == (B, K, ADIM)
    assert out.abs().max() < 1.6
    assert (out.abs() <= 1.0).float().mean() > 0.9


def test_without_the_step_the_same_stats_blow_up():
    """The failure being avoided, asserted so the test above is not vacuous."""
    absolute = absolute_batch()
    rel = np.concatenate([to_chunk_relative(c)[0] for c in absolute]).astype(np.float64)
    out = run(prefix(False, stats_of(rel)), absolute)
    assert out.abs().max() > 5.0, "absolute actions should be far outside the range"
