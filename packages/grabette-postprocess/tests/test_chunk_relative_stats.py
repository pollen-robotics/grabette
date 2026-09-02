"""Tests for relative-action stats.

The point of the module: pi05 normalises ACTION with QUANTILES between the
relative step and the model, so stats computed on absolute poses put the
rotation channels outside [-1, 1]. These tests assert the fix works and that the
broken case is genuinely broken, so the check cannot go vacuous.
"""

import json

import numpy as np
import pandas as pd
import pytest

from grabette_postprocess.chunk_relative import to_chunk_relative
from grabette_postprocess.chunk_relative_stats import (
    relative_action_stats,
    write_relative_action_stats,
)

NAMES = ["x", "y", "z", "ax", "ay", "az", "proximal", "distal"]


def make_dataset(tmp_path, n_ep=4, t=60, seed=0):
    """A minimal dataset root: absolute poses from an arbitrary SLAM origin."""
    r = np.random.default_rng(seed)
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    rows = []
    for e in range(n_ep):
        pos = np.cumsum(r.normal(scale=0.01, size=(t, 3)), axis=0) + r.normal(scale=5, size=3)
        rot = np.cumsum(r.normal(scale=0.02, size=(t, 3)), axis=0) + r.normal(scale=1, size=3)
        grip = np.abs(r.normal(scale=0.3, size=(t, 2)))
        act = np.concatenate([pos, rot, grip], axis=1)
        rows += [{"episode_index": e, "action": a.astype(np.float32)} for a in act]
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")

    (root / "meta" / "info.json").write_text(json.dumps({"fps": 50}))
    absolute = np.stack([r_["action"] for r_ in rows]).astype(np.float64)
    (root / "meta" / "stats.json").write_text(json.dumps({
        "action": {
            "min": absolute.min(0).tolist(), "max": absolute.max(0).tolist(),
            "mean": absolute.mean(0).tolist(), "std": absolute.std(0).tolist(),
            "q01": np.quantile(absolute, 0.01, axis=0).tolist(),
            "q99": np.quantile(absolute, 0.99, axis=0).tolist(),
        },
        "observation.state": {"min": [0, 0], "max": [1, 1]},
    }))
    return root, absolute


def quantile_normalise(arr, stats):
    """What pi05's QUANTILES mode does: q01..q99 -> [-1, 1]."""
    q01, q99 = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    denom = np.where((q99 - q01) == 0, 1.0, q99 - q01)
    return (arr - q01) / denom * 2.0 - 1.0


def relative_actions_of(root, chunk_size):
    df = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    out = []
    for _e, g in df.groupby("episode_index"):
        a = np.stack(g["action"].to_numpy()).astype(np.float64)
        for t in range(0, max(len(a) - chunk_size, 0), chunk_size):
            out.append(to_chunk_relative(a[t:t + chunk_size])[0])
    return np.concatenate(out).astype(np.float64)


# ── the property the module exists for ──────────────────────────────────

def test_absolute_stats_put_relative_actions_out_of_range(tmp_path):
    """The broken case, asserted so the fix below is not vacuous. This is the
    documented failure: values the model never saw during training."""
    root, _ = make_dataset(tmp_path)
    absolute_stats = json.loads((root / "meta" / "stats.json").read_text())["action"]
    v = quantile_normalise(relative_actions_of(root, 50), absolute_stats)
    assert np.abs(v).max() > 1.5, "expected absolute stats to mis-scale badly"


def test_relative_stats_bring_actions_into_range(tmp_path):
    """With matching stats, q01..q99 maps to [-1, 1] by construction; only the
    1% tails may exceed it."""
    root, _ = make_dataset(tmp_path)
    stats, _rep = relative_action_stats(root, chunk_size=50)
    v = quantile_normalise(relative_actions_of(root, 50), stats)
    inside = np.mean(np.all(np.abs(v) <= 1.0, axis=1))
    assert inside > 0.80, f"only {inside:.0%} of actions inside [-1, 1]"
    assert np.abs(np.quantile(v, 0.5, axis=0)).max() < 0.9, "distribution off-centre"


def test_quantile_keys_are_present(tmp_path):
    """pi05 refuses to train without q01/q99, and lerobot's compute_stats does
    not emit them."""
    root, _ = make_dataset(tmp_path)
    stats, _ = relative_action_stats(root, chunk_size=50)
    for k in ("min", "max", "mean", "std", "q01", "q99"):
        assert k in stats and len(stats[k]) == len(NAMES), k


# ── chunk-length dependence ─────────────────────────────────────────────

def test_stats_scale_with_chunk_size(tmp_path):
    """The stats are horizon-dependent, which is why --chunk-size must match the
    policy's chunk_size."""
    root, _ = make_dataset(tmp_path)
    small, _ = relative_action_stats(root, chunk_size=10)
    large, _ = relative_action_stats(root, chunk_size=50)
    span_small = np.asarray(small["q99"][:3]) - np.asarray(small["q01"][:3])
    span_large = np.asarray(large["q99"][:3]) - np.asarray(large["q01"][:3])
    assert (span_large > span_small).all(), "a longer chunk must span further"


def test_gripper_stats_are_unchanged_by_the_representation(tmp_path):
    """The gripper passes through the relative step, so its stats must match the
    absolute ones — a difference there means the passthrough broke."""
    root, absolute = make_dataset(tmp_path)
    stats, _ = relative_action_stats(root, chunk_size=50)
    for i in (6, 7):
        assert abs(stats["max"][i] - absolute[:, i].max()) < 1e-3


# ── writing it back ─────────────────────────────────────────────────────

def test_write_preserves_the_absolute_stats_and_other_keys(tmp_path):
    """Reversible and auditable: a dataset whose action stats silently no longer
    describe its stored actions is undiagnosable later."""
    root, _ = make_dataset(tmp_path)
    before = json.loads((root / "meta" / "stats.json").read_text())
    report = write_relative_action_stats(root, chunk_size=50)
    after = json.loads((root / "meta" / "stats.json").read_text())

    assert after["action_absolute"] == before["action"]
    assert after["observation.state"] == before["observation.state"]
    assert after["action"] != before["action"]
    assert after["action_relative_meta"]["chunk_size"] == 50
    assert report["chunks"] > 0


def test_write_is_idempotent(tmp_path):
    """Re-running must not overwrite the archived absolute stats with relative
    ones — that would destroy the only record of the original."""
    root, _ = make_dataset(tmp_path)
    write_relative_action_stats(root, chunk_size=50)
    first = json.loads((root / "meta" / "stats.json").read_text())["action_absolute"]
    write_relative_action_stats(root, chunk_size=50)
    second = json.loads((root / "meta" / "stats.json").read_text())["action_absolute"]
    assert first == second


def test_a_chunk_longer_than_every_episode_is_refused(tmp_path):
    """Silently producing stats from zero chunks would normalise everything to
    garbage."""
    root, _ = make_dataset(tmp_path, t=20)
    with pytest.raises(ValueError, match="no chunk"):
        relative_action_stats(root, chunk_size=200)
