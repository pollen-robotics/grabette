"""Tier A — lerobot API-surface tripwires.

Asserts the exact lerobot symbols/signatures grabette-postprocess (dataset.py)
and the DiffusionPolicy integration depend on. Fails loudly + specifically when
a lerobot bump removes/renames/moves one — the class of change that has silently
broken the pipeline before (vcodec kwarg dropped, `datasets` moved to an extra).

Skipped entirely when lerobot isn't installed (the fast pure-Python CI lane).
"""
import inspect

import pytest

pytest.importorskip("lerobot", reason="lerobot not installed (fast lane)")


def test_lerobot_dataset_importable_from_top_level():
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata  # noqa: F401


def test_rgb_encoder_config_importable():
    from lerobot.configs import RGBEncoderConfig  # noqa: F401


def test_create_takes_rgb_encoder_not_vcodec():
    from lerobot.datasets import LeRobotDataset
    params = inspect.signature(LeRobotDataset.create).parameters
    assert "rgb_encoder" in params, f"create() params: {list(params)}"
    assert "vcodec" not in params, "vcodec still present — dataset.py migration mismatch"


def test_rgb_encoder_config_accepts_vcodec():
    from lerobot.configs import RGBEncoderConfig
    RGBEncoderConfig(vcodec="h264")


def test_create_takes_streaming_encoding():
    from lerobot.datasets import LeRobotDataset
    assert "streaming_encoding" in inspect.signature(LeRobotDataset.create).parameters


def test_recompute_stats_available():
    # Used by DiffusionPolicy/convert_dataset.py
    from lerobot.datasets.dataset_tools import recompute_stats  # noqa: F401


def test_hf_datasets_present():
    # `datasets` (HF) was historically moved to an extra; the pipeline needs it.
    import datasets  # noqa: F401
