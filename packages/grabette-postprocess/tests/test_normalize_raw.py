"""Tests for the raw-capture normalisation surgery.

The camera rename touches FOUR places and missing any one leaves a dataset that
either fails to load or silently loses its video, so each place gets an
assertion. Built on a synthetic minimal tree rather than a real dataset: the
functions under test are pure metadata edits, and a fixture we can read at a
glance is worth more here than fidelity to a 300 MB download.
"""

import json

import pandas as pd
import pytest

from grabette_postprocess.normalize_raw import (
    rename_video_feature,
    strip_name_prefix,
)

OLD = "observation.images.right_cam0"
NEW = "observation.images.cam0"


@pytest.fixture
def ds(tmp_path):
    root = tmp_path / "ds"
    (root / "meta" / "episodes").mkdir(parents=True)
    (root / "videos" / OLD / "chunk-000").mkdir(parents=True)
    (root / "videos" / OLD / "chunk-000" / "file-000.mp4").write_bytes(b"fake")
    (root / "meta" / "info.json").write_text(json.dumps({
        "features": {
            OLD: {"dtype": "video", "shape": [3, 360, 480]},
            "action": {"dtype": "float32", "shape": [8], "names": [
                "right_x", "right_y", "right_z", "right_ax", "right_ay",
                "right_az", "right_proximal", "right_distal"]},
            "observation.state": {"dtype": "float32", "shape": [2],
                                  "names": ["right_proximal", "right_distal"]},
        }
    }))
    (root / "meta" / "stats.json").write_text(json.dumps({
        OLD: {"mean": [0.5]}, "action": {"mean": [0.0] * 8}}))
    pd.DataFrame({
        "episode_index": [0],
        f"videos/{OLD}/chunk_index": [0],
        f"videos/{OLD}/from_timestamp": [0.0],
        f"stats/{OLD}/mean": [0.5],
        "stats/action/mean": [0.0],
    }).to_parquet(root / "meta" / "episodes" / "file-000.parquet", index=False)
    return root


def test_rename_updates_all_four_places(ds):
    rename_video_feature(ds, OLD, NEW)

    info = json.loads((ds / "meta" / "info.json").read_text())
    assert NEW in info["features"] and OLD not in info["features"]

    assert (ds / "videos" / NEW / "chunk-000" / "file-000.mp4").is_file()
    assert not (ds / "videos" / OLD).exists()

    cols = list(pd.read_parquet(ds / "meta" / "episodes" / "file-000.parquet").columns)
    assert f"videos/{NEW}/chunk_index" in cols
    assert f"stats/{NEW}/mean" in cols
    assert not any(OLD in c for c in cols)

    stats = json.loads((ds / "meta" / "stats.json").read_text())
    assert NEW in stats and OLD not in stats


def test_rename_preserves_unrelated_metadata(ds):
    rename_video_feature(ds, OLD, NEW)
    info = json.loads((ds / "meta" / "info.json").read_text())
    assert info["features"]["action"]["shape"] == [8]
    cols = list(pd.read_parquet(ds / "meta" / "episodes" / "file-000.parquet").columns)
    assert "stats/action/mean" in cols and "episode_index" in cols


def test_rename_keeps_feature_order(ds):
    """info.json feature order is what a reviewer diffs; churning it hides the
    real change."""
    before = list(json.loads((ds / "meta" / "info.json").read_text())["features"])
    rename_video_feature(ds, OLD, NEW)
    after = list(json.loads((ds / "meta" / "info.json").read_text())["features"])
    assert after == [NEW if k == OLD else k for k in before]


def test_rename_refuses_a_missing_feature(ds):
    with pytest.raises(SystemExit, match="not a feature"):
        rename_video_feature(ds, "observation.images.nope", NEW)


def test_rename_refuses_to_clobber(ds):
    info_path = ds / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"][NEW] = {"dtype": "video", "shape": [3, 360, 480]}
    info_path.write_text(json.dumps(info))
    with pytest.raises(SystemExit, match="already exists"):
        rename_video_feature(ds, OLD, NEW)


def test_strip_prefix_renames_channels(ds):
    strip_name_prefix(ds, "right_", ("action", "observation.state"))
    info = json.loads((ds / "meta" / "info.json").read_text())
    assert info["features"]["action"]["names"] == [
        "x", "y", "z", "ax", "ay", "az", "proximal", "distal"]
    assert info["features"]["observation.state"]["names"] == ["proximal", "distal"]


def test_strip_prefix_is_needed_for_channel_lookup(ds):
    """Why this matters: our tooling locates the gripper BY NAME, and the publish
    gate errors on the prefixed names."""
    from grabette_postprocess.grasp_projection_convert import _gripper_columns

    info = json.loads((ds / "meta" / "info.json").read_text())
    # substring matching already finds them, so the strip is about the publish
    # gate and downstream readability, not about breaking the projection
    assert _gripper_columns(info["features"]["action"]["names"], 8) == (6, 7)
    strip_name_prefix(ds, "right_", ("action",))
    info = json.loads((ds / "meta" / "info.json").read_text())
    assert _gripper_columns(info["features"]["action"]["names"], 8) == (6, 7)


def test_strip_prefix_leaves_non_matching_names_alone(ds):
    strip_name_prefix(ds, "left_", ("action",))
    info = json.loads((ds / "meta" / "info.json").read_text())
    assert info["features"]["action"]["names"][0] == "right_x"
