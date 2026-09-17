"""Tests for the metadata half of add_camera.py.

The script's risky parts are guarded at runtime: `verify` checks the alignment
guarantees against the real files and `check_result` decodes a frame out of the
built dataset. Neither can run without ~350 MB of Hub download, so what is
tested here is the pure file surgery in between — the two functions that decide
what the published dataset *says* about its cameras.

Both have a failure mode that produces a dataset which loads and trains:

  patch_info      wrong key ORDER. `config.image_features` is iterated to build
                  the prefix, so key order is token order — and token order is
                  what the attention tool reads back to split a map per camera.
                  Appending the new camera before cam0 would silently relabel
                  every overlay from every earlier run.
  patch_episodes  a stale `right_cam1` column left behind, naming a feature the
                  dataset does not have, or offsets that are not cam0's.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_PI05 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PI05))


def _load():
    spec = importlib.util.spec_from_file_location("_add_camera", _PI05 / "add_camera.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ac = _load()
IMG = "observation.images"


def _info(extra_features: dict | None = None) -> dict:
    """An info.json shaped like the sugar datasets': one camera between a
    leading and a trailing non-camera feature, so ordering is observable."""
    features = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        f"{IMG}.cam0": {
            "dtype": "video",
            "shape": [3, 360, 480],
            "names": ["channels", "height", "width"],
            "info": {"video.height": 360, "video.width": 480,
                     "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.fps": 50, "video.channels": 3,
                     "video.is_depth_map": False, "has_audio": False},
        },
        "action": {"dtype": "float32", "shape": [8]},
    }
    features.update(extra_features or {})
    return {"codebase_version": "v3.0", "total_frames": 6, "features": features}


def _episode_table(*, stale: bool) -> pa.Table:
    """Two episodes. `stale` adds the residual right_cam1 columns that the
    chunk-relative datasets carry and the delta ones do not — both branches."""
    columns = {
        "episode_index": pa.array([0, 1]),
        "length": pa.array([3, 3]),
        f"videos/{IMG}.cam0/chunk_index": pa.array([0, 0]),
        f"videos/{IMG}.cam0/file_index": pa.array([0, 0]),
        f"videos/{IMG}.cam0/from_timestamp": pa.array([0.0, 1.5]),
        f"videos/{IMG}.cam0/to_timestamp": pa.array([1.5, 3.0]),
    }
    for field in ac.STAT_FIELDS:
        columns[f"stats/{IMG}.cam0/{field}"] = pa.array([1.0, 2.0])
    if stale:
        for field in ac.OFFSET_FIELDS:
            columns[f"videos/{IMG}.right_cam1/{field}"] = pa.array([9.0, 9.0])
        for field in ac.STAT_FIELDS:
            columns[f"stats/{IMG}.right_cam1/{field}"] = pa.array([9.0, 9.0])
    return pa.table(columns)


def _raw_episode_table() -> pa.Table:
    """The raw's table: it supplies only the new camera's per-episode stats."""
    columns = {"episode_index": pa.array([0, 1])}
    for field in ac.STAT_FIELDS:
        columns[f"stats/{IMG}.{ac.RAW_CAMERA}/{field}"] = pa.array([7.0, 8.0])
    return pa.table(columns)


def _tree(tmp_path: Path, *, stale: bool) -> tuple[Path, Path]:
    staged, raw = tmp_path / "staged", tmp_path / "raw"
    for root in (staged, raw):
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (staged / "meta" / "info.json").write_text(json.dumps(_info()))
    (staged / "meta" / "stats.json").write_text(
        json.dumps({f"{IMG}.cam0": {"mean": [[[0.5]]]}, "action": {"mean": [0.0]}})
    )
    (raw / "meta" / "stats.json").write_text(
        json.dumps({f"{IMG}.{ac.RAW_CAMERA}": {"mean": [[[0.45]]]}})
    )
    pq.write_table(_episode_table(stale=stale),
                   staged / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(_raw_episode_table(),
                   raw / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return staged, raw


# ── patch_info ──────────────────────────────────────────────────────────

def test_new_camera_lands_directly_after_cam0(tmp_path):
    """Token order is camera order. The new view must come second, and the
    non-camera features must keep their places around it."""
    staged, _ = _tree(tmp_path, stale=False)
    ac.patch_info(staged, 480, 360)
    keys = list(json.loads((staged / "meta" / "info.json").read_text())["features"])
    assert keys == ["observation.state", f"{IMG}.cam0",
                    f"{IMG}.{ac.NEW_CAMERA}", "action"]


def test_new_camera_copies_cam0_video_info_with_its_own_size(tmp_path):
    staged, _ = _tree(tmp_path, stale=False)
    ac.patch_info(staged, 320, 240)
    features = json.loads((staged / "meta" / "info.json").read_text())["features"]
    new = features[f"{IMG}.{ac.NEW_CAMERA}"]
    assert new["shape"] == [3, 240, 320]
    assert new["info"]["video.width"] == 320
    assert new["info"]["video.height"] == 240
    # Codec, pix_fmt and fps are inherited: encode() reproduces cam0's.
    assert new["info"]["video.codec"] == "h264"
    assert new["info"]["video.fps"] == 50
    # Monochrome content, three-channel container.
    assert new["info"]["video.channels"] == 3
    # cam0 itself is untouched.
    assert features[f"{IMG}.cam0"]["info"]["video.width"] == 480


def test_patch_info_is_idempotent(tmp_path):
    """A second pass must not produce two features with one name, nor move the
    camera. `verify` already refuses a tree that has the new camera, so this is
    a property of the helper rather than a path the script takes."""
    staged, _ = _tree(tmp_path, stale=False)
    ac.patch_info(staged, 480, 360)
    ac.patch_info(staged, 480, 360)
    keys = list(json.loads((staged / "meta" / "info.json").read_text())["features"])
    assert keys.count(f"{IMG}.{ac.NEW_CAMERA}") == 1


# ── patch_episodes ──────────────────────────────────────────────────────

@pytest.mark.parametrize("stale", [False, True], ids=["delta", "chunkrel"])
def test_offsets_are_copies_of_cam0(tmp_path, stale):
    """The whole no-reconversion shortcut rests on this: the raw cuts both
    cameras identically, so the new camera's offsets ARE cam0's."""
    staged, raw = _tree(tmp_path, stale=stale)
    ac.patch_episodes(staged, raw)
    table = pq.read_table(
        staged / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    for field in ac.OFFSET_FIELDS:
        assert (table.column(f"videos/{IMG}.{ac.NEW_CAMERA}/{field}").to_pylist()
                == table.column(f"videos/{IMG}.cam0/{field}").to_pylist())


@pytest.mark.parametrize("stale", [False, True], ids=["delta", "chunkrel"])
def test_stats_come_from_the_raw_not_from_cam0(tmp_path, stale):
    """cam0's stats describe the wrong camera; the raw's describe the right one
    (at the pre-resize resolution, which pi0.5 never reads — VISUAL features
    normalise with IDENTITY)."""
    staged, raw = _tree(tmp_path, stale=stale)
    ac.patch_episodes(staged, raw)
    table = pq.read_table(
        staged / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    for field in ac.STAT_FIELDS:
        assert (table.column(f"stats/{IMG}.{ac.NEW_CAMERA}/{field}").to_pylist()
                == [7.0, 8.0])


def test_stale_columns_are_dropped(tmp_path):
    staged, raw = _tree(tmp_path, stale=True)
    dropped = ac.patch_episodes(staged, raw)
    table = pq.read_table(
        staged / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert dropped == len(ac.OFFSET_FIELDS) + len(ac.STAT_FIELDS)
    assert not [c for c in table.column_names if ac.RAW_CAMERA in c]


def test_nothing_to_drop_when_there_is_no_residue(tmp_path):
    staged, raw = _tree(tmp_path, stale=False)
    assert ac.patch_episodes(staged, raw) == 0


def test_cam0_columns_and_row_count_survive(tmp_path):
    """Surgery on a copy must not disturb the camera that was already there."""
    staged, raw = _tree(tmp_path, stale=True)
    before = _episode_table(stale=True)
    ac.patch_episodes(staged, raw)
    table = pq.read_table(
        staged / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert table.num_rows == before.num_rows
    for name in before.column_names:
        if ac.RAW_CAMERA in name:
            continue
        assert table.column(name).to_pylist() == before.column(name).to_pylist()


# ── patch_stats ─────────────────────────────────────────────────────────

def test_global_stats_gain_the_new_camera_only(tmp_path):
    staged, raw = _tree(tmp_path, stale=False)
    ac.patch_stats(staged, raw)
    stats = json.loads((staged / "meta" / "stats.json").read_text())
    assert stats[f"{IMG}.{ac.NEW_CAMERA}"] == {"mean": [[[0.45]]]}
    assert stats[f"{IMG}.cam0"] == {"mean": [[[0.5]]]}
    assert stats["action"] == {"mean": [0.0]}
