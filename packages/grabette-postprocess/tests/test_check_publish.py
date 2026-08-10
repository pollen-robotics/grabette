"""Tests for the publication gate (checks.publish).

Each test reproduces one defect that reached a real dataset, so a regression here
means that defect can ship again. Fixtures are minimal local roots — the Hub-only
checks (git tag) are covered by their skip path, not by network calls.
"""
import json

import pandas as pd
import pytest

from grabette_postprocess.checks.publish import check_publish

CARD = """---
tags:
- LeRobot
---
<a href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=user/ds"></a>
"""


def make_root(
    tmp_path,
    tasks=("pick up the red can",),
    action_names=("dx", "dy", "dz", "proximal", "distal"),
    state_names=("proximal", "distal"),
    action_max=(0.1, 0.1, 0.1, 1.5, 1.4),
    fps=50,
    video_fps=50,
    card=CARD,
    stats=True,
    episode_tasks=None,
):
    """A minimal but structurally valid dataset root."""
    root = tmp_path / "ds"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    info = {
        "fps": fps,
        "codebase_version": "v3.0",
        "total_episodes": 1,
        "total_frames": 100,
        "total_tasks": len(tasks),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.cam0": {
                "dtype": "video", "shape": [3, 360, 480],
                "info": {"video.fps": video_fps},
            },
            "action": {"dtype": "float32", "shape": [len(action_names)],
                       "names": list(action_names)},
            "observation.state": {"dtype": "float32", "shape": [len(state_names)],
                                  "names": list(state_names)},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))

    if stats:
        (root / "meta" / "stats.json").write_text(json.dumps({
            "action": {"min": [0.0] * len(action_names), "max": list(action_max)},
            "observation.state": {"min": [0.0] * len(state_names),
                                  "max": list(action_max[-len(state_names):])},
        }))

    df = pd.DataFrame({"task_index": range(len(tasks))}, index=list(tasks))
    df.index.name = "task"
    df.to_parquet(root / "meta" / "tasks.parquet")

    ep = pd.DataFrame({
        "episode_index": [0],
        "tasks": [list(episode_tasks) if episode_tasks else [tasks[0]]],
        "length": [100],
    })
    ep.to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    if card is not None:
        (root / "README.md").write_text(card)
    return root


def test_a_clean_dataset_passes(tmp_path):
    st = check_publish(make_root(tmp_path))
    assert st["errors"] == []
    assert st["warnings"] == []


# ── task strings ────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["test_pick_mustard_200", "pick_up_the_can",
                                 "grasp_and_lift_cube"])
def test_identifier_task_strings_are_an_error(tmp_path, bad):
    """The real defect: a directory name where a VLA's language prompt belongs."""
    st = check_publish(make_root(tmp_path, tasks=(bad,)))
    assert any("identifier" in e for e in st["errors"]), st


def test_language_task_strings_are_accepted(tmp_path):
    st = check_publish(make_root(tmp_path, tasks=("pick up the mustard bottle",)))
    assert st["errors"] == []


def test_episode_tasks_disagreeing_with_the_mapping_warns(tmp_path):
    """Both pick3 datasets shipped with the episodes column holding recording
    names while tasks.parquet held the language strings."""
    st = check_publish(make_root(tmp_path, tasks=("pick up the red can",),
                                 episode_tasks=("test_pick_can_200",)))
    assert st["errors"] == []
    assert any("disagrees with tasks.parquet" in w for w in st["warnings"]), st


# ── gripper channels ────────────────────────────────────────────────────

def test_unnamed_channels_are_an_error(tmp_path):
    """Unnamed channels make downstream fall back to "the last two", which cannot
    tell 1.0 radian from a normalised full close."""
    root = make_root(tmp_path)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"]["action"]["names"] = None
    (root / "meta" / "info.json").write_text(json.dumps(info))
    st = check_publish(root)
    assert any("unnamed" in e for e in st["errors"]), st


def test_projected_channels_above_one_are_an_error(tmp_path):
    """Named (strategy, closure) but holding radians — the mislabel that would
    send a full close as a partial one."""
    st = check_publish(make_root(
        tmp_path,
        action_names=("dx", "dy", "dz", "strategy", "closure"),
        state_names=("strategy", "closure"),
        action_max=(0.1, 0.1, 0.1, 1.5, 1.4),
    ))
    assert any("cannot leave [0, 1]" in e for e in st["errors"]), st


def test_projected_channels_within_one_pass(tmp_path):
    st = check_publish(make_root(
        tmp_path,
        action_names=("dx", "dy", "dz", "strategy", "closure"),
        state_names=("strategy", "closure"),
        action_max=(0.1, 0.1, 0.1, 0.9, 1.0),
    ))
    assert st["errors"] == []


def test_raw_channels_that_never_exceed_one_are_flagged(tmp_path):
    """A raw dataset's gripper q99 measured 0.894, so range alone cannot separate
    raw from projected — the ambiguity has to reach a human."""
    st = check_publish(make_root(tmp_path, action_max=(0.1, 0.1, 0.1, 0.89, 0.4)))
    assert st["errors"] == []
    assert any("indistinguishable" in w for w in st["warnings"]), st


def test_unknown_channel_names_are_an_error(tmp_path):
    st = check_publish(make_root(tmp_path,
                                 action_names=("dx", "dy", "dz", "g1", "g2"),
                                 state_names=("g1", "g2")))
    assert any("expected" in e for e in st["errors"]), st


# ── card, stats, fps ────────────────────────────────────────────────────

def test_a_missing_card_warns(tmp_path):
    st = check_publish(make_root(tmp_path, card=None))
    assert any("no README.md" in w for w in st["warnings"]), st


def test_a_card_without_the_viewer_link_warns(tmp_path):
    st = check_publish(make_root(tmp_path, card="---\ntags:\n- LeRobot\n---\nhi\n"))
    assert any("viewer" in w for w in st["warnings"]), st


def test_the_wrong_viewer_parameter_warns(tmp_path):
    """`?dataset=` renders a blank page, which reads as a broken dataset."""
    card = ("---\ntags:\n- LeRobot\n---\n"
            "https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=user/ds\n")
    st = check_publish(make_root(tmp_path, card=card))
    assert any("path=" in w for w in st["warnings"]), st


def test_a_card_without_the_lerobot_tag_warns(tmp_path):
    card = ("---\ntags:\n- robotics\n---\n"
            "https://huggingface.co/spaces/lerobot/visualize_dataset?path=user/ds\n")
    st = check_publish(make_root(tmp_path, card=card))
    assert any("LeRobot" in w for w in st["warnings"]), st


def test_missing_stats_are_an_error(tmp_path):
    st = check_publish(make_root(tmp_path, stats=False))
    assert any("stats.json" in e for e in st["errors"]), st


def test_fps_disagreeing_with_the_video_warns(tmp_path):
    """Three different rates were live on this project at once (46 / 50 / 30)."""
    st = check_publish(make_root(tmp_path, fps=50, video_fps=30))
    assert any("video.fps" in w for w in st["warnings"]), st


def test_the_git_tag_check_is_skipped_for_a_local_root(tmp_path):
    """It cannot be answered locally, so it must not masquerade as a failure."""
    st = check_publish(make_root(tmp_path))
    assert any("skipped the git-tag check" in i for i in st["info"])
    assert not any("git tag" in e for e in st["errors"])


def test_missing_info_json_stops_early(tmp_path):
    """Nothing else is meaningful without it, and it must not raise."""
    root = tmp_path / "empty"
    root.mkdir()
    st = check_publish(root)
    assert any("info.json unreadable" in e for e in st["errors"])


def test_a_nonexistent_target_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_publish(tmp_path / "nope")
