"""Frame sources and frame selection.

The dump_obs reader is tested against files written the way evaluate.py writes
them (cv2.imwrite, so BGR on disk). The dataset source is exercised in the
GPU integration test; here we test the selection logic that both share.
"""
import json

import numpy as np
import pytest

from grabette_attention.sources import DumpObsSource, select_frames

cv2 = pytest.importorskip("cv2")


def write_dump(tmp_path, n=3):
    for i in range(n):
        # A frame that is unambiguously RED in RGB terms.
        rgb = np.zeros((8, 12, 3), np.uint8)
        rgb[:, :, 0] = 255
        cv2.imwrite(str(tmp_path / f"obs_{i:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    lines = [json.dumps({"step": i, "state": [0.1 * i, 0.2]}) for i in range(n)]
    (tmp_path / "state.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_dump_obs_frames_come_back_as_rgb(tmp_path):
    write_dump(tmp_path)
    source = DumpObsSource(tmp_path, task="pick the sugar cube", camera_key="observation.images.cam0")
    frames = list(source.frames())
    assert len(frames) == 3
    first = frames[0].images["observation.images.cam0"]
    # Red in RGB means channel 0 is hot. If the BGR conversion were skipped,
    # channel 2 would be hot instead.
    assert first[0, 0, 0] == 255
    assert first[0, 0, 2] == 0


def test_dump_obs_pairs_each_frame_with_its_state(tmp_path):
    write_dump(tmp_path)
    source = DumpObsSource(tmp_path, task="t", camera_key="cam0")
    frames = list(source.frames())
    assert frames[2].state[0] == pytest.approx(0.2)
    assert frames[1].frame == 1


def test_grasp_selection_finds_the_first_closure_crossing():
    gripper = np.array([0.0, 0.0, 0.1, 0.6, 0.9, 0.9, 0.2], np.float32)
    indices, note = select_frames(gripper, n_frames=7, mode="grasp")
    assert indices == [3]
    assert "closure" in note


def test_grasp_selection_can_return_a_window_around_the_grasp():
    gripper = np.array([0.0, 0.0, 0.1, 0.6, 0.9], np.float32)
    indices, _ = select_frames(gripper, n_frames=5, mode="grasp", count=3)
    assert indices == [2, 3, 4]


def test_grasp_selection_falls_back_to_a_stride_and_says_so():
    indices, note = select_frames(None, n_frames=10, mode="grasp", count=3)
    assert len(indices) == 3
    assert "stride" in note and "no gripper" in note


def test_a_never_closing_episode_falls_back_to_a_stride():
    gripper = np.zeros(10, np.float32)
    indices, note = select_frames(gripper, n_frames=10, mode="grasp", count=2)
    assert len(indices) == 2
    assert "never" in note


def test_stride_mode_spreads_frames_over_the_episode():
    indices, _ = select_frames(None, n_frames=10, mode="stride", count=5)
    assert indices == [0, 2, 4, 6, 8]


def test_explicit_indices_are_returned_untouched():
    indices, note = select_frames(None, n_frames=10, mode=[2, 7])
    assert indices == [2, 7]
    assert "explicit" in note


def test_an_out_of_range_explicit_index_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        select_frames(None, n_frames=5, mode=[9])


def test_stride_note_reports_actual_count_not_requested():
    indices, note = select_frames(None, n_frames=3, mode="stride", count=5)
    # Should return only 3 frames (all available), not 5
    assert len(indices) == 3
    assert indices == [0, 1, 2]
    # Note should report actual count sampled, not requested count
    assert "3" in note and "5" not in note
