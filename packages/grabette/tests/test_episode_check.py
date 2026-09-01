"""Local completeness screen for a recorded episode (episode_check).

The failure this guards against is specific and expensive: an episode that
LOOKS recorded but lacks an artifact the conversion needs is uploaded in full,
rejected on the Space, and — in a bimanual build — takes the peer arm's good
recording down with it. The screen has to catch that before the upload, so the
required set is pinned here against the pipeline's own contract.
"""
from grabette.episode_check import REQUIRED_FILES, missing_files


def _write_episode(tmp_path, *, skip=(), depth="mkv"):
    """A complete episode dir, minus whatever `skip` names."""
    ep = tmp_path / "20250101_120000"
    ep.mkdir()
    for name in REQUIRED_FILES:
        if name in skip:
            continue
        (ep / name).write_bytes(b"x")
    if depth == "mkv" and "oakd_depth.mkv" not in skip:
        (ep / "oakd_depth.mkv").write_bytes(b"x")
    elif depth == "dir":
        d = ep / "oakd_depth"
        d.mkdir()
        (d / "00000000.png").write_bytes(b"x")
    return ep


def test_complete_episode_has_nothing_missing(tmp_path):
    assert missing_files(_write_episode(tmp_path)) == []


def test_depth_png_sequence_satisfies_depth(tmp_path):
    # stop_recording muxes the PNG dir into the .mkv, so an episode may carry
    # either form depending on when it was recorded. Both are convertible.
    assert missing_files(_write_episode(tmp_path, depth="dir")) == []


def test_missing_calibration_is_reported_by_name(tmp_path):
    # THE regression: this is the file whose silent absence sent a whole session
    # to the Space only to be rejected there, episode by episode.
    ep = _write_episode(tmp_path, skip=("oakd_calib_offline.json",))

    assert missing_files(ep) == ["oakd_calib_offline.json"]


def test_missing_depth_in_both_forms_is_reported(tmp_path):
    ep = _write_episode(tmp_path, skip=("oakd_depth.mkv",), depth="none")

    assert "oakd_depth.mkv" in missing_files(ep)


def test_zero_byte_file_counts_as_missing(tmp_path):
    # An interrupted recording leaves an empty video behind; upstream rejects it
    # exactly like an absent one, so it must not pass the screen.
    ep = _write_episode(tmp_path)
    (ep / "oakd_left.mp4").write_bytes(b"")

    assert missing_files(ep) == ["oakd_left.mp4"]


def test_empty_depth_dir_counts_as_missing(tmp_path):
    ep = _write_episode(tmp_path, skip=("oakd_depth.mkv",), depth="none")
    (ep / "oakd_depth").mkdir()

    assert "oakd_depth.mkv" in missing_files(ep)


def test_several_missing_files_are_all_reported(tmp_path):
    ep = _write_episode(tmp_path, skip=("oakd_imu.json", "angle_data.json"))

    assert missing_files(ep) == ["angle_data.json", "oakd_imu.json"]
