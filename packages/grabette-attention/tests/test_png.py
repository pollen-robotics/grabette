"""The PNG front end: one overlay per camera per frame, plus a summary.

Must work headless — this runs over ssh on the GPU box — so matplotlib is
forced onto Agg and nothing opens a window.
"""
import pathlib
import tempfile

import numpy as np
import pytest

from grabette_attention.frontends.png import (
    _colour_limits,
    write_overlays,
    write_summary,
)
from grabette_attention.records import (
    CameraAttention,
    FrameAnalysis,
    FrameObservation,
    ViewAblation,
)

pytest.importorskip("matplotlib")


def analysis(cameras=("cam0", "cam1")) -> FrameAnalysis:
    return FrameAnalysis(
        episode=3, frame=42,
        cameras={
            c: CameraAttention(
                grid=np.linspace(0, 1, 12 * 16, dtype=np.float32).reshape(12, 16),
                mass=0.4,
            )
            for c in cameras
        },
        language_mass=0.2,
        ablations={c: ViewAblation(delta_mm=8.4, per_axis_mm=(1.0, 2.0, 8.1)) for c in cameras},
        baseline_mm=42.0,
        provenance={"checkpoint": "user/model_best", "denoise_step": "last"},
    )


def observation(cameras=("cam0", "cam1")) -> FrameObservation:
    return FrameObservation(
        episode=3, frame=42,
        images={c: np.zeros((720, 960, 3), np.uint8) for c in cameras},
        state=np.zeros(2, np.float32), task="pick the sugar cube",
    )


def test_one_overlay_is_written_per_camera():
    out = pathlib.Path(tempfile.mkdtemp())
    paths = write_overlays(analysis(), observation(), out)
    assert len(paths) == 2
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_the_filename_carries_the_frame_the_camera_name_and_the_step():
    out = pathlib.Path(tempfile.mkdtemp())
    paths = write_overlays(analysis(), observation(), out)
    names = sorted(p.name for p in paths)
    assert names[0] == "frame_00042_cam0_steplast_attn.png"
    assert names[1] == "frame_00042_cam1_steplast_attn.png"


def test_per_step_overlays_of_one_frame_do_not_overwrite_each_other():
    # A per-step sweep writes many records for the SAME frame and camera. If
    # the step were missing from the name they would all collide on one path
    # and the sweep would silently produce a single file.
    out = pathlib.Path(tempfile.mkdtemp())
    written = []
    for step in (0, 7, 10):
        record = analysis(("cam0",))
        record.provenance["denoise_step"] = str(step)
        written += write_overlays(record, observation(("cam0",)), out)
    assert len({p.name for p in written}) == 3
    # Zero-padded so a listing sorts in step order, not lexically.
    assert sorted(p.name for p in written) == [
        "frame_00042_cam0_step00_attn.png",
        "frame_00042_cam0_step07_attn.png",
        "frame_00042_cam0_step10_attn.png",
    ]


def test_a_sink_cell_does_not_take_over_the_colour_scale():
    # The defect this guards: one register cell an order of magnitude above
    # everything else made the ramp span [0, peak], rendering the remaining
    # 94% of the mass as flat black. The clip must sit near the bulk of the
    # data, not at the outlier.
    grid = np.full((12, 16), 0.001, np.float32)
    grid[10, 1] = 0.036                        # 36x the median, as measured
    vmin, vmax = _colour_limits(grid)
    assert vmax < 0.036
    assert vmax == pytest.approx(0.001, abs=1e-4)


def test_a_uniform_grid_still_gets_a_usable_colour_range():
    # A percentile of a constant grid equals its minimum, which would give a
    # degenerate all-one-colour ramp or a matplotlib error.
    vmin, vmax = _colour_limits(np.full((4, 4), 0.5, np.float32))
    assert vmax > vmin


def test_a_three_camera_frame_writes_three_overlays():
    out = pathlib.Path(tempfile.mkdtemp())
    cams = ("cam0", "cam1", "wrist")
    paths = write_overlays(analysis(cams), observation(cams), out)
    assert len(paths) == 3


def test_the_summary_reports_mass_and_millimetres_per_camera():
    out = pathlib.Path(tempfile.mkdtemp())
    path = write_summary([analysis()], out)
    text = path.read_text()
    assert "cam0" in text and "cam1" in text
    assert "0.40" in text          # mass
    assert "8.4" in text           # ablation delta in mm
    assert "mm" in text


def test_the_summary_labels_each_axis_with_its_physical_meaning():
    # Three bare millimetre figures invite being read against the wrong axis.
    # Vertical and depth behave very differently in this action space, so a
    # positional 'xyz' label is not enough.
    out = pathlib.Path(tempfile.mkdtemp())
    text = write_summary([analysis()], out).read_text()
    assert "y/vert 2.0" in text     # per_axis_mm=(1.0, 2.0, 8.1)
    assert "z/depth 8.1" in text
    assert "x/lat 1.0" in text


def test_the_summary_names_the_language_mass_separately():
    out = pathlib.Path(tempfile.mkdtemp())
    text = write_summary([analysis()], out).read_text()
    assert "language" in text
    assert "0.20" in text


def test_the_summary_carries_the_provenance():
    out = pathlib.Path(tempfile.mkdtemp())
    text = write_summary([analysis()], out).read_text()
    assert "user/model_best" in text
    assert "denoise_step" in text


def test_the_summary_repeats_the_frame_selection_note():
    out = pathlib.Path(tempfile.mkdtemp())
    text = write_summary(
        [analysis()], out, notes={3: "gripper never closes; even stride of 3"}
    ).read_text()
    assert "never closes" in text


def test_the_summary_warns_that_a_broad_map_is_normal():
    # The review's interpretation guard, in the artefact itself.
    out = pathlib.Path(tempfile.mkdtemp())
    text = write_summary([analysis()], out).read_text()
    assert "broad" in text.lower()


def test_two_cameras_sharing_a_trailing_segment_both_survive():
    # A stereo pair is exactly the scenario this package exists to serve:
    # "observation.images.left.cam0" and "observation.images.right.cam0" both
    # collapse to "cam0" under a naive last-dot-segment split, which would
    # write both overlays to the same filename and silently drop one.
    cams = ("observation.images.left.cam0", "observation.images.right.cam0")
    out = pathlib.Path(tempfile.mkdtemp())
    paths = write_overlays(analysis(cams), observation(cams), out)
    assert len(paths) == 2
    assert len({p.name for p in paths}) == 2   # distinct filenames
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
