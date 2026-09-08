"""The PNG front end: one overlay per camera per frame, plus a summary.

Must work headless — this runs over ssh on the GPU box — so matplotlib is
forced onto Agg and nothing opens a window.
"""
import pathlib
import tempfile

import numpy as np
import pytest

from grabette_attention.frontends.png import write_overlays, write_summary
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


def test_the_filename_carries_the_frame_and_the_camera_name():
    out = pathlib.Path(tempfile.mkdtemp())
    paths = write_overlays(analysis(), observation(), out)
    names = sorted(p.name for p in paths)
    assert names[0] == "frame_00042_cam0_attn.png"
    assert names[1] == "frame_00042_cam1_attn.png"


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
