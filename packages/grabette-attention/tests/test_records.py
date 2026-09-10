"""The records are plain data: keyed by camera NAME, never by position."""
import dataclasses

import numpy as np
import pytest

from grabette_attention.records import (
    CameraAttention,
    FrameAnalysis,
    FrameObservation,
    ViewAblation,
)


def test_frame_observation_keeps_cameras_keyed_by_name():
    obs = FrameObservation(
        episode=3,
        frame=42,
        images={"observation.images.cam0": np.zeros((720, 960, 3), np.uint8)},
        state=np.zeros(2, np.float32),
        task="pick the sugar cube",
    )
    assert list(obs.images) == ["observation.images.cam0"]
    assert obs.images["observation.images.cam0"].shape == (720, 960, 3)


def test_frame_analysis_holds_one_entry_per_camera_and_is_immutable():
    cams = {
        "cam0": CameraAttention(grid=np.ones((12, 16), np.float32), mass=0.7),
        "cam1": CameraAttention(grid=np.ones((12, 16), np.float32), mass=0.1),
    }
    analysis = FrameAnalysis(
        episode=3,
        frame=42,
        cameras=cams,
        language_mass=0.2,
        ablations={"cam0": ViewAblation(delta_mm=8.4, per_axis_mm=(1.0, 2.0, 8.1))},
        baseline_mm=42.0,
        provenance={"checkpoint": "x", "denoise_step": "last"},
    )
    assert set(analysis.cameras) == {"cam0", "cam1"}
    assert analysis.ablations["cam0"].delta_mm == 8.4
    with pytest.raises(dataclasses.FrozenInstanceError):
        analysis.episode = 4


def test_masses_over_the_prefix_sum_to_one():
    cams = {
        "cam0": CameraAttention(grid=np.ones((12, 16), np.float32), mass=0.55),
        "cam1": CameraAttention(grid=np.ones((12, 16), np.float32), mass=0.25),
    }
    analysis = FrameAnalysis(
        episode=0, frame=0, cameras=cams, language_mass=0.20,
        ablations={}, provenance={},
        baseline_mm=42.0,
    )
    total = sum(c.mass for c in analysis.cameras.values()) + analysis.language_mass
    assert total == pytest.approx(1.0)
