"""The second front end, against a fake rerun module.

We do not want the tests to depend on rerun being installed, and we do want to
assert the entity paths and that BOTH front ends read the same records.
"""
import sys
import types

import numpy as np
import pytest

from grabette_attention.records import (
    CameraAttention,
    FrameAnalysis,
    FrameObservation,
    ViewAblation,
)


class FakeRerun(types.ModuleType):
    def __init__(self):
        super().__init__("rerun")
        self.logged: list[tuple[str, str]] = []
        self.times: list[int] = []

    def init(self, name, spawn=False):
        self.app = name

    def set_time(self, _timeline, *, sequence=None, timestamp=None):
        self.times.append(sequence if sequence is not None else timestamp)

    def log(self, path, entity):
        self.logged.append((path, type(entity).__name__))

    def Image(self, _array):            # noqa: N802 - mirrors rerun's API
        return types.SimpleNamespace()

    def Scalars(self, _value):          # noqa: N802
        return types.SimpleNamespace()


@pytest.fixture
def fake_rerun(monkeypatch):
    module = FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", module)
    return module


def analysis() -> FrameAnalysis:
    return FrameAnalysis(
        episode=3, frame=42,
        cameras={
            "observation.images.cam0": CameraAttention(
                grid=np.ones((12, 16), np.float32), mass=0.8
            )
        },
        language_mass=0.2,
        ablations={
            "observation.images.cam0": ViewAblation(
                delta_mm=8.4, per_axis_mm=(1.0, 2.0, 8.1)
            )
        },
        provenance={"checkpoint": "user/m"},
    )


def observation() -> FrameObservation:
    return FrameObservation(
        episode=3, frame=42,
        images={"observation.images.cam0": np.zeros((720, 960, 3), np.uint8)},
        state=np.zeros(2, np.float32), task="t",
    )


def test_it_logs_the_frame_and_the_overlay_under_the_camera_entity(fake_rerun):
    from grabette_attention.frontends import rerun_logger

    rerun_logger.log_analysis(analysis(), observation())
    paths = [p for p, _ in fake_rerun.logged]
    assert "camera_feed/cam0" in paths
    assert "camera_feed/cam0/attention" in paths


def test_it_logs_mass_and_ablation_as_scalar_series(fake_rerun):
    from grabette_attention.frontends import rerun_logger

    rerun_logger.log_analysis(analysis(), observation())
    paths = [p for p, _ in fake_rerun.logged]
    assert "metrics/mass/cam0" in paths
    assert "metrics/ablation_mm/cam0" in paths
    assert "metrics/mass/language" in paths


def test_the_frame_index_drives_the_timeline(fake_rerun):
    from grabette_attention.frontends import rerun_logger

    rerun_logger.log_analysis(analysis(), observation())
    assert fake_rerun.times == [42]


def test_a_missing_rerun_install_gives_a_clear_message(monkeypatch):
    from grabette_attention.frontends import rerun_logger

    monkeypatch.setitem(sys.modules, "rerun", None)
    with pytest.raises(ImportError, match="rerun"):
        rerun_logger.log_analysis(analysis(), observation())
