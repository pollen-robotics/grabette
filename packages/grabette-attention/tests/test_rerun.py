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


class _Image:
    """Distinguishable image entity that carries its payload."""
    def __init__(self, array):
        self.array = array


class _Scalars:
    """Distinguishable scalar entity that carries its payload."""
    def __init__(self, value):
        self.value = value


class FakeRerun(types.ModuleType):
    def __init__(self):
        super().__init__("rerun")
        self.logged: list[tuple[str, object]] = []
        self.times: list[int] = []

    def init(self, name, spawn=False):
        self.app = name

    def set_time(self, _timeline, *, sequence=None, timestamp=None):
        self.times.append(sequence if sequence is not None else timestamp)

    def log(self, path, entity):
        self.logged.append((path, entity))

    def Image(self, array):             # noqa: N802 - mirrors rerun's API
        return _Image(array)

    def Scalars(self, value):           # noqa: N802
        return _Scalars(value)


@pytest.fixture
def fake_rerun(monkeypatch):
    module = FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", module)
    return module


def analysis() -> FrameAnalysis:
    return FrameAnalysis(
        episode=3, frame=42,
        cameras={
            # Non-constant, so a test can tell "upsampled and normalized" apart
            # from "upsampled and left at its raw, near-black scale".
            "observation.images.cam0": CameraAttention(
                grid=np.linspace(0, 1, 12 * 16, dtype=np.float32).reshape(12, 16),
                mass=0.8,
            )
        },
        language_mass=0.2,
        ablations={
            "observation.images.cam0": ViewAblation(
                delta_mm=8.4, per_axis_mm=(1.0, 2.0, 8.1)
            )
        },
        baseline_mm=42.0,
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

    obs = observation()
    rerun_logger.log_analysis(analysis(), obs)
    logged_dict = {p: e for p, e in fake_rerun.logged}

    # Camera frame should be an Image entity with the observation's frame for cam0
    frame_entity = logged_dict["camera_feed/cam0"]
    assert isinstance(frame_entity, _Image)
    assert np.array_equal(frame_entity.array, obs.images["observation.images.cam0"])

    # Overlay should be an Image entity, upsampled to the FRAME's resolution --
    # the rerun viewer draws a child entity at its own pixel size, so a raw
    # 12x16 grid does not stretch over its 720x960 parent (Finding 5). Asserting
    # only "equals the grid", as this test used to, cannot catch that.
    grid = analysis().cameras["observation.images.cam0"].grid
    overlay_entity = logged_dict["camera_feed/cam0/attention"]
    assert isinstance(overlay_entity, _Image)
    assert overlay_entity.array.shape[:2] == obs.images["observation.images.cam0"].shape[:2]
    assert overlay_entity.array.shape != grid.shape
    # And normalized: raw grid values (of order one over the cell count) sit
    # near the bottom of the default float display range and render
    # near-black; the logged overlay must span the full [0, 1] range instead.
    assert float(overlay_entity.array.min()) == pytest.approx(0.0)
    assert float(overlay_entity.array.max()) == pytest.approx(1.0)


def test_it_logs_mass_and_ablation_as_scalar_series(fake_rerun):
    from grabette_attention.frontends import rerun_logger

    obs = observation()
    ana = analysis()
    rerun_logger.log_analysis(ana, obs)
    logged_dict = {p: e for p, e in fake_rerun.logged}

    # Mass should be a Scalars entity with the camera's mass value
    mass_entity = logged_dict["metrics/mass/cam0"]
    assert isinstance(mass_entity, _Scalars)
    assert mass_entity.value == ana.cameras["observation.images.cam0"].mass

    # Ablation should be a Scalars entity with the delta_mm value
    ablation_entity = logged_dict["metrics/ablation_mm/cam0"]
    assert isinstance(ablation_entity, _Scalars)
    assert ablation_entity.value == ana.ablations["observation.images.cam0"].delta_mm

    # Language mass should be a Scalars entity with the language mass value
    lang_entity = logged_dict["metrics/mass/language"]
    assert isinstance(lang_entity, _Scalars)
    assert lang_entity.value == ana.language_mass


def test_the_frame_index_drives_the_timeline(fake_rerun):
    from grabette_attention.frontends import rerun_logger

    rerun_logger.log_analysis(analysis(), observation())
    assert fake_rerun.times == [42]


def test_a_missing_rerun_install_gives_a_clear_message(monkeypatch):
    from grabette_attention.frontends import rerun_logger

    monkeypatch.setitem(sys.modules, "rerun", None)
    with pytest.raises(ImportError, match="rerun"):
        rerun_logger.log_analysis(analysis(), observation())


def test_two_cameras_sharing_a_trailing_segment_both_get_distinct_entities(fake_rerun):
    # A stereo pair: "observation.images.left.cam0" and
    # "observation.images.right.cam0" both collapse to "cam0" under a naive
    # last-dot-segment split, which would log both under the same entity path
    # and silently lose one view (Finding 4).
    from grabette_attention.frontends import rerun_logger

    left, right = "observation.images.left.cam0", "observation.images.right.cam0"
    ana = FrameAnalysis(
        episode=3, frame=42,
        cameras={
            left: CameraAttention(grid=np.zeros((12, 16), np.float32), mass=0.5),
            right: CameraAttention(grid=np.ones((12, 16), np.float32), mass=0.5),
        },
        language_mass=0.0, ablations={}, baseline_mm=42.0,
    )
    obs = FrameObservation(
        episode=3, frame=42,
        images={
            left: np.zeros((720, 960, 3), np.uint8),
            right: np.zeros((720, 960, 3), np.uint8),
        },
        state=np.zeros(2, np.float32), task="t",
    )
    rerun_logger.log_analysis(ana, obs)
    logged_paths = {p for p, _ in fake_rerun.logged}

    assert left != right
    assert f"camera_feed/{left}" in logged_paths
    assert f"camera_feed/{right}" in logged_paths
    assert "camera_feed/cam0" not in logged_paths
