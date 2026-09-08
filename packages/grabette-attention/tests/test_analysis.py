"""Orchestration: one capture pass, then one ablation pass per camera.

Tested against a fake adapter, so this file is about wiring and pass-counting
rather than about pi0.5. The pass count matters: the baseline must be reused,
not recomputed per camera.
"""
import numpy as np
import pytest

from grabette_attention.analysis import analyse, analyse_frame
from grabette_attention.adapters.base import RunResult
from grabette_attention.layout import LetterboxGeometry, TokenLayout
from grabette_attention.records import FrameObservation


class FakeAdapter:
    """Two cameras; only cam0 influences the chunk."""

    patch = 14

    def __init__(self, cameras=("cam0", "cam1")):
        self._cameras = cameras
        self.calls: list[tuple[bool, str | None]] = []
        self.draw_noise_count = 0

    @property
    def camera_keys(self):
        return tuple(self._cameras)

    def geometry(self, obs, camera):
        frame = obs.images[camera]
        return LetterboxGeometry.from_shapes(
            src_hw=(frame.shape[0], frame.shape[1]), dst_hw=(224, 224)
        )

    def layout(self, obs, *, drop_camera=None):
        masked = {drop_camera} if drop_camera else set()
        return TokenLayout(
            camera_keys=tuple(self._cameras), tokens_per_image=256,
            grid_rows=16, grid_cols=16, language_tokens=200,
            masked_cameras=frozenset(masked),
        )

    def draw_noise(self):
        self.draw_noise_count += 1
        return "fixed-noise"

    def run(self, obs, noise, *, capture, drop_camera=None):
        assert noise == "fixed-noise"          # the same noise every pass
        self.calls.append((capture, drop_camera))
        chunk = np.zeros((50, 11), np.float32)
        if drop_camera == "cam0":
            chunk[:, 2] = 0.004                # 4 mm on z when cam0 is removed
        captures = {}
        if capture:
            # Must match THIS adapter's own camera count, not a hard-coded
            # two: reduce_attention now asserts the captured key axis against
            # the derived layout (Finding 2), and a single-camera adapter
            # reporting a two-camera-shaped capture is exactly the mismatch
            # that check exists to catch.
            keys = len(self._cameras) * 256 + 200 + 50
            captures = {(0, 0): np.ones((8, 50, keys), np.float32)}
        return RunResult(chunk=chunk, captures=captures)


def observation():
    return FrameObservation(
        episode=3, frame=42,
        images={c: np.zeros((720, 960, 3), np.uint8) for c in ("cam0", "cam1")},
        state=np.zeros(2, np.float32), task="pick the sugar cube",
    )


def test_it_produces_one_map_and_one_ablation_per_camera():
    adapter = FakeAdapter()
    result = analyse_frame(adapter, observation())
    assert set(result.cameras) == {"cam0", "cam1"}
    assert set(result.ablations) == {"cam0", "cam1"}


def test_the_baseline_runs_once_and_each_ablation_once():
    adapter = FakeAdapter()
    analyse_frame(adapter, observation())
    assert adapter.calls == [
        (True, None), (False, "cam0"), (False, "cam1"),
    ]


def test_the_ablation_delta_is_reported_in_millimetres_per_axis():
    result = analyse_frame(FakeAdapter(), observation())
    assert result.ablations["cam0"].delta_mm == pytest.approx(4.0)
    assert result.ablations["cam0"].per_axis_mm == pytest.approx((0.0, 0.0, 4.0))
    assert result.ablations["cam1"].delta_mm == pytest.approx(0.0)


def test_ablation_can_be_switched_off():
    adapter = FakeAdapter()
    result = analyse_frame(adapter, observation(), ablate=False)
    assert result.ablations == {}
    assert adapter.calls == [(True, None)]


def test_provenance_records_how_the_map_was_made():
    result = analyse_frame(
        FakeAdapter(), observation(),
        denoise_step="last", layers="all",
        provenance={"checkpoint": "user/model_best"},
    )
    assert result.provenance["checkpoint"] == "user/model_best"
    assert result.provenance["denoise_step"] == "last"
    assert result.provenance["layers"] == "all"


def test_the_episode_and_frame_are_carried_through():
    result = analyse_frame(FakeAdapter(), observation())
    assert (result.episode, result.frame) == (3, 42)


def test_analyse_streams_over_many_frames():
    adapter = FakeAdapter()
    results = list(analyse(adapter, [observation(), observation()]))
    assert len(results) == 2


def test_a_single_camera_policy_needs_no_special_case():
    adapter = FakeAdapter(cameras=("cam0",))
    obs = FrameObservation(
        episode=0, frame=0,
        images={"cam0": np.zeros((720, 960, 3), np.uint8)},
        state=np.zeros(2, np.float32), task="t",
    )
    result = analyse_frame(adapter, obs)
    assert set(result.cameras) == {"cam0"}
    assert result.cameras["cam0"].mass == pytest.approx(256 / 456)


def test_noise_is_drawn_once_per_frame():
    adapter = FakeAdapter()
    analyse_frame(adapter, observation())
    assert adapter.draw_noise_count == 1


def test_no_usable_camera_raises_without_running_the_model():
    adapter = FakeAdapter(cameras=())  # No cameras
    obs = FrameObservation(
        episode=0, frame=0,
        images={},
        state=np.zeros(2, np.float32), task="t",
    )
    with pytest.raises(ValueError, match="no usable camera"):
        analyse_frame(adapter, obs)
    # The check happens before any run(), so no calls should be recorded
    assert adapter.calls == []


def test_provenance_always_records_what_was_actually_computed():
    # Even if caller passes provenance with denoise_step/layers, the actual
    # values used take precedence (unconditional assignment, not setdefault).
    result = analyse_frame(
        FakeAdapter(), observation(),
        denoise_step="first", layers=[0, 1],
        provenance={"denoise_step": "wrong", "layers": "wrong", "model": "v1"},
    )
    assert result.provenance["denoise_step"] == "first"
    assert result.provenance["layers"] == "[0, 1]"
    assert result.provenance["model"] == "v1"  # caller values preserved
