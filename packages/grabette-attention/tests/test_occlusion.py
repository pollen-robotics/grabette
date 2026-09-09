"""Occlusion saliency: cover a region, re-run, measure the chunk's movement.

Tested against a fake adapter whose chunk depends on a KNOWN region of the
image, so the sweep must find that region and no other. The pass count and the
noise sharing are checked too: a per-block noise draw would make every cell
measure sampling variance instead of the region's effect.
"""
import numpy as np
import pytest

from grabette_attention.adapters.base import RunResult
from grabette_attention.occlusion import occlusion_saliency
from grabette_attention.records import FrameObservation

CAM = "cam0"
# The fake is sensitive to this source-pixel box only.
HOT = (slice(0, 60), slice(0, 120))


class FakeAdapter:
    """Chunk translation responds only to the mean brightness of HOT."""

    def __init__(self):
        self.runs: list[np.ndarray] = []
        self.noise_draws = 0

    @property
    def camera_keys(self):
        return (CAM,)

    def draw_noise(self):
        self.noise_draws += 1
        return "fixed-noise"

    def run(self, obs, noise, *, capture, drop_camera=None):
        assert noise == "fixed-noise", "every pass must share one noise tensor"
        self.runs.append(obs.images[CAM])
        chunk = np.zeros((10, 8), np.float32)
        # 1 mm of z per unit of mean brightness in the hot box, in metres.
        chunk[:, 2] = obs.images[CAM][HOT].mean() * 1e-5
        return RunResult(chunk=chunk)


def observation(value=200):
    image = np.full((240, 320, 3), value, np.uint8)
    return FrameObservation(
        episode=0, frame=7, images={CAM: image},
        state=np.zeros(2, np.float32), task="t",
    )


def test_the_grid_has_the_requested_shape():
    result = occlusion_saliency(
        FakeAdapter(), observation(), camera=CAM, rows=4, cols=6
    )
    assert result.grid.shape == (4, 6)
    assert result.camera == CAM
    assert result.fill == "mean"


def test_it_finds_the_region_the_action_depends_on():
    # HOT covers source rows 0-60 of 240 and cols 0-120 of 320, so on a 4x4
    # grid that is cell (0,0) and part of (0,1). Those must dominate.
    result = occlusion_saliency(
        FakeAdapter(), observation(), camera=CAM, rows=4, cols=4, fill="black"
    )
    grid = result.grid
    assert grid[0, 0] > 0
    assert grid[0, 1] > 0
    # Everything outside the top-left band must be untouched.
    assert grid[1:, :].max() == pytest.approx(0.0)
    assert grid[0, 2:].max() == pytest.approx(0.0)
    assert grid[0, 0] > 10 * max(grid[1:, :].max(), 1e-9)


def test_one_pass_per_block_plus_one_baseline():
    adapter = FakeAdapter()
    occlusion_saliency(adapter, observation(), camera=CAM, rows=3, cols=5)
    assert len(adapter.runs) == 3 * 5 + 1


def test_the_noise_is_drawn_once_for_the_whole_sweep():
    # A draw per block would make each cell measure sampling variance rather
    # than the region's causal effect -- the same guarantee the view ablation
    # depends on.
    adapter = FakeAdapter()
    occlusion_saliency(adapter, observation(), camera=CAM, rows=3, cols=3)
    assert adapter.noise_draws == 1


def test_a_supplied_noise_tensor_is_used_and_none_is_drawn():
    adapter = FakeAdapter()
    occlusion_saliency(
        adapter, observation(), camera=CAM, rows=2, cols=2, noise="fixed-noise"
    )
    assert adapter.noise_draws == 0


def test_the_callers_frame_is_never_mutated():
    # fill must DIFFER from the frame's own pixels, or this cannot detect a
    # mutation: with the default "mean" fill on a uniform frame the fill value
    # equals the pixels it replaces, and an in-place write is invisible.
    obs = observation()
    original = obs.images[CAM].copy()
    occlusion_saliency(FakeAdapter(), obs, camera=CAM, rows=3, cols=3, fill="black")
    assert np.array_equal(obs.images[CAM], original)


def test_the_baseline_magnitude_is_reported_in_millimetres():
    # The fake's baseline chunk is 200 * 1e-5 m = 2 mm of z on every step.
    result = occlusion_saliency(
        FakeAdapter(), observation(value=200), camera=CAM, rows=2, cols=2
    )
    assert result.baseline_mm == pytest.approx(2.0)


def test_an_unknown_camera_is_rejected():
    with pytest.raises(KeyError, match="cam9"):
        occlusion_saliency(FakeAdapter(), observation(), camera="cam9", rows=2, cols=2)


def test_a_degenerate_grid_is_rejected():
    with pytest.raises(ValueError, match="at least 1x1"):
        occlusion_saliency(FakeAdapter(), observation(), camera=CAM, rows=0, cols=4)


def test_an_unknown_fill_is_rejected():
    with pytest.raises(ValueError, match="unknown fill"):
        occlusion_saliency(
            FakeAdapter(), observation(), camera=CAM, rows=2, cols=2, fill="chartreuse"
        )


def test_the_mean_fill_preserves_overall_brightness():
    # A uniform frame's mean equals its own value, so covering a block with the
    # mean changes nothing and every cell reads zero. That is the point of the
    # default: it removes local structure, not global brightness.
    result = occlusion_saliency(
        FakeAdapter(), observation(value=200), camera=CAM, rows=3, cols=3, fill="mean"
    )
    assert result.grid.max() == pytest.approx(0.0)


def test_grid_cells_tile_the_source_image_without_gaps():
    # A covered cell must actually differ from the original somewhere, for
    # every cell -- an off-by-one in the block arithmetic would silently leave
    # edge cells uncovered and reading zero for the wrong reason.
    adapter = FakeAdapter()
    obs = observation()
    occlusion_saliency(adapter, obs, camera=CAM, rows=5, cols=7, fill="black")
    covered = adapter.runs[1:]           # index 0 is the baseline
    assert len(covered) == 35
    union = np.zeros(obs.images[CAM].shape[:2], bool)
    for image in covered:
        union |= (image != obs.images[CAM]).any(axis=2)
    assert union.all(), "some source pixels were never covered by any block"
