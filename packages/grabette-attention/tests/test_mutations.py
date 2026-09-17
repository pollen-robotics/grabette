"""Assertions that would break under the specific mistakes the spec names.

Each test here corresponds to a line in the spec's mutation list. They overlap
with earlier tests on purpose: this file is where a reviewer looks to confirm
the genericity requirements are actually enforced.
"""
import numpy as np
import pytest

from grabette_attention.layout import LetterboxGeometry, TokenLayout
from grabette_attention.metrics import translation_delta
from grabette_attention.reduce import reduce_attention


class _SameGeometryEverywhere(dict):
    """The 4:3 live-camera geometry, for whatever camera is asked."""

    def __missing__(self, _camera):
        return LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224))


GEOM = _SameGeometryEverywhere()


def test_a_hard_coded_256_token_block_would_break_a_smaller_grid():
    # A 112-pixel input at patch 14 gives an 8x8 grid, 64 tokens.
    layout = TokenLayout(
        camera_keys=("cam0",), tokens_per_image=64, grid_rows=8, grid_cols=8,
        language_tokens=10, masked_cameras=frozenset(),
    )
    caps = {(0, 0): np.ones((8, 50, 64 + 10 + 50), np.float32)}
    geoms = {
        "cam0": LetterboxGeometry.from_shapes(src_hw=(112, 112), dst_hw=(112, 112))
    }
    cams, _ = reduce_attention(caps, layout, geoms, patch=14)
    assert cams["cam0"].grid.shape == (8, 8)


def test_a_hard_coded_single_camera_would_break_a_three_camera_prefix():
    layout = TokenLayout(
        camera_keys=("a", "b", "c"), tokens_per_image=256, grid_rows=16,
        grid_cols=16, language_tokens=200, masked_cameras=frozenset(),
    )
    caps = {(0, 0): np.ones((8, 50, 3 * 256 + 200 + 50), np.float32)}
    cams, lang = reduce_attention(caps, layout, GEOM, patch=14)
    assert set(cams) == {"a", "b", "c"}
    assert sum(c.mass for c in cams.values()) + lang == pytest.approx(1.0)


def test_transposing_the_token_grid_would_move_the_hot_cell():
    # Row-major: token r*cols + c is (row r, col c). A column-major reshape puts
    # this peak at (3, 5) instead of (5, 3) -> (3, 3) after cropping 2 pad rows.
    layout = TokenLayout(
        camera_keys=("cam0",), tokens_per_image=256, grid_rows=16, grid_cols=16,
        language_tokens=200, masked_cameras=frozenset(),
    )
    caps = {(0, 0): np.zeros((8, 50, 456 + 50), np.float32)}
    caps[(0, 0)][:, :, 5 * 16 + 3] = 1.0
    cams, _ = reduce_attention(caps, layout, GEOM, patch=14)
    grid = cams["cam0"].grid
    assert np.unravel_index(int(grid.argmax()), grid.shape) == (3, 3)


def test_renormalising_per_camera_would_hide_which_camera_is_used():
    # cam0 gets nine times cam1's attention. Per-camera renormalisation would
    # make both maps look identical and both masses 1.0.
    layout = TokenLayout(
        camera_keys=("cam0", "cam1"), tokens_per_image=256, grid_rows=16,
        grid_cols=16, language_tokens=200, masked_cameras=frozenset(),
    )
    caps = {(0, 0): np.zeros((8, 50, 2 * 256 + 200 + 50), np.float32)}
    caps[(0, 0)][:, :, 0:256] = 0.9
    caps[(0, 0)][:, :, 256:512] = 0.1
    cams, _ = reduce_attention(caps, layout, GEOM, patch=14)
    assert cams["cam0"].mass > 8 * cams["cam1"].mass


def test_padding_rows_must_not_be_plotted_as_content():
    layout = TokenLayout(
        camera_keys=("cam0",), tokens_per_image=256, grid_rows=16, grid_cols=16,
        language_tokens=200, masked_cameras=frozenset(),
    )
    caps = {(0, 0): np.zeros((8, 50, 456 + 50), np.float32)}
    caps[(0, 0)][:, :, 0:32] = 1.0            # grid rows 0-1: pure padding
    cams, _ = reduce_attention(caps, layout, GEOM, patch=14)
    assert cams["cam0"].grid.max() == 0.0     # cropped away entirely


def test_scaling_the_delta_twice_would_double_the_millimetres():
    baseline = np.zeros((4, 11), np.float32)
    ablated = baseline.copy()
    ablated[:, 0] = 0.001
    assert translation_delta(baseline, ablated).delta_mm == pytest.approx(1.0)
