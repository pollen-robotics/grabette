"""Reducing (step, layer, head, query, key) attention to per-camera maps.

The reduction order matters and the mass definition matters. Masses are
fractions of the PREFIX, so cameras plus language sum to 1 — that is what makes
"which camera does the policy rely on" readable without renormalising away the
answer.
"""
import numpy as np
import pytest

from grabette_attention.layout import LetterboxGeometry, TokenLayout
from grabette_attention.reduce import captured_steps, reduce_attention

PATCH = 14


class _SameGeometryEverywhere(dict):
    """Test fixture: the 4:3 live-camera geometry, for whatever camera is asked.

    Real callers pass one entry per camera; these tests mostly use views that
    share a resolution, so a mapping that answers for any name keeps them short.
    `test_cameras_with_different_geometries_are_cropped_independently` uses a
    real per-camera dict, which is the case this mapping must not paper over.
    """

    def __missing__(self, _camera):
        return LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224))


GEOM = _SameGeometryEverywhere()


def layout(n_cams: int, masked=frozenset()) -> TokenLayout:
    return TokenLayout(
        camera_keys=tuple(f"cam{i}" for i in range(n_cams)),
        tokens_per_image=256, grid_rows=16, grid_cols=16,
        language_tokens=200, masked_cameras=masked,
    )


def captures(n_cams: int, steps: int = 10, layers: int = 18, value: float = 1.0):
    keys = n_cams * 256 + 200 + 50          # prefix + the 50 action tokens
    return {
        (s, l): np.full((8, 50, keys), value, np.float32)
        for s in range(steps) for l in range(layers)
    }


def test_uniform_attention_splits_mass_by_token_count():
    cams, lang = reduce_attention(
        captures(2), layout(2), GEOM, patch=PATCH,
    )
    # 256 + 256 image tokens and 200 language tokens, all equal.
    assert cams["cam0"].mass == pytest.approx(256 / 712)
    assert cams["cam1"].mass == pytest.approx(256 / 712)
    assert lang == pytest.approx(200 / 712)


def test_the_masses_sum_to_one_over_the_prefix():
    cams, lang = reduce_attention(captures(3), layout(3), GEOM, patch=PATCH)
    assert sum(c.mass for c in cams.values()) + lang == pytest.approx(1.0)


def test_action_token_attention_is_excluded_from_the_mass():
    # Pile weight onto the action-token keys; the prefix fractions must not move.
    caps = captures(1)
    for w in caps.values():
        w[:, :, 456:] = 100.0
    cams, lang = reduce_attention(caps, layout(1), GEOM, patch=PATCH)
    assert cams["cam0"].mass == pytest.approx(256 / 456)
    assert lang == pytest.approx(200 / 456)


def test_padding_rows_are_cropped_from_the_returned_grid():
    cams, _ = reduce_attention(captures(1), layout(1), GEOM, patch=PATCH)
    # 28 px of padding at patch 14 leaves grid rows 2..13, i.e. 12 rows.
    assert cams["cam0"].grid.shape == (12, 16)


def test_a_masked_camera_is_absent_from_the_result():
    cams, _ = reduce_attention(
        captures(2), layout(2, masked=frozenset({"cam1"})), GEOM, patch=PATCH,
    )
    assert set(cams) == {"cam0"}


def test_the_last_denoising_step_is_the_default():
    caps = captures(1, steps=3, layers=1, value=0.0)
    caps[(2, 0)][:] = 5.0                     # only the last step is hot
    cams, _ = reduce_attention(caps, layout(1), GEOM, patch=PATCH)
    assert cams["cam0"].grid.max() == pytest.approx(5.0)


def test_a_specific_denoising_step_can_be_selected():
    caps = captures(1, steps=3, layers=1, value=0.0)
    caps[(0, 0)][:] = 7.0
    cams, _ = reduce_attention(
        caps, layout(1), GEOM, patch=PATCH, denoise_step=0,
    )
    assert cams["cam0"].grid.max() == pytest.approx(7.0)


def test_layers_can_be_restricted_to_a_subset():
    caps = captures(1, steps=1, layers=4, value=0.0)
    caps[(0, 2)][:] = 3.0
    cams, _ = reduce_attention(
        caps, layout(1), GEOM, patch=PATCH, layers=[2],
    )
    assert cams["cam0"].grid.max() == pytest.approx(3.0)


def test_a_hot_patch_lands_at_the_right_grid_cell():
    # Token 5*16+3 of cam0 is grid row 5, col 3; after cropping 2 pad rows it
    # must appear at row 3.
    caps = captures(1, steps=1, layers=1, value=0.0)
    caps[(0, 0)][:, :, 5 * 16 + 3] = 9.0
    cams, _ = reduce_attention(caps, layout(1), GEOM, patch=PATCH)
    grid = cams["cam0"].grid
    assert np.unravel_index(int(grid.argmax()), grid.shape) == (3, 3)


def test_an_empty_capture_set_is_an_error_not_an_empty_map():
    with pytest.raises(ValueError, match="no attention"):
        reduce_attention({}, layout(1), GEOM, patch=PATCH)


def test_cameras_with_different_geometries_are_cropped_independently():
    # cam0 is 4:3 and padded top and bottom; cam1 is square and unpadded. Each
    # camera's crop must come from ITS OWN geometry, so the two grids differ in
    # height. One shared geometry would give both the same shape and silently
    # crop real content off the square view.
    geometries = {
        "cam0": LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224)),
        "cam1": LetterboxGeometry.from_shapes(src_hw=(480, 480), dst_hw=(224, 224)),
    }
    cams, _ = reduce_attention(captures(2), layout(2), geometries, patch=PATCH)
    assert cams["cam0"].grid.shape == (12, 16)
    assert cams["cam1"].grid.shape == (16, 16)


def test_a_missing_geometry_is_an_error_not_a_guess():
    with pytest.raises(KeyError):
        reduce_attention(captures(2), layout(2), {"cam0": LetterboxGeometry
                         .from_shapes(src_hw=(720, 960), dst_hw=(224, 224))},
                         patch=PATCH)


def test_a_key_axis_that_disagrees_with_the_layout_is_rejected():
    # If the patch size used to derive the layout is wrong, tokens-per-image
    # and grid rows*cols stay mutually consistent BY CONSTRUCTION (both come
    # from the same wrong patch), so the reshape would still succeed and every
    # camera block would be silently misaligned. This must be checked against
    # what was ACTUALLY captured, not re-derived from the layout alone:
    # captures(2) has 2*256+200+50=762 keys, but layout(1) expects a prefix of
    # 456 (+ 50 queries = 506).
    with pytest.raises(ValueError, match="key axis"):
        reduce_attention(captures(2), layout(1), GEOM, patch=PATCH)


def test_denoise_step_all_is_rejected_rather_than_silently_averaged():
    # 'all' must not be a silent synonym for 'mean'. This function returns one
    # grid per camera, so it has no way to express a per-step result; the error
    # has to point at the function that does, or the caller will reach for
    # 'mean' and believe they compared the steps.
    with pytest.raises(ValueError, match="analyse_frame_steps"):
        reduce_attention(captures(1), layout(1), GEOM, patch=PATCH, denoise_step="all")


def test_the_captured_steps_are_reported_in_order():
    # The per-step caller must not have to know how many denoising steps the
    # sampler ran; that is a property of the checkpoint.
    # Out of order, and repeated across layers: the step list must be sorted
    # and deduplicated, since every layer fires once per step.
    grid = np.zeros((1, 1, 1), np.float32)
    caps = {(3, 0): grid, (1, 5): grid, (1, 0): grid, (2, 17): grid}
    assert captured_steps(caps) == [1, 2, 3]


def test_captured_steps_rejects_an_empty_capture():
    with pytest.raises(ValueError, match="no attention was captured"):
        captured_steps({})
