"""Token index <-> (camera, row, col), for ANY number of cameras.

These tests are the multi-camera acceptance criteria in executable form. If
someone hard-codes 256 tokens or a single view, the two- and three-camera cases
here fail.
"""
import pytest

from grabette_attention.layout import TokenLayout


def one_camera() -> TokenLayout:
    return TokenLayout(
        camera_keys=("cam0",), tokens_per_image=256, grid_rows=16, grid_cols=16,
        language_tokens=200, masked_cameras=frozenset(),
    )


def three_cameras() -> TokenLayout:
    return TokenLayout(
        camera_keys=("cam0", "cam1", "cam2"), tokens_per_image=256,
        grid_rows=16, grid_cols=16, language_tokens=200,
        masked_cameras=frozenset(),
    )


def test_the_single_camera_prefix_is_the_repos_default_length():
    # 256 image tokens + 200 language tokens, the shape a Grabette pi0.5 sees.
    assert one_camera().prefix_len == 456


def test_the_prefix_grows_with_each_camera():
    assert three_cameras().prefix_len == 3 * 256 + 200


def test_each_camera_occupies_its_own_block():
    layout = three_cameras()
    assert layout.camera_slice("cam0") == slice(0, 256)
    assert layout.camera_slice("cam1") == slice(256, 512)
    assert layout.camera_slice("cam2") == slice(512, 768)


def test_language_follows_the_last_camera():
    assert three_cameras().language_slice() == slice(768, 968)


def test_a_token_index_resolves_to_camera_row_and_column():
    layout = three_cameras()
    # Row-major within an image: token 17 of cam1 is row 1, col 1.
    assert layout.token_to_cell(256 + 17) == ("cam1", 1, 1)
    assert layout.token_to_cell(0) == ("cam0", 0, 0)
    assert layout.token_to_cell(255) == ("cam0", 15, 15)


def test_a_language_token_index_is_rejected_as_a_cell():
    layout = one_camera()
    with pytest.raises(ValueError, match="language"):
        layout.token_to_cell(300)


def test_a_non_square_grid_is_handled():
    # Nothing may assume rows == cols.
    layout = TokenLayout(
        camera_keys=("cam0",), tokens_per_image=32, grid_rows=4, grid_cols=8,
        language_tokens=10, masked_cameras=frozenset(),
    )
    assert layout.token_to_cell(9) == ("cam0", 1, 1)
    assert layout.prefix_len == 42


def test_masked_cameras_keep_their_slot_but_are_not_visible():
    # An absent camera still occupies tokens (padded, mask 0). It must keep its
    # slot so other cameras' indices stay right, and be excluded from output.
    layout = TokenLayout(
        camera_keys=("cam0", "cam1"), tokens_per_image=256, grid_rows=16,
        grid_cols=16, language_tokens=200, masked_cameras=frozenset({"cam1"}),
    )
    assert layout.camera_slice("cam1") == slice(256, 512)
    assert layout.visible_cameras() == ("cam0",)


def test_an_unknown_camera_is_an_error_not_a_guess():
    with pytest.raises(KeyError):
        one_camera().camera_slice("wrist")
