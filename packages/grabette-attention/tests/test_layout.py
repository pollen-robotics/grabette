"""Letterbox geometry, derived from the actual shapes — never hard-coded.

The numbers in these tests are the ones lerobot's resize_with_pad_torch
produces; if this file's arithmetic drifts from it, the overlays land on the
wrong pixels and nothing else catches it.
"""
import pytest

from grabette_attention.layout import LetterboxGeometry


def test_a_four_by_three_frame_is_padded_top_and_bottom():
    # The live camera: 960x720 into a 224 square.
    g = LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224))
    assert g.ratio == pytest.approx(960 / 224)
    assert (g.pad_top, g.pad_bottom) == (28, 28)
    assert (g.pad_left, g.pad_right) == (0, 0)


def test_the_downscaled_training_copy_gives_the_same_padding():
    # 480x360 is the training copy; same aspect, so the same 28-pixel bands.
    g = LetterboxGeometry.from_shapes(src_hw=(360, 480), dst_hw=(224, 224))
    assert (g.pad_top, g.pad_bottom) == (28, 28)
    assert g.ratio == pytest.approx(480 / 224)


def test_the_extra_pixel_goes_to_the_bottom():
    # An odd leftover must not be split evenly; lerobot gives it to bottom/right.
    g = LetterboxGeometry.from_shapes(src_hw=(99, 224), dst_hw=(224, 224))
    assert g.pad_bottom == g.pad_top + 1


def test_a_square_frame_needs_no_padding():
    g = LetterboxGeometry.from_shapes(src_hw=(480, 480), dst_hw=(224, 224))
    assert (g.pad_top, g.pad_bottom, g.pad_left, g.pad_right) == (0, 0, 0, 0)


def test_the_centre_of_the_letterbox_maps_to_the_centre_of_the_source():
    g = LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224))
    x, y = g.to_source_pixel(112.0, 112.0)
    assert x == pytest.approx(480.0, abs=1.0)
    assert y == pytest.approx(360.0, abs=1.0)


def test_the_top_of_the_content_maps_to_the_top_of_the_source():
    g = LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224))
    _, y = g.to_source_pixel(0.0, float(g.pad_top))
    assert y == pytest.approx(0.0, abs=1.0)


def test_padding_rows_are_excluded_from_the_content_rows():
    # 28 px of padding at patch 14 means grid rows 0-1 and 14-15 are pure pad.
    g = LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224))
    assert g.content_rows(patch=14) == (2, 14)


def test_an_unpadded_frame_keeps_every_row():
    g = LetterboxGeometry.from_shapes(src_hw=(480, 480), dst_hw=(224, 224))
    assert g.content_rows(patch=14) == (0, 16)
