"""The ablation metric, and its units.

We have already shipped a metres-labelled-as-millimetres bug in the offline
gate, and then over-corrected it by scaling twice. So the unit is pinned by a
test with a hand-computed answer, not by a comment.
"""
import numpy as np
import pytest

from grabette_attention.metrics import translation_delta


def test_a_one_millimetre_shift_reads_as_one_millimetre():
    baseline = np.zeros((4, 11), np.float32)
    ablated = baseline.copy()
    ablated[:, 0] = 0.001                     # 1 mm on x, every step, in METRES
    result = translation_delta(baseline, ablated)
    assert result.delta_mm == pytest.approx(1.0)
    assert result.per_axis_mm[0] == pytest.approx(1.0)


def test_an_identical_chunk_gives_exactly_zero():
    chunk = np.random.default_rng(0).normal(size=(50, 11)).astype(np.float32)
    result = translation_delta(chunk, chunk.copy())
    assert result.delta_mm == 0.0
    assert result.per_axis_mm == (0.0, 0.0, 0.0)


def test_the_per_axis_breakdown_isolates_the_vertical_component():
    # The question we care about: does removing a view change HEIGHT?
    baseline = np.zeros((10, 11), np.float32)
    ablated = baseline.copy()
    ablated[:, 2] = 0.005                     # 5 mm on z only
    result = translation_delta(baseline, ablated)
    assert result.per_axis_mm == pytest.approx((0.0, 0.0, 5.0))
    assert result.delta_mm == pytest.approx(5.0)


def test_the_norm_combines_axes_in_quadrature():
    baseline = np.zeros((3, 11), np.float32)
    ablated = baseline.copy()
    ablated[:, 0] = 0.003
    ablated[:, 1] = 0.004                     # 3-4-5 triangle
    assert translation_delta(baseline, ablated).delta_mm == pytest.approx(5.0)


def test_it_is_an_rms_over_chunk_steps_not_a_sum():
    # A difference on half the steps must not scale with chunk length.
    baseline = np.zeros((4, 11), np.float32)
    ablated = baseline.copy()
    ablated[:2, 0] = 0.002
    # RMS over 4 steps of (2, 2, 0, 0) mm = sqrt((4+4)/4) = sqrt(2).
    assert translation_delta(baseline, ablated).delta_mm == pytest.approx(
        np.sqrt(2.0)
    )


def test_an_eight_dimensional_chunk_relative_action_works_too():
    # Chunk-relative checkpoints emit 8 dims, not 11. The first three are still
    # translation, so the metric must not assume a width.
    baseline = np.zeros((5, 8), np.float32)
    ablated = baseline.copy()
    ablated[:, 1] = 0.002
    assert translation_delta(baseline, ablated).per_axis_mm == pytest.approx(
        (0.0, 2.0, 0.0)
    )


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        translation_delta(np.zeros((4, 11)), np.zeros((5, 11)))
