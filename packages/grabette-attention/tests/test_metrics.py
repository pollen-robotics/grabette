"""The ablation metric, and its units.

We have already shipped a metres-labelled-as-millimetres bug in the offline
gate, and then over-corrected it by scaling twice. So the unit is pinned by a
test with a hand-computed answer, not by a comment.
"""
import numpy as np
import pytest

from grabette_attention.metrics import translation_delta, translation_magnitude_mm


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


def test_the_per_axis_breakdown_keeps_each_channel_in_its_own_slot():
    # Channel order is load-bearing: slot 1 is VERTICAL and slot 2 is DEPTH in
    # this action space (see ViewAblation), and the two answer different
    # questions. A breakdown that leaked one into the other would be read as a
    # physical claim about the wrong axis.
    baseline = np.zeros((10, 11), np.float32)
    ablated = baseline.copy()
    ablated[:, 1] = 0.005                     # 5 mm on y (vertical) only
    result = translation_delta(baseline, ablated)
    assert result.per_axis_mm == pytest.approx((0.0, 5.0, 0.0))
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


def test_the_magnitude_is_the_rms_of_the_translation_in_millimetres():
    chunk = np.zeros((6, 8), np.float32)
    chunk[:, 0] = 0.003
    chunk[:, 1] = 0.004                       # 3-4-5, so 5 mm every step
    assert translation_magnitude_mm(chunk) == pytest.approx(5.0)


def test_the_magnitude_ignores_the_non_translation_channels():
    # Rotation, strategy and closure live in the later channels and are on
    # completely different scales; including them would make the figure
    # meaningless as a spatial magnitude.
    chunk = np.zeros((4, 11), np.float32)
    chunk[:, 2] = 0.002
    chunk[:, 3:] = 500.0
    assert translation_magnitude_mm(chunk) == pytest.approx(2.0)


def test_the_magnitude_is_what_makes_representations_comparable():
    # The reason this helper exists. A chunk-relative checkpoint emits
    # cumulative offsets; a delta checkpoint emits per-step motion of the same
    # trajectory. Their raw millimetres differ by the chunk length, so a bare
    # delta cannot be compared across the two -- but each divided by its own
    # magnitude can.
    steps = 50
    per_step = np.zeros((steps, 8), np.float32)
    per_step[:, 2] = 0.001                            # 1 mm per step
    cumulative = per_step.copy()
    cumulative[:, 2] = np.cumsum(per_step[:, 2])      # same motion, integrated

    small = translation_magnitude_mm(per_step)
    large = translation_magnitude_mm(cumulative)
    assert small == pytest.approx(1.0)
    assert large > 20 * small                          # ~29x for 50 steps

    # An intervention that perturbs each representation by 10% of its own
    # motion reads as the SAME fraction, though wildly different millimetres.
    for chunk in (per_step, cumulative):
        perturbed = chunk.copy()
        perturbed[:, 2] *= 1.10
        delta = translation_delta(chunk, perturbed).delta_mm
        fraction = delta / translation_magnitude_mm(chunk)
        assert fraction == pytest.approx(0.10, rel=1e-6)
