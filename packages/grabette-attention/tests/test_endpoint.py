"""Endpoint divergence: the one metric comparable across representations.

An RMS over chunk steps means different things for cumulative offsets and for
per-step deltas, and normalising each by its own magnitude — while better —
still embeds a choice that reversed the delta-vs-chunk-relative ranking
depending on which was picked. Endpoint divergence embeds none: it is the
distance between two places the gripper would arrive.

The linchpin test is the last one: the SAME physical trajectory, written in
both representations, must give the same endpoint.
"""
import numpy as np
import pytest

from grabette_attention.metrics import (
    OFFSETS,
    PER_STEP_DELTAS,
    chunk_endpoint_m,
    endpoint_divergence_mm,
)

IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def offsets_chunk(points):
    chunk = np.zeros((len(points), 8), np.float32)
    chunk[:, :3] = points
    return chunk


def deltas_chunk(steps, six=IDENTITY_6D):
    chunk = np.zeros((len(steps), 11), np.float32)
    chunk[:, :3] = steps
    chunk[:, 3:9] = six
    return chunk


def test_for_offsets_the_endpoint_is_simply_the_last_row():
    chunk = offsets_chunk([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0.004, 0]])
    end = chunk_endpoint_m(chunk, representation=OFFSETS)
    assert end == pytest.approx([0.03, 0.004, 0.0])


def test_for_per_step_deltas_the_endpoint_is_the_composed_sum():
    # Identity rotations, so composition reduces to a plain sum.
    chunk = deltas_chunk([[0.001, 0, 0]] * 5)
    end = chunk_endpoint_m(chunk, representation=PER_STEP_DELTAS)
    assert end == pytest.approx([0.005, 0.0, 0.0])


def test_per_step_deltas_compose_the_ROTATIONS_not_just_the_translations():
    # Two steps of 1 mm along x, with a 90 deg rotation about z between them.
    # Naively summing gives (2, 0, 0) mm; composing correctly turns the second
    # step onto +y, giving (1, 1, 0) mm. This is the whole reason the
    # representation needs stating.
    quarter_turn = (0.0, 1.0, 0.0, -1.0, 0.0, 0.0)   # 6D for Rz(-90 deg)
    chunk = np.zeros((2, 11), np.float32)
    chunk[:, :3] = [[0.001, 0, 0], [0.001, 0, 0]]
    chunk[0, 3:9] = quarter_turn      # applied AFTER step 0
    chunk[1, 3:9] = IDENTITY_6D
    end = chunk_endpoint_m(chunk, representation=PER_STEP_DELTAS)
    assert end[0] == pytest.approx(0.001)
    assert abs(end[1]) == pytest.approx(0.001)
    assert np.linalg.norm(end) == pytest.approx(0.001 * np.sqrt(2))


def test_a_degenerate_rotation_row_falls_back_to_the_identity():
    # All-zero actions appear as padding at the end of an episode; Gram-Schmidt
    # on them would be singular.
    chunk = np.zeros((3, 11), np.float32)
    chunk[:, :3] = [[0.002, 0, 0]] * 3        # rotation block left at zero
    end = chunk_endpoint_m(chunk, representation=PER_STEP_DELTAS)
    assert end == pytest.approx([0.006, 0.0, 0.0])


def test_identical_chunks_diverge_by_exactly_zero():
    chunk = deltas_chunk([[0.001, 0.002, -0.001]] * 6)
    assert endpoint_divergence_mm(
        chunk, chunk.copy(), representation=PER_STEP_DELTAS
    ) == 0.0


def test_the_divergence_is_a_distance_in_millimetres():
    a = offsets_chunk([[0.0, 0, 0], [0.010, 0, 0]])
    b = offsets_chunk([[0.0, 0, 0], [0.013, 0.004, 0]])
    # endpoints 3 mm apart on x and 4 mm on y -> 5 mm
    assert endpoint_divergence_mm(a, b, representation=OFFSETS) == pytest.approx(5.0)


def test_an_unstated_or_unknown_representation_is_rejected():
    chunk = offsets_chunk([[0.01, 0, 0]])
    with pytest.raises(ValueError, match="unknown representation"):
        chunk_endpoint_m(chunk, representation="whatever")
    with pytest.raises(TypeError):
        chunk_endpoint_m(chunk)          # keyword is required, never inferred


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        endpoint_divergence_mm(
            np.zeros((4, 8)), np.zeros((5, 8)), representation=OFFSETS
        )


def test_the_same_trajectory_in_both_representations_gives_one_endpoint():
    # THE point of this metric. A straight 50-step run of 1 mm steps, written
    # once as per-step deltas and once as cumulative offsets. Their RMS
    # magnitudes differ by ~29x (see test_metrics), but the endpoint is the
    # same physical place and must come back identical.
    steps = np.tile([0.001, 0.0005, -0.0002], (50, 1))
    deltas = deltas_chunk(steps)
    offsets = offsets_chunk(np.cumsum(steps, axis=0))

    from_deltas = chunk_endpoint_m(deltas, representation=PER_STEP_DELTAS)
    from_offsets = chunk_endpoint_m(offsets, representation=OFFSETS)
    assert from_deltas == pytest.approx(from_offsets, rel=1e-6)

    # And a perturbation of the same physical size reads the same in both.
    bump = np.zeros_like(steps)
    bump[:, 0] = 0.00002                          # +0.02 mm per step
    assert endpoint_divergence_mm(
        deltas, deltas_chunk(steps + bump), representation=PER_STEP_DELTAS
    ) == pytest.approx(
        endpoint_divergence_mm(
            offsets, offsets_chunk(np.cumsum(steps + bump, axis=0)),
            representation=OFFSETS,
        ),
        rel=1e-5,
    )
