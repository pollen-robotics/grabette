"""Unit tests for grabette_postprocess.dataset pure helpers."""

import numpy as np

from grabette_postprocess.dataset import _nearest_frame_indices


def test_nearest_frame_indices_basic():
    """Each query timestamp maps to the index of its nearest frame timestamp."""
    frame_ts = np.array([0.0, 1.0, 2.0, 3.0])
    query = np.array([0.4, 1.6, 2.9])
    np.testing.assert_array_equal(
        _nearest_frame_indices(query, frame_ts), [0, 2, 3],
    )


def test_nearest_frame_indices_ties_go_left():
    """An exact midpoint resolves to the lower frame index."""
    # Exact midpoint: query - left == right - query -> picks the lower index.
    frame_ts = np.array([0.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        _nearest_frame_indices(np.array([0.5, 1.5]), frame_ts), [0, 1],
    )


def test_nearest_frame_indices_clamps_out_of_range():
    """Queries before the first / after the last frame clamp to the end indices."""
    frame_ts = np.array([10.0, 20.0])
    # Queries before the first / after the last frame clamp to the ends.
    np.testing.assert_array_equal(
        _nearest_frame_indices(np.array([-5.0, 100.0]), frame_ts), [0, 1],
    )


def test_nearest_frame_indices_degenerate_returns_zeros():
    """Fewer than 2 frames returns all-zeros, one per query."""
    # Fewer than 2 frames -> all-zeros, one per query.
    assert _nearest_frame_indices(np.array([1.0, 2.0]), np.array([5.0])).tolist() == [0, 0]
    assert _nearest_frame_indices(np.array([1.0]), np.array([])).tolist() == [0]
