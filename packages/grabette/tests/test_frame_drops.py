"""Which streams of a stopped take get flagged as having lost frames.

The cases come from a bimanual test recording filmed against a stopwatch: the
Arducam routinely loses a few frames at startup (harmless), and once lost 27 in
a row (563 ms) while its average rate still looked fine at 43 fps.
"""

from grabette.backend.rpi import FPS, _dropped_streams


def _stream(name, period_ms, n, holes=(), fps=FPS):
    """(name, n_frames, span_ms, max_gap_ms, fps) for n frames at period_ms,
    with each hole widening one interval to that many milliseconds."""
    gaps = [period_ms] * (n - 1)
    for i, hole in enumerate(holes):
        gaps[10 + i] = hole
    return (name, n, sum(gaps), max(gaps), fps)


def test_regular_streams_pass():
    assert _dropped_streams([_stream("rgb", 20.12, 240), _stream("depth", 33.31, 150, fps=30)]) == []


def test_startup_losses_of_a_few_frames_pass():
    assert _dropped_streams([_stream("rgb", 20.12, 240, holes=(80.5, 60.4, 40.2))]) == []


def test_a_single_long_hole_is_flagged_despite_a_good_average():
    stream = _stream("rgb", 20.12, 245, holes=(563.5,))
    _, n, span, _, _ = stream
    assert (n - 1) / (span / 1000) > 0.8 * FPS  # the average alone misses it
    assert _dropped_streams([stream]) == ["rgb"]


def test_a_low_average_rate_is_still_flagged():
    assert _dropped_streams([_stream("depth", 33.31, 150, fps=30),
                             _stream("rgb", 30.0, 200)]) == ["rgb"]


def test_a_depth_hole_is_flagged():
    assert _dropped_streams([_stream("depth", 33.31, 150, holes=(400.0,), fps=30)]) == ["depth"]
