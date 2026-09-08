"""How much did removing a camera change the commanded motion?

The chunk's translation channels are in METRES; everything reported here is in
MILLIMETRES, converted exactly once. `delta_mm` is the RMS over chunk steps of
the 3-D difference, so it does not grow with the chunk length. `per_axis_mm` is
the same RMS per axis, which is the figure that says whether a view carries the
vertical (range) information.

For a chunk-relative checkpoint the chunk holds offsets rather than per-step
deltas; the metric still measures how far the two predictions diverge, which is
what the ablation asks.
"""

import numpy as np

from .records import ViewAblation

_METRES_TO_MM = 1000.0


def translation_delta(
    baseline: np.ndarray, ablated: np.ndarray, *, n_translation: int = 3
) -> ViewAblation:
    """RMS difference of the translation channels, in millimetres."""
    if baseline.shape != ablated.shape:
        raise ValueError(
            f"shape mismatch: baseline {baseline.shape} vs ablated {ablated.shape}"
        )
    diff = np.asarray(ablated[:, :n_translation], dtype=np.float64) - np.asarray(
        baseline[:, :n_translation], dtype=np.float64
    )
    norm = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    per_axis = np.sqrt(np.mean(diff**2, axis=0))
    return ViewAblation(
        delta_mm=norm * _METRES_TO_MM,
        per_axis_mm=tuple(float(a * _METRES_TO_MM) for a in per_axis),
    )
