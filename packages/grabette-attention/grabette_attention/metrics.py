"""How much did removing a camera change the commanded motion?

The chunk's translation channels are in METRES; everything reported here is in
MILLIMETRES, converted exactly once. `delta_mm` is the RMS over chunk steps of
the 3-D difference, so it does not grow with the chunk length. `per_axis_mm` is
the same RMS per axis.

Vertical and range are DIFFERENT axes and they do not behave alike, so read
`per_axis_mm` against the convention recorded in `records.ViewAblation` rather
than assuming the interesting axis is the last one.

For a chunk-relative checkpoint the chunk holds offsets rather than per-step
deltas; the metric still measures how far the two predictions diverge, which is
what the ablation asks.
"""

import numpy as np

from .records import ViewAblation

_METRES_TO_MM = 1000.0


def translation_magnitude_mm(chunk: np.ndarray, *, n_translation: int = 3) -> float:
    """RMS translation magnitude of one chunk, in millimetres.

    The scale a delta should be read against. It also makes deltas comparable
    ACROSS ACTION REPRESENTATIONS, which a bare millimetre figure is not: a
    chunk-relative checkpoint's chunk holds cumulative offsets from the current
    pose (tens of mm), while a plain delta checkpoint's holds per-step motion
    (a few mm), so the same intervention reads ~20-50x larger on the former
    purely because of how its actions are parameterised. Dividing each model's
    delta by its own magnitude asks both the same question: what fraction of
    the commanded motion did this change?
    """
    translation = np.asarray(chunk[:, :n_translation], dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(translation**2, axis=1)))) * _METRES_TO_MM


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
