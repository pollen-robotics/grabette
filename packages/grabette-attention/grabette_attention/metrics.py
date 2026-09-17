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


OFFSETS = "offsets"
PER_STEP_DELTAS = "per_step_deltas"
_REPRESENTATIONS = (OFFSETS, PER_STEP_DELTAS)


def _six_d_to_rotation(six: np.ndarray) -> np.ndarray:
    """Zhou et al. 6D rotation representation -> 3x3 matrix.

    The first two columns of R, re-orthonormalised by Gram-Schmidt. Degenerate
    rows (all-zero actions, which appear as padding at the end of an episode)
    fall back to the identity rather than producing a singular matrix.
    """
    a1 = np.asarray(six[:3], dtype=np.float64)
    a2 = np.asarray(six[3:6], dtype=np.float64)
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        return np.eye(3)
    b1 = a1 / n1
    a2p = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2p)
    if n2 < 1e-8:
        return np.eye(3)
    b2 = a2p / n2
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def chunk_endpoint_m(
    chunk: np.ndarray,
    *,
    representation: str,
    n_translation: int = 3,
    rotation: slice = slice(3, 9),
) -> np.ndarray:
    """Where the commanded trajectory ENDS, relative to the current pose, in metres.

    `representation` must be stated, never inferred, because the two look
    identical as arrays and mean completely different things:

    OFFSETS -- each row is already a displacement from the current pose, so the
        endpoint is simply the last row. Chunk-relative checkpoints.

    PER_STEP_DELTAS -- each row is motion relative to the PREVIOUS step's
        frame, so the endpoint requires composing the rotations along the
        chunk: p = d0 + R0 d1 + R0 R1 d2 + ... Plain delta checkpoints. The
        rotation slice defaults to the 6D block at channels 3:9 of an 11-dim
        graspproj action.

    This is the one quantity that IS comparable across representations: it is a
    physical displacement, so it needs no normalisation and carries no choice
    about what to divide by.
    """
    array = np.asarray(chunk, dtype=np.float64)
    if representation == OFFSETS:
        return array[-1, :n_translation].copy()
    if representation == PER_STEP_DELTAS:
        point = np.zeros(n_translation, dtype=np.float64)
        frame = np.eye(3)
        for step in array:
            point = point + frame @ step[:n_translation]
            frame = frame @ _six_d_to_rotation(step[rotation])
        return point
    raise ValueError(
        f"unknown representation {representation!r}; expected one of "
        f"{_REPRESENTATIONS}"
    )


def endpoint_divergence_mm(
    baseline: np.ndarray,
    perturbed: np.ndarray,
    *,
    representation: str,
    n_translation: int = 3,
    rotation: slice = slice(3, 9),
) -> float:
    """How far apart the two commanded trajectories END, in millimetres.

    Prefer this to `translation_delta` when comparing ACROSS representations.
    An RMS over chunk steps means different things for offsets and for per-step
    deltas -- and dividing each by its own magnitude, while better, still
    embeds a choice that can reverse the ranking. Endpoint divergence embeds
    none: it is the distance between two places the gripper would arrive.
    """
    if baseline.shape != perturbed.shape:
        raise ValueError(
            f"shape mismatch: baseline {baseline.shape} vs "
            f"perturbed {perturbed.shape}"
        )
    kwargs = {
        "representation": representation,
        "n_translation": n_translation,
        "rotation": rotation,
    }
    start = chunk_endpoint_m(baseline, **kwargs)
    end = chunk_endpoint_m(perturbed, **kwargs)
    return float(np.linalg.norm(end - start)) * _METRES_TO_MM


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
