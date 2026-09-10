"""Where in the image does the commanded motion actually come from?

The view ablation asks "does this camera matter" by removing the whole view.
This asks the same question of each REGION of one view: cover a block of the
image, re-run, and measure how far the commanded chunk moves. The result is a
causal map in millimetres, not an attention map -- which matters because on
this policy attention mass does not track causal effect (see
docs/attention_maps_findings.md), and the attention map's dominant component is
locked to the gripper rather than to the scene.

The occlusion grid divides the SOURCE image into blocks. The attention grid has
its letterbox padding rows already cropped, so it too covers exactly the source
content: pass the attention grid's own shape and the two maps are comparable
cell for cell.

Cost is one forward pass per block plus one baseline, so a 6x8 grid is 49
passes. Prefer a coarse grid for exploration and a fine one only where it
earns itself.

One honest caveat about the fill value. A covered block has to be filled with
something, and every choice is off distribution: the policy has never seen a
grey rectangle in the middle of a scene. `-1` after normalisation is what the
model sees for absent image content, but in training that only ever appears at
the borders of a letterboxed frame, never mid-image. So read this map as
"disrupting this region changes the action by X", not as "the model would do X
if this region were empty". The comparison BETWEEN regions is the trustworthy
part, since every region gets the same treatment.
"""

from dataclasses import replace

import numpy as np

from .metrics import (
    endpoint_divergence_mm,
    translation_delta,
    translation_magnitude_mm,
)
from .records import FrameObservation, OcclusionMap

_FILLS = ("mean", "grey", "black")


def _fill_value(image: np.ndarray, fill: str):
    """The colour a covered block is painted with."""
    if fill == "mean":
        # The least informative colour available for THIS frame: it preserves
        # the image's overall brightness, so the disruption is the removal of
        # local structure rather than a global brightness shift.
        return image.reshape(-1, image.shape[-1]).mean(axis=0).astype(image.dtype)
    if fill == "grey":
        return np.full(image.shape[-1], 128, dtype=image.dtype)
    if fill == "black":
        # Maps to -1 after normalisation, which is what the policy sees for
        # padding -- in distribution at a border, not mid-image.
        return np.zeros(image.shape[-1], dtype=image.dtype)
    raise ValueError(f"unknown fill {fill!r}; expected one of {_FILLS}")


def occlusion_saliency(
    adapter,
    obs: FrameObservation,
    *,
    camera: str,
    rows: int,
    cols: int,
    fill: str = "mean",
    metric: str = "rms",
    representation: str | None = None,
    noise=None,
) -> OcclusionMap:
    """Per-region causal effect on the commanded chunk, in millimetres.

    `rows` and `cols` set the occlusion grid over the source image. Passing the
    attention grid's shape makes the two maps directly comparable.

    Every pass -- the baseline and each occluded run -- shares ONE noise
    tensor, so a difference means the region mattered rather than the sampler
    drew differently. This is the same guarantee the view ablation relies on.
    """
    if camera not in obs.images:
        raise KeyError(f"{camera!r} is not one of this frame's cameras "
                       f"{tuple(obs.images)}")
    if rows < 1 or cols < 1:
        raise ValueError(f"occlusion grid must be at least 1x1, got {rows}x{cols}")
    if metric not in ("rms", "endpoint"):
        raise ValueError(f"unknown metric {metric!r}; expected 'rms' or 'endpoint'")
    if metric == "endpoint" and representation is None:
        # Never guessed: offsets and per-step deltas are the same shape and
        # composing one as the other silently returns a wrong distance.
        raise ValueError(
            "metric='endpoint' needs representation='offsets' or "
            "'per_step_deltas' -- it cannot be inferred from the chunk"
        )

    image = obs.images[camera]
    height, width = image.shape[:2]
    value = _fill_value(image, fill)

    noise = adapter.draw_noise() if noise is None else noise
    baseline = adapter.run(obs, noise, capture=False)

    grid = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        r0, r1 = int(r * height / rows), int((r + 1) * height / rows)
        for c in range(cols):
            c0, c1 = int(c * width / cols), int((c + 1) * width / cols)
            covered = image.copy()
            covered[r0:r1, c0:c1] = value
            # A new observation rather than a mutated one: the caller's frame
            # must survive the sweep unchanged, and it is reused every block.
            occluded = replace(obs, images={**obs.images, camera: covered})
            result = adapter.run(occluded, noise, capture=False)
            grid[r, c] = (
                translation_delta(baseline.chunk, result.chunk).delta_mm
                if metric == "rms"
                else endpoint_divergence_mm(
                    baseline.chunk, result.chunk, representation=representation
                )
            )

    # RMS translation magnitude of the untouched chunk, so a delta can be read
    # against the size of the motion it perturbs.
    baseline_mm = translation_magnitude_mm(baseline.chunk)

    return OcclusionMap(
        camera=camera,
        grid=grid.astype(np.float32),
        baseline_mm=baseline_mm,
        fill=fill,
        metric=metric,
    )
