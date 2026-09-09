"""Turn captured attention tensors into one map and one number per camera.

Reduction order, following the recipe every 2026 VLA-diagnosis paper uses: mean
over heads, mean over the action-token queries, then mean over the selected
layers and denoising steps. Queries are the action tokens; keys are the prefix
followed by the action tokens themselves.

Mass is a fraction of the PREFIX (cameras plus language), so the numbers sum to
one and a camera the policy barely uses reads as a small number rather than
being renormalised into looking important.
"""

from typing import Mapping, Sequence

import numpy as np

from .layout import LetterboxGeometry, TokenLayout
from .records import CameraAttention


def captured_steps(captures: dict[tuple[int, int], np.ndarray]) -> list[int]:
    """The denoising step indices present in a capture, in order.

    The caller needs these to iterate steps without knowing how many denoising
    steps the policy ran, which is a property of the checkpoint's sampler.
    """
    if not captures:
        raise ValueError("no attention was captured; were the hooks installed?")
    return sorted({step for step, _ in captures})


def _select(
    captures: dict[tuple[int, int], np.ndarray],
    denoise_step: str | int,
    layers: str | Sequence[int],
) -> list[np.ndarray]:
    if not captures:
        raise ValueError("no attention was captured; were the hooks installed?")

    steps = sorted({s for s, _ in captures})
    if denoise_step == "last":
        wanted_steps = {steps[-1]}
    elif denoise_step == "first":
        wanted_steps = {steps[0]}
    elif denoise_step == "mean":
        wanted_steps = set(steps)
    elif denoise_step == "all":
        # This function returns ONE grid per camera, so "all" has no meaning
        # here: collapsing every step into one grid is just "mean" under a
        # misleading name, and that is precisely the per-step comparison it
        # would destroy. A real per-step result is a sequence of records, which
        # is `analysis.analyse_frame_steps` -- it reduces these same captures
        # once per step, so it costs no extra inference.
        raise ValueError(
            "denoise_step='all' is not a single-grid reduction: collapsing "
            "every step into one grid is what 'mean' already does. For a true "
            "per-step comparison use analysis.analyse_frame_steps(), which "
            "returns one record per step from a single forward pass."
        )
    elif isinstance(denoise_step, int):
        wanted_steps = {denoise_step}
    else:
        raise ValueError(f"unknown denoise_step {denoise_step!r}")

    if layers in ("all", "mean"):
        wanted_layers = {l for _, l in captures}
    else:
        wanted_layers = set(layers)

    chosen = [
        w for (s, l), w in captures.items()
        if s in wanted_steps and l in wanted_layers
    ]
    if not chosen:
        raise ValueError(
            f"no attention matched denoise_step={denoise_step!r} layers={layers!r}"
        )
    return chosen


def reduce_attention(
    captures: dict[tuple[int, int], np.ndarray],
    layout: TokenLayout,
    geometries: Mapping[str, LetterboxGeometry],
    *,
    patch: int,
    denoise_step: str | int = "last",
    layers: str | Sequence[int] = "all",
) -> tuple[dict[str, CameraAttention], float]:
    """Reduce captures to per-camera grids plus each camera's prefix mass.

    `geometries` holds ONE ENTRY PER VISIBLE CAMERA, keyed by camera name. Each
    camera's padding crop comes from its own frame, because views may differ in
    resolution and aspect ratio; a camera with no entry is a KeyError rather
    than a guess.

    Returns (per-camera attention keyed by camera name, language mass).
    """
    chosen = _select(captures, denoise_step, layers)

    # (heads, queries, keys) each -> mean over layers/steps, heads, then queries.
    stacked = np.stack(chosen, axis=0).mean(axis=0)      # (heads, queries, keys)

    # If the patch size is wrong, tokens-per-image and grid rows*cols stay
    # mutually consistent BY CONSTRUCTION (both derived from the same wrong
    # patch), so the reshape below would still succeed and every camera block
    # would be silently misaligned. Check the captured tensor itself, not just
    # the layout's own arithmetic: keys must be exactly the prefix plus the
    # query count (the action tokens appended after the prefix).
    queries, keys = stacked.shape[-2], stacked.shape[-1]
    expected_keys = layout.prefix_len + queries
    if keys != expected_keys:
        raise ValueError(
            f"captured key axis ({keys}) does not match the derived layout: "
            f"expected prefix_len ({layout.prefix_len}) + query count "
            f"({queries}) = {expected_keys}. This usually means the patch size "
            "used to derive the layout does not match what actually produced "
            "this capture."
        )

    over_keys = stacked.mean(axis=(0, 1))                # (keys,)

    prefix = over_keys[: layout.prefix_len]
    total = float(prefix.sum())
    if total <= 0.0:
        raise ValueError("the prefix received no attention mass")

    cameras: dict[str, CameraAttention] = {}
    for key in layout.visible_cameras():
        block = prefix[layout.camera_slice(key)]
        grid = block.reshape(layout.grid_rows, layout.grid_cols)
        geometry = geometries[key]          # KeyError if a camera was forgotten
        row0, row1 = geometry.content_rows(patch)
        col0, col1 = geometry.content_cols(patch)
        cameras[key] = CameraAttention(
            grid=np.ascontiguousarray(grid[row0:row1, col0:col1], dtype=np.float32),
            mass=float(block.sum()) / total,
        )

    language_mass = float(prefix[layout.language_slice()].sum()) / total
    return cameras, language_mass
