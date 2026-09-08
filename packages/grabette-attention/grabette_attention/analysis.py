"""Compose an adapter and a frame into a `FrameAnalysis`.

One capture pass gives the chunk and the attention together; then one pass per
camera with that camera masked gives the ablation deltas. All of them share the
adapter's single noise tensor, which is what makes the deltas mean "the camera
changed this" rather than "the sampler drew different noise".
"""

from typing import Iterable, Iterator, Sequence

from .adapters.base import PolicyAdapter
from .metrics import translation_delta
from .records import FrameAnalysis, FrameObservation
from .reduce import reduce_attention


def analyse_frame(
    adapter: PolicyAdapter,
    obs: FrameObservation,
    *,
    denoise_step: str | int = "last",
    layers: str | Sequence[int] = "all",
    ablate: bool = True,
    provenance: dict[str, str] | None = None,
) -> FrameAnalysis:
    layout = adapter.layout(obs)
    baseline = adapter.run(obs, adapter.draw_noise(), capture=True)

    # Geometry is per camera: views may differ in resolution or aspect ratio, so
    # each camera's padding crop must come from its own frame.
    visible = layout.visible_cameras()
    if not visible:
        raise ValueError(f"frame {obs.frame} has no usable camera")
    geometries = {camera: adapter.geometry(obs, camera) for camera in visible}

    cameras, language_mass = reduce_attention(
        baseline.captures,
        layout,
        geometries,
        patch=adapter.patch,
        denoise_step=denoise_step,
        layers=layers,
    )

    ablations = {}
    if ablate:
        for camera in visible:
            dropped = adapter.run(
                obs, adapter.draw_noise(), capture=False, drop_camera=camera
            )
            ablations[camera] = translation_delta(baseline.chunk, dropped.chunk)

    record = dict(provenance or {})
    record.setdefault("denoise_step", str(denoise_step))
    record.setdefault("layers", str(layers))
    return FrameAnalysis(
        episode=obs.episode,
        frame=obs.frame,
        cameras=cameras,
        language_mass=language_mass,
        ablations=ablations,
        provenance=record,
    )


def analyse(
    adapter: PolicyAdapter,
    frames: Iterable[FrameObservation],
    **kwargs,
) -> Iterator[FrameAnalysis]:
    """Stream over frames, so a long episode does not have to fit in memory."""
    for obs in frames:
        yield analyse_frame(adapter, obs, **kwargs)
