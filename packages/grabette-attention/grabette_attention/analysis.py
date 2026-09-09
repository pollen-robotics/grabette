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
from .reduce import captured_steps, reduce_attention


def _run_passes(adapter: PolicyAdapter, obs: FrameObservation, *, ablate: bool):
    """Every forward pass this frame needs, sharing one noise draw.

    Returns (layout, geometries, baseline, ablations). Split out because the
    per-step variant needs exactly the same passes and must not repeat them:
    the captures already hold every denoising step.
    """
    layout = adapter.layout(obs)

    # Geometry is per camera: views may differ in resolution or aspect ratio, so
    # each camera's padding crop must come from its own frame.
    visible = layout.visible_cameras()
    if not visible:
        raise ValueError(f"frame {obs.frame} has no usable camera")

    # Draw noise once and reuse it across all passes (baseline + ablations).
    # This is the structural guarantee that "camera removed" deltas measure
    # camera effect, not sampling variance.
    noise = adapter.draw_noise()

    baseline = adapter.run(obs, noise, capture=True)
    geometries = {camera: adapter.geometry(obs, camera) for camera in visible}

    ablations = {}
    if ablate:
        for camera in visible:
            dropped = adapter.run(
                obs, noise, capture=False, drop_camera=camera
            )
            ablations[camera] = translation_delta(baseline.chunk, dropped.chunk)

    return layout, geometries, baseline, ablations


def analyse_frame(
    adapter: PolicyAdapter,
    obs: FrameObservation,
    *,
    denoise_step: str | int = "last",
    layers: str | Sequence[int] = "all",
    ablate: bool = True,
    provenance: dict[str, str] | None = None,
) -> FrameAnalysis:
    layout, geometries, baseline, ablations = _run_passes(
        adapter, obs, ablate=ablate
    )

    cameras, language_mass = reduce_attention(
        baseline.captures,
        layout,
        geometries,
        patch=adapter.patch,
        denoise_step=denoise_step,
        layers=layers,
    )

    record = dict(provenance or {})
    record["denoise_step"] = str(denoise_step)
    record["layers"] = str(layers)
    return FrameAnalysis(
        episode=obs.episode,
        frame=obs.frame,
        cameras=cameras,
        language_mass=language_mass,
        ablations=ablations,
        provenance=record,
    )


def analyse_frame_steps(
    adapter: PolicyAdapter,
    obs: FrameObservation,
    *,
    layers: str | Sequence[int] = "all",
    ablate: bool = True,
    provenance: dict[str, str] | None = None,
) -> list[FrameAnalysis]:
    """One record per denoising step, from ONE forward pass.

    The map moves between steps -- early steps and the final step attend to
    different places -- so comparing them is a real question. The captures from
    a single pass already contain every step, so this costs one reduction per
    step and no extra inference.

    The ablation is a property of the emitted chunk, not of any one denoising
    step, so the same figures are attached to every record. Each record gets
    its own copy so a consumer mutating one cannot corrupt the others.
    """
    layout, geometries, baseline, ablations = _run_passes(
        adapter, obs, ablate=ablate
    )

    out = []
    for step in captured_steps(baseline.captures):
        cameras, language_mass = reduce_attention(
            baseline.captures,
            layout,
            geometries,
            patch=adapter.patch,
            denoise_step=step,
            layers=layers,
        )
        record = dict(provenance or {})
        record["denoise_step"] = str(step)
        record["layers"] = str(layers)
        # Says plainly that these millimetres do not vary across the records,
        # so nobody reads a per-step ablation trend into them.
        record["ablation_scope"] = "whole chunk (identical across steps)"
        out.append(
            FrameAnalysis(
                episode=obs.episode,
                frame=obs.frame,
                cameras=cameras,
                language_mass=language_mass,
                ablations=dict(ablations),
                provenance=record,
            )
        )
    return out


def analyse(
    adapter: PolicyAdapter,
    frames: Iterable[FrameObservation],
    **kwargs,
) -> Iterator[FrameAnalysis]:
    """Stream over frames, so a long episode does not have to fit in memory.

    `denoise_step="all"` fans one frame out into one record per step; every
    other value yields exactly one record per frame.
    """
    for obs in frames:
        if kwargs.get("denoise_step") == "all":
            per_step = {k: v for k, v in kwargs.items() if k != "denoise_step"}
            yield from analyse_frame_steps(adapter, obs, **per_step)
        else:
            yield analyse_frame(adapter, obs, **kwargs)
