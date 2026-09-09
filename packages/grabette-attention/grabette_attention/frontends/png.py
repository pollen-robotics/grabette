"""Write attention overlays as PNGs and the numbers as a text summary.

Headless by construction: matplotlib is switched to Agg before pyplot is
imported, matching `integrations/DiffusionPolicy/offline_eval.py`. Frames are
RGB in memory throughout; nothing here writes BGR.
"""

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ..records import FrameAnalysis, FrameObservation
from . import camera_labels

_GUARD = (
    "NOTE: pi0.5 spreads attention broadly with low peaks, and action "
    "fine-tuning makes it more diffuse. A broad map is NORMAL and is not "
    "evidence of anything. The ablation millimetres are the interventional "
    "measurement; the map is a hypothesis. See docs/attention_saliency_review.md."
)

# Attention grids routinely carry one or two "register" cells -- high-norm
# sink tokens parked in featureless background -- an order of magnitude above
# everything else. Autoscaling the colour ramp to the raw maximum hands the
# entire ramp to those cells: a measured peak at 36x the median held only 6%
# of the camera's mass yet rendered as the only visible structure. Clip the
# ramp at a high percentile so the remaining 94% is actually legible. The
# percentile goes in the title, so a clipped peak reads as clipped rather
# than as absent.
_CLIP_PERCENTILE = 99.0


def _colour_limits(grid: np.ndarray) -> tuple[float, float]:
    """Robust (vmin, vmax) for one attention grid."""
    vmin = float(grid.min())
    vmax = float(np.percentile(grid, _CLIP_PERCENTILE))
    if not vmax > vmin:
        # The bulk of the grid is flat, so the clip landed on the minimum. Do
        # NOT fall back to grid.max() here: that hands the ramp straight back
        # to the sink cells this function exists to demote, and it is exactly
        # the flat-bulk-plus-one-spike grid where that matters most. Nudge the
        # top of the range instead and let the outliers saturate.
        vmax = vmin + max(abs(vmin) * 1e-3, 1e-12)
    return vmin, vmax


# Physical meaning of the translation channels, in channel order. See
# records.ViewAblation for the convention and the README for how it was
# measured. Labelled here because three bare millimetre figures invite being
# read against the wrong axis.
_AXIS_LABELS = ("x/lat", "y/vert", "z/depth")


def _step_tag(analysis: FrameAnalysis) -> str:
    """Which denoising step this analysis shows, safe for a filename.

    Numeric steps are zero-padded so a directory of a per-step sweep sorts in
    step order rather than lexically (step2 before step10).
    """
    raw = analysis.provenance.get("denoise_step", "last")
    return f"{int(raw):02d}" if raw.lstrip("-").isdigit() else raw


def _per_axis(ablation) -> str:
    """`x/lat 8.1  y/vert 13.1  z/depth 4.1 mm`, tolerating an odd width."""
    values = ablation.per_axis_mm
    labels = _AXIS_LABELS if len(values) == len(_AXIS_LABELS) else (
        tuple(f"axis{i}" for i in range(len(values)))
    )
    return "  ".join(f"{name} {value:.1f}" for name, value in zip(labels, values))


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_overlays(
    analysis: FrameAnalysis, obs: FrameObservation, out_dir: Path | str
) -> list[Path]:
    """One overlay per visible camera: the frame with its attention on top."""
    plt = _pyplot()
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    labels = camera_labels(analysis.cameras)
    step = _step_tag(analysis)

    written = []
    for camera, attention in analysis.cameras.items():
        frame = obs.images[camera]
        short = labels[camera]
        # The step is always in the name, so a per-step sweep of one frame
        # cannot overwrite itself and every file says which step it shows.
        path = directory / f"frame_{analysis.frame:05d}_{short}_step{step}_attn.png"

        figure, axis = plt.subplots(figsize=(6, 4.5), dpi=110)
        axis.imshow(frame)
        # The grid already has its padding rows removed, so stretching it over
        # the whole frame is the correct mapping back to source pixels.
        vmin, vmax = _colour_limits(attention.grid)
        axis.imshow(
            attention.grid,
            extent=(0, frame.shape[1], frame.shape[0], 0),
            interpolation="bilinear",
            alpha=0.55,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        peak = float(attention.grid.max())
        axis.set_title(
            f"ep{analysis.episode} frame {analysis.frame} — {short}"
            f"  [step {step}]\n"
            f"mass {attention.mass:.2f}"
            + (
                f", ablation {analysis.ablations[camera].delta_mm:.1f} mm"
                if camera in analysis.ablations
                else ""
            )
            + f"\ncolour clipped at p{_CLIP_PERCENTILE:g}"
            + (f" (peak {peak / vmax:.1f}x above clip)" if peak > vmax else ""),
            fontsize=8,
        )
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
        written.append(path)
    return written


def write_summary(
    analyses: Iterable[FrameAnalysis],
    out_dir: Path | str,
    *,
    notes: Mapping[int, str] | None = None,
) -> Path:
    """The numbers, one row per camera per frame, plus provenance."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "summary.txt"

    lines: list[str] = []
    provenance: dict[str, str] = {}
    steps_seen: list[str] = []
    for analysis in analyses:
        # The denoising step is per record, so it cannot be merged into one
        # provenance block: with --denoise-step all, several records describe
        # the same frame and a single merged value would name only the last
        # one. Collect them instead, and put each record's own step in its
        # header so the rows can never be attributed to the wrong step.
        rest = dict(analysis.provenance)
        step = rest.pop("denoise_step", None)
        if step is not None and step not in steps_seen:
            steps_seen.append(step)
        provenance.update(rest)
        header = f"episode {analysis.episode}  frame {analysis.frame}"
        if step is not None:
            header += f"  step {step}"
        if notes and analysis.episode in notes:
            header += f"  [{notes[analysis.episode]}]"
        lines.append(header)
        for camera, attention in analysis.cameras.items():
            row = f"  {camera:<34} mass {attention.mass:.2f}"
            ablation = analysis.ablations.get(camera)
            if ablation is not None:
                row += f"   ablate -> {ablation.delta_mm:.1f} mm"
            lines.append(row)
            if ablation is not None:
                lines.append(f"  {'':<34} {_per_axis(ablation)} mm")
        lines.append(f"  {'language (task + state)':<34} mass {analysis.language_mass:.2f}")
        lines.append("")

    if steps_seen:
        provenance["denoise_step"] = ",".join(steps_seen)
    if provenance:
        lines.append("provenance")
        for key in sorted(provenance):
            lines.append(f"  {key}: {provenance[key]}")
        lines.append("")
    lines.append(_GUARD)

    path.write_text("\n".join(lines) + "\n")
    return path
