"""Write attention overlays as PNGs and the numbers as a text summary.

Headless by construction: matplotlib is switched to Agg before pyplot is
imported, matching `integrations/DiffusionPolicy/offline_eval.py`. Frames are
RGB in memory throughout; nothing here writes BGR.
"""

from pathlib import Path
from typing import Iterable, Mapping

from ..records import FrameAnalysis, FrameObservation
from . import camera_labels

_GUARD = (
    "NOTE: pi0.5 spreads attention broadly with low peaks, and action "
    "fine-tuning makes it more diffuse. A broad map is NORMAL and is not "
    "evidence of anything. The ablation millimetres are the interventional "
    "measurement; the map is a hypothesis. See docs/attention_saliency_review.md."
)


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

    written = []
    for camera, attention in analysis.cameras.items():
        frame = obs.images[camera]
        short = labels[camera]
        path = directory / f"frame_{analysis.frame:05d}_{short}_attn.png"

        figure, axis = plt.subplots(figsize=(6, 4.5), dpi=110)
        axis.imshow(frame)
        # The grid already has its padding rows removed, so stretching it over
        # the whole frame is the correct mapping back to source pixels.
        axis.imshow(
            attention.grid,
            extent=(0, frame.shape[1], frame.shape[0], 0),
            interpolation="bilinear",
            alpha=0.55,
            cmap="inferno",
        )
        axis.set_title(
            f"ep{analysis.episode} frame {analysis.frame} — {short}\n"
            f"mass {attention.mass:.2f}"
            + (
                f", ablation {analysis.ablations[camera].delta_mm:.1f} mm"
                if camera in analysis.ablations
                else ""
            ),
            fontsize=9,
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
    for analysis in analyses:
        provenance.update(analysis.provenance)
        header = f"episode {analysis.episode}  frame {analysis.frame}"
        if notes and analysis.episode in notes:
            header += f"  [{notes[analysis.episode]}]"
        lines.append(header)
        for camera, attention in analysis.cameras.items():
            row = f"  {camera:<34} mass {attention.mass:.2f}"
            ablation = analysis.ablations.get(camera)
            if ablation is not None:
                axes = ", ".join(f"{a:.1f}" for a in ablation.per_axis_mm)
                row += f"   ablate -> {ablation.delta_mm:.1f} mm  (xyz {axes} mm)"
            lines.append(row)
        lines.append(f"  {'language (task + state)':<34} mass {analysis.language_mass:.2f}")
        lines.append("")

    if provenance:
        lines.append("provenance")
        for key in sorted(provenance):
            lines.append(f"  {key}: {provenance[key]}")
        lines.append("")
    lines.append(_GUARD)

    path.write_text("\n".join(lines) + "\n")
    return path
