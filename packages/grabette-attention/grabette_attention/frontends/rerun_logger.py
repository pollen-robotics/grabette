"""Log the same `FrameAnalysis` records to a rerun timeline.

Second front end, added after the PNG writer. It computes nothing: both front
ends read identical records, which is the whole point of keeping the analysis
layer free of presentation.

Entity naming follows the repo's existing visualiser
(`packages/grabette-postprocess/scripts/visualize/visualize_rgbd_trajectory.py`):
`camera_feed/<name>` for imagery, `metrics/...` for scalar series.
"""

from typing import Any

from ..records import FrameAnalysis, FrameObservation
from . import camera_labels


def _rerun():
    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError(
            "rerun is not installed; install the 'rerun' extra of "
            "grabette-attention, or use the PNG front end"
        ) from exc
    return rr


def open_recording(name: str = "grabette-attention") -> Any:
    rr = _rerun()
    rr.init(name, spawn=True)
    return rr


def log_analysis(
    analysis: FrameAnalysis, obs: FrameObservation, *, recording: Any = None
) -> None:
    """Log one frame: imagery per camera, plus mass and ablation as scalars."""
    rr = recording or _rerun()
    rr.set_time("frame", sequence=analysis.frame)

    labels = camera_labels(analysis.cameras)
    for camera, attention in analysis.cameras.items():
        short = labels[camera]
        rr.log(f"camera_feed/{short}", rr.Image(obs.images[camera]))
        rr.log(f"camera_feed/{short}/attention", rr.Image(attention.grid))
        rr.log(f"metrics/mass/{short}", rr.Scalars(attention.mass))
        ablation = analysis.ablations.get(camera)
        if ablation is not None:
            rr.log(
                f"metrics/ablation_mm/{short}", rr.Scalars(ablation.delta_mm)
            )
    rr.log("metrics/mass/language", rr.Scalars(analysis.language_mass))
