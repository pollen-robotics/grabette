"""Log the same `FrameAnalysis` records to a rerun timeline.

Second front end, added after the PNG writer. It computes nothing: both front
ends read identical records, which is the whole point of keeping the analysis
layer free of presentation.

Entity naming follows the repo's existing visualiser
(`packages/grabette-postprocess/scripts/visualize/visualize_rgbd_trajectory.py`):
`camera_feed/<name>` for imagery, `metrics/...` for scalar series.
"""

from typing import Any

import numpy as np

from ..records import FrameAnalysis, FrameObservation
from . import camera_labels


def _upsample_and_normalize(grid: np.ndarray, height: int, width: int) -> np.ndarray:
    """Stretch a small attention grid over a full frame and rescale to [0, 1].

    The rerun viewer draws an image entity at its OWN pixel resolution, so a
    twelve-by-sixteen child does not stretch over its parent frame the way the
    PNG path's `extent=` does -- it lands as a small block. And raw attention
    values, of order one over the cell count, sit near the bottom of the
    default float display range and render near-black. Nearest-neighbour
    upsampling (no interpolation library is guaranteed at this layer) plus a
    min-max rescale gives the same "stretched over the frame, visibly hot
    where it's hot" behaviour the PNG overlay gets from matplotlib's `extent`
    and autoscaling.
    """
    rows, cols = grid.shape
    row_idx = np.clip((np.arange(height) * rows) // height, 0, rows - 1)
    col_idx = np.clip((np.arange(width) * cols) // width, 0, cols - 1)
    upsampled = grid[row_idx][:, col_idx]

    lo, hi = float(upsampled.min()), float(upsampled.max())
    if hi - lo < 1e-12:
        return np.zeros_like(upsampled, dtype=np.float32)
    return ((upsampled - lo) / (hi - lo)).astype(np.float32)


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
        frame = obs.images[camera]
        rr.log(f"camera_feed/{short}", rr.Image(frame))
        overlay = _upsample_and_normalize(attention.grid, frame.shape[0], frame.shape[1])
        rr.log(f"camera_feed/{short}/attention", rr.Image(overlay))
        rr.log(f"metrics/mass/{short}", rr.Scalars(attention.mass))
        ablation = analysis.ablations.get(camera)
        if ablation is not None:
            rr.log(
                f"metrics/ablation_mm/{short}", rr.Scalars(ablation.delta_mm)
            )
    rr.log("metrics/mass/language", rr.Scalars(analysis.language_mass))
