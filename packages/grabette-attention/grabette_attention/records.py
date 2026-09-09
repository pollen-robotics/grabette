"""Plain data passed between the four layers of the tool.

Nothing here knows about tokens, patches, torch or file paths. Both front ends
consume `FrameAnalysis`, which is what makes "PNG now, rerun later" free.

Every per-camera quantity is a mapping keyed by the camera's feature name. This
is deliberate: the tool must work unchanged when a second or third view is
added, so no code may index cameras by position.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FrameObservation:
    """One observation, exactly as recorded — full resolution, un-normalised.

    The policy's own preprocessing does the resizing and normalisation, which is
    the property that lets a dataset frame and a `--dump_obs` frame be used
    interchangeably.

    images: camera feature name -> HWC uint8 RGB.
    """

    episode: int
    frame: int
    images: dict[str, np.ndarray]
    state: np.ndarray
    task: str


@dataclass(frozen=True)
class CameraAttention:
    """Attention over one camera's image patches.

    grid: (rows, cols) float32. Letterbox padding rows are ALREADY removed, so
        the grid maps onto real image content only.
    mass: this camera's share of the prefix attention. Shares over all cameras
        plus `FrameAnalysis.language_mass` sum to 1.
    """

    grid: np.ndarray
    mass: float


@dataclass(frozen=True)
class ViewAblation:
    """How much the commanded chunk changed when this camera was removed.

    Both figures are in MILLIMETRES and are computed on the translation channels
    of the chunk. `delta_mm` is the RMS over chunk steps of the 3-D difference;
    `per_axis_mm` is the RMS per axis, in channel order.

    AXIS CONVENTION for this project's action space -- the standard OpenCV
    CAMERA frame, so the axes are relative to the gripper-mounted view, not to
    the world:

        per_axis_mm[0]  x  lateral, +right
        per_axis_mm[1]  y  VERTICAL, +DOWN
        per_axis_mm[2]  z  DEPTH / range, +FORWARD along the optical axis

    Vertical and range are different axes and they behave very differently, so
    do not read either off the wrong slot. The README records how this was
    measured (172 episodes) and how to re-verify it for another robot.
    """

    delta_mm: float
    per_axis_mm: tuple[float, float, float]


@dataclass(frozen=True)
class FrameAnalysis:
    """Everything computed for one frame. No plotting, no paths.

    provenance carries what a reader needs months later: checkpoint, which
    layers were aggregated, which denoising step, the noise seed, and how the
    frame was chosen.
    """

    episode: int
    frame: int
    cameras: dict[str, CameraAttention]
    language_mass: float
    ablations: dict[str, ViewAblation]
    provenance: dict[str, str] = field(default_factory=dict)
