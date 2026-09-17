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
class PromptAblation:
    """How much the commanded chunk changed under a different prompt.

    Same metric as `ViewAblation`, applied to the language input instead of a
    camera: MILLIMETRES of RMS change in the chunk's translation channels.
    `per_axis_mm` follows the same axis convention.
    """

    prompt: str
    delta_mm: float
    per_axis_mm: tuple[float, ...]


@dataclass(frozen=True)
class PromptSensitivity:
    """A frame's response to a set of alternative prompts.

    baseline_mm is the RMS translation magnitude the policy commands under its
    real prompt, so a variant's delta can be read as a fraction of the motion
    it perturbs rather than as a bare number.
    """

    baseline_prompt: str
    baseline_mm: float
    variants: tuple[PromptAblation, ...]


@dataclass(frozen=True)
class OcclusionMap:
    """Per-region causal effect of covering part of one camera's image.

    grid: (rows, cols) float32, each cell the change in the commanded chunk's
        translation, in MILLIMETRES, when that block of the source image was
        covered — see `metric` for which change. Larger means the region
        carried more of the motion. The grid spans the source image, so
        passing `CameraAttention.grid`'s shape makes the two directly
        comparable cell for cell.
    baseline_mm: RMS translation magnitude of the untouched chunk, so a cell
        can be read against the size of the motion it perturbs.
    fill: what covered blocks were painted with. Every choice is off
        distribution, so comparisons BETWEEN cells are meaningful while an
        absolute cell value is not.
    """

    camera: str
    grid: np.ndarray
    baseline_mm: float
    fill: str
    # Which quantity the grid holds. "rms" is the RMS difference over chunk
    # steps; "endpoint" is the distance between where the two trajectories
    # END. They look identical as arrays and are not interchangeable -- only
    # "endpoint" is comparable across action representations.
    metric: str = "rms"


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
    # RMS translation magnitude of the untouched chunk, in millimetres. Carried
    # because an ablation delta is only interpretable against the motion it
    # perturbs -- and because it is the ONLY way to compare across action
    # representations: see metrics.translation_magnitude_mm. Deliberately has no
    # default, so a caller cannot silently report a delta with no scale.
    baseline_mm: float
    provenance: dict[str, str] = field(default_factory=dict)
