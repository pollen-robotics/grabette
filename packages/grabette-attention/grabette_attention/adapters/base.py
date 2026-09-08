"""The interface the analysis layer talks to.

Nothing above an adapter knows about tokens, patches, letterboxing or torch. A
second policy (the Diffusion adapter, next) implements the same five members;
its per-camera "attention" comes from a CNN's spatial-softmax layer rather than
token attention, which is exactly why the analysis layer is written against
this protocol and not against pi0.5.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..layout import LetterboxGeometry, TokenLayout
from ..records import FrameObservation


@dataclass
class RunResult:
    """One forward pass: the predicted chunk and, optionally, its attention.

    Both come from the SAME pass, so a baseline is not paid for twice.
    captures is keyed by (denoise step, layer); empty when capture was off.
    """

    chunk: np.ndarray
    captures: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


class PolicyAdapter(Protocol):
    """What the analysis layer needs from any policy."""

    @property
    def camera_keys(self) -> tuple[str, ...]:
        """Cameras in the order the POLICY assembles them."""

    @property
    def patch(self) -> int:
        """Side of one spatial cell in model-input pixels."""

    def geometry(self, obs: FrameObservation, camera: str) -> LetterboxGeometry:
        """How that camera's recorded frame maps into the model input."""

    def layout(
        self, obs: FrameObservation, *, drop_camera: str | None = None
    ) -> TokenLayout:
        """Spatial layout for this observation, including which views are masked."""

    def draw_noise(self) -> Any:
        """One noise tensor, shared by the baseline and every ablation."""

    def run(
        self,
        obs: FrameObservation,
        noise: Any,
        *,
        capture: bool,
        drop_camera: str | None = None,
    ) -> RunResult:
        """Predict a chunk, optionally capturing attention, optionally with a
        camera removed."""
