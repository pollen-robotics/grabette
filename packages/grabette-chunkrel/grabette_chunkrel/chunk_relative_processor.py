"""LeRobot processor steps for chunk-relative actions.

A thin torch/lerobot shell around `chunk_relative`, which holds the maths. Kept
separate so the maths stays testable without torch and without lerobot.

WHY A CUSTOM STEP AND NOT `use_relative_actions`
------------------------------------------------
LeRobot's built-in relative-action processor subtracts `observation.state`
elementwise. That is correct for joint-space actions (each dim a 1-DOF rotation
about a fixed axis, which commutes) and wrong for a Cartesian pose: subtraction is
not rotation composition, and it yields world-frame offsets, which move with the
arbitrary SLAM origin. See `docs/relative_actions_lerobot_native.md`.

DEPLOYMENT NOTE — read before shipping a checkpoint
---------------------------------------------------
A saved pipeline stores step *names*, resolved through `ProcessorStepRegistry` at
load time, and the registry is populated by import side-effect. So this module
must be imported wherever a checkpoint using it is loaded: the training container,
the inference server, the offline gates, and the eval loop. Plain lerobot cannot
load such a checkpoint — the price of leaving "stock upstream everywhere".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, TransitionKey

from grabette_chunkrel.chunk_relative import (
    DEFAULT_JUMP_CAP_DEG,
    DEFAULT_JUMP_CAP_M,
    from_chunk_relative,
    to_chunk_relative,
)

STEP_NAME = "grabette_chunk_relative_actions"
INVERSE_STEP_NAME = "grabette_absolute_from_chunk_relative"


@ProcessorStepRegistry.register(STEP_NAME)
@dataclass
class ChunkRelativeActionsStep(ProcessorStep):
    """Absolute pose action chunks -> offsets from each chunk's first pose.

    Operates on `(B, T, D)` action tensors. A `(B, D)` action (single step, as at
    inference) has no chunk to be relative to and is passed through unchanged —
    the inverse step is what runs on the policy's output.

    Caches each batch's reference poses so the paired inverse step can restore
    absolute actions.
    """

    enabled: bool = True
    rot: str = "rotvec"
    jump_cap_m: float = DEFAULT_JUMP_CAP_M
    jump_cap_deg: float = DEFAULT_JUMP_CAP_DEG
    _last_reference: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_report: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or action.ndim != 3:
            return new_transition

        # The reference is the chunk's FIRST action, not observation.state: it keeps
        # the state gripper-only (no absolute SLAM pose in the policy input) and
        # makes the step self-contained.
        self._last_reference = action[:, 0, :6].detach().clone()

        arr = action.detach().cpu().numpy().astype(np.float64)
        out, jumps, rot_jumps = [], 0, 0
        for chunk in arr:
            rel, rep = to_chunk_relative(
                chunk, rot=self.rot,
                jump_cap_m=self.jump_cap_m, jump_cap_deg=self.jump_cap_deg,
            )
            out.append(rel)
            jumps += rep["n_jumps_spliced"]
            rot_jumps += rep["n_rot_jumps_spliced"]
        self._last_report = {"n_jumps": jumps, "n_rot_jumps": rot_jumps}

        new_transition[TransitionKey.ACTION] = torch.as_tensor(
            np.stack(out), dtype=action.dtype, device=action.device
        )
        return new_transition

    def get_reference(self) -> torch.Tensor | None:
        """Reference poses of the last batch, for the paired inverse step."""
        return self._last_reference

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "rot": self.rot,
            "jump_cap_m": self.jump_cap_m,
            "jump_cap_deg": self.jump_cap_deg,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # `rotvec` preserves the action dimension. `r6d` would widen 8 -> 11 and
        # therefore has to rewrite the action feature; refuse rather than emit a
        # spec that disagrees with the tensors.
        if self.enabled and self.rot != "rotvec":
            raise NotImplementedError(
                f"rot={self.rot!r} changes the action dimension; the action "
                "PolicyFeature must be rewritten before this is usable"
            )
        return features


@ProcessorStepRegistry.register(INVERSE_STEP_NAME)
@dataclass
class AbsoluteFromChunkRelativeStep(ProcessorStep):
    """Offsets -> absolute poses, using a reference pose.

    At training/replay time the reference comes from the paired forward step. At
    inference there is no forward step in the loop, so the caller sets
    `reference` explicitly to the robot's MEASURED pose — which is what "the pose
    at prediction time" means once a real arm is involved.
    """

    enabled: bool = True
    rot: str = "rotvec"
    relative_step: ChunkRelativeActionsStep | None = None
    reference: torch.Tensor | None = None

    def set_reference(self, pose: torch.Tensor | np.ndarray) -> None:
        """Set the absolute pose the next conversion is relative to (6D)."""
        self.reference = torch.as_tensor(np.asarray(pose, dtype=np.float32))

    def _resolve_reference(self, batch: int) -> torch.Tensor | None:
        if self.reference is not None:
            ref = self.reference
            return ref.unsqueeze(0).expand(batch, -1) if ref.ndim == 1 else ref
        if self.relative_step is not None:
            return self.relative_step.get_reference()
        return None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        single = action.ndim == 2                       # (T, D) — one sample
        act = action.unsqueeze(0) if single else action
        if act.ndim != 3:
            return new_transition

        ref = self._resolve_reference(act.shape[0])
        if ref is None:
            raise ValueError(
                "no reference pose available: set_reference() with the robot's "
                "measured pose, or pair this step with a ChunkRelativeActionsStep"
            )

        arr = act.detach().cpu().numpy().astype(np.float64)
        ref_np = ref.detach().cpu().numpy().astype(np.float64)
        out = [from_chunk_relative(c, r, rot=self.rot) for c, r in zip(arr, ref_np)]
        res = torch.as_tensor(np.stack(out), dtype=action.dtype, device=action.device)

        new_transition[TransitionKey.ACTION] = res.squeeze(0) if single else res
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "rot": self.rot}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
