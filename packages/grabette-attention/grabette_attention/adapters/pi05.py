"""pi0.5 adapter: hooks for attention, mask mutation for ablation.

Two facts from the verified internals make this small:

1. `sample_actions` forces eager attention and the stock Gemma attention module
   returns `(output, weights)`, so a forward hook is enough (see capture.py).
2. `predict_action_chunk` is a six-line wrapper around `_preprocess_images` plus
   `sample_actions(images, img_masks, tokens, masks, noise=...)`. Reproducing
   those six lines lets us mutate the image masks in place, which is how a view
   is removed without disturbing token order.

Removing a view by zeroing its mask (rather than dropping the batch key) matters
twice over: absent keys are appended AFTER the present ones, which would change
token order and position ids; and pi0.5 is trained with missing-camera masking,
so a masked view is in distribution while a grey rectangle is not.
"""

from typing import Any, Callable

import numpy as np
import torch

from ..capture import AttentionCapture
from ..layout import LetterboxGeometry, TokenLayout
from ..records import FrameObservation
from .base import RunResult


class Pi05Adapter:
    """Wraps a loaded pi0.5 policy and its checkpoint preprocessor."""

    def __init__(
        self,
        policy: Any,
        preprocessor: Callable[[dict], dict],
        postprocessor: Callable[[Any], Any],
        *,
        device: str = "cpu",
        seed: int = 0,
    ):
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._device = device
        self._seed = seed
        self._config = policy.config
        self._expert_layers = self._find_expert_layers(policy)

    # ---- static description -------------------------------------------------

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return tuple(self._config.image_features)

    @property
    def patch(self) -> int:
        """Side of one spatial cell in model-input pixels, read from the model.

        No fallback: a wrong guess here would keep tokens-per-image and grid
        rows*cols mutually consistent by construction while silently
        disagreeing with the tensor `reduce_attention` actually receives, which
        is exactly the "looks plausible and means nothing" failure
        `_find_expert_layers` below refuses to risk.
        """
        vision = getattr(
            getattr(getattr(self._policy, "model", None), "paligemma_with_expert", None),
            "paligemma", None,
        )
        try:
            return int(vision.model.vision_tower.config.patch_size)
        except AttributeError as exc:
            raise AttributeError(
                "expected policy.model.paligemma_with_expert.paligemma.model."
                "vision_tower.config.patch_size; "
                f"{type(self._policy).__name__} does not expose it"
            ) from exc

    def _resolution(self) -> tuple[int, int]:
        return tuple(self._config.image_resolution)

    def geometry(self, obs: FrameObservation, camera: str) -> LetterboxGeometry:
        frame = obs.images[camera]
        return LetterboxGeometry.from_shapes(
            src_hw=(frame.shape[0], frame.shape[1]), dst_hw=self._resolution()
        )

    def layout(
        self, obs: FrameObservation, *, drop_camera: str | None = None
    ) -> TokenLayout:
        """Present cameras first, absent ones appended — the policy's own order.

        A dropped camera is reported as masked but keeps its slot, because the
        ablation zeroes its mask in place rather than removing it.
        """
        present = tuple(k for k in self.camera_keys if k in obs.images)
        absent = tuple(k for k in self.camera_keys if k not in obs.images)
        masked = set(absent)
        if drop_camera is not None:
            if drop_camera not in self.camera_keys:
                raise KeyError(
                    f"{drop_camera!r} is not one of {self.camera_keys}"
                )
            masked.add(drop_camera)

        dst_h, dst_w = self._resolution()
        patch = self.patch
        rows, cols = dst_h // patch, dst_w // patch
        return TokenLayout(
            camera_keys=present + absent,
            tokens_per_image=rows * cols,
            grid_rows=rows,
            grid_cols=cols,
            language_tokens=int(self._config.tokenizer_max_length),
            masked_cameras=frozenset(masked),
        )

    # ---- running -------------------------------------------------------------

    def draw_noise(self) -> torch.Tensor:
        """Deterministic from the adapter's seed, so a whole run is repeatable.

        Shape follows `sample_actions`: (1, chunk_size, max_action_dim).
        """
        generator = torch.Generator(device="cpu").manual_seed(self._seed)
        return torch.randn(
            (1, int(self._config.chunk_size), int(self._config.max_action_dim)),
            generator=generator,
            dtype=torch.float32,
        ).to(self._device)

    def run(
        self,
        obs: FrameObservation,
        noise: Any,
        *,
        capture: bool,
        drop_camera: str | None = None,
    ) -> RunResult:
        if drop_camera is not None and drop_camera not in self.camera_keys:
            raise KeyError(f"{drop_camera!r} is not one of {self.camera_keys}")

        batch = self._preprocessor(self._build_batch(obs))
        images, img_masks = self._policy._preprocess_images(batch)

        if drop_camera is not None:
            index = self.layout(obs).camera_index(drop_camera)
            # Exactly how the policy represents an absent camera.
            images[index] = torch.ones_like(images[index]) * -1
            img_masks[index] = torch.zeros_like(img_masks[index])

        tokens = batch["observation.language.tokens"]
        masks = batch["observation.language.attention_mask"]

        if not capture:
            with torch.no_grad():
                chunk = self._policy.model.sample_actions(
                    images, img_masks, tokens, masks, noise=noise
                )
            chunk = self._finalize_chunk(chunk)
            return RunResult(chunk=self._to_numpy(chunk))

        with AttentionCapture(self._expert_layers) as cap, torch.no_grad():
            chunk = self._policy.model.sample_actions(
                images, img_masks, tokens, masks, noise=noise
            )
            captures = dict(cap.captures)
        chunk = self._finalize_chunk(chunk)
        return RunResult(chunk=self._to_numpy(chunk), captures=captures)

    # ---- helpers -------------------------------------------------------------

    def _finalize_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Truncate to the real action width, then convert to real units.

        `sample_actions` returns `(batch, chunk_size, max_action_dim)`: padded,
        and in the policy's normalized (quantile) action space. `predict_action_chunk`
        does both these steps before returning; calling `sample_actions` directly,
        as `run()` does to reach the attention hooks, skips them, so this mirrors
        `predict_action_chunk`'s truncation and applies the postprocessor
        (Unnormalizer + AbsoluteActions) that converts to real units.
        """
        original_action_dim = int(self._config.output_features["action"].shape[0])
        chunk = chunk[:, :, :original_action_dim]
        return self._postprocessor(chunk)

    def _build_batch(self, obs: FrameObservation) -> dict:
        """Frames as the policy expects them: CHW float32 in [0, 1], batch of 1.

        No resizing here; `_preprocess_images` letterboxes internally, which is
        what keeps a dataset frame and a dump_obs frame interchangeable.
        """
        batch: dict[str, Any] = {}
        for key, frame in obs.images.items():
            tensor = torch.from_numpy(np.ascontiguousarray(frame)).float() / 255.0
            batch[key] = tensor.permute(2, 0, 1).unsqueeze(0).to(self._device)
        batch["observation.state"] = (
            torch.from_numpy(np.asarray(obs.state, dtype=np.float32))
            .unsqueeze(0)
            .to(self._device)
        )
        batch["task"] = obs.task
        return batch

    def _to_numpy(self, chunk: Any) -> np.ndarray:
        array = chunk.detach().to("cpu").float().numpy()
        return array[0] if array.ndim == 3 else array

    @staticmethod
    def _find_expert_layers(policy: Any) -> list[Any]:
        """The action expert's attention modules — the queries we want.

        Resolved by the real attribute path only. A policy that does not expose
        it is a programming error to surface, not something to guess around: a
        fallback here would silently hook the wrong modules and produce maps
        that look plausible and mean nothing.
        """
        try:
            expert = policy.model.paligemma_with_expert.gemma_expert
        except AttributeError as exc:
            raise AttributeError(
                "expected policy.model.paligemma_with_expert.gemma_expert; "
                f"{type(policy).__name__} does not expose the pi0.5 module tree"
            ) from exc
        return [layer.self_attn for layer in expert.model.layers]
