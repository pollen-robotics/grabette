# Attention Maps and View Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline tool that shows, per camera, where a pi0.5 policy's action tokens attend on each frame, and how much the predicted action chunk changes when each camera is removed.

**Architecture:** A new workspace package `packages/grabette-attention` with four layers: frame sources (dataset or `--dump_obs`), a policy adapter holding every model-specific fact, a pure analysis step producing plain records, and two front ends (PNG first, rerun after) that consume those records. Attention comes from forward hooks on the action expert's attention modules; no `lerobot` code is modified.

**Tech Stack:** Python >= 3.11, numpy (declared), torch and lerobot (host-provided, never declared), pytest, matplotlib or opencv for the PNG front end, rerun for the second front end. Build backend hatchling.

**Spec:** `docs/superpowers/specs/2026-09-08-attention-maps-design.md`

## Global Constraints

Copied verbatim from the spec. Every task's requirements implicitly include these.

- **Multi-camera genericity.** Tokens per image, grid shape, camera order and the pixel mapping are read from the model and its config at runtime. Hard-coding any of them fails a test. Every map and every scalar is keyed by camera name, never by position; outputs are mappings. Ablation works for any number of views. Absent or padded camera blocks are excluded explicitly.
- **Camera order** is the order the policy assembles them: present cameras in `config.image_features` order first, then absent ones appended after, matching `_preprocess_images`.
- **Ablation mechanism** is zeroing the ablated camera's image mask in place and setting its pixels to the padding value, never dropping the batch key (which reorders tokens and changes position ids).
- **Determinism.** The baseline and every ablation for a frame share one pre-drawn noise tensor, passed explicitly. No global seeding.
- **Units.** Deltas are returned in millimetres. Pin this with an explicit test; we have shipped a metre-labelled-as-millimetre bug once already.
- **Offline only.** No runtime signal on the robot, no remote inference, no modification to `lerobot`.
- **dtype.** pi0.5 runs fp32 because the bf16 flow path is broken. `compile_model` is forced to `False` so hooks are not swallowed by a graph.
- **Dependency discipline.** The package declares `numpy` only. torch and lerobot are host-provided, following `packages/grabette-chunkrel/pyproject.toml`. Any lerobot pin carries the marker `; python_version >= '3.12'`.
- **Lint.** Workspace ruff selects `E4`, `E7`, `E9`, `F`. No unused imports, no undefined names.
- **Interpretation guard.** pi0.5 attention is broad and low-peak by nature and action fine-tuning makes it diffuse. Nothing in the output may imply a crisp map is expected.

---

### Task 1: Package scaffold and analysis records

**Files:**
- Create: `packages/grabette-attention/pyproject.toml`
- Create: `packages/grabette-attention/grabette_attention/__init__.py`
- Create: `packages/grabette-attention/grabette_attention/records.py`
- Test: `packages/grabette-attention/tests/test_records.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FrameObservation(episode: int, frame: int, images: dict[str, np.ndarray], state: np.ndarray, task: str)`, `CameraAttention(grid: np.ndarray, mass: float)`, `ViewAblation(delta_mm: float, per_axis_mm: tuple[float, float, float])`, `FrameAnalysis(episode: int, frame: int, cameras: dict[str, CameraAttention], language_mass: float, ablations: dict[str, ViewAblation], provenance: dict[str, str])`.

- [ ] **Step 1: Write the failing test**

```python
# packages/grabette-attention/tests/test_records.py
"""The records are plain data: keyed by camera NAME, never by position."""
import numpy as np
import pytest

from grabette_attention.records import (
    CameraAttention,
    FrameAnalysis,
    FrameObservation,
    ViewAblation,
)


def test_frame_observation_keeps_cameras_keyed_by_name():
    obs = FrameObservation(
        episode=3,
        frame=42,
        images={"observation.images.cam0": np.zeros((720, 960, 3), np.uint8)},
        state=np.zeros(2, np.float32),
        task="pick the sugar cube",
    )
    assert list(obs.images) == ["observation.images.cam0"]
    assert obs.images["observation.images.cam0"].shape == (720, 960, 3)


def test_frame_analysis_holds_one_entry_per_camera_and_is_immutable():
    cams = {
        "cam0": CameraAttention(grid=np.ones((12, 16), np.float32), mass=0.7),
        "cam1": CameraAttention(grid=np.ones((12, 16), np.float32), mass=0.1),
    }
    analysis = FrameAnalysis(
        episode=3,
        frame=42,
        cameras=cams,
        language_mass=0.2,
        ablations={"cam0": ViewAblation(delta_mm=8.4, per_axis_mm=(1.0, 2.0, 8.1))},
        provenance={"checkpoint": "x", "denoise_step": "last"},
    )
    assert set(analysis.cameras) == {"cam0", "cam1"}
    assert analysis.ablations["cam0"].delta_mm == 8.4
    with pytest.raises(Exception):
        analysis.episode = 4


def test_masses_over_the_prefix_sum_to_one():
    cams = {
        "cam0": CameraAttention(grid=np.ones((12, 16), np.float32), mass=0.55),
        "cam1": CameraAttention(grid=np.ones((12, 16), np.float32), mass=0.25),
    }
    analysis = FrameAnalysis(
        episode=0, frame=0, cameras=cams, language_mass=0.20,
        ablations={}, provenance={},
    )
    total = sum(c.mass for c in analysis.cameras.values()) + analysis.language_mass
    assert total == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_records.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grabette_attention'`

- [ ] **Step 3: Write the scaffold and the records**

```toml
# packages/grabette-attention/pyproject.toml
[project]
name = "grabette-attention"
version = "0.1.0"
description = "Offline attention maps and view-ablation diagnostics for Grabette policies."
requires-python = ">=3.11"

# DELIBERATELY MINIMAL, for the same reason as grabette-chunkrel: this package is
# installed into environments that already provide torch and lerobot (the eval
# venv, the Pi05 integration). Declaring either would reproduce the
# URL-pin-vs-version-pin conflict those environments already carry. The analysis
# core (records, layout, reduction) needs numpy alone and imports neither.
dependencies = ["numpy"]

[project.optional-dependencies]
# The PNG front end. matplotlib only; the repo already uses the Agg backend for
# offline plots (integrations/DiffusionPolicy/offline_eval.py).
png = ["matplotlib"]
# The second front end, added after the PNG one.
rerun = ["rerun-sdk"]
# torch is needed to exercise the forward-hook capture with stub modules.
# lerobot carries the workspace's usual >= 3.12 marker: it declares
# Python>=3.12, and an unguarded pin makes `uv lock` fail on the 3.11 split.
test = ["pytest", "torch", "matplotlib", "lerobot==0.6.0 ; python_version >= '3.12'"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["grabette_attention"]
```

```python
# packages/grabette-attention/grabette_attention/__init__.py
"""Offline attention-map and view-ablation diagnostics for Grabette policies.

Read `docs/attention_saliency_review.md` before drawing conclusions from the
maps: on pi0.5 an interventional score is measurably more faithful than
attention, so the maps are hypothesis generators and the ablation numbers are
the evidence.
"""
```

```python
# packages/grabette-attention/grabette_attention/records.py
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
    `per_axis_mm` is the RMS per axis, which is what tells us whether a view
    carries the vertical (range) information.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_records.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add packages/grabette-attention
git commit -m "feat(attention): package scaffold and analysis records"
```

---

### Task 2: Letterbox geometry

**Files:**
- Create: `packages/grabette-attention/grabette_attention/layout.py`
- Test: `packages/grabette-attention/tests/test_layout.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `LetterboxGeometry` with `from_shapes(src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> LetterboxGeometry`, fields `src_h, src_w, dst_h, dst_w, ratio, pad_top, pad_bottom, pad_left, pad_right`, methods `to_source_pixel(u: float, v: float) -> tuple[float, float]` and `content_rows(patch: int) -> tuple[int, int]`.

This mirrors `resize_with_pad_torch` in lerobot's `modeling_pi05.py`: `ratio = max(src_w/dst_w, src_h/dst_h)`, resized sides are `int(src/ratio)`, padding is centred with the extra pixel going to bottom and right.

- [ ] **Step 1: Write the failing test**

```python
# packages/grabette-attention/tests/test_layout.py
"""Letterbox geometry, derived from the actual shapes — never hard-coded.

The numbers in these tests are the ones lerobot's resize_with_pad_torch
produces; if this file's arithmetic drifts from it, the overlays land on the
wrong pixels and nothing else catches it.
"""
import pytest

from grabette_attention.layout import LetterboxGeometry


def test_a_four_by_three_frame_is_padded_top_and_bottom():
    # The live camera: 960x720 into a 224 square.
    g = LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224))
    assert g.ratio == pytest.approx(960 / 224)
    assert (g.pad_top, g.pad_bottom) == (28, 28)
    assert (g.pad_left, g.pad_right) == (0, 0)


def test_the_downscaled_training_copy_gives_the_same_padding():
    # 480x360 is the training copy; same aspect, so the same 28-pixel bands.
    g = LetterboxGeometry.from_shapes(src_hw=(360, 480), dst_hw=(224, 224))
    assert (g.pad_top, g.pad_bottom) == (28, 28)
    assert g.ratio == pytest.approx(480 / 224)


def test_the_extra_pixel_goes_to_the_bottom():
    # An odd leftover must not be split evenly; lerobot gives it to bottom/right.
    g = LetterboxGeometry.from_shapes(src_hw=(100, 224), dst_hw=(224, 224))
    assert g.pad_bottom == g.pad_top + 1


def test_a_square_frame_needs_no_padding():
    g = LetterboxGeometry.from_shapes(src_hw=(480, 480), dst_hw=(224, 224))
    assert (g.pad_top, g.pad_bottom, g.pad_left, g.pad_right) == (0, 0, 0, 0)


def test_the_centre_of_the_letterbox_maps_to_the_centre_of_the_source():
    g = LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224))
    x, y = g.to_source_pixel(112.0, 112.0)
    assert x == pytest.approx(480.0, abs=1.0)
    assert y == pytest.approx(360.0, abs=1.0)


def test_the_top_of_the_content_maps_to_the_top_of_the_source():
    g = LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224))
    _, y = g.to_source_pixel(0.0, float(g.pad_top))
    assert y == pytest.approx(0.0, abs=1.0)


def test_padding_rows_are_excluded_from_the_content_rows():
    # 28 px of padding at patch 14 means grid rows 0-1 and 14-15 are pure pad.
    g = LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224))
    assert g.content_rows(patch=14) == (2, 14)


def test_an_unpadded_frame_keeps_every_row():
    g = LetterboxGeometry.from_shapes(src_hw=(480, 480), dst_hw=(224, 224))
    assert g.content_rows(patch=14) == (0, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_layout.py -v`
Expected: FAIL with `ImportError: cannot import name 'LetterboxGeometry'`

- [ ] **Step 3: Write the implementation**

```python
# packages/grabette-attention/grabette_attention/layout.py
"""Geometry and token bookkeeping. Pure numpy; no torch, no lerobot.

This module is where multi-camera genericity is won or lost. Every number it
returns is derived from shapes and config passed in by the caller. Nothing here
may assume a patch count, a grid size, or how many cameras exist.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LetterboxGeometry:
    """Maps a letterboxed model input back to the source frame's pixels.

    Mirrors lerobot's `resize_with_pad_torch`: the aspect ratio is preserved,
    the shorter axis is padded symmetrically, and an odd leftover pixel goes to
    the bottom (or right). Padding is black, which after the policy's `*2-1`
    becomes -1, so those regions carry no image content and their attention must
    not be plotted.
    """

    src_h: int
    src_w: int
    dst_h: int
    dst_w: int
    ratio: float
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    @classmethod
    def from_shapes(
        cls, src_hw: tuple[int, int], dst_hw: tuple[int, int]
    ) -> "LetterboxGeometry":
        src_h, src_w = src_hw
        dst_h, dst_w = dst_hw
        # max(), so the whole source fits inside the target and we pad rather
        # than crop. This is lerobot's choice, not ours to change.
        ratio = max(src_w / dst_w, src_h / dst_h)
        resized_h = int(src_h / ratio)
        resized_w = int(src_w / ratio)
        pad_top, rem_h = divmod(dst_h - resized_h, 2)
        pad_left, rem_w = divmod(dst_w - resized_w, 2)
        return cls(
            src_h=src_h,
            src_w=src_w,
            dst_h=dst_h,
            dst_w=dst_w,
            ratio=ratio,
            pad_top=pad_top,
            pad_bottom=pad_top + rem_h,
            pad_left=pad_left,
            pad_right=pad_left + rem_w,
        )

    def to_source_pixel(self, u: float, v: float) -> tuple[float, float]:
        """Letterboxed pixel (u across, v down) -> source pixel (x, y)."""
        return ((u - self.pad_left) * self.ratio, (v - self.pad_top) * self.ratio)

    def content_rows(self, patch: int) -> tuple[int, int]:
        """Half-open range of grid rows that carry image content.

        A row that is only partly padded is KEPT, which is the conservative
        choice: we would rather show a slightly-too-tall map than silently drop
        real content.
        """
        first = self.pad_top // patch
        content_end = self.dst_h - self.pad_bottom
        last = -(-content_end // patch)  # ceiling division
        return first, last

    def content_cols(self, patch: int) -> tuple[int, int]:
        """Half-open range of grid columns that carry image content."""
        first = self.pad_left // patch
        content_end = self.dst_w - self.pad_right
        last = -(-content_end // patch)
        return first, last
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_layout.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add packages/grabette-attention
git commit -m "feat(attention): letterbox geometry derived from actual shapes"
```

---

### Task 3: Token layout

**Files:**
- Modify: `packages/grabette-attention/grabette_attention/layout.py`
- Test: `packages/grabette-attention/tests/test_token_layout.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `TokenLayout(camera_keys: tuple[str, ...], tokens_per_image: int, grid_rows: int, grid_cols: int, language_tokens: int, masked_cameras: frozenset[str])` with property `prefix_len: int`, methods `camera_slice(key: str) -> slice`, `language_slice() -> slice`, `token_to_cell(index: int) -> tuple[str, int, int]`, and `visible_cameras() -> tuple[str, ...]`.

The layout is `[cam_0 tokens][cam_1 tokens]...[language tokens]`, patch order row-major, exactly as `embed_prefix` assembles it.

- [ ] **Step 1: Write the failing test**

```python
# packages/grabette-attention/tests/test_token_layout.py
"""Token index <-> (camera, row, col), for ANY number of cameras.

These tests are the multi-camera acceptance criteria in executable form. If
someone hard-codes 256 tokens or a single view, the two- and three-camera cases
here fail.
"""
import pytest

from grabette_attention.layout import TokenLayout


def one_camera() -> TokenLayout:
    return TokenLayout(
        camera_keys=("cam0",), tokens_per_image=256, grid_rows=16, grid_cols=16,
        language_tokens=200, masked_cameras=frozenset(),
    )


def three_cameras() -> TokenLayout:
    return TokenLayout(
        camera_keys=("cam0", "cam1", "cam2"), tokens_per_image=256,
        grid_rows=16, grid_cols=16, language_tokens=200,
        masked_cameras=frozenset(),
    )


def test_the_single_camera_prefix_is_the_repos_default_length():
    # 256 image tokens + 200 language tokens, the shape a Grabette pi0.5 sees.
    assert one_camera().prefix_len == 456


def test_the_prefix_grows_with_each_camera():
    assert three_cameras().prefix_len == 3 * 256 + 200


def test_each_camera_occupies_its_own_block():
    layout = three_cameras()
    assert layout.camera_slice("cam0") == slice(0, 256)
    assert layout.camera_slice("cam1") == slice(256, 512)
    assert layout.camera_slice("cam2") == slice(512, 768)


def test_language_follows_the_last_camera():
    assert three_cameras().language_slice() == slice(768, 968)


def test_a_token_index_resolves_to_camera_row_and_column():
    layout = three_cameras()
    # Row-major within an image: token 17 of cam1 is row 1, col 1.
    assert layout.token_to_cell(256 + 17) == ("cam1", 1, 1)
    assert layout.token_to_cell(0) == ("cam0", 0, 0)
    assert layout.token_to_cell(255) == ("cam0", 15, 15)


def test_a_language_token_index_is_rejected_as_a_cell():
    layout = one_camera()
    with pytest.raises(ValueError, match="language"):
        layout.token_to_cell(300)


def test_a_non_square_grid_is_handled():
    # Nothing may assume rows == cols.
    layout = TokenLayout(
        camera_keys=("cam0",), tokens_per_image=32, grid_rows=4, grid_cols=8,
        language_tokens=10, masked_cameras=frozenset(),
    )
    assert layout.token_to_cell(9) == ("cam0", 1, 1)
    assert layout.prefix_len == 42


def test_masked_cameras_keep_their_slot_but_are_not_visible():
    # An absent camera still occupies tokens (padded, mask 0). It must keep its
    # slot so other cameras' indices stay right, and be excluded from output.
    layout = TokenLayout(
        camera_keys=("cam0", "cam1"), tokens_per_image=256, grid_rows=16,
        grid_cols=16, language_tokens=200, masked_cameras=frozenset({"cam1"}),
    )
    assert layout.camera_slice("cam1") == slice(256, 512)
    assert layout.visible_cameras() == ("cam0",)


def test_an_unknown_camera_is_an_error_not_a_guess():
    with pytest.raises(KeyError):
        one_camera().camera_slice("wrist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_token_layout.py -v`
Expected: FAIL with `ImportError: cannot import name 'TokenLayout'`

- [ ] **Step 3: Append the implementation to `layout.py`**

```python
# append to packages/grabette-attention/grabette_attention/layout.py


@dataclass(frozen=True)
class TokenLayout:
    """Where each camera's image tokens sit in the policy's prefix.

    The prefix is `[cam_0][cam_1]...[language]`, with patches in row-major order
    inside each camera's block, exactly as `embed_prefix` assembles it.

    `camera_keys` must be given in the order the POLICY assembles them, which is
    present cameras in `config.image_features` order followed by absent ones —
    not the order they appear in a batch. Absent cameras keep their slot (their
    tokens exist, padded, with mask 0) so that the indices of the cameras after
    them stay correct; `masked_cameras` records them so they are excluded from
    output instead of being plotted as attention on nothing.
    """

    camera_keys: tuple[str, ...]
    tokens_per_image: int
    grid_rows: int
    grid_cols: int
    language_tokens: int
    masked_cameras: frozenset[str]

    @property
    def prefix_len(self) -> int:
        return len(self.camera_keys) * self.tokens_per_image + self.language_tokens

    @property
    def image_tokens(self) -> int:
        return len(self.camera_keys) * self.tokens_per_image

    def camera_index(self, key: str) -> int:
        try:
            return self.camera_keys.index(key)
        except ValueError as exc:
            raise KeyError(
                f"{key!r} is not one of this policy's cameras {self.camera_keys}"
            ) from exc

    def camera_slice(self, key: str) -> slice:
        start = self.camera_index(key) * self.tokens_per_image
        return slice(start, start + self.tokens_per_image)

    def language_slice(self) -> slice:
        return slice(self.image_tokens, self.image_tokens + self.language_tokens)

    def visible_cameras(self) -> tuple[str, ...]:
        return tuple(k for k in self.camera_keys if k not in self.masked_cameras)

    def token_to_cell(self, index: int) -> tuple[str, int, int]:
        """Prefix token index -> (camera key, grid row, grid column)."""
        if index < 0 or index >= self.prefix_len:
            raise ValueError(f"token {index} is outside the prefix ({self.prefix_len})")
        if index >= self.image_tokens:
            raise ValueError(
                f"token {index} is a language token, not an image patch"
            )
        camera = index // self.tokens_per_image
        within = index % self.tokens_per_image
        return (self.camera_keys[camera], within // self.grid_cols, within % self.grid_cols)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_token_layout.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add packages/grabette-attention
git commit -m "feat(attention): token layout for any number of cameras"
```

---

### Task 4: Attention capture with forward hooks

**Files:**
- Create: `packages/grabette-attention/grabette_attention/capture.py`
- Test: `packages/grabette-attention/tests/test_capture.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `AttentionCapture(modules: Sequence[Any])` as a context manager, with `.captures -> dict[tuple[int, int], np.ndarray]` keyed by `(denoise_step, layer)` and `.reset()`.

The bookkeeping rule from the spec: each layer's hook fires once per denoising step, so the denoising step is that layer's own invocation count. Never infer the layer from a global counter.

- [ ] **Step 1: Write the failing test**

```python
# packages/grabette-attention/tests/test_capture.py
"""Forward-hook capture and its (step, layer) bookkeeping.

pi0.5 runs 18 expert layers per denoising step and 10 steps per chunk, so a hook
on every layer fires 180 times. Getting the (step, layer) assignment wrong
silently mixes early and late denoising, which is exactly the axis nobody has
published — so it is pinned here.
"""
import numpy as np
import torch
from torch import nn

from grabette_attention.capture import AttentionCapture


class FakeAttention(nn.Module):
    """Stands in for GemmaAttention: returns (output, attention_weights)."""

    def __init__(self, layer: int, heads: int = 8, queries: int = 50, keys: int = 506):
        super().__init__()
        self.layer = layer
        self.heads, self.queries, self.keys = heads, queries, keys
        self.calls = 0

    def forward(self, x):
        # Weights encode (layer, call) so the test can assert the assignment.
        w = torch.full((1, self.heads, self.queries, self.keys), float(self.layer))
        w[0, 0, 0, 0] = float(self.calls)
        self.calls += 1
        return x, w


def test_it_captures_one_tensor_per_layer_per_step():
    layers = [FakeAttention(i) for i in range(18)]
    with AttentionCapture(layers) as cap:
        for _ in range(10):                       # 10 denoising steps
            for layer in layers:
                layer(torch.zeros(1))
    assert len(cap.captures) == 180
    assert set(cap.captures) == {(s, l) for s in range(10) for l in range(18)}


def test_the_step_is_each_layers_own_invocation_count():
    layers = [FakeAttention(i) for i in range(3)]
    with AttentionCapture(layers) as cap:
        for _ in range(4):
            for layer in layers:
                layer(torch.zeros(1))
    for (step, layer_idx), w in cap.captures.items():
        assert w[0, 0, 0] == step        # the call counter we stamped in
        assert w[0, 1, 1] == layer_idx   # the layer id we stamped in


def test_captures_are_numpy_with_the_batch_dimension_dropped():
    layers = [FakeAttention(0)]
    with AttentionCapture(layers) as cap:
        layers[0](torch.zeros(1))
    w = cap.captures[(0, 0)]
    assert isinstance(w, np.ndarray)
    assert w.shape == (8, 50, 506)       # (heads, queries, keys)


def test_hooks_are_removed_on_exit_so_later_passes_are_not_captured():
    layers = [FakeAttention(0)]
    with AttentionCapture(layers) as cap:
        layers[0](torch.zeros(1))
    layers[0](torch.zeros(1))            # an ablation pass, outside the block
    assert len(cap.captures) == 1


def test_hooks_are_removed_even_when_the_body_raises():
    layers = [FakeAttention(0)]
    try:
        with AttentionCapture(layers):
            raise RuntimeError("cuda oom")
    except RuntimeError:
        pass
    layers[0](torch.zeros(1))
    assert layers[0]._forward_hooks == {}


def test_reset_clears_the_counters_so_the_next_frame_starts_at_step_zero():
    layers = [FakeAttention(0)]
    with AttentionCapture(layers) as cap:
        layers[0](torch.zeros(1))
        cap.reset()
        layers[0](torch.zeros(1))
    assert set(cap.captures) == {(0, 0)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grabette_attention.capture'`

- [ ] **Step 3: Write the implementation**

```python
# packages/grabette-attention/grabette_attention/capture.py
"""Capture attention probabilities from a policy without modifying lerobot.

Why a plain forward hook is enough: at inference `sample_actions` sets
`_attn_implementation = "eager"` on both towers, and the stock Gemma attention
module returns `(attn_output, attn_weights)`. The caller discards the second
element, but a forward hook sees the full return value. The fused
`compute_layer_complete` path that drops the weights is TRAINING-ONLY and is not
on this code path.
"""

from typing import Any, Sequence

import numpy as np


class AttentionCapture:
    """Context manager that records attention weights per (denoise step, layer).

    Bookkeeping rule: each layer's hook fires exactly once per denoising step, so
    a layer's own invocation count IS the step index. Do not derive the layer
    from a global call counter — that breaks the moment a layer is skipped or the
    step count changes.

    Captures are numpy arrays of shape (heads, queries, keys) with the batch
    dimension dropped; batch size is 1 for this tool.
    """

    def __init__(self, modules: Sequence[Any]):
        self._modules = list(modules)
        self._handles: list[Any] = []
        self._calls: dict[int, int] = {}
        self.captures: dict[tuple[int, int], np.ndarray] = {}

    def reset(self) -> None:
        """Forget everything, so the next frame starts again at step 0."""
        self._calls.clear()
        self.captures.clear()

    def _make_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            # GemmaAttention returns (attn_output, attn_weights). SDPA returns
            # None for the weights, which means eager was not in force.
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if weights is None:
                raise RuntimeError(
                    "attention weights are None: the policy is not running eager "
                    "attention. pi0.5 sets this itself in sample_actions; a "
                    "vision tower needs _attn_implementation='eager' set first."
                )
            step = self._calls.get(layer_index, 0)
            self._calls[layer_index] = step + 1
            self.captures[(step, layer_index)] = (
                weights.detach().to("cpu", copy=True).float().numpy()[0]
            )

        return hook

    def __enter__(self) -> "AttentionCapture":
        for index, module in enumerate(self._modules):
            self._handles.append(module.register_forward_hook(self._make_hook(index)))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_capture.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add packages/grabette-attention
git commit -m "feat(attention): forward-hook capture with per-layer step bookkeeping"
```

---

### Task 5: Reduce captures to per-camera maps and mass

**Files:**
- Create: `packages/grabette-attention/grabette_attention/reduce.py`
- Test: `packages/grabette-attention/tests/test_reduce.py`

**Interfaces:**
- Consumes: `TokenLayout` and `LetterboxGeometry` from `layout.py` (Tasks 2-3), `CameraAttention` from `records.py` (Task 1).
- Produces: `reduce_attention(captures: dict[tuple[int, int], np.ndarray], layout: TokenLayout, geometries: Mapping[str, LetterboxGeometry], *, patch: int, denoise_step: str | int = "last", layers: str | Sequence[int] = "all") -> tuple[dict[str, CameraAttention], float]`.

**Note:** `geometries` is a mapping keyed by camera name, one entry per visible camera — NOT a single geometry. Views may differ in resolution and aspect ratio, so each camera's padding crop must come from its own frame. Applying one camera's geometry to all of them would assume every view shares a resolution, which the Global Constraints forbid.

- [ ] **Step 1: Write the failing test**

```python
# packages/grabette-attention/tests/test_reduce.py
"""Reducing (step, layer, head, query, key) attention to per-camera maps.

The reduction order matters and the mass definition matters. Masses are
fractions of the PREFIX, so cameras plus language sum to 1 — that is what makes
"which camera does the policy rely on" readable without renormalising away the
answer.
"""
import numpy as np
import pytest

from grabette_attention.layout import LetterboxGeometry, TokenLayout
from grabette_attention.reduce import reduce_attention

PATCH = 14


class _SameGeometryEverywhere(dict):
    """Test fixture: the 4:3 live-camera geometry, for whatever camera is asked.

    Real callers pass one entry per camera; these tests mostly use views that
    share a resolution, so a mapping that answers for any name keeps them short.
    `test_cameras_with_different_geometries_are_cropped_independently` uses a
    real per-camera dict, which is the case this mapping must not paper over.
    """

    def __missing__(self, _camera):
        return LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224))


GEOM = _SameGeometryEverywhere()


def layout(n_cams: int, masked=frozenset()) -> TokenLayout:
    return TokenLayout(
        camera_keys=tuple(f"cam{i}" for i in range(n_cams)),
        tokens_per_image=256, grid_rows=16, grid_cols=16,
        language_tokens=200, masked_cameras=masked,
    )


def captures(n_cams: int, steps: int = 10, layers: int = 18, value: float = 1.0):
    keys = n_cams * 256 + 200 + 50          # prefix + the 50 action tokens
    return {
        (s, l): np.full((8, 50, keys), value, np.float32)
        for s in range(steps) for l in range(layers)
    }


def test_uniform_attention_splits_mass_by_token_count():
    cams, lang = reduce_attention(
        captures(2), layout(2), GEOM, patch=PATCH,
    )
    # 256 + 256 image tokens and 200 language tokens, all equal.
    assert cams["cam0"].mass == pytest.approx(256 / 712)
    assert cams["cam1"].mass == pytest.approx(256 / 712)
    assert lang == pytest.approx(200 / 712)


def test_the_masses_sum_to_one_over_the_prefix():
    cams, lang = reduce_attention(captures(3), layout(3), GEOM, patch=PATCH)
    assert sum(c.mass for c in cams.values()) + lang == pytest.approx(1.0)


def test_action_token_attention_is_excluded_from_the_mass():
    # Pile weight onto the action-token keys; the prefix fractions must not move.
    caps = captures(1)
    for w in caps.values():
        w[:, :, 456:] = 100.0
    cams, lang = reduce_attention(caps, layout(1), GEOM, patch=PATCH)
    assert cams["cam0"].mass == pytest.approx(256 / 456)
    assert lang == pytest.approx(200 / 456)


def test_padding_rows_are_cropped_from_the_returned_grid():
    cams, _ = reduce_attention(captures(1), layout(1), GEOM, patch=PATCH)
    # 28 px of padding at patch 14 leaves grid rows 2..13, i.e. 12 rows.
    assert cams["cam0"].grid.shape == (12, 16)


def test_a_masked_camera_is_absent_from_the_result():
    cams, _ = reduce_attention(
        captures(2), layout(2, masked=frozenset({"cam1"})), GEOM, patch=PATCH,
    )
    assert set(cams) == {"cam0"}


def test_the_last_denoising_step_is_the_default():
    caps = captures(1, steps=3, layers=1, value=0.0)
    caps[(2, 0)][:] = 5.0                     # only the last step is hot
    cams, _ = reduce_attention(caps, layout(1), GEOM, patch=PATCH)
    assert cams["cam0"].grid.max() == pytest.approx(5.0)


def test_a_specific_denoising_step_can_be_selected():
    caps = captures(1, steps=3, layers=1, value=0.0)
    caps[(0, 0)][:] = 7.0
    cams, _ = reduce_attention(
        caps, layout(1), GEOM, patch=PATCH, denoise_step=0,
    )
    assert cams["cam0"].grid.max() == pytest.approx(7.0)


def test_layers_can_be_restricted_to_a_subset():
    caps = captures(1, steps=1, layers=4, value=0.0)
    caps[(0, 2)][:] = 3.0
    cams, _ = reduce_attention(
        caps, layout(1), GEOM, patch=PATCH, layers=[2],
    )
    assert cams["cam0"].grid.max() == pytest.approx(3.0)


def test_a_hot_patch_lands_at_the_right_grid_cell():
    # Token 5*16+3 of cam0 is grid row 5, col 3; after cropping 2 pad rows it
    # must appear at row 3.
    caps = captures(1, steps=1, layers=1, value=0.0)
    caps[(0, 0)][:, :, 5 * 16 + 3] = 9.0
    cams, _ = reduce_attention(caps, layout(1), GEOM, patch=PATCH)
    grid = cams["cam0"].grid
    assert np.unravel_index(int(grid.argmax()), grid.shape) == (3, 3)


def test_an_empty_capture_set_is_an_error_not_an_empty_map():
    with pytest.raises(ValueError, match="no attention"):
        reduce_attention({}, layout(1), GEOM, patch=PATCH)


def test_cameras_with_different_geometries_are_cropped_independently():
    # cam0 is 4:3 and padded top and bottom; cam1 is square and unpadded. Each
    # camera's crop must come from ITS OWN geometry, so the two grids differ in
    # height. One shared geometry would give both the same shape and silently
    # crop real content off the square view.
    geometries = {
        "cam0": LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224)),
        "cam1": LetterboxGeometry.from_shapes(src_hw=(480, 480), dst_hw=(224, 224)),
    }
    cams, _ = reduce_attention(captures(2), layout(2), geometries, patch=PATCH)
    assert cams["cam0"].grid.shape == (12, 16)
    assert cams["cam1"].grid.shape == (16, 16)


def test_a_missing_geometry_is_an_error_not_a_guess():
    with pytest.raises(KeyError):
        reduce_attention(captures(2), layout(2), {"cam0": LetterboxGeometry
                         .from_shapes(src_hw=(720, 960), dst_hw=(224, 224))},
                         patch=PATCH)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_reduce.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grabette_attention.reduce'`

- [ ] **Step 3: Write the implementation**

```python
# packages/grabette-attention/grabette_attention/reduce.py
"""Turn captured attention tensors into one map and one number per camera.

Reduction order, following the recipe every 2026 VLA-diagnosis paper uses: mean
over heads, mean over the action-token queries, then mean over the selected
layers and denoising steps. Queries are the action tokens; keys are the prefix
followed by the action tokens themselves.

Mass is a fraction of the PREFIX (cameras plus language), so the numbers sum to
one and a camera the policy barely uses reads as a small number rather than
being renormalised into looking important.
"""

from typing import Mapping, Sequence

import numpy as np

from .layout import LetterboxGeometry, TokenLayout
from .records import CameraAttention


def _select(
    captures: dict[tuple[int, int], np.ndarray],
    denoise_step: str | int,
    layers: str | Sequence[int],
) -> list[np.ndarray]:
    if not captures:
        raise ValueError("no attention was captured; were the hooks installed?")

    steps = sorted({s for s, _ in captures})
    if denoise_step == "last":
        wanted_steps = {steps[-1]}
    elif denoise_step == "first":
        wanted_steps = {steps[0]}
    elif denoise_step in ("mean", "all"):
        wanted_steps = set(steps)
    elif isinstance(denoise_step, int):
        wanted_steps = {denoise_step}
    else:
        raise ValueError(f"unknown denoise_step {denoise_step!r}")

    if layers in ("all", "mean"):
        wanted_layers = {l for _, l in captures}
    else:
        wanted_layers = set(layers)

    chosen = [
        w for (s, l), w in captures.items()
        if s in wanted_steps and l in wanted_layers
    ]
    if not chosen:
        raise ValueError(
            f"no attention matched denoise_step={denoise_step!r} layers={layers!r}"
        )
    return chosen


def reduce_attention(
    captures: dict[tuple[int, int], np.ndarray],
    layout: TokenLayout,
    geometries: Mapping[str, LetterboxGeometry],
    *,
    patch: int,
    denoise_step: str | int = "last",
    layers: str | Sequence[int] = "all",
) -> tuple[dict[str, CameraAttention], float]:
    """Reduce captures to per-camera grids plus each camera's prefix mass.

    `geometries` holds ONE ENTRY PER VISIBLE CAMERA, keyed by camera name. Each
    camera's padding crop comes from its own frame, because views may differ in
    resolution and aspect ratio; a camera with no entry is a KeyError rather
    than a guess.

    Returns (per-camera attention keyed by camera name, language mass).
    """
    chosen = _select(captures, denoise_step, layers)

    # (heads, queries, keys) each -> mean over layers/steps, heads, then queries.
    stacked = np.stack(chosen, axis=0).mean(axis=0)      # (heads, queries, keys)
    over_keys = stacked.mean(axis=(0, 1))                # (keys,)

    prefix = over_keys[: layout.prefix_len]
    total = float(prefix.sum())
    if total <= 0.0:
        raise ValueError("the prefix received no attention mass")

    cameras: dict[str, CameraAttention] = {}
    for key in layout.visible_cameras():
        block = prefix[layout.camera_slice(key)]
        grid = block.reshape(layout.grid_rows, layout.grid_cols)
        geometry = geometries[key]          # KeyError if a camera was forgotten
        row0, row1 = geometry.content_rows(patch)
        col0, col1 = geometry.content_cols(patch)
        cameras[key] = CameraAttention(
            grid=np.ascontiguousarray(grid[row0:row1, col0:col1], dtype=np.float32),
            mass=float(block.sum()) / total,
        )

    language_mass = float(prefix[layout.language_slice()].sum()) / total
    return cameras, language_mass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_reduce.py -v`
Expected: PASS, 12 tests

- [ ] **Step 5: Commit**

```bash
git add packages/grabette-attention
git commit -m "feat(attention): reduce captures to per-camera maps and prefix mass"
```

---

### Task 6: The ablation delta metric

**Files:**
- Create: `packages/grabette-attention/grabette_attention/metrics.py`
- Test: `packages/grabette-attention/tests/test_metrics.py`

**Interfaces:**
- Consumes: `ViewAblation` from `records.py` (Task 1).
- Produces: `translation_delta(baseline: np.ndarray, ablated: np.ndarray, *, n_translation: int = 3) -> ViewAblation`.

- [ ] **Step 1: Write the failing test**

```python
# packages/grabette-attention/tests/test_metrics.py
"""The ablation metric, and its units.

We have already shipped a metres-labelled-as-millimetres bug in the offline
gate, and then over-corrected it by scaling twice. So the unit is pinned by a
test with a hand-computed answer, not by a comment.
"""
import numpy as np
import pytest

from grabette_attention.metrics import translation_delta


def test_a_one_millimetre_shift_reads_as_one_millimetre():
    baseline = np.zeros((4, 11), np.float32)
    ablated = baseline.copy()
    ablated[:, 0] = 0.001                     # 1 mm on x, every step, in METRES
    result = translation_delta(baseline, ablated)
    assert result.delta_mm == pytest.approx(1.0)
    assert result.per_axis_mm[0] == pytest.approx(1.0)


def test_an_identical_chunk_gives_exactly_zero():
    chunk = np.random.default_rng(0).normal(size=(50, 11)).astype(np.float32)
    result = translation_delta(chunk, chunk.copy())
    assert result.delta_mm == 0.0
    assert result.per_axis_mm == (0.0, 0.0, 0.0)


def test_the_per_axis_breakdown_isolates_the_vertical_component():
    # The question we care about: does removing a view change HEIGHT?
    baseline = np.zeros((10, 11), np.float32)
    ablated = baseline.copy()
    ablated[:, 2] = 0.005                     # 5 mm on z only
    result = translation_delta(baseline, ablated)
    assert result.per_axis_mm == pytest.approx((0.0, 0.0, 5.0))
    assert result.delta_mm == pytest.approx(5.0)


def test_the_norm_combines_axes_in_quadrature():
    baseline = np.zeros((3, 11), np.float32)
    ablated = baseline.copy()
    ablated[:, 0] = 0.003
    ablated[:, 1] = 0.004                     # 3-4-5 triangle
    assert translation_delta(baseline, ablated).delta_mm == pytest.approx(5.0)


def test_it_is_an_rms_over_chunk_steps_not_a_sum():
    # A difference on half the steps must not scale with chunk length.
    baseline = np.zeros((4, 11), np.float32)
    ablated = baseline.copy()
    ablated[:2, 0] = 0.002
    # RMS over 4 steps of (2, 2, 0, 0) mm = sqrt((4+4)/4) = sqrt(2).
    assert translation_delta(baseline, ablated).delta_mm == pytest.approx(
        np.sqrt(2.0)
    )


def test_an_eight_dimensional_chunk_relative_action_works_too():
    # Chunk-relative checkpoints emit 8 dims, not 11. The first three are still
    # translation, so the metric must not assume a width.
    baseline = np.zeros((5, 8), np.float32)
    ablated = baseline.copy()
    ablated[:, 1] = 0.002
    assert translation_delta(baseline, ablated).per_axis_mm == pytest.approx(
        (0.0, 2.0, 0.0)
    )


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        translation_delta(np.zeros((4, 11)), np.zeros((5, 11)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grabette_attention.metrics'`

- [ ] **Step 3: Write the implementation**

```python
# packages/grabette-attention/grabette_attention/metrics.py
"""How much did removing a camera change the commanded motion?

The chunk's translation channels are in METRES; everything reported here is in
MILLIMETRES, converted exactly once. `delta_mm` is the RMS over chunk steps of
the 3-D difference, so it does not grow with the chunk length. `per_axis_mm` is
the same RMS per axis, which is the figure that says whether a view carries the
vertical (range) information.

For a chunk-relative checkpoint the chunk holds offsets rather than per-step
deltas; the metric still measures how far the two predictions diverge, which is
what the ablation asks.
"""

import numpy as np

from .records import ViewAblation

_METRES_TO_MM = 1000.0


def translation_delta(
    baseline: np.ndarray, ablated: np.ndarray, *, n_translation: int = 3
) -> ViewAblation:
    """RMS difference of the translation channels, in millimetres."""
    if baseline.shape != ablated.shape:
        raise ValueError(
            f"shape mismatch: baseline {baseline.shape} vs ablated {ablated.shape}"
        )
    diff = np.asarray(ablated[:, :n_translation], dtype=np.float64) - np.asarray(
        baseline[:, :n_translation], dtype=np.float64
    )
    norm = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    per_axis = np.sqrt(np.mean(diff**2, axis=0))
    return ViewAblation(
        delta_mm=norm * _METRES_TO_MM,
        per_axis_mm=tuple(float(a * _METRES_TO_MM) for a in per_axis),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_metrics.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add packages/grabette-attention
git commit -m "feat(attention): translation delta metric in millimetres"
```

---

### Task 7: The pi0.5 adapter

**Files:**
- Create: `packages/grabette-attention/grabette_attention/adapters/__init__.py`
- Create: `packages/grabette-attention/grabette_attention/adapters/base.py`
- Create: `packages/grabette-attention/grabette_attention/adapters/pi05.py`
- Test: `packages/grabette-attention/tests/test_pi05_adapter.py`

**Interfaces:**
- Consumes: `TokenLayout`, `LetterboxGeometry` (Tasks 2-3), `AttentionCapture` (Task 4), `FrameObservation` (Task 1).
- Produces: `RunResult(chunk: np.ndarray, captures: dict[tuple[int, int], np.ndarray])`; `PolicyAdapter` protocol with `camera_keys`, `patch`, `layout(obs)`, `draw_noise()`, `run(obs, noise, *, capture, drop_camera=None)`; `Pi05Adapter(policy, preprocessor, *, device, seed=0)`.

One forward pass yields both the chunk and the attention, so the baseline pass is not paid for twice.

- [ ] **Step 1: Write the failing test**

```python
# packages/grabette-attention/tests/test_pi05_adapter.py
"""The pi0.5 adapter, against a stub policy.

The stub reproduces the three things the real policy does that the adapter
depends on: `_preprocess_images` returns (images, masks) with absent cameras
appended and masked; `sample_actions` accepts an explicit `noise`; and the
attention modules return `(output, weights)`. No GPU, no checkpoint.
"""
import numpy as np
import pytest
import torch
from torch import nn

from grabette_attention.adapters.pi05 import Pi05Adapter
from grabette_attention.records import FrameObservation


class StubAttention(nn.Module):
    def forward(self, x):
        return x, torch.rand(1, 8, 50, 506)


class StubDecoderLayer(nn.Module):
    """Mirrors a Gemma decoder layer: the attention hangs off `self_attn`."""

    def __init__(self):
        super().__init__()
        self.self_attn = StubAttention()


class StubExpertInner(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(StubDecoderLayer() for _ in range(n_layers))


class StubExpert(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.model = StubExpertInner(n_layers)


class StubTwoTower(nn.Module):
    """Mirrors `paligemma_with_expert`: the adapter resolves this exact path."""

    def __init__(self, n_layers):
        super().__init__()
        self.gemma_expert = StubExpert(n_layers)


class StubConfig:
    def __init__(self, cameras):
        self.image_features = {c: None for c in cameras}
        self.image_resolution = (224, 224)
        self.chunk_size = 50
        self.max_action_dim = 32
        self.tokenizer_max_length = 200
        self.compile_model = False


class StubModel(nn.Module):
    """Its action depends on the SUM of the unmasked images, so masking a
    camera provably changes the output — that is what the ablation must see."""

    def __init__(self, n_layers=18):
        super().__init__()
        self.paligemma_with_expert = StubTwoTower(n_layers)
        self.steps = 10

    def sample_actions(self, images, img_masks, tokens, masks, noise=None):
        for _ in range(self.steps):
            for layer in self.paligemma_with_expert.gemma_expert.model.layers:
                layer.self_attn(torch.zeros(1))
        signal = sum(
            float(img.mean()) * float(m.float().mean())
            for img, m in zip(images, img_masks)
        )
        chunk = noise[:, :, :11] * 0.0 + signal * 0.001
        return chunk


class StubPolicy(nn.Module):
    def __init__(self, cameras):
        super().__init__()
        self.config = StubConfig(cameras)
        self.model = StubModel()

    def _preprocess_images(self, batch):
        present = [k for k in self.config.image_features if k in batch]
        missing = [k for k in self.config.image_features if k not in batch]
        images, masks = [], []
        for key in present:
            images.append(batch[key])
            masks.append(torch.ones(1, dtype=torch.bool))
        for _ in missing:
            images.append(torch.ones_like(images[-1]) * -1)
            masks.append(torch.zeros(1, dtype=torch.bool))
        return images, masks


def observation(cameras) -> FrameObservation:
    rng = np.random.default_rng(0)
    return FrameObservation(
        episode=0, frame=0,
        images={c: rng.integers(0, 255, (720, 960, 3), dtype=np.uint8) for c in cameras},
        state=np.zeros(2, np.float32), task="pick the sugar cube",
    )


def make_adapter(cameras):
    policy = StubPolicy(cameras)

    def preprocessor(batch):
        batch = dict(batch)
        batch["observation.language.tokens"] = torch.zeros(1, 200, dtype=torch.long)
        batch["observation.language.attention_mask"] = torch.ones(1, 200, dtype=torch.long)
        return batch

    return Pi05Adapter(policy, preprocessor, device="cpu", seed=0)


def test_camera_keys_come_from_the_config_in_order():
    adapter = make_adapter(["cam_a", "cam_b"])
    assert adapter.camera_keys == ("cam_a", "cam_b")


def test_the_layout_is_derived_from_the_model_not_hard_coded():
    adapter = make_adapter(["cam_a", "cam_b"])
    layout = adapter.layout(observation(["cam_a", "cam_b"]))
    assert layout.tokens_per_image == 256      # (224/14)^2, computed
    assert layout.grid_rows == layout.grid_cols == 16
    assert layout.prefix_len == 2 * 256 + 200


def test_an_absent_camera_is_marked_masked_and_keeps_its_slot():
    adapter = make_adapter(["cam_a", "cam_b"])
    layout = adapter.layout(observation(["cam_a"]))     # cam_b not recorded
    assert layout.masked_cameras == frozenset({"cam_b"})
    assert layout.visible_cameras() == ("cam_a",)
    assert layout.camera_slice("cam_b") == slice(256, 512)


def test_one_pass_returns_both_the_chunk_and_the_attention():
    adapter = make_adapter(["cam_a"])
    result = adapter.run(observation(["cam_a"]), adapter.draw_noise(), capture=True)
    assert result.chunk.shape == (50, 11)
    assert len(result.captures) == 180          # 10 steps x 18 layers


def test_capture_can_be_switched_off_for_ablation_passes():
    adapter = make_adapter(["cam_a"])
    result = adapter.run(observation(["cam_a"]), adapter.draw_noise(), capture=False)
    assert result.captures == {}


def test_the_same_noise_gives_a_bitwise_identical_chunk():
    adapter = make_adapter(["cam_a"])
    obs, noise = observation(["cam_a"]), adapter.draw_noise()
    first = adapter.run(obs, noise, capture=False).chunk
    second = adapter.run(obs, noise, capture=False).chunk
    np.testing.assert_array_equal(first, second)


def test_draw_noise_is_deterministic_for_a_given_seed():
    a = Pi05Adapter(StubPolicy(["cam_a"]), lambda b: b, device="cpu", seed=7)
    b = Pi05Adapter(StubPolicy(["cam_a"]), lambda b: b, device="cpu", seed=7)
    np.testing.assert_array_equal(a.draw_noise().numpy(), b.draw_noise().numpy())


def test_dropping_a_camera_changes_the_chunk():
    adapter = make_adapter(["cam_a", "cam_b"])
    obs, noise = observation(["cam_a", "cam_b"]), adapter.draw_noise()
    base = adapter.run(obs, noise, capture=False).chunk
    dropped = adapter.run(obs, noise, capture=False, drop_camera="cam_b").chunk
    assert not np.allclose(base, dropped)


def test_dropping_a_camera_preserves_the_token_order():
    # The mask is zeroed IN PLACE; the camera keeps its slot, so a later camera
    # does not shift. Asserted through the layout the adapter reports.
    adapter = make_adapter(["cam_a", "cam_b"])
    obs = observation(["cam_a", "cam_b"])
    layout = adapter.layout(obs, drop_camera="cam_a")
    assert layout.camera_keys == ("cam_a", "cam_b")
    assert layout.camera_slice("cam_b") == slice(256, 512)
    assert layout.masked_cameras == frozenset({"cam_a"})


def test_dropping_an_unknown_camera_is_an_error():
    adapter = make_adapter(["cam_a"])
    with pytest.raises(KeyError):
        adapter.run(observation(["cam_a"]), adapter.draw_noise(),
                    capture=False, drop_camera="wrist")


def test_the_patch_size_is_read_from_the_vision_config_when_reachable():
    # A real checkpoint exposes it; the adapter must prefer the model's own
    # value over any constant, so a differently-configured SigLIP still works.
    policy = StubPolicy(["cam_a"])
    tower = type("Tower", (), {})()
    tower.config = type("VisionCfg", (), {"patch_size": 16})()
    inner = type("Inner", (), {})()
    inner.vision_tower = tower
    policy.model.paligemma_with_expert.paligemma = type("PG", (), {})()
    policy.model.paligemma_with_expert.paligemma.model = inner
    adapter = Pi05Adapter(policy, lambda b: b, device="cpu", seed=0)
    assert adapter.patch == 16


def test_the_patch_size_falls_back_to_the_siglip_default():
    # SigLIP-so400m as PaliGemma configures it uses patch 14. This fallback is
    # the documented default, not test scaffolding.
    adapter = make_adapter(["cam_a"])
    assert adapter.patch == 14


def test_a_policy_without_the_pi05_module_tree_is_rejected_clearly():
    class NotPi05(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = StubConfig(["cam_a"])
            self.model = nn.Module()

    with pytest.raises(AttributeError, match="paligemma_with_expert"):
        Pi05Adapter(NotPi05(), lambda b: b, device="cpu", seed=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_pi05_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grabette_attention.adapters'`

- [ ] **Step 3: Write the base protocol**

```python
# packages/grabette-attention/grabette_attention/adapters/__init__.py
"""Policy adapters: every model-specific fact lives behind this boundary."""

from .base import PolicyAdapter, RunResult

__all__ = ["PolicyAdapter", "RunResult"]
```

```python
# packages/grabette-attention/grabette_attention/adapters/base.py
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
```

- [ ] **Step 4: Write the pi0.5 adapter**

```python
# packages/grabette-attention/grabette_attention/adapters/pi05.py
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

# SigLIP-so400m as PaliGemma configures it. Read from the model where possible;
# this is the fallback when the vision config is not reachable.
_DEFAULT_PATCH = 14


class Pi05Adapter:
    """Wraps a loaded pi0.5 policy and its checkpoint preprocessor."""

    def __init__(
        self,
        policy: Any,
        preprocessor: Callable[[dict], dict],
        *,
        device: str = "cpu",
        seed: int = 0,
    ):
        self._policy = policy
        self._preprocessor = preprocessor
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
        vision = getattr(
            getattr(getattr(self._policy, "model", None), "paligemma_with_expert", None),
            "paligemma", None,
        )
        try:
            return int(vision.model.vision_tower.config.patch_size)
        except AttributeError:
            return _DEFAULT_PATCH

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
            return RunResult(chunk=self._to_numpy(chunk))

        with AttentionCapture(self._expert_layers) as cap, torch.no_grad():
            chunk = self._policy.model.sample_actions(
                images, img_masks, tokens, masks, noise=noise
            )
            captures = dict(cap.captures)
        return RunResult(chunk=self._to_numpy(chunk), captures=captures)

    # ---- helpers -------------------------------------------------------------

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_pi05_adapter.py -v`
Expected: PASS, 13 tests

- [ ] **Step 6: Commit**

```bash
git add packages/grabette-attention
git commit -m "feat(attention): pi0.5 adapter with hooked attention and in-place mask ablation"
```

---

### Task 8: Frame sources and frame selection

**Files:**
- Create: `packages/grabette-attention/grabette_attention/sources.py`
- Test: `packages/grabette-attention/tests/test_sources.py`

**Interfaces:**
- Consumes: `FrameObservation` (Task 1).
- Produces: `DumpObsSource(directory, *, task, camera_key)` and `DatasetSource(repo_id, *, episodes, camera_keys, task, root=None)`, both with `frames() -> Iterator[FrameObservation]`; `select_frames(gripper: np.ndarray | None, n_frames: int, *, mode, threshold=0.5, count=1) -> tuple[list[int], str]`.

- [ ] **Step 1: Write the failing test**

```python
# packages/grabette-attention/tests/test_sources.py
"""Frame sources and frame selection.

The dump_obs reader is tested against files written the way evaluate.py writes
them (cv2.imwrite, so BGR on disk). The dataset source is exercised in the
GPU integration test; here we test the selection logic that both share.
"""
import json

import numpy as np
import pytest

from grabette_attention.sources import DumpObsSource, select_frames

cv2 = pytest.importorskip("cv2")


def write_dump(tmp_path, n=3):
    for i in range(n):
        # A frame that is unambiguously RED in RGB terms.
        rgb = np.zeros((8, 12, 3), np.uint8)
        rgb[:, :, 0] = 255
        cv2.imwrite(str(tmp_path / f"obs_{i:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    lines = [json.dumps({"step": i, "state": [0.1 * i, 0.2]}) for i in range(n)]
    (tmp_path / "state.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_dump_obs_frames_come_back_as_rgb():
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    write_dump(tmp)
    source = DumpObsSource(tmp, task="pick the sugar cube", camera_key="observation.images.cam0")
    frames = list(source.frames())
    assert len(frames) == 3
    first = frames[0].images["observation.images.cam0"]
    # Red in RGB means channel 0 is hot. If the BGR conversion were skipped,
    # channel 2 would be hot instead.
    assert first[0, 0, 0] == 255
    assert first[0, 0, 2] == 0


def test_dump_obs_pairs_each_frame_with_its_state():
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    write_dump(tmp)
    source = DumpObsSource(tmp, task="t", camera_key="cam0")
    frames = list(source.frames())
    assert frames[2].state[0] == pytest.approx(0.2)
    assert frames[1].frame == 1


def test_grasp_selection_finds_the_first_closure_crossing():
    gripper = np.array([0.0, 0.0, 0.1, 0.6, 0.9, 0.9, 0.2], np.float32)
    indices, note = select_frames(gripper, n_frames=7, mode="grasp")
    assert indices == [3]
    assert "closure" in note


def test_grasp_selection_can_return_a_window_around_the_grasp():
    gripper = np.array([0.0, 0.0, 0.1, 0.6, 0.9], np.float32)
    indices, _ = select_frames(gripper, n_frames=5, mode="grasp", count=3)
    assert indices == [2, 3, 4]


def test_grasp_selection_falls_back_to_a_stride_and_says_so():
    indices, note = select_frames(None, n_frames=10, mode="grasp", count=3)
    assert len(indices) == 3
    assert "stride" in note and "no gripper" in note


def test_a_never_closing_episode_falls_back_to_a_stride():
    gripper = np.zeros(10, np.float32)
    indices, note = select_frames(gripper, n_frames=10, mode="grasp", count=2)
    assert len(indices) == 2
    assert "never" in note


def test_stride_mode_spreads_frames_over_the_episode():
    indices, _ = select_frames(None, n_frames=10, mode="stride", count=5)
    assert indices == [0, 2, 4, 6, 8]


def test_explicit_indices_are_returned_untouched():
    indices, note = select_frames(None, n_frames=10, mode=[2, 7])
    assert indices == [2, 7]
    assert "explicit" in note


def test_an_out_of_range_explicit_index_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        select_frames(None, n_frames=5, mode=[9])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grabette_attention.sources'`

- [ ] **Step 3: Write the implementation**

```python
# packages/grabette-attention/grabette_attention/sources.py
"""Where frames come from: a LeRobot dataset, or a `--dump_obs` capture.

Both yield `FrameObservation` at full resolution and un-normalised, which is the
property that makes them interchangeable: the policy's own preprocessing does
the letterboxing either way.
"""

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .records import FrameObservation


class DumpObsSource:
    """Reads one episode directory written by `evaluate.py --dump_obs`.

    That writer uses `cv2.imwrite`, so the PNGs are BGR on disk and must be
    converted back to RGB here. It records one camera; the key it belongs to is
    supplied by the caller because the PNG carries no name.
    """

    def __init__(self, directory: Path | str, *, task: str, camera_key: str):
        self._dir = Path(directory)
        self._task = task
        self._camera_key = camera_key

    def _states(self) -> dict[int, np.ndarray]:
        path = self._dir / "state.jsonl"
        if not path.exists():
            return {}
        states = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            states[int(record["step"])] = np.asarray(
                record["state"], dtype=np.float32
            )
        return states

    def frames(self) -> Iterator[FrameObservation]:
        import cv2

        states = self._states()
        for path in sorted(self._dir.glob("obs_*.png")):
            index = int(path.stem.split("_")[1])
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"could not read {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield FrameObservation(
                episode=0,
                frame=index,
                images={self._camera_key: rgb},
                state=states.get(index, np.zeros(0, np.float32)),
                task=self._task,
            )


class DatasetSource:
    """Reads held-out frames from a LeRobot dataset.

    Follows the offline loading already used by
    `integrations/Pi05/smoke_generation.py`: one episode at a time, frames
    indexed directly. Camera keys come from the CHECKPOINT, not the dataset, so
    a dataset missing a view is handled by the adapter's masking rather than by
    guessing here.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        episodes: Sequence[int],
        camera_keys: Sequence[str],
        task: str,
        root: Path | str | None = None,
        selection: str | Sequence[int] = "grasp",
        count: int = 1,
        gripper_channel: int = -1,
    ):
        self._repo_id = repo_id
        self._episodes = list(episodes)
        self._camera_keys = list(camera_keys)
        self._task = task
        self._root = root
        self._selection = selection
        self._count = count
        self._gripper_channel = gripper_channel
        self.notes: dict[int, str] = {}

    def frames(self) -> Iterator[FrameObservation]:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        for episode in self._episodes:
            dataset = LeRobotDataset(
                self._repo_id, root=self._root, episodes=[episode]
            )
            n_frames = len(dataset)
            gripper = self._gripper_track(dataset, n_frames)
            indices, note = select_frames(
                gripper, n_frames, mode=self._selection, count=self._count
            )
            self.notes[episode] = note
            for index in indices:
                item = dataset[index]
                images = {}
                for key in self._camera_keys:
                    if key not in item:
                        continue          # absent view: the adapter masks it
                    chw = np.asarray(item[key], dtype=np.float32)
                    images[key] = (
                        np.clip(chw.transpose(1, 2, 0) * 255.0, 0, 255)
                        .astype(np.uint8)
                    )
                yield FrameObservation(
                    episode=episode,
                    frame=index,
                    images=images,
                    state=np.asarray(item["observation.state"], dtype=np.float32),
                    task=self._task,
                )

    def _gripper_track(self, dataset, n_frames: int) -> np.ndarray | None:
        """The gripper channel over the episode, for grasp-frame selection."""
        try:
            return np.asarray(
                [
                    float(np.asarray(dataset[i]["action"])[self._gripper_channel])
                    for i in range(n_frames)
                ],
                dtype=np.float32,
            )
        except (KeyError, IndexError, TypeError):
            return None


def select_frames(
    gripper: np.ndarray | None,
    n_frames: int,
    *,
    mode: str | Sequence[int],
    threshold: float = 0.5,
    count: int = 1,
) -> tuple[list[int], str]:
    """Choose which frames to analyse, and say how the choice was made.

    `grasp` is the default because the literature reports the failure signal
    concentrating at the grasp. When there is no gripper channel, or the episode
    never closes, this falls back to an even stride and RETURNS THAT FACT so the
    summary can print it instead of silently analysing the wrong frames.
    """
    if not isinstance(mode, str):
        indices = [int(i) for i in mode]
        for index in indices:
            if index < 0 or index >= n_frames:
                raise ValueError(
                    f"frame {index} out of range for an episode of {n_frames}"
                )
        return indices, "explicit indices"

    if mode == "stride":
        return _stride(n_frames, count), f"even stride of {count}"

    if mode != "grasp":
        raise ValueError(f"unknown frame selection {mode!r}")

    if gripper is None:
        return (
            _stride(n_frames, count),
            f"no gripper channel found; even stride of {count}",
        )
    closed = np.flatnonzero(np.asarray(gripper) >= threshold)
    if closed.size == 0:
        return (
            _stride(n_frames, count),
            f"gripper never closes (threshold {threshold}); even stride of {count}",
        )

    first = int(closed[0])
    half = count // 2
    start = max(0, min(first - half, n_frames - count))
    indices = list(range(start, min(start + count, n_frames)))
    return indices, f"gripper closure at frame {first} (threshold {threshold})"


def _stride(n_frames: int, count: int) -> list[int]:
    if count <= 0 or n_frames <= 0:
        return []
    step = max(1, n_frames // count)
    return [i * step for i in range(count) if i * step < n_frames]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_sources.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add packages/grabette-attention
git commit -m "feat(attention): dataset and dump_obs frame sources with grasp-frame selection"
```

---

### Task 9: Analysis orchestration

**Files:**
- Create: `packages/grabette-attention/grabette_attention/analysis.py`
- Test: `packages/grabette-attention/tests/test_analysis.py`

**Interfaces:**
- Consumes: `PolicyAdapter`, `RunResult` (Task 7), `reduce_attention` (Task 5), `translation_delta` (Task 6), records (Task 1).
- Produces: `analyse_frame(adapter, obs, *, denoise_step="last", layers="all", ablate=True, provenance=None) -> FrameAnalysis` and `analyse(adapter, frames, **kwargs) -> Iterator[FrameAnalysis]`.

- [ ] **Step 1: Write the failing test**

```python
# packages/grabette-attention/tests/test_analysis.py
"""Orchestration: one capture pass, then one ablation pass per camera.

Tested against a fake adapter, so this file is about wiring and pass-counting
rather than about pi0.5. The pass count matters: the baseline must be reused,
not recomputed per camera.
"""
import numpy as np
import pytest

from grabette_attention.analysis import analyse, analyse_frame
from grabette_attention.adapters.base import RunResult
from grabette_attention.layout import LetterboxGeometry, TokenLayout
from grabette_attention.records import FrameObservation


class FakeAdapter:
    """Two cameras; only cam0 influences the chunk."""

    patch = 14

    def __init__(self, cameras=("cam0", "cam1")):
        self._cameras = cameras
        self.calls: list[tuple[bool, str | None]] = []

    @property
    def camera_keys(self):
        return tuple(self._cameras)

    def geometry(self, obs, camera):
        frame = obs.images[camera]
        return LetterboxGeometry.from_shapes(
            src_hw=(frame.shape[0], frame.shape[1]), dst_hw=(224, 224)
        )

    def layout(self, obs, *, drop_camera=None):
        masked = {drop_camera} if drop_camera else set()
        return TokenLayout(
            camera_keys=tuple(self._cameras), tokens_per_image=256,
            grid_rows=16, grid_cols=16, language_tokens=200,
            masked_cameras=frozenset(masked),
        )

    def draw_noise(self):
        return "fixed-noise"

    def run(self, obs, noise, *, capture, drop_camera=None):
        assert noise == "fixed-noise"          # the same noise every pass
        self.calls.append((capture, drop_camera))
        chunk = np.zeros((50, 11), np.float32)
        if drop_camera == "cam0":
            chunk[:, 2] = 0.004                # 4 mm on z when cam0 is removed
        captures = {}
        if capture:
            keys = 2 * 256 + 200 + 50
            captures = {(0, 0): np.ones((8, 50, keys), np.float32)}
        return RunResult(chunk=chunk, captures=captures)


def observation():
    return FrameObservation(
        episode=3, frame=42,
        images={c: np.zeros((720, 960, 3), np.uint8) for c in ("cam0", "cam1")},
        state=np.zeros(2, np.float32), task="pick the sugar cube",
    )


def test_it_produces_one_map_and_one_ablation_per_camera():
    adapter = FakeAdapter()
    result = analyse_frame(adapter, observation())
    assert set(result.cameras) == {"cam0", "cam1"}
    assert set(result.ablations) == {"cam0", "cam1"}


def test_the_baseline_runs_once_and_each_ablation_once():
    adapter = FakeAdapter()
    analyse_frame(adapter, observation())
    assert adapter.calls == [
        (True, None), (False, "cam0"), (False, "cam1"),
    ]


def test_the_ablation_delta_is_reported_in_millimetres_per_axis():
    result = analyse_frame(FakeAdapter(), observation())
    assert result.ablations["cam0"].delta_mm == pytest.approx(4.0)
    assert result.ablations["cam0"].per_axis_mm == pytest.approx((0.0, 0.0, 4.0))
    assert result.ablations["cam1"].delta_mm == pytest.approx(0.0)


def test_ablation_can_be_switched_off():
    adapter = FakeAdapter()
    result = analyse_frame(adapter, observation(), ablate=False)
    assert result.ablations == {}
    assert adapter.calls == [(True, None)]


def test_provenance_records_how_the_map_was_made():
    result = analyse_frame(
        FakeAdapter(), observation(),
        denoise_step="last", layers="all",
        provenance={"checkpoint": "user/model_best"},
    )
    assert result.provenance["checkpoint"] == "user/model_best"
    assert result.provenance["denoise_step"] == "last"
    assert result.provenance["layers"] == "all"


def test_the_episode_and_frame_are_carried_through():
    result = analyse_frame(FakeAdapter(), observation())
    assert (result.episode, result.frame) == (3, 42)


def test_analyse_streams_over_many_frames():
    adapter = FakeAdapter()
    results = list(analyse(adapter, [observation(), observation()]))
    assert len(results) == 2


def test_a_single_camera_policy_needs_no_special_case():
    adapter = FakeAdapter(cameras=("cam0",))
    obs = FrameObservation(
        episode=0, frame=0,
        images={"cam0": np.zeros((720, 960, 3), np.uint8)},
        state=np.zeros(2, np.float32), task="t",
    )
    result = analyse_frame(adapter, obs)
    assert set(result.cameras) == {"cam0"}
    assert result.cameras["cam0"].mass == pytest.approx(256 / 456)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grabette_attention.analysis'`

- [ ] **Step 3: Write the implementation**

```python
# packages/grabette-attention/grabette_attention/analysis.py
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
from .reduce import reduce_attention


def analyse_frame(
    adapter: PolicyAdapter,
    obs: FrameObservation,
    *,
    denoise_step: str | int = "last",
    layers: str | Sequence[int] = "all",
    ablate: bool = True,
    provenance: dict[str, str] | None = None,
) -> FrameAnalysis:
    layout = adapter.layout(obs)
    baseline = adapter.run(obs, adapter.draw_noise(), capture=True)

    # Geometry is per camera: views may differ in resolution or aspect ratio, so
    # each camera's padding crop must come from its own frame.
    visible = layout.visible_cameras()
    if not visible:
        raise ValueError(f"frame {obs.frame} has no usable camera")
    geometries = {camera: adapter.geometry(obs, camera) for camera in visible}

    cameras, language_mass = reduce_attention(
        baseline.captures,
        layout,
        geometries,
        patch=adapter.patch,
        denoise_step=denoise_step,
        layers=layers,
    )

    ablations = {}
    if ablate:
        for camera in visible:
            dropped = adapter.run(
                obs, adapter.draw_noise(), capture=False, drop_camera=camera
            )
            ablations[camera] = translation_delta(baseline.chunk, dropped.chunk)

    record = dict(provenance or {})
    record.setdefault("denoise_step", str(denoise_step))
    record.setdefault("layers", str(layers))
    return FrameAnalysis(
        episode=obs.episode,
        frame=obs.frame,
        cameras=cameras,
        language_mass=language_mass,
        ablations=ablations,
        provenance=record,
    )


def analyse(
    adapter: PolicyAdapter,
    frames: Iterable[FrameObservation],
    **kwargs,
) -> Iterator[FrameAnalysis]:
    """Stream over frames, so a long episode does not have to fit in memory."""
    for obs in frames:
        yield analyse_frame(adapter, obs, **kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_analysis.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add packages/grabette-attention
git commit -m "feat(attention): analysis orchestration with a shared noise tensor"
```

---

### Task 10: PNG front end

**Files:**
- Create: `packages/grabette-attention/grabette_attention/frontends/__init__.py`
- Create: `packages/grabette-attention/grabette_attention/frontends/png.py`
- Test: `packages/grabette-attention/tests/test_png.py`

**Interfaces:**
- Consumes: `FrameAnalysis`, `FrameObservation` (Task 1).
- Produces: `write_overlays(analysis, obs, out_dir) -> list[Path]` and `write_summary(analyses, out_dir, *, notes=None) -> Path`.

- [ ] **Step 1: Write the failing test**

```python
# packages/grabette-attention/tests/test_png.py
"""The PNG front end: one overlay per camera per frame, plus a summary.

Must work headless — this runs over ssh on the GPU box — so matplotlib is
forced onto Agg and nothing opens a window.
"""
import pathlib
import tempfile

import numpy as np
import pytest

from grabette_attention.frontends.png import write_overlays, write_summary
from grabette_attention.records import (
    CameraAttention,
    FrameAnalysis,
    FrameObservation,
    ViewAblation,
)

pytest.importorskip("matplotlib")


def analysis(cameras=("cam0", "cam1")) -> FrameAnalysis:
    return FrameAnalysis(
        episode=3, frame=42,
        cameras={
            c: CameraAttention(
                grid=np.linspace(0, 1, 12 * 16, dtype=np.float32).reshape(12, 16),
                mass=0.4,
            )
            for c in cameras
        },
        language_mass=0.2,
        ablations={c: ViewAblation(delta_mm=8.4, per_axis_mm=(1.0, 2.0, 8.1)) for c in cameras},
        provenance={"checkpoint": "user/model_best", "denoise_step": "last"},
    )


def observation(cameras=("cam0", "cam1")) -> FrameObservation:
    return FrameObservation(
        episode=3, frame=42,
        images={c: np.zeros((720, 960, 3), np.uint8) for c in cameras},
        state=np.zeros(2, np.float32), task="pick the sugar cube",
    )


def test_one_overlay_is_written_per_camera():
    out = pathlib.Path(tempfile.mkdtemp())
    paths = write_overlays(analysis(), observation(), out)
    assert len(paths) == 2
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_the_filename_carries_the_frame_and_the_camera_name():
    out = pathlib.Path(tempfile.mkdtemp())
    paths = write_overlays(analysis(), observation(), out)
    names = sorted(p.name for p in paths)
    assert names[0] == "frame_00042_cam0_attn.png"
    assert names[1] == "frame_00042_cam1_attn.png"


def test_a_three_camera_frame_writes_three_overlays():
    out = pathlib.Path(tempfile.mkdtemp())
    cams = ("cam0", "cam1", "wrist")
    paths = write_overlays(analysis(cams), observation(cams), out)
    assert len(paths) == 3


def test_the_summary_reports_mass_and_millimetres_per_camera():
    out = pathlib.Path(tempfile.mkdtemp())
    path = write_summary([analysis()], out)
    text = path.read_text()
    assert "cam0" in text and "cam1" in text
    assert "0.40" in text          # mass
    assert "8.4" in text           # ablation delta in mm
    assert "mm" in text


def test_the_summary_names_the_language_mass_separately():
    out = pathlib.Path(tempfile.mkdtemp())
    text = write_summary([analysis()], out).read_text()
    assert "language" in text
    assert "0.20" in text


def test_the_summary_carries_the_provenance():
    out = pathlib.Path(tempfile.mkdtemp())
    text = write_summary([analysis()], out).read_text()
    assert "user/model_best" in text
    assert "denoise_step" in text


def test_the_summary_repeats_the_frame_selection_note():
    out = pathlib.Path(tempfile.mkdtemp())
    text = write_summary(
        [analysis()], out, notes={3: "gripper never closes; even stride of 3"}
    ).read_text()
    assert "never closes" in text


def test_the_summary_warns_that_a_broad_map_is_normal():
    # The review's interpretation guard, in the artefact itself.
    out = pathlib.Path(tempfile.mkdtemp())
    text = write_summary([analysis()], out).read_text()
    assert "broad" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_png.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grabette_attention.frontends'`

- [ ] **Step 3: Write the implementation**

```python
# packages/grabette-attention/grabette_attention/frontends/__init__.py
"""Front ends. Both consume `FrameAnalysis`; neither computes anything."""
```

```python
# packages/grabette-attention/grabette_attention/frontends/png.py
"""Write attention overlays as PNGs and the numbers as a text summary.

Headless by construction: matplotlib is switched to Agg before pyplot is
imported, matching `integrations/DiffusionPolicy/offline_eval.py`. Frames are
RGB in memory throughout; nothing here writes BGR.
"""

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from ..records import FrameAnalysis, FrameObservation

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

    written = []
    for camera, attention in analysis.cameras.items():
        frame = obs.images[camera]
        short = camera.rsplit(".", 1)[-1]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_png.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add packages/grabette-attention
git commit -m "feat(attention): headless PNG overlays and text summary"
```

---

### Task 11: CLI, checkpoint loading, and wiring

**Files:**
- Create: `packages/grabette-attention/grabette_attention/loader.py`
- Create: `packages/grabette-attention/grabette_attention/cli.py`
- Create: `packages/grabette-attention/README.md`
- Modify: `packages/grabette-attention/pyproject.toml` (add the console script)
- Modify: `integrations/Pi05/pyproject.toml` (path dependency)
- Modify: `integrations/Pi05/README.md` (document the tool)
- Test: `packages/grabette-attention/tests/test_cli.py`

**Interfaces:**
- Consumes: `Pi05Adapter` (Task 7), sources (Task 8), `analyse` (Task 9), PNG front end (Task 10).
- Produces: `load_pi05(checkpoint, *, device, fp32=True) -> tuple[Any, Callable]`; `build_parser() -> argparse.ArgumentParser`; `main(argv=None) -> int`; console script `grabette-attn`.

- [ ] **Step 1: Write the failing test**

```python
# packages/grabette-attention/tests/test_cli.py
"""CLI argument handling. No checkpoint, no GPU — parsing and validation only."""
import pytest

from grabette_attention.cli import build_parser, main


def test_dataset_mode_parses_episodes_as_integers():
    args = build_parser().parse_args(
        ["--checkpoint", "user/m", "--dataset", "user/d", "--episodes", "3", "7", "11"]
    )
    assert args.episodes == [3, 7, 11]


def test_dump_obs_mode_parses_a_directory():
    args = build_parser().parse_args(
        ["--checkpoint", "user/m", "--dump-obs", "out/ep003"]
    )
    assert args.dump_obs == "out/ep003"
    assert args.dataset is None


def test_grasp_is_the_default_frame_selection():
    args = build_parser().parse_args(["--checkpoint", "user/m", "--dataset", "user/d"])
    assert args.frames == "grasp"


def test_the_last_denoising_step_is_the_default():
    args = build_parser().parse_args(["--checkpoint", "user/m", "--dataset", "user/d"])
    assert args.denoise_step == "last"


def test_fp32_is_the_default_because_the_bf16_flow_path_is_broken():
    args = build_parser().parse_args(["--checkpoint", "user/m", "--dataset", "user/d"])
    assert args.fp32 is True


def test_ablation_is_on_by_default_and_can_be_turned_off():
    parser = build_parser()
    assert parser.parse_args(["--checkpoint", "c", "--dataset", "d"]).ablate is True
    assert parser.parse_args(
        ["--checkpoint", "c", "--dataset", "d", "--no-ablation"]
    ).ablate is False


def test_explicit_frame_indices_are_accepted():
    args = build_parser().parse_args(
        ["--checkpoint", "c", "--dataset", "d", "--frames", "12", "40"]
    )
    assert args.frames == ["12", "40"]


def test_giving_neither_input_is_rejected():
    with pytest.raises(SystemExit):
        main(["--checkpoint", "user/m"])


def test_giving_both_inputs_is_rejected():
    with pytest.raises(SystemExit):
        main(["--checkpoint", "c", "--dataset", "d", "--dump-obs", "out/ep0"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grabette_attention.cli'`

- [ ] **Step 3: Write the loader**

```python
# packages/grabette-attention/grabette_attention/loader.py
"""Load a pi0.5 checkpoint the way the rest of the repo does.

Copied deliberately from `integrations/Pi05/smoke_generation.py` rather than
invented: CPU config first, `compile_model` off so forward hooks are not
swallowed by a graph, fp32 because the pi05 port has a bf16 clash in its flow
path, and camera keys taken from the CHECKPOINT rather than the dataset.
"""

from typing import Any, Callable

import torch


def load_pi05(
    checkpoint: str, *, device: str = "cuda", fp32: bool = True
) -> tuple[Any, Callable[[dict], dict]]:
    """Return (policy, preprocessor) ready for the adapter."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    # Chunk-relative checkpoints register their processor steps on import. Import
    # it if present so `make_pre_post_processors` can resolve them; a plain delta
    # checkpoint is unaffected.
    try:
        import grabette_chunkrel.chunk_relative_processor  # noqa: F401
    except ImportError:
        pass

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cpu"
    config.compile_model = False

    policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
    policy = policy.to(dtype=torch.float32 if fp32 else torch.bfloat16).eval()
    policy = policy.to(device)
    policy.config.device = device

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor
```

- [ ] **Step 4: Write the CLI**

```python
# packages/grabette-attention/grabette_attention/cli.py
"""`grabette-attn`: offline attention maps and view ablation.

Deliberately offline. The maps are a hypothesis and the ablation millimetres are
the measurement; neither belongs on the robot's control path.
"""

import argparse
import sys
from pathlib import Path

from .analysis import analyse
from .frontends.png import write_overlays, write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grabette-attn",
        description=(
            "Show where a policy's action tokens attend on each camera, and how "
            "much the commanded chunk changes when each camera is removed."
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="local path or Hub repo id")
    parser.add_argument("--dataset", default=None, help="LeRobot dataset repo id")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--dump-obs", default=None, help="an episode dir from evaluate.py --dump_obs")
    parser.add_argument(
        "--camera-key",
        default="observation.images.cam0",
        help="which camera the dump_obs PNGs belong to (they carry no name)",
    )
    parser.add_argument("--task", default="", help="language prompt; required for dump_obs")
    parser.add_argument(
        "--frames",
        nargs="*",
        default="grasp",
        help="'grasp' (default), 'stride', or explicit frame indices",
    )
    parser.add_argument("--count", type=int, default=1, help="frames per episode")
    parser.add_argument("--denoise-step", default="last", help="last, first, mean, or an index")
    parser.add_argument("--layers", default="all", help="'all' or comma-separated indices")
    parser.add_argument("--no-ablation", dest="ablate", action="store_false")
    parser.add_argument("--bf16", dest="fp32", action="store_false")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="attention_out")
    parser.set_defaults(ablate=True, fp32=True)
    return parser


def _frames_mode(raw):
    if isinstance(raw, str):
        return raw
    if len(raw) == 1 and raw[0] in ("grasp", "stride"):
        return raw[0]
    return [int(i) for i in raw]


def _layers(raw: str):
    if raw in ("all", "mean"):
        return raw
    return [int(i) for i in raw.split(",")]


def _denoise_step(raw: str):
    if raw in ("last", "first", "mean", "all"):
        return raw
    return int(raw)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if (args.dataset is None) == (args.dump_obs is None):
        parser.error("give exactly one of --dataset or --dump-obs")

    from .adapters.pi05 import Pi05Adapter
    from .loader import load_pi05
    from .sources import DatasetSource, DumpObsSource

    policy, preprocessor = load_pi05(
        args.checkpoint, device=args.device, fp32=args.fp32
    )
    adapter = Pi05Adapter(
        policy, preprocessor, device=args.device, seed=args.seed
    )

    if args.dump_obs is not None:
        source = DumpObsSource(
            args.dump_obs, task=args.task, camera_key=args.camera_key
        )
        notes = {0: f"dump_obs capture {args.dump_obs}"}
    else:
        source = DatasetSource(
            args.dataset,
            episodes=args.episodes,
            camera_keys=adapter.camera_keys,
            task=args.task,
            root=args.dataset_root,
            selection=_frames_mode(args.frames),
            count=args.count,
        )
        notes = source.notes

    out_root = Path(args.out)
    analyses = []
    for obs in source.frames():
        analysis = next(
            analyse(
                adapter,
                [obs],
                denoise_step=_denoise_step(args.denoise_step),
                layers=_layers(args.layers),
                ablate=args.ablate,
                provenance={
                    "checkpoint": args.checkpoint,
                    "seed": str(args.seed),
                    "dtype": "fp32" if args.fp32 else "bf16",
                },
            )
        )
        episode_dir = out_root / f"ep{obs.episode:03d}"
        write_overlays(analysis, obs, episode_dir)
        analyses.append(analysis)
        print(
            f"ep{obs.episode:03d} frame {obs.frame}: "
            + "  ".join(
                f"{c.rsplit('.', 1)[-1]} mass {a.mass:.2f}"
                + (
                    f" ablate {analysis.ablations[c].delta_mm:.1f}mm"
                    if c in analysis.ablations
                    else ""
                )
                for c, a in analysis.cameras.items()
            )
        )

    summary = write_summary(analyses, out_root, notes=notes)
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Add the console script and the path dependency**

```toml
# append to packages/grabette-attention/pyproject.toml
[project.scripts]
grabette-attn = "grabette_attention.cli:main"
```

In `integrations/Pi05/pyproject.toml`, add `"grabette-attention"` to the
`dependencies` list and add the workspace source next to the existing
`grabette-chunkrel` entry:

```toml
[tool.uv.sources]
grabette-chunkrel = { path = "../../packages/grabette-chunkrel", editable = true }
grabette-attention = { path = "../../packages/grabette-attention", editable = true }
```

- [ ] **Step 6: Write the package README**

```markdown
# grabette-attention

Offline debugging for GRABETTE policies: where does the policy attend on each
camera, and how much does the commanded chunk change when a camera is removed?

**Read `docs/attention_saliency_review.md` first.** On pi0.5 an interventional
score is measurably more faithful than attention, so the maps here are
hypothesis generators and the ablation millimetres are the evidence. A broad,
low-peak map is normal for pi0.5 and more so after action fine-tuning.

## Usage

```bash
# held-out dataset episodes, at the grasp frame
grabette-attn --checkpoint <user>/<model>_best \
              --dataset <user>/<dataset>_graspproj \
              --episodes 3 7 11 --task "pick the sugar cube" \
              --out attention_out

# the exact observations from a robot run
grabette-attn --checkpoint <user>/<model>_best \
              --dump-obs eval_dump/ep003 --task "pick the sugar cube"
```

Output per episode: one overlay per frame per camera, plus `summary.txt` with
each camera's attention mass, its ablation delta in millimetres and the per-axis
breakdown, the language mass, and the provenance.

## What it does not do

No interventional saliency yet, no Grad-CAM, no Diffusion Policy adapter, and
nothing on the robot's control path. Remote (Ficelle) inference returns only
actions, so this needs a local checkpoint.

## Design notes

- Attention comes from forward hooks on the action expert's attention modules.
  No `lerobot` modification: `sample_actions` forces eager attention and the
  attention module returns its probabilities.
- A view is removed by zeroing its image mask in place, not by dropping the
  batch key. Dropping the key would reorder tokens; and pi0.5 trains with
  missing-camera masking, so a masked view is in distribution.
- The baseline and every ablation for a frame share one noise tensor, so a delta
  means the camera mattered rather than the sampler drew differently.
- Tokens per image, grid shape, camera order and the pixel mapping are derived
  at runtime. Adding a second camera needs no change here.
```

- [ ] **Step 7: Run the tests and the whole package suite**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests -v`
Expected: PASS, all tests from Tasks 1-11

- [ ] **Step 8: Document the tool in the Pi05 README**

Add to the tools table in `integrations/Pi05/README.md`, next to
`smoke_generation.py` and the other probes:

```markdown
| `grabette-attn` (from `packages/grabette-attention`) | Offline attention maps per camera plus view-ablation deltas in mm. Answers "where is it looking" and "which camera does it rely on". The maps are hypotheses; the ablation numbers are the measurement. See `docs/attention_saliency_review.md`. |
```

- [ ] **Step 9: Verify the lock still resolves**

Run: `uv lock --check`
Expected: no changes required, or a clean `uv lock` if the new member shifted the graph

- [ ] **Step 10: Commit**

```bash
git add packages/grabette-attention integrations/Pi05 uv.lock
git commit -m "feat(attention): grabette-attn CLI, checkpoint loader and wiring"
```

---

### Task 12: Rerun front end

**Files:**
- Create: `packages/grabette-attention/grabette_attention/frontends/rerun_logger.py`
- Modify: `packages/grabette-attention/grabette_attention/cli.py` (add `--rerun`)
- Test: `packages/grabette-attention/tests/test_rerun.py`

**Interfaces:**
- Consumes: `FrameAnalysis`, `FrameObservation` (Task 1).
- Produces: `log_analysis(analysis, obs, *, recording=None) -> None` and `open_recording(name: str) -> Any`.

Entity naming follows the existing visualiser in
`packages/grabette-postprocess/scripts/visualize/visualize_rgbd_trajectory.py`.

- [ ] **Step 1: Write the failing test**

```python
# packages/grabette-attention/tests/test_rerun.py
"""The second front end, against a fake rerun module.

We do not want the tests to depend on rerun being installed, and we do want to
assert the entity paths and that BOTH front ends read the same records.
"""
import sys
import types

import numpy as np
import pytest

from grabette_attention.records import (
    CameraAttention,
    FrameAnalysis,
    FrameObservation,
    ViewAblation,
)


class FakeRerun(types.ModuleType):
    def __init__(self):
        super().__init__("rerun")
        self.logged: list[tuple[str, str]] = []
        self.times: list[int] = []

    def init(self, name, spawn=False):
        self.app = name

    def set_time(self, _timeline, *, sequence=None, timestamp=None):
        self.times.append(sequence if sequence is not None else timestamp)

    def log(self, path, entity):
        self.logged.append((path, type(entity).__name__))

    def Image(self, _array):            # noqa: N802 - mirrors rerun's API
        return types.SimpleNamespace()

    def Scalars(self, _value):          # noqa: N802
        return types.SimpleNamespace()


@pytest.fixture
def fake_rerun(monkeypatch):
    module = FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", module)
    return module


def analysis() -> FrameAnalysis:
    return FrameAnalysis(
        episode=3, frame=42,
        cameras={
            "observation.images.cam0": CameraAttention(
                grid=np.ones((12, 16), np.float32), mass=0.8
            )
        },
        language_mass=0.2,
        ablations={
            "observation.images.cam0": ViewAblation(
                delta_mm=8.4, per_axis_mm=(1.0, 2.0, 8.1)
            )
        },
        provenance={"checkpoint": "user/m"},
    )


def observation() -> FrameObservation:
    return FrameObservation(
        episode=3, frame=42,
        images={"observation.images.cam0": np.zeros((720, 960, 3), np.uint8)},
        state=np.zeros(2, np.float32), task="t",
    )


def test_it_logs_the_frame_and_the_overlay_under_the_camera_entity(fake_rerun):
    from grabette_attention.frontends import rerun_logger

    rerun_logger.log_analysis(analysis(), observation())
    paths = [p for p, _ in fake_rerun.logged]
    assert "camera_feed/cam0" in paths
    assert "camera_feed/cam0/attention" in paths


def test_it_logs_mass_and_ablation_as_scalar_series(fake_rerun):
    from grabette_attention.frontends import rerun_logger

    rerun_logger.log_analysis(analysis(), observation())
    paths = [p for p, _ in fake_rerun.logged]
    assert "metrics/mass/cam0" in paths
    assert "metrics/ablation_mm/cam0" in paths
    assert "metrics/mass/language" in paths


def test_the_frame_index_drives_the_timeline(fake_rerun):
    from grabette_attention.frontends import rerun_logger

    rerun_logger.log_analysis(analysis(), observation())
    assert fake_rerun.times == [42]


def test_a_missing_rerun_install_gives_a_clear_message(monkeypatch):
    from grabette_attention.frontends import rerun_logger

    monkeypatch.setitem(sys.modules, "rerun", None)
    with pytest.raises(ImportError, match="rerun"):
        rerun_logger.log_analysis(analysis(), observation())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests/test_rerun.py -v`
Expected: FAIL with `ImportError: cannot import name 'rerun_logger'`

- [ ] **Step 3: Write the implementation**

```python
# packages/grabette-attention/grabette_attention/frontends/rerun_logger.py
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


def _rerun():
    import sys

    module = sys.modules.get("rerun", "absent")
    if module is None:
        raise ImportError(
            "rerun is not installed; install the 'rerun' extra of "
            "grabette-attention, or use the PNG front end"
        )
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - exercised via the fake
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

    for camera, attention in analysis.cameras.items():
        short = camera.rsplit(".", 1)[-1]
        rr.log(f"camera_feed/{short}", rr.Image(obs.images[camera]))
        rr.log(f"camera_feed/{short}/attention", rr.Image(attention.grid))
        rr.log(f"metrics/mass/{short}", rr.Scalars(attention.mass))
        ablation = analysis.ablations.get(camera)
        if ablation is not None:
            rr.log(
                f"metrics/ablation_mm/{short}", rr.Scalars(ablation.delta_mm)
            )
    rr.log("metrics/mass/language", rr.Scalars(analysis.language_mass))
```

- [ ] **Step 4: Wire `--rerun` into the CLI**

In `cli.py`, add the flag to `build_parser`:

```python
    parser.add_argument(
        "--rerun", action="store_true", help="also log to a rerun timeline"
    )
```

and inside `main`, before the per-frame loop:

```python
    recording = None
    if args.rerun:
        from .frontends.rerun_logger import open_recording

        recording = open_recording()
```

and inside the loop, after `write_overlays`:

```python
        if recording is not None:
            from .frontends.rerun_logger import log_analysis

            log_analysis(analysis, obs, recording=recording)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests -v`
Expected: PASS, every test including the four rerun ones

- [ ] **Step 6: Commit**

```bash
git add packages/grabette-attention
git commit -m "feat(attention): rerun front end over the same analysis records"
```

---

### Task 13: Mutation checks and the GPU integration test

**Files:**
- Create: `packages/grabette-attention/tests/test_mutations.py`
- Create: `packages/grabette-attention/tests/test_integration_gpu.py`
- Modify: `packages/grabette-attention/pyproject.toml` (register the marker)

**Interfaces:**
- Consumes: everything from Tasks 1-12.
- Produces: no new public API.

The mutation file states, as executable assertions, the failures the spec says must be caught. The GPU test is marked and skipped by default; it is the one run by hand on the box with a real checkpoint.

- [ ] **Step 1: Write the mutation checks**

```python
# packages/grabette-attention/tests/test_mutations.py
"""Assertions that would break under the specific mistakes the spec names.

Each test here corresponds to a line in the spec's mutation list. They overlap
with earlier tests on purpose: this file is where a reviewer looks to confirm
the genericity requirements are actually enforced.
"""
import numpy as np
import pytest

from grabette_attention.layout import LetterboxGeometry, TokenLayout
from grabette_attention.metrics import translation_delta
from grabette_attention.reduce import reduce_attention


class _SameGeometryEverywhere(dict):
    """The 4:3 live-camera geometry, for whatever camera is asked."""

    def __missing__(self, _camera):
        return LetterboxGeometry.from_shapes(src_hw=(720, 960), dst_hw=(224, 224))


GEOM = _SameGeometryEverywhere()


def test_a_hard_coded_256_token_block_would_break_a_smaller_grid():
    # A 112-pixel input at patch 14 gives an 8x8 grid, 64 tokens.
    layout = TokenLayout(
        camera_keys=("cam0",), tokens_per_image=64, grid_rows=8, grid_cols=8,
        language_tokens=10, masked_cameras=frozenset(),
    )
    caps = {(0, 0): np.ones((8, 50, 64 + 10 + 50), np.float32)}
    geoms = {
        "cam0": LetterboxGeometry.from_shapes(src_hw=(112, 112), dst_hw=(112, 112))
    }
    cams, _ = reduce_attention(caps, layout, geoms, patch=14)
    assert cams["cam0"].grid.shape == (8, 8)


def test_a_hard_coded_single_camera_would_break_a_three_camera_prefix():
    layout = TokenLayout(
        camera_keys=("a", "b", "c"), tokens_per_image=256, grid_rows=16,
        grid_cols=16, language_tokens=200, masked_cameras=frozenset(),
    )
    caps = {(0, 0): np.ones((8, 50, 3 * 256 + 200 + 50), np.float32)}
    cams, lang = reduce_attention(caps, layout, GEOM, patch=14)
    assert set(cams) == {"a", "b", "c"}
    assert sum(c.mass for c in cams.values()) + lang == pytest.approx(1.0)


def test_transposing_the_token_grid_would_move_the_hot_cell():
    # Row-major: token r*cols + c is (row r, col c). A column-major reshape puts
    # this peak at (3, 5) instead of (5, 3) -> (3, 3) after cropping 2 pad rows.
    layout = TokenLayout(
        camera_keys=("cam0",), tokens_per_image=256, grid_rows=16, grid_cols=16,
        language_tokens=200, masked_cameras=frozenset(),
    )
    caps = {(0, 0): np.zeros((8, 50, 456 + 50), np.float32)}
    caps[(0, 0)][:, :, 5 * 16 + 3] = 1.0
    cams, _ = reduce_attention(caps, layout, GEOM, patch=14)
    grid = cams["cam0"].grid
    assert np.unravel_index(int(grid.argmax()), grid.shape) == (3, 3)


def test_renormalising_per_camera_would_hide_which_camera_is_used():
    # cam0 gets nine times cam1's attention. Per-camera renormalisation would
    # make both maps look identical and both masses 1.0.
    layout = TokenLayout(
        camera_keys=("cam0", "cam1"), tokens_per_image=256, grid_rows=16,
        grid_cols=16, language_tokens=200, masked_cameras=frozenset(),
    )
    caps = {(0, 0): np.zeros((8, 50, 2 * 256 + 200 + 50), np.float32)}
    caps[(0, 0)][:, :, 0:256] = 0.9
    caps[(0, 0)][:, :, 256:512] = 0.1
    cams, _ = reduce_attention(caps, layout, GEOM, patch=14)
    assert cams["cam0"].mass > 8 * cams["cam1"].mass


def test_padding_rows_must_not_be_plotted_as_content():
    layout = TokenLayout(
        camera_keys=("cam0",), tokens_per_image=256, grid_rows=16, grid_cols=16,
        language_tokens=200, masked_cameras=frozenset(),
    )
    caps = {(0, 0): np.zeros((8, 50, 456 + 50), np.float32)}
    caps[(0, 0)][:, :, 0:32] = 1.0            # grid rows 0-1: pure padding
    cams, _ = reduce_attention(caps, layout, GEOM, patch=14)
    assert cams["cam0"].grid.max() == 0.0     # cropped away entirely


def test_scaling_the_delta_twice_would_double_the_millimetres():
    baseline = np.zeros((4, 11), np.float32)
    ablated = baseline.copy()
    ablated[:, 0] = 0.001
    assert translation_delta(baseline, ablated).delta_mm == pytest.approx(1.0)
```

- [ ] **Step 2: Write the GPU integration test**

```python
# packages/grabette-attention/tests/test_integration_gpu.py
"""One end-to-end check against a real checkpoint. Skipped unless asked for.

Run on the GPU box:
    GRABETTE_ATTN_CKPT=<user>/<model>_best \
    GRABETTE_ATTN_DATASET=<user>/<dataset>_graspproj \
    uv run pytest -m gpu packages/grabette-attention/tests/test_integration_gpu.py -v

It asserts shapes and invariants, not values: the point is that the hooks fire
against the real module tree and that the masses are a proper distribution.
"""
import os

import pytest

pytestmark = pytest.mark.gpu

CKPT = os.environ.get("GRABETTE_ATTN_CKPT")
DATASET = os.environ.get("GRABETTE_ATTN_DATASET")


@pytest.mark.skipif(not CKPT or not DATASET, reason="set GRABETTE_ATTN_CKPT and _DATASET")
def test_a_real_checkpoint_yields_maps_masses_and_an_ablation():
    from grabette_attention.adapters.pi05 import Pi05Adapter
    from grabette_attention.analysis import analyse_frame
    from grabette_attention.loader import load_pi05
    from grabette_attention.sources import DatasetSource

    policy, preprocessor = load_pi05(CKPT, device="cuda", fp32=True)
    adapter = Pi05Adapter(policy, preprocessor, device="cuda", seed=0)

    source = DatasetSource(
        DATASET, episodes=[0], camera_keys=adapter.camera_keys,
        task="pick the sugar cube", selection="grasp", count=1,
    )
    obs = next(iter(source.frames()))
    analysis = analyse_frame(adapter, obs)

    # The hooks fired against the real expert layers.
    assert set(analysis.cameras) == set(adapter.camera_keys)
    # Masses are a distribution over the prefix.
    total = sum(c.mass for c in analysis.cameras.values()) + analysis.language_mass
    assert total == pytest.approx(1.0, abs=1e-5)
    # The grid lost its letterbox padding rows.
    grid = next(iter(analysis.cameras.values())).grid
    assert grid.ndim == 2 and grid.shape[0] < grid.shape[1]
    # Removing the only camera must change the commanded motion.
    assert next(iter(analysis.ablations.values())).delta_mm > 0.0
```

- [ ] **Step 3: Register the marker**

```toml
# append to packages/grabette-attention/pyproject.toml
[tool.pytest.ini_options]
markers = ["gpu: needs a GPU and a real checkpoint; run by hand"]
addopts = "-m 'not gpu'"
```

- [ ] **Step 4: Run the suite and confirm the GPU test is skipped**

Run: `uv run --project packages/grabette-attention pytest packages/grabette-attention/tests -v`
Expected: PASS, every unit and mutation test; the GPU test deselected by `addopts`

- [ ] **Step 5: Run ruff over the new package**

Run: `uv run ruff check packages/grabette-attention`
Expected: no findings (workspace selects E4, E7, E9, F)

- [ ] **Step 6: Commit**

```bash
git add packages/grabette-attention
git commit -m "test(attention): mutation checks and a marked GPU integration test"
```

---

## Plan Self-Review

**Spec coverage.** Walking the spec section by section:

| Spec section | Task |
|---|---|
| §4.1 frame sources, frame selection | 8 |
| §4.2 adapter as the genericity boundary | 7 (protocol in `base.py`) |
| §4.3 analysis records | 1, 9 |
| §5 hooks, bookkeeping, slicing, aggregation, denoise step, mass, letterbox | 2, 3, 4, 5 |
| §5 interpretation guard | 10 (in `summary.txt`), 11 (README) |
| §6 ablation mechanism, determinism, metric, self-test | 6, 7, 9 |
| §7 six multi-camera acceptance criteria | 3, 5, 7, 13 |
| §8 PNG output and summary contents | 10 |
| §9 CLI and flags | 11 |
| §10 testing plan including mutation list | every task, gathered in 13 |
| §11 language mass caveat | 10 (labelled "language (task + state)") |

Two spec items are deliberately deferred rather than dropped: the per-head view
(§5 says per-layer maps stay available behind a flag; `--layers` covers layers,
per-head is not exposed in v1) and splitting language mass into task text versus
proprioception, which the spec itself defers.

**Placeholder scan.** No TBD, no "add error handling", no "similar to Task N".
Every code step carries runnable code. Test bodies are complete.

**Type consistency.** Checked across tasks: `TokenLayout` fields and
`camera_index`/`camera_slice`/`visible_cameras` are used identically in Tasks 5,
7 and 13. `LetterboxGeometry.content_rows`/`content_cols` take `patch` in Tasks
2 and 5. `RunResult(chunk, captures)` is produced in Task 7 and consumed in Task
9. `ViewAblation(delta_mm, per_axis_mm)` is produced in Task 6 and read in Tasks
9, 10 and 12. `reduce_attention` keeps the same keyword-only signature in Tasks
5, 9 and 13. `translation_delta` returns `ViewAblation` everywhere.

One naming decision worth flagging for the executor: the rerun module is
`frontends/rerun_logger.py`, not `frontends/rerun.py`, so that `import rerun`
inside it resolves to the real package rather than to itself.

---

## Execution Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-09-08-attention-maps.md`. Two execution options:

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between
tasks, fast iteration.

**2. Inline Execution** — tasks executed in this session with batch checkpoints
for review.
