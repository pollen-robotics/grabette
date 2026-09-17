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
    """Mirrors `paligemma_with_expert`: the adapter resolves this exact path.

    `patch_size=None` omits the `.paligemma` vision-config path entirely, for
    the one test that exercises its absence; every other test wants a
    reachable default (SigLIP-so400m's own patch size, 14), since `.patch` no
    longer falls back to a constant when it is unreachable (Finding 2).
    """

    def __init__(self, n_layers, patch_size: int | None = 14):
        super().__init__()
        self.gemma_expert = StubExpert(n_layers)
        if patch_size is not None:
            inner = type("Inner", (), {})()
            inner.vision_tower = type("Tower", (), {})()
            inner.vision_tower.config = type("VisionCfg", (), {"patch_size": patch_size})()
            self.paligemma = type("PG", (), {})()
            self.paligemma.model = inner


class _Feature:
    """Duck-types `lerobot.configs.PolicyFeature` for the one attribute used
    here: `.shape`."""

    def __init__(self, shape):
        self.shape = shape


class StubConfig:
    def __init__(self, cameras):
        self.image_features = {c: None for c in cameras}
        self.image_resolution = (224, 224)
        self.chunk_size = 50
        self.max_action_dim = 32
        self.tokenizer_max_length = 200
        self.compile_model = False
        # Real action width is 11 (8 pose dims + gripper, or similar); the
        # padded model output is max_action_dim (32) wide, and `run()` must
        # truncate down to this before returning.
        self.output_features = {"action": _Feature(shape=(11,))}


class StubModel(nn.Module):
    """Its action depends on the SUM of the unmasked images, so masking a
    camera provably changes the output — that is what the ablation must see."""

    def __init__(self, n_layers=18, patch_size: int | None = 14):
        super().__init__()
        self.paligemma_with_expert = StubTwoTower(n_layers, patch_size=patch_size)
        self.steps = 10

    def sample_actions(self, images, img_masks, tokens, masks, noise=None):
        for _ in range(self.steps):
            for layer in self.paligemma_with_expert.gemma_expert.model.layers:
                layer.self_attn(torch.zeros(1))
        signal = sum(
            float(img.mean()) * float(m.float().mean())
            for img, m in zip(images, img_masks)
        )
        # Padded to max_action_dim (32), like the real `sample_actions` -- the
        # adapter, not the model, is responsible for truncating to the real
        # action width (11) and for converting to real units.
        chunk = noise * 0.0 + signal * 0.001
        return chunk


class StubPolicy(nn.Module):
    def __init__(self, cameras, patch_size: int | None = 14):
        super().__init__()
        self.config = StubConfig(cameras)
        self.model = StubModel(patch_size=patch_size)

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


class RecordingStubModel(StubModel):
    """Like StubModel, but keeps what it was actually handed by `run()`, so a
    test can inspect `images`/`img_masks` at the operational layer instead of
    only through `layout()`'s metadata."""

    def sample_actions(self, images, img_masks, tokens, masks, noise=None):
        self.received_images = list(images)
        self.received_img_masks = list(img_masks)
        return super().sample_actions(images, img_masks, tokens, masks, noise=noise)


class RecordingStubPolicy(StubPolicy):
    def __init__(self, cameras):
        super().__init__(cameras)
        self.model = RecordingStubModel()


def observation(cameras) -> FrameObservation:
    rng = np.random.default_rng(0)
    return FrameObservation(
        episode=0, frame=0,
        images={c: rng.integers(0, 255, (720, 960, 3), dtype=np.uint8) for c in cameras},
        state=np.zeros(2, np.float32), task="pick the sugar cube",
    )


def make_adapter(cameras, policy=None, postprocessor=None):
    policy = policy or StubPolicy(cameras)
    postprocessor = postprocessor or (lambda chunk: chunk)   # identity by default

    def preprocessor(batch):
        batch = dict(batch)
        batch["observation.language.tokens"] = torch.zeros(1, 200, dtype=torch.long)
        batch["observation.language.attention_mask"] = torch.ones(1, 200, dtype=torch.long)
        return batch

    return Pi05Adapter(policy, preprocessor, postprocessor, device="cpu", seed=0)


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
    a = Pi05Adapter(StubPolicy(["cam_a"]), lambda b: b, lambda c: c, device="cpu", seed=7)
    b = Pi05Adapter(StubPolicy(["cam_a"]), lambda b: b, lambda c: c, device="cpu", seed=7)
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


def test_dropping_a_camera_masks_in_place_not_by_deletion():
    """Guards `run()` itself, not just `layout()`'s metadata: if the mask
    mutation at pi05.py were ever regressed to deleting the list entries
    instead of assigning them in place, the length assertion below would
    catch it — `StubModel.sample_actions`'s `zip(images, img_masks)` would
    otherwise tolerate a shortened list with no complaint of its own, so a
    chunk-differs assertion alone cannot distinguish "masked in place" from
    "removed, so the next camera slid into its slot"."""
    policy = RecordingStubPolicy(["cam_a", "cam_b"])
    adapter = make_adapter(["cam_a", "cam_b"], policy=policy)
    obs, noise = observation(["cam_a", "cam_b"]), adapter.draw_noise()

    adapter.run(obs, noise, capture=False)
    baseline_cam_a = policy.model.received_images[0].clone()

    adapter.run(obs, noise, capture=False, drop_camera="cam_b")
    images = policy.model.received_images
    masks = policy.model.received_img_masks

    # One entry per configured camera, still — a `del` would shrink this.
    assert len(images) == len(masks) == len(adapter.camera_keys) == 2

    # cam_b (index 1, the dropped one): padding value, mask cleared.
    assert torch.all(images[1] == -1)
    assert not bool(masks[1].any())

    # cam_a (index 0, untouched): bitwise identical to the un-ablated pass,
    # so it is still at its own slot rather than having shifted.
    assert torch.equal(images[0], baseline_cam_a)
    assert bool(masks[0].all())


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
    adapter = Pi05Adapter(policy, lambda b: b, lambda c: c, device="cpu", seed=0)
    assert adapter.patch == 16


def test_the_patch_size_raises_when_the_vision_config_is_unreachable():
    # No SigLIP-default fallback (Finding 2): a wrong guess would keep
    # tokens-per-image and grid rows*cols mutually consistent BY CONSTRUCTION
    # while silently disagreeing with the captured attention tensor, so every
    # camera block would be misaligned while looking fine. A policy that does
    # not expose the vision config is a programming error to surface.
    policy = StubPolicy(["cam_a"], patch_size=None)
    adapter = make_adapter(["cam_a"], policy=policy)
    with pytest.raises(AttributeError, match="vision_tower"):
        adapter.patch


def test_a_policy_without_the_pi05_module_tree_is_rejected_clearly():
    class NotPi05(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = StubConfig(["cam_a"])
            self.model = nn.Module()

    with pytest.raises(AttributeError, match="paligemma_with_expert"):
        Pi05Adapter(NotPi05(), lambda b: b, lambda c: c, device="cpu", seed=0)


def test_the_postprocessor_converts_the_raw_chunk_to_real_units():
    """Pins Finding 1 (CRITICAL). `sample_actions` returns the chunk in the
    policy's normalized (quantile) action space; only the postprocessor's
    Unnormalizer + AbsoluteActions steps convert it to real units. A stub
    postprocessor that scales by a known factor lets this test see whether the
    adapter actually applies it -- returning the raw model output straight
    through, unscaled, is exactly the bug this pins.
    """
    scale = 1000.0

    def scaling_postprocessor(chunk):
        return chunk * scale

    obs = observation(["cam_a"])
    noise = torch.zeros(1, 50, 32)     # zeroed out by StubModel; shape only

    raw = make_adapter(["cam_a"]).run(obs, noise, capture=False).chunk
    scaled = make_adapter(
        ["cam_a"], postprocessor=scaling_postprocessor
    ).run(obs, noise, capture=False).chunk

    np.testing.assert_allclose(scaled, raw * scale)
