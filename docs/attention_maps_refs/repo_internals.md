# Repo context for an attention-map / saliency tool (pi0.5 + Diffusion Policy)

Legend: **[V]** = verified by reading source; **[I]** = inferred (standard behaviour / not executed here).

## 0. Which lerobot is in play

- All four venvs that matter carry **byte-identical** `modeling_pi05.py` (md5 `a5758a20…`) and
  `modeling_diffusion.py` (md5 `0272d025…`) [V]:
  - `…/worktrees/attention-maps/.venv` and `…/worktrees/relative-actions/.venv` -> lerobot **0.6.0** (PyPI)
  - `…/worktrees/relative-actions/integrations/Pi05/.venv` -> lerobot **0.6.1** from git commit `e40b58a8` (`direct_url.json`)
  - `…/integrations/DiffusionPolicy/.venv` -> lerobot 0.6.0
  - (`/home/steve/Project/Repo/GRABETTE/openarm_gripette_simu/.venv` is stale lerobot 0.5.1 with a *different* file; ignore.)
- Readable source copy used for line numbers below (identical to what is installed):
  `LR = /home/steve/.cache/uv/git-v0/checkouts/b2400a7a62d6a7cf/e40b58a8/src/lerobot/policies`
- Stack in the Pi05 venv: `transformers 5.5.4`, `torch 2.11.0+cu128` [V]. transformers paths below are
  `TF = …/relative-actions/integrations/Pi05/.venv/lib/python3.12/site-packages/transformers`.

---

## 1. pi0.5 image -> tokens -> prefix layout

### 1.1 Image preprocessing (`PI05Policy._preprocess_images`, `LR/pi05/modeling_pi05.py:1149-1213`) [V]
- Iterates `present_img_keys = [key for key in self.config.image_features if key in batch]` (l.1161) —
  **camera order = order of `config.image_features`** (dict order of `input_features` filtered to VISUAL), not batch order.
- Per image: cast to float32 (l.1179-1180), to HWC (l.1183-1187), then
  `if img.shape[1:3] != self.config.image_resolution: img = resize_with_pad_torch(img, *self.config.image_resolution)` (l.1190-1191),
  then `img = img * 2.0 - 1.0` (l.1194, SigLIP range), back to CHW.
- `resize_with_pad_torch` (l.162-233) — **letterbox**, aspect preserved:
  `ratio = max(cur_width / width, cur_height / height)`; `resized_h = int(cur_h/ratio)`, `resized_w = int(cur_w/ratio)`;
  bilinear `F.interpolate` (align_corners=False); `pad_h0, rem = divmod(height - resized_h, 2)`, `pad_h1 = pad_h0 + rem`
  (same for w); `F.pad(..., (pad_w0, pad_w1, pad_h0, pad_h1))` i.e. **centered, extra pixel goes bottom/right**, pad value 0.0
  (=black in [0,1] -> becomes -1 after the `*2-1`). Only uint8/float32 accepted (l.207-212, raises otherwise).
  - Example for the Grabette 960x720 frames: ratio = 960/224 = 4.2857; resized = (168, 224); pad_h0 = 28, pad_h1 = 28, pad_w = 0.
    Inverse map for letterboxed pixel (u, v): x_orig = u * ratio, y_orig = (v - 28) * ratio.
- Missing cameras (`missing_img_keys`, l.1162, 1206-1211): `img = torch.ones_like(img) * -1` and `mask = zeros` — appended **after** the present ones.
  `empty_cameras` (`configuration_pi05.py:68`, `validate_features` l.126-132) adds keys `observation.images.empty_camera_{i}` to `input_features`,
  so they are simply extra (missing) image features. **Grabette trains with `--policy.empty_cameras=0`** (Pi05 README l.82, 110, 125; l.197 "do NOT zero-pad camera slots").

### 1.2 SigLIP tokens (`PaliGemmaWithExpertModel.embed_image`, l.446-455) [V]
- `image_outputs = self.paligemma.model.get_image_features(image); features = image_outputs.pooler_output` — in
  `TF/models/paligemma/modeling_paligemma.py:249-257`: `selected_image_feature = image_outputs.last_hidden_state;
  image_features = self.multi_modal_projector(selected_image_feature)` -> **all patch tokens, no pooling, no CLS**
  (`vision_use_head=False`, `configuration_paligemma.py:75`). Projector = single `nn.Linear(1152 -> 2048)` (`modeling_paligemma.py:92-100`).
- Vision config: lerobot builds `CONFIG_MAPPING["paligemma"]()` (l.365) whose default vision config is SigLIP-so400m
  `hidden_size=1152, patch_size=14, image_size=224, num_hidden_layers=27, num_attention_heads=16` (`configuration_paligemma.py:66-75`);
  lerobot then sets `vision_config.image_size = image_size` (224, l.379), `intermediate_size = 4304`, `projection_dim = 2048` (l.380-381).
- **Tokens per image = (224/14)^2 = 256**, `num_image_tokens` formula at `configuration_paligemma.py:97`.
- **Patch order is row-major**: `SiglipVisionEmbeddings.forward` (`TF/models/siglip/modeling_siglip.py:175-179`):
  `patch_embeds = self.patch_embedding(pixel_values)  # [*, width, grid, grid]; embeddings = patch_embeds.flatten(2).transpose(1, 2)`
  -> token `t` within an image = `row * 16 + col`, patch (row, col) covers letterboxed pixels `[14*row, 14*row+14) x [14*col, 14*col+14)`.
- Vision tower + projector are kept **float32** even in bf16 mode (`params_to_keep_float32`, l.415-427) and `embed_image` casts the image to fp32 (l.449-450), casting features back to `image.dtype` (l.453-454).

### 1.3 Prefix assembly (`PI05Pytorch.embed_prefix`, l.653-693) [V]
```
for img, img_mask in zip(images, img_masks):        # camera order as in 1.1
    img_emb = embed_image(img)                        # (B, 256, 2048)
    embs.append(img_emb); pad_masks.append(img_mask[:, None].expand(B, 256)); att_masks += [0]*256
lang_emb = embed_language_tokens(tokens)              # (B, 200, 2048)
embs.append(lang_emb); pad_masks.append(masks); att_masks += [0]*200
embs = torch.cat(embs, dim=1)
```
- Language: `TokenizerProcessorStep(tokenizer_name="google/paligemma-3b-pt-224", max_length=config.tokenizer_max_length (200),
  padding_side="right", padding="max_length")` (`processor_pi05.py:151-156`, `configuration_pi05.py:70`). Prompt string is
  `f"Task: {cleaned_text}, State: {state_str};\nAction: "` with the state discretised to 256 bins (`processor_pi05.py:77-83`) —
  so **state lives in the language tokens**, there is no separate state token.
- **Prefix layout (N_cam real + N_empty cameras)**: `[cam_0: 0..255] [cam_1: 256..511] … [lang: 256*N_img .. 256*N_img+199]`.
  `token_index -> (camera = idx // 256, row = (idx % 256) // 16, col = idx % 16)` for `idx < 256*N_img`; else language token `idx - 256*N_img`.
  Grabette default (1 camera, 0 empty): **prefix_len = 456**; images 0..255, language 256..455.
- Attention mask inside the prefix: all `att_masks = 0` -> `make_att_2d_masks` (l.110-139): `cumsum` all zeros ->
  `att_2d = cumsum[:,None,:] <= cumsum[:,:,None]` is all-True, ANDed with `pad_2d = pad[:,None,:] * pad[:,:,None]`.
  -> **fully bidirectional attention between images and language**; padded language tokens (and empty-camera tokens, `img_mask=0`)
  are masked as keys *and* queries. Position ids: `torch.cumsum(prefix_pad_masks, dim=1) - 1` (l.820).

### 1.4 Suffix (action tokens) (`embed_suffix`, l.695-740) [V]
- `chunk_size` (=50 default) action tokens: `action_in_proj(noisy_actions)` (B, 50, 1024); time enters only via AdaRMS `adarms_cond = time_mlp(...)`.
- `att_masks += [1] + [0]*(chunk_size-1)` (l.733): one block -> cumsum = 1 for all action tokens -> they attend to **all prefix**
  (cumsum 0 <= 1) and to **all other action tokens** (bidirectional within the chunk); prefix cannot see actions.
  In `denoise_step` (l.885-887) this is built explicitly: `full_att_2d_masks = cat([prefix_pad_masks expanded (B, 50, prefix_len), suffix_att_2d_masks], dim=2)`.
- Position ids of action tokens continue after the prefix: `prefix_offsets + cumsum(suffix_pad) - 1` (l.889-890).

---

## 2. Attention path and hook points

### 2.1 Inference control flow (`sample_actions`, l.791-869; `denoise_step`, l.871-908) [V]
1. `embed_prefix` -> `make_att_2d_masks` -> `_prepare_attention_masks_4d` (l.632-635: `torch.where(mask, 0.0, OPENPI_ATTENTION_MASK_VALUE)` -> additive 4D float mask `(B,1,Q,K)`).
2. `self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"` (l.823) then
   `paligemma_with_expert.forward(inputs_embeds=[prefix_embs, None], use_cache=True)` (l.825-831) -> **prefix prefill** through the 18 PaliGemma (Gemma-2B) layers
   (`PaliGemmaWithExpertModel.forward` branch `inputs_embeds[1] is None`, l.471-482), returning `past_key_values` (a `DynamicCache`).
3. Loop `for step in range(num_steps)` (10 default, `configuration_pi05.py:44`), `time = 1.0 + step*dt`, `dt = -1/num_steps`:
   `denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)` (l.871-908):
   - `self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"` (l.893)
   - `past_key_values = clone_past_key_values(past_key_values)` (l.895; l.142-148 clones every layer's K/V) — **the prefix KV cache is reused
     unchanged by every denoising step; only the 50 suffix tokens are recomputed** through the 18 **action-expert (Gemma-300M)** layers
     (`forward` branch `inputs_embeds[0] is None`, l.483-494 -> `gemma_expert.model.forward(..., past_key_values=...)`).
   - The fused two-tower `compute_layer_complete` (l.237-305, calls `modeling_gemma.eager_attention_forward` at l.273 and discards
     weights: `att_output, _ = …`) is **training-only** (both embeddings present); it is *not* used at inference.
4. `x_t = x_t + dt * v_t`.

So per `predict_action_chunk`: 18 PaliGemma-layer attentions once (prefix<->prefix), then 10 x 18 expert-layer attentions (actions -> prefix+actions).

### 2.2 Where weights are computed and dropped [V]
- `PiGemmaModel.forward` (`LR/pi_gemma.py:212-323`) -> per layer `_PiGemmaDecoderLayerBase.forward` (l.157-188):
  `hidden_states, _ = self.self_attn(hidden_states, attention_mask=…, past_key_values=…, position_embeddings=…)` (l.171) — **attention probabilities discarded here**.
  (`output_attentions` plumbing at l.288-310 is dead: the layer returns a plain tensor, so `layer_outputs[1]` would fail — do not rely on it.)
- `self_attn` is stock `transformers GemmaAttention` (`pi_gemma.py:145`). `GemmaAttention.forward` (`TF/models/gemma/modeling_gemma.py:262-304`):
  `attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)`;
  `attn_output, attn_weights = attention_interface(...)`; **`return attn_output, attn_weights`** (l.304).
- `eager_attention_forward` (`modeling_gemma.py:210-232`): `attn_weights = softmax(QK^T*scaling + mask, dim=-1, dtype=float32).to(query.dtype)`; `return attn_output, attn_weights`.
  `repeat_kv` (l.220) expands the single KV head to 8 query heads (both variants: `num_heads=8, num_kv_heads=1, head_dim=256`, l.322-339).
- `AttentionInterface.get_interface` (`TF/modeling_utils.py:4857-4869`): `"eager"` is **not** a key of `_global_mapping` (only `"sdpa"`, `"flash_*"`, `"paged|eager"`, …),
  so `get_interface("eager", default)` returns the `default` argument = the **module-level `eager_attention_forward` looked up at call time** in
  `modeling_gemma`. Consequence: monkeypatching `transformers.models.gemma.modeling_gemma.eager_attention_forward` *would* take effect, but it is unnecessary (see 2.3).
- The 4D float mask passes straight through `create_causal_mask` ("It can also be an already prepared 4D mask, in which case it is returned as-is", `TF/masking_utils.py:906`, early-exit l.939-942).

### 2.3 Cleanest hook (no lerobot modification): `register_forward_hook` on each `GemmaAttention` [V for shapes/semantics, I that nothing else interferes]
Because `GemmaAttention.forward` returns `(attn_output, attn_weights)` and lerobot forces `"eager"` before every call, a **forward hook on the
attention modules receives the probabilities in `output[1]`**:
```python
m = policy.model.paligemma_with_expert
expert_attn = [l.self_attn for l in m.gemma_expert.model.layers]                  # 18 layers, actions -> prefix+actions
vlm_attn    = [l.self_attn for l in m.paligemma.model.language_model.layers]      # 18 layers, prefix -> prefix (prefill only)
h = expert_attn[i].register_forward_hook(lambda mod, inp, out: store(i, out[1]))
```
- Expert hook fires **once per layer per denoising step** (10 x 18 calls); `out[1]` shape **`(B, 8, 50, prefix_len + 50)`**
  (queries = 50 action tokens, keys = cached prefix `prefix_len` then the 50 action tokens). Slice `[..., :prefix_len]` and re-normalise if you want
  prefix-only mass; index `[:, :, :, cam*256 : (cam+1)*256].reshape(B, 8, 50, 16, 16)` for camera `cam`.
  dtype = query dtype (fp32 in the repo's fp32 deployment; bf16 if loaded bf16).
- VLM hook fires once per layer per `predict_action_chunk`; `out[1]` shape `(B, 8, prefix_len, prefix_len)` (image<->image, language->image).
- Distinguish denoising steps by counting calls (18 per step) or by hooking `policy.model.denoise_step` via a wrapper (it is a plain method, not decorated).
- Alternative hook point if you want head-averaged, layer-output-based rollout: hook the `self_attn` *input* too (`inp[0]` is the normed hidden state).
- **SigLIP attention (patch<->patch inside the vision tower)**: `SiglipAttention.forward` (`TF/models/siglip/modeling_siglip.py:283-308`) also returns
  `attn_weights`, BUT lerobot never sets `_attn_implementation` on the vision config; transformers defaults to SDPA when available [I; `modeling_utils.py:1786-1787`
  mentions the "None to sdpa" fallback] and `sdpa_attention_forward` returns `(attn_output, None)` (`TF/integrations/sdpa_attention.py:104`).
  To capture it: `m.paligemma.model.vision_tower.config._attn_implementation = "eager"` (config object is shared by all 27 layers) before the forward; then a hook
  on `vision_tower.vision_model.encoder.layers[i].self_attn` gives `(B, 16, 256, 256)`. Check the actual value with `print(vision_tower.config._attn_implementation)`.
- RTC: `rtc_config` is `None` by default (`configuration_pi05.py:60`); ignore `RTCProcessor.denoise_step` (l.848-860).
- `compile_model` is forced off by every repo loader (evaluate.py l.51-52, smoke_generation.py l.212-213, ficelle `policy_host.py` uses config as saved but README serves with `--dtype float32`), so hooks are not swallowed by `torch.compile`. If a checkpoint had `compile_model=True`, `sample_actions` would be wrapped (l.599-603) and hooks on submodules may still fire but with graph breaks [I].

---

## 3. Gradients (Grad-CAM / integrated gradients) for pi0.5

- **`@torch.no_grad()` on**: `PI05Pytorch.sample_actions` (l.791), `PI05Policy.select_action` (l.1220), `PI05Policy.predict_action_chunk` (l.1237). [V]
- **Not decorated**: `_preprocess_images` (l.1149), `embed_prefix` (l.653), `embed_suffix` (l.695), `denoise_step` (l.871), `PaliGemmaWithExpertModel.forward` (l.460),
  `embed_image` (l.446), `PI05Pytorch.forward` (training loss, l.742). [V]
- `torch.no_grad()` used as a decorator goes through `torch.utils._contextlib.context_decorator` which uses `@functools.wraps(func)` (`TF-venv/torch/utils/_contextlib.py:118-124`) —
  so **`PI05Pytorch.sample_actions.__wrapped__`** is the undecorated function [V]. Either call
  `type(policy.model).sample_actions.__wrapped__(policy.model, images, img_masks, tokens, masks)` inside `torch.enable_grad()`, or (cleaner, and lets you pick the
  denoising step to attribute) re-implement the ~15-line loop of l.818-864 with `embed_prefix` + `paligemma_with_expert.forward(use_cache=True)` + `denoise_step`.
  Note `@torch.no_grad()` is a decorator here, not `inference_mode`, so tensors created inside are ordinary (no inference-tensor errors when mixing) [V].
- **Gradient path to pixels** [V]: `batch[key]` (float32 CHW in [0,1]) -> `_preprocess_images` (`permute`, `resize_with_pad_torch`: `F.interpolate` + `F.pad`, `*2-1`) ->
  `embed_image` (`.to(float32)` is a no-op copy for fp32 input; SigLIP conv patch embedding, 27 layers, projector) -> prefix -> PaliGemma prefill -> KV cache -> expert.
  All differentiable. Make the image tensor a leaf with `requires_grad_(True)` *before* `_preprocess_images`. Note `resize_with_pad_torch` raises for dtypes other than uint8/float32 (l.212).
- **Grad-CAM feature layer candidates**: (a) output of `m.paligemma.model.multi_modal_projector` — `(B, 256, 2048)` per camera call (one call per camera, in camera order);
  (b) `vision_tower.vision_model.encoder.layers[-1]` output `(B, 256, 1152)`; (c) the prefix embedding `prefix_embs` returned by `embed_prefix` (`(B, prefix_len, 2048)`, slice per camera).
  Target scalar: e.g. `v_t[:, :k, dims].sum()` from one `denoise_step`, or the final `x_t` action chunk.
- **Memory / dtype** [V for facts, I for numbers]: Grabette pi05 checkpoints are trained with `--policy.dtype=bfloat16` but **every repo loader casts to fp32**
  (`evaluate.py:54-55`, `smoke_generation.py:215` + docstring l.16 "the port has a bf16 flow-path dtype clash", Pi05 README l.223-224, 242: "fp32 is 16.6 GB of weights … use a >=24GB card").
  Backward through prefill (2B) + SigLIP (400M) at 456 tokens roughly adds activations of a few GB on top of 16.6 GB — plan on a 32-48 GB card or attribute only
  through the expert (freeze prefix as constant KV: gradients w.r.t. pixels then vanish, so that only works for attention-based maps, not Grad-CAM) [I].
  In bf16 mode: vision tower/projector/norms stay fp32 (l.415-427), embeddings are cast to bf16 at l.276-277 of `pi_gemma.py` and l.755-756 of the training forward;
  the attention softmax is computed in fp32 then cast to query dtype (`modeling_gemma.py:227`) — fine for visualisation, but bf16 IG sums are noisy [I].
- `_apply_checkpoint` only checkpoints when `self.training` (l.626) — irrelevant in eval.

---

## 4. lerobot Diffusion Policy (`LR/diffusion/modeling_diffusion.py`) and the Grabette config

### 4.1 Encoder [V]
- `DiffusionRgbEncoder` (l.473-558): optional `torchvision.transforms.Resize(config.resize_shape)` (l.482-483; a (H,W) tuple -> **non-aspect-preserving squash**),
  optional crop (`CenterCrop` in eval, `RandomCrop` in train, l.487-497, 546-553); backbone = `nn.Sequential(*list(resnet.children())[:-2])` (l.505, drops avgpool+fc) ->
  `SpatialSoftmax(feature_map_shape, num_kp=spatial_softmax_num_keypoints)` (l.532) -> `flatten` -> `Linear(2K, 2K)` + ReLU (l.534-535, 555-557). `feature_dim = 2*K`.
- GroupNorm replaces BatchNorm when `use_group_norm` (l.506-515: `nn.GroupNorm(num_groups=x.num_features // 16, …)`, only allowed with `pretrained_backbone_weights=None`).
- `SpatialSoftmax.forward` (l.451-470): `features = self.nets(features)` (1x1 conv C->K, l.437) -> `attention = F.softmax(features.reshape(-1, H*W), dim=-1)` (l.464) ->
  `expected_xy = attention @ pos_grid` -> `(B, K, 2)` in **[-1,1] normalised crop coordinates** (`pos_grid` from `np.meshgrid(linspace(-1,1,W), linspace(-1,1,H))`, l.445-449; x first, y second).
  -> the per-keypoint softmax `attention` **is an intrinsic K-channel attention map** (K x H x W). Capture with a forward hook on `encoder.pool.nets` (gives logits `(B, K, H, W)`; softmax over H*W yourself),
  and the keypoints themselves from a hook on `encoder.pool` (`(B, K, 2)`) to draw as points.
- **Cleanest Grad-CAM layer**: `encoder.backbone[7]` (= ResNet `layer4`, last conv stage; `children()` order conv1,bn1,relu,maxpool,layer1..4 -> indices 0..7 after `[:-2]`) [I on index, V on construction];
  or simply hook `encoder.backbone` output: `(N, 512, 7, 7)` for a 224x224 crop (stride 32) [I on 7x7]. Everything downstream (`pool`, `out`, UNet) is differentiable.

### 4.2 Multi-camera layout (`DiffusionModel`, l.190-209; `_prepare_global_conditioning`, l.269-305) [V]
- `use_separate_rgb_encoder_per_camera=True` -> `nn.ModuleList` of one encoder per camera (l.199-202), features computed per camera and concatenated `"(n b s) ... -> b s (n ...)"` (l.277-288).
- `False` -> **one shared encoder**; images flattened `"b s n ... -> (b s n) ..."` (l.292) so the encoder batch index is `(b*S + s)*N + n` (b sample, s obs step, n camera); features `"(b s n) ... -> b s (n ...)"` (l.296-298).
- `batch["observation.images"]` is built by stacking `config.image_features` keys along `dim=-4` (`select_action` l.151, `predict_action_chunk` l.119, `forward` l.169) -> `(B, n_obs_steps, N_cam, C, H, W)`; camera order = `config.image_features` order.
- `global_cond = cat([state (B,S,state_dim), img_features (B,S,N*2K)], -1).flatten(1)` (l.272-305) -> `(B, S*(state_dim + N*2K))`, FiLM-conditioning the 1D UNet (l.647, 726).
- Inference: `predict_action_chunk` and `select_action` are `@torch.no_grad()` (l.102, 123); `generate_actions` (l.307) and `conditional_sample` (l.233) are **not** — call
  `policy.diffusion.generate_actions(batch)` under grad for Grad-CAM (batch must already be time-stacked `(B, S, …)` with `OBS_IMAGES` key; see `predict_action_chunk` l.113-119 for how it builds it offline).

### 4.3 Grabette config (`integrations/DiffusionPolicy/train.py:473-520`) [V]
`DiffusionConfig(n_obs_steps=args.n_obs_steps (default 2, l.281-283), horizon=16, n_action_steps=args.n_action_steps (default 8), vision_backbone="resnet18",
resize_shape=(236, 236), crop_ratio=0.95  # -> crop_shape (224, 224) via configuration_diffusion.py:191-196, crop_is_random=not args.no_random_crop,
pretrained_backbone_weights=None, use_group_norm=True, use_separate_rgb_encoder_per_camera=False, spatial_softmax_num_keypoints=32,
down_dims=(256, 512, 1024), kernel_size=5, n_groups=8, diffusion_step_embed_dim=128, use_film_scale_modulation=True,
noise_scheduler_type="DDIM", num_train_timesteps=50, num_inference_steps=16, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True,
normalization_mapping VISUAL=MEAN_STD, STATE=MIN_MAX, ACTION=MIN_MAX)`.
Cameras: `--cameras` default `["observation.images.cam0"]` (l.309); input features = dataset features filtered to those cameras (l.440-444).
- So a Grabette DP checkpoint: **shared encoder, K=32 keypoints (64-D per image), 2 obs frames, 1 camera** -> `global_cond_dim = 2*(2 + 64) = 132`.
- Pixel mapping for DP: dataset/live frame (W x H, e.g. 480x360 training copy or 960x720 live) -> `Resize((236,236))` (squash) -> `CenterCrop(224)` (offset 6 px each side) -> 7x7 map.
  Crop pixel (u,v) -> original `x = (u+6)/236 * W`, `y = (v+6)/236 * H`. SpatialSoftmax keypoint `(kx, ky) in [-1,1]` -> crop pixel `u = (kx+1)/2 * 223`, `v = (ky+1)/2 * 223` [I on the linspace endpoints, V on formula].
- DP `README.md` l.17, 65-69, 247-250: datasets stored 960x720, training copy 480x360 (`resize_dataset_videos.py` docstring), "both meet at the encoder's internal 236x236 resize".

---

## 5. Repo scripts

### 5.1 `integrations/openarm/openarm_gripette_simu/examples/evaluate.py` (2373 lines) [V]
- `--dump_obs DIR` (l.509-512): "Directory to dump the EXACT observations fed to the policy (obs_XXXXX.png + state.jsonl, one subdir per episode). Use with --num_episodes 1 for a train/deploy distribution check (ood_check.py)."
  - Sync path (l.1234-1244): **per control step**, `cv2.imwrite(dump_dir / f"obs_{step:05d}.png", cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR), [IMWRITE_PNG_COMPRESSION, 1])` and one JSON line
    `{"step": step, "state": [floats]}` appended to `state.jsonl`. `camera_image` is **RGB HWC uint8 full-res (960x720 live), pre-normalisation, pre-letterbox**.
  - Async path (l.1804-1828): same PNG/state.jsonl keyed by `executor.sent_count`, plus `traj.jsonl` lines `{"t", "tick", "meas", "meas_r6d", "cmd", "cmd_r6d", "grip_cmd", "grip_obs"}`.
  - Per-episode subdir: `f"{args.dump_obs}/ep{ep:03d}"` (l.2268, 2296). No action/chunk is dumped in sync mode. Consumer: `integrations/DiffusionPolicy/ood_check.py` (`--images DIR`, docstring l.1-31).
- Policy loading `_load_policy_any` (l.37-56): `cfg = PreTrainedConfig.from_pretrained(ckpt); cfg.compile_model = False; policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg); if cfg.type in ("pi05","pi0"): policy = policy.to(dtype=torch.float32)`.
  Then `policy.to(device); policy.eval(); make_pre_post_processors(policy.config, pretrained_path=ckpt, preprocessor_overrides={"device_processor": {"device": str(device)}})` (l.2064-2074).
- Batch construction (l.1246-1259): `image_tensor = torch.from_numpy(camera_image).float() / 255.0; .permute(2,0,1)`; batch = `{"observation.state": (1, D), "observation.images.cam0": (1,3,H,W), "task": task}`, then `batch = preprocessor(batch)` (l.1271) and `policy.select_action(batch)` under `torch.no_grad()` (l.1274-1276, 1391-1393).
  **No letterbox/resize in evaluate.py for the local path** — pi05 letterboxes internally (`_preprocess_images`), DP squashes to 236 internally. Frames come from `CameraStream` (l.657-729): gRPC `StreamState` JPEG -> `cv2.imdecode` -> RGB, plus gripper motor positions.
- Remote (`ficelle_client`) path (l.1341-1383, 1974-2047): `client = open_client(args.policy_addr, jpeg_quality=…, resize=…)`; `metadata = client.metadata` gives `n_obs_steps`, `n_action_steps`, `action_dim`, `observations[key]["frame_shape"]` (l.2004-2042);
  frames are `cv2.resize(img, remote_img_wh)` to the server's declared `frame_shape` (l.1361-1373) and sent as `{"observation.images.cam0": uint8 HWC (or (2,H,W,3) stacked for n_obs_steps=2), "observation.state": float32, "task": str}`;
  `reply = client.infer(obs); action_queue.extend(reply["actions"][:remote_k])` (l.1379-1380).
- `--debug` (l.65-91, 1584-1594): `cv2.putText` overlay + `debug_display()` which tries `cv2.imshow("Evaluation", img_bgr)` and falls back to `cv2.imwrite("eval_debug_frames/frame_%05d.png")` on headless OpenCV — **the repo's convention for overlays on camera frames: RGB in memory, BGR only at imwrite/imshow, headless-safe fallback to PNGs.**

### 5.2 `ficelle` (`/home/steve/Project/Repo/ficelle`) [V]
- Client `client/ficelle_client/policy_client.py:102-127`: msgpack-numpy over websocket; `infer(obs) -> {"actions": (n_action_steps, action_dim) float32, "server_timing": {"infer_ms"}}`.
- Server `policy_host.py:198-230`: `prepare_observation_for_inference` -> checkpoint preprocessor -> `with torch.no_grad(): policy.reset(); actions = policy.predict_action_chunk(batch)` -> per-step postprocess -> **only actions are returned**.
  `serve.py:6` "one msgpack frame {"actions": ..., "server_timing": {...}}"; args `--checkpoint --host --port --api_key --device --dtype {float32,bfloat16} --transport {websocket,iroh}` (l.123-136).
  -> **Attention cannot be obtained remotely today**; it would need a server-side hook + an extra reply key (e.g. `"attention": {layer: (8,50,prefix_len)}`), and the wire protocol string `PROTOCOL = "ficelle/1"` (l.43) bumped. Local checkpoint path is the natural place for the tool.

### 5.3 `integrations/Pi05/smoke_generation.py` (447 lines) [V]
- Offline load (l.210-219): `cfg = PreTrainedConfig.from_pretrained(ckpt); cfg.device = "cpu"; cfg.compile_model = False; policy = get_policy_class(args.policy_type).from_pretrained(ckpt, config=cfg);
  policy = policy.to(dtype=torch.float32 if args.fp32 else torch.bfloat16).eval(); policy = policy.to(device); policy.config.device = device; pre, post = make_pre_post_processors(policy.config, args.checkpoint)`.
  `--fp32` help (l.182-186): "the pi05 port has a bf16 dtype clash in its flow path … fp32 is 16.6GB of WEIGHTS … use a >=24GB card".
- **Camera keys from the checkpoint, not the dataset**: `cams = [k for k in policy.config.input_features if "image" in k]` (l.228); missing ones zero-filled: `batch[k] = (item[k] if k in item else torch.zeros_like(item[cams[0]])).unsqueeze(0).to(device)` (l.256-260).
- Dataset frames: `ds = LeRobotDataset(repo_id, root=…, episodes=[ep]); item = ds[frame]` (l.253-254) — `item[k]` is float32 CHW in [0,1]; state `torch.as_tensor(np.asarray(item["observation.state"], dtype=np.float32)).unsqueeze(0)`; `batch["task"] = args.task` (l.255).
- `predict(batch)` (l.232-240): `policy.reset(); with torch.no_grad(): out = policy.predict_action_chunk(pre(batch)) if chunk_relative else policy.select_action(pre(batch)); return post(out).squeeze(0).float().cpu().numpy()`.
  Chunk-relative checkpoints need `import grabette_chunkrel.chunk_relative_processor` before `make_pre_post_processors` (l.197-208) and drop `AbsoluteFromChunkRelative*` post steps (l.223-225).
- `probe_task_sensitivity.py` is the remote (ficelle) twin: same idea over `open_client` (PEP-723 header pins lerobot `@e40b58a8`).

### 5.4 Existing visualisation conventions to follow [V]
- `integrations/DiffusionPolicy/offline_eval.py:165-200`: matplotlib `Agg`, `fig.savefig(out_dir / f"offline_eval_ep{ep:03d}.png", dpi=110)`, graceful "matplotlib not installed — skipping plots". Batch building at l.126-131 mirrors smoke_generation.
- `integrations/DiffusionPolicy/ood_check.py:45-60`: `FeatureExtractor` hooks `policy.diffusion.rgb_encoder` directly with the checkpoint preprocessor (`self.encoder = policy.diffusion.rgb_encoder`) — precedent for reaching into `policy.diffusion.*` from a tool.
- `integrations/DiffusionPolicy/vision_check.py` (docstring l.1-45): "Does the policy actually USE the image?" — image-swap / pixel-shift sensitivity probes; the attention tool is the natural complement (it should be listed next to it in the DP README table).
- `packages/grabette-postprocess/scripts/visualize/visualize_rgbd_trajectory.py`: rerun (`rr.init(app_id, spawn=True)`, `rr.set_time("time", timestamp=…)`, `rr.log("camera_feed", rr.Image(frame_disp))` after `cv2.resize(frame, (640, 400))`, `rr.SeriesLines`/`rr.Scalars`). If you want a rerun front-end, this is the house style (entity paths `world/...`, `camera_feed`, `imu/...`).
- `integrations/openarm/openarm_gripette_simu/examples/replay_dataset.py:55, 124`: `to_bgr()` helper "LeRobot video frame -> HWC uint8 BGR for cv2.imshow"; `openarm_gripette/examples/view_camera.py`: matplotlib QtAgg live view because lerobot pulls headless OpenCV (l.3-35).
- `cv2.applyColorMap`/`cv2.addWeighted` are **not used anywhere** in the repo yet (grep) — no existing heat-map overlay to match.

### 5.5 Existing mentions of "attention" / "saliency" / "grad-cam" / "heatmap" [V]
- `rtk proxy grep -rniE "attention|saliency|grad.?cam|heatmap"` over the worktree (py/md/yaml/toml/sh, excluding `.venv`/`.git`) -> **no matches** (exit 1). The only in-repo "attention" is
  inside `.venv` lerobot/transformers code. There is no prior art to extend; `vision_check.py`/`ood_check.py`/`probe_task_sensitivity.py` are the closest "is the policy looking?" tools.

---

## 6. Quick-reference numbers (Grabette defaults)
| | pi0.5 | Diffusion |
|---|---|---|
| input to model | letterbox 224x224 (pad top/bottom 28 px for 4:3), [-1,1] | Resize 236x236 (squash) -> CenterCrop 224, MEAN_STD |
| spatial grid | 16x16 patches (14 px), 256 tokens/cam | 7x7 (stride 32) ResNet18 layer4; 32 spatial-softmax keypoints |
| prefix | `[cam0 0..255][lang 256..455]` (1 cam, `empty_cameras=0`, 200 lang tokens) | `global_cond = [state, 64-D img feat] x n_obs_steps=2` |
| attention tensor | expert: `(B, 8, 50, 456+50)` per layer (18) per step (10); VLM prefill: `(B, 8, 456, 456)` x 18 | `pool.nets` logits `(N, 32, 7, 7)` -> softmax = keypoint heat maps |
| no_grad | `sample_actions`, `select_action`, `predict_action_chunk` (use `__wrapped__` or re-implement l.818-864) | `select_action`, `predict_action_chunk`; `generate_actions` free |
| dtype at eval | fp32 (repo casts; bf16 flow path broken) 16.6 GB | fp32 |
