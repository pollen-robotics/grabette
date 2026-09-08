# Design: attention maps and view ablation for GRABETTE policies

Date: 2026-09-08. Branch `attention-maps` (off `develop` @ d41ae9f).
Background and citations: `docs/attention_saliency_review.md`.
Verified internals: `docs/attention_maps_refs/repo_internals.md`.

## 1. Goal

When a policy misses a grasp, answer two questions offline:

1. **Where is it looking?** A spatial map, per camera, of the action tokens'
   attention over that camera's image patches.
2. **Which camera does it actually depend on?** A number per camera: how much
   the predicted action chunk changes when that view is removed.

The second exists because the first is not evidence. The review establishes
that on pi0.5 an interventional score is more faithful than attention, and that
saliency-style maps are hypothesis generators. Pairing a cheap map with a cheap
intervention is the smallest honest tool.

## 2. Decisions already taken

| Decision | Choice |
|---|---|
| v1 scope | Attention extraction + view ablation. No interventional saliency, no Grad-CAM. |
| Presentation | Shared computation, PNG writer in v1, rerun logger added afterwards against the same records. |
| Input | Dataset episodes first, `--dump_obs` captures behind the same interface. |
| Policy in v1 | pi0.5 only. The adapter interface is designed and tested for a second policy; the Diffusion implementation is v2. |
| Multi-camera | First-class. Nothing may assume one view. Acceptance criteria in §7. |

## 3. Non-goals

- Interventional saliency (v2), Grad-CAM or gradient attribution (cost, see the
  review), Diffusion Policy adapter (v2).
- Any runtime signal on the robot. This is offline analysis only. The
  failure-detection literature shows attention statistics lose to hidden-state
  probes and even those have unusable false-alarm rates.
- Remote inference. Ficelle's wire reply carries only actions and timing, so
  attention would need a server-side hook and a protocol bump. Out of scope.
- Any change to `lerobot`. Everything is hooks and thin re-implementation of a
  six-line wrapper.

## 4. Architecture

Four layers, so that the presentation choice and the input choice are both
swappable without touching the analysis.

```
frame sources          policy adapter           analysis            front ends
-------------          --------------           --------            ----------
DatasetFrames  ---\                                          /--- png_writer   (v1)
                   >--- Pi05Adapter ---> FrameAnalysis ------
DumpObsFrames  ---/     (DiffusionAdapter v2)                 \--- rerun_logger (next)
```

New workspace package `packages/grabette-attention`, mirroring
`grabette-chunkrel`: core depends on torch and numpy only; the PNG front end
needs matplotlib or opencv behind an import guard; rerun likewise. Consumed by
`integrations/Pi05` as a path dependency.

### 4.1 Frame sources

Both yield the same record: an ordered mapping of camera key to a full
resolution RGB frame, the state vector, the task string, and an episode and
frame index. Neither resizes or normalises; the policy's own preprocessing does
that, which is the property that makes the two sources interchangeable.

- **Dataset source**: `LeRobotDataset(repo_id, episodes=[ep])[frame]`, following
  the offline loading already in `integrations/Pi05/smoke_generation.py`.
- **dump_obs source**: reads `obs_XXXXX.png` plus `state.jsonl` from an episode
  subdirectory. Note the PNGs are written BGR by `cv2.imwrite`, so the reader
  converts back to RGB.

Frame selection: `--frames` accepts explicit indices, `grasp`, or a stride.
Default is `grasp`, because the literature reports the failure signal
concentrating there. `grasp` means the first frame whose gripper closure channel
crosses the closure threshold, read from the episode's own action or state
channels; when no gripper channel is present it falls back to an even stride
over the episode and says so in the summary rather than failing.

### 4.2 Policy adapter: the genericity boundary

Every model-specific fact lives behind this interface. Nothing above it knows
about tokens, patches or letterboxing.

- `camera_keys` -> ordered list, read from `config.image_features`.
- `token_layout` -> for each camera, its token slice, its grid shape, and the
  geometry needed to map a grid cell back to original pixels. **Derived at
  runtime, never constant.**
- `attention(frame)` -> per-camera spatial maps plus per-camera and language
  attention mass.
- `predict(frame, noise)` -> the baseline action chunk.
- `ablate(frame, camera, noise)` -> the chunk with that camera removed.

### 4.3 Analysis records

A plain data record per frame: the per-camera maps, the mass per camera and for
language, the ablation delta per camera, and the provenance needed to read it
later (checkpoint, layer aggregation, denoising step, noise seed). No plotting,
no file paths. Both front ends consume this, which is what makes "PNG now,
rerun later" cost nothing extra.

## 5. pi0.5 attention extraction

Verified facts this rests on. The fused training path that discards attention
weights is not used at inference. `sample_actions` forces eager attention on
both towers, and the stock Gemma attention module returns the probabilities as
its second output. So a forward hook suffices.

- **Hook**: `register_forward_hook` on each of the 18
  `gemma_expert.model.layers[i].self_attn`; the weights arrive as `output[1]`
  with shape `(batch, 8 heads, 50 action tokens, prefix_len + 50)`.
- **Bookkeeping**: each layer's hook fires once per denoising step, 18 per step
  and 180 per chunk. Each hook knows its own layer index and counts its own
  invocations, so the denoising step is the invocation count. Do not infer the
  layer from a global counter.
- **Slicing**: take `[..., :prefix_len]` for prefix-only attention. Camera `v`
  occupies `[v*tokens_per_image, (v+1)*tokens_per_image)`, reshaped to the grid.
  Language occupies the remainder. Masked blocks, meaning absent cameras and
  padded language, are excluded explicitly rather than plotted as attention on
  nothing.
- **Aggregation**: mean over heads and over the 50 action tokens. Default over
  layers is the mean of all 18, following the one paper that states its choice
  for pi0.5; per-layer maps stay available behind a flag, since one paper finds
  middle layers most meaningful and another shows individual expert heads carry
  distinct spatial roles.
- **Denoising step**: default is the last step, the only choice stated in the
  literature. `--denoise-step {last,first,mean,all}` because the prefix cache is
  computed once and reused unchanged by every step, so a per-step comparison is
  clean to interpret and has never been published.
- **Mass**: reported two ways. Fractions over the whole prefix, so language mass
  is visible next to the cameras and the numbers sum to one. And per-camera maps
  normalised for display. Keeping both avoids the trap of a map that looks
  confident because it was renormalised over a view the model barely used.
- **Letterbox**: our frames become 224 wide by 168 high inside a 224 square with
  28 pixels of padding top and bottom, extra pixel to the bottom. Only grid rows
  2 to 13 carry image content. Padding rows are dropped before upsampling, and
  the grid-to-pixel map is computed from the actual resize ratio.
- **Expectation setting**: pi0.5 is reported to spread attention broadly with
  low peaks, and action fine-tuning makes maps diffuse. A broad map is normal.
  The tool must not imply otherwise, which is another reason the ablation number
  carries the argument.

## 6. View ablation

The design point that makes this cheap and defensible: pi0.5 is trained with
missing-camera masking, so a removed view is in distribution, whereas a grey
rectangle is not.

- **Mechanism**: call `_preprocess_images(batch)` to get the image and mask
  lists, then for the ablated camera set its mask to zero and its pixels to the
  padding value, and run `sample_actions` on the mutated lists. This reproduces
  exactly how the model represents an absent camera.
- **Why not drop the batch key**: absent keys are appended *after* the present
  ones, which changes token order and therefore position ids. Mutating the mask
  in place preserves the layout, so maps stay comparable by token index between
  the baseline and ablated runs.
- **Determinism**: `sample_actions` takes an explicit `noise` argument, and
  `predict_action_chunk` forwards keyword arguments to it. So the baseline and
  every ablation share one pre-drawn noise tensor. This is mandatory: without it
  flow-sampling variance is confounded with the ablation effect, and the review
  found no paper that discusses seeding at all.
- **Metric**: the change in the predicted chunk, reported in millimetres for the
  translation channels, and additionally per axis. Per axis matters because the
  open question is vertical: if range perception depends on the one camera, the
  vertical component should dominate.
- **Self-test**: running the baseline twice with the same noise must give a
  delta of exactly zero. That is a correctness check, and it goes in the tests.

## 7. Multi-camera acceptance criteria

These are testable requirements, not aspirations.

1. Every map and every scalar is keyed by camera name, never by position.
   Outputs are mappings.
2. Tokens per image, grid shape, camera order and the pixel mapping are read
   from the model and its config at runtime. Hard-coding any of them fails a
   test.
3. Per-camera mass and per-camera ablation delta are first-class outputs, so
   "which camera does the policy rely on" is answerable without inspecting
   images.
4. Ablation works for any number of views.
5. Absent or padded camera blocks are excluded explicitly.
6. The interface admits a policy whose per-camera features come from a CNN
   rather than tokens, since the Diffusion adapter follows. When it does, note
   that our config shares one encoder across cameras and flattens views into the
   encoder batch, so per-camera attribution there must unflatten that index.

## 8. Outputs

PNG front end writes, per episode directory, one overlay per frame per camera
plus a summary. Follows house conventions: RGB in memory and BGR only at write
time, matplotlib on the Agg backend, and a graceful skip if the plotting
dependency is absent.

```
ep003/
  frame_00042_cam0_attn.png
  summary.txt
```

The summary carries the numbers, one row per camera: attention mass, ablation
delta in millimetres, and the per-axis breakdown. Language mass appears as its
own row.

The rerun logger, added next, logs the same records on a timeline with the
overlay under each camera entity and the masses and deltas as scalar series,
following the existing rerun visualiser's entity naming.

## 9. CLI

```
grabette-attn --checkpoint <path_or_repo_id> \
              --dataset <repo_id> --episodes 3 7 11 \
              --frames grasp \
              --out <dir>

grabette-attn --checkpoint <path_or_repo_id> \
              --dump-obs <dir>/ep003 \
              --out <dir>
```

Flags: `--denoise-step`, `--layers`, `--per-head`, `--no-ablation`,
`--fp32/--bf16`. Defaults follow the repo: fp32, because the bf16 flow path is
broken, and `compile_model` forced off so hooks are not swallowed by a graph.
Chunk-relative checkpoints need `grabette_chunkrel.chunk_relative_processor`
imported before the processors are built, as the existing scripts do.

## 10. Testing plan

Unit tests, no GPU, using a stub policy whose attention modules return known
weights.

- Token layout: index to camera, row and column round-trips for one, two and
  three cameras. A masked camera is excluded.
- Letterbox: corners map to the expected original pixels; padding rows are
  dropped; the ratio is derived, not assumed.
- Hook bookkeeping: 18 layers times 10 steps yields 180 captures with correct
  layer and step assignment.
- Aggregation: head and action-token means match a hand-computed reduction;
  prefix mass sums to one across cameras and language.
- Determinism: identical inputs with identical noise give a bitwise identical
  chunk and a zero ablation delta.
- Ablation semantics: a stub policy that ignores one camera by construction
  reports approximately zero delta for it and non-zero for the camera it uses.
- Units: the delta is returned in millimetres. Pin this with an explicit test,
  the way `test_step_error_is_returned_in_millimetres` pins the gate, because we
  have already shipped a metre-labelled-as-millimetre bug once.
- Front end: the writer produces one file per frame per camera and does not
  require a display; the rerun logger is import-guarded.

Mutation checks that must fail: hard-coding 256 tokens, hard-coding one camera,
dropping the noise argument so ablation becomes stochastic, using columns
instead of rows when reshaping the token grid, and renormalising mass per camera
where the prefix-wide fraction is reported.

One integration test behind a marker, run manually on the GPU box against a
real checkpoint, asserting shapes and that the mass fractions sum to one.

## 11. Risks and open questions

- **The maps may be uninformative.** pi0.5 attention is reported diffuse, and
  our models are action fine-tunes, which makes it more so. Mitigation: the
  ablation number does not depend on the map being crisp, so v1 still yields a
  result. If the maps are useless, that is itself a finding and argues for going
  straight to interventional saliency in v2.
- **One camera today.** With a single view the ablation reduces to "how much
  does it use vision at all", which overlaps with the existing
  `vision_check.py`. Still worth having, since it is measured per axis and the
  interesting comparison arrives with the second view.
- **Language mass interpretation.** The state is discretised into the language
  prompt rather than a separate token, so language mass mixes task text and
  proprioception. Splitting them by token position is possible and deferred.
- **Denoising step choice** is unpublished territory. Treat whatever we find as
  a measurement to record, not a settled default.

## 12. What success looks like

For a held-out sugar-cup episode at the grasp frame, the tool prints the
attention mass on each camera and the millimetre change in the commanded chunk
when each camera is removed, and writes an overlay showing where on the frame
the action tokens are looking. That is enough to distinguish the two hypotheses
in the review: attention sitting on the gripper rather than the target, versus a
single view that cannot resolve range.
