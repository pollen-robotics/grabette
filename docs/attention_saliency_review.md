# Attention maps and saliency maps as debugging tools for our policies

Status: **review / state of the art**. No code yet. Written 2026-09-08 on branch
`attention-maps` (off `develop` @ d41ae9f).

Purpose: decide what to build so that, when a policy misses a grasp, we can ask
*what was it looking at* and get an answer we are allowed to believe. The
immediate target is pi0.5 (`lerobot` `pi05`), the second is our Diffusion
Policy. **The design must stay generic over the number of camera views** — we
run one camera today (`cam0`), the Gripette stereo pair and its depth image are
available and will likely be added, so nothing may hard-code "one image".

---

## 1. Why we want this

Two open questions from the chunk-relative work motivated it:

- All three sugar-cup models tested — including the delta baseline — misjudge
  arrival by more than the 16 mm cube in about half of held-out episodes
  (pooled median 21.6 mm). That is representation-independent, so the suspicion
  is **range perception**: the policy may not be using the visual cue that
  fixes depth, or may be reading the gripper rather than the target.
- The offline gate can tell us *that* a model is wrong (per-step error, closure
  gap, never-close count) but not *why*. Eval loss ranks checkpoints badly; we
  need a diagnostic that looks at the input.

So the deliverable is a debugging tool, not a research contribution: given an
observation, show where in each camera image the policy's action tokens are
looking, and separately show which pixels the predicted action actually depends
on. The two are different things (§2 vs §3) and the literature is clear that
conflating them is the classic mistake.

---

## 2. Attention maps

### 2.1 What the 2026 VLA-diagnosis literature actually does

The recipe is remarkably uniform, and it is the *simplest* one: **raw
attention**, with

- **query** = the action tokens (for pi0.5, the noisy-action suffix tokens),
- **key** = the image-patch tokens,
- softmax weights averaged over heads and over the action tokens,
- reshaped to the patch grid and upsampled onto the frame.

Nobody in the VLA literature uses attention rollout, Chefer relevance, or GMAR.
Those remain ViT/VQA-classifier methods (§2.2).

Where the papers differ is the **layer aggregation**:

| Work | Layer choice |
|---|---|
| VLA Knows Its Limits (arXiv 2602.21445, pi0.5 + GR00T N1.5) | mean over all transformer blocks and heads |
| DTP (arXiv 2601.16065) | per-layer map weighted by that layer's share of total visual attention, then summed |
| Don't Blind Your VLA (arXiv 2510.25616, OpenVLA) | a single middle layer band, "layers 14-24, where vision-language fusion is most active" |

**Denoising step.** For a flow-matching head the map also depends on the
sampling step. Only one paper states its choice: *VLA Knows Its Limits* reads
attention at **the last sampling step**. VLA-Trace does not state it in the
text. **No paper compares maps across denoising steps** — so for us that is an
open knob, and worth a look, since our 10-step loop re-attends the cached
prefix at every step.

That same paper documents the token layout we need: in its pi0.5 setting,
vision tokens are indices 0-767 (three views x 256 tokens), language 768-967,
then the action tokens. This is exactly the structure that makes the
multi-camera question answerable (§4).

### 2.2 The methods the interpretability literature would recommend

They exist, they are better on classifiers, and none has been applied to a VLA.

- **Attention rollout / attention flow** (Abnar & Zuidema, ACL 2020,
  arXiv 2005.00928). Treat per-layer attention plus the residual identity as a
  graph and take the matrix product from input to the layer of interest.
  Gradient-free. Their own point is the one that should make us cautious about
  raw attention: after a few layers token identities are mixed, so a late-layer
  raw map is *not* an attribution to input pixels. Adapting it here means
  respecting PaliGemma's block-causal mask and deciding whether to roll through
  the prefix layers as well as the expert's.
- **Chefer et al., CVPR 2021** (arXiv 2012.09838) — LRP-style relevance
  combined with the gradient of the attention map w.r.t. a class score, then
  rolled out. **Chefer et al., ICCV 2021** (arXiv 2103.15679) drops LRP and
  keeps gradient-weighted attention with separate update rules for self- and
  cross-attention; this is the bi-modal version, with code
  (`hila-chefer/Transformer-MM-Explainability`).
- **GMAR** (arXiv 2504.19414) — weight heads by the norm of the gradient of the
  prediction w.r.t. each head's attention map before the rollout product.
  ViT classifiers only.

All three need a **scalar target** to differentiate. For a classifier that is
the class logit. For a flow-matching action expert there is no logit, so the
target is a design choice — a component of the predicted velocity, the norm of
the predicted delta, the gripper channel. That choice is ours to make and is
shared with the saliency side (§3), which is an argument for picking it once.

### 2.3 What these maps have shown about VLAs

This is the part that makes the tool worth building, because the findings are
directly about the failure mode we are chasing.

- **VLA-Trace** (arXiv 2605.30117, code Apache-2.0 at
  `github.com/VLA-Trace/VLA-Trace`) is the only open-source tool found that
  ships action-token to image-patch heatmaps **for pi0.5 and OpenVLA**, plus
  attention knockout and attention-IoU against simulator masks. Its headline
  comparison: pi0.5 "distributes attention broadly (higher mass, lower peak-hit
  rates)" while OpenVLA "concentrates it sparsely". Both route attention to
  **robot-object interaction regions**, not purely to the semantic target.
  Note for us: a broad, low-peak map on pi0.5 is *normal*, so a diagnostic that
  expects a crisp blob on the sugar cube will mislead.
- **Don't Blind Your VLA** (arXiv 2510.25616): the pretrained VLM has clear
  object-aligned attention; after action fine-tuning "the maps become diffuse,
  noisy, and weakly correlated with the target object" and "leak into
  irrelevant background regions or concentrate on distractor objects". Since
  every model we run is an action fine-tune of `pi05_base`, expect this.
- **DTP** (arXiv 2601.16065): failure episodes show significantly more
  attention on task-irrelevant regions than successes (p < 0.001), **peaking
  during the grasp phase**. This is the single most encouraging result for us:
  attention mass off-target is a measurable failure correlate, and it peaks
  exactly where our models fail.
- **Cloak** (arXiv 2606.22836) fine-tunes pi0.5 with the end-effector masked in
  the wrist view, on the argument that the gripper occupies a large, consistent
  region and invites a shortcut. Worth knowing that the paper gives **no
  attention evidence** for the shortcut — the transfer result is the argument.
  **Cross-View Action Consistency** (arXiv 2608.06965) is blunter: it keeps the
  wrist stream masked throughout "to prevent an unperturbed visual shortcut
  from confounding attribution to scene-camera variation". If we add a wrist
  view, this is a stated confound for any attribution we compute.
- **GuidedVLA** (arXiv 2605.12369) supervises three heads of **pi0's action
  expert** with grounding / phase / depth signals and shows head-level maps.
  Evidence that individual expert heads in exactly our architecture can be read
  as spatial maps — i.e. a per-head view may be more informative than the
  head-mean everyone plots.

### 2.4 Attention as a failure signal at inference time

Tempting, and mostly a trap on the evidence so far. Attention-derived detectors
exist (navigation-head entropy, arXiv 2603.13782: 44.6% detection at 11.7%
false positives; attention-entropy token selection, arXiv 2604.05323), but the
strong failure detectors in 2026 are **hidden-state probes**, not attention
(arXiv 2606.29699 reports AUROC 0.972 retrospectively on layer-16 MLP
activations, yet 3.32 false warnings per clean episode, and calls itself
"retrospective separability rather than prediction"). Recommendation: treat
attention as an *offline diagnostic*, and do not put it on the robot's critical
path as a runtime abort signal.

---

## 3. Saliency maps

Attention says where the model *routes information*. Saliency says what the
prediction *depends on*. For a debugging tool the second is the stronger claim,
and the literature says so explicitly for our exact architecture (§3.4).

### 3.1 The scalar-target problem, which is the whole design decision

Every attribution method needs a scalar `s(x)` to differentiate or perturb
against. A classifier hands you a class logit. We have a 50-step chunk in
R^(50x11), so we must choose. The options, and what each one asks:

| Target | Question it answers |
|---|---|
| one action dimension at one step, or summed over the chunk (e.g. the gripper channel, or delta-z) | which pixels drive *this* command |
| `\|\|a(x') - a(x)\|\|` against the unperturbed prediction | which pixels, if changed, change the action at all (the only sensible target for perturbation methods) |
| `\|\|a\|\|` or the sum of all outputs | class-agnostic map; components cancel, structure is hidden — avoid |
| the flow/diffusion loss at a fixed `(noise, t)` | one network evaluation instead of a sampler; a proxy |

For our failure mode the interesting targets are concrete: **the gripper
closure channel** (why did it close there) and **the vertical component of the
commanded delta** (the range-perception hypothesis). Picking per-dimension
targets and stacking the maps beats one summed map.

### 3.2 Gradient methods

- **Vanilla gradients** (arXiv 1312.6034) and **SmoothGrad** (arXiv 1706.03825):
  one and N backward passes. Nothing in either assumes a logit. Noisy at pixel
  level; gradients saturate where the output is locally flat even though the
  feature matters.
- **Integrated Gradients** (arXiv 1703.01365): path integral from a baseline,
  20-100 steps, and its *completeness* property is actually meaningful here —
  attributions sum to the change in the action between baseline and real image,
  so a signed per-dimension map tells you which pixels push the gripper closed
  versus open. Baseline choice dominates; a blurred baseline beats black.
  Captum takes a tuple of inputs and baselines, which is what we need per
  camera.
- **Grad-CAM family** (arXiv 1610.02391; ++ arXiv 1710.11063; **HiResCAM**
  arXiv 2011.08891; **LayerCAM**, IEEE TIP 2021): one forward and one backward
  pass, resolution limited by the layer grid. Two practical notes: Grad-CAM's
  spatial averaging can highlight regions where the gradient is zero, which is
  HiResCAM's whole argument, and the built-in `ReLU` keeps only what *increases*
  the target, so for a signed regression target either drop it or run twice
  with `s` and `-s`. LayerCAM works on earlier, higher-resolution layers, which
  matters because a 7x7 map cannot resolve gripper fingertips.
- **Sanity check that constrains our choice**: Adebayo et al. (arXiv 1810.03292)
  show Guided Backprop and Guided Grad-CAM are nearly independent of the trained
  weights — they are edge detectors that pass the eye test. Plain Grad-CAM
  survives the randomisation test. **Do not use guided variants**, and run the
  weight-randomisation check on whatever we build.
- **Through a flow-matching head**: no paper found derives gradient attribution
  through a flow sampler. Three options, increasing in cost: fix `(eps, t)` and
  target the one-step velocity or loss; backprop through all K steps with a
  fixed seed (10 steps at batch 1 is feasible with checkpointing); or target the
  model's `x_0` prediction at an intermediate `t`.

### 3.3 Perturbation methods

- **Occlusion** (arXiv 1311.2901), **RISE** (arXiv 1806.07421), **meaningful
  perturbation** (arXiv 1704.03296), **extremal perturbations** (arXiv
  1910.08485). All transfer to a chunk by using `||a(x_masked) - a(x)||` as the
  readout, which conveniently removes the need for an arbitrary baseline. Cost
  is 100s to 1000s of forward passes per frame per camera, batchable.
- The occluder is itself out of distribution: a grey square was never in
  training, and the action change it causes may say nothing about importance.
  **Blur fills are milder** (Greydanus) and dataset-mean fills milder still.
- **Seed fixing is mandatory and unremarked in the literature.** For a
  stochastic head, sampling variance is otherwise confounded with the occlusion
  effect. Nothing we found discusses it; treat it as a requirement we impose.
- Extremal perturbations give a nice framing for our case: fix the mask area
  and find "the 10% of pixels that suffice to reproduce the action".

### 3.4 What has been done on policies, and the one paper that matters most

- **Embodied Interpretability** (Zhang et al., arXiv 2605.00321, ICML 2026) is
  the only work found that does attribution **on pi0.5 with a continuous action
  chunk**, and it is close to a blueprint. Method: an *Interventional
  Significance Score* — Bernoulli-mask the visual tokens (p = 0.3), fill from a
  Gaussian-blurred copy of the image rather than zeroing, 100 Monte-Carlo
  interventions per frame, readout = **MSE between predicted action chunks**.
  It defines a **Nuisance Mass Ratio**, the attribution mass falling on
  task-irrelevant regions, and shows it **predicts generalisation failure under
  distribution shift**. It analyses three camera views separately (front,
  overhead, wrist) and finds wrist attribution concentrated on gripper and
  object while front and overhead stay diffuse. Crucially, its baselines
  include the attention score, and **the interventional score is more faithful
  than attention** — the "attention is not explanation" result, reproduced in
  our architecture.
- **VLA-Pruner** (arXiv 2511.16449) reinforces it from the efficiency side:
  pruning tokens by the VLM's *semantic* attention hurts manipulation, because
  the tokens the VLM attends to are not the tokens the action head needs. So
  SigLIP/Gemma-side attention is a poor proxy for action attribution in a
  two-tower pi0-style model.
- **PointMapPolicy** (arXiv 2510.20406) is the closest published Diffusion
  Policy recipe: Grad-CAM++ with **the diffusion loss as the target**, hooked at
  the final convolutional block, computed **separately per camera view**. It
  does not state whether noise and timestep are fixed.
- **Greydanus et al.** (arXiv 1711.00138) already used a vector-output
  delta-norm readout for Atari actors, so the formulation is old and reusable.
- **Atrey et al.** (arXiv 1912.05743, ICLR 2020) is the discipline we must
  adopt: they took hypotheses generated from saliency maps of RL agents,
  intervened on the game state, and found that in most cases the action did not
  change as the map implied. **Saliency maps are hypothesis generators, not
  explanations.** Their protocol is the one to implement: compute the map, form
  a hypothesis, mask that region and nothing else, re-run with the same seed,
  and compare against masking a random region of equal area. Only that last
  comparison turns a picture into evidence.
- **Shortcut precedent worth knowing**: Govi et al. (arXiv 2304.08230) used
  saliency to catch a 6-DoF pose regressor fixating on ArUco markers in the
  background rather than the object. We collect data around a ChArUco
  calibration board, so this failure class is not hypothetical for us.
- Spatial-softmax keypoints in Diffusion Policy's ResNet are themselves a
  zero-cost "where is the encoder looking" readout (PRISM, arXiv 2606.15232),
  worth plotting before building anything.

### 3.5 Libraries

- **Captum** (BSD-3, maintained) is the best fit: every method accepts a
  **tuple of input tensors** and returns a tuple of attributions, which is
  exactly the multi-camera shape, and regression targets are native (index into
  the output, or wrap a custom scalar in a `forward_func`). Has `Occlusion` and
  a SmoothGrad wrapper.
- **pytorch-grad-cam** (MIT, maintained) has the ViT `reshape_transform` and
  callable targets, but takes a single input tensor, so it needs one call per
  camera with the others held in a closure.
- **TorchRay** archived since 2021. **Zennit** is LRP/CNN-oriented; attention
  rules are not standard.
- Neither library backprops through a multi-step sampler. The wrapper we need
  is the same in both cases: `forward(images_dict) -> action_chunk` with the
  seed fixed inside, after which the policy is just a deterministic regressor.
  For the ISS/RISE-style token masking no library is needed at all — it is N
  forward passes and an MSE.

Full annotated bibliography with per-entry verification status:
`scratchpad/biblio_saliency.md`.

---

## 4. Multi-camera: the constraint, and an unpublished measurement

We will add views. The stereo pair and depth exist on Gripette, and the design
must not assume one image.

Two things follow from the survey.

**(a) The token layout makes per-view attribution trivial. Per-camera
*saliency* has been published once; per-camera *attention mass* has not.**
PaliGemma concatenates a fixed block of image tokens per view (256 for a 224 px
input at patch 14, i.e. a 16x16 grid). Given the layout — view *v* occupies
token indices `[256v, 256v+256)` — the **fraction of action-to-image attention
mass landing on each view** is a sum over that block.

On the saliency side this is already done: Embodied Interpretability (§3.4)
reports per-camera interventional attribution on pi0.5 across front, overhead
and wrist views, and PointMapPolicy (§3.4) shows per-camera Grad-CAM++ for a
diffusion policy. One caveat from the former: per-view masses are only
comparable if the *same* masking distribution and fill are used for every view.

On the attention side I found nothing — no paper reports per-camera attention
mass for pi0/pi0.5 or for OpenVLA-OFT's multi-image input. Existing multi-view
evidence is instead

- *behavioural*: LIBERO-Plus (arXiv 2510.13626) blacks out the third-person
  view and still gets 43.6-67.3% success from the wrist camera alone; all-black
  goes to ~0; camera-viewpoint shift is the most damaging perturbation of the
  seven axes (95% to under 30%);
- *architectural*: a learned Camera Router scoring each view from the prompt
  (arXiv 2602.15543); UniviewVLA (arXiv 2606.21501) picking "the most
  action-informative view" by **lowest mean action-token entropy**, training-free,
  re-evaluated every 30 steps, taking occluded-task success from 40.0% to 73.3%;
- *a training control*: view dropout (SkillMoV, arXiv 2606.17615, p=0.2,
  keep >= 2 views), and the masked-wrist controls above.

- *behavioural, on real policies*: a multi-policy surgical-suture study
  (arXiv 2605.28736) varied the camera set across ACT, Diffusion Policy,
  SmolVLA and pi0 and found the side camera supports depth along the approach
  axis while the on-arm camera supports lateral alignment — on-arm only gives
  40-65% depth error, side only 30-45% lateral error. That is the closest
  published answer to "which camera carries which error axis", and it is
  directly relevant to a range-perception failure.

So "which camera does the policy rely on" is answerable cheaply, and the field
has answered it mostly by ablation. Per-view attention mass is a one-line
reduction once the layout is known; per-view saliency mass follows the ISS
recipe; and **view ablation** is the behavioural cross-check that validates
both.

One important implementation detail for the ablation: for pi0.5 the cleanest
intervention is **dropping that camera's image tokens via the attention mask**,
not occluding its pixels. pi0/pi0.5 are trained with missing-camera masking, so
a dropped view is *in distribution* while a grey rectangle is not. No paper
found formalises this as an attribution score, which makes it both the cheapest
and the most defensible per-view measure available to us.

**(b) Genericity requirements this imposes on the design.** Concretely, and
these are the acceptance criteria for the implementation:

1. Every map is **keyed by camera key** (`observation.images.<name>`), never by
   position or a hard-coded single view. Output is a dict, not an array.
2. The token-to-pixel mapping is **derived at runtime** from the model's own
   image-token layout (tokens per image, grid side, view order as the policy
   assembles it), not from a constant. It must handle the letterbox: our
   480x360 frames become 224x168 padded to 224x224 with 28 px top and bottom,
   so only grid rows 2-13 carry image content, and the padding rows must be
   dropped rather than plotted as attention on nothing.
3. A **per-view scalar** (attention mass, and later saliency mass) is a
   first-class output alongside the spatial map, so the multi-camera question is
   answerable without eyeballing images.
4. **View ablation** is supported for N views, reported as change-in-action per
   ablated view.
5. Nothing in the interface assumes the number of views is 1, 2, or 3;
   `empty_cameras` padding, if any, must be excluded explicitly rather than
   plotted.
6. The same interface must accept the Diffusion Policy, whose per-camera
   features come from a CNN rather than tokens — i.e. the abstraction is
   "per-camera spatial map + per-camera scalar", not "attention".

---

## 5. Implementation notes for `lerobot` pi0.5 and Diffusion

_(pending: repo/tooling agent still running — attention hook points, whether
weights are discarded, prefix KV cache across denoising steps, gradient
availability under `no_grad`, Diffusion Grad-CAM layer, `--dump_obs` reuse.)_

---

## 6. Prior art we can start from

| Repo | What it gives | Models | Licence |
|---|---|---|---|
| `VLA-Trace/VLA-Trace` | action-token to patch heatmaps, attention knockout, attention-IoU, CKA | **pi0.5**, OpenVLA, OFT, X-VLA | Apache-2.0 |
| `hila-chefer/Transformer-MM-Explainability` | gradient-weighted relevancy, self- + cross-attention rules | LXMERT, VisualBERT, DETR, CLIP | needs adaptation |
| `vla-mech-interp/mechanistic-steering-vlas` | working hooks into **LeRobot pi0** internals (FFN value vectors, steering) | OpenVLA, pi0/pi0-FAST | not attention viz, but the hooks |
| `zjysteven/VLM-Visualizer` | LLM attention x ViT attention overlay recipe | LLaVA-style | closest generic VLM recipe |
| `sylvestf/LIBERO-plus` | perturbation benchmark incl. camera blackout / viewpoint shift | OpenVLA-OFT, pi0 | behavioural validation |
| `meta-pytorch/captum` | IG, occlusion, SmoothGrad, layer attribution; **tuple inputs = per-camera**, regression targets native | any | BSD-3, maintained |
| `jacobgil/pytorch-grad-cam` | CAM family + ViT `reshape_transform`, callable targets | any | MIT, maintained, one input tensor |
| `greydanus/visualize_atari`, `KDL-umass/saliency_maps` | perturbation saliency and the counterfactual critique protocol | RL | the methodology, not the code |

Neither `openpi` nor `lerobot` ships an attention-visualisation utility; expect
to hook the eager attention path ourselves. Full annotated bibliography with
verification status per entry: `scratchpad/biblio_vla_attention.md`.

---

## 7. Recommendation

### 7.1 Build both, but do not trust them equally

The literature is unambiguous on the ranking, and it was tested on our exact
model. Embodied Interpretability included the attention score as a baseline for
pi0.5 and found the **interventional score more faithful**; VLA-Pruner found
that the tokens the VLM attends to are not the tokens the action head needs.
So:

- **Attention maps are the cheap first look.** One instrumented forward pass,
  no target choice, no seed problem. They tell us where information is routed
  and, per view, how much. Treat every conclusion drawn from them as a
  hypothesis.
- **Saliency is the evidence.** Interventional (mask-and-remeasure) attribution
  with the action-chunk MSE readout is what the one pi0.5 paper uses, it needs
  no gradients through the flow sampler, and it survives its own faithfulness
  checks.
- **Neither is a picture we get to interpret freely.** Atrey's counterfactual
  discipline is not optional: mask the region the map points at, re-run with the
  same seed, and compare against masking a random region of equal area.

### 7.2 Proposed staging

1. **Attention extraction for pi0.5**, action tokens as queries, image tokens as
   keys, mean over heads and action tokens, per layer *and* aggregated. Report
   per-view attention mass alongside the spatial map. Sweep the denoising step
   once to see whether it matters — nobody has published this, and our 10-step
   loop re-attends the cached prefix at every step, so the answer is cheap and
   ours to find. Expect a broad, low-peak map: that is normal for pi0.5, and a
   diagnostic tuned to expect a crisp blob on the sugar cube will mislead.
2. **View ablation for N views**, via the attention mask rather than pixel
   occlusion, since missing-camera masking is in distribution for pi0.5. Report
   change-in-chunk per dropped view. This is the measure that will answer "does
   the second camera actually help" when the stereo view is added, and it is the
   cheapest thing in this document.
3. **Interventional saliency** following the ISS recipe: Bernoulli token masks,
   blur fill, fixed sampler seed, action-chunk MSE readout, ~100 passes per
   frame, per camera with an identical masking distribution across views.
   Then the **nuisance-mass** style summary: how much attribution falls outside
   the object and gripper region.
4. **Counterfactual check** as a first-class command, not an afterthought.
5. **Diffusion Policy**: Grad-CAM++ (or HiResCAM) on the last conv block with a
   fixed-`(noise, t)` loss target, per camera, following PointMapPolicy. Plot
   the spatial-softmax keypoints first — they are a free readout and may answer
   the question before we build anything.
6. **Sanity check**: run the weight-randomisation test from Adebayo et al. on
   whatever we ship, and never use guided variants.

### 7.3 The question we are actually trying to answer

The range-perception hypothesis is testable with this tool: if the models
misjudge arrival height, then either (a) attention and saliency at the grasp
phase sit on the gripper rather than on the target and its support surface,
which is the shortcut story and matches Cloak's premise, or (b) they sit on the
target but the map is depth-uninformative from a single view, which the view
ablation will show as a large dependence on the one camera we have and predicts
that the stereo pair will help. DTP's result — attention on task-irrelevant
regions is significantly higher in failures and **peaks during the grasp
phase** — says we should look at the grasp frames specifically, not at episode
averages.

### 7.4 Scope discipline

This is a debugging tool. It should be an offline analysis over held-out
episodes and over `--dump_obs` captures, not a runtime signal on the robot: the
2026 failure-detection literature shows attention statistics lose to
hidden-state probes, and even those report unusable false-alarm rates. Keeping
it offline also keeps it out of the control loop entirely, which is the right
place for anything experimental given the arm.
