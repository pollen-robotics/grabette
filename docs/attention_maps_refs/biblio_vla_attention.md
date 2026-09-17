# Annotated bibliography: visualising attention maps of VLA policies (pi0 / pi0.5, OpenVLA, Octo, RT-2)

Compiled 2026-09-08 from ~17 web searches + ~25 abstract/README fetches (arXiv, GitHub, CVF).
Verification legend:
- **[verified]** abstract or full-text HTML fetched and the statement is quoted/paraphrased from it.
- **[partial]** only search-snippet / abstract level; method details not confirmed.
- **[not found]** searched, nothing matching was located.

Nothing below is invented; where a search yielded nothing I say so.

---

## 0. Take-aways for a pi0 / pi0.5 (PaliGemma + action expert) implementation

1. The de-facto recipe used by the 2026 VLA-diagnosis papers is *raw attention*: take the **action-expert (noisy-action) tokens as queries**, the **image-patch tokens as keys**, softmax attention weights, **average over heads and over the action tokens**, and either (a) average over all layers (VLA Knows Its Limits, §1.6), (b) weight each layer by its total image-attention mass and sum (DTP, §2.3), or (c) pick a single middle/late layer (Don't Blind Your VLA: "middle layers 14-24" of OpenVLA's Llama-2). None of the VLA papers uses rollout / Chefer relevance / GMAR — those remain ViT/VQA-classifier methods.
2. **Denoising step**: the only paper stating it explicitly is *VLA Knows Its Limits* (arXiv 2602.21445), which reads attention at **"the last sampling step"** of pi0.5's flow-matching loop. VLA-Trace (2605.30117) does not state the step in the text I could fetch. No paper found that compares attention maps across denoising steps.
3. **Multi-camera**: no paper found that reports per-view attention mass for the PaliGemma "256 tokens per image, concatenated" layout. Existing multi-view evidence is *behavioural* (blacking out a camera, LIBERO-Plus), *architectural* (camera routers, view entropy), or a *training control* (masking the wrist stream). Per-view attention mass = summing the same action->image attention over each view's 256-token block is trivial to implement but I could not find it published.
4. **Open-source, VLA-specific**: **VLA-Trace** (github.com/VLA-Trace/VLA-Trace, Apache-2.0) is the only repo found that ships action-token->image-patch attention heatmaps for pi0.5 and OpenVLA (+OpenVLA-OFT, X-VLA adapters). Generic tools: hila-chefer/Transformer-Explainability, hila-chefer/Transformer-MM-Explainability, sehyeongjo/GMAR, zjysteven/VLM-Visualizer.

---

## 1. Attribution methods for a spatial map over image tokens

### 1.1 Raw attention (last layer, mean over heads) — baseline
- No single citation; this is the baseline in every paper below. Practical variants seen in VLA papers:
  - *query = action tokens, key = image tokens, mean over heads and action tokens, mean over all layers* — VLA Knows Its Limits (§1.6) **[verified]**.
  - *per-layer heatmap weighted by the layer's share of total visual attention, then summed over layers* — DTP (§2.3) **[verified]**.
  - *single middle layer* — Don't Blind Your VLA: "the strongest and most semantically meaningful attention patterns typically emerge in the middle transformer layers (layers 14-24), where vision-language fusion is the most active" (OpenVLA / Llama-2 7B, 32 layers) **[verified]**.
- Limitation (Abnar & Zuidema): after a few layers token identities are mixed, so a late-layer raw map is not an attribution to input pixels.

### 1.2 Abnar & Zuidema 2020 — Attention rollout / attention flow **[verified, abstract]**
- "Quantifying Attention Flow in Transformers", ACL 2020, pp. 4190-4197. https://aclanthology.org/2020.acl-main.385/ (arXiv 2005.00928)
- Method: treat per-layer attention (mean over heads, plus identity for the residual, re-normalised) as a graph; **rollout** = matrix product of the layer matrices from input to the layer of interest; **attention flow** = max-flow on the same graph. Both are post-hoc, gradient-free.
- Showed: raw attention is unreliable as an explanation past the first layers; rollout/flow correlate better with input-gradient ablation on BERT.
- Limitations: assumes all heads equally important and ignores value/MLP nonlinearity; flow is expensive (max-flow per token). Designed for encoder self-attention; for a decoder-only LLM with block-causal masks (PaliGemma) the product must respect the mask, and for an action-expert whose weights differ from the prefix one must decide whether to roll through prefix layers too.

### 1.3 Chefer, Gur, Wolf 2021 (CVPR) — "Transformer Interpretability Beyond Attention Visualization" **[verified, abstract]**
- CVPR 2021, pp. 782-791. https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html (arXiv 2012.09838). Code: https://github.com/hila-chefer/Transformer-Explainability
- Method: LRP-style relevance (Deep Taylor Decomposition) propagated through attention layers and skip connections; per layer, relevance is combined with the **gradient of the attention map w.r.t. the class score** (positive part, mean over heads), then rolled out like Abnar & Zuidema.
- Showed: better segmentation/perturbation scores than rollout, GradCAM, LRP on ViT / BERT classification.
- Limitations: requires a scalar target for the gradient (class logit); for a flow-matching action expert one must pick a scalar (e.g. a component of the predicted velocity, or its norm) — no VLA paper found that does this. Needs custom LRP hooks per layer type.

### 1.4 Chefer, Gur, Wolf 2021 (ICCV oral) — bi-modal / encoder-decoder extension **[verified, abstract]**
- "Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers", ICCV 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Chefer_Generic_Attention-Model_Explainability_for_Interpreting_Bi-Modal_and_Encoder-Decoder_Transformers_ICCV_2021_paper.html (arXiv 2103.15679). Code: https://github.com/hila-chefer/Transformer-MM-Explainability (examples for LXMERT/VisualBERT VQA, DETR, CLIP).
- Method: drops LRP; keeps only gradient-weighted attention (`grad * attn`, clamp >= 0, mean over heads) and defines update rules for self-attention and cross-attention relevancy matrices (R_tt, R_ii, R_ti, R_it), accumulated layer by layer. Directly applicable to a decoder that mixes text/image/action tokens.
- Showed: SOTA on VQA / DETR explanation benchmarks, "first method to explain prediction by any Transformer-based architecture".
- Limitations: still needs a scalar target and a backward pass; for pi0-style joint attention (prefix + suffix sharing one attention op but different weights) the cross-/self-attention bookkeeping must be adapted. No VLA application found.

### 1.5 GMAR 2025 — Gradient-Driven Multi-Head Attention Rollout **[verified, abstract + README]**
- Jo, Jang et al., arXiv 2504.19414 (Apr 2025), preprint. https://arxiv.org/abs/2504.19414 . Code: https://github.com/sehyeongjo/GMAR (no license file mentioned).
- Method: per layer, compute gradient of the prediction w.r.t. each head's attention map, take L1/L2 norm as a head-importance score, normalise across heads, use it as weights when aggregating heads *before* the rollout product (instead of the uniform mean of Abnar & Zuidema).
- Showed: consistently better than plain rollout on ViT classification/detection/segmentation.
- Limitations: ViT classifiers only; not evaluated on multimodal decoders or policies; head weights still depend on a scalar target.

### 1.6 Which tokens are the "query" and which denoising step, for an action-expert VLA
- **VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies** — Wang, Zhang, Yan, Kompella, Liu, arXiv 2602.21445 (v2 June 2026), preprint **[verified, HTML]**. https://arxiv.org/abs/2602.21445
  - Models: **pi0.5** (horizons 10 and 50) and GR00T N1.5.
  - Attention read at **"the last sampling step"**; post-softmax matrix S with **queries = action tokens**, keys = [vision, language, action]; **"averaged across all transformer blocks and attention heads"**. For their pi0.5 setting vision tokens are indices 0-767 (= 3 images x 256), language 768-967, rest actions — i.e. the per-view 256-token blocks are directly addressable.
  - Finding: "Intra-chunk actions consistently attend to the same vision-language tokens" (invariance within the chunk); later actions do not adapt. They use *action-to-action* attention (radial "action sinks", first turning point of the attention trajectory) to set the executed horizon at test time.
  - Limitations: no per-step (denoising) comparison; no explicit limitations section.
- **VLA-Trace** (see §2.1): "For each rollout step, we average action-query attention over visual patches to obtain an action-conditioned heatmap" — query = action tokens; layer and denoising step not stated in the fetched text **[verified quote, details missing]**.
- **DTP** (see §2.3): per action token, per layer, attention to visual tokens; layer-weighted sum **[verified]**.
- For **OpenVLA** (autoregressive), the query is simply the action-token positions of the Llama decoder (7 action tokens); for **OpenVLA-OFT** the parallel-decoded action tokens. For pi0 the natural choice is the 50 noisy-action suffix tokens (state token excluded). **No paper found** that averages over denoising steps or reports how the map changes with tau — this is an open point.

---

## 2. What VLAs actually attend to; attention for diagnosis / failure prediction

### 2.1 VLA-Trace: Diagnosing VLA Models through Representation and Behavior Tracing **[verified, HTML + README]**
- Shi et al. (12 authors), arXiv 2605.30117 (May 2026, rev. Aug 2026), preprint. https://arxiv.org/abs/2605.30117 . Code: https://github.com/VLA-Trace/VLA-Trace (Apache-2.0, released 2026-06-06). Project: https://vla-trace.github.io/
- Method: three stages — cross-modal / checkpoint-drift CKA; **attention knockout** (causal necessity of layer/modality paths); behavioural probes incl. **attention IoU** with simulator masks (Robot / Object regions), patch masking, semantic input editing. Attention tensors `[layers, heads, query, key]`, action-token queries, averaged, reshaped to patch grid (`plot-attention`).
- Findings: **"pi0.5 distributes attention broadly (higher mass, lower peak-hit rates), while OpenVLA concentrates it sparsely (higher peak-hit rates, lower mass and IoU)"**; both show "consistent overlap between high-attention patches and Robot+Object regions", i.e. attention is routed to **robot-object interaction regions**, not purely to the semantic target. OpenVLA has "broader vulnerable regions" under knockout; pi0.5 "routes action-critical visual information through a narrower bottleneck". VLAs "excel at visually grounded trajectory generation" but are "limited in fine-grained semantic following".
- Multi-view: pi0.5 run with multi-view images but no per-view attention breakdown reported.
- Limitations (theirs): knockout is a necessity probe only; IoU depends on simulator masks and patch resolution ("conservative localisation measure"); grounding probes confounded by whether the checkpoint still emits language.

### 2.2 Don't Blind Your VLA: Aligning Visual Representations for OOD Generalization **[verified, HTML]**
- Kachaev, Kolosov, Zelezetsky, Kovalev, Panov, arXiv 2510.25616 (Oct 2025), preprint. https://arxiv.org/abs/2510.25616 . Project: https://blind-vla-paper.github.io
- Method: probing + attention-map analysis of Qwen2.5-VL, Prismatic VLM, OpenVLA-7B before/after action fine-tuning; proposes VL-alignment losses during fine-tuning. Attention "for visual patch embeddings from the middle layers (14-24)".
- Findings: pretrained VLM has "clear and relevant object-aligned attention"; after action fine-tuning "the maps become diffuse, noisy, and weakly correlated with the target object", "leak into irrelevant background regions or concentrate on distractor objects"; alignment restores "crisp, object-centric attention maps".
- Limitations: fine-tuning only (no pretraining), LoRA, modest dataset diversity. Query token for the maps not stated in fetched text.

### 2.3 DTP: Distracting Token Pruning for VLA **[verified, HTML]**
- Li et al., arXiv 2601.16065 (Jan 2026), preprint. https://arxiv.org/abs/2601.16065
- Method: "For each action token and each layer, we extract the attention weights from the token to all visual tokens, producing a layer-specific heatmap. We then weight each layer's heatmap by the proportion of total visual attention in that layer, and aggregate across layers." Then prunes tokens outside the "important region".
- Findings: **failure episodes show significantly higher attention to task-irrelevant regions than successes (p<0.001) across SpatialVLA, Nora, UniVLA**, peaking during the grasp phase — direct evidence that attention mass on irrelevant regions is a failure correlate.
- Limitations (theirs): not tested on OpenVLA-OFT / pi0; important-region construction is attention-derived only; per-model tolerance hyper-parameter.

### 2.4 AVA-VLA: Active Visual Attention **[verified, HTML]**
- Xiao et al., arXiv 2511.18960, **CVPR 2026**. https://arxiv.org/abs/2511.18960
- Method: POMDP view; a recurrent belief state produces soft importance weights for visual tokens, applied as a soft attention-mask on all LLM layers (OpenVLA-OFT backbone). Real robot: one third-person + two wrist cameras, no special multi-view fusion.
- Findings: qualitative maps show "the vanilla OpenVLA-OFT baseline fails to locate the task-critical 'stove' switch" while AVA keeps stable focus using history. SOTA on LIBERO / CALVIN.
- Limitations: attention-map computation details not given in fetched text; no code link in the paper text fetched; efficiency analysis "preliminary".

### 2.5 ReconVLA: Reconstructive VLA as Effective Robot Perceiver **[verified, abstract]**
- Song, Zhou et al., arXiv 2508.10333 (Aug 2025), **AAAI 2026**. https://arxiv.org/abs/2508.10333 . Code: https://github.com/OpenHelix-Team/ReconVLA
- Method: implicit grounding — a diffusion transformer reconstructs the *gaze region* (target object crop) from the VLA's visual output tokens, forcing attention to concentrate on the manipulated object. 100k-trajectory pretraining set.
- Finding: motivated by "current VLAs struggle to allocate visual attention to target regions"; improved LIBERO / real-robot success. Attention-visualisation method not verified.

### 2.6 FocusVLA **[partial, abstract]**
- Zhang et al., arXiv 2603.28740 (Mar 2026), preprint. https://arxiv.org/abs/2603.28740
- Claims VLA performance is limited by *how* visual info is used, not representation quality; identifies "architectural bias causing visual oversight" (shortcut pathways), too many visual tokens diluting attention, irrelevant noise. Adds Modality Cascaded Attention (removes shortcuts) and Focus Attention (selects patches). 18 figures reportedly include attention analyses — not verified.

### 2.7 GuidedVLA: Plug-and-Play Action Attention Specialization **[verified, HTML]**
- Fudan / SJTU, arXiv 2605.12369 (May 2026), preprint. https://arxiv.org/abs/2605.12369
- Method: on **pi0's action expert**, three attention heads are supervised with auxiliary signals (object grounding via attention supervision, skill phase classification, depth features). Shows head-level attention visualisations ("object attention focuses on the manipulation target") and t-SNE of head features.
- Relevance: demonstrates that individual action-expert heads can be read/steered as spatial maps in pi0. Limitation (theirs): predefined factors.

### 2.8 Oat-VLA: Object-Agent-centric Tokenization **[partial, abstract]**
- Bendikas et al., arXiv 2509.23655, **CoRL 2025**. https://arxiv.org/abs/2509.23655
- Reduces visual tokens to object-centric + agent (gripper) tokens; 2x faster convergence than OpenVLA. Relevant as an inductive-bias statement that "the agent's own visual information" is what the policy needs; no attention analysis verified.

### 2.9 Cloak: Masking the End-Effector from the VLA **[verified, HTML]**
- Piseno et al. (Stanford TML), arXiv 2606.22836 (June 2026), preprint. https://arxiv.org/abs/2606.22836 . https://tml.stanford.edu/cloak/
- Method: pi0.5 fine-tuned on DROID-style data with the end-effector masked in the **wrist** view (rendered from known geometry; external camera not masked); enables zero-shot transfer to other grippers / a 5-finger hand.
- Note: **no attention/saliency evidence** of end-effector shortcut is given; the argument is "the end-effector occupies a large and consistent region of the wrist view" plus the transfer result as indirect validation. Limitations (theirs): two-tip skills only, gripper-style data, IK regularisation, transfer not lossless.

### 2.10 LIBERO-Plus: In-depth Robustness Analysis of VLAs **[verified, HTML]**
- arXiv 2510.13626 (Oct 2025), preprint. https://arxiv.org/abs/2510.13626 . Code: https://github.com/sylvestf/LIBERO-plus
- Behavioural (not attention-based) analysis over 7 perturbation axes. Key: models "may not fully utilize the language modality"; OpenVLA-OFT keeps high success with *blank* instructions; goal-replacement shows "fixed vision-action mappings". Camera-viewpoint change drops success from 95% to <30%. **Third-person view blacked out: 43.6-67.3% success from the wrist camera alone**; all-black -> ~0. Wrist camera explains illumination robustness.
- Relevance: gives the behavioural baseline against which a per-view attention-mass measure should be validated.

### 2.11 Attention / internal-state signals for failure detection at inference time
- **Your VLA Already Has Attention Heads For Path Deviation Detection** — Jeong et al., arXiv 2603.13782 (Mar 2026) **[verified, abstract]**. Navigation VLA; 3 "Navigation Heads" out of thousands; entropy of those heads gives a training-free deviation detector (44.6% detection, 11.7% FP), triggers an RL recovery policy; real robot. Limitation: modest detection rate; navigation, not manipulation.
- **VLA-InfoEntropy** — Liu et al., arXiv 2604.05323 (Apr 2026) **[verified, HTML]**. OpenVLA on LIBERO; text->vision attention from all layers/heads aggregated per visual token, Shannon entropy: "task-relevant visual tokens yield concentrated, low-entropy attention, whereas irrelevant tokens produce diffuse, high-entropy attention"; used for KV-cache token selection (1.53x speedup, 76.4% success). Not a failure detector per se but an attention-entropy recipe.
- **Early Warning Signals for OpenVLA Failure under Visual Distribution Shift** — Mahato & Ren, arXiv 2606.29699 (June 2026) **[verified, abstract]**. Linear probes on **layer-16 MLP activations** (not attention) of frozen OpenVLA, LIBERO-10 with occlusion; AUROC 0.972 retrospectively but 3.32 false warnings per clean episode; partial transfer to camera jitter (AUROC 0.689). Explicitly "retrospective separability rather than prediction".
- **Hide-and-Seek in Trajectories** — arXiv 2605.30834 (May 2026) **[partial]**. Contrastive inter/intra-trajectory objectives to localise failure-indicative actions from trajectory-level labels; SOTA multi-task failure detection under conformal prediction. Not attention-based.
- **VLA-FAIL** — arXiv 2606.21386 (June 2026) **[partial]**. Last-layer Mahalanobis distance on token features + action-chunk consistency. Not attention-based.
- **SAFECAST** — arXiv 2608.04246 (Aug 2026) **[partial]**. Hidden-state probes with contrast-set perturbations and calibration. Not attention-based.
- Snippet-level claim (unverified source, from a related VLM paper in the results): "spatial attention is likewise near chance" as a failure signal — i.e. the literature trend is that hidden-state probes beat simple attention statistics for failure prediction.

### 2.12 [not found]
- No paper titled or explicitly claiming "VLA attention is diffuse" as its thesis; the closest verified statements are VLA-Trace ("pi0.5 distributes attention broadly") and Don't Blind Your VLA ("maps become diffuse, noisy").
- No paper found that visualises attention for **Octo** or **RT-2** specifically (Octo is a small transformer with readout tokens; RT-2 is closed).

---

## 3. Multi-camera / multi-view attribution

### 3.1 [not found] Per-view attention mass on concatenated image-token blocks
- Searched for "per-view / per-camera attention mass", "view importance attention VLA", "wrist camera dominates attention". No paper found that reports the fraction of action->image attention landing on each camera's 256-token block for PaliGemma/pi0 or for OpenVLA-OFT's multi-image input. The token layout needed is documented (VLA Knows Its Limits: vision tokens 0-767 for 3 views in pi0.5), so the measurement is straightforward but unpublished as far as I could find.

### 3.2 Behavioural view-importance (blackout / ablation)
- **LIBERO-Plus** (§2.10) **[verified]**: third-person blackout -> 43.6-67.3% success, wrist-only; wrist camera drives illumination robustness; camera-viewpoint shift is the most damaging perturbation.
- **VLANeXt: Recipes for Building Strong VLA Models** — arXiv 2602.18532 **[partial, snippet]**: combining third-person + wrist "significantly improves performance", complementary geometric cues.

### 3.3 Learned view weighting / selection
- **Selective Perception for Robot: Task-Aware Attention in Multimodal VLA** — Son, Lee, Choi, Ko, Lim, arXiv 2602.15543 (Feb 2026) **[verified, abstract]**. A lightweight **Camera Router** predicts a relevance score per camera view from the text prompt + wrist image; low-utility views get attenuated compute; VLM-generated labels for training. Better efficiency and success on real tasks. Gives explicit per-view importance weights (learned, not attention-derived). Limitations not discussed.
- **UniviewVLA** — Xu et al., arXiv 2606.21501 (June 2026) **[verified, HTML]**. Agent-view + wrist-view; generates compressed auxiliary views (16 tokens each) with a world model; selects "the most action-informative view" as the one with **lowest mean action-token entropy**, re-evaluated every 30 steps, training-free. Occluded-task success 40.0 -> 73.3%. View importance measured through the *action* distribution, not attention.
- **SkillMoV** — arXiv 2606.17615 **[partial, snippet]**: stochastic **view dropout** (p=0.2, keep >= 2 views) to avoid reliance on a dominant view. Not a VLA (proficiency estimation) but the regularisation idea is transferable.

### 3.4 Multi-view representation / robustness training
- **Cross-View Action Consistency for Camera-Robust VLA Policies** — Huang et al., arXiv 2608.06965 (Aug 2026) **[verified, abstract]**. Flow-based VLA; action-equivalent scene-camera pairs from reset MuJoCo states; consistency loss on flow velocities. Methodologically relevant: **"the wrist stream remains masked throughout to prevent an unperturbed visual shortcut from confounding attribution to scene-camera variation"** — an explicit statement that the wrist view is a shortcut that must be controlled for when attributing to the scene camera. LIBERO-Plus 87.2%; real robot 53.3 -> 74.4% on held-out camera.
- **Multi-View Unified Camera Fields (MVUCF)** — Yang et al., arXiv 2608.01826 (Aug 2026) **[verified, abstract]**. Training-only geometry objectives (coordinate-query depth, cross-view correspondence) because "existing multi-camera VLAs usually concatenate view tokens, leaving action representations weak in metric depth and inconsistent across cameras"; RGB-only at deployment. No per-view attention analysis.
- **AVA-VLA** (§2.4) and **Cloak** (§2.9) both use 1 third-person + wrist cameras with plain token concatenation; neither reports per-view attention.

---

## 4. Open-source code

| Repo | What it gives you | Models | Notes |
|---|---|---|---|
| https://github.com/VLA-Trace/VLA-Trace **[verified README]** | `plot-attention` (action-token -> image-patch heatmaps), attention knockout, CKA, attention-IoU vs sim masks, LIBERO eval | **pi0.5, OpenVLA**, + OpenVLA-OFT, X-VLA adapters | Apache-2.0, released 2026-06-06. Best starting point. Denoising step / layer choice must be read from code. |
| https://github.com/hila-chefer/Transformer-MM-Explainability **[verified]** | Gradient-weighted relevancy for bi-modal / enc-dec transformers (self- and cross-attention rules) | LXMERT, VisualBERT, DETR, CLIP examples | Needs adaptation to a decoder-only LLM + action expert and a scalar target. |
| https://github.com/hila-chefer/Transformer-Explainability **[verified]** | LRP + grad-attention rollout for ViT/BERT | ViT, BERT | Classifier-oriented. |
| https://github.com/sehyeongjo/GMAR **[verified README]** | Gradient-weighted multi-head rollout | ViT classification/detection | No license file mentioned; ViT only. |
| https://github.com/zjysteven/VLM-Visualizer **[partial]** | Combines LLM attention (generated token -> image tokens) with ViT attention to overlay on the input image | LLaVA-style VLMs | Closest generic VLM recipe; would need PaliGemma/pi0 hooks. |
| https://github.com/junyangwang0410/Attention-LLaVA **[partial]** | Hot-pluggable LLaVA attention visualiser | LLaVA | |
| https://github.com/vla-mech-interp/mechanistic-steering-vlas **[verified README]** | CoRL 2025 "Mechanistic Interpretability for Steering VLAs": FFN value-vector projection, activation steering, LIBERO/LeRobot eval | OpenVLA-7B, pi0 / pi0-FAST (LeRobot) | **Not attention visualisation**, but has working hooks into LeRobot pi0 internals. |
| https://github.com/sylvestf/LIBERO-plus **[verified]** | Perturbation benchmark incl. camera blackout / viewpoint shift | OpenVLA-OFT, pi0 and others | For behavioural validation of view-importance claims. |
| https://github.com/OpenHelix-Team/ReconVLA **[verified link]** | Reconstructive grounding VLA | own model | |
| https://github.com/Physical-Intelligence/openpi , https://github.com/huggingface/lerobot (pi0 / SmolVLA) **[partial]** | Reference pi0/pi0.5 implementations | | **No built-in attention-visualisation utility found.** LeRobot pi0 uses a custom block-causal mask (issue #862 discusses flex-attention); whether `output_attentions`-style hooks are exposed was not verified — expect to hook the attention op manually (eager attention path). |
| GitHub searches "openvla attention map", "pi0 attention visualization", "lerobot attention", "vla explainability" | **[not found]** No dedicated standalone repo beyond VLA-Trace. Generic LLM heatmap tools (ronniross/attention-heatmap-visualizer, wln20/Attention-Viewer) are text-only. |

---

## 5. Sources consulted (all links above) — additional search hits not used as primary items
- Survey: "Vision-Language-Action Models: Concepts, Progress, Applications and Challenges" arXiv 2505.04769 (no attention-visualisation content verified).
- AimBot (CoRL 2025, arXiv 2508.08113): multi-view reticle overlays; **no attention analysis verified** despite an early search snippet suggesting OpenVLA-OFT attention heatmaps "upsampled like GradCAM" — I could not attribute that snippet to a specific paper; treat as **unverified**.
- AttenA+ (arXiv 2605.13548): velocity-based action reweighting, OpenVLA-OFT / FastWAM; not attention-visualisation despite the name.
- Where Reliability Lives in VLMs (arXiv 2605.08200), Visuals Lie, Consistency Speaks (arXiv 2606.17389): VLM (not VLA) mechanistic studies of attention vs reliability; not fetched.
