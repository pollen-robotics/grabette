# Annotated bibliography: input-saliency / attribution for vision-conditioned robot policies

Scope: debugging tool to see which pixels drive the predicted action, for (a) Diffusion Policy with ResNet encoder(s), one or several cameras, and (b) a transformer VLA (pi0.5: SigLIP + Gemma + flow-matching action expert) emitting a continuous 50-step action chunk.

Compiled 2026-09-08 from web search + arXiv abstract/HTML fetches. Verification legend:

- **[V]** abstract / HTML page fetched during this search, details quoted from it.
- **[S]** seen only in a search-result snippet; not opened.
- **[M]** classic paper, arXiv id / venue given from memory, not re-fetched this session (ids are the standard ones but double-check before citing).
- **[?]** claim not verified.

Throughout, "scalar target" means the quantity `s(theta, x)` you backprop or perturb against. For a continuous action chunk `a in R^{T x D}` the natural choices are: (i) `||a_pred - a_ref||` with `a_ref` the unperturbed prediction (a *self-referential* delta, the only choice that makes sense for perturbation methods), (ii) one action dimension summed over the chunk or at one step (e.g. gripper command, or delta-z), (iii) `||a||` or `sum(a)` (the "sum all outputs" trick; degenerate when components cancel), (iv) the diffusion / flow-matching training loss at a fixed `(noise, t)`.

---

## 1. Gradient-based methods and their adaptation to regression targets

### 1.1 Vanilla gradients (Simonyan, Vedaldi, Zisserman) [M]
- arXiv 1312.6034, ICLR 2014 workshop. "Deep Inside Convolutional Networks".
- Method: `|d s / d x|`, max over channels. One backward pass.
- Regression: trivially applicable; `s` is any differentiable scalar of the action chunk. Nothing in the method assumes a logit.
- Cost: 1 fwd + 1 bwd.
- Failure modes: extremely noisy at pixel level (ReLU/GELU shattered gradients), saturation (gradient ~0 where the output is flat even if the feature mattered), high sensitivity to input scale. For a diffusion/flow head, the gradient goes through the *sampler*: with K denoising steps you backprop through K network evaluations unless you use the one-step target (see 1.6). Gradient *w.r.t. the image* of a diffusion sample depends on the noise seed; fix it.

### 1.2 SmoothGrad (Smilkov et al.) [M]
- arXiv 1706.03825, 2017 (ICML workshop).
- Method: average vanilla gradients over N (~10-50) Gaussian-noised copies of the input, `sigma` ~10-20 % of input range.
- Regression: no change. Also works as a wrapper around IG or Grad-CAM (Captum `NoiseTunnel`).
- Cost: N fwd+bwd.
- Failure modes: choice of sigma; smoothing hides genuinely sharp features (thin gripper fingers). For a stochastic head, the input noise and the action-sampling noise compound; keep the sampler seed fixed so only the input noise varies.

### 1.3 Integrated Gradients (Sundararajan, Taly, Yan) [M]
- arXiv 1703.01365, ICML 2017.
- Method: path integral of gradients from a baseline `x'` (black / blurred / mean image) to `x`; satisfies completeness `sum(attr) = s(x) - s(x')`. Typically 20-100 steps.
- Regression: directly applicable and *completeness* is actually meaningful here: attributions sum to the change in action (e.g. change in gripper command) between baseline image and real image. Per action dimension you get a signed map (which pixels push the gripper to close vs open).
- Cost: M fwd+bwd (M = 20-300; convergence check via the completeness gap). With multi-camera input, each camera is a separate input tensor and gets its own attribution (Captum accepts a tuple of inputs and baselines).
- Failure modes: baseline choice dominates (black baseline -> attribution concentrated on bright pixels; blurred baseline is usually better for natural images); pixel-level noise; for a diffusion head the "baseline" also needs a fixed seed, otherwise `s(x')` is a different sample.

### 1.4 Grad-CAM / Grad-CAM++ / HiResCAM / LayerCAM [M]
- Grad-CAM: Selvaraju et al., arXiv 1610.02391, ICCV 2017. `L = ReLU(sum_k alpha_k A^k)`, `alpha_k = mean_{ij} d s / d A^k_ij` on a chosen conv layer (usually last ResNet block, 7x7 or 8x8 at 224-256 px input).
- Grad-CAM++: Chattopadhay et al., arXiv 1710.11063, WACV 2018. Pixel-wise positive weights from 2nd/3rd order derivative terms; better for multiple instances of the salient object.
- HiResCAM: Draelos & Carin, arXiv 2011.08891, 2020. Element-wise `ReLU(sum_k dS/dA^k * A^k)` (no spatial averaging); provably faithful for the last layer when the head is linear in the features. Same cost as Grad-CAM, generally preferable when you want "where does the gradient actually flow" rather than a smoothed heatmap.
- LayerCAM: Jiang et al., IEEE TIP 2021 (no arXiv id known to me). Uses `ReLU(grad)` element-wise as weights, so it works on *earlier, higher-resolution* layers where Grad-CAM breaks down. Useful because a 7x7 Grad-CAM cannot resolve gripper fingertips; LayerCAM on layer2/layer3 of a ResNet-18 gives 28x28 / 14x14 maps.
- Regression adaptation: all CAM variants only need `d s / d A`; `s` can be (i) one action dimension (gripper), (ii) `||a_pred - a_ref||^2` (for the *delta* interpretation; note the gradient at `a_pred = a_ref` is zero, so `a_ref` must come from a *different* forward pass, e.g. the unperturbed/previous frame, or use `||a||` instead), (iii) the sum of the chunk. The `ReLU` in Grad-CAM keeps only regions that *increase* s; for a signed regression target you usually want both signs, i.e. drop the ReLU or run twice with `s` and `-s`. pytorch-grad-cam's `targets` argument is any callable on the model output, so a `lambda out: out[..., gripper_idx].sum()` works out of the box [V, README].
- Diffusion Policy specifics: Diffusion Policy (Chi et al., arXiv 2303.04137, RSS 2023 / IJRR 2024 [M]) uses a ResNet-18 with GroupNorm and *spatial softmax* replacing global average pooling. Spatial softmax preserves the 2D layout, so Grad-CAM on the last conv block is meaningful, and the keypoint coordinates it outputs are themselves a coarse "where does the encoder look" diagnostic (see 3.4, Rethinking Implicit Spatial Representation). With `n_obs_steps=2` the observation stack is two frames per camera; attribute each frame separately.
- Grad-CAM on ViT / SigLIP tokens: the standard recipe (pytorch-grad-cam `reshape_transform`, [V]) hooks a transformer block output of shape `(B, N_tokens(+1 cls), C)`, drops the cls token if any, reshapes to `(B, C, H/p, W/p)` and runs Grad-CAM as if it were a conv layer. Which layer: the last block is customary for ViT classifiers (jacobgil's tutorial uses `blocks[-1].norm1`), but for a ViT the last block's tokens are already heavily mixed by attention, so heatmaps are blurrier and less local than for a CNN; empirically people try the last 2-4 blocks. For SigLIP in pi0/pi0.5 (no cls token, 224x224 / 14 px patches -> 16x16 grid of 256 tokens per camera) the reshape is a straight `view(B, 16, 16, C).permute(0, 3, 1, 2)`. Because the action expert reads the *Gemma* KV cache rather than the SigLIP tokens directly, one can also hook the Gemma hidden states at the image-token positions (they keep the per-camera token order) and reshape those. Alternative for transformers: Chefer et al., "Transformer Interpretability Beyond Attention Visualization", arXiv 2012.09838, CVPR 2021 [M] (gradient-weighted attention rollout / LRP-like relevance propagated through attention); Abnar & Zuidema attention rollout, arXiv 2005.00928, ACL 2020 [M]. Both are defined for a scalar target and therefore work for one action dimension; both need modification for the pi0.5 prefix/expert two-tower attention pattern.
- Cost: 1 fwd + 1 bwd for Grad-CAM/HiResCAM/LayerCAM; Grad-CAM++ same order.
- Failure modes: (a) resolution limited by the layer grid; (b) Grad-CAM's global-average weighting can highlight regions where the gradient is zero (HiResCAM paper's argument); (c) sanity checks (Adebayo et al., "Sanity Checks for Saliency Maps", arXiv 1810.03292, NeurIPS 2018 [M]) show Guided Backprop / Guided Grad-CAM are nearly independent of the trained weights, i.e. edge detectors; plain Grad-CAM passes the randomisation test, guided variants do not — avoid guided variants for debugging; (d) for a stochastic head the gradient depends on the sampled noise; average over a few seeds or fix one.

### 1.5 "Sum all outputs" Grad-CAM for dense / vector regression [S]
- Searched: "Grad-CAM regression continuous output"; the operator-learning paper arXiv 2607.02203 [S] and several engineering blogs describe defining the scalar target as the sum (or L2 norm) of all predicted values and reporting the result as a "class-agnostic" map. Works, but cancellations between components hide structure; prefer per-dimension or per-step targets and stack them.
- Related: "Uncovering the Background-Induced bias in RGB based 6-DoF Object Pose Estimation" (Govi et al., arXiv 2304.08230, 2023, EURASIP JIVP) [V] — uses saliency maps on a 6-DoF *regression* network to show it fixates on ArUco markers in the LINEMOD background rather than the object: a clean example of saliency exposing a shortcut in a continuous-output vision model. The abstract does not state the scalar target used [?].

### 1.6 Attribution for diffusion / flow-matching heads (how to define the target)
- No paper found that formally derives gradient attribution for a flow-matching *action* head (2023-2026). Practical options, in increasing cost:
  1. **One-step / loss target**: fix `(eps, t)` and use the denoising or flow loss `||v_theta(a_t, t, x) - (eps - a_0)||^2` or the network output `v_theta` itself as `s`. One network evaluation, no sampler. This is what PointMapPolicy does with Grad-CAM++ (see 3.3). Interpretation: "which pixels change the predicted velocity/noise for this noised action" — proxy for the action, valid to the extent the first step determines the sample.
  2. **Full sampler with fixed seed**: run the K-step sampler with a fixed Gaussian seed, define `s` on the final chunk, backprop through all K steps (memory ~K x network). For pi0.5 (10 flow steps) and Diffusion Policy (DDIM 16 steps or DDPM 100) this is feasible on a single GPU at batch 1 with gradient checkpointing.
  3. **Denoised-mean target**: use the model's `x_0` prediction at one intermediate `t` (e.g. `t=0.5`) as a cheaper proxy for the sample.
- Related but not robotics: DAAM (Tang et al., "What the DAAM: Interpreting Stable Diffusion Using Cross Attention", arXiv 2210.04885, ACL 2023) [M] — attribution in a diffusion model via aggregated cross-attention maps; the analogous trick for pi0.5 would be to aggregate the action-expert -> image-token attention over the flow steps.
- Whether papers fix the noise seed: the Embodied Interpretability paper (3.2) sidesteps it by comparing *action means* (MSE between predicted chunks) [V]; PointMapPolicy does not state it [V, "not specified"]. In general the diffusion-attribution literature does not discuss this; **treat seed-fixing as a required but unverified-by-literature practice** [?].

---

## 2. Perturbation-based methods

### 2.1 Occlusion sensitivity (Zeiler & Fergus) [M]
- arXiv 1311.2901, ECCV 2014.
- Method: slide a grey/blur/mean patch over the image; map = change in `s`. Model-agnostic.
- Regression: use `s = ||a_pred(x_masked) - a_pred(x)||` (chunk norm) or one dimension; the *reference* is the unperturbed prediction, so no arbitrary baseline logit is needed. For a stochastic head **you must fix the sampler seed** across the masked and unmasked passes, otherwise the sampling variance is confounded with the occlusion effect. Report `E_seed[...]` if the policy is multimodal.
- Cost: `(H/stride)*(W/stride)` forward passes, e.g. 16x16 grid at 224 px with stride 14 = 256 passes *per camera* per frame; batchable. For a full K-step sampler this is 256 x K network evaluations.
- Failure modes: patch size trades resolution vs. effect size; the occluder itself is out-of-distribution (grey square never seen in training) and may produce large action changes unrelated to "importance" (Fong & Vedaldi's "artifacts" argument); blur occluders (Greydanus) are milder.

### 2.2 RISE (Petsiuk, Das, Saenko) [M]
- arXiv 1806.07421, BMVC 2018.
- Method: N random binary low-res masks (Bernoulli p ~0.5, upsampled, random shift); saliency = mask-weighted average of `s(x * m)`. Black-box, only needs forward passes.
- Regression: original uses class probability; for a continuous output replace with `s = -||a(x*m) - a(x)||` (masks that *preserve* the action indicate unimportant pixels) or with the one-dim signed change. This is exactly what the ISS estimator in 3.2 does on *tokens* for pi0.5 (Bernoulli masks on visual tokens, action-MSE readout, N=100).
- Cost: N = 1000-8000 forward passes in the original for ImageNet; robotics papers use N ~100 with a 16x16 token grid. Fix the sampler seed.
- Failure modes: variance at low N (speckle), mask-resolution bias, multiplicative masking to zero is again OOD (use blur or dataset-mean fill instead).

### 2.3 Meaningful perturbation (Fong & Vedaldi) [M]
- arXiv 1704.03296, ICCV 2017. "Interpretable Explanations of Black Boxes by Meaningful Perturbation".
- Method: optimise a soft mask `m` so that blurring/noising the region `1-m` maximally *drops* `s` (deletion game) while keeping the mask small and smooth (TV + L1 penalties). Gradient-based on the mask, ~300 iterations.
- Regression: `s = ||a(phi(x, m)) - a(x)||` or one dimension; the optimisation finds the smallest region whose removal changes the action by a threshold. Good fit for the counterfactual question "what minimal region must I hide to change the gripper command".
- Cost: ~300 fwd+bwd (per camera or jointly over cameras with a mask per camera). For a K-step sampler multiply by K, or use the one-step target.
- Failure modes: adversarial masks (the optimiser finds a scribble that fools the network rather than a semantically meaningful region), hyperparameter sensitivity (lambda_1, TV weight, blur sigma); handled partially by extremal perturbations below.

### 2.4 Extremal perturbations (Fong, Patrick, Vedaldi) [M] and TorchRay
- arXiv 1910.08485, ICCV 2019. Fix the mask *area* `a` (e.g. 5 %, 10 %, 20 %) and a smoothness via a parametric smooth-max mask, then find the region of that area that *maximally preserves* `s`. Removes the L1/TV hyper-parameters; gives a monotone family of regions.
- Regression: preservation objective becomes `-||a(x*m) - a(x)||`; per area you get "the 10 % of pixels that suffice to reproduce the action". Nicely suited to a "sufficient region" counterfactual test.
- Cost: similar to 2.3 (few hundred iterations per area); implemented in TorchRay (see 5.3, archived).
- Failure modes: still an optimisation with local minima; area sweep multiplies cost; results depend on the perturbation operator (blur vs. fade).

### 2.5 Cost-vs-fidelity references
- Hooker et al., ROAR "A Benchmark for Interpretability Methods in Deep Neural Networks", arXiv 1806.10758, NeurIPS 2019 [M]: remove-and-retrain shows most gradient methods barely beat random; SmoothGrad-Squared and VarGrad ensembles do best. Expensive (retraining) — not a debugging tool, but a warning that pixel-level gradient maps are weak evidence.
- Petsiuk (RISE) deletion / insertion AUC curves [M]: cheap fidelity metric for a fixed model; for regression use action-error-vs-fraction-removed curves.
- Huber, Limmer, Andre, "Benchmarking Perturbation-based Saliency Maps for Explaining Atari Agents", arXiv 2101.07312, 2021-22 [V]: compares five perturbation methods (incl. Greydanus blur, RISE, SARFA-style) on DRL agents with (i) parameter-randomisation sanity checks and (ii) input-degradation fidelity; finds no universal winner and fixes a bug in one method; recommends choosing by context. Closest published cost/fidelity study for *policies* rather than classifiers.
- Ancona et al., "Towards better understanding of gradient-based attribution methods", arXiv 1711.06104, ICLR 2018 [M]: unifies Gradient*Input, IG, DeepLIFT, LRP-eps; useful for choosing which gradient method is actually different.

---

## 3. Robotics-specific

### 3.1 Perturbation saliency for RL policies and its critique
- **Greydanus, Koul, Dodge, Fern, "Visualizing and Understanding Atari Agents"**, arXiv 1711.00138, ICML 2018 [V]. Blur a Gaussian-shaped region at each location (stride 5 px), saliency = `0.5 * ||pi(x') - pi(x)||^2` for the actor and the same for the critic value — i.e. already a *vector-output* delta-norm formulation, directly reusable for an action chunk. Cost: one forward per location (~hundreds per frame). Code: github.com/greydanus/visualize_atari. They also compare against Jacobian (vanilla-gradient) saliency and find the perturbation maps far more interpretable. Failure modes: blur is a mild perturbation, so small objects with high contrast dominate; frame-stack inputs require perturbing all stacked frames.
- **Atrey, Clary, Jensen, "Exploratory Not Explanatory: Counterfactual Analysis of Saliency Maps for Deep RL"**, arXiv 1912.05743, ICLR 2020 [V]. Tests hypotheses generated from Jacobian, perturbation (Greydanus) and object-based saliency by *intervening on the game state* (moving/removing the object the agent supposedly attends to) and checking whether the action changes as the map implies; in most cases it does not. Conclusion: saliency maps are hypothesis generators, not explanations; every claim needs a counterfactual test. Code: github.com/KDL-umass/saliency_maps. Directly relevant methodology for the "mask the salient region and re-run" step.
- Puri et al., SARFA "Explain Your Move: Understanding Agent Actions Using Specific and Relevant Feature Attribution", arXiv 1912.12191, ICLR 2020 [M]: perturbation saliency that scores *specificity* (change in the chosen action's Q) balanced against *relevance* (KL of the other actions), reducing the "everything is salient" problem of raw delta-norms. Regression analogue: weight the change in the *commanded* dimension against changes in the others.

### 3.2 Saliency for VLA policies (transformer, continuous chunk)
- **Zhang et al., "Embodied Interpretability: Linking Causal Understanding to Generalization in Vision-Language-Action Models"**, arXiv 2605.00321, ICML 2026 [V, HTML read]. The only paper found that does attribution *on pi0.5 with a continuous action chunk*. Method: **Interventional Significance Score (ISS)** — Bernoulli-mask visual tokens (p=0.3), replace masked tokens' pixels by a Gaussian-blurred version of the image (Greydanus-style, not zeroing), N=100 Monte-Carlo interventions per frame, read-out = **MSE between predicted action chunks** ("Action MSE as a computationally efficient proxy for distributional divergence"), attribution per token = expected action change when that token is masked. Also defines **Nuisance Mass Ratio (NMR)** = attribution mass on task-irrelevant regions, and shows NMR predicts generalisation failure under distribution shift. Baselines: attention score (ATT, attention mass received by each visual token in the SigLIP encoder), token norm (NORM), gradient saliency. Cameras: three RLBench views (front, overhead, wrist) analysed separately; wrist attribution concentrates on gripper/object, front and overhead are diffuse. Faithfulness: structured perturbations (texture / geometric / patch) of nuisance regions, Pearson correlation between saliency change and action MSE. Cost: 100 forward passes of pi0.5 per frame (the paper does not say whether the flow sampler seed is fixed; comparing means partly hides this [?]). Failure modes: token-level (16x16) resolution; attention-score baseline is shown to be *less* faithful than the interventional score, echoing the NLP "attention is not explanation" literature.
- **Liu et al., "Bridging the Semantic-Action Gap in Visual Token Pruning for Efficient VLA Inference" (VLA-Pruner)**, arXiv 2511.16449, 2025-26 [V]. Not an interpretability paper, but it ranks visual tokens by "temporally smoothed action relevance" (attention from the action-decoding side) in addition to prefill semantic attention, across several VLA architectures, and shows that pruning by *semantic* attention alone hurts manipulation — i.e. the tokens the VLM attends to are not the tokens the action head needs. Useful as evidence that VLM-side attention maps are a poor proxy for action attribution in pi0-style two-tower models. Exact scoring mechanism not read [?].
- VLA-InfoEntropy, arXiv 2604.05323 [S]: uses entropy of vision-attention distributions in a VLA at inference; tangential.
- "What Frozen VLAs Already Know About Success" (probing), arXiv 2605.28527 [S]; OpenVLA hidden-state linear probes for symbolic state [S]: internal-representation interpretability, not input saliency.

### 3.3 Saliency for Diffusion Policy / BC policies
- **Jia et al., "PointMapPolicy: Structured Point Cloud Processing for Multi-Modal Imitation Learning"**, arXiv 2510.20406 (v3 Jan 2026) [V, HTML read]. Diffusion-based policy, ConvNeXt-v2 encoder. "We apply Grad-CAM++ using the **diffusion loss as the target signal**", hooked at "the final convolutional block before normalization", computed **separately for each of three camera views** (static left, static right, wrist) and for each modality (RGB, XYZ). Finding: activations "commonly focus on the robot gripper, the manipulated object, or the goal location, depending on the modality and perspective". Does not state whether noise / timestep are fixed [?]. This is the closest published recipe to "Grad-CAM on a Diffusion Policy encoder": one network evaluation, loss target, per-camera maps.
- **Yardi, Biruduganti, Ankile, "Bridging the Sim2Real Gap: Vision Encoder Pre-Training for Visuomotor Policy Transfer"**, arXiv 2501.16389, 2025 [V, HTML read]. 23 frozen encoders scored by an "Action Score" (how well a small head regresses actions) and a "Domain Invariance Score"; Grad-CAM used as supporting evidence: the best encoder (MCR) "has a sharp focus on the robot end-effectors and the objects", the worst (DINOv2-B) "does not have a clear focus". Layer and scalar target not specified [?]. Evidence that focus-on-gripper-and-object in Grad-CAM correlates with downstream action quality.
- **Chen et al., "Rethinking Implicit Spatial Representation in Visuomotor Policy Learning" (PRISM)**, arXiv 2606.15232, 2026 [V, abstract only]. Analyses the spatial-softmax pooling used by Diffusion Policy's ResNet; uses "saliency analysis" to show the encoder focuses on task-relevant regions (a search snippet adds: "localized and consistent saliency around task-relevant objects, the gripper, and robot-object interaction regions" [S]). Method for the saliency maps not read [?]. Relevant because spatial-softmax keypoints are themselves a built-in, zero-cost attention read-out for Diffusion Policy.
- Dunion & Albrecht, "Multi-view Disentanglement for RL with Multiple Cameras", arXiv 2404.14064, RLC 2024 [V, abstract]. Search snippet [S] reports per-camera saliency maps showing the shared representation attends to goal positions visible in all cameras while the first-person private representation highlights the end-effector. Abstract does not mention saliency; method unverified [?].
- Nothing found that applies attribution specifically to ACT (Zhao et al. 2023). ACT's transformer decoder cross-attends from action queries to image tokens, so the raw cross-attention maps per query (per future timestep) are a free, if unfaithful, visualisation [?, not found in literature].

### 3.4 Shortcut learning / causal confusion and how saliency is used to detect it
- de Haan, Jayaraman, Levine, "Causal Confusion in Imitation Learning", arXiv 1905.11979, NeurIPS 2019 [M]. Canonical statement: BC policies latch onto nuisance correlates (the brake indicator light) — more information can hurt. Detection is by *intervention* (environment interaction), not saliency; motivates counterfactual checks.
- Sun et al., "Artificial Foveated Perception for Mitigating Shortcut Learning in Robotic Foundation Models", arXiv 2607.10655, 2026 [V]. States the shortcut problem for VLA / world-action models ("policies often fail to distinguish causally relevant visual structure from spurious scene-level correlations") and mitigates it with task-conditioned masks over objects / robot / action-critical regions used as a grounding signal during fine-tuning. Detection method is not saliency; but the masks it needs are exactly what a saliency audit would compare against (a natural "nuisance mass" metric as in 3.2).
- CIVIL: Causal and Intuitive Visual Imitation Learning, arXiv 2504.17959 [S]; "Initial State Interventions for Deconfounded Imitation Learning", arXiv 2307.15980 [S]; "Fidelity-Aware Data Composition for Robust Robot Generalization", arXiv 2509.24797 [S]: all address spurious features in visual IL through data/intervention, not saliency; not opened.
- Govi et al. 2304.08230 (see 1.5) [V]: saliency exposing an ArUco-marker shortcut in a pose regressor — same failure class (background fiducials) that a data-collection rig with calibration boards can induce.
- Counterfactual check protocol used across the above (Atrey 2020; Embodied Interpretability 2026): (1) compute map, (2) form a hypothesis ("the policy uses the gripper / the board"), (3) mask, blur or in-paint that region *and nothing else*, (4) re-run with the same seed, (5) measure action delta vs. masking a random region of equal area. Only step 5 turns a map into evidence.

---

## 4. Multi-camera attribution

- **Per-view attribution mass**: Embodied Interpretability (3.2) [V] is the only paper found that reports per-camera attribution for a VLA (front / overhead / wrist on pi0.5): wrist attribution is concentrated on manipulation components, front and overhead diffuse. Summing normalised attribution per view gives a "which camera does the policy rely on" number, but note that with the ISS / RISE readout the masses are only comparable across views if the same masking distribution and fill are used for every view.
- **PointMapPolicy** (3.3) [V] shows Grad-CAM++ per camera (two static + wrist) for a diffusion policy, qualitatively only.
- **View ablation as the cheap attribution**: Wang et al., "Imitation Learning for Robot Assistance in Open Surgery: A Multi-Policy Evaluation on Suture Following", arXiv 2605.28736, 2026 [V]. Compares ACT, Diffusion Policy, SmolVLA and pi0 (28 models, 32 configurations) with camera viewpoint as an explicit variable; search snippet [S] reports that removing either camera degrades all policies, with the side camera supporting depth along the approach axis and the on-arm camera lateral alignment (on-arm only: depth errors 40-65 %; side only: lateral errors 30-45 %). Whether the ablation is train-time or test-time is not stated in the abstract [?]. Closest thing found to a measured "which camera does the policy rely on, per error axis".
- Dunion & Albrecht 2404.14064 (RL, RLC 2024) [V]: train multi-camera, test single-camera; the MVD auxiliary loss makes the agent robust to dropping views — a training-side counterpart.
- Cheap practical view-attribution (not from a paper, standard practice): (i) replace one camera's image by its dataset mean / a blurred copy / a frame from another episode and measure `||delta a||` over a validation set; (ii) shuffle one camera across the batch (destroys the correlation with the label while staying in-distribution); (iii) for pi0.5, drop the corresponding 256 image tokens via the attention mask (the model is trained with missing-camera masking, so this is *in-distribution*, unlike pixel occlusion). No paper found that formalises (iii) as attribution [?].

---

## 5. Libraries

| Library | License | Status (as of 2026-09) | Multi-image inputs | Transformer token grids | Regression targets |
|---|---|---|---|---|---|
| Captum (meta-pytorch/captum) | BSD-3-Clause [V] | maintained (~5.7k stars, active issues/PRs [V]; release version not read [?]) | **Yes**: every method accepts a tuple of input tensors and returns a tuple of attributions; baselines also a tuple [V, docs] | Via `LayerGradCam` / `LayerIntegratedGradients` on any module; you reshape the token output yourself (no built-in reshape helper) | **Yes**: `target=None` for scalar output, or an int / tuple index into an `N x T x D` output; for a custom scalar wrap the model in a `forward_func` that returns it [V, FAQ] |
| pytorch-grad-cam (jacobgil; PyPI `grad-cam`) | MIT [V] | maintained, ~13k stars [V]; last commit date not read [?] | Batch of images yes; *multiple distinct image inputs* not directly (`input_tensor` is one tensor) — wrap the policy so the second camera is a closure/attribute and call once per camera | **Yes**: `reshape_transform` argument, tutorial for ViT / Swin [V] | **Yes** in practice: `targets` is any callable on the model output ("filter it out for the specific scalar output we want to explain") [V]; no explicit regression docs |
| TorchRay (facebookresearch) | CC-BY-NC 4.0 [M, ?] | **archived / read-only since 2021-09-08** [V] | Single image tensor | No | Objective is a class index; would need patching |
| Zennit (chr5tphr) | LGPL-3.0 [M, ?] | maintained; v1.0.0 released July 31 (year not shown on page, likely 2025 [?]), "support for multiple inputs, keyword arguments, and multiple outputs for rules" [V] | Yes since 1.0.0 [V] | LRP rules for attention are not standard; Zennit targets CNN/MLP LRP | Yes: relevance is initialised from any output tensor / one-hot-like selector |

Notes:
- Captum has `Occlusion` (sliding window, multi-input) and `NoiseTunnel` (SmoothGrad / VarGrad wrapper), plus `Lime` / `KernelShap` (present in recent versions [M]; README fetch did not list them [?]).
- Neither library backpropagates through a multi-step sampler for you; wrap the policy so that `forward(img1, img2, ...) -> action_chunk` with a fixed seed set inside, then the libraries treat it as any deterministic regressor.
- For ISS/RISE-style token masking on pi0.5 no library is needed: it is N forward passes with an attention/pixel mask and an MSE.

---

## Gaps / not found (state of the art as searched)
- No paper derives or benchmarks gradient attribution *through* a flow-matching action expert; the only pi0.5 attribution work (2605.00321) is perturbation-based on tokens.
- No paper found applying Grad-CAM / IG to ACT or LeRobot-style Diffusion Policy with an explicit per-action-dimension target; PointMapPolicy uses the diffusion loss target.
- No paper found that formalises per-camera token dropping in a VLA as an attribution score, though the training-time camera dropout in pi0/pi0.5 makes it the cleanest in-distribution intervention.
- Seed-fixing for stochastic policies is not discussed anywhere found; it is an obvious requirement for perturbation methods.
