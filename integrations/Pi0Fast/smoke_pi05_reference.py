"""Health smoke: does the pi05 port on OUR pinned lerobot revision generate
sane, input-dependent actions with the known-good libero reference checkpoint?
(Same verification pattern that validated the pi0fast port before spending on
training — run BEFORE any pi05 fine-tune money.)

Healthy = both synthetic inputs produce finite, sane-scale action chunks that
DIFFER from each other. Note pi05 is flow-matching: no token decode, so the
failure modes to watch are NaNs, absurd scales, or input-independence.

Needs ~7 GB GPU for the 2.3B-param bf16 model — run on the 5090 (the laptop
4070 OOMs at load). Usage:  uv run python smoke_pi05_reference.py
"""
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

CKPT = "lerobot/pi05_libero_finetuned_v044"

cfg = PreTrainedConfig.from_pretrained(CKPT)
cfg.device = "cpu"
# Compile is a training/deployment optimization and (with the checkpoint's
# max-autotune setting) surfaces a Float-vs-BFloat16 mismatch inside the
# inductor graph on this port. Irrelevant for a generation-path smoke: off.
cfg.compile_model = False
policy = get_policy_class("pi05").from_pretrained(CKPT, config=cfg)
policy = policy.to(dtype=torch.bfloat16).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
policy = policy.to(device)
policy.config.device = device
DTYPE_FALLBACK_DONE = False


def fallback_to_fp32():
    """bf16 dtype clash in eager too -> run full fp32 (fits a 24GB+ card)."""
    global policy, DTYPE_FALLBACK_DONE
    print("  (bf16 dtype clash — falling back to full float32)")
    policy = policy.to(dtype=torch.float32).eval()
    DTYPE_FALLBACK_DONE = True


pre, post = make_pre_post_processors(policy.config, CKPT)
cams = [k for k in policy.config.input_features if "image" in k]
print(f"loaded {CKPT} on {device} | cams {cams} | chunk {policy.config.chunk_size} | "
      f"denoise steps {policy.config.num_inference_steps}")

outs = []
for seed in (0, 1):
    g = torch.Generator().manual_seed(seed)
    batch = {"task": "pick up the black bowl and place it on the plate"}
    for k in cams:
        batch[k] = torch.rand(1, 3, 224, 224, generator=g).to(device)
    batch["observation.state"] = (torch.rand(1, 8, generator=g) * 2 - 1).to(device)
    policy.reset()
    try:
        with torch.no_grad():
            try:
                a = post(policy.select_action(pre(batch))).squeeze(0).float().cpu().numpy()
            except RuntimeError as e:
                if "dtype" in str(e) and not DTYPE_FALLBACK_DONE:
                    fallback_to_fp32()
                    policy.reset()
                    a = post(policy.select_action(pre(batch))).squeeze(0).float().cpu().numpy()
                else:
                    raise
        outs.append(a)
        print(f"seed {seed}: action[:7] = {np.round(a[:7], 4)}  finite={np.all(np.isfinite(a))}")
    except Exception as e:
        outs.append(None)
        print(f"seed {seed}: ERROR — {type(e).__name__}: {str(e)[:250]}")

if all(o is not None for o in outs):
    diff = float(np.abs(np.asarray(outs[0]) - np.asarray(outs[1])).mean())
    print(f"\nmean |a(seed0)-a(seed1)| = {diff:.6f}")
    print("VERDICT: " + ("PASS — sane, input-dependent flow generation on the pinned rev"
                         if diff > 1e-6 else "SUSPICIOUS — input-independent"))
else:
    print("\nVERDICT: FAIL — pi05 inference path broken in this environment")
