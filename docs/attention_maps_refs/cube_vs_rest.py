"""The finger-versus-cube question, finally answerable.

With a validated cube region (Grounding DINO + SAM2, agreeing cell-for-cell
with hand annotation) every map already computed can be read on the object
rather than by eye:

    attention          does either model LOOK at the cube?
    causal, RMS        does covering the cube change the commanded chunk?
    causal, endpoint   does covering the cube move where the gripper ARRIVES?

for both action representations, at the approach frame and the grasp frame.

The mug is segmented by the same pipeline as a reference object, and
"elsewhere" is every cell in neither region. Ratios are against elsewhere, so
1.0 means the region is unremarkable.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from grabette_attention.sources import DatasetSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G

DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
CAM = "observation.images.cam0"
EPISODE = 0
FRAMES = {2: "approach 0%", 48: "GRASP"}
ROWS, COLS = 12, 16
DINO = "IDEA-Research/grounding-dino-base"
SAM2 = "facebook/sam2.1-hiera-small"
PHRASES = {"cube": "a small beige sugar cube.", "mug": "a blue and white mug."}


def cells_of(mask):
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    return {
        (min(int(y // (h / ROWS)), ROWS - 1), min(int(x // (w / COLS)), COLS - 1))
        for y, x in zip(ys, xs)
    }


def segment(rgb, phrase, dino_proc, dino, sam_proc, sam):
    from PIL import Image

    pil = Image.fromarray(rgb)
    h, w = rgb.shape[:2]
    inputs = dino_proc(images=pil, text=phrase, return_tensors="pt")
    with torch.no_grad():
        out = dino(**inputs)
    res = dino_proc.post_process_grounded_object_detection(
        out, inputs["input_ids"], threshold=0.2, text_threshold=0.2,
        target_sizes=[(h, w)],
    )[0]
    if not len(res["scores"]):
        return None, None
    best = int(np.argmax([float(s) for s in res["scores"]]))
    x0, y0, x1, y1 = (float(v) for v in res["boxes"][best])
    sam_in = sam_proc(images=pil, input_boxes=[[[x0, y0, x1, y1]]],
                      return_tensors="pt")
    with torch.no_grad():
        sam_out = sam(**sam_in, multimask_output=False)
    mask = np.asarray(sam_proc.post_process_masks(
        sam_out.pred_masks, sam_in["original_sizes"])[0][0][0], dtype=bool)
    return mask, float(res["scores"][best])


def main() -> None:
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
    )

    dino_proc = AutoProcessor.from_pretrained(DINO)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO).eval()
    sam_proc = AutoProcessor.from_pretrained(SAM2)
    sam = Sam2Model.from_pretrained(SAM2).eval()

    source = DatasetSource(
        G.REPO, episodes=[EPISODE], camera_keys=(CAM,), task=G.TASK,
        root=G.ROOT, selection=sorted(FRAMES),
    )
    frames = {obs.frame: obs.images[CAM] for obs in source.frames()}

    chunk_attn = {}
    data = np.load(DST / "pnp_grids.npz")
    for i, (e, f) in enumerate(data["meta"]):
        if int(e) == EPISODE and int(f) in FRAMES:
            chunk_attn[int(f)] = data["grids"][i]

    regions, report = {}, {}
    for f, rgb in sorted(frames.items()):
        regions[f] = {}
        for name, phrase in PHRASES.items():
            mask, score = segment(rgb, phrase, dino_proc, dino, sam_proc, sam)
            if mask is None:
                print(f"f{f}: no {name} found")
                continue
            regions[f][name] = cells_of(mask)
            np.save(DST / f"{name}_mask_ep{EPISODE}_f{f}.npy", mask)
            print(f"f{f} {name}: score {score:.3f}, "
                  f"{len(regions[f][name])} cells "
                  f"({', '.join(f'r{r}c{c}' for r, c in sorted(regions[f][name]))})")

    everything = {(r, c) for r in range(ROWS) for c in range(COLS)}
    maps = {
        ("attention", "delta"): lambda f: np.load(
            DST / f"delta_attn_ep{EPISODE}_f{f}.npy"),
        ("attention", "chunkrel"): lambda f: chunk_attn[f],
        ("causal RMS", "delta"): lambda f: np.load(
            DST / f"delta_occl_ep{EPISODE}_f{f}.npy"),
        ("causal RMS", "chunkrel"): lambda f: np.load(
            DST / f"occlusion_ep{EPISODE}_f{f}.npy"),
        ("causal endpoint", "delta"): lambda f: np.load(
            DST / f"endpoint_delta_ep{EPISODE}_f{f}.npy"),
        ("causal endpoint", "chunkrel"): lambda f: np.load(
            DST / f"endpoint_chunkrel_ep{EPISODE}_f{f}.npy"),
    }

    print(f"\n{'frame':<13}{'map':<17}{'model':<10}"
          f"{'cube':>9}{'mug':>9}{'else':>9}{'cube/else':>11}{'mug/else':>10}")
    print("-" * 88)
    for f in sorted(FRAMES):
        for (kind, model), get in maps.items():
            grid = get(f)
            cube = regions[f].get("cube", set())
            mug = regions[f].get("mug", set())
            rest = everything - cube - mug
            vals = {
                "cube": np.mean([grid[c] for c in cube]) if cube else np.nan,
                "mug": np.mean([grid[c] for c in mug]) if mug else np.nan,
                "else": np.mean([grid[c] for c in rest]),
            }
            report[f"{f}|{kind}|{model}"] = {
                k: float(v) for k, v in vals.items()
            } | {"cube_ratio": float(vals["cube"] / vals["else"]),
                 "mug_ratio": float(vals["mug"] / vals["else"])}
            print(f"{FRAMES[f]:<13}{kind:<17}{model:<10}"
                  f"{vals['cube']:9.4f}{vals['mug']:9.4f}{vals['else']:9.4f}"
                  f"{vals['cube']/vals['else']:10.2f}x{vals['mug']/vals['else']:9.2f}x")
        print()

    (DST / "cube_vs_rest.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {DST / 'cube_vs_rest.json'}")


if __name__ == "__main__":
    main()
