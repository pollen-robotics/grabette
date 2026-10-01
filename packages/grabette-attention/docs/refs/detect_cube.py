"""Locate the sugar cube automatically: Grounding DINO for the box, SAM2 for the mask.

Colour thresholding failed four times on this scene -- the cube's tan is not
unique against a patterned cloth, and every mask either claimed a third of the
frame or latched onto a shadow. This replaces the heuristic with models.

Division of labour, because it is easy to get wrong: SAM2 is PROMPTABLE, not
open-vocabulary. It segments what you point at; it cannot find "the sugar cube"
from a description. So Grounding DINO does text -> box, and SAM2 turns that box
into a precise mask. Both run locally (transformers 5.5.4 has them natively),
so no robot imagery leaves the machine.

Validated against hand annotations read off a grid-labelled zoom of the same
two frames -- not against my impression of a heatmap, which has a poor record
in this investigation. Reports centre error and IoU so a failure is loud.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from grabette_attention.sources import DatasetSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pnp_grid as G

DST = Path("/home/steve/Project/Repo/GRABETTE/GRABETTE_RELEASE/attention_out")
CAM = "observation.images.cam0"
EPISODE = 0
ROWS, COLS = 12, 16

DINO = "IDEA-Research/grounding-dino-base"
SAM2 = "facebook/sam2.1-hiera-small"
# Grounding DINO wants lowercase phrases separated by periods.
PROMPT = "a small beige sugar cube. a blue and white mug."
CUBE_PHRASE = "sugar cube"

# Hand-annotated ground truth, read off zoom_ep0_f*.png (x0, x1, y0, y1).
TRUTH = {
    2:  {"cube": (248, 286, 224, 248), "mug": (115, 175, 130, 215)},
    48: {"cube": (208, 250, 232, 260), "mug": (0, 110, 60, 150)},
}


def iou(a, b):
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / max(union, 1e-9)


def cells_of(mask, h, w):
    ys, xs = np.nonzero(mask)
    return {
        (min(int(y // (h / ROWS)), ROWS - 1), min(int(x // (w / COLS)), COLS - 1))
        for y, x in zip(ys, xs)
    }


def main() -> None:
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
    )
    from PIL import Image

    print(f"loading {DINO}")
    dino_proc = AutoProcessor.from_pretrained(DINO)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO).eval()
    print(f"loading {SAM2}")
    sam_proc = AutoProcessor.from_pretrained(SAM2)
    sam = Sam2Model.from_pretrained(SAM2).eval()

    source = DatasetSource(
        G.REPO, episodes=[EPISODE], camera_keys=(CAM,), task=G.TASK,
        root=G.ROOT, selection=sorted(TRUTH),
    )
    frames = {obs.frame: obs.images[CAM] for obs in source.frames()}

    figure, axes = plt.subplots(1, len(frames), figsize=(16, 6.2), dpi=110)
    for panel, (frame, rgb) in enumerate(sorted(frames.items())):
        h, w = rgb.shape[:2]
        pil = Image.fromarray(rgb)
        print(f"\n=== ep{EPISODE} f{frame} ({w}x{h}) ===")

        inputs = dino_proc(images=pil, text=PROMPT, return_tensors="pt")
        with torch.no_grad():
            outputs = dino(**inputs)
        results = dino_proc.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=0.15, text_threshold=0.15,
            target_sizes=[(h, w)],
        )[0]

        print(f"  Grounding DINO returned {len(results['scores'])} boxes")
        best = None
        for score, label, box in zip(results["scores"], results["text_labels"],
                                     results["boxes"]):
            x0, y0, x1, y1 = (float(v) for v in box)
            area = (x1 - x0) * (y1 - y0)
            print(f"    {label!r:28s} score {float(score):.3f}  "
                  f"x {x0:.0f}-{x1:.0f} y {y0:.0f}-{y1:.0f}  area {area:.0f} px")
            if CUBE_PHRASE in str(label) and (best is None or score > best[0]):
                best = (float(score), (x0, x1, y0, y1))

        truth = TRUTH[frame]["cube"]
        if best is None:
            print(f"  NO '{CUBE_PHRASE}' box found — detector failed on this frame")
            axes[panel].imshow(rgb)
            axes[panel].add_patch(mpatches.Rectangle(
                (truth[0], truth[2]), truth[1] - truth[0], truth[3] - truth[2],
                fill=False, edgecolor="#00e5ff", linewidth=2,
                label="hand annotation"))
            axes[panel].set_title(f"ep{EPISODE} f{frame} — DINO found no cube",
                                  fontsize=10)
            axes[panel].set_axis_off()
            continue

        score, box = best
        overlap = iou(box, truth)
        cx, cy = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
        tx, ty = (truth[0] + truth[1]) / 2, (truth[2] + truth[3]) / 2
        print(f"  best cube box: score {score:.3f}  "
              f"x {box[0]:.0f}-{box[1]:.0f} y {box[2]:.0f}-{box[3]:.0f}")
        print(f"  hand truth:                 "
              f"x {truth[0]}-{truth[1]} y {truth[2]}-{truth[3]}")
        print(f"  IoU {overlap:.2f}   centre error "
              f"{np.hypot(cx - tx, cy - ty):.1f} px")

        # SAM2 refines the box into a mask.
        sam_inputs = sam_proc(
            images=pil, input_boxes=[[[box[0], box[2], box[1], box[3]]]],
            return_tensors="pt",
        )
        with torch.no_grad():
            sam_out = sam(**sam_inputs, multimask_output=False)
        masks = sam_proc.post_process_masks(
            sam_out.pred_masks, sam_inputs["original_sizes"]
        )[0]
        mask = np.asarray(masks[0][0], dtype=bool)
        cells = cells_of(mask, h, w)
        print(f"  SAM2 mask: {mask.sum()} px ({100*mask.mean():.2f}% of frame), "
              f"{len(cells)} cells")
        print(f"  cells: {', '.join(f'r{r}c{c}' for r, c in sorted(cells))}")
        np.save(DST / f"cube_mask_ep{EPISODE}_f{frame}.npy", mask)

        ax = axes[panel]
        ax.imshow(rgb)
        ax.imshow(np.ma.masked_where(~mask, mask.astype(float)),
                  cmap="autumn", alpha=0.55, vmin=0, vmax=1)
        ax.add_patch(mpatches.Rectangle(
            (truth[0], truth[2]), truth[1] - truth[0], truth[3] - truth[2],
            fill=False, edgecolor="#00e5ff", linewidth=2))
        ax.add_patch(mpatches.Rectangle(
            (box[0], box[2]), box[1] - box[0], box[3] - box[2],
            fill=False, edgecolor="#ffd400", linewidth=1.6, linestyle="--"))
        ax.set_title(f"ep{EPISODE} f{frame}  IoU {overlap:.2f}\n"
                     "cyan = hand annotation, dashed = DINO box, "
                     "fill = SAM2 mask", fontsize=9.5)
        ax.set_axis_off()

    figure.suptitle(
        "Automatic cube localisation: Grounding DINO (text to box) + SAM2 "
        "(box to mask), checked against hand annotation", fontsize=12,
    )
    figure.tight_layout()
    out = DST / "cube_detection_check.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
