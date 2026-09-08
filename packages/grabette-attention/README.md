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

Output: one overlay per frame per camera, under a per-episode directory, plus a
single `summary.txt` at the output root with a header per episode, carrying
each camera's attention mass, its ablation delta in millimetres and the
per-axis breakdown, the language mass, and the provenance.

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
