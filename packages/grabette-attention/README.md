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

# one map per denoising step of the same frame, from a single forward pass
grabette-attn --checkpoint <user>/<model>_best \
              --dataset <user>/<dataset>_graspproj \
              --denoise-step all --task "pick the sugar cube"
```

`--denoise-step` takes `last` (default), `first`, `mean`, an explicit index, or
`all`. `all` fans each frame out into one record per step — the map moves
between steps, and the captures from one pass already hold every step, so this
costs no extra inference. The ablation is a property of the emitted chunk, not
of any one step, so the same millimetres appear on every per-step record and
`provenance["ablation_scope"]` says so.

Overlay colour is clipped at the 99th percentile of each grid, and the title
reports the clip and how far the peak exceeded it. Without this, a single
high-norm "register" cell parked in featureless background — measured at 36×
the median while holding only 6% of the camera's mass — takes the whole colour
ramp and renders everything else as flat black.

Output: one overlay per frame per camera, under a per-episode directory, plus a
single `summary.txt` at the output root with a header per episode, carrying
each camera's attention mass, its ablation delta in millimetres and the
per-axis breakdown, the language mass, and the provenance.

## Reading the per-axis breakdown

The per-axis millimetres are in action-channel order, and the channels are the
standard OpenCV **camera** frame — relative to the gripper-mounted view, not to
the world:

| slot | axis | meaning |
| --- | --- | --- |
| 0 | x | lateral, +right |
| 1 | y | **vertical**, +down |
| 2 | z | **depth / range**, +forward along the optical axis |

Vertical and range are different axes and they do not behave alike, so a
figure read from the wrong slot is a claim about the wrong thing.

How this was established, over 172 episodes of `mustard_graspproj`
(`scratchpad/axis_convention.py`):

- The approach phase is pinned to **+z** in 99% of episodes, with 70 mm of mean
  travel. A gravity-aligned world frame would scatter that direction across
  channels as the object's placement varied, so the frame is camera-relative.
- The lift after closure is **−y**, 44 mm, 91% agreement, while z's sign
  agreement collapses to 51% — noise. So y is vertical and +y is down.

This agrees independently with the calibration in the project `CLAUDE.md`:
`T_b_c1` is a 180° rotation about x, so camera y = −imu y, and imu y is up.

**To re-verify for another robot or action space**, re-run that script against
the new dataset: read `action.names` from `meta/info.json`, then check which
channel carries the approach and which carries the lift. If the answer differs,
update `_AXIS_LABELS` in `frontends/png.py` and the convention block in
`records.ViewAblation`.

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
