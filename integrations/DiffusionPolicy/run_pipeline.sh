#!/usr/bin/env bash
#
# One-shot DATA-PREP pipeline for the Diffusion Policy (filtering + conversion).
#
# Chains the prep steps so you don't run them by hand:
#
#   raw dataset (8D, with is_lost)  ──▶  clean_dataset.py          (reject unrecoverable-lost episodes)
#                                   ──▶  convert_dataset.py        (camera-local deltas + despike)
#                                   ──▶  analyze_dataset.py        (QA, optional)
#                                   ──▶  resize_dataset_videos.py  (480x360 training copy: the policy
#                                        consumes 236x236 internally, so training on full-res video is a
#                                        measured 2-3x cost for nothing; --no-resize to skip)
#                                   ──▶  grasp_projection_convert   (--grasp-projection: re-express the two
#                                        gripper angles as (strategy, closure) so the policy can command a
#                                        FULL close and let the object stop the fingers)
#
# WITHOUT --grasp-projection the gripper channels stay RAW JOINT ANGLES, and a
# position servo replaying a demonstrated angle under-closes: the recorded angle
# is where the human's fingers sat while PRESSING the object, so reproducing it
# stops just short and grips nothing (demos use only 38-60% of the proximal
# range). See docs/grasp_projection.md.
#
# Training is intentionally NOT run here — it's long-running and you'll want to
# launch it deliberately (GPU, steps, wandb, ...). The script prints the exact
# `train.py` command to run next, wired to the converted dataset.
#
# The raw dataset must carry the `is_lost` feature (built by the postprocess
# generate_dataset.py). Intermediate datasets are written under --work and
# threaded between steps via --root.
#
# Usage:
#   ./run_pipeline.sh <raw_repo_id> [options]
#
# Options:
#   --raw-root DIR        local root of the raw dataset (omit if it's on the Hub)
#   --work DIR            scratch dir for intermediate datasets
#                         (default: ~/.cache/grabette_pipeline/<name> — must be on
#                         real disk, NOT /tmp: /tmp is often RAM-backed tmpfs)
#   --proprioception M    convert mode: none (default) | relative
#   --max-lost-run N      clean: reject if longest lost run > N (default: script's 10)
#   --smooth-poses N      Savitzky-Golay window (odd, frames) smoothing the absolute
#                         poses before differencing (recommended: 9 at 50fps). Removes
#                         SLAM pose jitter that dominates grasp-phase delta supervision.
#                         Default: off (omit the flag) until A/B-validated.
#   --cameras "C ..."     camera stream(s) to KEEP (default: "cam0", the camera the
#                         policy trains on). Extra recorded streams are removed:
#                         they double training decode cost and can crash training
#                         if corrupt, even though the policy never reads them.
#                         Pass --cameras all to keep everything.
#   --grasp-projection    re-express the gripper channels as (strategy, closure).
#                         Recommended: it is what lets the policy command a full
#                         close. Measured on hardware, 5/5 Diffusion grasps
#                         reached 60-68 deg of proximal travel against 48-51 deg
#                         demonstrated, with no grasp assist.
#   --rest-is-closed      --grasp-projection: recording convention where the
#                         operator holds the gripper CLOSED when idle. Must match
#                         how the episodes were actually recorded — see
#                         docs/grasp_projection_recording_procedure.md.
#   --projection-repo-id ID
#                         --grasp-projection: repo_id used to open the converted
#                         dataset for its (mandatory) stats pass. Defaults to the
#                         raw input's id. Must RESOLVE on the Hub — LeRobotDataset
#                         queries revisions even for a local root, so an invented
#                         id 404s. The data always comes from the local root.
#   --no-qa               skip the analyze_dataset.py QA step
#   --no-resize           skip the 480x360 training copy (train on full res —
#                         debugging only; costs 2-3x more per training step)
#   -h, --help            show this help
#
# Examples:
#   ./run_pipeline.sh local/test_pick_can_100_fixed --raw-root /tmp/lerobot_out_100_fixed
#   ./run_pipeline.sh <user>/my_raw --proprioception relative

set -euo pipefail

# Print the whole leading comment block, however long it is. A fixed line range
# silently truncates the help text the moment the header grows (it did).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

RAW="" RAW_ROOT="" WORK="" PROPRIO="none" MAX_LOST_RUN="" CAMERAS="cam0" SMOOTH_POSES="" DO_QA=1 DO_RESIZE=1
DO_PROJECTION=0 REST_IS_CLOSED=0 PROJ_REPO_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-root)       RAW_ROOT="$2"; shift 2 ;;
    --work)           WORK="$2"; shift 2 ;;
    --proprioception) PROPRIO="$2"; shift 2 ;;
    --max-lost-run)   MAX_LOST_RUN="$2"; shift 2 ;;
    --smooth-poses)   SMOOTH_POSES="$2"; shift 2 ;;
    --cameras)        CAMERAS="$2"; shift 2 ;;
    --no-qa)          DO_QA=0; shift ;;
    --no-resize)      DO_RESIZE=0; shift ;;
    --grasp-projection) DO_PROJECTION=1; shift ;;
    --rest-is-closed) REST_IS_CLOSED=1; shift ;;
    --projection-repo-id) PROJ_REPO_ID="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    -*)               echo "Unknown option: $1" >&2; exit 1 ;;
    *)                if [[ -z "$RAW" ]]; then RAW="$1"; shift; else echo "Unexpected arg: $1" >&2; exit 1; fi ;;
  esac
done

[[ -n "$RAW" ]] || { echo "ERROR: raw dataset repo_id is required." >&2; echo; usage; exit 1; }

# Run everything from this script's directory (the DiffusionPolicy uv project).
cd "$(dirname "$0")"

BASE="${RAW##*/}"
# Default work dir on REAL DISK (~/.cache), never /tmp: on many machines /tmp is
# tmpfs (RAM) — datasets parked there eat gigabytes of memory, get truncated
# under pressure (corrupt videos), starve training (OOM kills), and vanish on
# reboot. All four happened.
WORK="${WORK:-${XDG_CACHE_HOME:-$HOME/.cache}/grabette_pipeline/$BASE}"
# Create the work dir BEFORE probing its filesystem: df on a not-yet-existing
# path returns nonzero, and under `set -e` a failing $(...) in an assignment
# kills the whole script — silently, since df's stderr is suppressed. (Bit us:
# first run on a fresh machine died with no output at all.) `|| true` keeps the
# probe advisory no matter what.
mkdir -p "$WORK"
WORK_FSTYPE=$(df -PT "$WORK" 2>/dev/null | awk 'NR==2{print $2}' || true)
if [[ "$WORK_FSTYPE" == "tmpfs" || "$WORK_FSTYPE" == "ramfs" ]]; then
  echo "WARNING: work dir '$WORK' is on $WORK_FSTYPE (RAM-backed)." >&2
  echo "         Datasets there consume RAM, can be silently truncated (corrupt" >&2
  echo "         videos), and are lost on reboot. Pass --work <dir-on-disk>." >&2
fi
CLEAN_ID="local/${BASE}_clean"
CART_ID="local/${BASE}_cartesian"
CLEAN_ROOT="$WORK/clean"
CART_ROOT="$WORK/cartesian"

# Optional args as arrays (robust to spaces / empty).
RAW_ROOT_ARG=();  [[ -n "$RAW_ROOT" ]]     && RAW_ROOT_ARG=(--root "$RAW_ROOT")
CLEAN_EXTRA=();   [[ -n "$MAX_LOST_RUN" ]] && CLEAN_EXTRA=(--max_lost_run "$MAX_LOST_RUN")
# Camera filter: keep only the training camera(s) unless --cameras all.
# (word-splitting of $CAMERAS is intentional: --cameras "cam0 cam1")
if [[ "$CAMERAS" != "all" ]]; then
  # shellcheck disable=SC2206
  CLEAN_EXTRA+=(--keep_cameras $CAMERAS)
fi

echo "════════════════════════════════════════════════════════════════"
echo "  Diffusion Policy — data prep (filter + convert)"
echo "    raw dataset   : $RAW ${RAW_ROOT:+(root $RAW_ROOT)}"
echo "    work dir      : $WORK"
echo "    proprioception: $PROPRIO   cameras kept: $CAMERAS"
echo "════════════════════════════════════════════════════════════════"

echo; echo "==> [1] clean — reject episodes with unrecoverable SLAM loss"
uv run python clean_dataset.py \
  --repo_id "$RAW" "${RAW_ROOT_ARG[@]}" \
  --output_repo_id "$CLEAN_ID" --output_root "$CLEAN_ROOT" --overwrite_output \
  "${CLEAN_EXTRA[@]}"

CONVERT_EXTRA=(); [[ -n "$SMOOTH_POSES" ]] && CONVERT_EXTRA=(--smooth_poses "$SMOOTH_POSES")
echo; echo "==> [2] convert — camera-local deltas + per-frame despike"
uv run python convert_dataset.py \
  --repo_id "$CLEAN_ID" --root "$CLEAN_ROOT" \
  --proprioception "$PROPRIO" \
  --output_repo_id "$CART_ID" --output_root "$CART_ROOT" --overwrite_output \
  "${CONVERT_EXTRA[@]}"

if [[ "$DO_QA" == 1 ]]; then
  echo; echo "==> [3] analyze — QA on the converted dataset"
  uv run python analyze_dataset.py --repo_id "$CART_ID" --root "$CART_ROOT"
fi

# The dataset-tools steps above RE-ENCODE video; an occasional encoder glitch
# writes an invalid packet that only surfaces hours into training ("Could not
# push packet to decoder"). Decode-check every episode NOW, through the same
# path training uses, and fail the pipeline loudly instead.
echo; echo "==> [4] video integrity — decode-check every episode"
if ! uv run python check_dataset_videos.py --repo_id "$CART_ID" --dataset_root "$CART_ROOT"; then
  echo "ERROR: the converted dataset has corrupt video segment(s) — see the episode" >&2
  echo "       list above. Re-run this pipeline (a fresh re-encode usually fixes it)." >&2
  echo "       If the same episodes fail repeatedly, inspect their raw recordings." >&2
  exit 1
fi

# Training dataset defaults to the converted output; the resize step below
# replaces it with the 480x360 copy when enabled.
TRAIN_ID="$CART_ID"
TRAIN_ROOT="$CART_ROOT"

if [[ "$DO_RESIZE" == 1 ]]; then
  RESIZE_ID="local/${BASE}_cartesian_480"
  RESIZE_ROOT="$WORK/cartesian_480"
  echo; echo "==> [5] resize — 480x360 training copy (policy consumes 236x236; full-res"
  echo "        training is a measured 2-3x slowdown for zero benefit)"
  # resize_dataset_videos.py refuses an existing output (non-destructive by
  # design); pipeline re-runs regenerate it like every other intermediate.
  rm -rf "$RESIZE_ROOT"
  uv run python resize_dataset_videos.py \
    --repo_id "$CART_ID" --root "$CART_ROOT" \
    --output_root "$RESIZE_ROOT"

  echo; echo "==> [6] video integrity — decode-check the resized copy"
  if ! uv run python check_dataset_videos.py --repo_id "$RESIZE_ID" --dataset_root "$RESIZE_ROOT"; then
    echo "ERROR: the resized dataset has corrupt video segment(s). Re-run the" >&2
    echo "       pipeline; if it persists, train on the full-res converted output" >&2
    echo "       ($CART_ROOT) with --no-resize and report the issue." >&2
    exit 1
  fi
  TRAIN_ID="$RESIZE_ID"
  TRAIN_ROOT="$RESIZE_ROOT"
fi

if [[ "$DO_PROJECTION" == 1 ]]; then
  PROJ_ID="local/${BASE}_graspproj"
  PROJ_ROOT="$WORK/graspproj"
  PROJ_EXTRA=(); [[ "$REST_IS_CLOSED" == 1 ]] && PROJ_EXTRA=(--rest_is_closed)
  # The converter's stats pass opens the dataset through LeRobotDataset, which
  # queries the Hub for revisions even when `root` is local — so the id has to be
  # one that RESOLVES. An invented `local/..._graspproj` 404s. The raw input's id
  # is the right one (the data still comes from --dst_root); override with
  # --projection-repo-id when the raw input is itself local-only.
  PROJ_REPO_ID="${PROJ_REPO_ID:-$RAW}"
  echo; echo "==> [7] grasp projection — gripper angles -> (strategy, closure)"
  # Runs in the grabette-postprocess project, not this one: the converter and the
  # projection itself live there (and in gripette), and this uv project carries
  # neither.
  if ! uv run --project ../../packages/grabette-postprocess \
      python -m grabette_postprocess.grasp_projection_convert \
      --src_root "$TRAIN_ROOT" --dst_root "$PROJ_ROOT" --overwrite \
      --repo_id "$PROJ_REPO_ID" "${PROJ_EXTRA[@]}"; then
    echo "ERROR: the grasp projection step failed." >&2
    echo "       If it 404'd on '$PROJ_REPO_ID': the stats pass needs a repo_id that" >&2
    echo "       resolves on the Hub. Pass --projection-repo-id <a real dataset id>." >&2
    echo "       Stats are NOT optional here — closure is 0..1 where the raw angle" >&2
    echo "       was 0..1.6 rad, so stale stats mis-normalise the channel." >&2
    exit 1
  fi
  TRAIN_ID="$PROJ_ID"
  TRAIN_ROOT="$PROJ_ROOT"
fi

echo; echo "════════════════════════════════════════════════════════════════"
echo "  Data prep complete. Training dataset:"
echo "    repo_id : $TRAIN_ID"
echo "    root    : $TRAIN_ROOT"
# State the gripper representation EVERY time. Getting this wrong is silent in
# both directions and is the single most expensive mistake available here.
if [[ "$DO_PROJECTION" == 1 ]]; then
  echo "    gripper : (strategy, closure) — projected"
  echo "              eval with: evaluate.py --grasp_projection on"
else
  echo "    gripper : RAW JOINT ANGLES (radians) — not projected"
  echo
  echo "  NOTE: a policy trained on raw angles UNDER-CLOSES. The demonstrated"
  echo "        angle is where the human's fingers sat while pressing the object;"
  echo "        a position servo replaying it stops short and grips nothing."
  echo "        Re-run with --grasp-projection, or project this dataset with:"
  echo "          uv run --project ../../packages/grabette-postprocess \\"
  echo "            python -m grabette_postprocess.grasp_projection_convert \\"
  echo "            --src_root $TRAIN_ROOT --dst_root $WORK/graspproj \\"
  echo "            --repo_id local/${BASE}_graspproj"
fi
[[ "$DO_RESIZE" == 1 ]] && echo "    (full-res converted copy kept at: $CART_ROOT)"
echo
# Persist the full certified train command — the console print gets lost in
# scrollback, and training MUST target the converted output, not the raw repo.
TRAIN_CMD_FILE="$WORK/train_command.txt"
cat > "$TRAIN_CMD_FILE" <<EOF
# Generated by run_pipeline.sh — train on the PREPARED dataset (not the raw repo id!)
uv run python train.py \\
    --dataset_repo_id $TRAIN_ID --dataset_root $TRAIN_ROOT \\
    --output_dir outputs/${BASE}_diffusion \\
    --training_steps 50000 --batch_size 64 --bf16 \\
    --num_workers 4 --prefetch_factor 2 \\
    --color_jitter --state_noise_std 0.01 \\
    --eval_freq 500 --save_freq 5000
# optional: --push_to_hub <user>/<model>   --wandb_project <proj>
# run inside screen/tmux, log with: ... 2>&1 | tee train.log
# if /tmp is tmpfs (train.py warns): prefix with TMPDIR=\$HOME/tmp (mkdir it first)
EOF
echo "  Train with (full certified command, also saved to $TRAIN_CMD_FILE):"
sed 's/^/    /' "$TRAIN_CMD_FILE"
echo "════════════════════════════════════════════════════════════════"
