#!/usr/bin/env bash
#
# One-shot DATA-PREP pipeline for the Diffusion Policy (filtering + conversion).
#
# Chains the prep steps so you don't run them by hand:
#
#   raw dataset (8D, with is_lost)  ──▶  clean_dataset.py   (reject unrecoverable-lost episodes)
#                                   ──▶  convert_dataset.py (camera-local deltas + despike)
#                                   ──▶  analyze_dataset.py (QA, optional)
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
#   --cameras "C ..."     camera stream(s) to KEEP (default: "cam0", the camera the
#                         policy trains on). Extra recorded streams are removed:
#                         they double training decode cost and can crash training
#                         if corrupt, even though the policy never reads them.
#                         Pass --cameras all to keep everything.
#   --no-qa               skip the analyze_dataset.py QA step
#   -h, --help            show this help
#
# Examples:
#   ./run_pipeline.sh local/test_pick_can_100_fixed --raw-root /tmp/lerobot_out_100_fixed
#   ./run_pipeline.sh <user>/my_raw --proprioception relative

set -euo pipefail

usage() { sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; }

RAW="" RAW_ROOT="" WORK="" PROPRIO="none" MAX_LOST_RUN="" CAMERAS="cam0" DO_QA=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-root)       RAW_ROOT="$2"; shift 2 ;;
    --work)           WORK="$2"; shift 2 ;;
    --proprioception) PROPRIO="$2"; shift 2 ;;
    --max-lost-run)   MAX_LOST_RUN="$2"; shift 2 ;;
    --cameras)        CAMERAS="$2"; shift 2 ;;
    --no-qa)          DO_QA=0; shift ;;
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
WORK_FSTYPE=$(df -PT "$(dirname "$WORK")" 2>/dev/null | awk 'NR==2{print $2}')
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

echo; echo "==> [2] convert — camera-local deltas + per-frame despike"
uv run python convert_dataset.py \
  --repo_id "$CLEAN_ID" --root "$CLEAN_ROOT" \
  --proprioception "$PROPRIO" \
  --output_repo_id "$CART_ID" --output_root "$CART_ROOT" --overwrite_output

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

echo; echo "════════════════════════════════════════════════════════════════"
echo "  Data prep complete. Converted dataset ready for training:"
echo "    repo_id : $CART_ID"
echo "    root    : $CART_ROOT"
echo
# Persist the full certified train command — the console print gets lost in
# scrollback, and training MUST target the converted output, not the raw repo.
TRAIN_CMD_FILE="$WORK/train_command.txt"
cat > "$TRAIN_CMD_FILE" <<EOF
# Generated by run_pipeline.sh — train on the CONVERTED dataset (not the raw repo id!)
uv run python train.py \\
    --dataset_repo_id $CART_ID --dataset_root $CART_ROOT \\
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
