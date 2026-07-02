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
#                         (default: /tmp/grabette_pipeline/<name>)
#   --proprioception M    convert mode: none (default) | relative
#   --max-lost-run N      clean: reject if longest lost run > N (default: script's 10)
#   --no-qa               skip the analyze_dataset.py QA step
#   -h, --help            show this help
#
# Examples:
#   ./run_pipeline.sh local/test_pick_can_100_fixed --raw-root /tmp/lerobot_out_100_fixed
#   ./run_pipeline.sh SteveNguyen/my_raw --proprioception relative

set -euo pipefail

usage() { sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; }

RAW="" RAW_ROOT="" WORK="" PROPRIO="none" MAX_LOST_RUN="" DO_QA=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-root)       RAW_ROOT="$2"; shift 2 ;;
    --work)           WORK="$2"; shift 2 ;;
    --proprioception) PROPRIO="$2"; shift 2 ;;
    --max-lost-run)   MAX_LOST_RUN="$2"; shift 2 ;;
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
WORK="${WORK:-/tmp/grabette_pipeline/$BASE}"
CLEAN_ID="local/${BASE}_clean"
CART_ID="local/${BASE}_cartesian"
CLEAN_ROOT="$WORK/clean"
CART_ROOT="$WORK/cartesian"

# Optional args as arrays (robust to spaces / empty).
RAW_ROOT_ARG=();  [[ -n "$RAW_ROOT" ]]     && RAW_ROOT_ARG=(--root "$RAW_ROOT")
CLEAN_EXTRA=();   [[ -n "$MAX_LOST_RUN" ]] && CLEAN_EXTRA=(--max_lost_run "$MAX_LOST_RUN")

echo "════════════════════════════════════════════════════════════════"
echo "  Diffusion Policy — data prep (filter + convert)"
echo "    raw dataset   : $RAW ${RAW_ROOT:+(root $RAW_ROOT)}"
echo "    work dir      : $WORK"
echo "    proprioception: $PROPRIO"
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

echo; echo "════════════════════════════════════════════════════════════════"
echo "  Data prep complete. Converted dataset ready for training:"
echo "    repo_id : $CART_ID"
echo "    root    : $CART_ROOT"
echo
echo "  Train with (separate step — pick your steps/batch/output/wandb):"
echo "    uv run python train.py \\"
echo "        --dataset_repo_id $CART_ID --dataset_root $CART_ROOT \\"
echo "        --training_steps 20000 --batch_size 64"
echo "════════════════════════════════════════════════════════════════"
