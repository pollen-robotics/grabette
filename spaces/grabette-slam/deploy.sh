#!/usr/bin/env bash
# Assemble the self-contained Space and upload it to a HuggingFace Space.
#
# Usage:
#   hf auth login                      # once, needs a write token
#   ./deploy.sh <repo_id>              # e.g. ./deploy.sh pollen-robotics/grabette-slam
#
# Vendors the local grabette-postprocess package (working tree, so it includes
# uncommitted changes) into the build context, then uploads everything.
#
# The `hf` CLI must work. If the global one is broken, run with a working one:
#   HF="uv run --project ../../packages/grabette-postprocess hf" ./deploy.sh <repo_id>
set -euo pipefail

REPO_ID="${1:?usage: ./deploy.sh <org/space-name>}"
: "${HF:=hf}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/../../packages/grabette-postprocess" && pwd)"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Space files
cp "$HERE/Dockerfile" "$HERE/app.py" "$HERE/pipeline.py" \
   "$HERE/views.py" "$HERE/controller.py" "$HERE/review.py" \
   "$HERE/requirements.txt" "$HERE/README.md" "$STAGE/"

# Vendored package (working tree), minus caches
cp -r "$PKG" "$STAGE/grabette-postprocess"
find "$STAGE/grabette-postprocess" -name __pycache__ -type d -prune -exec rm -rf {} +

echo "Staged Space at $STAGE:"
ls "$STAGE"

# Create the Space if needed (private), then upload the folder.
$HF repo create "$REPO_ID" --repo-type space --space-sdk docker --private || true
$HF upload "$REPO_ID" "$STAGE" . --repo-type space

echo "Done → https://huggingface.co/spaces/$REPO_ID"
