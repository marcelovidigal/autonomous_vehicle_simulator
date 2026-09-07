#!/usr/bin/env bash
# Publish the build to your Hugging Face Space (SDK: Static).
#
# Usage:
#   1) create a "Static" Space at huggingface.co/new-space
#   2) SPACE_REPO=https://huggingface.co/spaces/YOUR_USER/autonomous-vehicle-simulator \
#      bash scripts/deploy_hf.sh
#
# Requires: git, git-lfs and login (huggingface-cli login  or  a token in the remote).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${SPACE_REPO:?set SPACE_REPO=https://huggingface.co/spaces/USER/NAME}"

bash scripts/build_web.sh

TMP="$(mktemp -d)"
git clone "$SPACE_REPO" "$TMP"
cp -rf build/web/* "$TMP"/
cp -f scripts/assets/hf_space_readme.md "$TMP"/README.md   # Space YAML header

cd "$TMP"
git add -A
git commit -m "deploy: Autonomous Vehicle Simulator (pygbag build)"
git push
echo "OK -> $SPACE_REPO"
