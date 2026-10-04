#!/usr/bin/env bash
# The LoRA run of scripts/train_lora.sh with one change: the goldfish loss (Hans et al., NeurIPS 2024)
# replaces the standard next-token loss. Defaults are the official repo's released config
# (hash-table mask, k=4, context width h=13); everything else is identical.
#
# Env vars: those of scripts/train_lora.sh (OUTPUT_DIR defaults to checkpoints/capimagine-lora-goldfish), plus
#   K_GOLDFISH   drop about 1 in k supervised tokens (default: 4)
#   GOLDFISH_H   tokens hashed per drop decision (default: 13)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/capimagine-lora-goldfish}"

exec bash "$REPO_DIR/scripts/train_lora.sh" \
  --goldfish_strategy hash-table \
  --k_goldfish "${K_GOLDFISH:-4}" \
  --goldfish_context_width "${GOLDFISH_H:-13}" \
  "$@"
