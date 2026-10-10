#!/usr/bin/env bash
# The LoRA run of scripts/train_lora.sh as a dual-branch model (train/dual_branch.py): every token is
# predicted from (1 - w) * general + w * personalised final representations, where the general branch is
# the frozen pretrained model and the personalised branch adds the LoRA adapters; w starts at 0.1 and is
# either learned or kept fixed (PERSONALIZATION_TRAINABLE).
# Everything else is identical to the standard run.
#
# Env vars: those of scripts/train_lora.sh (OUTPUT_DIR defaults to checkpoints/capimagine-lora-dual), plus
#   PERSONALIZATION_INIT       initial (or fixed) weight of the personalised branch (default: 0.1)
#   PERSONALIZATION_TRAINABLE  true = learn the weight (default), false = keep it fixed at PERSONALIZATION_INIT
#   PERSONALIZATION_LR         learning rate of a trainable weight (default: 1e-2)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/capimagine-lora-dual}"

exec bash "$REPO_DIR/scripts/train_lora.sh" \
  --dual_branch True \
  --personalization_init "${PERSONALIZATION_INIT:-0.1}" \
  --personalization_trainable "${PERSONALIZATION_TRAINABLE:-true}" \
  --personalization_lr "${PERSONALIZATION_LR:-1e-2}" \
  "$@"
