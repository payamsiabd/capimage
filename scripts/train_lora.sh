#!/usr/bin/env bash
# LoRA fine-tune Qwen2.5-VL-7B-Instruct on CapImagine-Data (recipe: see train/train_lora.py).
#
# Env vars:
#   DATA          annotation file(s), space-separated (default: the only *.json/*.jsonl in data/CapImagine-Data)
#   IMAGE_ROOT    base dir for relative image paths (default: data/CapImagine-Data)
#   BASE_MODEL    default: models/Qwen2.5-VL-7B-Instruct if present, else Qwen/Qwen2.5-VL-7B-Instruct
#   OUTPUT_DIR    default: checkpoints/capimagine-lora
#   NGPU          GPUs to use (default: all visible)
#   GLOBAL_BATCH  effective batch size (default: 128 = the paper's 8 GPUs x batch 1 x grad-accum 16)
# Extra arguments go to train.train_lora, e.g. --learning_rate 2e-4 --num_train_epochs 2 --report_to wandb.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_DIR/data/CapImagine-Data"

if [ -z "${DATA:-}" ]; then
  mapfile -t found < <(find "$DATA_DIR" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' \) | sort)
  if [ "${#found[@]}" -ne 1 ]; then
    echo "Set DATA=<annotation file>; found ${#found[@]} candidates in $DATA_DIR: ${found[*]:-none}" >&2
    exit 1
  fi
  DATA="${found[0]}"
fi
read -ra DATA_FILES <<< "$DATA"
IMAGE_ROOT="${IMAGE_ROOT:-$DATA_DIR}"
if [ -z "${BASE_MODEL:-}" ]; then
  BASE_MODEL="$REPO_DIR/models/Qwen2.5-VL-7B-Instruct"
  [ -d "$BASE_MODEL" ] || BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
fi
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/checkpoints/capimagine-lora}"
NGPU="${NGPU:-$(python -c 'import torch; print(max(torch.cuda.device_count(), 1))')}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
BSZ=1

if (( GLOBAL_BATCH % (NGPU * BSZ) )); then
  echo "GLOBAL_BATCH=$GLOBAL_BATCH is not divisible by NGPU*batch=$((NGPU * BSZ))" >&2
  exit 1
fi
GRAD_ACCUM=$(( GLOBAL_BATCH / (NGPU * BSZ) ))
echo "Base: $BASE_MODEL | data: ${DATA_FILES[*]} | $NGPU GPU(s) x batch $BSZ x accum $GRAD_ACCUM = $GLOBAL_BATCH"

cd "$REPO_DIR"
torchrun --nproc-per-node "$NGPU" -m train.train_lora \
  --model_name_or_path "$BASE_MODEL" \
  --data_path "${DATA_FILES[@]}" \
  --image_root "$IMAGE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$BSZ" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  "$@"
