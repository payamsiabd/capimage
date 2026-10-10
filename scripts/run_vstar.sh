#!/usr/bin/env bash
# Evaluate CapImagine-7B (or the Qwen2.5-VL-7B baseline, or your own fine-tune) on V* Bench with
# VLMEvalKit, then print Attribute / Spatial / Overall accuracy next to the paper's Table 1.
#
# Env vars:
#   MODEL        capimagine (default) | qwen25vl | custom (a merged fine-tune; needs MODEL_PATH)
#                | dual (a dual-branch adapter from train_lora_dual.sh; needs ADAPTER_PATH)
#   MODEL_PATH   weights dir or HF repo id (default: models/<name> if present, else the HF repo id);
#                for MODEL=dual, the base model (default: Qwen2.5-VL-7B-Instruct)
#   ADAPTER_PATH dual-branch adapter dir (MODEL=dual)
#   MODEL_NAME   output name for MODEL=custom/dual (default: basename of MODEL_PATH/ADAPTER_PATH)
#   JUDGE        VLMEvalKit judge (default: chatgpt-0125 = gpt-3.5-turbo-0125, VLMEvalKit's MCQ
#                default at the pinned commit). exact_matching disables the LLM judge (not the
#                paper protocol).
#   USE_VLLM     1 = VLMEvalKit's vLLM backend; default 0 = HF transformers (official Qwen2.5-VL code).
#                Not available for MODEL=dual.
#   NGPU         data-parallel processes for the transformers backend (default: 1)
#   MODE         all (default) | infer | eval   (passed to VLMEvalKit's --mode)
#   REUSE        1 = reuse the latest earlier predictions (VLMEvalKit --reuse); default 1 for
#                MODE=eval, else 0 (without it VLMEvalKit moves today's earlier outputs to bak_*)
#   WORK_DIR     output root (default: outputs/)
#   VLMEVAL_DIR  VLMEvalKit checkout (default: third_party/VLMEvalKit)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLMEVAL_DIR="${VLMEVAL_DIR:-$REPO_DIR/third_party/VLMEvalKit}"
VLMEVAL_COMMIT=0bac5c064e8216037141d2cbd525e66b4b1dce99
MODEL="${MODEL:-capimagine}"
JUDGE="${JUDGE:-chatgpt-0125}"
USE_VLLM="${USE_VLLM:-0}"
NGPU="${NGPU:-1}"
MODE="${MODE:-all}"
if [ "$MODE" = eval ]; then REUSE="${REUSE:-1}"; else REUSE="${REUSE:-0}"; fi
WORK_DIR="$(mkdir -p "${WORK_DIR:-$REPO_DIR/outputs}" && cd "${WORK_DIR:-$REPO_DIR/outputs}" && pwd)"

ADAPTER_PATH="${ADAPTER_PATH:-}"
case "$MODEL" in
  capimagine) CONFIG="$REPO_DIR/configs/vstar_capimagine.json"; NAME=CapImagine-7B; HF_ID=Michael4933/CapImagine-7B ;;
  qwen25vl)   CONFIG="$REPO_DIR/configs/vstar_qwen25vl.json"; NAME=Qwen2.5-VL-7B-Instruct; HF_ID=Qwen/Qwen2.5-VL-7B-Instruct ;;
  custom)
    : "${MODEL_PATH:?MODEL=custom needs MODEL_PATH (e.g. a train/merge_lora.py output dir)}"
    CONFIG="$REPO_DIR/configs/vstar_capimagine.json"  # same protocol as CapImagine-7B
    NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"; HF_ID=""; REFERENCE="${REFERENCE:-CapImagine-7B}" ;;
  dual)
    : "${ADAPTER_PATH:?MODEL=dual needs ADAPTER_PATH (a scripts/train_lora_dual.sh output dir)}"
    [ "$USE_VLLM" != 1 ] || { echo "MODEL=dual runs on the transformers backend only; unset USE_VLLM" >&2; exit 1; }
    CONFIG="$REPO_DIR/configs/vstar_capimagine.json"  # same protocol as CapImagine-7B
    NAME="${MODEL_NAME:-$(basename "$ADAPTER_PATH")}"; HF_ID=Qwen/Qwen2.5-VL-7B-Instruct
    REFERENCE="${REFERENCE:-CapImagine-7B}"
    ADAPTER_PATH="$(cd "$ADAPTER_PATH" && pwd)" ;;
  *) echo "Unknown MODEL=$MODEL (expected capimagine, qwen25vl, custom or dual)" >&2; exit 1 ;;
esac
REFERENCE="${REFERENCE:-$NAME}"
if [ -z "${MODEL_PATH:-}" ]; then
  LOCAL="$REPO_DIR/models/$([ "$MODEL" = dual ] && echo Qwen2.5-VL-7B-Instruct || echo "$NAME")"
  if [ -d "$LOCAL" ]; then MODEL_PATH="$LOCAL"; else MODEL_PATH="$HF_ID"; fi
fi
# run.py executes inside VLMEVAL_DIR, so make local paths absolute.
if [ -d "$MODEL_PATH" ]; then MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"; fi

actual_commit="$(git -C "$VLMEVAL_DIR" rev-parse HEAD)"
if [ "$actual_commit" != "$VLMEVAL_COMMIT" ]; then
  echo "WARNING: VLMEvalKit is at $actual_commit, not the pinned $VLMEVAL_COMMIT; MCQ matching may differ." >&2
fi

if [ "$MODE" != infer ] && [ "$JUDGE" != exact_matching ]; then
  (cd "$VLMEVAL_DIR" && python "$REPO_DIR/scripts/check_judge.py" --judge "$JUDGE")
fi

# Point the config at the chosen weights (and, for MODEL=dual, the dual-branch model class and adapter).
RENDERED="$WORK_DIR/${NAME}_vstar_config.json"
python - "$CONFIG" "$NAME" "$MODEL_PATH" "$RENDERED" "$ADAPTER_PATH" <<'EOF'
import json, sys
src, name, model_path, dst, adapter_path = sys.argv[1:]
with open(src) as f:
    cfg = json.load(f)
(entry,) = cfg['model'].values()
cfg['model'] = {name: {**entry, 'model_path': model_path}}
if adapter_path:
    cfg['model'][name].update({'class': 'DualBranchQwen2VLChat', 'adapter_path': adapter_path})
with open(dst, 'w') as f:
    json.dump(cfg, f, indent=2)
EOF
echo "Model: $NAME <- $MODEL_PATH${ADAPTER_PATH:+ + dual-branch adapter $ADAPTER_PATH} | judge: $JUDGE | vLLM: $USE_VLLM | config: $RENDERED"

ARGS=(--config "$RENDERED" --work-dir "$WORK_DIR" --judge "$JUDGE" --mode "$MODE")
if [ "$REUSE" = 1 ]; then ARGS+=(--reuse); fi
RUN=(run.py)
if [ "$MODEL" = dual ]; then RUN=("$REPO_DIR/scripts/vlmeval_dual.py" run.py); fi  # registers DualBranchQwen2VLChat
cd "$VLMEVAL_DIR"
if [ "$USE_VLLM" = 1 ]; then
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  python "${RUN[@]}" "${ARGS[@]}" --use-vllm
elif [ "$NGPU" -gt 1 ]; then
  torchrun --nproc-per-node="$NGPU" "${RUN[@]}" "${ARGS[@]}"
else
  python "${RUN[@]}" "${ARGS[@]}"
fi
cd "$REPO_DIR"

if [ "$MODE" != infer ]; then
  python "$REPO_DIR/scripts/summarize_vstar.py" --work-dir "$WORK_DIR" --model-name "$NAME" --judge "$JUDGE" \
    --reference "$REFERENCE"
fi
