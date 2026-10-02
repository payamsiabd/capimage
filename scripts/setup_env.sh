#!/usr/bin/env bash
# Build the evaluation environment for reproducing CapImagine-7B on V* Bench.
#
# Library versions follow Monet's requirements.txt (vllm==0.10.0 -> torch==2.7.1,
# transformers==4.54.0): the CapImagine paper trains with the Monet codebase and
# evaluates "following the experimental protocol of Monet". VLMEvalKit is pinned
# to 0bac5c0 (2026-02-25), the commit current when the CapImagine weights were
# released; later commits changed MCQ answer matching and judge defaults.
#
# Env vars:
#   ENV_NAME     conda env name (default: capimagine)
#   SKIP_CONDA   set to 1 to install into the currently active Python instead
#   VLMEVAL_DIR  where to check out VLMEvalKit (default: third_party/VLMEvalKit)
#   SKIP_FLASH_ATTN  set to 1 to skip flash-attn (only OK if you run with USE_VLLM=1)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-capimagine}"
VLMEVAL_DIR="${VLMEVAL_DIR:-$REPO_DIR/third_party/VLMEvalKit}"
VLMEVAL_COMMIT=0bac5c064e8216037141d2cbd525e66b4b1dce99

if [ "${SKIP_CONDA:-0}" != 1 ]; then
  eval "$(conda shell.bash hook)"
  if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" python=3.10
  fi
  conda activate "$ENV_NAME"
fi

PINS=("vllm==0.10.0" "transformers==4.54.0" "qwen-vl-utils==0.0.11")

pip install "${PINS[@]}" accelerate
# LoRA fine-tuning (train/); 0.17.1 was released alongside transformers 4.54.
pip install "peft==0.17.1"
if [ "${SKIP_FLASH_ATTN:-0}" != 1 ]; then
  # VLMEvalKit's transformers backend for Qwen2.5-VL hard-codes flash_attention_2.
  pip install "flash-attn==2.8.2" --no-build-isolation
fi

if [ ! -d "$VLMEVAL_DIR/.git" ]; then
  mkdir -p "$VLMEVAL_DIR"
  git -C "$VLMEVAL_DIR" init -q
  git -C "$VLMEVAL_DIR" remote add origin https://github.com/open-compass/VLMEvalKit.git
fi
git -C "$VLMEVAL_DIR" fetch -q --depth 1 origin "$VLMEVAL_COMMIT"
git -C "$VLMEVAL_DIR" checkout -q "$VLMEVAL_COMMIT"
pip install -e "$VLMEVAL_DIR"
# vlmeval/dataset/foxbench.py imports `rouge`, which VLMEvalKit's requirements.txt omits at this commit.
pip install rouge

# VLMEvalKit's own requirements are unpinned; re-assert ours in case pip drifted.
pip install "${PINS[@]}"
python -c "import vlmeval" 2>/dev/null || { python -c "import vlmeval"; echo "import vlmeval failed" >&2; exit 1; }

python - <<'EOF'
import importlib.metadata as md
for pkg in ["torch", "transformers", "vllm", "qwen-vl-utils", "flash-attn", "peft", "vlmeval"]:
    try:
        print(f"{pkg:15s} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg:15s} (not installed)")
import torch
print("CUDA available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
EOF

echo
echo "VLMEvalKit is at: $(git -C "$VLMEVAL_DIR" rev-parse HEAD)"
echo "Next: put your judge API key in $VLMEVAL_DIR/.env (see README), then run scripts/download.sh"
