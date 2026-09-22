#!/usr/bin/env bash
# Download the CapImagine-7B weights and the V* Bench data used by VLMEvalKit.
#
# Env vars:
#   WITH_BASELINE  set to 1 to also download Qwen/Qwen2.5-VL-7B-Instruct (pipeline sanity check)
#   MODELS_DIR     where to put the weights (default: models/)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_DIR/models}"

download_model() {
  python - "$1" "$2" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo_id, local_dir = sys.argv[1:]
print(f"Downloading {repo_id} -> {local_dir}")
snapshot_download(repo_id=repo_id, local_dir=local_dir)
EOF
}

download_model Michael4933/CapImagine-7B "$MODELS_DIR/CapImagine-7B"
if [ "${WITH_BASELINE:-0}" = 1 ]; then
  download_model Qwen/Qwen2.5-VL-7B-Instruct "$MODELS_DIR/Qwen2.5-VL-7B-Instruct"
fi

# VLMEvalKit fetches VStarBench.tsv (191 questions, images embedded as base64)
# from huggingface.co/datasets/xjtupanda/VStar_Bench into $LMUData (default ~/LMUData)
# and checks its md5 (b18854d7075574be06b631cd5f7d2d6a). Build it once here so a
# download failure surfaces now instead of after the model has loaded.
python - <<'EOF'
from vlmeval.dataset import build_dataset
ds = build_dataset("VStarBench")
print(f"VStarBench ready: {len(ds.data)} questions; categories: {ds.data['category'].value_counts().to_dict()}")
EOF
