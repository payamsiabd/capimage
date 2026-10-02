#!/usr/bin/env bash
# Download CapImagine-Data (a json file plus an image zip) and the Qwen2.5-VL-7B-Instruct base model.
#
# Env vars:
#   DATA_DIR    where to put the dataset (default: data/CapImagine-Data)
#   MODELS_DIR  where to put the base model (default: models/)
#   SKIP_BASE   set to 1 to skip the base model download
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data/CapImagine-Data}"
MODELS_DIR="${MODELS_DIR:-$REPO_DIR/models}"

download() {
  python - "$1" "$2" "$3" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo_id, repo_type, local_dir = sys.argv[1:]
print(f"Downloading {repo_id} -> {local_dir}")
snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir)
EOF
}

download Michael4933/CapImagine-Data dataset "$DATA_DIR"

# Extract each image archive once, next to the json.
find "$DATA_DIR" -maxdepth 2 -name '*.zip' -print0 | while IFS= read -r -d '' archive; do
  if [ ! -f "$archive.extracted" ]; then
    echo "Extracting $archive"
    python -m zipfile -e "$archive" "$(dirname "$archive")"
    touch "$archive.extracted"
  fi
done

if [ "${SKIP_BASE:-0}" != 1 ]; then
  download Qwen/Qwen2.5-VL-7B-Instruct model "$MODELS_DIR/Qwen2.5-VL-7B-Instruct"
fi

echo
echo "Annotation files in $DATA_DIR:"
find "$DATA_DIR" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' \) -printf '  %p\n'
echo "Next: python -m train.inspect_data --data_path <one of the files above> --image_root $DATA_DIR"
