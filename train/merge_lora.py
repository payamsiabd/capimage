"""Merge a trained LoRA adapter into Qwen2.5-VL and save a standalone model.

The merged directory loads like any Qwen2.5-VL checkpoint, so it can be evaluated with
`MODEL=custom MODEL_PATH=<merged dir> bash scripts/run_vstar.sh`.

    python -m train.merge_lora --adapter checkpoints/capimagine-lora --output checkpoints/capimagine-lora-merged
"""
import argparse
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from train.dual_branch import is_dual_branch_adapter


def merge(adapter, output, base=None, dtype=torch.bfloat16):
    if is_dual_branch_adapter(adapter):
        raise ValueError(f'{adapter} is a dual-branch adapter: its general branch needs the unmerged base weights, so '
                         'it cannot become one model. Evaluate it with MODEL=dual ADAPTER_PATH=<adapter> '
                         'bash scripts/run_vstar.sh')
    base = base or PeftConfig.from_pretrained(adapter).base_model_name_or_path
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(output, safe_serialization=True)
    # The adapter dir carries the processor used in training (train_lora.py saves it there).
    has_processor = os.path.exists(os.path.join(adapter, 'preprocessor_config.json'))
    AutoProcessor.from_pretrained(adapter if has_processor else base).save_pretrained(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--adapter', required=True, help='dir with adapter_config.json (output_dir or a checkpoint-*)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--base', help='base model (default: the one recorded in adapter_config.json)')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    args = parser.parse_args()
    out = merge(args.adapter, args.output, args.base, getattr(torch, args.dtype))
    print(f'Merged model saved to {out}')


if __name__ == '__main__':
    main()
