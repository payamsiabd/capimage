"""Run VLMEvalKit's run.py with an extra model class for the dual-branch Qwen2.5-VL (train/dual_branch.py).

`DualBranchQwen2VLChat` is VLMEvalKit's `Qwen2VLChat` (same prompt, resolution, decoding and
judge) whose model is the base Qwen2.5-VL turned dual-branch with a trained adapter, so each
generated token comes from the fused general + personalised representation. Config entry:

    {"class": "DualBranchQwen2VLChat", "model_path": <base model>, "adapter_path": <adapter>, ...}

    python scripts/vlmeval_dual.py <VLMEvalKit>/run.py --config <config.json> [run.py args]

scripts/run_vstar.sh calls this for MODEL=dual.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import vlmeval.vlm  # noqa: E402
from vlmeval.vlm import Qwen2VLChat  # noqa: E402

from train.dual_branch import load_dual_branch  # noqa: E402


class DualBranchQwen2VLChat(Qwen2VLChat):
    def __init__(self, model_path, adapter_path, **kwargs):
        if kwargs.get('use_vllm') or kwargs.get('use_lmdeploy'):
            raise ValueError('the dual-branch model runs on the transformers backend only (no USE_VLLM)')
        super().__init__(model_path=model_path, **kwargs)
        self.model = load_dual_branch(self.model, adapter_path)


vlmeval.vlm.DualBranchQwen2VLChat = DualBranchQwen2VLChat

if __name__ == '__main__':
    run_py = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(run_py)))
    runpy.run_path(run_py, run_name='__main__')
