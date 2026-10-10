"""Dual-branch Qwen2.5-VL: fusion math, per-branch KV caches in generate(), training and reloading.

Runs a tiny random Qwen2.5-VL on CPU; needs torch, transformers, peft and qwen-vl-utils.
"""
import json
import math
import os
import sys

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('peft')
pytest.importorskip('qwen_vl_utils')
from peft import LoraConfig, get_peft_model  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from tiny_qwen_vl import build_tiny_qwen_vl  # noqa: E402
from train.collator import QwenVLSFTCollator  # noqa: E402
from train.data import CapImagineDataset  # noqa: E402
from train.dual_branch import (GATE_MODULE, DualBranchCache, PersonalizationGate, is_dual_branch_adapter,  # noqa: E402
                               load_dual_branch, personalization_weight, to_dual_branch)
from train.train_lora import LLM_LINEAR_REGEX  # noqa: E402

QUESTION = 'What is the color of the umbrella?\nOptions:\nA. red\nB. blue'
REPLY = 'The umbrella is red. \\boxed{A}'


@pytest.fixture(scope='module')
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp('dual')
    records = []
    for i, size in enumerate([(112, 84), (84, 112)]):
        Image.new('RGB', size, color=(60 * i, 120, 30)).save(root / f'{i}.png')
        records.append({'data': [
            {'role': 'user', 'content': [{'type': 'image', 'image': f'{i}.png'}, {'type': 'text', 'text': QUESTION}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': REPLY}]}]})
    (root / 'train.json').write_text(json.dumps(records))
    model_dir = build_tiny_qwen_vl(str(root / 'tiny'), [QUESTION, REPLY] * 20)
    processor = AutoProcessor.from_pretrained(model_dir)
    dataset = CapImagineDataset([str(root / 'train.json')], str(root))
    batch = QwenVLSFTCollator(processor)([dataset[0], dataset[1]])
    batch.pop('labels')
    prompt_messages = [m for m in dataset[0] if m['role'] != 'assistant']
    text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt = processor(text=[text], images=[Image.open(root / '0.png')], return_tensors='pt')
    return root, model_dir, dict(batch), dict(prompt)


def base_model(model_dir):
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir).eval()


def dual_model(model_dir, random_lora=True):
    torch.manual_seed(0)
    config = LoraConfig(r=8, lora_alpha=64, target_modules=LLM_LINEAR_REGEX, modules_to_save=[GATE_MODULE],
                        init_lora_weights=not random_lora)  # False: random B, so the two branches differ
    return get_peft_model(to_dual_branch(base_model(model_dir)), config).eval()


def test_gate_is_a_weighted_average_starting_at_one_tenth():
    gate = PersonalizationGate(0.1)
    assert gate.weight.item() == pytest.approx(0.1, abs=1e-6)
    general, personal = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
    torch.testing.assert_close(gate(general, personal), 0.9 * general + 0.1 * personal)
    with pytest.raises(ValueError):
        PersonalizationGate(1.0)


def test_fixed_gate_has_no_parameters_but_keeps_its_value_in_the_state_dict():
    gate = PersonalizationGate(0.1, trainable=False)
    assert list(gate.parameters()) == [] and list(gate.state_dict()) == ['logit']
    assert gate.weight.item() == pytest.approx(0.1, abs=1e-6)
    general, personal = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
    torch.testing.assert_close(gate(general, personal), 0.9 * general + 0.1 * personal)
    trainable = PersonalizationGate(0.5)
    trainable.load_state_dict(gate.state_dict())  # same key either way, so adapters load in both modes
    assert trainable.weight.item() == pytest.approx(0.1, abs=1e-6)


def test_logits_come_from_the_fused_final_representations(workspace):
    _, model_dir, batch, _ = workspace
    model = dual_model(model_dir)
    inner = model.get_base_model()
    with torch.no_grad():
        logits = model(**batch).logits
        personal = inner.model(**batch).last_hidden_state
        with model.disable_adapter():
            general = inner.model(**batch).last_hidden_state
        reference = base_model(model_dir)(**batch).logits
    assert personalization_weight(model) == pytest.approx(0.1, abs=1e-6)
    assert not torch.allclose(general, personal, atol=1e-3), 'LoRA should change the personalised branch'
    torch.testing.assert_close(logits, inner.lm_head(0.9 * general + 0.1 * personal))
    torch.testing.assert_close(inner.lm_head(general), reference)  # the general branch is the pretrained model


def test_untrained_or_disabled_adapter_reproduces_the_pretrained_model(workspace):
    _, model_dir, batch, _ = workspace
    with torch.no_grad():
        reference = base_model(model_dir)(**batch).logits
        fresh = dual_model(model_dir, random_lora=False)  # LoRA B = 0: both branches are the base model
        torch.testing.assert_close(fresh(**batch).logits, reference)
        trained = dual_model(model_dir)
        with trained.disable_adapter():
            torch.testing.assert_close(trained(**batch).logits, reference)
        assert not torch.allclose(trained(**batch).logits, reference, atol=1e-3)  # re-enabled afterwards


def test_each_branch_keeps_its_own_cache(workspace):
    _, model_dir, _, prompt = workspace
    with torch.no_grad():
        out = dual_model(model_dir)(**prompt, use_cache=True)
    cache = out.past_key_values
    assert isinstance(cache, DualBranchCache)
    n = prompt['input_ids'].shape[1]
    assert cache.get_seq_length() == cache.general.get_seq_length() == n
    assert not torch.allclose(cache.layers[0].keys, cache.general.layers[0].keys, atol=1e-4)


def test_cached_decoding_matches_recomputing_both_branches(workspace):
    _, model_dir, _, prompt = workspace
    model = dual_model(model_dir)
    out = model.generate(**prompt, max_new_tokens=12, min_new_tokens=12, do_sample=False,
                         return_dict_in_generate=True, output_logits=True)
    cached = torch.stack(out.logits, dim=1)  # each step decoded from the two per-branch KV caches
    with torch.no_grad():  # one no-cache pass over the generated sequence re-runs both branches from scratch
        full = model(input_ids=out.sequences, attention_mask=torch.ones_like(out.sequences),
                     pixel_values=prompt['pixel_values'], image_grid_thw=prompt['image_grid_thw'],
                     use_cache=False).logits
    n = prompt['input_ids'].shape[1]
    torch.testing.assert_close(cached, full[:, n - 1:-1], atol=1e-4, rtol=1e-4)


def test_cache_operations_apply_to_both_branches():
    cache = DualBranchCache()
    for c, offset in ((cache, 0.0), (cache.general, 100.0)):
        states = (torch.arange(3.0) + offset).view(3, 1, 1, 1).expand(3, 2, 4, 8).contiguous()
        c.update(states, states.clone(), layer_idx=0)

    def rows(c):
        return c.layers[0].keys[:, 0, 0, 0].tolist()

    cache.reorder_cache(torch.tensor([2, 0, 1]))  # beam search
    assert rows(cache) == [2, 0, 1] and rows(cache.general) == [102, 100, 101]
    cache.batch_repeat_interleave(2)
    assert rows(cache.general) == [102, 102, 100, 100, 101, 101]
    cache.batch_select_indices(torch.tensor([1, 4]))
    assert rows(cache) == [2, 1] and rows(cache.general) == [102, 101]
    cache.crop(3)
    assert cache.get_seq_length() == cache.general.get_seq_length() == 3


def test_beam_search_runs(workspace):
    _, model_dir, _, prompt = workspace
    out = dual_model(model_dir).generate(**prompt, max_new_tokens=6, min_new_tokens=6, num_beams=3, do_sample=False)
    assert out.shape == (1, prompt['input_ids'].shape[1] + 6)


def test_training_learns_the_weight_and_the_adapter_reloads(workspace, tmp_path):
    from train.merge_lora import merge
    from train.train_lora import main
    root, model_dir, batch, _ = workspace
    out = tmp_path / 'dual'
    trainer = main([
        '--model_name_or_path', model_dir, '--data_path', str(root / 'train.json'), '--output_dir', str(out),
        '--attn_implementation', 'sdpa', '--bf16', 'False', '--max_steps', '4', '--gradient_accumulation_steps', '1',
        '--learning_rate', '1e-2', '--lr_scheduler_type', 'constant', '--warmup_steps', '0',
        '--dataloader_num_workers', '0', '--save_strategy', 'no', '--lora_r', '8', '--lora_alpha', '16',
        '--dual_branch', 'True', '--personalization_lr', '0.05'])

    history = trainer.state.log_history
    losses = [h['loss'] for h in history if 'loss' in h]
    weights = [h['personalization_weight'] for h in history if 'personalization_weight' in h]
    vocab = trainer.model.get_base_model().config.vocab_size
    assert abs(losses[0] - math.log(vocab)) < 0.5, losses
    assert weights[0] == pytest.approx(0.1, abs=1e-6)  # step 1: LoRA B = 0, both branches agree, no gradient on w
    assert abs(weights[-1] - 0.1) > 1e-3, weights  # then w trains
    gate_group = next(g for g in trainer.optimizer.param_groups if g['lr'] == 0.05)
    assert gate_group['weight_decay'] == 0.0 and len(gate_group['params']) == 1

    assert is_dual_branch_adapter(str(out))
    reloaded = load_dual_branch(model_dir, str(out))
    assert personalization_weight(reloaded) == pytest.approx(weights[-1], abs=1e-6)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**batch).logits, trainer.model.eval()(**batch).logits)
    with pytest.raises(ValueError, match='dual-branch'):
        merge(str(out), str(tmp_path / 'merged'), base=model_dir)


def test_training_with_a_fixed_weight(workspace, tmp_path):
    from train.train_lora import main
    root, model_dir, batch, _ = workspace
    out = tmp_path / 'dual-fixed'
    trainer = main([
        '--model_name_or_path', model_dir, '--data_path', str(root / 'train.json'), '--output_dir', str(out),
        '--attn_implementation', 'sdpa', '--bf16', 'False', '--max_steps', '4', '--gradient_accumulation_steps', '1',
        '--learning_rate', '1e-2', '--lr_scheduler_type', 'constant', '--warmup_steps', '0',
        '--dataloader_num_workers', '0', '--save_strategy', 'no', '--lora_r', '8', '--lora_alpha', '16',
        '--dual_branch', 'True', '--personalization_init', '0.25', '--personalization_trainable', 'False',
        '--personalization_lr', '0.05'])

    weights = [h['personalization_weight'] for h in trainer.state.log_history if 'personalization_weight' in h]
    assert len(weights) >= 4 and all(w == pytest.approx(0.25, abs=1e-6) for w in weights), weights  # steps + summary
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n for n in trainable)  # only LoRA trains; w is not a parameter
    optimized = sum(len(g['params']) for g in trainer.optimizer.param_groups)
    assert optimized == len(trainable)
    assert any(p.abs().sum() > 0 for n, p in trainer.model.named_parameters() if 'lora_B' in n)  # LoRA did train

    reloaded = load_dual_branch(model_dir, str(out))  # builds a 0.1 gate, then loads the saved 0.25
    assert personalization_weight(reloaded) == pytest.approx(0.25, abs=1e-6)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**batch).logits, trainer.model.eval()(**batch).logits)
