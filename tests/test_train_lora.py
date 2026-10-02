"""LoRA training pipeline tests: data normalisation, label masking, and a CPU end-to-end run.

The end-to-end tests train a tiny random Qwen2.5-VL for a few steps, so they need torch,
transformers, peft and qwen-vl-utils (scripts/setup_env.sh) but no GPU or downloads.
"""
import json
import os
import sys

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('peft')
pytest.importorskip('qwen_vl_utils')
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from tiny_qwen_vl import build_tiny_qwen_vl  # noqa: E402
from train.collator import IGNORE_INDEX, QwenVLSFTCollator, resize_by_token_budget  # noqa: E402
from train.data import CapImagineDataset, DataFormatError, normalize_record  # noqa: E402

QUESTION = 'What is the color of the umbrella?\nOptions:\nA. red\nB. blue'
REPLY = ('I need to find the umbrella first.<think_image>Suppose the image had been zoomed into the lower left, '
         'showing a person holding a red umbrella.</think_image> The umbrella is red. \\boxed{A}')


def monet_record(image):
    return {'data': [
        {'role': 'system', 'content': [{'type': 'text', 'text': 'Some other system prompt.'}]},
        {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': QUESTION}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': REPLY}]},
    ], 'metadata': {'source': 'Visual_CoT'}}


@pytest.fixture(autouse=True)
def grad_enabled():
    # Importing vlmeval (tests/test_vlmevalkit_integration.py) runs torch.set_grad_enabled(False) globally.
    with torch.enable_grad():
        yield


@pytest.fixture(scope='module')
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp('capimagine')
    (root / 'images').mkdir()
    records = []
    for i, size in enumerate([(112, 84), (140, 112), (84, 84), (168, 112)]):
        Image.new('RGB', size, color=(40 * i, 90, 160)).save(root / 'images' / f'{i}.png')
        records.append(monet_record(f'images/{i}.png'))
    with open(root / 'train.json', 'w') as f:
        json.dump(records, f)
    model_dir = build_tiny_qwen_vl(str(root / 'tiny-qwen-vl'), [QUESTION, REPLY] * 20)
    return root, model_dir


def test_normalize_monet_record_overrides_system_prompt():
    messages = normalize_record(monet_record('images/0.png'), '/data')
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant']
    assert messages[0]['content'][0]['text'] == 'You are a helpful assistant.'
    assert messages[1]['content'][0] == {'type': 'image', 'image': '/data/images/0.png'}
    assert normalize_record(monet_record('x.png'), '/d', system_prompt=None)[0]['content'][0]['text'] == \
        'Some other system prompt.'


def test_normalize_llava_record():
    record = {'image': 'a.jpg', 'conversations': [
        {'from': 'human', 'value': '<image>\nWhat is shown?'}, {'from': 'gpt', 'value': 'A cat.'}]}
    messages = normalize_record(record, '/d')
    assert messages[1]['content'] == [{'type': 'image', 'image': '/d/a.jpg'}, {'type': 'text', 'text': 'What is shown?'}]
    assert messages[2]['content'] == [{'type': 'text', 'text': 'A cat.'}]


def test_assistant_images_error_or_drop():
    record = monet_record('q.png')
    record['data'][2]['content'] = [
        {'type': 'text', 'text': 'Zoom in.<abs_vis_token></abs_vis_token>'},
        {'type': 'image', 'image': 'aux.png'}, {'type': 'text', 'text': ' It is red.'}]
    with pytest.raises(DataFormatError, match='assistant turn contains an image'):
        normalize_record(record, '/d')
    reply = normalize_record(record, '/d', assistant_images='drop')[2]['content']
    assert reply == [{'type': 'text', 'text': 'Zoom in.'}, {'type': 'text', 'text': ' It is red.'}]


def test_dataset_reports_bad_record(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps([monet_record('a.png'), {'foo': 1}]))
    with pytest.raises(DataFormatError, match='record 1'):
        CapImagineDataset([str(path)], str(tmp_path))


def test_resize_by_token_budget():
    small = [Image.new('RGB', (280, 280))]
    assert resize_by_token_budget(small, global_max_pixels=280 * 280) is small
    big = [Image.new('RGB', (560, 560)), Image.new('RGB', (560, 280))]
    out = resize_by_token_budget(big, global_max_pixels=100 * 28 * 28, per_img_max_pixels=60 * 28 * 28)
    assert all(im.width % 28 == 0 and im.height % 28 == 0 for im in out)
    assert sum(im.width * im.height for im in out) <= 100 * 28 * 28


def test_token_weighted_ce_matches_one_big_batch():
    from types import SimpleNamespace
    from train.train_lora import token_weighted_ce
    torch.manual_seed(0)
    logits = torch.randn(2, 7, 11)
    labels = torch.randint(0, 11, (2, 7))
    labels[0, :5] = IGNORE_INDEX  # micro-batches with different numbers of supervised tokens
    full = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1),
                                             ignore_index=IGNORE_INDEX)
    n_total = (labels[:, 1:] != IGNORE_INDEX).sum()
    accumulated = sum(token_weighted_ce(SimpleNamespace(logits=logits[i:i + 1]), labels[i:i + 1], n_total)
                      for i in range(2))
    torch.testing.assert_close(accumulated, full)


def test_labels_cover_only_assistant_reply(workspace):
    from transformers import AutoProcessor
    root, model_dir = workspace
    processor = AutoProcessor.from_pretrained(model_dir)
    processor.tokenizer.padding_side = 'right'
    dataset = CapImagineDataset([str(root / 'train.json')], str(root))
    batch = QwenVLSFTCollator(processor)([dataset[0], dataset[1]])

    assert batch['pixel_values'].shape[0] > 0 and batch['image_grid_thw'].shape == (2, 3)
    for row in range(2):
        supervised = batch['input_ids'][row][batch['labels'][row] != IGNORE_INDEX]
        assert processor.tokenizer.decode(supervised) == REPLY + '<|im_end|>'
    # The shorter sample is right-padded; padding is never supervised.
    assert (batch['labels'][batch['attention_mask'] == 0] == IGNORE_INDEX).all()


def test_train_merge_end_to_end(workspace, tmp_path):
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from train.merge_lora import merge
    from train.train_lora import main

    root, model_dir = workspace
    out = tmp_path / 'lora'
    trainer = main([
        '--model_name_or_path', model_dir, '--data_path', str(root / 'train.json'), '--output_dir', str(out),
        '--attn_implementation', 'sdpa', '--bf16', 'False', '--max_steps', '3', '--gradient_accumulation_steps', '2',
        '--learning_rate', '1e-2', '--dataloader_num_workers', '0', '--save_strategy', 'no', '--lora_r', '8',
        '--lora_alpha', '16'])

    losses = [h['loss'] for h in trainer.state.log_history if 'loss' in h]
    assert len(losses) == 3 and all(torch.isfinite(torch.tensor(losses)))
    # A random model scores ~ln(vocab) per token; gradient accumulation must not multiply it.
    vocab = trainer.model.get_base_model().config.vocab_size
    assert abs(losses[0] - torch.log(torch.tensor(float(vocab)))) < 0.5, losses
    lora_names = [n for n, _ in trainer.model.named_modules() if n.endswith('lora_A')]
    assert lora_names and all('language_model.layers' in n for n in lora_names)
    assert len(lora_names) == 2 * 7  # 2 layers x (q, k, v, o, gate, up, down)
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'preprocessor_config.json'):
        assert (out / name).exists(), name

    merged_dir = merge(str(out), str(tmp_path / 'merged'), base=model_dir, dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(merged_dir)
    batch = QwenVLSFTCollator(processor)([CapImagineDataset([str(root / 'train.json')], str(root))[0]])
    inputs = {k: v for k, v in batch.items() if k != 'labels'}
    with torch.no_grad():
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir).eval()
        base_logits = base(**inputs).logits
        peft_logits = PeftModel.from_pretrained(base, str(out)).eval()(**inputs).logits
        merged_logits = Qwen2_5_VLForConditionalGeneration.from_pretrained(merged_dir).eval()(**inputs).logits
    assert not torch.allclose(base_logits, peft_logits, atol=1e-4), 'adapter had no effect'
    torch.testing.assert_close(merged_logits, peft_logits, atol=1e-4, rtol=1e-4)
