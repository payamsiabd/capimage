"""Build a tiny, randomly initialised Qwen2.5-VL model and processor entirely offline.

Lets the training code run end to end on CPU in tests. The chat template is Qwen2.5-VL-7B-Instruct's.
"""
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor,
                          Qwen2TokenizerFast, Qwen2VLImageProcessor)
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

QWEN25VL_CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: "
    "{% endif %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SPECIAL_TOKENS = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>',
                  '<|image_pad|>', '<|video_pad|>']


def build_tiny_qwen_vl(out_dir, corpus):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus + ['system user assistant'], trainers.BpeTrainer(
        vocab_size=400, special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = Qwen2TokenizerFast(tokenizer_object=tok, eos_token='<|im_end|>', pad_token='<|endoftext|>',
                                   unk_token=None, additional_special_tokens=SPECIAL_TOKENS[1:])
    tokenizer.chat_template = QWEN25VL_CHAT_TEMPLATE
    processor = Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor(), tokenizer=tokenizer,
                                    video_processor=Qwen2VLVideoProcessor(), chat_template=QWEN25VL_CHAT_TEMPLATE)

    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
    config = Qwen2_5_VLConfig(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=len(tokenizer), max_position_embeddings=4096,
                         rope_scaling={'type': 'mrope', 'mrope_section': [2, 2, 4]}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           fullatt_block_indexes=[1], window_size=56, patch_size=14, spatial_merge_size=2,
                           temporal_patch_size=2),
        vocab_size=len(tokenizer), image_token_id=ids['<|image_pad|>'], video_token_id=ids['<|video_pad|>'],
        vision_start_token_id=ids['<|vision_start|>'], vision_end_token_id=ids['<|vision_end|>'],
        eos_token_id=ids['<|im_end|>'], pad_token_id=ids['<|endoftext|>'])
    model = Qwen2_5_VLForConditionalGeneration(config)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    return out_dir
