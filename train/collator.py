"""Batch chat records for Qwen2.5-VL supervised fine-tuning.

Loss is computed on assistant turns only: each assistant reply and its closing
``<|im_end|>`` are supervised; the system prompt, the question, image tokens and padding
are masked with -100.
"""
import logging
import math

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

IGNORE_INDEX = -100
VISION_TOKENS = ('<|vision_start|>', '<|vision_end|>', '<|image_pad|>', '<|video_pad|>')

logger = logging.getLogger(__name__)


def resize_by_token_budget(images, global_max_pixels=2000 * 28 * 28, per_img_max_pixels=1280 * 28 * 28, divisor=28):
    """Shrink a sample's images to fit a global (and then per-image) pixel budget.

    Same logic and defaults as Monet's src/utils.py (used by its SFT collator, which CapImagine
    trained with; Monet also returns the new sizes): images are left alone when their total fits
    the global budget, otherwise all are scaled by the same factor and each is further capped at
    `per_img_max_pixels`.
    """
    total = sum(img.width * img.height for img in images)
    if total <= global_max_pixels:
        return images
    ratio = math.sqrt(global_max_pixels / total)
    processed = []
    for img in images:
        w, h = int(img.width * ratio), int(img.height * ratio)
        w = max(divisor, (w // divisor) * divisor)
        h = max(divisor, (h // divisor) * divisor)
        if w * h > per_img_max_pixels:
            r = math.sqrt(per_img_max_pixels / (w * h))
            w = max(divisor, int(w * r) // divisor * divisor)
            h = max(divisor, int(h * r) // divisor * divisor)
        processed.append(img.resize((w, h), Image.BICUBIC))
    return processed


class QwenVLSFTCollator:
    def __init__(self, processor, max_length=8192, global_max_image_tokens=2000, per_image_max_tokens=1280):
        self.processor = processor
        self.max_length = max_length
        self.global_max_pixels = global_max_image_tokens * 28 * 28
        self.per_image_max_pixels = per_image_max_tokens * 28 * 28

        tok = processor.tokenizer
        # Truncation to max_length below cuts the batch's tail, which only drops padding when padding is on the right.
        tok.padding_side = 'right'
        self.im_start_id = tok.convert_tokens_to_ids('<|im_start|>')
        self.im_end_id = tok.convert_tokens_to_ids('<|im_end|>')
        self.assistant_ids = tok.encode('assistant', add_special_tokens=False)
        self.newline_ids = tok.encode('\n', add_special_tokens=False)
        self.vision_ids = [tok.convert_tokens_to_ids(t) for t in VISION_TOKENS]

    def build_labels(self, input_ids, attention_mask):
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        n_assistant = len(self.assistant_ids)
        for b, row in enumerate(input_ids.tolist()):
            i = 0
            while i < len(row):
                if row[i] == self.im_start_id and row[i + 1:i + 1 + n_assistant] == self.assistant_ids:
                    start = i + 1 + n_assistant
                    if row[start:start + len(self.newline_ids)] == self.newline_ids:
                        start += len(self.newline_ids)
                    end = start
                    while end < len(row) and row[end] != self.im_end_id:
                        end += 1
                    end = min(end + 1, len(row))  # supervise <|im_end|> so the model learns to stop
                    labels[b, start:end] = input_ids[b, start:end]
                    i = end
                else:
                    i += 1
        labels[attention_mask == 0] = IGNORE_INDEX
        for vid in self.vision_ids:
            labels[input_ids == vid] = IGNORE_INDEX
        return labels

    def __call__(self, batch):
        texts, images = [], []
        for messages in batch:
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            sample_images, _ = process_vision_info(messages)
            if sample_images:
                images.extend(resize_by_token_budget(sample_images, self.global_max_pixels, self.per_image_max_pixels))
        enc = self.processor(text=texts, images=images or None, return_tensors='pt', padding=True)
        enc['labels'] = self.build_labels(enc['input_ids'], enc['attention_mask'])

        if enc['input_ids'].shape[1] > self.max_length:
            tail = enc['input_ids'][:, self.max_length:]
            if any((tail == vid).any() for vid in self.vision_ids):
                raise ValueError(
                    f'A sample has image tokens beyond max_length={self.max_length}; raise --max_length '
                    'or lower --global_max_image_tokens.'
                )
            logger.warning('Truncating a batch of length %d to max_length=%d', enc['input_ids'].shape[1],
                           self.max_length)
            for key in ('input_ids', 'attention_mask', 'labels'):
                enc[key] = enc[key][:, :self.max_length]
        return enc
