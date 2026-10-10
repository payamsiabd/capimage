"""Dual-branch Qwen2.5-VL: a frozen general language branch and a LoRA-personalised one, fused per token.

Every forward pass (training, prefill, and each decoding step) runs the language model twice
on the same inputs:

    h_general  = final hidden state with the LoRA adapters disabled (the pretrained model)
    h_personal = final hidden state with the LoRA adapters enabled
    h          = (1 - w) * h_general + w * h_personal        w = sigmoid(logit), learnable, init 0.1
    logits     = lm_head(h)

The loss and the next token both come from the fused logits. The general branch has no
trainable parameters, so it runs under `torch.no_grad()` (one extra forward, no extra
activation memory); `w` is a sigmoid so the fusion stays a weighted average. The vision tower
has no adapters and is identical for both branches.

Decoding keeps one KV cache per branch: `DualBranchCache` is the personalised branch's
`DynamicCache` and carries the general branch's in `.general`, so `generate()` (greedy,
sampling, beam search) works unchanged.

Usage: load a stock Qwen2.5-VL, call `to_dual_branch(model)`, then wrap it with PEFT using a
LoraConfig whose `modules_to_save` includes `GATE_MODULE` (so `w` trains and is saved with the
adapter). `load_dual_branch` rebuilds the model from a trained adapter for inference.
"""
import contextlib
import json
import math
import os

import torch
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers import DynamicCache, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

GATE_MODULE = 'personalization_gate'


class PersonalizationGate(torch.nn.Module):
    """Weighted average of the two branches; the personalised weight is sigmoid(logit)."""

    def __init__(self, init_weight=0.1):
        super().__init__()
        if not 0 < init_weight < 1:
            raise ValueError(f'personalisation weight must be in (0, 1), got {init_weight}')
        self.logit = torch.nn.Parameter(torch.tensor(math.log(init_weight / (1 - init_weight))))

    @property
    def weight(self):
        return torch.sigmoid(self.logit)

    def forward(self, general, personal):
        w = self.weight.to(device=general.device, dtype=general.dtype)  # device_map may split norm and lm_head
        return (1 - w) * general + w * personal


class DualBranchCache(DynamicCache):
    """KV cache of the personalised branch, carrying the general branch's cache in `.general`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.general = DynamicCache()

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        self.general.reorder_cache(beam_idx)

    def crop(self, max_length):
        super().crop(max_length)
        self.general.crop(max_length)

    def batch_repeat_interleave(self, repeats):
        super().batch_repeat_interleave(repeats)
        self.general.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self.general.batch_select_indices(indices)


class DualBranchQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL whose logits come from the fused general + personalised representations."""

    @contextlib.contextmanager
    def _adapters_disabled(self):
        # Same switch PEFT's `disable_adapter()` uses, limited to the LoRA layers (the gate stays active).
        layers = [m for m in self.modules() if isinstance(m, BaseTunerLayer) and not m.disable_adapters]
        for layer in layers:
            layer.enable_adapters(False)
        try:
            yield
        finally:
            for layer in layers:
                layer.enable_adapters(True)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None,
                rope_deltas=None, cache_position=None, second_per_grid_ts=None, logits_to_keep=0, **kwargs):
        if self.training:
            use_cache = False
        elif use_cache is None:
            use_cache = self.config.get_text_config().use_cache
        if past_key_values is None and use_cache:
            past_key_values = DualBranchCache()
        elif past_key_values is not None and not isinstance(past_key_values, DualBranchCache):
            if type(past_key_values) is not DynamicCache or past_key_values.get_seq_length() > 0:
                raise ValueError('Dual-branch decoding needs a DualBranchCache (an empty DynamicCache is replaced).')
            past_key_values = DualBranchCache()

        kwargs.pop('return_dict', None)  # always returns the output dataclass
        branch_inputs = dict(
            input_ids=input_ids, pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds,
            use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=True, cache_position=cache_position, **kwargs)
        with torch.no_grad(), self._adapters_disabled():
            general = self.model(past_key_values=past_key_values.general if use_cache else None, **branch_inputs)
        personal = self.model(past_key_values=past_key_values if use_cache else None, **branch_inputs)

        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden = getattr(self, GATE_MODULE)(general.last_hidden_state[:, keep], personal.last_hidden_state[:, keep])
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=past_key_values if use_cache else None,
            hidden_states=personal.hidden_states, attentions=personal.attentions, rope_deltas=personal.rope_deltas)


def to_dual_branch(model, init_weight=0.1):
    """Turn a loaded Qwen2.5-VL into the dual-branch model in place (weights, devices and hooks are kept)."""
    if isinstance(model, DualBranchQwen2_5_VLForConditionalGeneration):
        return model
    if type(model) is not Qwen2_5_VLForConditionalGeneration:
        raise TypeError(f'expected Qwen2_5_VLForConditionalGeneration, got {type(model).__name__}')
    model.__class__ = DualBranchQwen2_5_VLForConditionalGeneration
    model.add_module(GATE_MODULE, PersonalizationGate(init_weight).to(model.lm_head.weight.device))
    return model


def personalization_weight(model):
    """Current personalised weight w of a (possibly PEFT-wrapped) dual-branch model."""
    gate = next(m for name, m in model.named_modules() if name.split('.')[-1] == GATE_MODULE)
    device = next(gate.parameters()).device
    with torch.no_grad():  # calling the gate lets a PEFT wrapper route to its trained copy
        return gate(torch.zeros(1, device=device), torch.ones(1, device=device)).item()


def is_dual_branch_adapter(adapter_dir):
    with open(os.path.join(adapter_dir, 'adapter_config.json')) as f:
        return GATE_MODULE in (json.load(f).get('modules_to_save') or [])


def load_dual_branch(base, adapter_dir, **from_pretrained_kwargs):
    """Dual-branch PEFT model for inference from a base model (instance or path) and a trained adapter."""
    if not is_dual_branch_adapter(adapter_dir):
        raise ValueError(f'{adapter_dir} is not a dual-branch adapter (no {GATE_MODULE} in modules_to_save)')
    if isinstance(base, str):
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, **from_pretrained_kwargs)
    return PeftModel.from_pretrained(to_dual_branch(base), adapter_dir).eval()
