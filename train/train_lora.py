"""LoRA fine-tuning of Qwen2.5-VL-7B-Instruct on CapImagine-Data with PEFT.

Recipe defaults follow the CapImagine paper (§5.1: Qwen2.5-VL-7B, batch size 1, gradient
accumulation 16) and the Monet SFT script it trained with (4 epochs, 10 warmup steps,
weight decay 0.01, bf16, frozen vision tower, per-sample image budget, system prompt
"You are a helpful assistant."). The paper fine-tunes all LLM weights; the LoRA settings
(rank 64, alpha 128, dropout 0.05, lr 1e-4, adapters on every attention and MLP projection
of the language model) are the usual LoRA counterparts, not values from the paper.

    torchrun --nproc-per-node 8 -m train.train_lora \
        --data_path data/CapImagine-Data/<file>.json --image_root data/CapImagine-Data \
        --output_dir checkpoints/capimagine-lora

Add `--goldfish_strategy hash-table --k_goldfish 4 --goldfish_context_width 13` to train with
the goldfish loss (train/goldfish.py) instead of the standard next-token loss.

Add `--dual_branch True` to train the dual-branch model (train/dual_branch.py): the loss is computed
on the fused representation (1 - w) * general + w * personalised, with w learnable from 0.1.

Every `transformers.TrainingArguments` flag is accepted as well.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoProcessor, HfArgumentParser, Qwen2_5_VLForConditionalGeneration, Trainer,
                          TrainingArguments, set_seed)
from transformers.trainer_utils import SaveStrategy

from train.collator import IGNORE_INDEX, QwenVLSFTCollator
from train.data import MONET_SYSTEM_PROMPT, CapImagineDataset
from train.dual_branch import GATE_MODULE, personalization_weight, to_dual_branch
from train.goldfish import STRATEGIES as GOLDFISH_STRATEGIES

logger = logging.getLogger(__name__)

# Attention and MLP projections of the language model only. The vision tower's MLP reuses the
# names gate_proj/up_proj/down_proj, but its layers live under `blocks.N`, not `layers.N`.
LLM_LINEAR_REGEX = r'.*\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'


@dataclass
class ModelArguments:
    model_name_or_path: str = 'Qwen/Qwen2.5-VL-7B-Instruct'
    attn_implementation: str = field(default='flash_attention_2', metadata={'help': 'flash_attention_2 | sdpa | eager'})


@dataclass
class DataArguments:
    data_path: list[str] = field(default=None, metadata={'help': 'CapImagine-Data .json/.jsonl file(s)'})
    image_root: Optional[str] = field(default=None, metadata={'help': 'base dir for relative image paths '
                                                                      '(default: dir of the first data file)'})
    system_prompt: Optional[str] = field(default=MONET_SYSTEM_PROMPT,
                                         metadata={'help': "replaces every system turn; 'none' keeps the data's own"})
    assistant_images: str = field(default='error', metadata={'help': 'error | drop images found in assistant turns'})
    max_length: int = 8192
    global_max_image_tokens: int = field(default=2000, metadata={'help': 'per-sample image token budget (Monet)'})
    per_image_max_tokens: int = field(default=1280, metadata={'help': 'per-image cap when over budget (Monet)'})
    max_samples: Optional[int] = field(default=None, metadata={'help': 'use only the first N records (debugging)'})


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = field(default=LLM_LINEAR_REGEX, metadata={'help': 'regex over module names'})


@dataclass
class GoldfishArguments:
    """Goldfish loss (Hans et al., NeurIPS 2024). Flag names and the h=13 default follow the official repo's config."""
    goldfish_strategy: Optional[str] = field(default=None, metadata={'help': "unset = standard loss; 'hash-table'"})
    k_goldfish: int = field(default=4, metadata={'help': 'drop about 1 in k supervised tokens'})
    goldfish_context_width: int = field(default=13, metadata={'help': 'tokens hashed per drop decision (h)'})


@dataclass
class DualBranchArguments:
    dual_branch: bool = field(default=False, metadata={'help': 'fuse a frozen general branch with the LoRA branch'})
    personalization_init: float = field(default=0.1, metadata={'help': 'initial weight w of the LoRA branch'})
    personalization_lr: float = field(default=1e-2, metadata={'help': 'learning rate of w (0 keeps it fixed). '
                                      'Adam moves a scalar by about lr per step, so at the LoRA lr w would barely move'})


@dataclass
class TrainArguments(TrainingArguments):
    learning_rate: float = 1e-4
    num_train_epochs: float = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_steps: int = 10
    weight_decay: float = 0.01
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: float = 1
    save_strategy: Union[SaveStrategy, str] = 'epoch'
    remove_unused_columns: Optional[bool] = False
    dataloader_num_workers: int = 4
    ddp_find_unused_parameters: Optional[bool] = False
    report_to: Union[None, str, list[str]] = 'none'


def token_weighted_ce(outputs, labels, num_items_in_batch=None):
    """Next-token cross-entropy summed over supervised tokens and divided by the tokens in the whole step.

    Qwen2.5-VL's forward (transformers 4.54) accepts but drops `num_items_in_batch`, so its built-in
    loss is a per-micro-batch mean that Trainer then sums over gradient-accumulation steps, inflating
    loss and gradients by the accumulation factor. Passing this as `compute_loss_func` makes one
    optimizer step equal a single large batch, across accumulation steps and GPUs.
    """
    logits = outputs.logits[:, :-1].float()
    targets = labels[:, 1:].to(logits.device)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                             ignore_index=IGNORE_INDEX, reduction='sum')
    n_tokens = num_items_in_batch if num_items_in_batch is not None else (targets != IGNORE_INDEX).sum()
    return loss / n_tokens


class LoraTrainer(Trainer):
    """Trainer that, for the dual-branch model, gives the fusion weight its own lr (no weight decay) and logs it."""

    def __init__(self, *args, personalization_lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.personalization_lr = personalization_lr

    def create_optimizer(self):
        if self.optimizer is not None or self.personalization_lr is None:
            return super().create_optimizer()
        params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        gate = {n for n, _ in params if GATE_MODULE in n}
        decay = set(self.get_decay_parameter_names(self.model)) - gate
        groups = [
            {'params': [p for n, p in params if n in decay], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in params if n not in decay and n not in gate], 'weight_decay': 0.0},
            {'params': [p for n, p in params if n in gate], 'weight_decay': 0.0, 'lr': self.personalization_lr},
        ]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
        self.optimizer = optimizer_cls([g for g in groups if g['params']], **optimizer_kwargs)
        return self.optimizer

    def log(self, logs, *args, **kwargs):
        if self.personalization_lr is not None:
            logs['personalization_weight'] = personalization_weight(self.model)
        super().log(logs, *args, **kwargs)


def build_model(model_args, lora_args, train_args, dual_args):
    dtype = torch.bfloat16 if train_args.bf16 else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, torch_dtype=dtype, attn_implementation=model_args.attn_implementation)
    model.config.use_cache = False
    if train_args.gradient_checkpointing:
        model.enable_input_require_grads()
    if dual_args.dual_branch:
        model = to_dual_branch(model, dual_args.personalization_init)

    lora_config = LoraConfig(r=lora_args.lora_r, lora_alpha=lora_args.lora_alpha, lora_dropout=lora_args.lora_dropout,
                             target_modules=lora_args.lora_target_modules, bias='none', task_type='CAUSAL_LM',
                             modules_to_save=[GATE_MODULE] if dual_args.dual_branch else None)
    model = get_peft_model(model, lora_config)
    lora_modules = [n for n, _ in model.named_modules() if n.endswith('.lora_A')]
    if not lora_modules:
        raise ValueError(f'--lora_target_modules matched no modules: {lora_args.lora_target_modules}')
    if any('.visual.' in n for n in lora_modules):
        raise ValueError('--lora_target_modules matched vision-tower layers; the vision tower stays frozen')
    return model


def log_example(dataset, collator, tokenizer):
    batch = collator([dataset[0]])
    labels = batch['labels'][0]
    supervised = tokenizer.decode(batch['input_ids'][0][labels != IGNORE_INDEX])
    logger.info('Example 0: %d tokens, %d supervised. Supervised text:\n%s',
                batch['input_ids'].shape[1], int((labels != IGNORE_INDEX).sum()), supervised[:2000])
    if collator.goldfish_k:
        standard = collator.build_labels(batch['input_ids'], batch['attention_mask'])[0]
        n_standard, n_kept = int((standard != IGNORE_INDEX).sum()), int((labels != IGNORE_INDEX).sum())
        logger.info('Goldfish loss (k=%d, h=%d) drops %d of %d supervised tokens in example 0 (about 1/k expected).',
                    collator.goldfish_k, collator.goldfish_context_width, n_standard - n_kept, n_standard)


def main(argv=None):
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, GoldfishArguments, DualBranchArguments,
                               TrainArguments))
    model_args, data_args, lora_args, goldfish_args, dual_args, train_args = \
        parser.parse_args_into_dataclasses(args=argv)
    if not data_args.data_path or not train_args.output_dir:
        parser.error('--data_path and --output_dir are required')
    if goldfish_args.goldfish_strategy is not None:
        if goldfish_args.goldfish_strategy not in GOLDFISH_STRATEGIES:
            parser.error(f'--goldfish_strategy must be one of {GOLDFISH_STRATEGIES}')
        if goldfish_args.k_goldfish < 2 or goldfish_args.goldfish_context_width < 1:
            parser.error('--k_goldfish must be >= 2 and --goldfish_context_width >= 1')
    if train_args.gradient_checkpointing and train_args.gradient_checkpointing_kwargs is None:
        train_args.gradient_checkpointing_kwargs = {'use_reentrant': False}
    logging.basicConfig(level=logging.INFO if train_args.local_rank in (-1, 0) else logging.WARNING,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    set_seed(train_args.seed)

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    processor.tokenizer.padding_side = 'right'

    system_prompt = None if (data_args.system_prompt or '').lower() == 'none' else data_args.system_prompt
    image_root = data_args.image_root or os.path.dirname(os.path.abspath(data_args.data_path[0]))
    dataset = CapImagineDataset(data_args.data_path, image_root, system_prompt=system_prompt,
                                assistant_images=data_args.assistant_images, max_samples=data_args.max_samples)
    logger.info('Loaded %d training samples (image root: %s)', len(dataset), image_root)

    collator = QwenVLSFTCollator(processor, data_args.max_length, data_args.global_max_image_tokens,
                                 data_args.per_image_max_tokens,
                                 goldfish_k=goldfish_args.k_goldfish if goldfish_args.goldfish_strategy else None,
                                 goldfish_context_width=goldfish_args.goldfish_context_width)
    if train_args.local_rank in (-1, 0):
        log_example(dataset, collator, processor.tokenizer)

    model = build_model(model_args, lora_args, train_args, dual_args)
    if train_args.local_rank in (-1, 0):
        model.print_trainable_parameters()

    trainer = LoraTrainer(model=model, args=train_args, train_dataset=dataset, data_collator=collator,
                          processing_class=processor, compute_loss_func=token_weighted_ce,
                          personalization_lr=dual_args.personalization_lr if dual_args.dual_branch else None)
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    trainer.save_model(train_args.output_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(train_args.output_dir)
    return trainer


if __name__ == '__main__':
    main()
