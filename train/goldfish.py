"""Goldfish loss token mask (Hans et al., "Be like a Goldfish, Don't Memorize!", NeurIPS 2024).

Port of the "hash-table" strategy of `apply_goldfish` in the official implementation
(github.com/ahans30/goldfish-loss, lit_gpt/utils.py @ 35d29f9), which the paper's main results
and released configs use. For every next-token target, take the window of the
`context_width` targets ending at it (the target itself included), multiply their token ids,
look the product up in a fixed pseudo-random table, and drop the target from the loss when
the value is below 1/k. The mask depends only on the local text, so a passage is masked the
same way every time it appears. The first `context_width - 1` targets are never dropped.

Quirks kept from the reference for fidelity: the product key ignores token order and is 0
whenever the window contains token id 0. The reference builds the table on the training
device, so its exact values on CUDA differ from this CPU-built table; the drop rule and
seed are the same, and on CPU the masks are identical (tests/test_goldfish.py).
"""
import functools

import torch

STRATEGIES = ('hash-table',)
TABLE_SIZE = 1_000_003
TABLE_SEED = 2971215073  # fib47, as in the reference


@functools.lru_cache(maxsize=None)
def hash_table():
    rng = torch.Generator().manual_seed(TABLE_SEED)
    return torch.rand(TABLE_SIZE, generator=rng)


def goldfish_drop_mask(targets, k, context_width):
    """Boolean mask over `targets` (batch, seq) of the next-token targets the goldfish loss drops."""
    drop = torch.zeros_like(targets, dtype=torch.bool)
    if targets.shape[1] < context_width:
        return drop
    keys = targets.long().unfold(1, context_width, 1).prod(dim=-1) % TABLE_SIZE
    drop[:, context_width - 1:] = hash_table()[keys] < 1 / k
    return drop


def apply_goldfish(labels, input_ids, k, context_width, ignore_index=-100):
    """Return `labels` (unshifted, aligned with `input_ids`) with goldfish-dropped targets set to `ignore_index`.

    The hash reads the true next tokens (`input_ids[:, 1:]`), as the reference reads its raw
    `targets`, so prompt tokens that carry no label still shape the mask of the reply that follows.
    """
    labels = labels.clone()
    drop = goldfish_drop_mask(input_ids[:, 1:], k, context_width)
    labels[:, 1:][drop] = ignore_index
    return labels
