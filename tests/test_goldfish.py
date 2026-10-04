"""Goldfish mask: bit-exact agreement with the official implementation, plus the paper's properties."""
import os
import sys

import pytest

torch = pytest.importorskip('torch')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train.goldfish import apply_goldfish, goldfish_drop_mask  # noqa: E402

QWEN_VOCAB = 151_665


# --- Verbatim from github.com/ahans30/goldfish-loss, lit_gpt/utils.py @ 35d29f9 (hash-table branch). ---
hash_table = None
table_size = 1_000_003


def _load_hash_table(device):
    global hash_table
    rng = torch.Generator(device=device)
    rng.manual_seed(2971215073)  # fib47 is prime
    hash_table = torch.rand(table_size, device=device, generator=rng)


def reference_apply_goldfish(targets, k, ignore_index, goldfish_context_width):
    device = targets.device
    masked_targets = targets.clone()
    global hash_table
    if hash_table is None:
        _load_hash_table(device)
    hashed_keys = hash_table[targets.unfold(1, goldfish_context_width, 1).prod(dim=-1) % table_size]
    dropped_token_indices = (hashed_keys < 1 / k)
    masked_targets[:, goldfish_context_width-1:][dropped_token_indices] = ignore_index
    return masked_targets
# --- end of reference code ---


@pytest.mark.parametrize('k, h', [(2, 1), (3, 4), (4, 13), (8, 13), (4, 32)])
def test_matches_reference_implementation(k, h):
    gen = torch.Generator().manual_seed(k * 100 + h)
    targets = torch.randint(0, QWEN_VOCAB, (3, 500), generator=gen)
    targets[1, 50:80] = 0  # token id 0 zeroes the product key in both implementations
    expected = reference_apply_goldfish(targets, k, ignore_index=-100, goldfish_context_width=h) == -100
    assert torch.equal(goldfish_drop_mask(targets, k, h), expected)


def test_drop_rate_is_one_in_k():
    targets = torch.randint(0, QWEN_VOCAB, (1, 200_000), generator=torch.Generator().manual_seed(0))
    for k in (3, 4, 8):
        rate = goldfish_drop_mask(targets, k, 13).float().mean().item()
        assert abs(rate - 1 / k) < 0.01, (k, rate)


def test_mask_depends_only_on_local_text():
    gen = torch.Generator().manual_seed(1)
    passage = torch.randint(0, QWEN_VOCAB, (60,), generator=gen)
    a = torch.cat([torch.randint(0, QWEN_VOCAB, (17,), generator=gen), passage])
    b = torch.cat([torch.randint(0, QWEN_VOCAB, (101,), generator=gen), passage,
                   torch.randint(0, QWEN_VOCAB, (5,), generator=gen)])
    h = 13
    drop_a = goldfish_drop_mask(a[None], 4, h)[0, 17:][h - 1:]
    drop_b = goldfish_drop_mask(b[None], 4, h)[0, 101:161][h - 1:]
    assert torch.equal(drop_a, drop_b)  # same passage, same mask, wherever it appears
    assert drop_a.any() and not drop_a.all()


def test_first_targets_and_short_sequences_are_never_dropped():
    targets = torch.randint(0, QWEN_VOCAB, (4, 200), generator=torch.Generator().manual_seed(3))
    drop = goldfish_drop_mask(targets, 2, 13)
    assert not drop[:, :12].any() and drop[:, 12:].any()
    assert not goldfish_drop_mask(targets[:, :12], 2, 13).any()


def test_apply_goldfish_only_removes_supervised_targets():
    gen = torch.Generator().manual_seed(2)
    input_ids = torch.randint(0, QWEN_VOCAB, (2, 300), generator=gen)
    labels = input_ids.clone()
    labels[:, :120] = -100  # prompt
    labels[1, 250:] = -100  # padding
    out = apply_goldfish(labels, input_ids, 4, 13)

    drop = goldfish_drop_mask(input_ids[:, 1:], 4, 13)  # targets are the true next tokens
    assert torch.equal(out[:, 1:], torch.where(drop, torch.full_like(labels[:, 1:], -100), labels[:, 1:]))
    assert torch.equal(out[:, 0], labels[:, 0])
    kept = out != -100
    assert (kept <= (labels != -100)).all()  # never adds supervision
    assert 0 < kept.sum() < (labels != -100).sum()
    assert torch.equal(labels[:, :120], torch.full_like(labels[:, :120], -100))  # input not modified in place
