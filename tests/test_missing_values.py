"""Unit tests for MISSING_PROB corruption in src/core.py RecurrenceDataset.

Covers: legacy path when disabled, corruption confined to positions >= init_len,
loss-mask alignment (target index = position - 1), test split corrupted with the
same rule as train, collate_fn routing to BatchTag.DYNAMIC_MIXED, seed
determinism, first_task_weight passthrough, and the mixed-rule equivalents.

Requires torch (run on the training server): python -m pytest tests/test_missing_values.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

from core import BatchTag, RecurrenceDataset, collate_fn
from rules import LinearRecurrenceRule


def _make_ds(missing_prob, seed=0, p=7, init_len=2, length=8, num_samples=20,
             num_mask=1, first_task_weight=1.0):
    def rec(seq, m):
        return (seq[-1] + seq[-2]) % m if init_len == 2 else (seq[-1] + seq[-2] + seq[-3]) % m
    random.seed(seed)
    ds = RecurrenceDataset(p=p, recurrence_fn=rec, init_len=init_len,
                           num_samples=num_samples, length=length, verbose=False,
                           missing_prob=missing_prob, num_mask=num_mask,
                           first_task_weight=first_task_weight)
    ds.run()
    return ds


def test_disabled_keeps_legacy_plain_tensors():
    ds = _make_ds(0.0)
    assert all(isinstance(s, torch.Tensor) for s in ds.train_samples)
    assert all(isinstance(s, torch.Tensor) for s in ds.test_samples)
    batch = collate_fn(ds.train_samples[:4])
    assert batch[1] == BatchTag.PLAIN


def test_train_windows_corrupted_and_masked():
    p, init_len, num_mask = 7, 2, 1
    ds = _make_ds(0.5, p=p, init_len=init_len, num_mask=num_mask)
    assert len(ds.train_samples) > 0
    saw_corruption = False
    for seq, mask in ds.train_samples:
        assert isinstance(seq, torch.Tensor) and isinstance(mask, torch.Tensor)
        assert mask.shape == (ds.length - 1,)
        # positions before init_len are never corrupted
        assert (seq[:init_len] < p).all()
        # mask prefix follows num_mask
        assert (mask[:num_mask] == 0).all()
        for pos in range(init_len, ds.length):
            if seq[pos].item() == p:  # corrupted (clean tokens are < p)
                saw_corruption = True
                assert mask[pos - 1].item() == 0.0
            elif pos - 1 >= num_mask:
                assert mask[pos - 1].item() == 1.0
    assert saw_corruption


def test_full_corruption_when_prob_one():
    p, init_len = 7, 3
    ds = _make_ds(1.0, p=p, init_len=init_len, num_mask=2)
    for seq, mask in ds.train_samples:
        assert (seq[:init_len] < p).all()
        assert (seq[init_len:] == p).all()
        # every corruptible position's prediction target is masked out
        assert (mask[init_len - 1:] == 0).all()


def test_test_split_corrupted_with_mask():
    """test windows are corrupted exactly like train: init values stay clean,
    every corruptible position becomes the missing token, and each corrupted
    position's prediction target is masked out."""
    p, init_len = 7, 2
    ds = _make_ds(1.0, p=p, init_len=init_len, num_mask=1)
    assert len(ds.test_samples) > 0
    for seq, mask in ds.test_samples:
        assert (seq[:init_len] < p).all()        # init values never corrupted
        assert (seq[init_len:] == p).all()       # prob=1.0 -> all later positions missing
        assert (mask[:1] == 0).all()             # num_mask prefix
        assert (mask[init_len - 1:] == 0).all()  # every corrupted target masked


def test_collate_routes_tuples_to_dynamic_mixed():
    ds = _make_ds(0.3)
    seqs, tag, masks = collate_fn(ds.train_samples[:4])
    assert tag == BatchTag.DYNAMIC_MIXED
    assert seqs.shape == (4, ds.length)
    assert masks.shape == (4, ds.length - 1)


def test_seed_determinism():
    a = _make_ds(0.4, seed=42)
    b = _make_ds(0.4, seed=42)
    for (sa, ma), (sb, mb) in zip(a.train_samples, b.train_samples):
        assert torch.equal(sa, sb) and torch.equal(ma, mb)


def test_first_task_weight_passthrough():
    ds = _make_ds(0.5, num_mask=2, first_task_weight=30.0, p=11, length=12, num_samples=40)
    seen_weighted = False
    for seq, mask in ds.train_samples:
        if mask[2].item() == 30.0:
            seen_weighted = True
        else:
            assert mask[2].item() == 0.0  # corrupted at position 3
    assert seen_weighted  # with prob 0.5 and 40 samples, position 3 stays clean somewhere


# ---------------- mixed_ab / mixed_abc (MixedRecurrenceDataset) ----------------

def _make_mixed(missing_prob, use_ab_tag, seed=0, p=7, length=8, num_samples=20,
                num_mask=2):
    from mixed_dataset import MixedRecurrenceDataset
    rules = [LinearRecurrenceRule(coeffs=(1, 1), p=p),
             LinearRecurrenceRule(coeffs=(1, 2), p=p)]
    random.seed(seed)
    return MixedRecurrenceDataset(rules=rules, num_samples=num_samples,
                                  length=length, verbose=False, use_ab_tag=use_ab_tag,
                                  missing_prob=missing_prob, num_mask=num_mask)


def test_mixed_disabled_keeps_pairs():
    from core import mixed_ab_collate_fn
    ds = _make_mixed(0.0, use_ab_tag=False)
    assert all(len(item) == 2 for item in ds.train_data)
    batch = mixed_ab_collate_fn(ds.train_data[:4])
    assert batch[1] == BatchTag.MIXED_AB


def test_mixed_basic_missing_alignment():
    from core import _unpack_batch, mixed_ab_collate_fn
    p, num_mask, miss = 7, 2, 7  # basic mode: missing token id == p
    ds = _make_mixed(0.5, use_ab_tag=False, p=p, num_mask=num_mask)
    saw = False
    for seq, mask, label in ds.train_data:
        assert label in (0, 1)
        assert mask.shape == (ds.length - 1,)
        assert (seq[:ds.order] < p).all()  # first `order` values stay clean
        for pos in range(ds.order, ds.length):
            corrupted = seq[pos].item() == miss
            saw = saw or corrupted
            expected = 0.0 if (corrupted or pos - 1 < num_mask) else 1.0
            assert mask[pos - 1].item() == expected, (pos, corrupted, mask)
    assert saw
    seqs, tag, labels, masks = mixed_ab_collate_fn(ds.train_data[:4])
    assert tag == BatchTag.MIXED_AB_MASKED
    assert labels.shape == (4,) and masks.shape == (4, ds.length - 1)
    x, loss_mask, kwargs, ab_labels = _unpack_batch((seqs, tag, labels, masks), 'cpu', None)
    assert loss_mask is not None and ab_labels is not None and kwargs == {}


def test_mixed_tag_missing_alignment():
    p, num_mask, n_rules = 7, 2, 2
    miss = p + n_rules  # tag mode: missing id avoids the flag tokens p..p+N-1
    ds = _make_mixed(0.5, use_ab_tag=True, p=p, num_mask=num_mask)
    saw = False
    for seq, mask, label in ds.train_data:
        assert seq.shape == (ds.length + 1,) and mask.shape == (ds.length,)
        assert seq[0].item() == p + label  # leading flag token
        # positions 1..order (x0..x_{order-1}) stay clean; corruptible from order+1
        assert (seq[1:ds.order + 1] < p).all()
        for pos in range(1, ds.length + 1):
            corrupted = seq[pos].item() == miss
            saw = saw or corrupted
            expected = 0.0 if (corrupted or pos - 1 < num_mask) else 1.0
            assert mask[pos - 1].item() == expected, (pos, corrupted, mask)
    assert saw


def test_mixed_test_split_corrupted():
    """Mixed test windows are corrupted like train, in both tag modes."""
    p, n_rules, order = 7, 2, 2
    for use_tag, miss in ((False, p), (True, p + n_rules)):
        ds = _make_mixed(1.0, use_ab_tag=use_tag, p=p)
        assert len(ds.test_data) > 0
        for seq, mask, label in ds.test_data:
            start = order + (1 if use_tag else 0)  # flag + init values stay clean
            assert (seq[:start] != miss).all()
            assert (seq[start:] == miss).all()     # prob=1.0 corrupts everything corruptible
            assert (mask == 0).all()               # prefix + all corrupted targets masked


def test_post_train_init_state_extraction_single():
    """The post-training generation test indexes raw sample lists; it must
    normalize (seq, mask) tuples via core._sample_seq."""
    from core import _sample_seq
    init_len = 2
    ds = _make_ds(0.5, init_len=init_len)
    for split_data in (ds.train_samples, ds.test_samples):
        for item in split_data:
            init_state = tuple(_sample_seq(item)[:init_len].tolist())
            assert len(init_state) == init_len
            assert all(v < ds.p for v in init_state)  # init values never corrupted


def test_post_train_init_state_extraction_mixed():
    """Mixed post-train loop: seq via _sample_seq, rule index is item[-1] for
    both (seq, label) and (seq, mask, label) items."""
    from core import _sample_seq
    for use_tag in (False, True):
        tag_offset = 1 if use_tag else 0
        ds = _make_mixed(0.5, use_ab_tag=use_tag)
        for item in ds.train_data:
            seq, rule_idx = _sample_seq(item), item[-1]
            assert rule_idx in (0, 1)
            init_state = tuple(seq[tag_offset:tag_offset + ds.order].tolist())
            assert len(init_state) == ds.order
            assert all(v < ds.p for v in init_state)


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('All tests passed.')