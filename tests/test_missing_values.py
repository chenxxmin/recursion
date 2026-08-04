"""Unit tests for MISSING_PROB corruption in src/core.py RecurrenceDataset.

Covers: legacy path when disabled, corruption confined to positions >= init_len,
loss-mask alignment (target index = position - 1), clean test split, collate_fn
routing to BatchTag.DYNAMIC_MIXED, seed determinism, first_task_weight passthrough.

Requires torch (run on the training server): python -m pytest tests/test_missing_values.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

from core import BatchTag, RecurrenceDataset, collate_fn


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


def test_test_split_stays_clean():
    p = 7
    ds = _make_ds(1.0, p=p, num_mask=1)
    assert len(ds.test_samples) > 0
    for seq, mask in ds.test_samples:
        assert (seq < p).all()  # no missing token in test data
        assert (mask[:1] == 0).all() and (mask[1:] == 1).all()  # default num_mask mask


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


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('All tests passed.')
