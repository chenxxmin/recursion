"""Unit tests for on-the-fly MISSING_PROB corruption (predict-only regime).

Datasets store CLEAN windows only; make_missing_collate /
make_mixed_missing_collate corrupt each batch with fresh randomness, so every
epoch sees different missing positions. Predict mode is the ONLY regime now:
collates return (views, PLAIN_TARGET, cleans) / (views, MIXED_AB_TARGET,
labels, cleans) — the loss/accuracy targets are the clean values, so corrupted
positions must be filled in. num_mask/first_task_weight are applied by the
training loop's _default_loss_mask, not by the collate. Covers: clean storage,
run structure (MISS_LEN), deterministic patterns at prob=1.0, per-call fresh
randomness, and unpack routing.

Requires torch (run on the training server): python tests/test_missing_values.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

from core import BatchTag, RecurrenceDataset, collate_fn
from datasets import make_missing_collate, make_mixed_missing_collate
from rules import LinearRecurrenceRule

P, INIT_LEN, LEN = 7, 2, 8


def _make_ds(seed=0, p=P, init_len=INIT_LEN, length=LEN, num_samples=20):
    def rec(seq, m):
        return (seq[-1] + seq[-2]) % m if init_len == 2 else (seq[-1] + seq[-2] + seq[-3]) % m
    random.seed(seed)
    ds = RecurrenceDataset(p=p, recurrence_fn=rec, init_len=init_len,
                           num_samples=num_samples, length=length, verbose=False)
    ds.run()
    return ds


def _collate(missing_prob=0.5, p=P, init_len=INIT_LEN, miss_len=1):
    return make_missing_collate(p=p, init_len=init_len, missing_prob=missing_prob,
                                miss_len=miss_len)


def test_dataset_always_stores_clean_windows():
    """RecurrenceDataset no longer corrupts at generation time: windows are
    plain clean tensors even though _corrupt params exist."""
    ds = _make_ds()
    assert all(isinstance(s, torch.Tensor) for s in ds.train_samples + ds.test_samples)
    assert all((s < ds.p).all() for s in ds.train_samples)
    batch = collate_fn(ds.train_samples[:4])
    assert batch[1] == BatchTag.PLAIN


def test_collate_corrupts_predict_mode():
    """predict 口径（唯一保留的模式）：返回 (views, PLAIN_TARGET, cleans)，
    污损位以干净值为目标；初始值永不污损。"""
    p, init_len = 7, 2
    ds = _make_ds(p=p, init_len=init_len)
    collate = _collate(0.5, p=p, init_len=init_len)
    views, tag, cleans = collate(ds.train_samples[:8])
    assert tag == BatchTag.PLAIN_TARGET
    assert views.shape == (8, ds.length) and cleans.shape == (8, ds.length)
    saw = False
    for view, clean in zip(views, cleans):
        assert (view[:init_len] < p).all()
        assert (clean < p).all()
        for pos in range(init_len, ds.length):
            if view[pos].item() == p:
                saw = True
            else:
                assert view[pos].item() == clean[pos].item()
    assert saw


def test_fresh_randomness_per_call():
    """Two collate calls on the same batch corrupt differently (the point of
    the on-the-fly regime: every epoch sees new missing positions)."""
    ds = _make_ds()
    collate = _collate(0.5)
    b1 = collate(ds.train_samples[:8])
    b2 = collate(ds.train_samples[:8])
    assert not torch.equal(b1[0], b2[0])


def test_full_corruption_when_prob_one():
    # prob=1.0 + miss_len=1: deterministic alternation — corrupt one, skip one.
    p, init_len = 7, 3
    ds = _make_ds(p=p, init_len=init_len)
    collate = _collate(1.0, p=p, init_len=init_len)
    views, _, cleans = collate(ds.train_samples[:8])
    for view in views:
        assert (view[:init_len] < p).all()
        for pos in range(init_len, ds.length):
            if pos % 2 == 1:  # 3,5,7
                assert view[pos].item() == p
            else:             # 4,6 forced clean after each run
                assert view[pos].item() < p


def _check_runs(seq, p, init_len, miss_len, start_pos=None):
    """Run-length invariants: every maximal run of missing tokens has length
    in [1, miss_len], is followed by a clean position, and the longest run
    equals miss_len (when any run exists). Returns the max run length."""
    pos = start_pos if start_pos is not None else init_len
    max_run = 0
    while pos < len(seq):
        if seq[pos].item() == p:
            end = pos
            while end < len(seq) and seq[end].item() == p:
                end += 1
            run = end - pos
            assert 1 <= run <= miss_len, f'run {run} at {pos}'
            if end < len(seq):
                assert seq[end].item() != p
            max_run = max(max_run, run)
            pos = end + 1
        else:
            pos += 1
    if max_run:
        assert max_run == miss_len, f'max run {max_run} != miss_len {miss_len}'
    return max_run


def test_miss_len_run_structure():
    """prob=1.0, miss_len=3: run lengths in [1,3], clean separators, longest
    run equals 3 (redo guarantee)."""
    ds = _make_ds(length=12, num_samples=40, p=11)
    collate = _collate(1.0, p=11, miss_len=3)
    views, _, cleans = collate(ds.train_samples[:20])
    for view in views:
        assert (view[:INIT_LEN] < 11).all()
        _check_runs(view, 11, INIT_LEN, 3)


def test_default_loss_mask_first_task_weight():
    from training import _default_loss_mask
    mask = _default_loss_mask(4, 7, num_mask=2, first_task_weight=30.0)
    assert mask.shape == (4, 7)
    assert (mask[:, :2] == 0).all() and mask[0, 2].item() == 30.0
    assert (mask[:, 3:] == 1).all()


def test_predict_missing_returns_clean_targets():
    from core import _unpack_batch
    p = 7
    ds = _make_ds(p=p)
    collate = _collate(0.5, p=p)
    views, tag, cleans = collate(ds.train_samples[:8])
    assert tag == BatchTag.PLAIN_TARGET
    saw = False
    for view, clean in zip(views, cleans):
        assert (clean < p).all()
        for pos in range(ds.length):
            if view[pos].item() == p:
                saw = True
                assert clean[pos].item() != p
            else:
                assert view[pos].item() == clean[pos].item()
    assert saw
    x, loss_mask, kwargs, ab_labels, t_override = _unpack_batch((views, tag, cleans), 'cpu', None)
    assert loss_mask is None and ab_labels is None
    assert torch.equal(t_override, cleans)


# ---------------- mixed_ab / mixed_abc (MixedRecurrenceDataset) ----------------

def _make_mixed(use_ab_tag=False, seed=0, p=7, length=8, num_samples=20):
    # use_ab_tag is accepted for call-site compatibility and ignored: tag mode
    # is deleted; MixedRecurrenceDataset is always basic layout.
    from datasets import MixedRecurrenceDataset
    rules = [LinearRecurrenceRule(coeffs=(1, 1), p=p),
             LinearRecurrenceRule(coeffs=(1, 2), p=p)]
    random.seed(seed)
    return MixedRecurrenceDataset(rules=rules, num_samples=num_samples,
                                  length=length, verbose=False)


def _mixed_collate(missing_prob=0.5, p=7, order=2, n_rules=2):
    return make_mixed_missing_collate(p=p, order=order, n_rules=n_rules,
                                      missing_prob=missing_prob)


def test_mixed_disabled_keeps_pairs():
    from core import mixed_ab_collate_fn
    ds = _make_mixed(use_ab_tag=False)
    assert all(len(item) == 2 for item in ds.train_data)
    batch = mixed_ab_collate_fn(ds.train_data[:4])
    assert batch[1] == BatchTag.MIXED_AB


def test_mixed_missing_alignment():
    from core import _unpack_batch
    p = 7  # basic 模式（唯一模式）：missing token id == p
    ds = _make_mixed(use_ab_tag=False, p=p)
    collate = _mixed_collate(0.5, p=p)
    views, tag, labels, cleans = collate(ds.train_data[:8])
    assert tag == BatchTag.MIXED_AB_TARGET and labels.shape == (8,)
    saw = False
    for (clean, _), view in zip(ds.train_data[:8], views):
        assert (view[:ds.order] < p).all()
        for pos in range(ds.order, ds.length):
            if view[pos].item() == p:
                saw = True
            else:
                assert view[pos].item() == clean[pos].item()
    assert saw
    x, loss_mask, kwargs, ab_labels, t_override = _unpack_batch(
        (views, tag, labels, cleans), 'cpu', None)
    assert loss_mask is None and ab_labels is not None and torch.equal(t_override, cleans)


def test_mixed_fresh_randomness_per_call():
    ds = _make_mixed(use_ab_tag=False)
    collate = _mixed_collate(0.5)
    b1 = collate(ds.train_data[:8])
    b2 = collate(ds.train_data[:8])
    assert not torch.equal(b1[0], b2[0])


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('All tests passed.')
