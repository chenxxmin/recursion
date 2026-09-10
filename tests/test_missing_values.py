"""Unit tests for on-the-fly MISSING_PROB corruption (2026-09-09 regime).

Datasets store CLEAN windows only; make_missing_collate /
make_mixed_missing_collate corrupt each batch with fresh randomness, so every
epoch sees different missing positions. Covers: clean storage, mask alignment
(target index = position - 1), run structure (MISS_LEN, MISS_SECOND),
deterministic patterns at prob=1.0, per-call fresh randomness, tag-mode token
avoidance, PREDICT_MISSING mode, and unpack routing.

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


def _collate(missing_prob=0.5, p=P, init_len=INIT_LEN, num_mask=1,
             first_task_weight=1.0, miss_len=1, miss_second=False,
             predict_missing=False):
    return make_missing_collate(p=p, init_len=init_len, missing_prob=missing_prob,
                                miss_len=miss_len, miss_second=miss_second,
                                num_mask=num_mask, first_task_weight=first_task_weight,
                                predict_missing=predict_missing)


def test_dataset_always_stores_clean_windows():
    """RecurrenceDataset no longer corrupts at generation time: windows are
    plain clean tensors even though _corrupt params exist."""
    ds = _make_ds()
    assert all(isinstance(s, torch.Tensor) for s in ds.train_samples + ds.test_samples)
    assert all((s < ds.p).all() for s in ds.train_samples)
    batch = collate_fn(ds.train_samples[:4])
    assert batch[1] == BatchTag.PLAIN


def test_collate_corrupts_and_masks():
    p, init_len, num_mask = 7, 2, 1
    ds = _make_ds(p=p, init_len=init_len)
    collate = _collate(0.5, p=p, init_len=init_len, num_mask=num_mask)
    views, tag, masks = collate(ds.train_samples[:8])
    assert tag == BatchTag.LOSS_MASK
    assert views.shape == (8, ds.length) and masks.shape == (8, ds.length - 1)
    saw = False
    for view, clean, mask in zip(views, ds.train_samples[:8], masks):
        assert (view[:init_len] < p).all()           # init values never corrupted
        assert (mask[:num_mask] == 0).all()          # mask prefix follows num_mask
        for pos in range(init_len, ds.length):
            if view[pos].item() == p:
                saw = True
                assert clean[pos].item() < p         # clean side holds true value
                assert mask[pos - 1].item() == 0.0
            elif pos - 1 >= num_mask:
                assert view[pos].item() == clean[pos].item()
                assert mask[pos - 1].item() == 1.0
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
    collate = _collate(1.0, p=p, init_len=init_len, num_mask=2)
    views, _, masks = collate(ds.train_samples[:8])
    expected_mask = torch.tensor([0., 0., 0., 1., 0., 1., 0.])
    for view, mask in zip(views, masks):
        assert (view[:init_len] < p).all()
        for pos in range(init_len, ds.length):
            if pos % 2 == 1:  # 3,5,7
                assert view[pos].item() == p and mask[pos - 1].item() == 0.0
            else:             # 4,6 forced clean after each run
                assert view[pos].item() < p and mask[pos - 1].item() == 1.0
        assert torch.equal(mask, expected_mask)


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
    collate = _collate(1.0, p=11, miss_len=3, num_mask=1)
    views, _, masks = collate(ds.train_samples[:20])
    for view, mask in zip(views, masks):
        assert (view[:INIT_LEN] < 11).all()
        _check_runs(view, 11, INIT_LEN, 3)
        for pos in range(INIT_LEN, 12):
            if view[pos].item() == 11:
                assert mask[pos - 1].item() == 0.0
            elif pos - 1 >= 1:
                assert mask[pos - 1].item() == 1.0


def test_miss_second_forced_run_deterministic():
    """miss_second=True: positions 1..miss_len always missing, position
    miss_len+1 always clean, random scan resumes at miss_len+2."""
    ds = _make_ds(length=10, num_samples=40, p=11)
    collate = _collate(1.0, p=11, miss_len=2, num_mask=1, miss_second=True)
    views, _, masks = collate(ds.train_samples[:20])
    for view, mask in zip(views, masks):
        assert view[0].item() != 11
        assert view[1].item() == 11 and view[2].item() == 11  # forced run
        assert view[3].item() != 11                            # guaranteed clean
        assert mask[0].item() == 0.0 and mask[1].item() == 0.0
        _check_runs(view, 11, 1, 2)


def test_first_task_weight_passthrough():
    ds = _make_ds(p=11, length=12, num_samples=40)
    collate = _collate(0.5, p=11, num_mask=2, first_task_weight=30.0)
    _, _, masks = collate(ds.train_samples[:40])
    seen_weighted = False
    for mask in masks:
        if mask[2].item() == 30.0:
            seen_weighted = True
        else:
            assert mask[2].item() == 0.0
    assert seen_weighted


def test_predict_missing_returns_clean_targets():
    from core import _unpack_batch
    p = 7
    ds = _make_ds(p=p)
    collate = _collate(0.5, p=p, predict_missing=True)
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

def _make_mixed(use_ab_tag, seed=0, p=7, length=8, num_samples=20):
    from datasets import MixedRecurrenceDataset
    rules = [LinearRecurrenceRule(coeffs=(1, 1), p=p),
             LinearRecurrenceRule(coeffs=(1, 2), p=p)]
    random.seed(seed)
    return MixedRecurrenceDataset(rules=rules, num_samples=num_samples,
                                  length=length, verbose=False, use_ab_tag=use_ab_tag)


def _mixed_collate(missing_prob=0.5, p=7, order=2, n_rules=2, use_ab_tag=False,
                   num_mask=2, miss_len=1, predict_missing=False):
    return make_mixed_missing_collate(p=p, order=order, n_rules=n_rules,
                                      use_ab_tag=use_ab_tag, missing_prob=missing_prob,
                                      miss_len=miss_len, num_mask=num_mask,
                                      predict_missing=predict_missing)


def test_mixed_disabled_keeps_pairs():
    from core import mixed_ab_collate_fn
    ds = _make_mixed(use_ab_tag=False)
    assert all(len(item) == 2 for item in ds.train_data)
    batch = mixed_ab_collate_fn(ds.train_data[:4])
    assert batch[1] == BatchTag.MIXED_AB


def test_mixed_missing_alignment():
    from core import _unpack_batch
    p, num_mask = 7, 2  # basic mode: missing token id == p
    ds = _make_mixed(use_ab_tag=False, p=p)
    collate = _mixed_collate(0.5, p=p, use_ab_tag=False, num_mask=num_mask)
    views, tag, labels, masks = collate(ds.train_data[:8])
    assert tag == BatchTag.MIXED_AB_MASKED
    assert labels.shape == (8,) and masks.shape == (8, ds.length - 1)
    saw = False
    for (clean, _), view, mask in zip(ds.train_data[:8], views, masks):
        assert (view[:ds.order] < p).all()  # first `order` values stay clean
        for pos in range(ds.order, ds.length):
            corrupted = view[pos].item() == p
            saw = saw or corrupted
            expected = 0.0 if (corrupted or pos - 1 < num_mask) else 1.0
            assert mask[pos - 1].item() == expected, (pos, corrupted, mask)
    assert saw
    x, loss_mask, kwargs, ab_labels, t_override = _unpack_batch(
        (views, tag, labels, masks), 'cpu', None)
    assert loss_mask is not None and ab_labels is not None and kwargs == {} and t_override is None


def test_mixed_tag_missing_alignment():
    p, num_mask, n_rules = 7, 2, 2
    miss = p + n_rules  # tag mode: missing id avoids the flag tokens p..p+N-1
    ds = _make_mixed(use_ab_tag=True, p=p)
    collate = _mixed_collate(0.5, p=p, use_ab_tag=True, num_mask=num_mask)
    views, tag, labels, masks = collate(ds.train_data[:8])
    saw = False
    for (clean, label), view, mask in zip(ds.train_data[:8], views, masks):
        assert view.shape == (ds.length + 1,) and mask.shape == (ds.length,)
        assert view[0].item() == p + label  # leading flag token stays clean
        assert (view[1:ds.order + 1] < p).all()
        for pos in range(1, ds.length + 1):
            corrupted = view[pos].item() == miss
            saw = saw or corrupted
            expected = 0.0 if (corrupted or pos - 1 < num_mask) else 1.0
            assert mask[pos - 1].item() == expected, (pos, corrupted, mask)
    assert saw


def test_mixed_fresh_randomness_per_call():
    ds = _make_mixed(use_ab_tag=False)
    collate = _mixed_collate(0.5, use_ab_tag=False)
    b1 = collate(ds.train_data[:8])
    b2 = collate(ds.train_data[:8])
    assert not torch.equal(b1[0], b2[0])


def test_predict_missing_mixed_tag_mode():
    """Tag mode with PREDICT_MISSING: the flag token is kept on BOTH the
    corrupted view and the clean target sequence."""
    from core import _unpack_batch
    p, n_rules = 7, 2
    miss = p + n_rules
    ds = _make_mixed(use_ab_tag=True, p=p)
    collate = _mixed_collate(0.5, p=p, use_ab_tag=True, predict_missing=True)
    views, tag, labels, cleans = collate(ds.train_data[:8])
    assert tag == BatchTag.MIXED_AB_TARGET and labels.shape == (8,)
    saw = False
    for view, clean, label in zip(views, cleans, labels):
        assert view[0].item() == p + label and clean[0].item() == p + label
        for pos in range(1, ds.length + 1):
            if view[pos].item() == miss:
                saw = True
                assert clean[pos].item() != miss
            else:
                assert view[pos].item() == clean[pos].item()
    assert saw
    x, loss_mask, kwargs, ab_labels, t_override = _unpack_batch(
        (views, tag, labels, cleans), 'cpu', None)
    assert loss_mask is None and ab_labels is not None and torch.equal(t_override, cleans)


def test_post_train_init_state_extraction():
    """The post-training generation test indexes raw sample lists; clean
    windows make initial-state reads trivially correct."""
    from core import _sample_seq
    ds = _make_ds()
    for split_data in (ds.train_samples, ds.test_samples):
        for item in split_data:
            init_state = tuple(_sample_seq(item)[:INIT_LEN].tolist())
            assert len(init_state) == INIT_LEN
            assert all(v < ds.p for v in init_state)


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('All tests passed.')
