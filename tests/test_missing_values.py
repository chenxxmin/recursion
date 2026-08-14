"""Unit tests for MISSING_PROB corruption in src/datasets.py RecurrenceDataset.

Covers: legacy path when disabled, run-based corruption (MISS_LEN, MISS_SECOND),
loss-mask alignment (target index = position - 1), test split corrupted with the
same rule as train, PREDICT_MISSING mode (input = corrupted view, targets =
clean sequence), explicit collate routing (PLAIN / DYNAMIC_MIXED / PLAIN_TARGET
/ MIXED_AB_*), seed determinism, first_task_weight passthrough, and the
mixed-rule equivalents.

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
             num_mask=1, first_task_weight=1.0, miss_len=1, miss_second=False):
    def rec(seq, m):
        return (seq[-1] + seq[-2]) % m if init_len == 2 else (seq[-1] + seq[-2] + seq[-3]) % m
    random.seed(seed)
    ds = RecurrenceDataset(p=p, recurrence_fn=rec, init_len=init_len,
                           num_samples=num_samples, length=length, verbose=False,
                           missing_prob=missing_prob, num_mask=num_mask,
                           first_task_weight=first_task_weight, miss_len=miss_len,
                           miss_second=miss_second)
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
    # prob=1.0 + miss_len=1: deterministic alternation — corrupt one, skip one.
    p, init_len = 7, 3
    ds = _make_ds(1.0, p=p, init_len=init_len, num_mask=2)
    # length=8: corrupted {3,5,7}, clean {0,1,2,4,6}, mask [0,0,0,1,0,1,0]
    expected_mask = torch.tensor([0., 0., 0., 1., 0., 1., 0.])
    for seq, mask in ds.train_samples:
        assert (seq[:init_len] < p).all()
        for pos in range(init_len, ds.length):
            if pos % 2 == 1:  # 3,5,7
                assert seq[pos].item() == p and mask[pos - 1].item() == 0.0
            else:             # 4,6 (forced clean after each run)
                assert seq[pos].item() < p and mask[pos - 1].item() == 1.0
        assert torch.equal(mask, expected_mask)


def _check_runs(seq, p, init_len, miss_len, start_pos=None):
    """Run-length invariants for a corrupted window: every maximal run of
    missing tokens has length in [1, miss_len], is followed by a clean
    position, and the longest run equals miss_len (when any run exists).
    Returns the max run length."""
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
                assert seq[end].item() != p  # clean after a run
            max_run = max(max_run, run)
            pos = end + 1
        else:
            pos += 1
    if max_run:
        assert max_run == miss_len, f'max run {max_run} != miss_len {miss_len}'
    return max_run


def test_miss_len_run_structure():
    """prob=1.0, miss_len=3: random run lengths in [1,3], clean separators,
    and every window's longest run equals 3 (redo guarantee)."""
    ds = _make_ds(1.0, miss_len=3, length=12, num_samples=40, p=11, num_mask=1)
    for seq, mask in ds.train_samples:
        assert (seq[:ds.init_len] < ds.p).all()
        _check_runs(seq, ds.p, ds.init_len, 3)
        for pos in range(ds.init_len, ds.length):
            if seq[pos].item() == ds.p:
                assert mask[pos - 1].item() == 0.0
            elif pos - 1 >= 1:
                assert mask[pos - 1].item() == 1.0


def test_miss_len_runs_separated_by_clean():
    """prob<1: runs have length in [1, miss_len], a clean position after each
    run, and max run == miss_len whenever the window has any corruption."""
    ds = _make_ds(0.5, miss_len=3, length=16, num_samples=60, p=11)
    saw_run = False
    for seq, mask in ds.train_samples:
        saw_run = _check_runs(seq, ds.p, ds.init_len, 3) > 0 or saw_run
    assert saw_run


def test_miss_second_forced_run_deterministic():
    """miss_second=True: positions 1..miss_len always missing, position
    miss_len+1 always clean (no merging), random scan resumes at miss_len+2
    with run lengths in [1, miss_len]."""
    ds = _make_ds(1.0, miss_len=2, length=10, num_samples=40, p=11, num_mask=1,
                  miss_second=True)
    for seq, mask in ds.train_samples + ds.test_samples:
        assert seq[0].item() != ds.p
        assert seq[1].item() == ds.p and seq[2].item() == ds.p  # forced run
        assert seq[3].item() != ds.p                            # guaranteed clean
        assert mask[0].item() == 0.0 and mask[1].item() == 0.0  # forced-run targets masked
        _check_runs(seq, ds.p, 1, 2)  # forced run starts at position 1


def test_miss_second_default_off_unchanged():
    """miss_second defaults to False: position 1 is never corrupted."""
    ds = _make_ds(1.0, miss_len=1, p=7)  # prob=1 without the flag
    for seq, mask in ds.train_samples:
        assert seq[0].item() < ds.p and seq[1].item() < ds.p


def test_test_split_corrupted_with_mask():
    """test windows follow the same run-corruption rule as train: prob=1.0 +
    miss_len=1 gives the deterministic corrupt-one/skip-one pattern."""
    p, init_len = 7, 2
    ds = _make_ds(1.0, p=p, init_len=init_len, num_mask=1)
    assert len(ds.test_samples) > 0
    # length=8: corrupted {2,4,6}, clean {0,1,3,5,7}, mask [0,0,1,0,1,0,1]
    expected_mask = torch.tensor([0., 0., 1., 0., 1., 0., 1.])
    for seq, mask in ds.test_samples:
        assert (seq[:init_len] < p).all()  # init values never corrupted
        for pos in range(init_len, ds.length):
            if pos % 2 == 0:  # 2,4,6
                assert seq[pos].item() == p and mask[pos - 1].item() == 0.0
            else:             # 3,5,7 forced clean
                assert seq[pos].item() < p and mask[pos - 1].item() == 1.0
        assert torch.equal(mask, expected_mask)


def test_collate_routes_tuples_to_dynamic_mixed():
    from core import collate_fn_masked
    ds = _make_ds(0.3)
    seqs, tag, masks = collate_fn_masked(ds.train_samples[:4])
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
                num_mask=2, miss_len=1):
    from datasets import MixedRecurrenceDataset
    rules = [LinearRecurrenceRule(coeffs=(1, 1), p=p),
             LinearRecurrenceRule(coeffs=(1, 2), p=p)]
    random.seed(seed)
    return MixedRecurrenceDataset(rules=rules, num_samples=num_samples,
                                  length=length, verbose=False, use_ab_tag=use_ab_tag,
                                  missing_prob=missing_prob, num_mask=num_mask,
                                  miss_len=miss_len)


def test_mixed_disabled_keeps_pairs():
    from core import mixed_ab_collate_fn
    ds = _make_mixed(0.0, use_ab_tag=False)
    assert all(len(item) == 2 for item in ds.train_data)
    batch = mixed_ab_collate_fn(ds.train_data[:4])
    assert batch[1] == BatchTag.MIXED_AB


def test_mixed_basic_missing_alignment():
    from core import _unpack_batch, mixed_ab_collate_fn_masked
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
    seqs, tag, labels, masks = mixed_ab_collate_fn_masked(ds.train_data[:4])
    assert tag == BatchTag.MIXED_AB_MASKED
    assert labels.shape == (4,) and masks.shape == (4, ds.length - 1)
    x, loss_mask, kwargs, ab_labels, t_override = _unpack_batch((seqs, tag, labels, masks), 'cpu', None)
    assert loss_mask is not None and ab_labels is not None and kwargs == {} and t_override is None


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
    """Mixed test windows use the same run-corruption rule as train, in both
    tag modes. prob=1.0 + miss_len=1 -> corrupt one, skip one."""
    p, n_rules, order = 7, 2, 2
    for use_tag, miss in ((False, p), (True, p + n_rules)):
        ds = _make_mixed(1.0, use_ab_tag=use_tag, p=p)
        assert len(ds.test_data) > 0
        start = order + (1 if use_tag else 0)  # first corruptible position
        seq_len = ds.length + (1 if use_tag else 0)
        for seq, mask, label in ds.test_data:
            for pos in range(start, seq_len):
                if (pos - start) % 2 == 0:  # corrupted run positions
                    assert seq[pos].item() == miss and mask[pos - 1].item() == 0.0
                else:                       # forced clean after each run
                    assert seq[pos].item() != miss and mask[pos - 1].item() == 1.0


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


# ---------------- PREDICT_MISSING (targets = clean sequence) ----------------

def _make_ds_predict(prob, seed=0, p=7, init_len=2, length=8, num_samples=20,
                     miss_len=1):
    def rec(seq, m):
        return (seq[-1] + seq[-2]) % m
    random.seed(seed)
    ds = RecurrenceDataset(p=p, recurrence_fn=rec, init_len=init_len,
                           num_samples=num_samples, length=length, verbose=False,
                           missing_prob=prob, num_mask=1, miss_len=miss_len,
                           predict_missing=True)
    ds.run()
    return ds


def test_predict_missing_items_carry_clean_targets():
    from core import _unpack_batch, collate_fn_predict
    p = 7
    ds = _make_ds_predict(0.5, p=p)
    saw = False
    for view, clean in ds.train_samples:
        assert (clean < p).all()                       # clean side never has missing tokens
        for pos in range(ds.length):
            if view[pos].item() == p:                  # corrupted in view...
                saw = True
                assert clean[pos].item() != p          # ...but holds the true value in clean
            else:
                assert view[pos].item() == clean[pos].item()  # uncorrupted positions agree
    assert saw
    views, tag, cleans = collate_fn_predict(ds.train_samples[:4])
    assert tag == BatchTag.PLAIN_TARGET
    x, loss_mask, kwargs, ab_labels, t_override = _unpack_batch((views, tag, cleans), 'cpu', None)
    assert loss_mask is None and ab_labels is None
    assert torch.equal(t_override, cleans)             # targets_override = clean sequences


def test_predict_missing_mixed_tag_mode():
    """Tag mode with PREDICT_MISSING: the flag token is prepended to BOTH the
    corrupted view and the clean target sequence."""
    from core import _unpack_batch, mixed_ab_collate_fn_predict
    from datasets import MixedRecurrenceDataset
    p, n_rules = 7, 2
    miss = p + n_rules
    rules = [LinearRecurrenceRule(coeffs=(1, 1), p=p),
             LinearRecurrenceRule(coeffs=(1, 2), p=p)]
    random.seed(0)
    ds = MixedRecurrenceDataset(rules=rules, num_samples=20, length=8, verbose=False,
                                use_ab_tag=True, missing_prob=0.5, num_mask=2,
                                predict_missing=True)
    saw = False
    for view, clean, label in ds.train_data:
        assert view[0].item() == p + label and clean[0].item() == p + label  # flag on both
        for pos in range(1, ds.length + 1):
            if view[pos].item() == miss:
                saw = True
                assert clean[pos].item() != miss
            else:
                assert view[pos].item() == clean[pos].item()
    assert saw
    views, tag, labels, cleans = mixed_ab_collate_fn_predict(ds.train_data[:4])
    assert tag == BatchTag.MIXED_AB_TARGET and labels.shape == (4,)
    x, loss_mask, kwargs, ab_labels, t_override = _unpack_batch(
        (views, tag, labels, cleans), 'cpu', None)
    assert loss_mask is None and ab_labels is not None and torch.equal(t_override, cleans)


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('All tests passed.')