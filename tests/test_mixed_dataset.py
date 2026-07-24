"""Unit tests for src/mixed_dataset.py (pure asserts; also pytest-compatible)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from mixed_dataset import MixedRecurrenceDataset
from rules import LinearRecurrenceRule


def test_train_quota_and_labels():
    p = 5
    rules = [LinearRecurrenceRule(coeffs=(1, 1), p=p),
             LinearRecurrenceRule(coeffs=(1, 2), p=p)]
    random.seed(0)
    ds = MixedRecurrenceDataset(rules=rules, num_samples=[10, 12], length=8, verbose=False)
    # per-state quota is exact: train windows per rule == num_samples entry
    assert len(ds.train_data) == 22
    assert len(ds.test_data) == 2 * p * p - 22
    assert sorted(l for _, l in ds.train_data) == [0] * 10 + [1] * 12
    assert sorted(l for _, l in ds.test_data) == [0] * (p * p - 10) + [1] * (p * p - 12)


def test_sequences_follow_their_rule():
    p = 7
    rules = [LinearRecurrenceRule(coeffs=(1, 1), p=p),
             LinearRecurrenceRule(coeffs=(2, 1), p=p)]
    random.seed(1)
    ds = MixedRecurrenceDataset(rules=rules, num_samples=[20, 20], length=8, verbose=False)
    for seq, idx in ds.train_data + ds.test_data:
        fn = rules[idx].next_fn()
        vals = seq.tolist()
        for k in range(rules[idx].order, len(vals)):
            assert vals[k] == fn(vals[k - rules[idx].order:k], p)


def test_use_ab_tag_prepends_flag_token():
    p = 7
    rules = [LinearRecurrenceRule(coeffs=(1, 1), p=p),
             LinearRecurrenceRule(coeffs=(1, 2), p=p)]
    random.seed(2)
    ds = MixedRecurrenceDataset(rules=rules, num_samples=[20, 20], length=8,
                                verbose=False, use_ab_tag=True)
    for seq, idx in ds.train_data + ds.test_data:
        vals = seq.tolist()
        assert vals[0] == p + idx
        assert len(vals) == 9  # length 8 + 1 flag token
        fn = rules[idx].next_fn()
        body = vals[1:]
        for k in range(rules[idx].order, len(body)):
            assert body[k] == fn(body[k - rules[idx].order:k], p)


def test_getitem_matches_flat_lists():
    p = 5
    rules = [LinearRecurrenceRule(coeffs=(1, 1), p=p)]
    random.seed(3)
    ds = MixedRecurrenceDataset(rules=rules, num_samples=[8], length=6, verbose=False)
    assert len(ds) == len(ds.train_data)
    for i in range(len(ds)):
        seq, idx = ds[i]
        assert seq is ds.train_data[i][0] and idx == ds.train_data[i][1]
    ds.split = 'test'
    assert len(ds) == len(ds.test_data)
    for i in range(len(ds)):
        seq, idx = ds[i]
        assert seq is ds.test_data[i][0] and idx == ds.test_data[i][1]


def test_mismatched_rules_rejected():
    try:
        MixedRecurrenceDataset(
            rules=[LinearRecurrenceRule(coeffs=(1, 1), p=5),
                   LinearRecurrenceRule(coeffs=(1, 2), p=7)],
            num_samples=4, length=6, verbose=False)
        assert False, "different p must be rejected"
    except AssertionError:
        pass
    try:
        MixedRecurrenceDataset(
            rules=[LinearRecurrenceRule(coeffs=(1, 1), p=5),
                   LinearRecurrenceRule(coeffs=(1, 1, 1), p=5)],
            num_samples=4, length=6, verbose=False)
        assert False, "different order must be rejected"
    except AssertionError:
        pass
    try:
        MixedRecurrenceDataset(
            rules=[LinearRecurrenceRule(coeffs=(1, 1), p=5),
                   LinearRecurrenceRule(coeffs=(1, 1), p=5)],
            num_samples=4, length=6, verbose=False)
        assert False, "duplicate coeffs must be rejected"
    except AssertionError:
        pass


if __name__ == '__main__':
    test_train_quota_and_labels()
    test_sequences_follow_their_rule()
    test_use_ab_tag_prepends_flag_token()
    test_getitem_matches_flat_lists()
    test_mismatched_rules_rejected()
    print("ALL TESTS PASSED: test_mixed_dataset.py")
