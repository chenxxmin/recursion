"""Compatibility smoke test (spec verification 2).

Same seed + legacy AB_PAIRS config: the new MixedRecurrenceDataset pipeline
must produce train/test data sample-for-sample identical to the legacy
core.MixedABDataset (sequences, order, and rule indices).
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import core
from mixed_dataset import MixedRecurrenceDataset
from rules import LinearRecurrenceRule


def _assert_identical(old, new):
    assert len(old.train_data) == len(new.train_data), "train size mismatch"
    assert len(old.test_data) == len(new.test_data), "test size mismatch"
    for (seq_old, idx_old), (seq_new, idx_new) in zip(old.train_data, new.train_data):
        assert idx_old == idx_new, "train rule index mismatch"
        assert seq_old.tolist() == seq_new.tolist(), "train sequence mismatch"
    for (seq_old, idx_old), (seq_new, idx_new) in zip(old.test_data, new.test_data):
        assert idx_old == idx_new, "test rule index mismatch"
        assert seq_old.tolist() == seq_new.tolist(), "test sequence mismatch"


def _run_case(p, ab_pairs, num_samples, length, use_ab_tag, seed):
    random.seed(seed)
    old = core.MixedABDataset(p=p, ab_pairs=ab_pairs, num_samples=num_samples,
                              length=length, verbose=False, use_ab_tag=use_ab_tag)
    random.seed(seed)
    rules = [LinearRecurrenceRule(coeffs=tuple(pair), p=p) for pair in ab_pairs]
    new = MixedRecurrenceDataset(rules=rules, num_samples=num_samples,
                                 length=length, verbose=False, use_ab_tag=use_ab_tag)
    _assert_identical(old, new)


def test_compat_two_rules_no_tag():
    _run_case(p=23, ab_pairs=[(1, 1), (1, 2)], num_samples=[100, 200],
              length=8, use_ab_tag=False, seed=0)


def test_compat_two_rules_with_tag():
    _run_case(p=23, ab_pairs=[(3, 5), (7, 11)], num_samples=[100, 200],
              length=8, use_ab_tag=True, seed=1)


def test_compat_three_rules_scalar_num_samples():
    _run_case(p=7, ab_pairs=[(1, 1), (1, 2), (2, 1)], num_samples=20,
              length=6, use_ab_tag=True, seed=42)


if __name__ == '__main__':
    test_compat_two_rules_no_tag()
    test_compat_two_rules_with_tag()
    test_compat_three_rules_scalar_num_samples()
    print("ALL TESTS PASSED: test_mixed_ab_compat.py")
