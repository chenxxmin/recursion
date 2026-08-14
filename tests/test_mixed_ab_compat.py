"""Compatibility smoke test (spec verification 2).

Same seed + legacy AB_PAIRS config: the new MixedRecurrenceDataset pipeline
must produce train/test data sample-for-sample identical to the legacy
core.MixedABDataset (sequences, order, and rule indices).
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch
from torch.utils.data import Dataset

import core
from datasets import MixedRecurrenceDataset
from rules import LinearRecurrenceRule


# Legacy implementation, kept here ONLY as the golden master for the
# compatibility tests below. Production code uses
# datasets.MixedRecurrenceDataset. Do not "improve" or restyle this
# class -- the tests pin sample-for-sample equivalence against it, including
# the exact global-RNG call sequence.
class MixedABDataset(Dataset):
    def __init__(self, p=127, ab_pairs=None, fixed_ab_idx=None, num_samples=1000, length=10,
                 verbose=True, use_ab_tag=False):
        self.p = p
        self.use_ab_tag = use_ab_tag
        self.ab_pairs = ab_pairs if ab_pairs is not None else [(3, 5), (7, 11)]
        self.train_samples = []
        self.test_samples = []
        self.ab_labels_train = []
        self.ab_labels_test = []
        self.split = 'train'

        if fixed_ab_idx is not None:
            a, b = self.ab_pairs[fixed_ab_idx]
            ns = num_samples[fixed_ab_idx] if isinstance(num_samples, (list, tuple)) else num_samples
            self._build_with_ab(a, b, ns, length, verbose)
        else:
            self._build_mixed(num_samples, length, verbose)

    def _build_with_ab(self, a, b, num_samples, length, verbose):
        def recurrence_fn(seq, p):
            return (a * seq[-1] + b * seq[-2]) % p

        ds = core.RecurrenceDataset(
            p=self.p, recurrence_fn=recurrence_fn,
            recurrence_name=f"X(k)={a}*X(k-1)+{b}*X(k-2)",
            init_len=2, num_samples=num_samples, length=length,
            verbose=verbose
        )
        ds.run()
        self.train_samples = ds.train_samples
        self.test_samples = ds.test_samples
        self.ab_labels_train = [(a, b)] * len(ds.train_samples)
        self.ab_labels_test = [(a, b)] * len(ds.test_samples)

    def _build_mixed(self, num_samples, length, verbose):
        if isinstance(num_samples, (list, tuple)):
            num_samples_list = list(num_samples)
        else:
            num_samples_list = [num_samples] * len(self.ab_pairs)
        all_train = []
        all_test = []
        all_labels_train = []
        all_labels_test = []
        per_rule_stats = []
        for idx, (a, b) in enumerate(self.ab_pairs):
            self._build_with_ab(a, b, num_samples_list[idx], length, False)
            all_train.extend(self.train_samples)
            all_test.extend(self.test_samples)
            all_labels_train.extend(self.ab_labels_train)
            all_labels_test.extend(self.ab_labels_test)
            per_rule_stats.append((a, b, len(self.train_samples), len(self.test_samples)))

        # Prepend rule token if use_ab_tag (DataLoader indexes the flat list, not __getitem__)
        if self.use_ab_tag:
            all_train = [
                torch.cat([torch.tensor([self.p + self.ab_pairs.index(l)], dtype=torch.long), seq], dim=0)
                for seq, l in zip(all_train, all_labels_train)
            ]
            all_test = [
                torch.cat([torch.tensor([self.p + self.ab_pairs.index(l)], dtype=torch.long), seq], dim=0)
                for seq, l in zip(all_test, all_labels_test)
            ]

        train_pairs = list(zip(all_train, all_labels_train))
        random.shuffle(train_pairs)
        self.train_samples = [s for s, _ in train_pairs]
        self.ab_labels_train = [l for _, l in train_pairs]
        self.test_samples = all_test
        self.ab_labels_test = all_labels_test

        # Flat lists that can be fed directly to DataLoader / BucketBatchSampler
        self.train_data = list(zip(self.train_samples, [self.ab_pairs.index(l) for l in self.ab_labels_train]))
        self.test_data  = list(zip(self.test_samples, [self.ab_pairs.index(l) for l in self.ab_labels_test]))

        if verbose:
            total_states = self.p * self.p
            print(f"[Dataset] Mixed mode: generated {len(self.train_samples)} + {len(self.test_samples)} samples, length {length}")
            print(f"AB parameter pairs: {self.ab_pairs}")
            for a, b, n_train, n_test in per_rule_stats:
                print(f"  - AB=({a},{b}): train {n_train} | test {n_test} | exposed ~{n_train/total_states*100:.1f}%")
            print("-" * 50)

    def __len__(self):
        seqs = self.train_samples if self.split == 'train' else self.test_samples
        return len(seqs)

    def __getitem__(self, idx):
        seqs = self.train_samples if self.split == 'train' else self.test_samples
        labels = self.ab_labels_train if self.split == 'train' else self.ab_labels_test
        ab_pair = labels[idx]
        ab_idx = self.ab_pairs.index(ab_pair)
        seq = seqs[idx]
        return seq, ab_idx


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
    old = MixedABDataset(p=p, ab_pairs=ab_pairs, num_samples=num_samples,
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
