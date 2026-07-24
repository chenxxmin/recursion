"""Mixed-rule recurrence dataset.

Generates data for several LinearRecurrenceRule rules (same modulus p, same
order), each with its own RecurrenceDataset cycle traversal, then concatenates
the per-rule samples and tags every sample with its rule index. Mirrors the
legacy core.MixedABDataset pipeline step for step (same global-random call
sequence), so given the same rules and the same seed the produced samples are
identical to the legacy implementation.
"""
import random

import torch
from torch.utils.data import Dataset

from core import RecurrenceDataset


class MixedRecurrenceDataset(Dataset):
    """Dataset mixed from several linear recurrence rules over Z/pZ.

    Samples are (seq, rule_idx) pairs; with use_ab_tag=True a leading flag
    token p + rule_idx is prepended to every sequence. Feed train_data /
    test_data to core.mixed_ab_collate_fn / BatchTag.MIXED_AB.
    """

    def __init__(self, rules, num_samples=1000, length=10, verbose=True, use_ab_tag=False):
        assert len(rules) >= 1, "MixedRecurrenceDataset needs at least one rule"
        self.p = rules[0].p
        self.order = rules[0].order
        for rule in rules:
            assert rule.p == self.p, "all rules must share the same modulus p"
            assert rule.order == self.order, "all rules must share the same order"
        coeffs_list = [rule.coeffs for rule in rules]
        assert len(set(coeffs_list)) == len(coeffs_list), \
            "duplicate rule coeffs are not allowed"
        self.rules = list(rules)
        self.use_ab_tag = use_ab_tag
        self.split = 'train'

        if isinstance(num_samples, (list, tuple)):
            num_samples_list = list(num_samples)
        else:
            num_samples_list = [num_samples] * len(self.rules)
        assert len(num_samples_list) == len(self.rules), \
            f"num_samples list length ({len(num_samples_list)}) must equal number of rules ({len(self.rules)})"

        all_train, all_test = [], []
        all_labels_train, all_labels_test = [], []
        per_rule_stats = []
        for idx, rule in enumerate(self.rules):
            ds = RecurrenceDataset(
                p=self.p, recurrence_fn=rule.next_fn(),
                recurrence_name=rule.name, init_len=rule.order,
                num_samples=num_samples_list[idx], length=length,
                verbose=False,
            )
            ds.run()
            all_train.extend(ds.train_samples)
            all_test.extend(ds.test_samples)
            all_labels_train.extend([idx] * len(ds.train_samples))
            all_labels_test.extend([idx] * len(ds.test_samples))
            per_rule_stats.append((rule, len(ds.train_samples), len(ds.test_samples)))

        # Prepend rule token if use_ab_tag (DataLoader indexes the flat lists, not __getitem__)
        if self.use_ab_tag:
            all_train = [
                torch.cat([torch.tensor([self.p + l], dtype=torch.long), seq], dim=0)
                for seq, l in zip(all_train, all_labels_train)
            ]
            all_test = [
                torch.cat([torch.tensor([self.p + l], dtype=torch.long), seq], dim=0)
                for seq, l in zip(all_test, all_labels_test)
            ]

        # One global shuffle of train (seq, rule_idx) pairs (same seed semantics
        # as the legacy MixedABDataset).
        train_pairs = list(zip(all_train, all_labels_train))
        random.shuffle(train_pairs)
        self.train_samples = [s for s, _ in train_pairs]
        self.rule_labels_train = [l for _, l in train_pairs]
        self.test_samples = all_test
        self.rule_labels_test = all_labels_test

        # Flat lists that can be fed directly to DataLoader / BucketBatchSampler
        self.train_data = list(zip(self.train_samples, self.rule_labels_train))
        self.test_data = list(zip(self.test_samples, self.rule_labels_test))

        if verbose:
            total_states = self.p ** self.order
            print(f"[Dataset] Mixed mode: generated {len(self.train_samples)} + {len(self.test_samples)} samples, length {length}")
            print(f"Rules: {[r.name for r in self.rules]}")
            for rule, n_train, n_test in per_rule_stats:
                print(f"  - {rule.name} {rule.coeffs}: train {n_train} | test {n_test} | exposed ~{n_train/total_states*100:.1f}%")
            print("-" * 50)

    def __len__(self):
        seqs = self.train_samples if self.split == 'train' else self.test_samples
        return len(seqs)

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.train_samples[idx], self.rule_labels_train[idx]
        return self.test_samples[idx], self.rule_labels_test[idx]
