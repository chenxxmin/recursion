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

    With missing_prob > 0 the per-rule RecurrenceDataset corrupts windows
    (see core.RecurrenceDataset): scanning from position >= order, a hit with
    probability missing_prob corrupts a run of miss_len consecutive tokens,
    in train and test splits alike (test-side accuracy then measures bridging
    over gaps). Samples then become (seq, loss_mask, rule_idx) triples routed
    through BatchTag.MIXED_AB_MASKED. The missing token id is p, or
    p + len(rules) when use_ab_tag=True (plain p would collide with rule 0's
    flag token).
    """

    def __init__(self, rules, num_samples=1000, length=10, verbose=True, use_ab_tag=False,
                 missing_prob=0.0, num_mask=0, first_task_weight=1.0, miss_len=1):
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
        self.missing_prob = missing_prob
        self.split = 'train'

        if isinstance(num_samples, (list, tuple)):
            num_samples_list = list(num_samples)
        else:
            num_samples_list = [num_samples] * len(self.rules)
        assert len(num_samples_list) == len(self.rules), \
            f"num_samples list length ({len(num_samples_list)}) must equal number of rules ({len(self.rules)})"

        # Corruption happens inside the per-rule dataset, before any flag token
        # is prepended, so "the first `order` values stay clean" is expressed in
        # clean coordinates. The tag shifts every target by one, hence the inner
        # mask prefix is one shorter.
        corrupt = missing_prob > 0
        inner_num_mask = max(0, num_mask - 1) if use_ab_tag else num_mask
        missing_token = self.p + len(self.rules) if use_ab_tag else self.p

        all_train, all_test = [], []
        all_masks_train, all_masks_test = [], []
        all_labels_train, all_labels_test = [], []
        per_rule_stats = []
        for idx, rule in enumerate(self.rules):
            ds = RecurrenceDataset(
                p=self.p, recurrence_fn=rule.next_fn(),
                recurrence_name=rule.name, init_len=rule.order,
                num_samples=num_samples_list[idx], length=length,
                verbose=False,
                missing_prob=missing_prob,
                miss_len=miss_len,
                num_mask=inner_num_mask,
                first_task_weight=first_task_weight,
                missing_token=missing_token,
            )
            ds.run()
            if corrupt:
                train_seqs = [s for s, _ in ds.train_samples]
                train_masks = [m for _, m in ds.train_samples]
                test_seqs = [s for s, _ in ds.test_samples]
                test_masks = [m for _, m in ds.test_samples]
            else:
                train_seqs, test_seqs = ds.train_samples, ds.test_samples
                train_masks = test_masks = None
            all_train.extend(train_seqs)
            all_test.extend(test_seqs)
            if corrupt:
                all_masks_train.extend(train_masks)
                all_masks_test.extend(test_masks)
            all_labels_train.extend([idx] * len(train_seqs))
            all_labels_test.extend([idx] * len(test_seqs))
            per_rule_stats.append((rule, len(train_seqs), len(test_seqs)))

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
            if corrupt:
                # The prepended flag shifts every target by one; grow the mask
                # with a leading 0 (predicting x0 from the flag alone is
                # unsupervisable).
                zero = torch.zeros(1, dtype=torch.float)
                all_masks_train = [torch.cat([zero, m], dim=0) for m in all_masks_train]
                all_masks_test = [torch.cat([zero, m], dim=0) for m in all_masks_test]

        # One global shuffle of train (seq, rule_idx) pairs (same seed semantics
        # as the legacy MixedABDataset).
        train_triples = list(zip(all_train, all_labels_train,
                                 all_masks_train if corrupt else [None] * len(all_train)))
        random.shuffle(train_triples)
        self.train_samples = [s for s, _, _ in train_triples]
        self.rule_labels_train = [l for _, l, _ in train_triples]
        self.train_masks = [m for _, _, m in train_triples] if corrupt else None
        self.test_samples = all_test
        self.rule_labels_test = all_labels_test
        self.test_masks = all_masks_test if corrupt else None

        # Flat lists that can be fed directly to DataLoader / BucketBatchSampler
        if corrupt:
            self.train_data = list(zip(self.train_samples, self.train_masks, self.rule_labels_train))
            self.test_data = list(zip(self.test_samples, self.test_masks, self.rule_labels_test))
        else:
            self.train_data = list(zip(self.train_samples, self.rule_labels_train))
            self.test_data = list(zip(self.test_samples, self.rule_labels_test))

        if verbose:
            total_states = self.p ** self.order
            print(f"[Dataset] Mixed mode: generated {len(self.train_samples)} + {len(self.test_samples)} samples, length {length}")
            print(f"Rules: {[r.name for r in self.rules]}")
            for rule, n_train, n_test in per_rule_stats:
                print(f"  - {rule.name} {rule.coeffs}: train {n_train} | test {n_test} | exposed ~{n_train/total_states*100:.1f}%")
            if corrupt:
                print(f"  - Missing-value corruption: prob={missing_prob}, miss_len={miss_len}, "
                      f"positions >= {self.order}, token id {missing_token}, "
                      f"train+test splits (loss masked at corrupted positions)")
            print("-" * 50)

    def __len__(self):
        seqs = self.train_samples if self.split == 'train' else self.test_samples
        return len(seqs)

    def __getitem__(self, idx):
        if self.split == 'train':
            if self.train_masks is not None:
                return self.train_samples[idx], self.train_masks[idx], self.rule_labels_train[idx]
            return self.train_samples[idx], self.rule_labels_train[idx]
        if self.test_masks is not None:
            return self.test_samples[idx], self.test_masks[idx], self.rule_labels_test[idx]
        return self.test_samples[idx], self.rule_labels_test[idx]
