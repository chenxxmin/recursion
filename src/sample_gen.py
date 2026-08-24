"""Standalone sample generator for the addition recurrence task.

Input: modulus p, sample length, recurrence coefficients (a, b), exposure
ratio, seed. Output: train/test LongTensors.

The generation logic mirrors src/datasets.py RecurrenceDataset (clean path:
no missing-value corruption, no mixed rules) step for step, including the
random call sequence, so given the same (p, length, a, b, ratio, seed) the
output is element-wise identical to the in-repo dataset. This file has no
in-repo imports and can be copied out on its own (deps: torch, stdlib random).

Note on "same trajectory stays together": the train/test split is enforced
PER STATE (shuffled state order, first num_samples states -> train), not per
trajectory. Trajectories fully inside either side stay intact; the single
trajectory crossing the num_samples boundary is split between train and test.
"""
import random

import torch


def generate_datasets(p, length, a, b, exposed_ratio=0.7, seed=0):
    """Generate (train, test) for X(k) = (a*X(k-1) + b*X(k-2)) mod p.

    Returns two LongTensors of shape (N, length). init_len is fixed to 2.
    """
    init_len = 2
    num_samples = max(1, int(p ** init_len * exposed_ratio))
    recurrence_fn = lambda seq, p: (a * seq[-1] + b * seq[-2]) % p

    random.seed(seed)

    def index_to_values(index):
        vals = []
        for _ in range(init_len):
            vals.insert(0, index % p)
            index //= p
        return vals

    state_space = p ** init_len
    seen_indices = set()
    train_samples, test_samples = [], []

    all_indices = list(range(state_space))
    random.shuffle(all_indices)

    for start_idx in all_indices:
        if start_idx in seen_indices:
            continue
        n_before = len(seen_indices)
        seen_indices.add(start_idx)

        # Traverse the whole trajectory; stop when it closes back to the start
        # state (pure cycle) or runs into an already-seen state (transient
        # trajectory merging into a processed cycle).
        seq = index_to_values(start_idx)
        while True:
            next_val = recurrence_fn(seq[-init_len:], p)
            current_state_idx = 0
            for val in seq[-init_len:]:
                current_state_idx = current_state_idx * p + val
            next_idx = (current_state_idx % (p ** (init_len - 1))) * p + next_val
            if next_idx == start_idx or next_idx in seen_indices:
                break
            seen_indices.add(next_idx)
            seq.append(next_val)

        # One window per new state; extend by continuing the recurrence.
        num_inits = len(seq) - (init_len - 1)
        while len(seq) < num_inits + length - 1:
            seq.append(recurrence_fn(seq[-init_len:], p))

        for i in range(num_inits):
            is_train = n_before + i < num_samples
            target = train_samples if is_train else test_samples
            target.append(seq[i:i + length])

    random.shuffle(train_samples)

    train = torch.tensor(train_samples, dtype=torch.long)
    test = torch.tensor(test_samples, dtype=torch.long)
    return train, test
