"""Equivalence test: src/sample_gen.generate_datasets vs RecurrenceDataset.

Same (p, length, a, b, ratio, seed) must yield element-wise identical
train/test samples (train compared after its shuffle, test in order).
Run from tests/: C:/Users/Chen/anaconda3/envs/torch/python.exe test_sample_gen.py
"""
import random
import sys

sys.path.insert(0, '../src')
import torch

from datasets import RecurrenceDataset
from sample_gen import generate_datasets


def check(p, length, a, b, ratio, seed):
    train, test = generate_datasets(p=p, length=length, a=a, b=b,
                                    exposed_ratio=ratio, seed=seed)

    random.seed(seed)
    recurrence_fn = lambda seq, p: (a * seq[-1] + b * seq[-2]) % p
    ds = RecurrenceDataset(p=p, recurrence_fn=recurrence_fn,
                           recurrence_name=f"X(k)=({a}*X(k-1)+{b}*X(k-2)) mod {p}",
                           init_len=2, num_samples=max(1, int(p * p * ratio)),
                           length=length, verbose=False)
    ds.run()
    ref_train = torch.stack(ds.train_samples) if ds.train_samples else torch.empty(0, length, dtype=torch.long)
    ref_test = torch.stack(ds.test_samples) if ds.test_samples else torch.empty(0, length, dtype=torch.long)

    assert train.dtype == torch.long and test.dtype == torch.long
    assert train.shape == ref_train.shape, \
        f"train shape {train.shape} != {ref_train.shape} (p={p}, ratio={ratio}, seed={seed})"
    assert test.shape == ref_test.shape, \
        f"test shape {test.shape} != {ref_test.shape} (p={p}, ratio={ratio}, seed={seed})"
    assert torch.equal(train, ref_train), \
        f"train mismatch (p={p}, length={length}, a={a}, b={b}, ratio={ratio}, seed={seed})"
    assert torch.equal(test, ref_test), \
        f"test mismatch (p={p}, length={length}, a={a}, b={b}, ratio={ratio}, seed={seed})"
    print(f"OK p={p} length={length} a={a} b={b} ratio={ratio} seed={seed}: "
          f"train {train.shape}, test {test.shape}")


def test_equivalence():
    for p, length, a, b, ratio, seed in [
        (53, 10, 1, 1, 0.7, 0),
        (53, 16, 1, 2, 0.7, 12345),
        (127, 64, 1, 1, 0.7, 0),
        (127, 64, 2, 3, 0.5, 999),
        (7, 6, 1, 1, 0.9, 42),
        (7, 6, 1, 0, 0.7, 0),   # transient states (non-bijective rule)
    ]:
        check(p, length, a, b, ratio, seed)


def test_defaults():
    train, test = generate_datasets(p=53, length=10, a=1, b=1)
    n = train.shape[0] + test.shape[0]
    assert n == 53 * 53, f"every state yields one window, got {n}"
    assert train.shape[0] == max(1, int(53 * 53 * 0.7))
    print(f"OK defaults: train {train.shape}, test {test.shape}")


if __name__ == '__main__':
    test_equivalence()
    test_defaults()
    print("All sample_gen tests passed.")
