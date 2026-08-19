"""Tests for DynamicMixedDataset missing-value corruption (predict mode)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from datasets import BatchTag, DynamicMixedDataset, dynamic_missing_collate_fn
from training import _unpack_batch

P, PAIRS, L = 7, [(1, 1), (1, 2)], 12


def _make(missing_prob=0.5, miss_len=3, seed=42, n=200, length=L):
    return DynamicMixedDataset(p=P, ab_pairs=PAIRS, num_samples=n,
                               length=length, seed=seed,
                               missing_prob=missing_prob, miss_len=miss_len)


def test_missing_only_on_value_positions():
    ds = _make()
    n_masked = 0
    for view, clean, mask in ds.samples:
        v, c = view.tolist(), clean.tolist()
        for i, tok in enumerate(v):
            if i >= 2 and i % 2 == 0:
                assert tok >= P + 1, f"flag position {i} corrupted: {tok}"
            if tok == P:
                assert i >= 3 and i % 2 == 1, f"M at non-value position {i}"
                n_masked += 1
        # view differs from clean only at M positions; clean layout intact
        for a, b in zip(v, c):
            assert a == b or a == P
        assert all(tok < P for i, tok in enumerate(c) if i % 2 == 1 or i < 2)
        assert all(tok >= P + 1 for i, tok in enumerate(c) if i >= 2 and i % 2 == 0)
    assert n_masked > 0


def test_run_lengths_and_spacing():
    ds = _make(missing_prob=0.4, miss_len=3, n=300)
    seen_masked = 0
    for view, clean, mask in ds.samples:
        v = view.tolist()
        masked_ks = [k for k in range(3, L + 1) if v[2 * k - 3] == P]
        if not masked_ks:
            continue  # all-clean trials are accepted by design
        seen_masked += 1
        runs, start = [], masked_ks[0]
        for a, b in zip(masked_ks, masked_ks[1:]):
            if b != a + 1:
                runs.append((start, a))
                start = b
        runs.append((start, masked_ks[-1]))
        lengths = [b - a + 1 for a, b in runs]
        assert max(lengths) == 3, f"longest run {lengths} != miss_len 3"
        # runs are separated by at least one clean value
        for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
            assert a2 - b1 >= 2, f"adjacent runs merged: {runs}"
    assert seen_masked > 0


def test_seed_reproducibility():
    a = _make(seed=7, n=50)
    b = _make(seed=7, n=50)
    for (v1, c1, m1), (v2, c2, m2) in zip(a.samples, b.samples):
        assert torch.equal(v1, v2) and torch.equal(c1, c2) and torch.equal(m1, m2)


def test_no_missing_format_unchanged():
    ds = DynamicMixedDataset(p=P, ab_pairs=PAIRS, num_samples=20, length=L, seed=3)
    for item in ds.samples:
        assert isinstance(item, tuple) and len(item) == 2
        seq, mask = item
        assert all(tok != P for tok in seq.tolist())
        assert mask.sum().item() == L - 2  # x3..x_L targets only
        for k in range(2, L):
            assert mask[2 * (k - 1)] == 1.0


def test_collate_action_miss():
    ds = _make(n=4)
    x, tag, clean, mask = dynamic_missing_collate_fn(ds.samples[:4])
    assert tag == BatchTag.ACTION_MISS
    assert x.shape == clean.shape == (4, 2 * L - 2)
    assert mask.shape == (4, 2 * L - 3)


def test_unpack_action_miss():
    ds = _make(n=4)
    batch = dynamic_missing_collate_fn(ds.samples[:4])
    x, loss_mask, kwargs, ab_labels, targets_override = _unpack_batch(list(batch), 'cpu', None)
    assert torch.equal(x, batch[0])
    assert torch.equal(loss_mask, batch[3])
    assert torch.equal(targets_override, batch[2])
    assert kwargs == {} and ab_labels is None
    try:
        _unpack_batch([batch[0], BatchTag.ACTION_MISS, batch[2]], 'cpu', None)
        assert False, "expected ValueError for malformed payload"
    except ValueError:
        pass


if __name__ == '__main__':
    test_missing_only_on_value_positions()
    test_run_lengths_and_spacing()
    test_seed_reproducibility()
    test_no_missing_format_unchanged()
    test_collate_action_miss()
    test_unpack_action_miss()
    print("ALL TESTS PASSED: test_dynamic_missing.py")
