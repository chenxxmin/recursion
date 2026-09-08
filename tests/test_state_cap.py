"""Unit tests for RecurrenceDataset's STATE_SPACE_CAP subsampling path
(pure asserts; also pytest-compatible)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from datasets import RecurrenceDataset


def _trib_fn(seq, p):
    return (seq[-1] + seq[-2] + seq[-3]) % p


def _init_states(samples, init_len):
    return {tuple(s[:init_len].tolist()) for s in samples}


def test_capped_split_and_disjoint():
    p, init_len, cap = 7, 3, 100
    random.seed(0)
    ds = RecurrenceDataset(p=p, recurrence_fn=_trib_fn, init_len=init_len,
                           num_samples=70, length=10, verbose=False,
                           state_cap=cap)
    ds.run()
    assert len(ds.train_samples) == 70
    assert len(ds.test_samples) == cap - 70
    train_inits = _init_states(ds.train_samples, init_len)
    test_inits = _init_states(ds.test_samples, init_len)
    # one window per sampled state, train/test states disjoint
    assert len(train_inits) == 70 and len(test_inits) == cap - 70
    assert not (train_inits & test_inits)
    # every window follows the recurrence
    for s in ds.train_samples + ds.test_samples:
        seq = s.tolist()
        for i in range(init_len, len(seq)):
            assert seq[i] == _trib_fn(seq[i - init_len:i], p)


def test_cap_above_state_space_falls_back_to_full():
    p, init_len = 5, 3  # 125 states < cap
    random.seed(0)
    ds = RecurrenceDataset(p=p, recurrence_fn=_trib_fn, init_len=init_len,
                           num_samples=87, length=10, verbose=False,
                           state_cap=10 ** 9)
    ds.run()
    assert len(ds.train_samples) + len(ds.test_samples) == p ** init_len
    assert len(ds.train_samples) == 87


if __name__ == '__main__':
    test_capped_split_and_disjoint()
    test_cap_above_state_space_falls_back_to_full()
    print("ALL TESTS PASSED: test_state_cap.py")
