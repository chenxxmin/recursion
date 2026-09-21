"""Tests for the explicit single-recurrence data generation modes."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from datasets import RecurrenceDataset


def _add_rule(seq, p):
    return (seq[-1] + seq[-2]) % p


def _initial_state(seq, order):
    return tuple(seq[:order].tolist())


def test_sample_unique_train_has_exact_unique_count():
    ds = RecurrenceDataset(
        p=7, recurrence_fn=_add_rule, recurrence_name='add',
        init_len=2, num_samples=30, length=8, verbose=False)
    rng = random.Random(123)
    train = ds.sample_unique_train(rng=rng)

    assert len(train) == 30
    states = [_initial_state(seq, 2) for seq in train]
    assert len(set(states)) == 30
    assert ds.test_samples == []


def test_sample_random_windows_is_with_replacement():
    class SameStateRng:
        def randrange(self, stop):
            return 5

    ds = RecurrenceDataset(
        p=7, recurrence_fn=_add_rule, recurrence_name='add',
        init_len=2, num_samples=0, length=8, verbose=False)
    samples = ds.sample_random_windows(4, rng=SameStateRng())

    assert len(samples) == 4
    states = [_initial_state(seq, 2) for seq in samples]
    assert states == [states[0]] * 4


def test_full_split_still_partitions_entire_state_space():
    ds = RecurrenceDataset(
        p=3, recurrence_fn=_add_rule, recurrence_name='add',
        init_len=2, num_samples=5, length=6, verbose=False)
    random.seed(0)
    ds.run_full_split()

    assert len(ds.train_samples) == 5
    assert len(ds.train_samples) + len(ds.test_samples) == 3 ** 2


if __name__ == '__main__':
    test_sample_unique_train_has_exact_unique_count()
    test_sample_random_windows_is_with_replacement()
    test_full_split_still_partitions_entire_state_space()
    print('ALL TESTS PASSED: test_data_generation_modes.py')
