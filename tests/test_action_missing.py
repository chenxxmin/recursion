"""Tests for action on-the-fly missing-value corruption (2026-09-09 regime).

ActionDataset always stores CLEAN (seq, loss_mask) pairs;
make_action_missing_collate corrupts value positions with fresh randomness
per batch (predict mode: targets = clean values).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from datasets import BatchTag, ActionDataset, make_action_missing_collate
from training import _unpack_batch

P, PAIRS, L = 7, [(1, 1), (1, 2)], 12


def _make(seed=42, n=200, length=L):
    return ActionDataset(p=P, ab_pairs=PAIRS, num_samples=n, length=length, seed=seed)


def _collate(missing_prob=0.5, miss_len=3):
    return make_action_missing_collate(p=P, missing_prob=missing_prob, miss_len=miss_len)


def test_dataset_always_clean():
    ds = _make()
    for item in ds.samples:
        assert isinstance(item, tuple) and len(item) == 2
        seq, mask = item
        assert all(tok != P for tok in seq.tolist())
        assert mask.sum().item() == L - 2  # x3..x_L targets only
        for k in range(2, L):
            assert mask[2 * (k - 1)] == 1.0


def test_missing_only_on_value_positions():
    ds = _make()
    collate = _collate()
    views, tag, cleans, masks = collate(ds.samples[:100])
    assert tag == BatchTag.ACTION_MISS
    n_masked = 0
    for view, clean in zip(views, cleans):
        v, c = view.tolist(), clean.tolist()
        for i, tok in enumerate(v):
            if i >= 2 and i % 2 == 0:
                assert tok >= P + 1, f"flag position {i} corrupted: {tok}"
            if tok == P:
                assert i >= 3 and i % 2 == 1, f"M at non-value position {i}"
                n_masked += 1
        for a, b in zip(v, c):
            assert a == b or a == P
    assert n_masked > 0


def test_run_lengths_and_spacing():
    ds = _make(n=300)
    collate = _collate(missing_prob=0.4, miss_len=3)
    views, _, _, _ = collate(ds.samples)
    seen_masked = 0
    for view in views:
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
        for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
            assert a2 - b1 >= 2, f"adjacent runs merged: {runs}"
    assert seen_masked > 0


def test_fresh_randomness_per_call():
    ds = _make(n=8)
    collate = _collate()
    b1 = collate(ds.samples[:8])
    b2 = collate(ds.samples[:8])
    assert not torch.equal(b1[0], b2[0])


def test_unpack_action_miss():
    ds = _make(n=4)
    collate = _collate()
    batch = collate(ds.samples[:4])
    assert batch[1] == BatchTag.ACTION_MISS
    assert batch[0].shape == batch[2].shape == (4, 2 * L - 2)
    assert batch[3].shape == (4, 2 * L - 3)
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


def test_run_experiment_action_missing_smoke():
    import contextlib
    import io
    import json
    import tempfile
    from core import run_experiment
    main = {
        'P': 7, 'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
        'BATCH_SIZE': 32, 'EPOCHS': 2, 'LR': 0.001, 'RANDOM_SEED': 42,
        'TRAIN_LEN': 8, 'OOD_LEN': 10, 'DROPOUT': 0.0,
        'WEIGHT_DECAY': 0.1, 'ENTROPY_PENALTY_WEIGHT': 0.0,
        'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0,
        'EARLY_STOP_NO_IMPROVE': 3000,
        'USE_LEARNABLE_PE': False, 'MLP_RATIO': 4,
        'TASK': 'action', 'AB_PAIRS': [[1, 1], [1, 2]],
        'NUM_TRAIN_SAMPLES': 200, 'NUM_TEST_SAMPLES': 50,
        'MISSING_PROB': 0.3, 'MISS_LEN': 2,
    }
    # On-the-fly regime: the prepared dataset stores CLEAN (seq, mask) pairs;
    # corruption happens in the collate.
    from experiment import _prepare_action
    prep = _prepare_action({'main': dict(main), '_BATCH_RUN_MERGED': True})
    sample = prep['train_dataset'].samples[0]
    assert isinstance(sample, tuple) and len(sample) == 2, (
        f"dataset must store clean (seq, mask) pairs, got {len(sample)}-tuple")
    assert prep['save_config']['missing_prob'] == 0.3
    assert prep['save_config']['miss_len'] == 2
    with tempfile.TemporaryDirectory() as tmpdir:
        main['SAVE_PATH'] = os.path.join(tmpdir, 'model.pth')
        cfg = {'main': main, '_BATCH_RUN_MERGED': True}
        cfg_path = os.path.join(tmpdir, 'config.json')
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_experiment(cfg_path)
        out = buf.getvalue()
        assert os.path.exists(main['SAVE_PATH']), "checkpoint was not saved"
        assert 'Train: Loss=' in out, "training loop output missing"
        assert 'nan' not in out.lower(), "NaN appeared in training"


if __name__ == '__main__':
    test_dataset_always_clean()
    test_missing_only_on_value_positions()
    test_run_lengths_and_spacing()
    test_fresh_randomness_per_call()
    test_unpack_action_miss()
    test_run_experiment_action_missing_smoke()
    print("ALL TESTS PASSED: test_action_missing.py")
