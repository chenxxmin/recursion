"""Test FRESH_TEST_PER_EVAL: action task rebuilds the test set at every eval.

Each eval must use a different fresh dataset seed (visible in the log),
while train data stays fixed.
"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from core import run_experiment


def _run(tmpdir, fresh, task='action'):
    main = {
        'TASK': task, 'P': 7, 'AB_PAIRS': [[1, 1], [2, 3]],
        'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1, 'BATCH_SIZE': 32,
        'EPOCHS': 3, 'LR': 0.001, 'RANDOM_SEED': 42,
        'TRAIN_LEN': 8, 'OOD_LEN': 8, 'DROPOUT': 0.0, 'MAX_UNIQUE_RATIO': 0.7,
        'WEIGHT_DECAY': 0.1, 'ENTROPY_PENALTY_WEIGHT': 0.0, 'FIRST_TASK_WEIGHT': 1.0,
        'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0, 'EARLY_STOP_NO_IMPROVE': 3000,
        'USE_LEARNABLE_PE': False, 'MLP_RATIO': 4, 'NUM_MASK': None,
        'NUM_TRAIN_SAMPLES': 200, 'NUM_TEST_SAMPLES': 32,
        'FRESH_TEST_PER_EVAL': fresh,
        'SAVE_PATH': os.path.join(tmpdir, 'model.pth'),
    }
    cfg = {'main': main, '_BATCH_RUN_MERGED': True}
    cfg_path = os.path.join(tmpdir, 'config.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_experiment(cfg_path)
    return buf.getvalue()


def test_fresh_test_per_eval_rebuilds_each_time():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = _run(tmpdir, fresh=True)
        seeds = re.findall(r'\[FreshTest\] rebuilt test set: n=32, len=8, seed=(\d+)', out)
        assert len(seeds) >= 3, f"expected >=3 fresh rebuilds, got {len(seeds)}"
        assert len(set(seeds)) == len(seeds), f"seeds must differ per eval: {seeds}"


def test_fresh_test_disabled_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = _run(tmpdir, fresh=False)
        assert '[FreshTest]' not in out


def test_fresh_test_single_recurrence():
    """Single-recurrence tasks (tribonacci) also rebuild the test set per eval:
    fresh seeds differ per eval, and the train set stays at the explicit
    NUM_TRAIN_SAMPLES rather than the MAX_UNIQUE_RATIO-derived count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = _run(tmpdir, fresh=True, task='tribonacci')
        seeds = re.findall(r'\[FreshTest\] rebuilt test set: n=32, len=8, seed=(\d+)', out)
        assert len(seeds) >= 3, f"expected >=3 fresh rebuilds, got {len(seeds)}"
        assert len(set(seeds)) == len(seeds), f"seeds must differ per eval: {seeds}"
        assert '200 + ' in out, "train set should be NUM_TRAIN_SAMPLES=200"


def test_fresh_test_mixed_recurrence():
    """mixed_ab also supports FRESH_TEST_PER_EVAL: per-rule fresh states each
    eval, seeds differ per eval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        main = {
            'TASK': 'mixed_ab', 'P': 7, 'AB_PAIRS': [[1, 1], [2, 3]],
            'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1, 'BATCH_SIZE': 32,
            'EPOCHS': 3, 'LR': 0.001, 'RANDOM_SEED': 42,
            'TRAIN_LEN': 8, 'OOD_LEN': 8, 'DROPOUT': 0.0, 'MAX_UNIQUE_RATIO': 0.7,
            'WEIGHT_DECAY': 0.1, 'ENTROPY_PENALTY_WEIGHT': 0.0, 'FIRST_TASK_WEIGHT': 1.0,
            'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0, 'EARLY_STOP_NO_IMPROVE': 3000,
            'USE_LEARNABLE_PE': False, 'MLP_RATIO': 4, 'NUM_MASK': None,
            'NUM_TEST_SAMPLES': 32, 'FRESH_TEST_PER_EVAL': True,
            'SAVE_PATH': os.path.join(tmpdir, 'model.pth'),
        }
        cfg = {'main': main, '_BATCH_RUN_MERGED': True}
        cfg_path = os.path.join(tmpdir, 'config.json')
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_experiment(cfg_path)
        out = buf.getvalue()
        seeds = re.findall(r'\[FreshTest\] rebuilt mixed test set: n=32/rule x 2, len=8, seed=(\d+)', out)
        assert len(seeds) >= 3, f"expected >=3 fresh rebuilds, got {len(seeds)}"
        assert len(set(seeds)) == len(seeds), f"seeds must differ per eval: {seeds}"


if __name__ == '__main__':
    test_fresh_test_per_eval_rebuilds_each_time()
    test_fresh_test_disabled_by_default()
    test_fresh_test_single_recurrence()
    test_fresh_test_mixed_recurrence()
    print("ALL TESTS PASSED: test_fresh_test_eval.py")
