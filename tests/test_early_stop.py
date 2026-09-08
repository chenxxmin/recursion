"""Tests for the regime-dependent early-stop rules (2026-09-07):

- static-split regime (no fresh test): stop DIRECTLY once test acc reaches
  EARLY_STOP_ACCURACY (no extra epochs);
- fresh-test regime (FRESH_TEST_PER_EVAL): stop only when the last-10-eval
  mean test acc exceeds EARLY_STOP_ACCURACY.
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

_BASE_MAIN = {
    'P': 7, 'A': 1, 'B': 1, 'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
    'BATCH_SIZE': 32, 'EPOCHS': 50, 'LR': 0.001, 'RANDOM_SEED': 42,
    'TRAIN_LEN': 8, 'OOD_LEN': 8, 'DROPOUT': 0.0, 'MAX_UNIQUE_RATIO': 0.7,
    'WEIGHT_DECAY': 0.1, 'ENTROPY_PENALTY_WEIGHT': 0.0, 'FIRST_TASK_WEIGHT': 1.0,
    'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 0.0, 'EARLY_STOP_NO_IMPROVE': 3000,
    'USE_LEARNABLE_PE': False, 'MLP_RATIO': 4, 'NUM_MASK': None,
    'NUM_TRAIN_SAMPLES': 200, 'NUM_TEST_SAMPLES': 32,
}


def _run(tmpdir, overrides):
    main = dict(_BASE_MAIN)
    main.update(overrides)
    main['SAVE_PATH'] = os.path.join(tmpdir, 'model.pth')
    cfg = {'main': main, '_BATCH_RUN_MERGED': True}
    cfg_path = os.path.join(tmpdir, 'config.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_experiment(cfg_path)
    return buf.getvalue()


def _epochs_seen(out):
    return [int(m) for m in re.findall(r'^Epoch\s+(\d+)', out, re.M)]


def test_static_split_stops_directly():
    # threshold 0.0 -> the very first eval already "reaches" it; with the old
    # extra-epochs rule training would have continued past epoch 0.
    with tempfile.TemporaryDirectory() as tmpdir:
        out = _run(tmpdir, {'TASK': 'tribonacci'})
        assert _epochs_seen(out) == [0], _epochs_seen(out)
        assert 'stopping directly' in out


def test_fresh_test_needs_10_eval_mean():
    # threshold 0.0 -> any 10-eval window exceeds it; the run must survive
    # until the window fills (epoch 9) and stop there, not earlier.
    with tempfile.TemporaryDirectory() as tmpdir:
        out = _run(tmpdir, {'TASK': 'action', 'AB_PAIRS': [[1, 1], [2, 3]],
                            'FRESH_TEST_PER_EVAL': True})
        assert _epochs_seen(out) == list(range(10)), _epochs_seen(out)
        assert 'last-10-eval mean test acc' in out


if __name__ == '__main__':
    test_static_split_stops_directly()
    test_fresh_test_needs_10_eval_mean()
    print("ALL TESTS PASSED: test_early_stop.py")
