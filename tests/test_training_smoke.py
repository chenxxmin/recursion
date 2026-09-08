"""End-to-end training smoke test: run_experiment on tiny configs (CPU).

Covers the main pipeline that unit tests don't reach: config routing,
dataset build, train_epoch/evaluate loop, checkpoint saving, and the
stage-3 final generation test output. Kept tiny (p=7, 2 epochs) so the
whole file runs in seconds.
"""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from core import run_experiment

_BASE_MAIN = {
    'P': 7, 'A': 1, 'B': 1, 'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
    'BATCH_SIZE': 32, 'EPOCHS': 2, 'LR': 0.001, 'RANDOM_SEED': 42,
    'TRAIN_LEN': 8, 'OOD_LEN': 10, 'DROPOUT': 0.0, 'MAX_UNIQUE_RATIO': 0.7,
    'WEIGHT_DECAY': 0.1, 'ENTROPY_PENALTY_WEIGHT': 0.0, 'FIRST_TASK_WEIGHT': 1.0,
    'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0, 'EARLY_STOP_NO_IMPROVE': 3000,
    'USE_LEARNABLE_PE': False, 'MLP_RATIO': 4, 'NUM_MASK': None,
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
    return buf.getvalue(), main['SAVE_PATH']


def _check_common(out, save_path):
    assert os.path.exists(save_path), "checkpoint was not saved"
    assert 'Train: Loss=' in out, "training loop output missing"
    assert 'Final generation test' in out, "stage-3 final test did not run"
    assert 'Exposed' in out and 'Unexposed' in out, "exposure stats missing"
    assert 'nan' not in out.lower(), "NaN appeared in training"
    # Rolling checkpoints: best (weights only) and latest (full resume state)
    base = os.path.splitext(save_path)[0]
    import torch
    best = torch.load(base + '_best.pth', weights_only=False)
    assert 'model_state_dict' in best and 'best_accuracy' in best
    latest = torch.load(base + '_latest.pth', weights_only=False)
    assert 'optimizer_state_dict' in latest and 'next_epoch' in latest


def test_run_experiment_addition():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run(tmpdir, {'TASK': 'addition'})
        _check_common(out, save_path)
        assert 'X(k)=(1*X(k-1)+1*X(k-2)) mod 7' in out


def test_run_experiment_mixed_ab():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run(tmpdir, {
            'TASK': 'mixed_ab', 'AB_PAIRS': [[1, 1], [1, 2]], 'USE_AB_TAG': False,
            'MIXED_AB_MAX_UNIQUE_RATIOS': [0.7, 0.7]})
        _check_common(out, save_path)
        assert 'Rule 1' in out and 'Rule 2' in out, "per-rule final test missing"


def test_run_experiment_nonlinear_mul():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run(tmpdir, {'TASK': 'nonlinear_mul'})
        _check_common(out, save_path)
        assert 'X(k)=(X(k-2)*X(k-1)^2) mod 7' in out


if __name__ == '__main__':
    test_run_experiment_addition()
    test_run_experiment_mixed_ab()
    test_run_experiment_nonlinear_mul()
    print("ALL TESTS PASSED: test_training_smoke.py")
