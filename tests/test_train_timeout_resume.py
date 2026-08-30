"""Tests for wall-clock timeout (MAX_TRAIN_HOURS) and resume (RESUME_FROM).

Covers: timeout saves a full resume checkpoint and exits with code 42 without
writing the final model; RESUME_FROM then continues training to completion.
Kept tiny (p=7, CPU) so the whole file runs in seconds.
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
    'BATCH_SIZE': 32, 'EPOCHS': 4, 'LR': 0.001, 'RANDOM_SEED': 42,
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
    code = 0
    with contextlib.redirect_stdout(buf):
        try:
            run_experiment(cfg_path)
        except SystemExit as e:
            code = e.code
    return buf.getvalue(), code, main['SAVE_PATH']


def test_timeout_saves_resume_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, code, save_path = _run(tmpdir, {'TASK': 'addition', 'MAX_TRAIN_HOURS': 0})
        resume_path = os.path.join(tmpdir, 'model_resume.pth')
        assert code == 42, f"expected exit code 42, got {code}"
        assert os.path.exists(resume_path), "resume checkpoint was not saved"
        assert not os.path.exists(save_path), "final model must not be saved on timeout"
        assert '[TIMEOUT]' in out
        assert 'Best test accuracy' not in out, "timeout must not look like a finished run"


def test_resume_from_checkpoint_completes():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, code, save_path = _run(tmpdir, {'TASK': 'addition', 'MAX_TRAIN_HOURS': 0})
        assert code == 42
        resume_path = os.path.join(tmpdir, 'model_resume.pth')

        # Resume without the time limit: must run remaining epochs to the end.
        out2, code2, save_path = _run(tmpdir, {'TASK': 'addition', 'RESUME_FROM': resume_path})
        assert code2 == 0, f"resume run exited with {code2}"
        assert os.path.exists(save_path), "final model missing after resume"
        assert '[RESUME] continue from epoch 1' in out2, "resume did not start at next epoch"
        assert 'Training epochs: 3' in out2, "resume did not run to final epoch"
        assert 'Best test accuracy' in out2


if __name__ == '__main__':
    test_timeout_saves_resume_checkpoint()
    test_resume_from_checkpoint_completes()
    print("ALL TESTS PASSED: test_train_timeout_resume.py")
