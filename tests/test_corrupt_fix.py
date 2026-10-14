"""Tests for the corrupt_fix task variant (TASK_VARIANT='corrupt_fix').

Covers: make_corrupt_fix_collate corruption/layout semantics, _unpack_batch
routing + evaluate integration, and an end-to-end run_experiment smoke test
(CPU, tiny config). Style follows test_training_smoke.py / test_missing_values.py.
"""
import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from datasets import BatchTag, make_corrupt_fix_collate
from models import FibonacciTransformer
from training import evaluate, _unpack_batch


def _addition_windows(p=7, n=8, length=6, seed=0):
    """A few clean addition (order-2) windows as plain LongTensors."""
    rng = random.Random(seed)
    seqs = []
    for _ in range(n):
        a, b = rng.randrange(p), rng.randrange(p)
        seq = [a, b]
        while len(seq) < length:
            seq.append((seq[-1] + seq[-2]) % p)
        seqs.append(torch.tensor(seq, dtype=torch.long))
    return seqs


def test_collate_layout():
    p, eos_id, pos_base, L = 7, 8, 9, 6
    seqs = _addition_windows(p=p, n=16, length=L)
    random.seed(123)
    collate = make_corrupt_fix_collate(p=p, eos_id=eos_id, pos_base=pos_base)
    x, tag, targets, mask = collate(seqs)
    assert tag == BatchTag.CORRUPT_FIX
    B = len(seqs)
    assert x.shape == (B, L + 2) and targets.shape == (B, L + 3)
    assert mask.shape == (B, L + 2)
    clean = torch.stack(seqs)
    for i in range(B):
        diff = (x[i, :L] != clean[i]).nonzero().flatten().tolist()
        assert len(diff) == 1, f"sample {i}: expected exactly 1 corrupted position, got {diff}"
        k = diff[0]
        assert int(x[i, k]) != int(clean[i, k]), "wrong value must differ from the original"
        assert 0 <= int(x[i, k]) < p, "wrong value must stay a value token"
        # EOS + teacher-forced position token at the tail of the input
        assert int(x[i, L]) == eos_id
        assert int(x[i, L + 1]) == pos_base + k
        # targets = targets_override[:, 1:]: position token at slot L, correct
        # value (the ORIGINAL clean value) at slot L+1
        assert int(targets[i, L + 1]) == pos_base + k
        assert int(targets[i, L + 2]) == int(clean[i, k])
        # loss mask: exactly the two output positions
        ones = mask[i].nonzero().flatten().tolist()
        assert ones == [L, L + 1], f"sample {i}: loss mask at {ones}"
        # targets alignment: targets[:, 1:] == x shifted, with value appended
        assert torch.equal(targets[i, 1:L + 1], x[i, 1:L + 1])


def test_collate_fresh_randomness():
    # Two calls on the same batch corrupt (with high probability) different
    # positions/values — corruption is online, not baked into the dataset.
    p, L = 11, 8
    seqs = _addition_windows(p=p, n=32, length=L)
    collate = make_corrupt_fix_collate(p=p, eos_id=p + 1, pos_base=p + 2)
    random.seed(1)
    x1, _, _, _ = collate(seqs)
    random.seed(2)
    x2, _, _, _ = collate(seqs)
    assert not torch.equal(x1, x2), "corruption must be fresh per call"


def test_unpack_and_evaluate_integration():
    p, L = 7, 6
    eos_id, pos_base = p + 1, p + 2
    vocab_size = p + 2 + L
    seqs = _addition_windows(p=p, n=16, length=L)
    collate = make_corrupt_fix_collate(p=p, eos_id=eos_id, pos_base=pos_base)
    batch = collate(seqs)
    x, loss_mask, kwargs, ab_labels, targets_override = _unpack_batch(batch, 'cpu', None)
    assert kwargs == {} and ab_labels is None
    assert loss_mask is not None and targets_override is not None
    assert torch.equal(targets_override[:, :-1], x)

    torch.manual_seed(0)
    model = FibonacciTransformer(p=p, d_model=32, n_head=1, n_layer=1,
                                 block_size=16, dropout=0.0, vocab_size=vocab_size)
    loader = DataLoader(seqs, batch_size=8, collate_fn=collate)
    loss, acc, per_pos_acc, group_acc = evaluate(model, loader, 'cpu', num_mask=0)
    assert math.isfinite(loss) and loss > 0
    assert 0.0 <= acc <= 1.0
    # accuracy is counted only at the two output positions
    assert set(per_pos_acc) <= {L, L + 1}, f"unexpected acc positions: {per_pos_acc}"
    assert set(per_pos_acc) == {L, L + 1}


def _run_experiment(tmpdir, overrides):
    from core import run_experiment
    main = {
        'P': 7, 'ORDER': 2, 'RULES': [[1, 1]], 'TASK': 'single',
        'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
        'BATCH_SIZE': 32, 'EPOCHS': 2, 'LR': 0.001, 'RANDOM_SEED': 42,
        'TRAIN_LEN': 8, 'OOD_LEN': 10, 'DROPOUT': 0.0, 'MAX_UNIQUE_RATIO': 0.7,
        'WEIGHT_DECAY': 0.1, 'FIRST_TASK_WEIGHT': 1.0,
        'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0, 'EARLY_STOP_NO_IMPROVE': 3000,
        'MLP_RATIO': 4, 'NUM_MASK': None, 'MISSING_PROB': 0.0,
    }
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


def test_run_experiment_corrupt_fix_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {'TASK_VARIANT': 'corrupt_fix'})
        assert os.path.exists(save_path), "checkpoint was not saved"
        assert 'Train: Loss=' in out, "training loop output missing"
        assert 'nan' not in out.lower(), "NaN appeared in training"
        assert '[corrupt_fix]' in out, "task banner missing variant tag"
        # 0.0% acc on a fresh model never beats best=0.0, so _best.pth is not
        # written this early; check the rolling full-state checkpoint instead.
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['variant'] == 'corrupt_fix'
        assert sc['scenario'] == 'single' and sc['rules'] == [[1, 1]]
        assert sc['eos_token_id'] == 8 and sc['pos_token_base'] == 9
        assert sc['vocab_size'] == 7 + 2 + 10  # max(TRAIN_LEN=8, OOD_LEN=10)
        assert sc['block_size'] == 16  # round_up_pow2(10 + 2)


def test_run_experiment_corrupt_fix_rejects_missing_prob():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'corrupt_fix',
                                     'MISSING_PROB': 0.1})
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected SystemExit(2) for MISSING_PROB conflict")


if __name__ == '__main__':
    test_collate_layout()
    print('PASS test_collate_layout')
    test_collate_fresh_randomness()
    print('PASS test_collate_fresh_randomness')
    test_unpack_and_evaluate_integration()
    print('PASS test_unpack_and_evaluate_integration')
    test_run_experiment_corrupt_fix_smoke()
    print('PASS test_run_experiment_corrupt_fix_smoke')
    test_run_experiment_corrupt_fix_rejects_missing_prob()
    print('PASS test_run_experiment_corrupt_fix_rejects_missing_prob')
    print('ALL TESTS PASSED: test_corrupt_fix.py')
