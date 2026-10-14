"""Tests for the fill_blank task variant (TASK_VARIANT='fill_blank').

Same shape as test_corrupt_fix.py: collate layout, fresh randomness,
_unpack_batch + evaluate integration, end-to-end smoke (CPU), config errors.
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

from datasets import BatchTag, make_fill_blank_collate
from models import FibonacciTransformer
from training import evaluate, _unpack_batch
from test_corrupt_fix import _addition_windows, _run_experiment


def test_collate_layout():
    p, eos_id, L = 7, 8, 6
    seqs = _addition_windows(p=p, n=16, length=L)
    random.seed(123)
    collate = make_fill_blank_collate(p=p, eos_id=eos_id)
    x, tag, targets, mask = collate(seqs)
    assert tag == BatchTag.FILL_BLANK
    B = len(seqs)
    assert x.shape == (B, L + 1) and targets.shape == (B, L + 2)
    assert mask.shape == (B, L + 1)
    clean = torch.stack(seqs)
    for i in range(B):
        # exactly one position replaced by the BLANK token (= p)
        diff = (x[i, :L] != clean[i]).nonzero().flatten().tolist()
        assert len(diff) == 1, f"sample {i}: expected exactly 1 blank, got {diff}"
        k = diff[0]
        assert int(x[i, k]) == p, "blanked position must carry the blank token id (= p)"
        assert int(x[i, L]) == eos_id
        # target: correct value at the EOS slot; targets = tgt[:, 1:]
        assert int(targets[i, L + 1]) == int(clean[i, k])
        ones = mask[i].nonzero().flatten().tolist()
        assert ones == [L], f"sample {i}: loss mask at {ones}"
        assert torch.equal(targets[i, 1:L + 1], x[i, 1:L + 1])


def test_collate_fresh_randomness():
    p, L = 11, 8
    seqs = _addition_windows(p=p, n=32, length=L)
    collate = make_fill_blank_collate(p=p, eos_id=p + 1)
    random.seed(1)
    x1, _, _, _ = collate(seqs)
    random.seed(2)
    x2, _, _, _ = collate(seqs)
    # blank positions differ between calls (compare against clean to locate)
    clean = torch.stack(seqs)
    k1 = [(x1[i, :L] != clean[i]).nonzero().item() for i in range(32)]
    k2 = [(x2[i, :L] != clean[i]).nonzero().item() for i in range(32)]
    assert k1 != k2, "blank positions must be fresh per call"


def test_unpack_and_evaluate_integration():
    p, L = 7, 6
    eos_id = p + 1
    vocab_size = p + 2
    seqs = _addition_windows(p=p, n=16, length=L)
    collate = make_fill_blank_collate(p=p, eos_id=eos_id)
    batch = collate(seqs)
    x, loss_mask, kwargs, ab_labels, targets_override = _unpack_batch(batch, 'cpu', None)
    assert kwargs == {} and ab_labels is None
    assert loss_mask is not None and targets_override is not None
    assert torch.equal(targets_override[:, :-1], x)

    torch.manual_seed(0)
    model = FibonacciTransformer(p=p, d_model=32, n_head=1, n_layer=1,
                                 block_size=8, dropout=0.0, vocab_size=vocab_size)
    loader = DataLoader(seqs, batch_size=8, collate_fn=collate)
    loss, acc, per_pos_acc, group_acc = evaluate(model, loader, 'cpu', num_mask=0)
    assert math.isfinite(loss) and loss > 0
    assert 0.0 <= acc <= 1.0
    # accuracy counted only at the single output position (EOS slot)
    assert set(per_pos_acc) == {L}, f"unexpected acc positions: {per_pos_acc}"


def test_run_experiment_fill_blank_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {'TASK_VARIANT': 'fill_blank'})
        assert os.path.exists(save_path), "checkpoint was not saved"
        assert 'Train: Loss=' in out, "training loop output missing"
        assert 'nan' not in out.lower(), "NaN appeared in training"
        assert '[fill_blank]' in out, "task banner missing variant tag"
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['variant'] == 'fill_blank'
        assert sc['scenario'] == 'single' and sc['rules'] == [[1, 1]]
        assert sc['eos_token_id'] == 8 and sc['blank_token_id'] == 7
        assert sc['vocab_size'] == 9          # P + 2
        assert sc['block_size'] == 16         # round_up_pow2(max(8,10) + 1)


def test_run_experiment_fill_blank_rejects_missing_prob():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'fill_blank',
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
    test_run_experiment_fill_blank_smoke()
    print('PASS test_run_experiment_fill_blank_smoke')
    test_run_experiment_fill_blank_rejects_missing_prob()
    print('PASS test_run_experiment_fill_blank_rejects_missing_prob')
    print('ALL TESTS PASSED: test_fill_blank.py')
