"""Unit and CPU smoke tests for TASK_VARIANT='clean_value_set'."""
import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile
from unittest import mock

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from datasets import BatchTag, make_clean_value_set_collate
from experiment import _normalize_experiment_config
from models import FibonacciTransformer, _lm_loss
from training import _unpack_batch, evaluate, train_epoch


def _windows(p=7, n=4, length=5, seed=0):
    rng = random.Random(seed)
    seqs = []
    for _ in range(n):
        seq = [rng.randrange(p), rng.randrange(p)]
        while len(seq) < length:
            seq.append((seq[-1] + seq[-2]) % p)
        seqs.append(torch.tensor(seq, dtype=torch.long))
    return seqs


def _run_experiment(tmpdir, overrides, filename='model.pth'):
    from core import run_experiment
    main = {
        'P': 7, 'ORDER': 2, 'RULES': [[1, 1]], 'TASK': 'single',
        'TASK_VARIANT': 'clean_value_set',
        'D_MODEL': 16, 'N_HEAD': 1, 'N_LAYER': 1, 'MLP_RATIO': 2,
        'BATCH_SIZE': 16, 'EPOCHS': 1, 'LR': 0.001, 'RANDOM_SEED': 42,
        'TRAIN_LEN': 5, 'OOD_LEN': 5, 'DROPOUT': 0.0,
        'MAX_UNIQUE_RATIO': 0.5, 'NUM_TRAIN_SAMPLES': 12,
        'WEIGHT_DECAY': 0.01, 'FIRST_TASK_WEIGHT': 1.0,
        'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0,
        'EARLY_STOP_NO_IMPROVE': 3000, 'NUM_MASK': None,
        'MISSING_PROB': 0.0,
    }
    main.update(overrides)
    save_path = os.path.join(tmpdir, filename)
    main['SAVE_PATH'] = save_path
    cfg_path = os.path.join(tmpdir, filename + '.config.json')
    with open(cfg_path, 'w') as f:
        json.dump({'main': main, '_BATCH_RUN_MERGED': True}, f)
    buf = io.StringIO()
    # Keep the smoke test on CPU regardless of the host's GPU availability.
    with mock.patch('torch.cuda.is_available', return_value=False):
        with contextlib.redirect_stdout(buf):
            run_experiment(cfg_path)
    return buf.getvalue(), save_path


def test_collate_layout_and_arbitrary_order():
    p, eos_id = 7, 8
    seq = torch.tensor([1, 1, 2, 3, 5])
    x, tag, target_seq, mask, value_sets = make_clean_value_set_collate(
        p=p, eos_id=eos_id, body_from=1)([seq])
    assert tag == BatchTag.CLEAN_VALUE_SET
    assert torch.equal(x[0], torch.tensor([1, 1, 2, 3, 5, 8]))
    assert torch.equal(target_seq[0], torch.tensor([1, 1, 2, 3, 5, 8, 0]))
    assert torch.equal(target_seq[:, 1:],
                       torch.tensor([[1, 2, 3, 5, 8, 0]]))
    assert torch.equal(mask[0], torch.tensor([0, 1, 1, 1, 0, 1]).float())
    assert value_sets == [[1, 2, 3, 5]]

    # L=6/order=3: three body targets, no len5/order2 special case.
    seq3 = torch.tensor([1, 2, 3, 6, 11, 3])
    x3, _, target3, mask3, sets3 = make_clean_value_set_collate(
        p=17, eos_id=18, body_from=2)([seq3])
    assert x3.shape == (1, 7) and target3.shape == (1, 8)
    assert torch.equal(mask3[0], torch.tensor([0, 0, 1, 1, 1, 0, 1]).float())
    assert sets3 == [[1, 2, 3, 6, 11]]


def test_set_loss_formula_and_placeholder_independence():
    p, eos_id = 7, 8
    seqs = [torch.tensor([1, 1, 2, 3, 5]),
            torch.tensor([2, 2, 2, 2, 2])]
    _, _, target_seq, mask, value_sets = make_clean_value_set_collate(
        p=p, eos_id=eos_id, body_from=1)(seqs)
    targets = target_seq[:, 1:]
    torch.manual_seed(9)
    logits = torch.randn(2, 6, 9, dtype=torch.float64)

    actual = _lm_loss(logits, targets, mask, value_sets=value_sets)
    log_probs = F.log_softmax(logits, dim=-1)
    expected_terms = []
    for row, values in enumerate(value_sets):
        body_sum = F.cross_entropy(logits[row, 1:4], targets[row, 1:4],
                                   reduction='sum')
        ids = torch.tensor(values, dtype=torch.long)
        set_nll = -torch.logsumexp(log_probs[row, 5].index_select(0, ids), dim=-1)
        expected_terms.extend([body_sum, set_nll])
    expected = sum(expected_terms) / 8.0  # two samples x four effective positions
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    for placeholder in (7, 8, 999, -999):
        changed = targets.clone()
        changed[:, -1] = placeholder
        changed_loss = _lm_loss(logits, changed, mask, value_sets=value_sets)
        torch.testing.assert_close(changed_loss, actual, rtol=0, atol=0)


class _FixedModel:
    def __init__(self, logits):
        self.logits = logits

    def eval(self):
        return self

    def __call__(self, x, targets, loss_mask, value_sets=None):
        return self.logits, _lm_loss(self.logits, targets, loss_mask,
                                     value_sets=value_sets)


def test_membership_accuracy_and_body_set_metrics():
    p, eos_id = 7, 8
    seqs = [torch.tensor([1, 1, 2, 3, 5]),
            torch.tensor([2, 2, 4, 6, 3])]
    collate = make_clean_value_set_collate(p=p, eos_id=eos_id, body_from=1)
    _, _, target_seq, _, value_sets = collate(seqs)
    targets = target_seq[:, 1:]

    # Full-vocab predictions default to EOS: body misses and set nonmembership.
    pred = torch.full((2, 6), eos_id, dtype=torch.long)
    pred[0, 1:4] = targets[0, 1:4]  # row 0: all three body positions correct
    pred[1, 2] = targets[1, 2]      # row 1: one body position correct
    pred[0, 5] = value_sets[0][0]   # set hit despite placeholder=0
    pred[1, 5] = eos_id             # EOS is not a set member
    logits = torch.zeros(2, 6, 9)
    logits.scatter_(2, pred.unsqueeze(-1), 10.0)

    loader = DataLoader(seqs, batch_size=2, collate_fn=collate)
    loss, acc, per_pos, aux = evaluate(_FixedModel(logits), loader, 'cpu', num_mask=0)
    assert math.isfinite(loss)
    assert math.isclose(acc, 5 / 8)          # (4 body + 1 set) / 8
    assert math.isclose(aux['body_acc'], 4 / 6)
    assert math.isclose(aux['set_acc'], 1 / 2)
    assert set(per_pos) == {1, 2, 3, 5}
    assert math.isclose(per_pos[1], 0.5)
    assert math.isclose(per_pos[2], 1.0)
    assert math.isclose(per_pos[3], 0.5)
    assert math.isclose(per_pos[5], 0.5)     # EOS slot uses membership


def test_small_model_forward_backward_and_train_epoch():
    p, eos_id = 7, 8
    seqs = _windows(p=p, n=4, length=5)
    collate = make_clean_value_set_collate(p=p, eos_id=eos_id, body_from=1)
    batch = collate(seqs)
    x, mask, kwargs, ab_labels, target_override = _unpack_batch(batch, 'cpu', None)
    assert ab_labels is None and kwargs['value_sets'] == batch[4]
    targets = target_override[:, 1:]

    torch.manual_seed(3)
    model = FibonacciTransformer(p=p, d_model=16, n_head=1, n_layer=1,
                                 block_size=8, dropout=0.0, mlp_ratio=2,
                                 vocab_size=p + 2)
    logits, loss = model(x, targets, mask, **kwargs)
    assert logits.shape == (4, 6, 9) and math.isfinite(loss.item())
    loss.backward()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(seqs, batch_size=4, collate_fn=collate)
    train_loss, acc, per_pos, grad_stats = train_epoch(
        model, loader, optimizer, 'cpu', num_mask=0)
    assert math.isfinite(train_loss) and 0.0 <= acc <= 1.0
    assert set(per_pos) == {1, 2, 3, 5}
    assert 0.0 <= grad_stats['body_acc'] <= 1.0
    assert 0.0 <= grad_stats['set_acc'] <= 1.0


def test_normalize_and_end_to_end_save_then_value_only_load():
    norm = _normalize_experiment_config('single', {
        'P': 7, 'ORDER': 2, 'RULES': [[1, 1]],
        'TASK_VARIANT': 'clean_value_set'})
    assert norm['scenario'] == 'single' and norm['variant'] == 'clean_value_set'
    for bad in ({'MISSING_PROB': 0.1},):
        cfg = {'P': 7, 'ORDER': 2, 'RULES': [[1, 1]],
               'TASK_VARIANT': 'clean_value_set', **bad}
        try:
            _normalize_experiment_config('single', cfg)
            raise AssertionError('clean_value_set must reject MISSING_PROB')
        except ValueError:
            pass

    with tempfile.TemporaryDirectory() as tmpdir:
        out, clean_path = _run_experiment(tmpdir, {})
        assert os.path.exists(clean_path)
        assert '[clean_value_set]' in out
        assert 'CleanValueSet:' in out and 'body_acc=' in out and 'set_acc=' in out
        assert 'PosProbe' not in out
        assert 'nan' not in out.lower()
        latest = torch.load(os.path.splitext(clean_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['variant'] == 'clean_value_set'
        assert sc['scenario'] == 'single' and sc['rules'] == [[1, 1]]
        assert sc['vocab_size'] == 9 and sc['eos_token_id'] == 8
        assert sc['block_size'] == 8 and sc['body_from'] == 1
        assert sc['body_supervision'] is True
        assert 'blank_token_id' not in sc

        # The existing second stage consumes the P+2 checkpoint without any
        # skipped wte/lm_head (or other) tensors.
        out2, value_path = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'corrupt_fix',
            'CORRUPT_FIX_VALUE_ONLY': True,
            'BODY_SUPERVISION': False,
            'POS_PROBE': False,
            'INIT_FROM': clean_path,
        }, filename='value_only.pth')
        assert os.path.exists(value_path)
        assert '[INIT_FROM]' in out2 and 'skipped 0: []' in out2
        assert 'nan' not in out2.lower()


def test_mixed_eos_only_collate_loss_and_accuracy():
    p, eos_id = 7, 8
    seqs = [torch.tensor([1, 1, 2, 3, 5]),
            torch.tensor([2, 2, 4, 6, 3]),
            torch.tensor([1, 2, 3, 5, 1]),
            torch.tensor([3, 1, 4, 5, 2])]
    random.seed(11)
    collate = make_clean_value_set_collate(
        p=p, eos_id=eos_id, body_from=1,
        body_supervision=False, corrupt_ratio=0.5)
    batch = collate(seqs)
    x, tag, target_seq, mask, value_sets = batch
    assert tag == BatchTag.CLEAN_VALUE_SET
    assert sum(values is None for values in value_sets) == 2
    assert sum(values is not None for values in value_sets) == 2
    for row, seq in enumerate(seqs):
        assert mask[row].sum().item() == 1.0 and mask[row, 5].item() == 1.0
        if value_sets[row] is None:
            changed = (x[row, :5] != seq).nonzero().flatten().tolist()
            assert len(changed) == 1
            k = changed[0]
            assert int(x[row, k]) != int(seq[k])
            assert int(target_seq[row, -1]) == int(seq[k])
        else:
            assert torch.equal(x[row], torch.cat([seq, torch.tensor([eos_id])]))
            assert value_sets[row] == sorted(set(seq.tolist()))
            assert int(target_seq[row, -1]) == 0

    targets = target_seq[:, 1:]
    torch.manual_seed(9)
    logits = torch.randn(4, 6, 9, dtype=torch.float64)
    actual = _lm_loss(logits, targets, mask, value_sets=value_sets)
    log_probs = F.log_softmax(logits, dim=-1)
    expected_terms = []
    for row, values in enumerate(value_sets):
        if values is None:
            expected_terms.append(F.cross_entropy(logits[row, -1:], targets[row, -1:],
                                                  reduction='sum'))
        else:
            ids = torch.tensor(values, dtype=torch.long)
            expected_terms.append(-torch.logsumexp(
                log_probs[row, -1].index_select(0, ids), dim=-1))
    expected = sum(expected_terms) / 4.0
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    # Only clean-row placeholders are sanitized; corrupt true values remain CE targets.
    changed_targets = targets.clone()
    for row, values in enumerate(value_sets):
        if values is not None:
            changed_targets[row, -1] = 999
    changed = _lm_loss(logits, changed_targets, mask, value_sets=value_sets)
    torch.testing.assert_close(actual, changed, rtol=0, atol=0)

    # Both clean rows hit their sets; both corrupt rows miss (EOS != value).
    pred = torch.full((4, 6), eos_id, dtype=torch.long)
    for row, values in enumerate(value_sets):
        if values is not None:
            pred[row, -1] = values[0]
    fixed_logits = torch.zeros(4, 6, 9)
    fixed_logits.scatter_(2, pred.unsqueeze(-1), 10.0)
    loss, acc, per_pos, aux = evaluate(
        _FixedModel(fixed_logits), [batch], 'cpu', num_mask=0)
    assert math.isfinite(loss) and math.isclose(acc, 0.5)
    assert math.isclose(aux['set_acc'], 1.0)
    assert math.isclose(aux['corrupt_value_acc'], 0.0)
    assert set(per_pos) == {5}


def test_mixed_normalize_and_end_to_end():
    norm = _normalize_experiment_config('single', {
        'P': 7, 'ORDER': 2, 'RULES': [[1, 1]],
        'TASK_VARIANT': 'clean_value_set',
        'CLEAN_VALUE_SET_CORRUPT_RATIO': 0.5,
        'BODY_SUPERVISION': False})
    assert norm['scenario'] == 'single' and norm['variant'] == 'clean_value_set'
    for bad in (-0.1, 1.1, None, '0.5'):
        try:
            _normalize_experiment_config('single', {
                'P': 7, 'ORDER': 2, 'RULES': [[1, 1]],
                'TASK_VARIANT': 'clean_value_set',
                'CLEAN_VALUE_SET_CORRUPT_RATIO': bad})
            raise AssertionError(f'invalid corrupt ratio accepted: {bad!r}')
        except ValueError:
            pass
    try:
        _normalize_experiment_config('single', {
            'P': 7, 'ORDER': 2, 'RULES': [[1, 1]],
            'TASK_VARIANT': 'corrupt_fix',
            'CLEAN_VALUE_SET_CORRUPT_RATIO': 0.5})
        raise AssertionError('mix ratio must be rejected outside clean_value_set')
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        out, clean_path = _run_experiment(tmpdir, {
            'CLEAN_VALUE_SET_CORRUPT_RATIO': 0.5,
            'BODY_SUPERVISION': False,
        })
        assert 'CorruptMix:' in out and 'corrupt_value_acc=' in out
        latest = torch.load(os.path.splitext(clean_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['variant'] == 'clean_value_set'
        assert sc['corrupt_ratio'] == 0.5 and sc['body_supervision'] is False
        out2, value_path = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'corrupt_fix',
            'CORRUPT_FIX_VALUE_ONLY': True,
            'BODY_SUPERVISION': False,
            'POS_PROBE': False,
            'INIT_FROM': clean_path,
        }, filename='mixed_value_only.pth')
        assert os.path.exists(value_path)
        assert '[INIT_FROM]' in out2 and 'skipped 0: []' in out2


if __name__ == '__main__':
    test_collate_layout_and_arbitrary_order()
    print('PASS test_collate_layout_and_arbitrary_order')
    test_set_loss_formula_and_placeholder_independence()
    print('PASS test_set_loss_formula_and_placeholder_independence')
    test_membership_accuracy_and_body_set_metrics()
    print('PASS test_membership_accuracy_and_body_set_metrics')
    test_small_model_forward_backward_and_train_epoch()
    print('PASS test_small_model_forward_backward_and_train_epoch')
    test_normalize_and_end_to_end_save_then_value_only_load()
    print('PASS test_normalize_and_end_to_end_save_then_value_only_load')
    test_mixed_eos_only_collate_loss_and_accuracy()
    print('PASS test_mixed_eos_only_collate_loss_and_accuracy')
    test_mixed_normalize_and_end_to_end()
    print('PASS test_mixed_normalize_and_end_to_end')
    print('ALL TESTS PASSED: test_clean_value_set.py')
