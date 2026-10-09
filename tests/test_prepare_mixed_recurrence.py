"""Wiring tests for experiment._prepare_mixed_recurrence（mixed 场景，任意 order）。"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

import core
from scenarios import normalize_task_config


def _base_main():
    return {
        'P': 5, 'D_MODEL': 16, 'N_HEAD': 1, 'N_LAYER': 1, 'BATCH_SIZE': 16,
        'TRAIN_LEN': 8, 'OOD_LEN': 10, 'DROPOUT': 0.0,
    }


def _prepare(main, task):
    return core._prepare_mixed_recurrence({'main': main}, 'cpu',
                                          normalize_task_config(task, main))


def test_prepare_mixed_ab_wiring():
    main = {**_base_main(), 'AB_PAIRS': [[1, 1], [1, 2]], 'MAX_UNIQUE_RATIO': 0.5}
    random.seed(0)
    torch.manual_seed(0)
    ctx = _prepare(main, 'mixed_ab')
    assert ctx['order'] == 2
    assert [r.coeffs for r in ctx['rules']] == [(1, 1), (1, 2)]
    # state space 5**2 = 25 per rule; ratio 0.5 -> 12 train samples per rule
    assert len(ctx['train_dataset']) == 24
    assert len(ctx['test_dataset']) == 2 * 25 - 24
    assert ctx['model'].vocab_size == 5 + 1  # 词表恒 P+1
    assert ctx['model'].pad_token_id == 5
    assert ctx['num_mask'] == 1  # 新默认 order-1
    assert ctx['save_config']['scenario'] == 'mixed'
    assert ctx['save_config']['order'] == 2
    assert ctx['save_config']['rules'] == [[1, 1], [1, 2]]
    assert ctx['save_config']['ab_pairs'] == [[1, 1], [1, 2]]
    assert type(ctx['model']).__name__ == 'FibonacciTransformer'


def test_prepare_mixed_abc_wiring():
    main = {**_base_main(), 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]],
            'MIXED_AB_MAX_UNIQUE_RATIOS': [0.4, 0.4]}
    random.seed(0)
    torch.manual_seed(0)
    ctx = _prepare(main, 'mixed_abc')
    assert ctx['order'] == 3
    assert [r.coeffs for r in ctx['rules']] == [(1, 1, 1), (1, 2, 3)]
    # state space 5**3 = 125 per rule; ratio 0.4 -> 50 train samples per rule
    assert len(ctx['train_dataset']) == 100
    assert len(ctx['test_dataset']) == 2 * 125 - 100
    assert ctx['model'].vocab_size == 5 + 1
    assert ctx['num_mask'] == 2  # order-1


def test_prepare_new_format_equals_legacy():
    # 新写法（TASK=mixed + ORDER + RULES + EXPOSURE_RATIOS）与旧写法同 seed 数据集一致
    old_main = {**_base_main(), 'AB_PAIRS': [[1, 1], [1, 2]], 'MAX_UNIQUE_RATIO': 0.5}
    new_main = {**_base_main(), 'ORDER': 2, 'RULES': [[1, 1], [1, 2]],
                'EXPOSURE_RATIOS': [0.5, 0.5]}
    random.seed(0)
    torch.manual_seed(0)
    old = _prepare(old_main, 'mixed_ab')
    random.seed(0)
    torch.manual_seed(0)
    new = _prepare(new_main, 'mixed')
    assert len(old['train_dataset']) == len(new['train_dataset'])
    for (so, lo), (sn, ln) in zip(old['train_dataset'], new['train_dataset']):
        assert lo == ln and so.tolist() == sn.tolist()


def test_prepare_num_train_samples_override():
    main = {**_base_main(), 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]], 'NUM_TRAIN_SAMPLES': 10}
    random.seed(0)
    torch.manual_seed(0)
    ctx = _prepare(main, 'mixed_abc')
    assert len(ctx['train_dataset']) == 20
    main = {**_base_main(), 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]],
            'NUM_TRAIN_SAMPLES': [10, 30]}
    random.seed(0)
    torch.manual_seed(0)
    ctx = _prepare(main, 'mixed_abc')
    assert len(ctx['train_dataset']) == 40
    labels = [idx for _, idx in ctx['train_dataset']]
    assert labels.count(0) == 10 and labels.count(1) == 30
    main = {**_base_main(), 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]],
            'NUM_TRAIN_SAMPLES': [10]}
    random.seed(0)
    torch.manual_seed(0)
    try:
        _prepare(main, 'mixed_abc')
        assert False, "NUM_TRAIN_SAMPLES/rules length mismatch must raise"
    except AssertionError:
        pass


def test_ratios_length_mismatch_raises():
    main = {**_base_main(), 'AB_PAIRS': [[1, 1], [1, 2]],
            'MIXED_AB_MAX_UNIQUE_RATIOS': [0.5]}
    try:
        _prepare(main, 'mixed_ab')
        assert False, "ratios/rules length mismatch must raise"
    except ValueError:
        pass  # 长度校验已上移到归一化层（ValueError）


if __name__ == '__main__':
    test_prepare_mixed_ab_wiring()
    test_prepare_mixed_abc_wiring()
    test_prepare_new_format_equals_legacy()
    test_prepare_num_train_samples_override()
    test_ratios_length_mismatch_raises()
    print("ALL TESTS PASSED: test_prepare_mixed_recurrence.py")
