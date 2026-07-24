"""Wiring tests for core._prepare_mixed_recurrence (mixed_ab order=2, mixed_abc order=3)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

import core


def _base_main():
    return {
        'P': 5, 'D_MODEL': 16, 'N_HEAD': 1, 'N_LAYER': 1, 'BATCH_SIZE': 16,
        'TRAIN_LEN': 8, 'OOD_LEN': 10, 'DROPOUT': 0.0,
        'ENTROPY_PENALTY_WEIGHT': 0.0, 'MAX_UNIQUE_RATIO': 0.5,
        'USE_AB_TAG': False,
    }


def test_prepare_mixed_ab_wiring():
    cfg = {'main': {**_base_main(), 'AB_PAIRS': [[1, 1], [1, 2]]}}
    random.seed(0)
    torch.manual_seed(0)
    ctx = core._prepare_mixed_recurrence(cfg, 'cpu', 2)
    assert ctx['post_train_mode'] == 'mixed_ab'
    assert ctx['order'] == 2
    assert [r.coeffs for r in ctx['rules']] == [(1, 1), (1, 2)]
    # state space 5**2 = 25 per rule; default ratio 0.5 -> 12 train samples per rule
    assert len(ctx['train_dataset']) == 24
    assert len(ctx['test_dataset']) == 2 * 25 - 24
    assert ctx['model'].vocab_size == 5 + 1  # no tag
    assert ctx['num_mask'] == 2
    assert ctx['save_config']['order'] == 2
    assert ctx['save_config']['ab_pairs'] == [[1, 1], [1, 2]]


def test_prepare_mixed_abc_wiring():
    cfg = {'main': {**_base_main(), 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]],
                    'USE_AB_TAG': True, 'MIXED_AB_MAX_UNIQUE_RATIOS': [0.4, 0.4]}}
    random.seed(0)
    torch.manual_seed(0)
    ctx = core._prepare_mixed_recurrence(cfg, 'cpu', 3)
    assert ctx['order'] == 3
    assert [r.coeffs for r in ctx['rules']] == [(1, 1, 1), (1, 2, 3)]
    # state space 5**3 = 125 per rule; ratio 0.4 -> 50 train samples per rule
    assert len(ctx['train_dataset']) == 100
    assert len(ctx['test_dataset']) == 2 * 125 - 100
    assert ctx['model'].vocab_size == 5 + 1 + 2  # tag: p + 1 + N_rules
    assert ctx['model'].pad_token_id == 5 + 2
    seq, idx = ctx['train_dataset'][0]
    assert seq[0].item() == 5 + idx  # leading flag token p + rule_idx
    assert len(seq) == 8 + 1         # TRAIN_LEN + flag token


def test_prepare_ratios_length_mismatch():
    cfg = {'main': {**_base_main(), 'AB_PAIRS': [[1, 1], [1, 2]],
                    'MIXED_AB_MAX_UNIQUE_RATIOS': [0.5]}}
    random.seed(0)
    torch.manual_seed(0)
    try:
        core._prepare_mixed_recurrence(cfg, 'cpu', 2)
        assert False, "ratios/rules length mismatch must raise"
    except AssertionError:
        pass


if __name__ == '__main__':
    test_prepare_mixed_ab_wiring()
    test_prepare_mixed_abc_wiring()
    test_prepare_ratios_length_mismatch()
    print("ALL TESTS PASSED: test_prepare_mixed_recurrence.py")
