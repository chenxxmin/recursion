"""Tests for order-generalized analysis tools and removed-checkpoint errors."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

import analyze_attention
from models import FibonacciTransformer


def test_action_query_mask_any_order():
    # order=2 必须与旧实现逐点一致（回归钉死）
    assert analyze_attention._make_action_query_mask(5, 2) == \
        [0, 0, 0, 1, 0, 1, 0, 1]
    # order=3: 序列 [x0 x1 x2 f x3 f x4]，x3 在 index 4，x4 在 index 6
    assert analyze_attention._make_action_query_mask(5, 3) == \
        [0, 0, 0, 0, 1, 0, 1]


def _save_ckpt(state_dict, config):
    fd, path = tempfile.mkstemp(suffix='.pth')
    os.close(fd)
    torch.save({'model_state_dict': state_dict, 'config': config}, path)
    return path


def test_load_model_rejects_legacy_mixed_checkpoint():
    model = FibonacciTransformer(p=7, d_model=16, n_head=1, n_layer=1, block_size=16)
    state = model.state_dict()
    state['rule_head.0.weight'] = torch.zeros(8, 16)  # 伪造旧 mixed 指纹
    path = _save_ckpt(state, {'p': 7, 'd_model': 16, 'n_head': 1, 'n_layer': 1,
                              'block_size': 16, 'ab_pairs': [[1, 1]], 'order': 2})
    try:
        analyze_attention.load_model(path)
        assert False, "legacy mixed checkpoint must raise"
    except RuntimeError as exc:
        assert 'rule_head' in str(exc)
    finally:
        os.remove(path)


def test_load_model_rejects_learnable_pe_checkpoint():
    model = FibonacciTransformer(p=7, d_model=16, n_head=1, n_layer=1, block_size=16)
    state = model.state_dict()
    state['wpe.weight'] = torch.zeros(16, 16)  # 伪造 wpe 指纹
    path = _save_ckpt(state, {'p': 7, 'd_model': 16, 'n_head': 1, 'n_layer': 1,
                              'block_size': 16})
    try:
        analyze_attention.load_model(path)
        assert False, "wpe checkpoint must raise"
    except RuntimeError as exc:
        assert 'wpe' in str(exc)
    finally:
        os.remove(path)


def test_load_model_new_action_checkpoint():
    model = FibonacciTransformer(p=7, d_model=16, n_head=1, n_layer=1,
                                 block_size=16, vocab_size=7 + 1 + 2)
    path = _save_ckpt(model.state_dict(),
                      {'p': 7, 'd_model': 16, 'n_head': 1, 'n_layer': 1,
                       'block_size': 16, 'vocab_size': 10, 'pad_token_id': 7,
                       'scenario': 'action', 'order': 3, 'rules': [[1, 1, 1]],
                       'ab_pairs': [[1, 1, 1]], 'recurrence': 'action',
                       'train_len': 8})
    try:
        loaded, ckpt = analyze_attention.load_model(path)
        assert isinstance(loaded, FibonacciTransformer)
        rec = analyze_attention._resolve_recurrence(ckpt['config'])
        assert rec['is_action'] and rec['order'] == 3 and rec['init_len'] == 3
    finally:
        os.remove(path)


def test_resolve_recurrence_new_single_checkpoint():
    rec = analyze_attention._resolve_recurrence(
        {'p': 23, 'scenario': 'single', 'order': 3, 'rules': [[1, 2, 3]],
         'recurrence': 'tribonacci'})
    assert not rec['is_action'] and not rec.get('is_mixed')
    assert rec['init_len'] == 3
    assert rec['next_val']([1, 2, 3]) == (1 * 3 + 2 * 2 + 3 * 1) % 23


if __name__ == '__main__':
    test_action_query_mask_any_order()
    test_load_model_rejects_legacy_mixed_checkpoint()
    test_load_model_rejects_learnable_pe_checkpoint()
    test_load_model_new_action_checkpoint()
    test_resolve_recurrence_new_single_checkpoint()
    print("ALL TESTS PASSED: test_scenario_tools.py")
