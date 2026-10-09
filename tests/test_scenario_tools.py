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


def test_eval_clean_action_build_model_legacy_order3():
    # eval_clean_action 在模块级 parse_args 并设 CUDA_VISIBLE_DEVICES，导入前后保护现场
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts')
    sys.path.insert(0, scripts_dir)
    argv, gpu = sys.argv, os.environ.get('CUDA_VISIBLE_DEVICES')
    sys.argv = ['eval_clean_action.py', 'dummy.pth']
    try:
        import eval_clean_action
    finally:
        sys.argv = argv
        if gpu is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    # 合成 legacy action_trib save_config：无 rules 新块，三系数 ab_pairs。
    # 修复前 build_model 一律按 'action'(order 2) 归一化，这里会抛
    # "规则长度 3 != order 2"；修复后按行长度走 'action_trib'。
    sc = {'p': 7, 'd_model': 16, 'n_head': 1, 'n_layer': 1, 'mlp_ratio': 4,
          'train_len': 8, 'ood_len': 8, 'missing_prob': 0.1, 'miss_len': 1,
          'ab_pairs': [[1, 1, 1], [1, 2, 3]]}
    ctx = eval_clean_action.build_model(sc)
    # 归一化结果正确：canonical rules = 逐条 reverse 的 ab_pairs，order=3
    assert ctx['save_config']['order'] == 3
    assert ctx['save_config']['rules'] == [[1, 1, 1], [3, 2, 1]]
    assert ctx['save_config']['ab_pairs'] == [[1, 1, 1], [1, 2, 3]]
    assert ctx['model'].vocab_size == 7 + 1 + 2


if __name__ == '__main__':
    test_action_query_mask_any_order()
    test_load_model_rejects_legacy_mixed_checkpoint()
    test_load_model_rejects_learnable_pe_checkpoint()
    test_load_model_new_action_checkpoint()
    test_resolve_recurrence_new_single_checkpoint()
    test_eval_clean_action_build_model_legacy_order3()
    print("ALL TESTS PASSED: test_scenario_tools.py")
