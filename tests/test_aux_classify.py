"""AUX_CLASSIFY: rule-classification auxiliary head (mixed scenario only).

Default ON for mixed since 2026-10-10 (user decision); AUX_CLASSIFY=false
turns it off. Covers the four layers the feature touches:
  1. scenarios.normalize_task_config validation (mixed-only legality; the key
     is validated but never copied into the normalized dict);
  2. models.FibonacciTransformer rule_head (n_rules > 0) and its rule loss;
  3. end-to-end smoke: mixed trains on CPU with the head; the default-on path
     (key absent) is bitwise identical to explicit AUX_CLASSIFY=true, and
     AUX_CLASSIFY=false produces a head-less checkpoint;
  4. analyze_attention.load_model checkpoint compatibility (rule_head loads,
     cond_wte/ab_emb still rejected).
"""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

import analyze_attention
from models import FibonacciTransformer, RULE_LOSS_WEIGHT
from scenarios import normalize_task_config
from core import run_experiment


# ==================== 1. scenarios validation ====================

def test_scenarios_aux_classify_legality():
    mixed_new = {'P': 5, 'ORDER': 2, 'RULES': [[1, 1], [1, 2]], 'AUX_CLASSIFY': True}
    norm = normalize_task_config('mixed', mixed_new)
    assert norm['scenario'] == 'mixed'
    assert 'AUX_CLASSIFY' not in norm, "normalization must not consume AUX_CLASSIFY"

    norm = normalize_task_config('mixed_ab',
                                 {'P': 5, 'AB_PAIRS': [[1, 1], [1, 2]], 'AUX_CLASSIFY': True})
    assert norm['scenario'] == 'mixed' and 'AUX_CLASSIFY' not in norm
    norm = normalize_task_config('mixed_abc',
                                 {'P': 5, 'ABC_PAIRS': [[1, 1, 1]], 'AUX_CLASSIFY': True})
    assert norm['scenario'] == 'mixed' and 'AUX_CLASSIFY' not in norm

    # 归一化层只校验 true 的合法性；false / 不写在归一化层都无行为
    # （默认开是 experiment._prepare_mixed_recurrence 的兜底，不在这一层）
    normalize_task_config('single', {'P': 5, 'ORDER': 2, 'RULES': [[1, 1]],
                                     'AUX_CLASSIFY': False})
    normalize_task_config('addition', {'P': 5, 'A': 1, 'B': 1})

    # true outside mixed: explicit misuse -> ValueError pointing at mixed
    bad_configs = [
        ('single', {'P': 5, 'ORDER': 2, 'RULES': [[1, 1]], 'AUX_CLASSIFY': True}),
        ('action', {'P': 5, 'ORDER': 2, 'RULES': [[1, 1], [1, 2]], 'AUX_CLASSIFY': True}),
        ('addition', {'P': 5, 'A': 1, 'B': 1, 'AUX_CLASSIFY': True}),
        ('action_trib', {'P': 5, 'ABC_PAIRS': [[1, 1, 1]], 'AUX_CLASSIFY': True}),
        ('multiplication', {'P': 5, 'AUX_CLASSIFY': True}),
    ]
    for task, cfg in bad_configs:
        try:
            normalize_task_config(task, cfg)
            assert False, f"AUX_CLASSIFY=true on {task} must raise"
        except ValueError as exc:
            assert 'mixed' in str(exc), f"error must name mixed as the only legal scenario: {exc}"


# ==================== 2. model rule_head ====================

def _toy_batch(p=7, b=4, t=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, p, (b, t), generator=g)
    targets = idx[:, 1:]
    loss_mask = torch.ones(b, t - 1)
    ab_labels = torch.arange(b) % 2
    return idx, targets, loss_mask, ab_labels


def test_model_rule_head_aux_loss():
    torch.manual_seed(0)
    model = FibonacciTransformer(p=7, d_model=16, n_head=1, n_layer=1,
                                 block_size=16, n_rules=2, order=2)
    assert model.rule_head is not None
    idx, targets, loss_mask, ab_labels = _toy_batch()

    out = model(idx, targets, loss_mask)
    assert isinstance(out, tuple) and len(out) == 2, "forward must stay a 2-tuple"
    _, loss_plain = out
    out = model(idx, targets, loss_mask, ab_labels=ab_labels)
    assert isinstance(out, tuple) and len(out) == 2
    _, loss_aux = out
    # rule loss is a plain cross-entropy (> 0), weighted by RULE_LOSS_WEIGHT
    assert RULE_LOSS_WEIGHT == 0.5
    assert loss_aux.item() > loss_plain.item()
    # ab_labels=None must not engage the head
    _, loss_none = model(idx, targets, loss_mask, ab_labels=None)
    assert torch.equal(loss_none, loss_plain)


def test_model_no_rule_head_ignores_ab_labels():
    torch.manual_seed(0)
    model = FibonacciTransformer(p=7, d_model=16, n_head=1, n_layer=1, block_size=16)
    assert model.rule_head is None
    idx, targets, loss_mask, ab_labels = _toy_batch()
    _, loss_plain = model(idx, targets, loss_mask)
    _, loss_with = model(idx, targets, loss_mask, ab_labels=ab_labels)
    assert torch.equal(loss_plain, loss_with), \
        "n_rules=0 must make ab_labels a no-op"


# ==================== 3. training smoke + false-path determinism ====================

_BASE_MAIN = {
    'P': 7, 'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
    'BATCH_SIZE': 32, 'EPOCHS': 2, 'LR': 0.001, 'RANDOM_SEED': 42,
    'TRAIN_LEN': 8, 'OOD_LEN': 10, 'DROPOUT': 0.0,
    'WEIGHT_DECAY': 0.1, 'FIRST_TASK_WEIGHT': 1.0,
    'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0, 'EARLY_STOP_NO_IMPROVE': 3000,
    'MLP_RATIO': 4, 'NUM_MASK': None,
    'TASK': 'mixed_ab', 'AB_PAIRS': [[1, 1], [1, 2]],
    'MIXED_AB_MAX_UNIQUE_RATIOS': [0.7, 0.7],
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


def test_smoke_mixed_aux_classify_true():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run(tmpdir, {'AUX_CLASSIFY': True})
        assert os.path.exists(save_path), "checkpoint was not saved"
        assert 'Train: Loss=' in out, "training loop output missing"
        assert 'nan' not in out.lower(), "NaN appeared in training"
        ckpt = torch.load(save_path, weights_only=False)
        assert any(k.startswith('rule_head.') for k in ckpt['model_state_dict']), \
            "AUX_CLASSIFY=true checkpoint must contain rule_head weights"
        assert ckpt['config']['aux_classify'] is True
        assert 'per-rule acc' in out, "per-rule eval stats missing"


def test_aux_classify_default_on_matches_explicit_true():
    # 默认开（2026-10-10）：不写 AUX_CLASSIFY 与显式 true 必须逐位一致
    # （同 seed 下 checkpoint 完全相同，且都含 rule_head 权重）。
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        _, p1 = _run(d1, {})
        _, p2 = _run(d2, {'AUX_CLASSIFY': True})
        c1 = torch.load(p1, weights_only=False)
        c2 = torch.load(p2, weights_only=False)
        s1, s2 = c1['model_state_dict'], c2['model_state_dict']
        assert any(k.startswith('rule_head.') for k in s1), \
            "mixed default must build the rule head"
        assert c1['config']['aux_classify'] is True
        assert set(s1) == set(s2)
        for k in s1:
            assert torch.equal(s1[k], s2[k]), f"default path diverged at {k}"


def test_aux_classify_false_disables_head():
    # 显式 false：无 rule_head，且与 pre-feature 架构一致（无额外参数）。
    with tempfile.TemporaryDirectory() as tmpdir:
        _, save_path = _run(tmpdir, {'AUX_CLASSIFY': False})
        ckpt = torch.load(save_path, weights_only=False)
        assert not any(k.startswith('rule_head.') for k in ckpt['model_state_dict'])
        assert ckpt['config']['aux_classify'] is False


# ==================== 4. load_model checkpoint compatibility ====================

def _save_ckpt(state_dict, config):
    fd, path = tempfile.mkstemp(suffix='.pth')
    os.close(fd)
    torch.save({'model_state_dict': state_dict, 'config': config}, path)
    return path


def test_load_model_rule_head_checkpoint():
    torch.manual_seed(0)
    model = FibonacciTransformer(p=7, d_model=16, n_head=1, n_layer=1,
                                 block_size=16, n_rules=2, order=2)
    # new-style config (aux_classify declared) and legacy-style config
    # (rule_head keys only, no aux_classify) must both load; n_rules is
    # inferred from the rule_head weight shape.
    for config in (
        {'p': 7, 'd_model': 16, 'n_head': 1, 'n_layer': 1, 'block_size': 16,
         'scenario': 'mixed', 'order': 2, 'rules': [[1, 1], [1, 2]],
         'ab_pairs': [[1, 1], [1, 2]], 'aux_classify': True},
        {'p': 7, 'd_model': 16, 'n_head': 1, 'n_layer': 1, 'block_size': 16,
         'order': 2, 'ab_pairs': [[1, 1], [1, 2]]},
    ):
        path = _save_ckpt(model.state_dict(), config)
        try:
            loaded, _ = analyze_attention.load_model(path)
            assert isinstance(loaded, FibonacciTransformer)
            assert loaded.rule_head is not None
            assert loaded.rule_head[-1].out_features == 2
        finally:
            os.remove(path)


def test_load_model_cond_wte_still_rejected():
    model = FibonacciTransformer(p=7, d_model=16, n_head=1, n_layer=1, block_size=16)
    state = model.state_dict()
    state['cond_wte.weight'] = torch.zeros(8, 16)  # 伪造 cond_wte 指纹
    path = _save_ckpt(state, {'p': 7, 'd_model': 16, 'n_head': 1, 'n_layer': 1,
                              'block_size': 16, 'ab_pairs': [[1, 1]], 'order': 2})
    try:
        analyze_attention.load_model(path)
        assert False, "cond_wte checkpoint must raise"
    except RuntimeError as exc:
        assert 'cond_wte' in str(exc)
    finally:
        os.remove(path)


if __name__ == '__main__':
    test_scenarios_aux_classify_legality()
    test_model_rule_head_aux_loss()
    test_model_no_rule_head_ignores_ab_labels()
    test_smoke_mixed_aux_classify_true()
    test_aux_classify_default_on_matches_explicit_true()
    test_aux_classify_false_disables_head()
    test_load_model_rule_head_checkpoint()
    test_load_model_cond_wte_still_rejected()
    print("ALL TESTS PASSED: test_aux_classify.py")
