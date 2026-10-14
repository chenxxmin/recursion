"""Unit tests for src/scenarios.py (scenario normalization layer)."""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from scenarios import (SCENARIOS, EXCEPTION_TASKS, scenario_of,
                       normalize_task_config)


def _expect_error(fn, *fragments):
    try:
        fn()
    except ValueError as exc:
        msg = str(exc)
        for frag in fragments:
            assert frag in msg, f"{frag!r} not in error: {msg}"
        return
    raise AssertionError(f"expected ValueError containing {fragments}")


def test_new_single_happy():
    norm = normalize_task_config('single', {'P': 53, 'ORDER': 3, 'RULES': [[1, 2, 3]]})
    assert norm == {'scenario': 'single', 'order': 3, 'rules': [[1, 2, 3]],
                    'exposure_ratios': 0.7, 'variant': None, 'const': 0,
                    'source': 'new'}


def test_new_mixed_scalar_exposure():
    norm = normalize_task_config('mixed', {'P': 23, 'ORDER': 2,
                                           'RULES': [[1, 1], [2, 3]],
                                           'EXPOSURE_RATIOS': 0.5})
    assert norm['scenario'] == 'mixed' and norm['order'] == 2
    assert norm['rules'] == [[1, 1], [2, 3]]
    assert norm['exposure_ratios'] == [0.5, 0.5]  # scalar -> per-rule list


def test_new_mixed_list_exposure_and_default():
    norm = normalize_task_config('mixed', {'P': 23, 'ORDER': 4, 'RULES': [[1, 0, 0, 1]]})
    assert norm['exposure_ratios'] == [0.7]  # default 0.7 per rule
    norm = normalize_task_config('mixed', {'P': 23, 'ORDER': 2,
                                           'RULES': [[1, 1], [2, 3], [3, 5]],
                                           'EXPOSURE_RATIOS': [0.3, 0.4, 0.5]})
    assert norm['exposure_ratios'] == [0.3, 0.4, 0.5]


def test_new_action_no_exposure():
    norm = normalize_task_config('action', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1], [2, 1]]})
    assert norm['scenario'] == 'action' and norm['exposure_ratios'] is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        norm = normalize_task_config('action', {'P': 53, 'ORDER': 2,
                                                'RULES': [[1, 1]],
                                                'EXPOSURE_RATIOS': 0.5})
    assert norm['exposure_ratios'] is None
    assert any('EXPOSURE_RATIOS' in str(w.message) for w in caught)


def test_exceptions_passthrough():
    for task in EXCEPTION_TASKS:
        norm = normalize_task_config(task, {'P': 53})
        assert norm == {'scenario': task, 'order': None, 'rules': None,
                        'exposure_ratios': None, 'variant': None, 'const': 0,
                        'source': task}


def test_new_format_errors():
    _expect_error(lambda: normalize_task_config('single', {'P': 53, 'RULES': [[1, 1]]}),
                  'ORDER')
    _expect_error(lambda: normalize_task_config('single', {'P': 53, 'ORDER': 3}),
                  'RULES')
    _expect_error(lambda: normalize_task_config('single', {'P': 53, 'ORDER': 0, 'RULES': [[]]}),
                  'ORDER')
    # ORDER=1 仅 single 合法（等比递推）；mixed/action 报错
    norm = normalize_task_config('single', {'P': 53, 'ORDER': 1, 'RULES': [[3]]})
    assert norm['order'] == 1 and norm['rules'] == [[3]]
    _expect_error(lambda: normalize_task_config('mixed', {'P': 53, 'ORDER': 1,
                                                          'RULES': [[1], [2]]}),
                  'ORDER=1')
    _expect_error(lambda: normalize_task_config('action', {'P': 53, 'ORDER': 1,
                                                           'RULES': [[1]]}),
                  'ORDER=1')
    _expect_error(lambda: normalize_task_config('mixed', {'P': 53, 'ORDER': 2,
                                                          'RULES': [[1, 1], [1, 2, 3]]}),
                  '长度')
    _expect_error(lambda: normalize_task_config('mixed', {'P': 53, 'ORDER': 2,
                                                          'RULES': [[1, 53]]}),
                  '越界')
    _expect_error(lambda: normalize_task_config('single', {'P': 53, 'ORDER': 2,
                                                           'RULES': [[1, 1], [2, 3]]}),
                  '恰好 1 条')
    _expect_error(lambda: normalize_task_config('mixed', {'P': 53, 'ORDER': 2,
                                                          'RULES': [[1, 1], [2, 3]],
                                                          'EXPOSURE_RATIOS': [0.5]}),
                  'EXPOSURE_RATIOS')
    err = None
    try:
        normalize_task_config('fibonacci', {'P': 53})
    except ValueError as exc:
        err = str(exc)
    assert err is not None and 'single' in err and 'mixed_ab' in err and 'action_trib' in err


def test_scenario_of():
    assert scenario_of('single') == 'single'
    assert scenario_of('nonlinear_mul') == 'nonlinear_mul'
    assert scenario_of('nonsense') is None


def test_legacy_single_mapping():
    assert normalize_task_config('addition', {'P': 23, 'A': 2, 'B': 3})['rules'] == [[2, 3]]
    norm = normalize_task_config('tribonacci', {'P': 23, 'A': 1, 'B': 2, 'C': 3})
    assert norm['scenario'] == 'single' and norm['order'] == 3
    assert norm['rules'] == [[1, 2, 3]] and norm['source'] == 'tribonacci'
    norm = normalize_task_config('tetranacci', {'P': 23, 'A': 1, 'B': 1, 'C': 1, 'D': 1})
    assert norm['order'] == 4 and norm['rules'] == [[1, 1, 1, 1]]
    # 系数缺省全 1（沿用 single_rule_from_task 的旧默认）
    assert normalize_task_config('addition', {'P': 23})['rules'] == [[1, 1]]


def test_legacy_mixed_mapping():
    norm = normalize_task_config('mixed_ab', {'P': 23, 'AB_PAIRS': [[3, 5], [7, 11]]})
    assert norm['scenario'] == 'mixed' and norm['order'] == 2
    assert norm['rules'] == [[3, 5], [7, 11]]  # mixed 旧键本就是 canonical，直接用
    norm = normalize_task_config('mixed_abc', {'P': 23, 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]]})
    assert norm['order'] == 3 and norm['rules'] == [[1, 1, 1], [1, 2, 3]]
    # 缺省 pairs（沿用旧 config.json main 段默认）
    assert normalize_task_config('mixed_ab', {'P': 23})['rules'] == [[1, 1], [1, 2]]
    assert normalize_task_config('mixed_abc', {'P': 23})['rules'] == [[1, 1, 1], [1, 2, 3]]


def test_legacy_action_mapping_reverses():
    # action 旧约定：第一个系数乘最老的值 -> canonical 需逐条 reverse。
    norm = normalize_task_config('action', {'P': 53, 'AB_PAIRS': [[1, 1], [1, 2]]})
    assert norm['scenario'] == 'action' and norm['order'] == 2
    assert norm['rules'] == [[1, 1], [2, 1]]
    norm = normalize_task_config('dynamic_mixed', {'P': 53, 'AB_PAIRS': [[3, 4]]})
    assert norm['scenario'] == 'action' and norm['rules'] == [[4, 3]]
    norm = normalize_task_config('action_trib', {'P': 53, 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]]})
    assert norm['order'] == 3 and norm['rules'] == [[1, 1, 1], [3, 2, 1]]
    # 缺省 pairs
    assert normalize_task_config('action', {'P': 53})['rules'] == [[1, 1], [2, 1]]
    assert normalize_task_config('action_trib', {'P': 53})['rules'] == [[1, 1, 1], [3, 2, 1]]


def test_legacy_exposure_keys():
    norm = normalize_task_config('addition', {'P': 23, 'MAX_UNIQUE_RATIO': 0.3})
    assert norm['exposure_ratios'] == 0.3
    norm = normalize_task_config('mixed_ab', {'P': 23, 'AB_PAIRS': [[1, 1], [2, 3]],
                                              'MIXED_AB_MAX_UNIQUE_RATIOS': [0.2, 0.4]})
    assert norm['exposure_ratios'] == [0.2, 0.4]
    # mixed 旧 JSON 只有标量 MAX_UNIQUE_RATIO 时广播到每条规则
    norm = normalize_task_config('mixed_ab', {'P': 23, 'AB_PAIRS': [[1, 1], [2, 3]],
                                              'MAX_UNIQUE_RATIO': 0.6})
    assert norm['exposure_ratios'] == [0.6, 0.6]


def test_removed_features_raise():
    # 规格 §1.5：每条已删/写死功能各一例，报错消息含键名。
    cases = [
        ({'PREDICT_MISSING': False}, 'PREDICT_MISSING'),
        ({'MISS_SECOND': True}, 'MISS_SECOND'),
        ({'USE_LEARNABLE_PE': True}, 'USE_LEARNABLE_PE'),
        ({'USE_AB_TAG': True}, 'USE_AB_TAG'),
        ({'USE_CONDITIONAL_WTE': True}, 'USE_CONDITIONAL_WTE'),
        ({'COND_WTE_SHARED_RATIO': 0.5}, 'COND_WTE_SHARED_RATIO'),
        ({'COND_FIX': 'WTE'}, 'COND_FIX'),
        ({'COND_FIX_START': 0.9}, 'COND_FIX_START'),
        ({'COND_FIX_START_A1': 0.9, 'COND_FIX_START_A2': 0.8}, 'COND_FIX_START_A1'),
        ({'USE_AMP': True}, 'USE_AMP'),
        ({'AMP_DTYPE': 'bfloat16'}, 'AMP_DTYPE'),
        ({'ALLOW_TF32': False}, 'ALLOW_TF32'),
        ({'ENTROPY_PENALTY_WEIGHT': 0.5}, 'ENTROPY_PENALTY_WEIGHT'),
    ]
    for extra, key in cases:
        cfg = {'P': 23, 'A': 1, 'B': 1, **extra}
        _expect_error(lambda c=cfg: normalize_task_config('addition', c), key)


def test_consistent_hardwired_keys_warn_not_raise():
    # 与写死值一致的旧键：警告后忽略，不报错。
    cfg = {'P': 23, 'A': 1, 'B': 1, 'USE_AMP': False, 'USE_AB_TAG': False,
           'USE_LEARNABLE_PE': False, 'PREDICT_MISSING': True, 'ALLOW_TF32': True,
           'ENTROPY_PENALTY_WEIGHT': 0.0, 'MISS_SECOND': False,
           'USE_CONDITIONAL_WTE': False, 'COND_WTE_SHARED_RATIO': 0.0,
           'SKIP_FINAL_GENERATION_TEST': True}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        norm = normalize_task_config('addition', cfg)
    assert norm['rules'] == [[1, 1]]


def test_conflicting_keys():
    # 一致 -> warning + 新键优先
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        norm = normalize_task_config('mixed', {'P': 23, 'ORDER': 2,
                                               'RULES': [[1, 1], [2, 3]],
                                               'AB_PAIRS': [[1, 1], [2, 3]]})
    assert norm['rules'] == [[1, 1], [2, 3]]
    assert any('AB_PAIRS' in str(w.message) for w in caught)
    # action 旧键按反转后与 canonical RULES 比较
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        norm = normalize_task_config('action', {'P': 53, 'ORDER': 2,
                                                'RULES': [[2, 1]],
                                                'AB_PAIRS': [[1, 2]]})
    assert norm['rules'] == [[2, 1]]
    # 不一致 -> 报错
    _expect_error(lambda: normalize_task_config('mixed', {'P': 23, 'ORDER': 2,
                                                          'RULES': [[1, 1]],
                                                          'AB_PAIRS': [[9, 9]]}),
                  '不一致')
    _expect_error(lambda: normalize_task_config('single', {'P': 23, 'ORDER': 2,
                                                           'RULES': [[1, 1]], 'B': 5}),
                  '不一致')
    _expect_error(lambda: normalize_task_config('tribonacci', {'P': 23, 'A': 1, 'B': 2,
                                                               'C': 3, 'RULES': [[1, 2, 9]]}),
                  '不一致')
    _expect_error(lambda: normalize_task_config('mixed_ab', {'P': 23,
                                                             'AB_PAIRS': [[1, 1]],
                                                             'ORDER': 3}),
                  'ORDER')


def test_action_exposure_mismatch_ignored():
    # Task 1 评审裁决：action 收到任何形状的 EXPOSURE_RATIOS（含长度不匹配
    # 的列表）都 warn 后忽略，exposure_ratios 恒 None，不做长度校验。
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        norm = normalize_task_config('action', {'P': 53, 'ORDER': 2,
                                                'RULES': [[1, 1]],
                                                'EXPOSURE_RATIOS': [0.1, 0.2, 0.3]})
    assert norm['exposure_ratios'] is None
    assert any('EXPOSURE_RATIOS' in str(w.message) for w in caught)


def test_task_variant_corrupt_fix():
    # 默认：无 TASK_VARIANT 键 -> variant=None
    norm = normalize_task_config('single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]]})
    assert norm['variant'] is None
    # 显式 null 等价
    norm = normalize_task_config('single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]],
                                            'TASK_VARIANT': None})
    assert norm['variant'] is None
    # single + corrupt_fix -> 通过
    norm = normalize_task_config('single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]],
                                            'TASK_VARIANT': 'corrupt_fix'})
    assert norm['variant'] == 'corrupt_fix'
    # 旧 task 名（映射到 single）也支持
    norm = normalize_task_config('addition', {'P': 23, 'TASK_VARIANT': 'corrupt_fix'})
    assert norm['variant'] == 'corrupt_fix' and norm['scenario'] == 'single'
    # 未知变体名 -> 报错
    _expect_error(lambda: normalize_task_config(
        'single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]], 'TASK_VARIANT': 'nope'}),
        'TASK_VARIANT')
    # mixed/action + corrupt_fix -> 尚未实现，报错
    _expect_error(lambda: normalize_task_config(
        'mixed', {'P': 23, 'ORDER': 2, 'RULES': [[1, 1], [2, 3]],
                  'TASK_VARIANT': 'corrupt_fix'}),
        'corrupt_fix', 'single')
    _expect_error(lambda: normalize_task_config(
        'action', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]], 'TASK_VARIANT': 'corrupt_fix'}),
        'corrupt_fix')
    # 例外 task + corrupt_fix -> 报错
    _expect_error(lambda: normalize_task_config(
        'multiplication', {'P': 53, 'TASK_VARIANT': 'corrupt_fix'}),
        'corrupt_fix')
    # 与 MISSING_PROB 互斥 -> 报错
    _expect_error(lambda: normalize_task_config(
        'single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]],
                   'TASK_VARIANT': 'corrupt_fix', 'MISSING_PROB': 0.1}),
        'MISSING_PROB')
    # NUM_MASK 无意义 -> warn 忽略，不报错
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        norm = normalize_task_config('single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]],
                                                'TASK_VARIANT': 'corrupt_fix',
                                                'NUM_MASK': 2})
    assert norm['variant'] == 'corrupt_fix'
    assert any('NUM_MASK' in str(w.message) for w in caught)


def test_task_variant_fill_blank():
    # single + fill_blank -> 通过
    norm = normalize_task_config('single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]],
                                            'TASK_VARIANT': 'fill_blank'})
    assert norm['variant'] == 'fill_blank'
    # 非 single -> 报错
    _expect_error(lambda: normalize_task_config(
        'mixed', {'P': 23, 'ORDER': 2, 'RULES': [[1, 1], [2, 3]],
                  'TASK_VARIANT': 'fill_blank'}),
        'fill_blank', 'single')
    # 与 MISSING_PROB 互斥
    _expect_error(lambda: normalize_task_config(
        'single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]],
                   'TASK_VARIANT': 'fill_blank', 'MISSING_PROB': 0.2}),
        'MISSING_PROB')


def test_const_term():
    # 默认 0；single 非零合法（等差数列 = ORDER=1 + RULES=[[1]] + CONST=d）
    norm = normalize_task_config('single', {'P': 1009, 'ORDER': 1, 'RULES': [[1]],
                                            'CONST': 1})
    assert norm['const'] == 1
    # CONST: 0 / null 显式给也合法
    assert normalize_task_config('single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]],
                                            'CONST': None})['const'] == 0
    # 越界 / 非整数 -> 报错
    _expect_error(lambda: normalize_task_config(
        'single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]], 'CONST': 53}), '越界')
    _expect_error(lambda: normalize_task_config(
        'single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]], 'CONST': 0.5}), '整数')
    # mixed/action/例外 task + CONST 非零 -> 报错
    _expect_error(lambda: normalize_task_config(
        'mixed', {'P': 23, 'ORDER': 2, 'RULES': [[1, 1], [2, 3]], 'CONST': 1}), 'single')
    _expect_error(lambda: normalize_task_config(
        'multiplication', {'P': 53, 'CONST': 1}), 'single')
    # 旧 single 名（映射到 single 场景）允许 CONST
    assert normalize_task_config('addition', {'P': 53, 'CONST': 2})['const'] == 2


def test_fill_blank_all_positions_gate():
    # 仅 fill_blank 支持；其他变体/无变体 -> 报错
    _expect_error(lambda: normalize_task_config(
        'single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]],
                   'FILL_BLANK_ALL_POSITIONS': True}), 'fill_blank')
    _expect_error(lambda: normalize_task_config(
        'single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]],
                   'TASK_VARIANT': 'corrupt_fix', 'FILL_BLANK_ALL_POSITIONS': True}),
        'fill_blank')
    norm = normalize_task_config('single', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1]],
                                            'TASK_VARIANT': 'fill_blank',
                                            'FILL_BLANK_ALL_POSITIONS': True})
    assert norm['variant'] == 'fill_blank'


if __name__ == '__main__':
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith('test_') and callable(_fn):
            _fn()
            print(f'PASS {_name}')
    print('ALL TESTS PASSED: test_scenarios.py')
