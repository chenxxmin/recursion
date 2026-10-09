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
                    'exposure_ratios': 0.7, 'source': 'new'}


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
                        'exposure_ratios': None, 'source': task}


def test_new_format_errors():
    _expect_error(lambda: normalize_task_config('single', {'P': 53, 'RULES': [[1, 1]]}),
                  'ORDER')
    _expect_error(lambda: normalize_task_config('single', {'P': 53, 'ORDER': 3}),
                  'RULES')
    _expect_error(lambda: normalize_task_config('single', {'P': 53, 'ORDER': 1, 'RULES': [[1]]}),
                  'ORDER')
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


if __name__ == '__main__':
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith('test_') and callable(_fn):
            _fn()
            print(f'PASS {_name}')
    print('ALL TESTS PASSED: test_scenarios.py')
