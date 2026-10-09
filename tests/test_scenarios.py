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


if __name__ == '__main__':
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith('test_') and callable(_fn):
            _fn()
            print(f'PASS {_name}')
    print('ALL TESTS PASSED: test_scenarios.py')
