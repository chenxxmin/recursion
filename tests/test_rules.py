"""Unit tests for src/rules.py (pure asserts; also pytest-compatible)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from rules import (LinearRecurrenceRule, rules_from_config,
                   single_rule_from_task, save_config_extra, task_from_save_config)


def test_order_property():
    assert LinearRecurrenceRule(coeffs=(1, 1), p=23).order == 2
    assert LinearRecurrenceRule(coeffs=(1, 2, 3), p=23).order == 3


def test_name_auto_generated():
    assert LinearRecurrenceRule(coeffs=(1, 2, 4), p=23).name == "a1b2c4"
    assert LinearRecurrenceRule(coeffs=(3, 5), p=23).name == "a3b5"
    assert LinearRecurrenceRule(coeffs=(1, 1), p=23, name="custom").name == "custom"


def test_next_fn_matches_legacy_order2():
    p = 23
    for a, b in [(1, 1), (3, 5), (7, 11), (1, 2)]:
        fn = LinearRecurrenceRule(coeffs=(a, b), p=p).next_fn()
        for x0 in range(p):
            for x1 in range(p):
                seq = [x0, x1]
                assert fn(seq, p) == (a * seq[-1] + b * seq[-2]) % p


def test_next_fn_matches_legacy_order3():
    p = 23
    for a, b, c in [(1, 1, 1), (1, 2, 3)]:
        fn = LinearRecurrenceRule(coeffs=(a, b, c), p=p).next_fn()
        for x0 in range(p):
            for x1 in range(p):
                for x2 in range(p):
                    seq = [x0, x1, x2]
                    assert fn(seq, p) == (a * seq[-1] + b * seq[-2] + c * seq[-3]) % p


def test_rules_from_config_order2():
    rules = rules_from_config([[3, 5], [7, 11]], 23)
    assert [r.coeffs for r in rules] == [(3, 5), (7, 11)]
    assert all(r.p == 23 for r in rules)
    assert rules[0].name == "a3b5"


def test_rules_from_config_any_order():
    rules = rules_from_config([[1, 2, 3]], 23)
    assert [r.coeffs for r in rules] == [(1, 2, 3)]
    rules = rules_from_config([[1, 2, 3, 4, 5]], 23)  # 任意 order，不再有 2/3 写死映射
    assert rules[0].order == 5


def test_rules_from_config_validation():
    for bad in (None, [], [[1, 1], [1, 2, 3]], [[1, 99]], [[1]]):
        try:
            rules_from_config(bad, 23)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


def test_single_rule_from_task():
    p = 23
    init_len, fn, name = single_rule_from_task('addition', {'P': p, 'A': 2, 'B': 3})
    assert init_len == 2 and fn([4, 5], p) == (2 * 5 + 3 * 4) % p and 'mod 23' in name
    # lowercase checkpoint keys and coefficient defaults
    init_len, fn, _ = single_rule_from_task('addition', {'p': p})
    assert init_len == 2 and fn([4, 5], p) == (5 + 4) % p
    init_len, fn, _ = single_rule_from_task('tribonacci', {'P': p, 'A': 1, 'B': 2, 'C': 3})
    assert init_len == 3 and fn([1, 2, 3], p) == (1 * 3 + 2 * 2 + 3 * 1) % p
    init_len, fn, name = single_rule_from_task('tetranacci', {'P': p, 'A': 1, 'B': 2, 'C': 3, 'D': 4})
    assert init_len == 4 and fn([1, 2, 3, 4], p) == (1 * 4 + 2 * 3 + 3 * 2 + 4 * 1) % p
    assert 'X(k-4)' in name
    # tetranacci coefficient default is 1 (lowercase checkpoint keys accepted)
    init_len, fn, _ = single_rule_from_task('tetranacci', {'p': p})
    assert init_len == 4 and fn([1, 2, 3, 4], p) == (4 + 3 + 2 + 1) % p
    _, fn, _ = single_rule_from_task('multiplication', {'P': p})
    assert fn([4, 5], p) == 20 % p
    _, fn, _ = single_rule_from_task('nonlinear', {'P': p})
    assert fn([4, 5], p) == (25 + 4) % p
    _, fn, name = single_rule_from_task('nonlinear_mul', {'P': p})
    assert fn([4, 5], p) == (4 * 5 * 5) % p and 'X(k-2)*X(k-1)^2' in name
    # x2=0 collapses to 0 regardless of x1 (non-bijective state map)
    assert fn([7, 0], p) == 0
    try:
        single_rule_from_task('mixed_ab', {'P': p})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_single_rule_from_task_single_scenario():
    p = 23
    init_len, fn, name = single_rule_from_task('single', {'P': p, 'RULES': [[2, 3, 1]]})
    assert init_len == 3 and fn([1, 2, 3], p) == (2 * 3 + 3 * 2 + 1 * 1) % p
    assert 'mod 23' in name
    # checkpoint 小写键
    init_len, fn, _ = single_rule_from_task('single', {'p': p, 'rules': [[5, 6]]})
    assert init_len == 2 and fn([4, 5], p) == (5 * 5 + 6 * 4) % p
    try:
        single_rule_from_task('single', {'P': p, 'RULES': [[1, 1], [2, 2]]})
        assert False, "single 多规则必须报"
    except ValueError:
        pass


def test_save_config_round_trip():
    """save_config_extra（写）与 task_from_save_config（读）必须一致：
    线性 single -> 'single'；例外 task 名原样往返；新块优先于旧字段。"""
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
    from scenarios import normalize_task_config
    for task in ('addition', 'tribonacci', 'tetranacci'):
        cfg = {'P': 23, 'A': 2, 'B': 3, 'C': 4, 'D': 5}
        norm = normalize_task_config(task, cfg)
        extra = save_config_extra(norm, cfg)
        assert task_from_save_config(extra) == 'single', (task, extra)
        assert extra['scenario'] == 'single' and extra['order'] == norm['order']
        assert extra['rules'] == [list(norm['rules'][0])]
    for task in ('multiplication', 'nonlinear', 'nonlinear_mul'):
        extra = save_config_extra(normalize_task_config(task, {'P': 23}), {'P': 23})
        assert task_from_save_config(extra) == task
    # 旧 checkpoint（无 scenario 块）反推不变
    assert task_from_save_config({'a': 1, 'b': 2, 'c': 3,
                                  'recurrence': 'tribonacci'}) == 'tribonacci'
    assert task_from_save_config({'ab_pairs': [[1, 1]], 'order': 2}) is None
    assert task_from_save_config({'recurrence': 'action', 'ab_pairs': [[1, 1]]}) is None
    assert task_from_save_config({'recurrence': 'dynamic_mixed', 'ab_pairs': [[1, 1]]}) is None
    assert task_from_save_config({'p': 23}) == 'addition'
    # 新块优先：scenario 与旧字段不一致时以 scenario 为准
    assert task_from_save_config({'scenario': 'single', 'order': 3,
                                  'rules': [[1, 2, 3]],
                                  'recurrence': 'addition'}) == 'single'
    assert task_from_save_config({'scenario': 'mixed', 'order': 2,
                                  'rules': [[1, 1]], 'ab_pairs': [[1, 1]]}) is None
    assert task_from_save_config({'scenario': 'action', 'order': 2,
                                  'rules': [[2, 1]]}) is None


if __name__ == '__main__':
    test_order_property()
    test_name_auto_generated()
    test_next_fn_matches_legacy_order2()
    test_next_fn_matches_legacy_order3()
    test_rules_from_config_order2()
    test_rules_from_config_any_order()
    test_rules_from_config_validation()
    test_single_rule_from_task()
    test_single_rule_from_task_single_scenario()
    test_save_config_round_trip()
    print("ALL TESTS PASSED: test_rules.py")
