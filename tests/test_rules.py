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
    cfg = {'P': 23, 'AB_PAIRS': [[3, 5], [7, 11]]}
    rules = rules_from_config(cfg, 2)
    assert [r.coeffs for r in rules] == [(3, 5), (7, 11)]
    assert all(r.p == 23 for r in rules)
    assert rules[0].name == "a3b5"


def test_rules_from_config_order3():
    cfg = {'P': 23, 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]]}
    rules = rules_from_config(cfg, 3)
    assert [r.coeffs for r in rules] == [(1, 1, 1), (1, 2, 3)]


def test_rules_from_config_validation():
    for bad_cfg, order in [
        ({'P': 23}, 2),                       # missing AB_PAIRS
        ({'P': 23, 'AB_PAIRS': []}, 2),       # empty rule list
        ({'P': 23, 'AB_PAIRS': [[1, 1, 1]]}, 2),  # wrong coefficient count
    ]:
        try:
            rules_from_config(bad_cfg, order)
            assert False, f"expected ValueError for {bad_cfg}"
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


def test_save_config_round_trip():
    """save_config_extra (writer) and task_from_save_config (reader) must agree,
    including the multiplication/multiplicative vocabulary split."""
    for task in ('addition', 'multiplication', 'tribonacci', 'tetranacci', 'nonlinear', 'nonlinear_mul'):
        cfg = {'P': 23, 'A': 2, 'B': 3, 'C': 4, 'D': 5}
        extra = save_config_extra(task, cfg)
        assert task_from_save_config(extra) == task, (task, extra)
    # mixed / action checkpoints are not single-rule (old name kept for compat)
    assert task_from_save_config({'ab_pairs': [[1, 1]], 'order': 2}) is None
    assert task_from_save_config({'recurrence': 'action', 'ab_pairs': [[1, 1]]}) is None
    assert task_from_save_config({'recurrence': 'dynamic_mixed', 'ab_pairs': [[1, 1]]}) is None
    # old minimal config defaults to addition
    assert task_from_save_config({'p': 23}) == 'addition'


if __name__ == '__main__':
    test_order_property()
    test_name_auto_generated()
    test_next_fn_matches_legacy_order2()
    test_next_fn_matches_legacy_order3()
    test_rules_from_config_order2()
    test_rules_from_config_order3()
    test_rules_from_config_validation()
    test_single_rule_from_task()
    test_save_config_round_trip()
    print("ALL TESTS PASSED: test_rules.py")
