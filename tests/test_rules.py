"""Unit tests for src/rules.py (pure asserts; also pytest-compatible)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from rules import LinearRecurrenceRule, rules_from_config


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


if __name__ == '__main__':
    test_order_property()
    test_name_auto_generated()
    test_next_fn_matches_legacy_order2()
    test_next_fn_matches_legacy_order3()
    test_rules_from_config_order2()
    test_rules_from_config_order3()
    test_rules_from_config_validation()
    print("ALL TESTS PASSED: test_rules.py")
