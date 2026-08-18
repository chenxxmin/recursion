"""Unit tests for the queue-driven missing-pattern analysis in rule_fit.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from rule_fit import (aggregate_buckets, child_patterns, patt_label,
                      rule_coeffs, select_C, visible_distance)


def test_visible_distance():
    P = (False, True)  # (x,M): d0 masked, d1 visible
    assert not visible_distance(P, 0)
    assert visible_distance(P, 1)
    assert visible_distance(P, 5)  # beyond the window: always visible in probes


def test_patt_label():
    assert patt_label((False, True)) == '(x,M)'
    assert patt_label((True, False, True)) == '(M,x,M)'


def test_child_patterns_root():
    # (x,x) with deps {0,1} -> mask any non-empty subset
    ch = set(child_patterns((False, False), {0, 1}, 5))
    assert ch == {(False, True), (True, False), (True, True)}, ch


def test_child_patterns_extends_window():
    # (x,M) with deps {1,2}: d1 -> (M,M); d2 outside window -> extend to (M,x,M);
    # both -> (M,M,M)
    ch = set(child_patterns((False, True), {1, 2}, 5))
    assert ch == {(True, True), (True, False, True), (True, True, True)}, ch


def test_child_patterns_depth_cap():
    # dep at d5 needs a length-6 pattern -> dropped when d_max=5
    assert child_patterns((False, True), {5}, 5) == []
    # dep at d4 fits exactly in length 5
    ch = child_patterns((False, True), {4}, 5)
    assert ch == [(True, False, False, False, True)], ch


def test_aggregate_buckets_and_select_C():
    buckets = {
        (False, False, False, False, True): {'n': 100, 'agree': 99,
            'attn': {(0, 0): {0: [90.0, 100], 1: [80.0, 100], 2: [5.0, 100]}}},
        (True, False, False, False, True): {'n': 50, 'agree': 40,
            'attn': {(0, 0): {1: [45.0, 50], 3: [20.0, 50]}}},
        (False, False, False, True, False): {'n': 70, 'agree': 70,
            'attn': {(0, 0): {0: [70.0, 70]}}},
    }
    n, agree, attn = aggregate_buckets(buckets, (False, True), 5)
    assert (n, agree) == (150, 139)
    # merged means: d0 = 90/150 = 0.60 (but masked in (x,M) -> excluded),
    # d1 = 125/150 >= 0.10, d2 = 5/150 < 0.10, d3 = 20/150 = 0.133 >= 0.10
    assert select_C(attn, (False, True)) == [1, 3]
    # suffix match works for full-depth pattern too
    n2, agree2, _ = aggregate_buckets(buckets, (False, False, False, True, False), 5)
    assert (n2, agree2) == (70, 70)


def test_rule_coeffs():
    next_fn = lambda s: (s[-1] + 2 * s[-2]) % 127
    coeffs, base = rule_coeffs(next_fn, 2, 127)
    assert coeffs == [1, 2] and base == 0
    next_fn3 = lambda s: (3 * s[-1] + s[-2] + 2 * s[-3]) % 127
    coeffs3, _ = rule_coeffs(next_fn3, 3, 127)
    assert coeffs3 == [3, 1, 2]


if __name__ == '__main__':
    test_visible_distance()
    test_patt_label()
    test_child_patterns_root()
    test_child_patterns_extends_window()
    test_child_patterns_depth_cap()
    test_aggregate_buckets_and_select_C()
    test_rule_coeffs()
    print("ALL TESTS PASSED: test_rule_fit_missing.py")
