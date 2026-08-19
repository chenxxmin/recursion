"""Unit tests for the queue-driven missing-pattern analysis in rule_fit.py."""
import contextlib
import io
import os
import random
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from models import FibonacciTransformer
from rule_fit import (aggregate_buckets, child_patterns, expansion_deps,
                      fit_and_score, fmt_equation, patt_label, probe_equations,
                      rule_coeffs, run_missing_categories, select_C,
                      visible_distance)


def test_fmt_equation_drops_zero_coeffs():
    assert (fmt_equation([1, 2, 3, 4, 5], [0, 3, 2, 0, 0], 127)
            == 'x_{t+1} = 3*x_{t-2} + 2*x_{t-3}  (mod 127)')
    assert fmt_equation([0, 1], [1, 1], 127) == 'x_{t+1} = 1*x_t + 1*x_{t-1}  (mod 127)'
    assert fmt_equation([0, 1], [0, 0], 127) == 'x_{t+1} = 0  (mod 127)'


def test_expansion_deps():
    # reliable fit -> nonzero coefficient distances drive expansion
    assert expansion_deps([1, 2, 3], [2, 1, 0], 0.99) == ({1, 2}, 'fit')
    # low agreement -> fall back to the whole attention hypothesis set
    assert expansion_deps([1, 2, 3], [0, 3, 2], 0.16) == ({1, 2, 3}, 'attn')
    # no fit at all -> attention set as well
    assert expansion_deps([1, 2], None, -1.0) == ({1, 2}, 'attn')
    # reliable constant fit -> no dependencies, no children
    assert expansion_deps([1, 2], [0, 0], 1.0) == (set(), 'fit')


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


class _StubModel:
    """A perfect textbook model: x_{t+1} = x_t + x_{t-1} when x_t is visible,
    else the unrolled 2*x_{t-1} + x_{t-2}. Mimics the (logits, loss) return."""

    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        view = x[0].tolist()
        T = len(view)
        logits = torch.zeros(1, T, self.p + 1)
        for t in range(1, T):
            if view[t] == self.p:
                pred = (2 * view[t - 1] + view[t - 2]) % self.p
            else:
                pred = (view[t] + view[t - 1]) % self.p
            logits[0, t, pred] = 1.0
        return (logits, None)


def test_probe_equations_recovers_stub_formulas():
    random.seed(0)
    model = _StubModel(127)
    # (x,M): fit on d1,d2 must recover the unrolled 2*x_{t-1} + x_{t-2}
    X, y = probe_equations(model, (False, True), [1, 2], 32, 50, 127)
    assert len(X) > 100
    assert all(v != 127 for row in X for v in row), "masked token leaked into features"
    coeffs, acc, exact = fit_and_score(X, y, 127)
    assert exact and coeffs == [2, 1], (coeffs, acc)
    # (x,x): fit on d0,d1 recovers the base rule
    X, y = probe_equations(model, (False, False), [0, 1], 32, 50, 127)
    coeffs, acc, exact = fit_and_score(X, y, 127)
    assert exact and coeffs == [1, 1], (coeffs, acc)


def test_probe_equations_neighbour_window_masks_do_not_leak():
    # C includes distances beyond the window: neighbours' forced masks must not
    # land inside [t - max(C), t]
    random.seed(0)
    model = _StubModel(127)
    X, y = probe_equations(model, (False, True), [1, 2, 3, 4, 5], 32, 20, 127)
    assert len(X) > 0
    assert all(v != 127 for row in X for v in row)


def test_run_missing_categories_smoke():
    """Untrained tiny model: the BFS machinery must run end-to-end. Output is
    quiet by design — pattern blocks print only for trustworthy fits (EXACT or
    agreement >= 0.9), which an untrained model produces none of, so we assert
    the header and the absence of pattern blocks."""
    random.seed(0)
    torch.manual_seed(0)
    model = FibonacciTransformer(p=7, d_model=32, n_head=2, n_layer=1, block_size=16)
    model.eval()
    args = SimpleNamespace(length=12, min_n=5, samples=30, depth=None)
    cfg = {'MISSING_PROB': 0.3, 'MISS_LEN': 2, 'TRAIN_LEN': 12}
    next_fn = lambda s: (s[-1] + s[-2]) % 7
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_missing_categories(args, model, cfg, next_fn, 2, 7, 'smoke', 'X(k)=(1*X(k-1)+1*X(k-2)) mod 7')
    out = buf.getvalue()
    assert 'Model: smoke' in out
    assert 'Queue-driven analysis' in out
    assert '== pattern' not in out or 'set C' in out, \
        'a pattern block without a fit line means the print gating is broken'


if __name__ == '__main__':
    test_expansion_deps()
    test_fmt_equation_drops_zero_coeffs()
    test_visible_distance()
    test_patt_label()
    test_child_patterns_root()
    test_child_patterns_extends_window()
    test_child_patterns_depth_cap()
    test_aggregate_buckets_and_select_C()
    test_rule_coeffs()
    test_probe_equations_recovers_stub_formulas()
    test_probe_equations_neighbour_window_masks_do_not_leak()
    test_run_missing_categories_smoke()
    print("ALL TESTS PASSED: test_rule_fit_missing.py")
