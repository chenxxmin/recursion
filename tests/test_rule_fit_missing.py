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
from rule_fit import (aggregate_buckets, fit_and_score, fmt_equation,
                      mask_children, patt_label, position_influence,
                      probe_attention, probe_equations, refine_children,
                      rule_coeffs, run_missing_categories, select_C,
                      visible_distance)


def test_fmt_equation_drops_zero_coeffs():
    assert (fmt_equation([1, 2, 3, 4, 5], [0, 3, 2, 0, 0], 127)
            == 'x_{t+1} = 3*x_{t-2} + 2*x_{t-3}  (mod 127)')
    assert fmt_equation([0, 1], [1, 1], 127) == 'x_{t+1} = 1*x_t + 1*x_{t-1}  (mod 127)'
    assert fmt_equation([0, 1], [0, 0], 127) == 'x_{t+1} = 0  (mod 127)'


def test_mask_children():
    # (x,x) with deps {0,1}: one child per used position
    assert set(mask_children((False, False), {0, 1}, 8)) == {(False, True), (True, False)}
    # (x,M) dep d1 -> (M,M); dep d2 outside window -> extend to (M,x,M)
    assert set(mask_children((False, True), {1, 2}, 8)) == {(True, True), (True, False, True)}
    # depth cap: d8 needs a length-9 child -> dropped; d7 fits exactly in 8
    assert mask_children((False, True), {8}, 8) == []
    assert mask_children((False, True), {7}, 8) == [(True, False, False, False, False, False, False, True)]
    # no deps -> no children
    assert mask_children((False, False), set(), 8) == []


def test_refine_children():
    # both states of the next-deeper position, prepended
    assert refine_children((False, True), 8) == [(False, False, True), (True, False, True)]
    # at the depth cap -> nothing
    assert refine_children((False,) * 8, 8) == []


def test_visible_distance():
    P = (False, True)  # (x,M): d0 masked, d1 visible
    assert not visible_distance(P, 0)
    assert visible_distance(P, 1)
    assert visible_distance(P, 5)  # beyond the window: always visible in probes


def test_patt_label():
    assert patt_label((False, True)) == '(x,M)'
    assert patt_label((True, False, True)) == '(M,x,M)'


def test_aggregate_buckets():
    buckets = {
        (False, False, False, False, True): {'n': 100, 'agree': 99},
        (True, False, False, False, True): {'n': 50, 'agree': 40},
        (False, False, False, True, False): {'n': 70, 'agree': 70},
    }
    # suffix merge over the two buckets ending (x,M)
    assert aggregate_buckets(buckets, (False, True), 5) == (150, 139)
    # full-depth pattern matches only its own bucket
    assert aggregate_buckets(buckets, (False, False, False, True, False), 5) == (70, 70)


def test_select_C():
    attn = {(0, 0): {0: [6.0, 10], 1: [8.0, 10], 2: [0.5, 10], 3: [1.5, 10]},
            (0, 1): {4: [0.9, 10]}}
    # (x,M): d0 masked -> excluded; d2 below 0.10 -> excluded; d4 below -> excluded
    assert select_C(attn, (False, True)) == [1, 3]
    # (x,x): d0 visible now
    assert select_C(attn, (False, False)) == [0, 1, 3]


def test_probe_attention_structure():
    random.seed(0)
    torch.manual_seed(0)
    model = FibonacciTransformer(p=7, d_model=32, n_head=2, n_layer=1, block_size=16)
    model.eval()
    attn = probe_attention(model, (False, True), 12, 10, 7)
    assert set(attn) == {(0, 0), (0, 1)}
    for dd in attn.values():
        # window forced at t = length-2 = 10 -> distances 0..10
        assert max(dd) == 10
        for d, (s, c) in dd.items():
            assert c == 10 and 0.0 <= s <= 10.0


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


def test_position_influence_stub():
    random.seed(0)
    model = _StubModel(127)
    # (x,M): stub computes 2*x_{t-1} + x_{t-2} -> d1,d2 always matter, d3 never
    inf = position_influence(model, (False, True), [1, 2, 3], 32, 50, 127)
    assert inf[1] == 1.0 and inf[2] == 1.0 and inf[3] == 0.0, inf
    # (x,x): stub computes x_t + x_{t-1} -> d0,d1 always matter, d2 never
    inf = position_influence(model, (False, False), [0, 1, 2], 32, 50, 127)
    assert inf[0] == 1.0 and inf[1] == 1.0 and inf[2] == 0.0, inf


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
    """Untrained tiny model: the BFS machinery must run end-to-end. Pattern
    blocks print only when there is something trustworthy (a good fit or
    significant influence); assert the headers and that no bare header prints
    without a content line."""
    random.seed(0)
    torch.manual_seed(0)
    model = FibonacciTransformer(p=7, d_model=32, n_head=2, n_layer=1, block_size=16)
    model.eval()
    args = SimpleNamespace(length=12, min_n=5, samples=30, depth=None, influence=False)
    cfg = {'MISSING_PROB': 0.3, 'MISS_LEN': 2, 'TRAIN_LEN': 12}
    next_fn = lambda s: (s[-1] + s[-2]) % 7
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_missing_categories(args, model, cfg, next_fn, 2, 7, 'smoke', 'X(k)=(1*X(k-1)+1*X(k-2)) mod 7')
    out = buf.getvalue()
    assert 'Model: smoke' in out
    assert 'Queue-driven analysis' in out
    assert '== pattern' not in out or ('set C' in out or 'influence' in out), \
        'a pattern block without a fit/influence line means the print gating is broken'


if __name__ == '__main__':
    test_mask_children()
    test_refine_children()
    test_fmt_equation_drops_zero_coeffs()
    test_visible_distance()
    test_patt_label()
    test_aggregate_buckets()
    test_select_C()
    test_probe_attention_structure()
    test_rule_coeffs()
    test_position_influence_stub()
    test_probe_equations_recovers_stub_formulas()
    test_probe_equations_neighbour_window_masks_do_not_leak()
    test_run_missing_categories_smoke()
    print("ALL TESTS PASSED: test_rule_fit_missing.py")
