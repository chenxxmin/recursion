"""GF(2^n) representation + recurrence generators (no torch)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from finite_field import (
    F128, GF2n, generate_recurrence_sequence, generate_action_sequence_gf,
    _clmul, _reduce, _IRR_X7_X_1,
)
from rules import LinearRecurrenceRule, single_rule_from_task, rules_from_config


def test_f128_basics():
    assert F128.n == 7 and F128.order == 128
    assert F128.add(0b1010, 0b1100) == 0b0110
    assert F128.sub(5, 5) == 0
    assert F128.mul(0, 99) == 0
    assert F128.mul(1, 99) == 99
    # x * x = x^2 for the indeterminate x (=2)
    assert F128.mul(2, 2) == 4


def test_mul_matches_poly_reduce():
    for a in (0, 1, 2, 3, 7, 16, 64, 127):
        for b in (0, 1, 2, 5, 17, 100, 127):
            expect = _reduce(_clmul(a, b), _IRR_X7_X_1, 7)
            assert F128.mul(a, b) == expect


def test_field_axioms_sample():
    # Additive abelian (XOR), multiplicative identity, distributivity spot-check.
    rng = random.Random(0)
    for _ in range(200):
        a, b, c = (rng.randrange(128) for _ in range(3))
        assert F128.add(a, b) == F128.add(b, a)
        assert F128.add(F128.add(a, b), c) == F128.add(a, F128.add(b, c))
        assert F128.mul(F128.add(a, b), c) == F128.add(F128.mul(a, c), F128.mul(b, c))
        assert F128.mul(1, a) == a


def test_not_integer_mod_128():
    # Classic counterexample: in Z/128Z, 64+64=0 and 2*64=128≡0, but field
    # add is XOR so 64+64=0 still; mul differs: 3*5 = 15 in Z, but in GF(2^7)
    # 0b11 * 0b101 = carryless 0b1111 = 15 — pick a case that diverges.
    # 7 * 9: integer 63; carryless: 0b111 * 0b1001 = 0b111111 = 63 — still same.
    # Use 6*6: int 36; clmul(6,6)=0b0110*0b0110 -> 0b010100 = 20.
    assert (6 * 6) % 128 == 36
    assert F128.mul(6, 6) == 20
    assert F128.mul(6, 6) != (6 * 6) % 128


def test_x_is_primitive():
    """Irreducible x^7+x+1 is primitive: powers of x (=2) hit all nonzero elts."""
    seen = set()
    g = 1
    for _ in range(127):
        g = F128.mul(g, 2)
        seen.add(g)
    assert 0 not in seen and len(seen) == 127


def test_linear_combo_fibonacci_like():
    # x_k = x_{k-1} + x_{k-2} over F128 (coeffs canonical: newest first)
    seq = generate_recurrence_sequence(F128, (1, 1), init=(1, 2), length=6)
    # 1, 2, 1⊕2=3, 2⊕3=1, 3⊕1=2, 1⊕2=3
    assert seq == [1, 2, 3, 1, 2, 3]


def test_rule_next_fn_uses_field():
    rule = LinearRecurrenceRule(coeffs=(1, 1), p=128, field=F128)
    fn = rule.next_fn()
    assert fn([1, 2], 128) == 3
    assert fn([2, 3], 128) == 1
    # Must differ from mod-128 integer arithmetic when mul involved
    rule2 = LinearRecurrenceRule(coeffs=(6, 6), p=128, field=F128)
    assert rule2.next_fn()([1, 1], 128) == F128.add(F128.mul(6, 1), F128.mul(6, 1))
    assert rule2.next_fn()([1, 1], 128) != (6 * 1 + 6 * 1) % 128


def test_rule_rejects_p_mismatch():
    try:
        LinearRecurrenceRule(coeffs=(1, 1), p=127, field=F128)
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_single_rule_field_cfg():
    init_len, fn, name = single_rule_from_task(
        'single', {'P': 128, 'RULES': [[1, 1]], 'FIELD': 'gf2n'})
    assert init_len == 2
    assert 'GF(2^7)' in name
    # Human-facing name uses polynomials, not bare ints for coeffs.
    assert '(1)*X(k-1)' in name and '(1)*X(k-2)' in name
    assert 'irr=' in name
    assert fn([1, 2], 128) == 3


def test_poly_str():
    from finite_field import poly_str
    assert poly_str(0) == '0'
    assert poly_str(1) == '1'
    assert poly_str(2) == 'x'
    assert poly_str(5) == 'x^2 + 1'
    assert poly_str(70) == 'x^6 + x^2 + x'
    assert F128.format(124) == 'x^6 + x^5 + x^4 + x^3 + x^2'


def test_rules_from_config_field():
    rules = rules_from_config([[1, 1], [2, 3]], 128, field=F128)
    assert rules[0].next_fn()([1, 2], 128) == 3
    # 2*2 + 3*1 over F128: mul(2,2)=4, mul(3,1)=3, 4⊕3=7
    assert rules[1].next_fn()([1, 2], 128) == 7


def test_action_generator_layout():
    rng = random.Random(0)
    seq, flag0 = generate_action_sequence_gf(
        F128, rules=[[1, 1], [2, 3]], length=5, rng=rng)
    assert flag0 == 129  # q+1, missing slot q=128 reserved
    # length=5 values => init 2 + 3 generated => 2 + 3*(flag+val) = 8 tokens
    assert len(seq) == 2 + 2 * 3
    values = seq[:2]
    i = 2
    while i < len(seq):
        flag = seq[i]
        val = seq[i + 1]
        assert flag >= flag0
        rule = [[1, 1], [2, 3]][flag - flag0]
        expect = F128.linear_combo(rule, values[-2:])
        assert val == expect
        values.append(val)
        i += 2


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'OK {name}')
    print('All finite_field tests passed.')
