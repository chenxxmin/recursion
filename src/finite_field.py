"""Binary extension fields GF(2^n).

Elements are integers in ``[0, 2^n)``: bit ``i`` is the coefficient of ``x^i``.
Addition is XOR. Multiplication is polynomial multiply reduced by a fixed
irreducible of degree ``n``.

This is **not** integer mod ``2^n``. In particular GF(2^7) (order 128) is not
``Z/128Z``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Degree-7 irreducible x^7 + x + 1 (0x83). Primitive over GF(2): the element
# x (= 0b10) generates the multiplicative group of order 127.
_IRR_X7_X_1 = 0b10000011  # x^7 + x + 1


def _clmul(a: int, b: int) -> int:
    """Carry-less product of two polynomials over GF(2)."""
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def _reduce(poly: int, irr: int, n: int) -> int:
    """Reduce ``poly`` modulo ``irr`` (degree n) into ``[0, 2^n)``."""
    while poly.bit_length() > n:
        shift = poly.bit_length() - 1 - n
        poly ^= irr << shift
    return poly


@dataclass(frozen=True)
class GF2n:
    """GF(2^n) with integer element encoding.

    Parameters
    ----------
    n:
        Extension degree. Field order is ``q = 2^n``.
    irreducible:
        Monic irreducible of degree ``n``, as an integer bit-polynomial.
        Default for ``n=7`` is ``x^7+x+1``.
    """

    n: int
    irreducible: int = field(default=_IRR_X7_X_1)
    _mul_table: tuple[tuple[int, ...], ...] | None = field(
        default=None, repr=False, compare=False, hash=False
    )

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.irreducible.bit_length() - 1 != self.n:
            raise ValueError(
                f"irreducible must have degree {self.n}, "
                f"got bit_length-1={self.irreducible.bit_length() - 1}"
            )
        # Precompute full mul table: tiny for n=7 (128^2), worth it for generators.
        q = self.order
        table = []
        for a in range(q):
            row = []
            for b in range(q):
                row.append(_reduce(_clmul(a, b), self.irreducible, self.n))
            table.append(tuple(row))
        object.__setattr__(self, '_mul_table', tuple(table))

    @property
    def order(self) -> int:
        """Cardinality ``q = 2^n`` (number of field elements)."""
        return 1 << self.n

    def check(self, x: int) -> int:
        x = int(x)
        if not 0 <= x < self.order:
            raise ValueError(f"element {x} outside GF(2^{self.n}) = [0, {self.order})")
        return x

    def add(self, a: int, b: int) -> int:
        return self.check(a) ^ self.check(b)

    def sub(self, a: int, b: int) -> int:
        return self.add(a, b)  # char 2

    def neg(self, a: int) -> int:
        return self.check(a)

    def mul(self, a: int, b: int) -> int:
        return self._mul_table[self.check(a)][self.check(b)]

    def mac(self, acc: int, c: int, x: int) -> int:
        """``acc + c*x`` in the field."""
        return self.add(acc, self.mul(c, x))

    def dot(self, coeffs, values) -> int:
        """``sum_i coeffs[i] * values[i]`` in the field (same length)."""
        if len(coeffs) != len(values):
            raise ValueError(
                f"dot length mismatch: {len(coeffs)} coeffs vs {len(values)} values")
        acc = 0
        for c, v in zip(coeffs, values):
            acc = self.mac(acc, c, v)
        return acc

    def linear_combo(self, coeffs, values, const: int = 0) -> int:
        """``const + sum_i coeffs[i] * values[i]`` (canonical newest-first values)."""
        return self.add(self.check(const), self.dot(coeffs, values))

    def format(self, e: int, var: str = "x") -> str:
        """Human-readable bit-polynomial for a field element."""
        return poly_str(self.check(e), var=var)


def poly_str(e: int, var: str = "x") -> str:
    """Human-readable bit-polynomial, e.g. ``5 → 'x^2 + 1'``.

    Code may store elements as ints; use this when showing sequences to humans.
    """
    e = int(e)
    if e == 0:
        return "0"
    if e < 0:
        raise ValueError(f"field element must be >= 0, got {e}")
    terms = []
    for i in range(e.bit_length() - 1, -1, -1):
        if (e >> i) & 1:
            if i == 0:
                terms.append("1")
            elif i == 1:
                terms.append(var)
            else:
                terms.append(f"{var}^{i}")
    return " + ".join(terms)


# Canonical F_128 used by Recurrence GF experiments.
F128 = GF2n(n=7, irreducible=_IRR_X7_X_1)


def generate_recurrence_sequence(
    field: GF2n,
    coeffs,
    init,
    length: int,
    *,
    const: int = 0,
    coeff_order: str = 'canonical',
):
    """Generate a length-``length`` linear recurrence over ``field``.

    Parameters
    ----------
    coeffs:
        Coefficient tuple. With ``coeff_order='canonical'`` (default, matches
        ``LinearRecurrenceRule``): ``c[0]`` multiplies the newest value.
        With ``coeff_order='action'``: ``c[0]`` multiplies the oldest value
        (action-family layout).
    init:
        Initial state, length = order = ``len(coeffs)``.
    length:
        Total number of field values in the returned sequence (includes init).
    const:
        Constant term (default 0).
    """
    coeffs = tuple(int(c) for c in coeffs)
    order = len(coeffs)
    if order < 1:
        raise ValueError("need at least one coefficient")
    if length < order:
        raise ValueError(f"length {length} < order {order}")
    seq = [field.check(x) for x in init]
    if len(seq) != order:
        raise ValueError(f"init length {len(seq)} != order {order}")
    field.check(const)

    for _ in range(order, length):
        window = seq[-order:]
        if coeff_order == 'canonical':
            # c1 * newest + c2 * next + ...
            values = list(reversed(window))
            paired = coeffs
        elif coeff_order == 'action':
            # c1 * oldest + ... + cn * newest
            values = window
            paired = coeffs
        else:
            raise ValueError(f"unknown coeff_order: {coeff_order!r}")
        seq.append(field.linear_combo(paired, values, const=const))
    return seq


def generate_action_sequence_gf(
    field: GF2n,
    rules,
    length: int,
    rng,
    *,
    order: int | None = None,
):
    """One action-style sample over a binary field (flags interleaved).

    Layout matches ``datasets.generate_action_sample``:
    ``[x0..x_{o-1}, flag, x_o, flag, x_{o+1}, ...]`` with action coefficient
    convention (first coeff × oldest). Returns ``(seq, flag_start_id)`` where
    value tokens are in ``[0, q)`` and flags are ``flag_start_id + rule_idx``.
    ``flag_start_id = q + 1`` mirrors the usual missing-token = ``q`` slot.
    """
    rules = [tuple(int(c) for c in r) for r in rules]
    if not rules:
        raise ValueError("need at least one rule")
    o = order if order is not None else len(rules[0])
    if o < 1 or any(len(r) != o for r in rules):
        raise ValueError("all rules must share the same order")
    q = field.order
    flag_start_id = q + 1  # leave q free for a missing/pad token
    values = [rng.randrange(q) for _ in range(o)]
    seq = list(values)
    for _ in range(o, length):
        rule_idx = rng.randrange(len(rules))
        x_next = field.linear_combo(rules[rule_idx], values[-o:], const=0)
        # action convention already: coeffs zip with oldest..newest = values[-o:]
        seq.append(flag_start_id + rule_idx)
        seq.append(x_next)
        values.append(x_next)
    return seq, flag_start_id
