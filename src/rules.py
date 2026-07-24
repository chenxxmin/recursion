"""Linear recurrence rules over Z/pZ.

A rule X(k) = (c1*X(k-1) + c2*X(k-2) + ... + cn*X(k-n)) mod p is described by
its coefficient tuple (c1, ..., cn); len(coeffs) is the recurrence order.
Shared by the mixed-rule tasks (mixed_ab / mixed_abc).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LinearRecurrenceRule:
    coeffs: tuple[int, ...]  # X(k) = (c1*X(k-1) + c2*X(k-2) + ...) mod p
    p: int
    name: str = ""           # e.g. "a1b2c4"; auto-generated when empty

    def __post_init__(self):
        # frozen dataclass: normalize via object.__setattr__
        object.__setattr__(self, 'coeffs', tuple(int(c) for c in self.coeffs))
        if not self.name:
            labels = 'abcdefgh'
            auto = ''.join(f"{labels[i]}{c}" for i, c in enumerate(self.coeffs))
            object.__setattr__(self, 'name', auto)

    @property
    def order(self) -> int:
        return len(self.coeffs)

    def next_fn(self):
        """Return a closure fn(seq, p) compatible with RecurrenceDataset.

        Numerically identical to the legacy inline closures:
        (a, b)    -> (a*seq[-1] + b*seq[-2]) % p
        (a, b, c) -> (a*seq[-1] + b*seq[-2] + c*seq[-3]) % p
        zip(coeffs, reversed(seq)) pairs c1 with seq[-1], c2 with seq[-2], ...
        """
        coeffs = self.coeffs

        def fn(seq, p):
            return sum(c * s for c, s in zip(coeffs, reversed(seq))) % p

        return fn


def rules_from_config(cfg, order):
    """Build the rule list for a mixed task from a merged config dict.

    order=2 reads AB_PAIRS, order=3 reads ABC_PAIRS; p comes from cfg['P'].
    Raises ValueError if the key is missing/empty or a rule's coefficient
    count differs from `order`.
    """
    key = {2: 'AB_PAIRS', 3: 'ABC_PAIRS'}.get(order)
    if key is None:
        raise ValueError(f"rules_from_config supports order 2 or 3, got {order}")
    raw = cfg.get(key)
    if not raw:
        raise ValueError(f"config key {key} must contain at least one rule")
    p = cfg['P']
    rules = []
    for entry in raw:
        coeffs = tuple(int(c) for c in entry)
        if len(coeffs) != order:
            raise ValueError(
                f"rule {entry} has {len(coeffs)} coefficients, expected {order}")
        rules.append(LinearRecurrenceRule(coeffs=coeffs, p=p))
    return rules
