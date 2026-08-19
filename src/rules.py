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


def single_rule_from_task(task, cfg):
    """Resolve a single-rule task to (init_len, next_fn, recurrence_name).

    task: 'addition' | 'multiplication' | 'tribonacci' | 'nonlinear' | 'nonlinear_mul'.
    cfg accepts merged-config keys (P, A/B/C) or checkpoint keys (p, a/b/c);
    missing coefficients default to 1. The returned next_fn takes (seq, p),
    matching RecurrenceDataset's recurrence_fn contract.
    """
    p = cfg.get('P', cfg.get('p'))
    get = lambda k: cfg.get(k.upper(), cfg.get(k.lower(), 1))
    if task == 'addition':
        a, b = get('a'), get('b')
        name = f"X(k)=({a}*X(k-1)+{b}*X(k-2)) mod {p}"
        return 2, (lambda seq, p: (a * seq[-1] + b * seq[-2]) % p), name
    if task == 'multiplication':
        return 2, (lambda seq, p: (seq[-1] * seq[-2]) % p), f"X(k)=(X(k-1)*X(k-2)) mod {p}"
    if task == 'tribonacci':
        a, b, c = get('a'), get('b'), get('c')
        name = f"X(k)=({a}*X(k-1)+{b}*X(k-2)+{c}*X(k-3)) mod {p}"
        return 3, (lambda seq, p: (a * seq[-1] + b * seq[-2] + c * seq[-3]) % p), name
    if task == 'nonlinear':
        # The state map (x,y) -> (y, y^2+x) is bijective for any p
        # (invert: x = z - y^2), so all states lie on pure cycles.
        return 2, (lambda seq, p: (seq[-1] * seq[-1] + seq[-2]) % p), f"X(k)=(X(k-1)^2+X(k-2)) mod {p}"
    if task == 'nonlinear_mul':
        # The state map (x,y) -> (y, x*y^2) is not bijective (any state with
        # y=0 flows into the (0,0) fixed point), so transient trajectories
        # exist; RecurrenceDataset handles them (truncated trajectories get
        # step-by-step extension in run(), see generate_cycle's comment).
        return 2, (lambda seq, p: (seq[-2] * seq[-1] * seq[-1]) % p), f"X(k)=(X(k-2)*X(k-1)^2) mod {p}"
    raise ValueError(f"unknown single-rule task: {task}")


def save_config_extra(task, cfg):
    """Checkpoint save_config entries for a single-rule task (the checkpoint
    vocabulary: lowercase a/b/c and the 'recurrence' name)."""
    get = lambda k: cfg.get(k.upper(), cfg.get(k.lower(), 1))
    if task == 'addition':
        return {'a': get('a'), 'b': get('b'), 'recurrence': 'addition'}
    if task == 'multiplication':
        return {'recurrence': 'multiplicative'}
    if task == 'tribonacci':
        return {'a': get('a'), 'b': get('b'), 'c': get('c'), 'recurrence': 'tribonacci'}
    if task == 'nonlinear':
        return {'recurrence': 'nonlinear'}
    if task == 'nonlinear_mul':
        return {'recurrence': 'nonlinear_mul'}
    raise ValueError(f"unknown single-rule task: {task}")


def task_from_save_config(config):
    """Map a checkpoint save_config to its single-rule task name.

    Returns None for mixed_ab/mixed_abc/action checkpoints (those save
    ab_pairs instead of a single recurrence spec). The old name
    'dynamic_mixed' (pre-rename checkpoints) is accepted for compatibility.
    Defaults to 'addition' for old a/b-only or minimal configs.
    """
    recurrence = config.get('recurrence')
    if recurrence in ('action', 'dynamic_mixed'):
        return None
    if recurrence is None and 'ab_pairs' in config:
        return None  # mixed_ab/mixed_abc
    if 'c' in config or recurrence == 'tribonacci':
        return 'tribonacci'
    if recurrence == 'multiplicative':
        return 'multiplication'
    if recurrence == 'nonlinear':
        return 'nonlinear'
    if recurrence == 'nonlinear_mul':
        return 'nonlinear_mul'
    return 'addition'


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
        if any(not 0 <= c < p for c in coeffs):
            raise ValueError(f"rule {entry} has a coefficient outside [0, {p})")
        rules.append(LinearRecurrenceRule(coeffs=coeffs, p=p))
    return rules
