"""Linear recurrence rules over Z/pZ.

A rule X(k) = (c1*X(k-1) + c2*X(k-2) + ... + cn*X(k-n)) mod p is described by
its coefficient tuple (c1, ..., cn); len(coeffs) is the recurrence order.
Shared by the mixed-rule tasks (mixed_ab / mixed_abc).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LinearRecurrenceRule:
    coeffs: tuple[int, ...]  # X(k) = (c1*X(k-1) + c2*X(k-2) + ... + const) mod p
    p: int
    name: str = ""           # e.g. "a1b2c4"; auto-generated when empty
    const: int = 0           # optional constant term (2026-10-14)

    def __post_init__(self):
        # frozen dataclass: normalize via object.__setattr__
        object.__setattr__(self, 'coeffs', tuple(int(c) for c in self.coeffs))
        object.__setattr__(self, 'const', int(self.const))
        if not self.name:
            labels = 'abcdefgh'
            auto = ''.join(f"{labels[i]}{c}" for i, c in enumerate(self.coeffs))
            if self.const:
                auto += f"+{self.const}"
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
        const = self.const

        def fn(seq, p):
            return (sum(c * s for c, s in zip(coeffs, reversed(seq))) + const) % p

        return fn


def single_rule_from_task(task, cfg):
    """Resolve a single-rule task to (init_len, next_fn, recurrence_name).

    task: 'addition' | 'multiplication' | 'tribonacci' | 'tetranacci' | 'nonlinear' | 'nonlinear_mul'.
    cfg accepts merged-config keys (P, A/B/C/D) or checkpoint keys (p, a/b/c/d);
    missing coefficients default to 1. The returned next_fn takes (seq, p),
    matching RecurrenceDataset's recurrence_fn contract.
    """
    p = cfg.get('P', cfg.get('p'))
    get = lambda k: cfg.get(k.upper(), cfg.get(k.lower(), 1))
    if task == 'single':
        # Scenario-unification single task: canonical rules live in
        # cfg['RULES'] (merged config) or cfg['rules'] (checkpoint save_config);
        # optional constant term in cfg['CONST']/cfg['const'] (default 0).
        rules = cfg.get('RULES', cfg.get('rules'))
        if not rules or len(rules) != 1:
            raise ValueError("task 'single' needs exactly one rule in RULES/rules")
        coeffs = [int(c) for c in rules[0]]
        const = int(cfg.get('CONST', cfg.get('const', 0)) or 0)
        name = ('X(k)=(' + '+'.join(f"{c}*X(k-{i + 1})" for i, c in enumerate(coeffs))
                + (f"+{const}" if const else '') + f") mod {p}")
        rule = LinearRecurrenceRule(coeffs=tuple(coeffs), p=p, const=const)
        return len(coeffs), rule.next_fn(), name
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
    if task == 'tetranacci':
        a, b, c, d = get('a'), get('b'), get('c'), get('d')
        name = f"X(k)=({a}*X(k-1)+{b}*X(k-2)+{c}*X(k-3)+{d}*X(k-4)) mod {p}"
        return 4, (lambda seq, p: (a * seq[-1] + b * seq[-2] + c * seq[-3] + d * seq[-4]) % p), name
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


# Legacy recurrence names still written for single-scenario checkpoints so old
# analysis tools can read them; order >= 5 has no legacy name.
_RECURRENCE_NAMES = {2: 'addition', 3: 'tribonacci', 4: 'tetranacci'}


def save_config_extra(norm, cfg):
    """Checkpoint save_config entries for a single-scenario or exception task.

    The new block (scenario/order/rules, canonical) is always written; legacy
    fields (lowercase a/b/c/d, the 'recurrence' name) are written alongside
    while they remain unambiguous, so old analysis tools read new checkpoints.
    """
    scenario = norm['scenario']
    if scenario == 'single':
        coeffs = [int(c) for c in norm['rules'][0]]
        order = norm['order']
        extra = {'scenario': 'single', 'order': order, 'rules': [coeffs],
                 'recurrence': _RECURRENCE_NAMES.get(order, f'single_o{order}'),
                 'const': int(norm.get('const', 0))}
        for letter, c in zip('abcd', coeffs):
            extra[letter] = c
        return extra
    # Exception tasks: legacy behavior + scenario tag.
    legacy = {'multiplication': {'recurrence': 'multiplicative'},
              'nonlinear': {'recurrence': 'nonlinear'},
              'nonlinear_mul': {'recurrence': 'nonlinear_mul'}}[scenario]
    return {**legacy, 'scenario': scenario}


def task_from_save_config(config):
    """Map a checkpoint save_config to its single-rule task name.

    New checkpoints (2026-10-08+) carry an explicit scenario/order/rules
    block which takes priority over the legacy fields: 'single' and the
    exception task names are returned as-is; 'mixed'/'action' return None.
    Legacy checkpoints (no 'scenario' key) keep the old inference: returns
    None for mixed_ab/mixed_abc/action checkpoints (those save ab_pairs
    instead of a single recurrence spec). The old name 'dynamic_mixed'
    (pre-rename checkpoints) is accepted for compatibility. Defaults to
    'addition' for old a/b-only or minimal configs.
    """
    scenario = config.get('scenario')
    if scenario is not None:
        # New checkpoints (2026-10-08+): the explicit block wins.
        # 'single' is returned as-is (callers read coefficients from
        # config['rules']); mixed/action are not single-rule tasks.
        if scenario in ('mixed', 'action'):
            return None
        return scenario
    recurrence = config.get('recurrence')
    if recurrence in ('action', 'dynamic_mixed'):
        return None
    if recurrence is None and 'ab_pairs' in config:
        return None  # mixed_ab/mixed_abc
    if 'd' in config or recurrence == 'tetranacci':
        return 'tetranacci'
    if 'c' in config or recurrence == 'tribonacci':
        return 'tribonacci'
    if recurrence == 'multiplicative':
        return 'multiplication'
    if recurrence == 'nonlinear':
        return 'nonlinear'
    if recurrence == 'nonlinear_mul':
        return 'nonlinear_mul'
    return 'addition'


def rules_from_config(rules_raw, p):
    """Build the rule list from canonical coefficient lists.

    rules_raw: list of coefficient lists in canonical order (first
    coefficient multiplies the newest value), all of the same length >= 2 —
    i.e. the 'rules' field of scenarios.normalize_task_config's output. The
    legacy {2: 'AB_PAIRS', 3: 'ABC_PAIRS'} key mapping is gone.
    """
    if not rules_raw:
        raise ValueError("rules must contain at least one rule")
    order = len(rules_raw[0])
    if order < 2:
        raise ValueError(f"recurrence order must be >= 2, got {order}")
    rules = []
    for entry in rules_raw:
        coeffs = tuple(int(c) for c in entry)
        if len(coeffs) != order:
            raise ValueError(
                f"rule {entry} has {len(coeffs)} coefficients, expected {order}")
        if any(not 0 <= c < p for c in coeffs):
            raise ValueError(f"rule {entry} has a coefficient outside [0, {p})")
        rules.append(LinearRecurrenceRule(coeffs=coeffs, p=p))
    return rules
