"""Scenario normalization layer (2026-10-08 scenario unification).

Single entry point mapping every accepted TASK spelling (new scenario names
single/mixed/action, legacy task names, and the three nonlinear exception
tasks) to one normalized dict:

    {'scenario': 'single' | 'mixed' | 'action' | <exception task name>,
     'order': int | None,                 # None for exception tasks
     'rules': [[c1, ...], ...] | None,    # canonical; None for exception tasks
     'exposure_ratios': float | [float, ...] | None,
     'source': 'new' | <legacy task name>}

Pure config translation: no torch, no RNG, no I/O. Canonical RULES convention
(x_k = c1*x_{k-1} + c2*x_{k-2} + ...) matches LinearRecurrenceRule everywhere.
The action family's historical AB_PAIRS/ABC_PAIRS convention (first
coefficient multiplies the OLDEST value) is reversed HERE, in _legacy_rules —
the only other reversal site is experiment._prepare_action's entry point
(canonical -> action-internal); see its comment.
"""
import warnings

SCENARIOS = ('single', 'mixed', 'action')
EXCEPTION_TASKS = ('multiplication', 'nonlinear', 'nonlinear_mul')

DEFAULT_RATIO = 0.7  # matches the pre-unification MAX_UNIQUE_RATIO default

# Legacy config.json coefficient defaults (pre-unification src/config.json).
_DEFAULT_AB_PAIRS = [[1, 1], [1, 2]]
_DEFAULT_ABC_PAIRS = [[1, 1, 1], [1, 2, 3]]

_LEGACY_SCENARIO = {
    'addition': 'single', 'tribonacci': 'single', 'tetranacci': 'single',
    'mixed_ab': 'mixed', 'mixed_abc': 'mixed',
    'action_trib': 'action', 'dynamic_mixed': 'action',
}

# 合法 task 名全集（报错信息用）。注意 'action' 不在 _LEGACY_SCENARIO 里：
# 它既是新场景名又是旧 task 名，由 _is_new_format 按配置内容消歧。
VALID_TASKS = SCENARIOS + EXCEPTION_TASKS + tuple(_LEGACY_SCENARIO)


def scenario_of(task):
    """Scenario name for any accepted task spelling; None if unknown."""
    if task in SCENARIOS or task in EXCEPTION_TASKS:
        return task
    return _LEGACY_SCENARIO.get(task)


def _is_new_format(task, cfg):
    """True when the config uses the new spelling (ORDER + RULES).

    'action' is both a new scenario name and a legacy task name: it is new
    format iff RULES is present, or ORDER is present without AB_PAIRS (a bare
    ORDER never appears in legacy action JSONs). Legacy action JSONs carry
    AB_PAIRS only and must keep working unchanged.
    """
    if task in ('single', 'mixed'):
        return True
    if task == 'action':
        return cfg.get('RULES') is not None or \
            (cfg.get('ORDER') is not None and not cfg.get('AB_PAIRS'))
    return False


def _validate_new_format(task, cfg):
    """New-format task (single/mixed/action): ORDER + RULES are mandatory."""
    p = cfg['P']
    order = cfg.get('ORDER')
    if order is None:
        raise ValueError(
            f"TASK={task!r}（新格式）必须显式写 ORDER（>=2 的整数）")
    if not isinstance(order, int) or isinstance(order, bool) or order < 2:
        raise ValueError(f"ORDER 必须是 >=2 的整数，got {order!r}")
    raw = cfg.get('RULES')
    if not raw:
        raise ValueError(
            f"TASK={task!r} 必须给 RULES（canonical：第一个系数乘最新的值）")
    rules = [list(map(int, r)) for r in raw]
    for r in rules:
        if len(r) != order:
            raise ValueError(f"RULE {r} 长度 {len(r)} != ORDER {order}")
        if any(not 0 <= c < p for c in r):
            raise ValueError(f"RULE {r} 系数越界 [0, {p})")
    if task == 'single' and len(rules) != 1:
        raise ValueError(f"single 场景恰好 1 条规则，got {len(rules)}")
    return order, rules


def _normalize_exposure(scenario, n_rules, cfg, legacy_value):
    """EXPOSURE_RATIOS（新键，优先）或旧键翻译值 -> float | list[float] | None."""
    raw = cfg.get('EXPOSURE_RATIOS', legacy_value)
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if scenario == 'single' else [float(raw)] * n_rules
    raw = [float(r) for r in raw]
    if len(raw) != n_rules:
        raise ValueError(
            f"EXPOSURE_RATIOS 长度 {len(raw)} != 规则数 {n_rules}")
    if scenario == 'single':
        return raw[0]
    return raw


def normalize_task_config(task, cfg):
    """See module docstring. cfg is the merged main section (must contain P)."""
    # Task 2 在这里插入：_check_removed_features(cfg)
    if scenario_of(task) is None:
        raise ValueError(
            f"未知 TASK {task!r}；合法值：新场景 {SCENARIOS}，例外 {EXCEPTION_TASKS}，"
            f"旧名 {tuple(_LEGACY_SCENARIO)}")
    if task in EXCEPTION_TASKS:
        return {'scenario': task, 'order': None, 'rules': None,
                'exposure_ratios': None, 'source': task}
    scenario = scenario_of(task)
    if _is_new_format(task, cfg):
        # Task 2 在这里插入：_resolve_conflicts(scenario, order, rules, cfg)
        order, rules = _validate_new_format(task, cfg)
        source = 'new'
        legacy_exposure = None
    else:
        # Task 2 在这里实现旧 task 名翻译（_legacy_rules / 冲突检查）；
        # Task 1 先不走到这里（旧名暂由 Task 2 补全）。
        raise NotImplementedError(f"legacy task {task!r} lands in Task 2")
    exposure = _normalize_exposure(scenario, len(rules), cfg, legacy_exposure)
    if scenario in ('single', 'mixed') and exposure is None:
        exposure = DEFAULT_RATIO if scenario == 'single' else [DEFAULT_RATIO] * len(rules)
    if scenario == 'action':
        if cfg.get('EXPOSURE_RATIOS') is not None:
            warnings.warn("action 场景不使用 EXPOSURE_RATIOS（规则逐步随机选择），已忽略",
                          stacklevel=2)
        exposure = None  # action 恒 None（规则逐步随机选择，无曝光比例概念）
    return {'scenario': scenario, 'order': order, 'rules': rules,
            'exposure_ratios': exposure, 'source': source}
