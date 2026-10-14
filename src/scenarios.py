"""Scenario normalization layer (2026-10-08 scenario unification).

Single entry point mapping every accepted TASK spelling (new scenario names
single/mixed/action, legacy task names, and the three nonlinear exception
tasks) to one normalized dict:

    {'scenario': 'single' | 'mixed' | 'action' | <exception task name>,
     'order': int | None,                 # None for exception tasks
     'rules': [[c1, ...], ...] | None,    # canonical; None for exception tasks
     'exposure_ratios': float | [float, ...] | None,
     'variant': None | 'corrupt_fix' | 'fill_blank',  # TASK_VARIANT；场景下一级的任务变体
     'const': int,                        # 常数项（CONST 键，默认 0；仅 single 可非零）
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

# 合法 TASK_VARIANT 取值（场景下一级的任务变体；None = 标准 next-token 预测）。
# corrupt_fix：恰好一位污损成随机错误值，尾部接 EOS，模型输出 [位置, 正确值]。
# fill_blank：恰好一位污损成空白 token（id=P），尾部接 EOS，模型只输出正确值。
TASK_VARIANTS = ('corrupt_fix', 'fill_blank')

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


def _check_removed_features(cfg):
    """规格 §1.5：旧配置显式启用已删除/写死功能 -> 明确报错（含替代写法）；
    与写死值一致的旧键警告后忽略。SKIP_FINAL_GENERATION_TEST 功能整体消失，
    直接忽略（没有需要保护的语义）。"""
    def err(key, replacement):
        raise ValueError(
            f"{key}={cfg[key]!r} 请求了已删除的功能（2026-10-08 场景统一）。{replacement}")

    if cfg.get('PREDICT_MISSING') is False:
        err('PREDICT_MISSING', '污损位恒以原值为目标（predict 口径），mask 模式已删除；'
                               '去掉该键即可。')
    if cfg.get('MISS_SECOND'):
        err('MISS_SECOND', '改用 MISSING_PROB + MISS_LEN。')
    if cfg.get('USE_LEARNABLE_PE'):
        err('USE_LEARNABLE_PE', '位置编码恒为 RoPE，wpe 已删除。')
    if cfg.get('USE_AB_TAG'):
        err('USE_AB_TAG', 'mixed 只剩 basic 模式，rule token / 词表扩展已删除。')
    if cfg.get('USE_CONDITIONAL_WTE'):
        err('USE_CONDITIONAL_WTE', 'cond_wte 已删除。')
    if cfg.get('COND_WTE_SHARED_RATIO'):
        err('COND_WTE_SHARED_RATIO', 'cond_wte 已删除。')
    for key in ('COND_FIX', 'COND_FIX_START', 'COND_FIX_START_A1', 'COND_FIX_START_A2'):
        if cfg.get(key) is not None:
            err(key, '参数冻结（freeze_partial）已删除。')
    if cfg.get('USE_AMP'):
        err('USE_AMP', 'AMP 已移除，训练恒为 fp32（依据 docs/AMP_PRECISION_NOTES.md）。')
    if cfg.get('AMP_DTYPE') is not None:
        err('AMP_DTYPE', 'AMP 已移除，训练恒为 fp32。')
    if cfg.get('ALLOW_TF32') is False:
        err('ALLOW_TF32', 'TF32 恒开。')
    if cfg.get('ENTROPY_PENALTY_WEIGHT'):
        err('ENTROPY_PENALTY_WEIGHT', '注意力熵惩罚已移除（恒 0）。')
    for key in ('PREDICT_MISSING', 'MISS_SECOND', 'USE_LEARNABLE_PE', 'USE_AB_TAG',
                'USE_CONDITIONAL_WTE', 'COND_WTE_SHARED_RATIO', 'USE_AMP',
                'ALLOW_TF32', 'ENTROPY_PENALTY_WEIGHT'):
        if cfg.get(key) is not None:
            warnings.warn(f"{key} 已写死/删除，与写死值一致，忽略", stacklevel=2)


def _normalize_const(scenario, cfg):
    """CONST：递推公式的常数项（x_k = c1*x_{k-1} + ... + CONST mod P），默认 0。

    目前仅 single 场景允许非零（等差数列 ORDER=1 + RULES=[[1]] + CONST=d）；
    其余场景/例外 task 设非零值报错。注意不能用键名 C——A/B/C/D 已被旧系数占用。
    """
    raw = cfg.get('CONST', 0)
    if raw is None:
        raw = 0
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"CONST 必须是整数，got {raw!r}")
    if not 0 <= raw < cfg['P']:
        raise ValueError(f"CONST 越界 [0, {cfg['P']})：{raw}")
    if raw != 0 and scenario != 'single':
        raise ValueError(
            f"CONST（常数项）目前仅 single 场景支持（TASK 属于 {scenario!r} 场景）")
    return raw


def _normalize_variant(scenario, cfg):
    """TASK_VARIANT（场景下一级的任务变体，正交于 scenario）-> None | 'corrupt_fix' | 'fill_blank'.

    变体目前仅 single 场景开放；mixed/action 留扩展位（报错信息里说明）。
    """
    raw = cfg.get('TASK_VARIANT')
    if cfg.get('FILL_BLANK_ALL_POSITIONS') and raw != 'fill_blank':
        raise ValueError(
            "FILL_BLANK_ALL_POSITIONS 仅 TASK_VARIANT='fill_blank' 支持"
            "（每个位置挖空各成一条训练样本）")
    if cfg.get('TEST_ALL_POSITIONS') and raw != 'fill_blank':
        raise ValueError(
            "TEST_ALL_POSITIONS 仅 TASK_VARIANT='fill_blank' 支持"
            "（测试集逐位置枚举挖空）")
    if (cfg.get('TEST_ALL_POSITIONS') and cfg.get('FILL_BLANK_ALL_POSITIONS')
            and raw == 'fill_blank'):
        warnings.warn(
            "FILL_BLANK_ALL_POSITIONS 已使测试集逐位置枚举，TEST_ALL_POSITIONS 冗余",
            stacklevel=3)
    if raw is None:
        return None
    if raw not in TASK_VARIANTS:
        raise ValueError(f"未知 TASK_VARIANT {raw!r}；合法值：{TASK_VARIANTS}")
    if scenario != 'single':
        raise ValueError(
            f"TASK_VARIANT={raw!r} 目前仅 single 场景支持"
            f"（TASK 属于 {scenario!r} 场景）；mixed/action 的变体尚未实现。")
    if cfg.get('MISSING_PROB'):
        raise ValueError(
            f"{raw} 变体的污损是任务内生的（恰好一位，见 settings 文档 §1.7/§1.8），"
            "与 MISSING_PROB 在线污损互斥；去掉 MISSING_PROB 即可。")
    if cfg.get('NUM_MASK') is not None:
        warnings.warn(
            f"{raw} 变体的 loss_mask 由 collate 给出（仅输出位），"
            "NUM_MASK 无意义，已忽略", stacklevel=3)
    return raw


def _validate_new_format(task, cfg):
    """New-format task (single/mixed/action): ORDER + RULES are mandatory.

    ORDER >= 2 in general; ORDER = 1（等比递推 x_k = c*x_{k-1}）仅 single 场景
    合法（状态空间 P^1，mixed/action 的多规则/flag 语义不覆盖一阶）。"""
    p = cfg['P']
    order = cfg.get('ORDER')
    if order is None:
        raise ValueError(
            f"TASK={task!r}（新格式）必须显式写 ORDER（>=2 的整数；single 允许 1）")
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError(f"ORDER 必须是 >=1 的整数，got {order!r}")
    if order == 1 and task != 'single':
        raise ValueError(f"ORDER=1 仅 single 场景支持（TASK={task!r}）")
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


def _legacy_rules(task, cfg):
    """(order, canonical rules) for a legacy task name.

    mixed_ab/mixed_abc 的 AB_PAIRS/ABC_PAIRS 本就是 canonical（第一个系数乘
    最新的值），直接用；action/action_trib/dynamic_mixed 的旧键约定相反
    （第一个系数乘最老的值），逐条 reverse —— 全仓库仅有的两处反转之一，
    另一处在 experiment._prepare_action 入口（canonical -> action 内部约定）。
    """
    if task == 'addition':
        order, rules = 2, [[int(cfg.get('A', 1)), int(cfg.get('B', 1))]]
    elif task == 'tribonacci':
        order, rules = 3, [[int(cfg.get(k, 1)) for k in 'ABC']]
    elif task == 'tetranacci':
        order, rules = 4, [[int(cfg.get(k, 1)) for k in 'ABCD']]
    elif task == 'mixed_ab':
        order, rules = 2, [list(map(int, p))
                           for p in (cfg.get('AB_PAIRS') or _DEFAULT_AB_PAIRS)]
    elif task == 'mixed_abc':
        order, rules = 3, [list(map(int, p))
                           for p in (cfg.get('ABC_PAIRS') or _DEFAULT_ABC_PAIRS)]
    elif task in ('action', 'dynamic_mixed'):
        pairs = cfg.get('AB_PAIRS') or _DEFAULT_AB_PAIRS
        order, rules = 2, [list(map(int, reversed(pair))) for pair in pairs]
    elif task == 'action_trib':
        pairs = cfg.get('ABC_PAIRS') or _DEFAULT_ABC_PAIRS
        order, rules = 3, [list(map(int, reversed(pair))) for pair in pairs]
    else:
        raise ValueError(f"unknown legacy task: {task}")
    for r in rules:
        if len(r) != order:
            raise ValueError(
                f"规则 {r} 长度 {len(r)} != order {order}（旧 task {task!r}）")
    return order, rules


def _legacy_exposure(scenario, n_rules, cfg):
    if scenario == 'single':
        return float(cfg.get('MAX_UNIQUE_RATIO', DEFAULT_RATIO))
    if scenario == 'mixed':
        ratios = cfg.get('MIXED_AB_MAX_UNIQUE_RATIOS')
        if ratios is None:
            ratios = cfg.get('MAX_UNIQUE_RATIO', DEFAULT_RATIO)
        return ratios
    return None  # action


def _resolve_conflicts(scenario, order, rules, cfg):
    """新格式 task 里新旧键同时出现：一致 -> warning + 新键优先；不一致 -> 报错。"""
    pairs_key = {2: 'AB_PAIRS', 3: 'ABC_PAIRS'}.get(order)
    if pairs_key and cfg.get(pairs_key):
        old = [list(map(int, r)) for r in cfg[pairs_key]]
        if scenario == 'action':
            # 旧键是 action 约定，反转成 canonical 再比（反转规则见 _legacy_rules）
            old = [list(reversed(r)) for r in old]
        if old != rules:
            raise ValueError(f"{pairs_key} 与 RULES 不一致"
                             f"{'（action 旧键按反转后比较）' if scenario == 'action' else ''}："
                             f"{old} != {rules}")
        warnings.warn(f"{pairs_key} 与 RULES 一致，以 RULES 为准", stacklevel=3)
    if scenario == 'single':
        letters = 'ABCD'[:order]
        for i, k in enumerate(letters):
            if k in cfg and cfg[k] is not None and int(cfg[k]) != rules[0][i]:
                raise ValueError(f"{k}={cfg[k]} 与 RULES[0][{i}]={rules[0][i]} 不一致")
        if any(k in cfg and cfg[k] is not None for k in letters):
            warnings.warn("A/B/C/D 与 RULES 一致，以 RULES 为准", stacklevel=3)
    for old_key in ('MAX_UNIQUE_RATIO', 'MIXED_AB_MAX_UNIQUE_RATIOS'):
        if cfg.get(old_key) is not None and cfg.get('EXPOSURE_RATIOS') is not None:
            if cfg[old_key] != cfg['EXPOSURE_RATIOS']:
                raise ValueError(f"{old_key}={cfg[old_key]} 与 "
                                 f"EXPOSURE_RATIOS={cfg['EXPOSURE_RATIOS']} 不一致")
            warnings.warn(f"{old_key} 与 EXPOSURE_RATIOS 一致，以 EXPOSURE_RATIOS 为准",
                          stacklevel=3)


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
    _check_removed_features(cfg)
    if scenario_of(task) is None:
        raise ValueError(
            f"未知 TASK {task!r}；合法值：新场景 {SCENARIOS}，例外 {EXCEPTION_TASKS}，"
            f"旧名 {tuple(_LEGACY_SCENARIO)}")
    # AUX_CLASSIFY 只做合法性校验（仅 mixed 合法），不进归一化结果；
    # mixed 场景默认开（experiment._prepare_mixed_recurrence 兜底默认 true），
    # 其他场景不写 -> 无行为。
    if cfg.get('AUX_CLASSIFY') and scenario_of(task) != 'mixed':
        raise ValueError(
            f"AUX_CLASSIFY=true 仅 mixed 场景支持（TASK={task!r} 属于 "
            f"{scenario_of(task)!r} 场景）；去掉该键即可。")
    variant = _normalize_variant(scenario_of(task), cfg)
    const = _normalize_const(scenario_of(task), cfg)
    if task in EXCEPTION_TASKS:
        return {'scenario': task, 'order': None, 'rules': None,
                'exposure_ratios': None, 'variant': variant, 'const': const,
                'source': task}
    scenario = scenario_of(task)
    if _is_new_format(task, cfg):
        order, rules = _validate_new_format(task, cfg)
        _resolve_conflicts(scenario, order, rules, cfg)
        source = 'new'
        legacy_exposure = None
    else:
        order, rules = _legacy_rules(task, cfg)
        p = cfg['P']
        for r in rules:
            if any(not 0 <= c < p for c in r):
                raise ValueError(f"规则 {r} 系数越界 [0, {p})")
        # 旧 task 名与新键同时出现：一致 -> warning + 新键优先；不一致 -> 报错。
        if cfg.get('RULES') is not None:
            new_rules = [list(map(int, r)) for r in cfg['RULES']]
            if new_rules != rules:
                raise ValueError(
                    f"RULES={new_rules} 与旧 task {task!r} 推出的规则 {rules} 不一致")
            warnings.warn(f"旧 task {task!r} 与 RULES 同时出现且一致，以 RULES 为准",
                          stacklevel=2)
            rules = new_rules
        if cfg.get('ORDER') is not None and cfg['ORDER'] != order:
            raise ValueError(
                f"ORDER={cfg['ORDER']} 与旧 task {task!r} 的阶数 {order} 不一致")
        source = task
        legacy_exposure = _legacy_exposure(scenario, len(rules), cfg)
        if task == 'dynamic_mixed':
            warnings.warn("task 'dynamic_mixed' 已废弃，按 'action' 处理", stacklevel=2)
    if scenario == 'action':
        # action 收到任何形状的 EXPOSURE_RATIOS（含长度不匹配的列表）都 warn 后
        # 忽略，不做长度校验；exposure_ratios 恒 None（规则逐步随机选择，无曝光比例概念）。
        if cfg.get('EXPOSURE_RATIOS') is not None:
            warnings.warn("action 场景不使用 EXPOSURE_RATIOS（规则逐步随机选择），已忽略",
                          stacklevel=2)
        exposure = None
    else:
        exposure = _normalize_exposure(scenario, len(rules), cfg, legacy_exposure)
        if exposure is None:
            exposure = DEFAULT_RATIO if scenario == 'single' else [DEFAULT_RATIO] * len(rules)
    return {'scenario': scenario, 'order': order, 'rules': rules,
            'exposure_ratios': exposure, 'variant': variant, 'const': const,
            'source': source}
