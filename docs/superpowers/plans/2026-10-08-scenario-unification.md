# 场景统一（single/mixed/action × order × miss_len）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把十余个 task 名收敛为 `TASK ∈ {single, mixed, action}`（+ `multiplication`/`nonlinear`/`nonlinear_mul` 三个例外）× `ORDER` × `MISSING_PROB`/`MISS_LEN` 的统一配置体系；新增 `src/scenarios.py` 归一化层做旧格式兼容翻译与已删功能报错；删除 MixedABTransformer/rule_head/cond_wte/wpe/AMP/COND_FIX/final generation test/MISS_SECOND/mask 污损口径；checkpoint save_config 追加 scenario/order/rules 三键；分析评估工具 order 参数化。设计规格：`docs/superpowers/specs/2026-10-08-scenario-unification-design.md`；参数决策：`docs/superpowers/specs/2026-10-08-config-params-review.md`。

**Architecture:** 配置面唯一入口是 `scenarios.normalize_task_config(task, cfg) -> dict`（纯翻译，无 torch/RNG/IO），`run_experiment` 开头调用后按 `scenario` 分派到三个 `_prepare_*`。canonical 约定：`RULES` 第一个系数乘最新的值（与 `LinearRecurrenceRule` 一致）；action 家族的旧 `AB_PAIRS`/`ABC_PAIRS` 约定相反，反转只出现在归一化层和 `_prepare_action` 入口各一次。模型层只剩 `FibonacciTransformer`（RoPE 恒用）。

**Tech Stack:** Python 3 / PyTorch，无新依赖

## Global Constraints

- 旧 `experiments/*.json`（约 3000 条）一律不改动；它们靠归一化层继续可跑，前提是没用到已删功能。
- `RULES` 全仓库统一 canonical 方向：第一个系数乘最新的值（`x_k = c1·x_{k-1} + c2·x_{k-2} + ...`）。
- action 系数反转全仓库只出现两次：`scenarios._legacy_rules`（旧 AB_PAIRS/ABC_PAIRS → canonical）和 `experiment._prepare_action` 入口（canonical → action 内部约定）；两处注释必须互相指向对方，不得出现第三处反转。
- 已删除/写死功能被旧配置**显式启用**时必须 `raise ValueError` 并给出替代写法（不允许静默降级）；与写死值一致的旧键（如 `USE_AMP: false`）警告后忽略；`SKIP_FINAL_GENERATION_TEST` 功能整体消失，该键直接忽略。
- 测试解释器：`C:/Users/Chen/anaconda3/envs/torch/python.exe`；**没有 pytest**，`tests/test_*.py` 是纯 assert 脚本带 `__main__` runner；`tests/test_sample_gen.py` 需要 `PYTHONPATH=src`。
- RNG 调用序不可变：`tests/test_mixed_ab_compat.py` 是 golden master，数据生成路径（RecurrenceDataset/MixedRecurrenceDataset/ActionDataset 的生成与 shuffle）不许增删任何 RNG 消耗；污损 `_corrupt` 删 MISS_SECOND 块是安全的（该块不消耗 RNG），其余扫描逻辑原样保留。
- git 提交粒度：每个 task 末尾一个 commit（`git add` 只加本 task 触及的文件）；**禁止 git push**、禁止任何其他 git 变更操作。
- 代码风格跟随现有文件：模块级 docstring 说明设计意图，中英文注释混排，测试文件末尾 `print("ALL TESTS PASSED: <file>")`。
- 基线事实（2026-10-08 本机实测）：`tests/test_prepare_mixed_recurrence.py`（num_mask 断言过时）和 `tests/test_fresh_test_eval.py`（single fresh-test 日志消息正则过时）在重构**前** already red；`tests/test_experiment_queue.py` 与 `tests/test_remote_experiment_runner.py` 依赖 `fcntl`，Windows 本机不可跑（Linux 服务器跑）。Task 12 负责修这两个既红文件并在最终回归中说明。
- Task 4/5/6 是纯删除任务，`experiment.py` 在 Task 7 才适配新签名——这中间的 commit 上跨文件端到端测试（test_chain_run / test_training_smoke / test_early_stop 等）不绿是**预期中间状态**，不要回头修，每个 task 只要求自己列出的 focused tests 绿。

## Task 1: `src/scenarios.py` 核心——新格式校验 + NormalizedTask

**Files:**
- Create: `src/scenarios.py`
- Test: `tests/test_scenarios.py`

**Interfaces:**
- Produces（后续所有 task 消费这些名字，签名冻结）:
  ```python
  SCENARIOS = ('single', 'mixed', 'action')
  EXCEPTION_TASKS = ('multiplication', 'nonlinear', 'nonlinear_mul')

  def scenario_of(task: str) -> str | None
      # 任何合法 task 拼写（新场景名/例外名/旧 task 名）-> scenario 名；未知返回 None

  def normalize_task_config(task: str, cfg: dict) -> dict
      # cfg = 合并后的 main 段（必须含 P）。返回：
      # {'scenario': 'single'|'mixed'|'action'|<例外task名>,
      #  'order': int | None,                 # None = 例外 task
      #  'rules': [[c1, ...], ...] | None,    # canonical 方向；None = 例外 task
      #  'exposure_ratios': float | list[float] | None,  # action 恒 None
      #  'source': 'new' | <旧task名>}
      # 所有校验失败/已删功能/新旧键冲突 -> raise ValueError（消息含原因与替代写法）
  ```

- [ ] **Step 1: 写 tests/test_scenarios.py 的新格式部分（此时无 src/scenarios.py，测试必失败）**

完整文件（本 task 先写到 `test_exceptions_passthrough` 为止的用例，Task 2 再追加；`__main__` 用自动发现，后续 task 直接向文件中部插函数即可）：

```python
"""Unit tests for src/scenarios.py (scenario normalization layer)."""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from scenarios import (SCENARIOS, EXCEPTION_TASKS, scenario_of,
                       normalize_task_config)


def _expect_error(fn, *fragments):
    try:
        fn()
    except ValueError as exc:
        msg = str(exc)
        for frag in fragments:
            assert frag in msg, f"{frag!r} not in error: {msg}"
        return
    raise AssertionError(f"expected ValueError containing {fragments}")


def test_new_single_happy():
    norm = normalize_task_config('single', {'P': 53, 'ORDER': 3, 'RULES': [[1, 2, 3]]})
    assert norm == {'scenario': 'single', 'order': 3, 'rules': [[1, 2, 3]],
                    'exposure_ratios': 0.7, 'source': 'new'}


def test_new_mixed_scalar_exposure():
    norm = normalize_task_config('mixed', {'P': 23, 'ORDER': 2,
                                           'RULES': [[1, 1], [2, 3]],
                                           'EXPOSURE_RATIOS': 0.5})
    assert norm['scenario'] == 'mixed' and norm['order'] == 2
    assert norm['rules'] == [[1, 1], [2, 3]]
    assert norm['exposure_ratios'] == [0.5, 0.5]  # scalar -> per-rule list


def test_new_mixed_list_exposure_and_default():
    norm = normalize_task_config('mixed', {'P': 23, 'ORDER': 4, 'RULES': [[1, 0, 0, 1]]})
    assert norm['exposure_ratios'] == [0.7]  # default 0.7 per rule
    norm = normalize_task_config('mixed', {'P': 23, 'ORDER': 2,
                                           'RULES': [[1, 1], [2, 3], [3, 5]],
                                           'EXPOSURE_RATIOS': [0.3, 0.4, 0.5]})
    assert norm['exposure_ratios'] == [0.3, 0.4, 0.5]


def test_new_action_no_exposure():
    norm = normalize_task_config('action', {'P': 53, 'ORDER': 2, 'RULES': [[1, 1], [2, 1]]})
    assert norm['scenario'] == 'action' and norm['exposure_ratios'] is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        norm = normalize_task_config('action', {'P': 53, 'ORDER': 2,
                                                'RULES': [[1, 1]],
                                                'EXPOSURE_RATIOS': 0.5})
    assert norm['exposure_ratios'] is None
    assert any('EXPOSURE_RATIOS' in str(w.message) for w in caught)


def test_exceptions_passthrough():
    for task in EXCEPTION_TASKS:
        norm = normalize_task_config(task, {'P': 53})
        assert norm == {'scenario': task, 'order': None, 'rules': None,
                        'exposure_ratios': None, 'source': task}


if __name__ == '__main__':
    for _name, _fn in sorted(list(globals().items())):
        if _name.startswith('test_') and callable(_fn):
            _fn()
            print(f'PASS {_name}')
    print('ALL TESTS PASSED: test_scenarios.py')
```

跑：`C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_scenarios.py`
预期：`ModuleNotFoundError: No module named 'scenarios'`（确认失败）。

- [ ] **Step 2: 写 src/scenarios.py 骨架 + 新格式校验（最小实现让 Step 1 通过）**

完整文件内容（本 task 的实现范围；Task 2 会在标记位置追加旧键映射与删除功能检查）：

```python
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
    if scenario == 'action' and cfg.get('EXPOSURE_RATIOS') is not None:
        warnings.warn("action 场景不使用 EXPOSURE_RATIOS（规则逐步随机选择），已忽略",
                      stacklevel=2)
    return {'scenario': scenario, 'order': order, 'rules': rules,
            'exposure_ratios': exposure, 'source': source}
```

跑：`C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_scenarios.py`
预期：5 个用例全 PASS，最后一行 `ALL TESTS PASSED: test_scenarios.py`。

- [ ] **Step 3: 补新格式校验的失败用例（先加测试确认失败，再确认实现已覆盖——本 task 的实现已在 Step 2 给全，此步是补测试证明校验生效）**

在 tests/test_scenarios.py 的 `test_exceptions_passthrough` 之后、`__main__` 之前追加：

```python
def test_new_format_errors():
    _expect_error(lambda: normalize_task_config('single', {'P': 53, 'RULES': [[1, 1]]}),
                  'ORDER')
    _expect_error(lambda: normalize_task_config('single', {'P': 53, 'ORDER': 3}),
                  'RULES')
    _expect_error(lambda: normalize_task_config('single', {'P': 53, 'ORDER': 1, 'RULES': [[1]]}),
                  'ORDER')
    _expect_error(lambda: normalize_task_config('mixed', {'P': 53, 'ORDER': 2,
                                                          'RULES': [[1, 1], [1, 2, 3]]}),
                  '长度')
    _expect_error(lambda: normalize_task_config('mixed', {'P': 53, 'ORDER': 2,
                                                          'RULES': [[1, 53]]}),
                  '越界')
    _expect_error(lambda: normalize_task_config('single', {'P': 53, 'ORDER': 2,
                                                           'RULES': [[1, 1], [2, 3]]}),
                  '恰好 1 条')
    _expect_error(lambda: normalize_task_config('mixed', {'P': 53, 'ORDER': 2,
                                                          'RULES': [[1, 1], [2, 3]],
                                                          'EXPOSURE_RATIOS': [0.5]}),
                  'EXPOSURE_RATIOS')
    err = None
    try:
        normalize_task_config('fibonacci', {'P': 53})
    except ValueError as exc:
        err = str(exc)
    assert err is not None and 'single' in err and 'mixed_ab' in err and 'action_trib' in err


def test_scenario_of():
    assert scenario_of('single') == 'single'
    assert scenario_of('nonlinear_mul') == 'nonlinear_mul'
    assert scenario_of('nonsense') is None
```

跑：`C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_scenarios.py`
预期：7 个用例全 PASS。

- [ ] **Step 4: commit**

```bash
git add src/scenarios.py tests/test_scenarios.py
git commit -m "scenarios: normalization layer core (new-format validation)"
```

## Task 2: scenarios.py 兼容映射——旧 task 名 / 旧键 / action 反转 / 删除功能报错 / 新旧键冲突

**Files:**
- Modify: `src/scenarios.py`（Step 2 给出的标记位置）
- Test: `tests/test_scenarios.py`

**Interfaces:**
- Consumes: Task 1 的 `normalize_task_config` / `scenario_of` / 模块常量。
- Produces: 不变（签名冻结）；内部新增 `_check_removed_features(cfg)`、`_legacy_rules(task, cfg) -> (order, rules)`、`_legacy_exposure(scenario, n_rules, cfg)`、`_resolve_conflicts(task, scenario, order, rules, cfg)`，均为模块私有。

- [ ] **Step 1: 写兼容映射测试（确认失败：旧名目前 raise NotImplementedError）**

在 tests/test_scenarios.py `__main__` 之前追加：

```python
def test_legacy_single_mapping():
    assert normalize_task_config('addition', {'P': 23, 'A': 2, 'B': 3})['rules'] == [[2, 3]]
    norm = normalize_task_config('tribonacci', {'P': 23, 'A': 1, 'B': 2, 'C': 3})
    assert norm['scenario'] == 'single' and norm['order'] == 3
    assert norm['rules'] == [[1, 2, 3]] and norm['source'] == 'tribonacci'
    norm = normalize_task_config('tetranacci', {'P': 23, 'A': 1, 'B': 1, 'C': 1, 'D': 1})
    assert norm['order'] == 4 and norm['rules'] == [[1, 1, 1, 1]]
    # 系数缺省全 1（沿用 single_rule_from_task 的旧默认）
    assert normalize_task_config('addition', {'P': 23})['rules'] == [[1, 1]]


def test_legacy_mixed_mapping():
    norm = normalize_task_config('mixed_ab', {'P': 23, 'AB_PAIRS': [[3, 5], [7, 11]]})
    assert norm['scenario'] == 'mixed' and norm['order'] == 2
    assert norm['rules'] == [[3, 5], [7, 11]]  # mixed 旧键本就是 canonical，直接用
    norm = normalize_task_config('mixed_abc', {'P': 23, 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]]})
    assert norm['order'] == 3 and norm['rules'] == [[1, 1, 1], [1, 2, 3]]
    # 缺省 pairs（沿用旧 config.json main 段默认）
    assert normalize_task_config('mixed_ab', {'P': 23})['rules'] == [[1, 1], [1, 2]]
    assert normalize_task_config('mixed_abc', {'P': 23})['rules'] == [[1, 1, 1], [1, 2, 3]]


def test_legacy_action_mapping_reverses():
    # action 旧约定：第一个系数乘最老的值 -> canonical 需逐条 reverse。
    norm = normalize_task_config('action', {'P': 53, 'AB_PAIRS': [[1, 1], [1, 2]]})
    assert norm['scenario'] == 'action' and norm['order'] == 2
    assert norm['rules'] == [[1, 1], [2, 1]]
    norm = normalize_task_config('dynamic_mixed', {'P': 53, 'AB_PAIRS': [[3, 4]]})
    assert norm['scenario'] == 'action' and norm['rules'] == [[4, 3]]
    norm = normalize_task_config('action_trib', {'P': 53, 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]]})
    assert norm['order'] == 3 and norm['rules'] == [[1, 1, 1], [3, 2, 1]]
    # 缺省 pairs
    assert normalize_task_config('action', {'P': 53})['rules'] == [[1, 1], [2, 1]]
    assert normalize_task_config('action_trib', {'P': 53})['rules'] == [[1, 1, 1], [3, 2, 1]]


def test_legacy_exposure_keys():
    norm = normalize_task_config('addition', {'P': 23, 'MAX_UNIQUE_RATIO': 0.3})
    assert norm['exposure_ratios'] == 0.3
    norm = normalize_task_config('mixed_ab', {'P': 23, 'AB_PAIRS': [[1, 1], [2, 3]],
                                              'MIXED_AB_MAX_UNIQUE_RATIOS': [0.2, 0.4]})
    assert norm['exposure_ratios'] == [0.2, 0.4]
    # mixed 旧 JSON 只有标量 MAX_UNIQUE_RATIO 时广播到每条规则
    norm = normalize_task_config('mixed_ab', {'P': 23, 'AB_PAIRS': [[1, 1], [2, 3]],
                                              'MAX_UNIQUE_RATIO': 0.6})
    assert norm['exposure_ratios'] == [0.6, 0.6]
```

跑：`C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_scenarios.py`
预期：新 4 个用例失败（NotImplementedError / ValueError）。

- [ ] **Step 2: 实现旧名映射与旧键翻译**

在 src/scenarios.py 做三处编辑：

(a) `normalize_task_config` 的 `else:` 分支（Step 2 标记的 NotImplementedError 处）替换为：

```python
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
```

(b) 在 `_validate_new_format` 之后新增两个私有函数：

```python
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
```

(c) `normalize_task_config` 里 `if _is_new_format(task, cfg):` 分支中 `order, rules = _validate_new_format(task, cfg)` 之后插入：

```python
        _resolve_conflicts(scenario, order, rules, cfg)
```

并新增私有函数：

```python
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
```

跑：`C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_scenarios.py`
预期：11 个用例全 PASS。

- [ ] **Step 3: 写删除功能报错 + 冲突处理测试（确认失败）**

`__main__` 之前追加：

```python
def test_removed_features_raise():
    # 规格 §1.5：每条已删/写死功能各一例，报错消息含键名。
    cases = [
        ({'PREDICT_MISSING': False}, 'PREDICT_MISSING'),
        ({'MISS_SECOND': True}, 'MISS_SECOND'),
        ({'USE_LEARNABLE_PE': True}, 'USE_LEARNABLE_PE'),
        ({'USE_AB_TAG': True}, 'USE_AB_TAG'),
        ({'USE_CONDITIONAL_WTE': True}, 'USE_CONDITIONAL_WTE'),
        ({'COND_WTE_SHARED_RATIO': 0.5}, 'COND_WTE_SHARED_RATIO'),
        ({'COND_FIX': 'WTE'}, 'COND_FIX'),
        ({'COND_FIX_START': 0.9}, 'COND_FIX_START'),
        ({'COND_FIX_START_A1': 0.9, 'COND_FIX_START_A2': 0.8}, 'COND_FIX_START_A1'),
        ({'USE_AMP': True}, 'USE_AMP'),
        ({'AMP_DTYPE': 'bfloat16'}, 'AMP_DTYPE'),
        ({'ALLOW_TF32': False}, 'ALLOW_TF32'),
        ({'ENTROPY_PENALTY_WEIGHT': 0.5}, 'ENTROPY_PENALTY_WEIGHT'),
    ]
    for extra, key in cases:
        cfg = {'P': 23, 'A': 1, 'B': 1, **extra}
        _expect_error(lambda c=cfg: normalize_task_config('addition', c), key)


def test_consistent_hardwired_keys_warn_not_raise():
    # 与写死值一致的旧键：警告后忽略，不报错。
    cfg = {'P': 23, 'A': 1, 'B': 1, 'USE_AMP': False, 'USE_AB_TAG': False,
           'USE_LEARNABLE_PE': False, 'PREDICT_MISSING': True, 'ALLOW_TF32': True,
           'ENTROPY_PENALTY_WEIGHT': 0.0, 'MISS_SECOND': False,
           'USE_CONDITIONAL_WTE': False, 'COND_WTE_SHARED_RATIO': 0.0,
           'SKIP_FINAL_GENERATION_TEST': True}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        norm = normalize_task_config('addition', cfg)
    assert norm['rules'] == [[1, 1]]


def test_conflicting_keys():
    # 一致 -> warning + 新键优先
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        norm = normalize_task_config('mixed', {'P': 23, 'ORDER': 2,
                                               'RULES': [[1, 1], [2, 3]],
                                               'AB_PAIRS': [[1, 1], [2, 3]]})
    assert norm['rules'] == [[1, 1], [2, 3]]
    assert any('AB_PAIRS' in str(w.message) for w in caught)
    # action 旧键按反转后与 canonical RULES 比较
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        norm = normalize_task_config('action', {'P': 53, 'ORDER': 2,
                                                'RULES': [[2, 1]],
                                                'AB_PAIRS': [[1, 2]]})
    assert norm['rules'] == [[2, 1]]
    # 不一致 -> 报错
    _expect_error(lambda: normalize_task_config('mixed', {'P': 23, 'ORDER': 2,
                                                          'RULES': [[1, 1]],
                                                          'AB_PAIRS': [[9, 9]]}),
                  '不一致')
    _expect_error(lambda: normalize_task_config('single', {'P': 23, 'ORDER': 2,
                                                           'RULES': [[1, 1]], 'B': 5}),
                  '不一致')
    _expect_error(lambda: normalize_task_config('tribonacci', {'P': 23, 'A': 1, 'B': 2,
                                                               'C': 3, 'RULES': [[1, 2, 9]]}),
                  '不一致')
    _expect_error(lambda: normalize_task_config('mixed_ab', {'P': 23,
                                                             'AB_PAIRS': [[1, 1]],
                                                             'ORDER': 3}),
                  'ORDER')
```

跑确认失败（`_check_removed_features` / `_resolve_conflicts` 旧键分支尚未接线）。

- [ ] **Step 4: 实现 `_check_removed_features` 并接线**

在 src/scenarios.py 新增（放在 `_validate_new_format` 之前），并在 `normalize_task_config` 第一行（`if scenario_of(task) is None:` 之前）调用 `_check_removed_features(cfg)`：

```python
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
```

跑：`C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_scenarios.py`
预期：14 个用例全 PASS，最后一行 `ALL TESTS PASSED: test_scenarios.py`。

- [ ] **Step 5: commit**

```bash
git add src/scenarios.py tests/test_scenarios.py
git commit -m "scenarios: legacy task mapping, action reversal, removed-feature errors"
```

## Task 3: rules.py——消费归一化结果 + save_config 三键 + task_from_save_config 新块优先

**Files:**
- Modify: `src/rules.py`（rules_from_config:126-149；save_config_extra:81-97；task_from_save_config:100-123；single_rule_from_task:44-78）
- Modify: `src/experiment.py:123`（`rules = rules_from_config(cfg, order)` 一处调用点适配）
- Test: `tests/test_rules.py`

**Interfaces:**
- Consumes: `scenarios.normalize_task_config` 的输出 dict（仅测试里构造；rules.py 本身不 import scenarios）。
- Produces（签名冻结）:
  ```python
  def rules_from_config(rules_raw, p) -> list[LinearRecurrenceRule]
      # rules_raw: canonical 系数列表的列表，长度任意但全部一致且 >= 2

  def save_config_extra(norm: dict, cfg: dict) -> dict
      # norm = normalize_task_config 的输出；cfg = 合并 main 段（例外 task 用）

  def task_from_save_config(config) -> str | None
      # 有 'scenario' 键优先：mixed/action -> None；'single'/例外名原样返回

  def single_rule_from_task(task, cfg)  # 新增 task == 'single' 分支
  ```

- [ ] **Step 1: 改写 tests/test_rules.py（确认失败）**

四处修改（保留未提及的既有用例不动）：

(a) `test_rules_from_config_order2` / `test_rules_from_config_order3` 改为新签名，并加一个任意 order 用例：

```python
def test_rules_from_config_order2():
    rules = rules_from_config([[3, 5], [7, 11]], 23)
    assert [r.coeffs for r in rules] == [(3, 5), (7, 11)]
    assert all(r.p == 23 for r in rules)
    assert rules[0].name == "a3b5"


def test_rules_from_config_any_order():
    rules = rules_from_config([[1, 2, 3]], 23)
    assert [r.coeffs for r in rules] == [(1, 2, 3)]
    rules = rules_from_config([[1, 2, 3, 4, 5]], 23)  # 任意 order，不再有 2/3 写死映射
    assert rules[0].order == 5


def test_rules_from_config_validation():
    for bad in (None, [], [[1, 1], [1, 2, 3]], [[1, 99]], [[1]]):
        try:
            rules_from_config(bad, 23)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass
```

（`[[1]]`：order 1 < 2 必须报；`[[1, 99]]`：99 >= p=23 越界；`[[1,1],[1,2,3]]`：长度不一致。）

(b) `test_save_config_round_trip` 整函数替换为：

```python
def test_save_config_round_trip():
    """save_config_extra（写）与 task_from_save_config（读）必须一致：
    线性 single -> 'single'；例外 task 名原样往返；新块优先于旧字段。"""
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
    from scenarios import normalize_task_config
    for task in ('addition', 'tribonacci', 'tetranacci'):
        cfg = {'P': 23, 'A': 2, 'B': 3, 'C': 4, 'D': 5}
        norm = normalize_task_config(task, cfg)
        extra = save_config_extra(norm, cfg)
        assert task_from_save_config(extra) == 'single', (task, extra)
        assert extra['scenario'] == 'single' and extra['order'] == norm['order']
        assert extra['rules'] == [list(norm['rules'][0])]
    for task in ('multiplication', 'nonlinear', 'nonlinear_mul'):
        extra = save_config_extra(normalize_task_config(task, {'P': 23}), {'P': 23})
        assert task_from_save_config(extra) == task
    # 旧 checkpoint（无 scenario 块）反推不变
    assert task_from_save_config({'a': 1, 'b': 2, 'c': 3,
                                  'recurrence': 'tribonacci'}) == 'tribonacci'
    assert task_from_save_config({'ab_pairs': [[1, 1]], 'order': 2}) is None
    assert task_from_save_config({'recurrence': 'action', 'ab_pairs': [[1, 1]]}) is None
    assert task_from_save_config({'recurrence': 'dynamic_mixed', 'ab_pairs': [[1, 1]]}) is None
    assert task_from_save_config({'p': 23}) == 'addition'
    # 新块优先：scenario 与旧字段不一致时以 scenario 为准
    assert task_from_save_config({'scenario': 'single', 'order': 3,
                                  'rules': [[1, 2, 3]],
                                  'recurrence': 'addition'}) == 'single'
    assert task_from_save_config({'scenario': 'mixed', 'order': 2,
                                  'rules': [[1, 1]], 'ab_pairs': [[1, 1]]}) is None
    assert task_from_save_config({'scenario': 'action', 'order': 2,
                                  'rules': [[2, 1]]}) is None
```

(c) 新增：

```python
def test_single_rule_from_task_single_scenario():
    p = 23
    init_len, fn, name = single_rule_from_task('single', {'P': p, 'RULES': [[2, 3, 1]]})
    assert init_len == 3 and fn([1, 2, 3], p) == (2 * 3 + 3 * 2 + 1 * 1) % p
    assert 'mod 23' in name
    # checkpoint 小写键
    init_len, fn, _ = single_rule_from_task('single', {'p': p, 'rules': [[5, 6]]})
    assert init_len == 2 and fn([4, 5], p) == (5 * 5 + 6 * 4) % p
    try:
        single_rule_from_task('single', {'P': p, 'RULES': [[1, 1], [2, 2]]})
        assert False, "single 多规则必须报"
    except ValueError:
        pass
```

(d) `__main__` 的调用清单：`test_rules_from_config_order3` 改名 `test_rules_from_config_any_order`，并加 `test_single_rule_from_task_single_scenario()`。

跑：`C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_rules.py`
预期：失败（TypeError/AssertionError，新签名尚未实现）。

- [ ] **Step 2: 实现 rules.py 四处改动**

(a) `rules_from_config`（rules.py:126-149）整函数替换：

```python
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
```

(b) `single_rule_from_task`（rules.py:44-78）在 `if task == 'addition':` 之前插入新分支：

```python
    if task == 'single':
        # Scenario-unification single task: canonical rules live in
        # cfg['RULES'] (merged config) or cfg['rules'] (checkpoint save_config).
        rules = cfg.get('RULES', cfg.get('rules'))
        if not rules or len(rules) != 1:
            raise ValueError("task 'single' needs exactly one rule in RULES/rules")
        coeffs = [int(c) for c in rules[0]]
        name = ('X(k)=(' + '+'.join(f"{c}*X(k-{i + 1})" for i, c in enumerate(coeffs))
                + f") mod {p}")
        rule = LinearRecurrenceRule(coeffs=tuple(coeffs), p=p)
        return len(coeffs), rule.next_fn(), name
```

（`p` 已在函数开头解析；name 格式与旧 addition/tribonacci/tetranacci 字符串完全一致。）

(c) `save_config_extra`（rules.py:81-97）整函数替换：

```python
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
                 'recurrence': _RECURRENCE_NAMES.get(order, f'single_o{order}')}
        for letter, c in zip('abcd', coeffs):
            extra[letter] = c
        return extra
    # Exception tasks: legacy behavior + scenario tag.
    legacy = {'multiplication': {'recurrence': 'multiplicative'},
              'nonlinear': {'recurrence': 'nonlinear'},
              'nonlinear_mul': {'recurrence': 'nonlinear_mul'}}[scenario]
    return {**legacy, 'scenario': scenario}
```

注意：旧实现里 `get` lambda 与 addition/tribonacci/tetranacci 的 a/b/c/d 提取由归一化层（Task 2 的 `_legacy_rules`）接管，此处删除。

(d) `task_from_save_config`（rules.py:100-123）在函数体开头（docstring 之后）插入：

```python
    scenario = config.get('scenario')
    if scenario is not None:
        # New checkpoints (2026-10-08+): the explicit block wins.
        # 'single' is returned as-is (callers read coefficients from
        # config['rules']); mixed/action are not single-rule tasks.
        if scenario in ('mixed', 'action'):
            return None
        return scenario
```

docstring 同步更新（说明新块优先）。

(e) `src/experiment.py:123` 一行适配（`_prepare_mixed_recurrence` 内，Task 7 才换 norm 版）：

```python
    # Temporary call-site adaptation (Task 3); Task 7 replaces this with
    # rules_from_config(norm['rules'], P) once the normalized config flows in.
    rules = rules_from_config(cfg.get({2: 'AB_PAIRS', 3: 'ABC_PAIRS'}[order]), P)
```

跑：`C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_rules.py`
预期：全部 PASS。再跑 `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_scenarios.py` 确认无回归。

- [ ] **Step 3: 验证既有调用方未破**

跑（都应 PASS；test_prepare_mixed_recurrence 基线已红，只需确认失败点仍是 line 34 的 num_mask 断言而非新引入的错误）：

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_mixed_ab_compat.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_prepare_mixed_recurrence.py
```

- [ ] **Step 4: commit**

```bash
git add src/rules.py src/experiment.py tests/test_rules.py
git commit -m "rules: consume normalized rules; save_config scenario/order/rules; task_from_save_config new block"
```

## Task 4: models.py——删 MixedABTransformer / rule_head / cond_wte / wpe / entropy penalty

**Files:**
- Modify: `src/models.py`
- Delete: `tests/test_rule_start_offset.py`
- Modify: `tests/test_activation_patch.py`（make_model:14-19）
- Modify: `tests/test_chain_run.py`（test_load_partial_checkpoint:17-42）

**Interfaces:**
- Produces（签名冻结）:
  ```python
  class FibonacciTransformer(nn.Module):
      def __init__(self, p=53, d_model=64, n_head=1, n_layer=1, block_size=128,
                   dropout=0.0, mlp_ratio=4, pad_token_id=None, vocab_size=None)
      def forward(self, idx, targets=None, loss_mask=None)  # -> (logits, loss)
  class CausalSelfAttention:  # __init__(n_embd, n_head, block_size, dropout=0.0, rope=None)
      def forward(self, x)    # -> y（不再返回 penalty）
  class TransformerBlock:     # __init__(n_embd, n_head, block_size, dropout=0.0, rope=None, mlp_ratio=4)
      def forward(self, x)    # -> x
  def _lm_loss(logits, targets, loss_mask)  # 不再有 total_penalty
  # 删除：MixedABTransformer、RULE_LOSS_WEIGHT
  ```
- 中间状态提醒：本 task 后 experiment.py（仍 import/实例化 MixedABTransformer）与 analyze_attention.load_model 的 MixedABTransformer 分支暂时破，Task 7/9 修复；只要求下面列出的 focused tests 绿。

- [ ] **Step 1: 先改测试（确认失败）**

(a) 删除 `tests/test_rule_start_offset.py`（整个文件测的就是被删的 MixedABTransformer rule_head 偏移）。

(b) `tests/test_activation_patch.py:14-19` 的 `make_model` 替换为：

```python
def make_model(seed=0):
    torch.manual_seed(seed)
    return models.FibonacciTransformer(
        p=11, d_model=32, n_head=2, n_layer=2, block_size=32,
        mlp_ratio=4).eval()
```

（activation_patch.py 只用 `model.transformer.*` 内部结构，不依赖 ab_labels/cond_wte，无需其他改动。）

(c) `tests/test_chain_run.py:17-42` 的 `test_load_partial_checkpoint` 整函数替换（不再用 MixedABTransformer；改用词表不同的两个 FibonacciTransformer 验证 name/shape 过滤）：

```python
def test_load_partial_checkpoint():
    p = 7
    src = FibonacciTransformer(p=p, d_model=32, n_head=1, n_layer=1, block_size=32)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, 'src.pth')
        torch.save({'model_state_dict': src.state_dict(),
                    'config': {'p': p, 'a': 1, 'b': 1, 'recurrence': 'addition'}}, ckpt)

        torch.manual_seed(123)
        # action 式词表扩展模型：wte/lm_head 形状不同，必须被跳过
        dst = FibonacciTransformer(p=p, d_model=32, n_head=1, n_layer=1,
                                   block_size=32, vocab_size=p + 3,
                                   pad_token_id=p)
        wte_before = dst.transformer.wte.weight.detach().clone()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _load_partial_checkpoint(dst, ckpt, 'cpu')
        out = buf.getvalue()

        # 同形同名的 backbone 权重被拷贝；形状不符的 wte/lm_head 保持初始化
        assert torch.equal(dst.transformer.h[0].attn.c_attn.weight,
                           src.transformer.h[0].attn.c_attn.weight)
        assert torch.equal(dst.transformer.wte.weight, wte_before)
        assert 'loaded' in out and 'skipped' in out
```

`test_chain_run.py:14` 的 import 改为 `from models import FibonacciTransformer`。

跑（确认失败）：

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_activation_patch.py
C:/Users/Chen/anaconda3/envs/torch/python.exe -c "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'tests'); import test_chain_run as t; t.test_load_partial_checkpoint(); print('PASS partial')"
```

预期：TypeError/AttributeError（`use_learnable_pe` 等参数还在、MixedABTransformer import 已删前是 OK 的——注意顺序：先提交测试改动前它们可能仍通过，以 models.py 改完后必须通过为准；本步只要求测试文件改完、语句可解析）。

- [ ] **Step 2: models.py 删除与瘦身（精确锚点）**

(a) `CausalSelfAttention.__init__`（models.py:43-59）：删 `entropy_penalty_weight=0.0` 参数与 `self.entropy_penalty_weight = ...`（line 51）。
(b) `CausalSelfAttention.forward`（61-102）：删 penalty 块（83-95 行，`penalty = torch.tensor(0.0...)` 到 `penalty = penalty + ...`）；`return y, penalty` 改为 `return y`。同时删除 76-79 行注释里关于 fp16 autocast 的注释段中已过时的表述——保留 clamp 代码本身（`neg = max(...)` 无害），但把两行注释改成 `# Clamp the mask value to the dtype's range (defensive; fp32 下是 no-op)。`
(c) `TransformerBlock`（104-121）：删 `entropy_penalty_weight=0.0` 参数、透传与 `forward` 的 penalty 返回；`forward` 变为：

```python
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

(d) `_lm_loss`（124-149）：删 `total_penalty` 参数，末尾 `return loss + total_penalty` 改为 `return loss`；docstring 删 "total_penalty is added at the end."。
(e) `FibonacciTransformer.__init__`（153-185）：签名删 `entropy_penalty_weight`、`use_learnable_pe`、`restricted_token_ids`；删 163-170 行的 `self.use_learnable_pe` 与 if/else wpe 分支，替换为 `self.rope = RotaryEmbedding(d_model // n_head)`；`TransformerBlock(...)` 调用删 `entropy_penalty_weight` 实参。
(f) `FibonacciTransformer.forward`（195-220）：删 201-204 行 wpe 分支；penalty 累计改为直接调用：

```python
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = _lm_loss(logits, targets, loss_mask)
        return logits, loss
```

(g) 整段删除 `RULE_LOSS_WEIGHT = 0.5`（224）与 `class MixedABTransformer`（227-340，最后一行 `return logits, loss, rule_logits`）。

- [ ] **Step 3: 跑 focused tests**

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_activation_patch.py
C:/Users/Chen/anaconda3/envs/torch/python.exe -c "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'tests'); import test_chain_run as t; t.test_load_partial_checkpoint(); print('PASS partial')"
```

预期：`10/10 tests passed` 与 `PASS partial`。

- [ ] **Step 4: commit**

```bash
git add src/models.py tests/test_activation_patch.py tests/test_chain_run.py
git rm tests/test_rule_start_offset.py
git commit -m "models: remove MixedABTransformer/rule_head/cond_wte/wpe/entropy penalty"
```

## Task 5: training.py——删 AMP / COND_FIX / freeze_partial；_unpack_batch 收缩

**Files:**
- Modify: `src/training.py`
- Modify: `tests/test_chain_run.py`（删 test_freeze_from_start:111-138 及 `__main__` 里的调用行）

**Interfaces:**
- Produces（签名冻结）:
  ```python
  def train_epoch(model, dataloader, optimizer, device, num_mask=1, extra_kwargs_fn=None,
                  first_task_weight=1.0, grad_accum_steps=1, grad_clip_norm=1.0,
                  opt_diag_interval=0, epoch=None)
  def run_training_engine(model, train_loader, test_loader, optimizer, scheduler, device,
                          epochs, eval_interval, early_stop_accuracy, early_stop_no_improve,
                          save_path, save_config, num_mask=1, extra_kwargs_fn=None,
                          first_task_weight=1.0, max_train_hours=None, resume_state=None,
                          test_loader_fn=None, grad_accum_steps=1, grad_clip_norm=1.0,
                          opt_diag_interval=0, extra_epochs_after_high_acc=0, epoch_cap=None)
  # 删除：freeze_partial、_sample_seq（Task 7 随 final_eval 一起删 core.py 引用）
  ```
- 中间状态提醒：experiment.py 仍按旧签名传 cond_fix*/use_amp 等 kwargs，Task 7 前 run_experiment 系测试会 TypeError，属预期。

- [ ] **Step 1: 删 test_chain_run.py 的 test_freeze_from_start**

删除 `tests/test_chain_run.py:111-138` 整个 `test_freeze_from_start` 函数及 `__main__` 中的 `test_freeze_from_start()` 行（222 行附近）。该用例测的就是被删的 COND_FIX 功能。

- [ ] **Step 2: training.py 删除（精确锚点）**

(a) 整段删 `freeze_partial`（training.py:34-60）。
(b) `_unpack_batch`（63-120）：删 `MIXED_AB_MASKED` 分支（100-105 行，`if tag == BatchTag.MIXED_AB_MASKED:` 整块）；docstring 里 `MIXED_AB_MASKED -> (ab_indices, loss_mask)` 行删除。（BatchTag 枚举本身 Task 6 删成员；此处先删消费分支。）
(c) `train_epoch`（184-291）：签名删 `frozen_param_states=None, grad_scaler=None`；删 scaler 分支（224-231 的 `if grad_scaler is not None:` 两处与 256-260 的 step/update）；`(loss / grad_accum_steps).backward()` 与 `optimizer.step()` 变为无条件路径；删 261-265 的 frozen 恢复块；`logits, loss, *_ = model(...)` 改为 `logits, loss = model(x, targets, loss_mask, **kwargs)`。
(d) `evaluate`（293-338）：同样改 `logits, loss = model(...)`（316 行）。
(e) `run_training_engine`（406-638）：签名删 `cond_fix=None, cond_fix_start=None, cond_fix_start_a1=None, cond_fix_start_a2=None`、`use_amp=False, amp_dtype='bfloat16'`；**保留 `extra_epochs_after_high_acc` 形参但默认值 200→0**（用户决定：完整保留接线，见决策表 §6）；删 AMP 块（416-427，`# Autocast for the forward pass...` 到 `print(f"[Config] USE_AMP enabled...`）；删 `frozen_param_states = None` / `cond_fix_triggered = False`（452-453）与 cond_fix_start==0 预冻块（460-466）；删 eval 后的 cond_fix 触发块（493-521，`# Conditional freeze:...` 到 `print(f"  per-rule acc: {accs}")`）；删两处 `with amp_ctx():`（473、488），内容提一级；删 `from contextlib import nullcontext`（15 行）；556-561 行的 tag 偏移：
```python
            offset = 0 if getattr(model, 'use_ab_tag', False) else 1
```
改为 `offset = 1`（tag 模式已删，目标 t 恒预测 x_{t+1}），并删 559 行注释。
(f) 454-458 行的早停注释块更新：删 "EARLY_STOP_EXTRA_EPOCHS / high_acc_epoch are legacy..." 句，改为说明 high_acc_epoch 仅为 resume checkpoint 兼容保留。
(g) 保留 `_sample_seq` 到 Task 7（final_eval 还在用它）。

- [ ] **Step 3: focused tests**

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_checkpoint_atomic.py
C:/Users/Chen/anaconda3/envs/torch/python.exe -c "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'tests'); import test_chain_run as t; t.test_load_partial_checkpoint(); print('PASS partial')"
C:/Users/Chen/anaconda3/envs/torch/python.exe -c "import sys; sys.path.insert(0, 'src'); import training; print('training imports OK')"
```

预期：全 PASS（test_checkpoint_atomic 只测 `_atomic_torch_save`，不受引擎改动影响）。

- [ ] **Step 4: commit**

```bash
git add src/training.py tests/test_chain_run.py
git commit -m "training: remove AMP/GradScaler, COND_FIX/freeze_partial"
```

## Task 6: datasets.py——删 MISS_SECOND / mask 污损口径 / use_ab_tag / missing_token_id；BatchTag 收缩

**Files:**
- Modify: `src/datasets.py`
- Modify: `tests/test_missing_values.py`（大改，见下）
- Modify: `tests/test_mixed_dataset.py`（删 tag 用例 :38-44 及 `__main__` 调用行）
- Modify: `tests/test_mixed_ab_compat.py`（删 tag 用例）

**Interfaces:**
- Produces（签名冻结）:
  ```python
  class BatchTag(IntEnum):
      MIXED_AB = 0; LOSS_MASK = 1; PLAIN = 2
      PLAIN_TARGET = 4; MIXED_AB_TARGET = 5; ACTION_MISS = 6
      # MIXED_AB_MASKED(=3) 删除；其余值不重编号
  def make_missing_collate(*, p, init_len, missing_prob, miss_len=1, missing_token=None)
      # 恒 predict 口径 -> (views, PLAIN_TARGET, cleans)
  def make_mixed_missing_collate(*, p, order, n_rules, missing_prob, miss_len=1)
      # 恒 predict 口径、恒无 tag -> (views, MIXED_AB_TARGET, labels, cleans)
  class MixedRecurrenceDataset:  # __init__(self, rules, num_samples=1000, length=10, verbose=True)
  def corrupt_window(window, is_train, *, p, init_len, missing_prob, miss_len=1, missing_token=None)
  class RecurrenceDataset:  # __init__ 删 miss_second / predict_missing 两参数，其余不动
  # 删除：missing_token_id
  ```
- 中间状态提醒：experiment.py / analyze_attention.py / verify_sample.py / error_breakdown.py / rule_fit.py / eval_clean_action.py 仍按旧签名调用，Task 7/9 修复；本 task 只跑下面列出的 focused tests。
- RNG 约束：`_corrupt` 只删 `if self.miss_second:` 预置块（datasets.py:317-324，不消耗 RNG）和 `max_run = self.miss_len if self.miss_second else 0` 改 `= 0`；随机扫描部分一个字符不动。

- [ ] **Step 1: 改写 tests/test_missing_values.py（确认失败）**

具体改动清单（未列出的用例不动）：

(a) `_collate` helper（:37-43）改为：

```python
def _collate(missing_prob=0.5, p=P, init_len=INIT_LEN):
    return make_missing_collate(p=p, init_len=init_len, missing_prob=missing_prob)
```

(b) `test_collate_corrupts_and_masks`（:56-75）整函数替换为 predict 口径断言：

```python
def test_collate_corrupts_predict_mode():
    """predict 口径（唯一保留的模式）：返回 (views, PLAIN_TARGET, cleans)，
    污损位以干净值为目标；初始值永不污损。"""
    p, init_len = 7, 2
    ds = _make_ds(p=p, init_len=init_len)
    collate = _collate(0.5, p=p, init_len=init_len)
    views, tag, cleans = collate(ds.train_samples[:8])
    assert tag == BatchTag.PLAIN_TARGET
    assert views.shape == (8, ds.length) and cleans.shape == (8, ds.length)
    saw = False
    for view, clean in zip(views, cleans):
        assert (view[:init_len] < p).all()
        assert (clean < p).all()
        for pos in range(init_len, ds.length):
            if view[pos].item() == p:
                saw = True
            else:
                assert view[pos].item() == clean[pos].item()
    assert saw
```

(c) `test_full_corruption_when_prob_one`（:88-102）：删 mask 相关断言与 `expected_mask`，只保留 view 结构断言（prob=1.0、miss_len=1 时 pos 3,5,7 污损、4,6 干净）；解包改为 `views, _, cleans = collate(...)`。
(d) `test_miss_len_run_structure`（:129-142）：解包改 `views, _, cleans`；删 mask 断言（139-142 行的 for 循环），保留 `_check_runs`。
(e) 整删 `test_miss_second_forced_run_deterministic`（:145-156）。
(f) 整删 `test_first_task_weight_passthrough`（:159-168）——first_task_weight 不再经 collate；替换为对训练侧默认 mask 的直接测试：

```python
def test_default_loss_mask_first_task_weight():
    from training import _default_loss_mask
    mask = _default_loss_mask(4, 7, num_mask=2, first_task_weight=30.0)
    assert mask.shape == (4, 7)
    assert (mask[:, :2] == 0).all() and mask[0, 2].item() == 30.0
    assert (mask[:, 3:] == 1).all()
```

(g) `test_predict_missing_returns_clean_targets`（:172-191）：`_collate(0.5, p=p, predict_missing=True)` 改 `_collate(0.5, p=p)`，其余不动（现在只剩 predict 模式，用例名保留）。
(h) `_mixed_collate` helper（:205-210）改为：

```python
def _mixed_collate(missing_prob=0.5, p=7, order=2, n_rules=2):
    return make_mixed_missing_collate(p=p, order=order, n_rules=n_rules,
                                      missing_prob=missing_prob)
```

(i) `test_mixed_missing_alignment`（:221-240）整函数替换：

```python
def test_mixed_missing_alignment():
    from core import _unpack_batch
    p = 7  # basic 模式（唯一模式）：missing token id == p
    ds = _make_mixed(use_ab_tag=False, p=p)
    collate = _mixed_collate(0.5, p=p)
    views, tag, labels, cleans = collate(ds.train_data[:8])
    assert tag == BatchTag.MIXED_AB_TARGET and labels.shape == (8,)
    saw = False
    for (clean, _), view in zip(ds.train_data[:8], views):
        assert (view[:ds.order] < p).all()
        for pos in range(ds.order, ds.length):
            if view[pos].item() == p:
                saw = True
            else:
                assert view[pos].item() == clean[pos].item()
    assert saw
    x, loss_mask, kwargs, ab_labels, t_override = _unpack_batch(
        (views, tag, labels, cleans), 'cpu', None)
    assert loss_mask is None and ab_labels is not None and torch.equal(t_override, cleans)
```

(j) 整删 `test_mixed_tag_missing_alignment`（:243-259）与 `test_predict_missing_mixed_tag_mode`（:270-292）（tag 模式删除）。
(k) 整删 `test_post_train_init_state_extraction`（:295-304）（其服务的 final generation test 已删；`_sample_seq` 在 Task 7 删除）。
(l) `test_mixed_fresh_randomness_per_call` / `test_mixed_disabled_keeps_pairs` / `test_dataset_always_stores_clean_windows` / `test_fresh_randomness_per_call` 不动。
(m) 文件 docstring（:1-11）更新：去掉 MISS_SECOND / mask 口径描述，说明 predict 是唯一口径。

跑：`C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_missing_values.py`
预期：失败（make_missing_collate 签名已变）。

- [ ] **Step 2: 改 test_mixed_dataset.py 与 test_mixed_ab_compat.py（确认失败）**

(a) `tests/test_mixed_dataset.py`：整删 `test_use_ab_tag_prepends_flag_token`（:38-44）及 `__main__` 里的 `test_use_ab_tag_prepends_flag_token()` 调用行（:101 附近）。
(b) `tests/test_mixed_ab_compat.py`：`_run_case`（:133-141）删 `use_ab_tag` 参数（新旧构造都不再传）；删 `test_compat_two_rules_with_tag`（:149-151）；`test_compat_three_rules_scalar_num_samples`（:153-156）的 `use_ab_tag=True` 改 `False`；`__main__` 调用清单删 `test_compat_two_rules_with_tag()`。golden master `MixedABDataset` 类一行不动（注释已声明其冻结地位）。

跑两个文件确认失败（MixedRecurrenceDataset 仍有 use_ab_tag 参数时 (b) 可能碰巧通过——以 datasets.py 改完后必须通过为准）。

- [ ] **Step 3: datasets.py 修改（精确锚点）**

(a) `RecurrenceDataset.__init__`（:18-57）：签名删 `miss_second=False, predict_missing=False`；删 `self.miss_second = ...`（:42）与 `self.predict_missing = ...` 及其注释块（:43-48）。docstring/注释里提到 miss_second/predict_missing 的地方同步删。
(b) `run()` 的 verbose 打印（:281-284）：删 `miss_second={self.miss_second}, ` 片段。
(c) `_corrupt`（:289-352）：删 miss_second 预置块（:317-324，`if self.miss_second:` 到 `pos = max(pos, end + 1)`）；`max_run = self.miss_len if self.miss_second else 0`（:331）改 `max_run = 0`；docstring 删 "when miss_second is set..." 整句及 redo 说明里的 miss_second 括号注。**其余扫描代码零改动**。
(d) `missing_token_id`（:363-373）整函数删除。
(e) `corrupt_window`（:376-389）：签名删 `miss_second=False`；构造 helper 时不传 miss_second。
(f) `BatchTag`（:448-455）：删 `MIXED_AB_MASKED = 3` 行，其余成员与值不动；类上方注释（:443-447）不变。
(g) `_corrupt_helper`（:463-469）：删 `miss_second` 参数与透传。
(h) `make_missing_collate`（:472-499）：签名改为 `(*, p, init_len, missing_prob, miss_len=1, missing_token=None)`；函数体改为：

```python
    """Collate with ON-THE-FLY missing-value corruption (predict regime).

    Input: a batch of CLEAN windows (plain LongTensors). Every call corrupts
    with fresh randomness. Always predict mode: (views, PLAIN_TARGET, cleans)
    — the loss/accuracy targets are the clean values, so corrupted positions
    must be filled in. num_mask/first_task_weight are applied by the training
    loop's _default_loss_mask, not here.
    """
    token = p if missing_token is None else missing_token

    def collate(batch):
        helper = _corrupt_helper(p, init_len, len(batch[0]), missing_prob,
                                 miss_len, missing_token=token)
        views, cleans = [], []
        for seq in batch:
            view = seq.clone()
            helper._corrupt(view, True)
            views.append(view)
            cleans.append(seq)
        return torch.stack(views), BatchTag.PLAIN_TARGET, torch.stack(cleans)

    return collate
```

(i) `make_mixed_missing_collate`（:511-557）：签名改为 `(*, p, order, n_rules, missing_prob, miss_len=1)`；函数体替换为：

```python
    """Mixed-rule on-the-fly corruption (predict regime): (clean_seq, rule_idx)
    batches -> (views, MIXED_AB_TARGET, labels, cleans). Tag mode is gone, so
    the missing token is always plain p."""
    token = p

    def collate(batch):
        seqs = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        helper = _corrupt_helper(p, order, len(seqs[0]), missing_prob, miss_len,
                                 missing_token=token)
        views, cleans = [], []
        for seq in seqs:
            view = seq.clone()
            helper._corrupt(view, True)
            views.append(view)
            cleans.append(seq)
        return (torch.stack(views), BatchTag.MIXED_AB_TARGET, labels,
                torch.stack(cleans))

    return collate
```

注意：`_corrupt` 需要 num_mask/first_task_weight 参数构造内部 mask——mask 在 predict 口径下被丢弃，传默认即可（`_corrupt_helper` 保留 num_mask/first_task_weight 参数位但调用处不再传）。同时更新 `_corrupt_helper` 签名保持 `(*, p, init_len, length, missing_prob, miss_len, num_mask=0, first_task_weight=1.0, missing_token)` 风格（位置参数维持现有顺序，仅删 miss_second）。
(j) `MixedRecurrenceDataset.__init__`（:733-800）：签名删 `use_ab_tag=False`；删 `self.use_ab_tag = ...`（:746）、tag 前置块（:774-779，`# Prepend rule token if use_ab_tag...` 整段）与类 docstring 里 use_ab_tag 相关句子（:724-731 段改为只描述 basic 布局与 collate 选择）。**shuffle 与 per-rule 生成顺序零改动（golden master）。**

- [ ] **Step 4: focused tests**

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_missing_values.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_mixed_dataset.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_mixed_ab_compat.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_data_generation_modes.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_state_cap.py
PYTHONPATH=src C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_sample_gen.py
```

预期：全部 PASS（test_mixed_ab_compat 证明 RNG 调用序未变）。

- [ ] **Step 5: commit**

```bash
git add src/datasets.py tests/test_missing_values.py tests/test_mixed_dataset.py tests/test_mixed_ab_compat.py
git commit -m "datasets: predict-only corruption, drop MISS_SECOND/tag mode/missing_token_id"
```

## Task 7: experiment.py 归一化路由 + 删 final generation test / final_eval.py + core.py

**Files:**
- Modify: `src/experiment.py`
- Delete: `src/final_eval.py`
- Modify: `src/core.py`
- Modify: `src/training.py`（删 `_sample_seq`，:123-131）
- Modify: `tests/test_prepare_mixed_recurrence.py`、`tests/test_action_trib.py`（test_prepare_wiring:92-119）、`tests/test_action_missing.py`（test_run_experiment_action_missing_smoke:108-150）、`tests/test_training_smoke.py`

**Interfaces:**
- Consumes: `scenarios.normalize_task_config`（Task 1-2）、`rules.rules_from_config(rules_raw, p)` / `save_config_extra(norm, cfg)` / `single_rule_from_task('single', ...)`（Task 3）、新 models/datasets/training 签名（Task 4-6）。
- Produces（签名冻结，后续 task 与测试依赖）:
  ```python
  def _prepare_single_recurrence(config, norm)  # norm['scenario']=='single' 或例外 task 名
  def _prepare_mixed_recurrence(config, device, norm)
  def _prepare_action(config, norm)
  # ctx 公共键：model, train_dataset, test_dataset, train_loader, test_loader,
  #   test_loader_fn, num_mask, save_config, cfg, p, train_len, ood_len
  # mixed 额外：rules, order。删除：post_train_mode, extra_kwargs_fn,
  #   recurrence_fn, init_len, recurrence_name, data_mode, num_test_samples
  ```
- 本 task 结束后全仓库不再有 MixedABTransformer / final_eval 引用，端到端测试恢复绿色。

- [ ] **Step 1: 改四个测试文件（确认失败）**

(a) `tests/test_training_smoke.py`：`_check_common`（:43-55）删两行——`assert 'Final generation test' in out...` 与 `assert 'Exposed' in out and 'Unexposed' in out...`，替换为 `assert 'Final generation test' not in out`；`test_run_experiment_mixed_ab`（:66-71）的 `assert 'Rule 1' in out and 'Rule 2' in out` 改为 `assert 'per-rule acc' in out`；文件 docstring 删 "and the stage-3 final generation test output"。
(b) `tests/test_prepare_mixed_recurrence.py` 全文替换为：

```python
"""Wiring tests for experiment._prepare_mixed_recurrence（mixed 场景，任意 order）。"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

import core
from scenarios import normalize_task_config


def _base_main():
    return {
        'P': 5, 'D_MODEL': 16, 'N_HEAD': 1, 'N_LAYER': 1, 'BATCH_SIZE': 16,
        'TRAIN_LEN': 8, 'OOD_LEN': 10, 'DROPOUT': 0.0,
    }


def _prepare(main, task):
    return core._prepare_mixed_recurrence({'main': main}, 'cpu',
                                          normalize_task_config(task, main))


def test_prepare_mixed_ab_wiring():
    main = {**_base_main(), 'AB_PAIRS': [[1, 1], [1, 2]], 'MAX_UNIQUE_RATIO': 0.5}
    random.seed(0)
    torch.manual_seed(0)
    ctx = _prepare(main, 'mixed_ab')
    assert ctx['order'] == 2
    assert [r.coeffs for r in ctx['rules']] == [(1, 1), (1, 2)]
    # state space 5**2 = 25 per rule; ratio 0.5 -> 12 train samples per rule
    assert len(ctx['train_dataset']) == 24
    assert len(ctx['test_dataset']) == 2 * 25 - 24
    assert ctx['model'].vocab_size == 5 + 1  # 词表恒 P+1
    assert ctx['model'].pad_token_id == 5
    assert ctx['num_mask'] == 1  # 新默认 order-1
    assert ctx['save_config']['scenario'] == 'mixed'
    assert ctx['save_config']['order'] == 2
    assert ctx['save_config']['rules'] == [[1, 1], [1, 2]]
    assert ctx['save_config']['ab_pairs'] == [[1, 1], [1, 2]]
    assert type(ctx['model']).__name__ == 'FibonacciTransformer'


def test_prepare_mixed_abc_wiring():
    main = {**_base_main(), 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]],
            'MIXED_AB_MAX_UNIQUE_RATIOS': [0.4, 0.4]}
    random.seed(0)
    torch.manual_seed(0)
    ctx = _prepare(main, 'mixed_abc')
    assert ctx['order'] == 3
    assert [r.coeffs for r in ctx['rules']] == [(1, 1, 1), (1, 2, 3)]
    # state space 5**3 = 125 per rule; ratio 0.4 -> 50 train samples per rule
    assert len(ctx['train_dataset']) == 100
    assert len(ctx['test_dataset']) == 2 * 125 - 100
    assert ctx['model'].vocab_size == 5 + 1
    assert ctx['num_mask'] == 2  # order-1


def test_prepare_new_format_equals_legacy():
    # 新写法（TASK=mixed + ORDER + RULES + EXPOSURE_RATIOS）与旧写法同 seed 数据集一致
    old_main = {**_base_main(), 'AB_PAIRS': [[1, 1], [1, 2]], 'MAX_UNIQUE_RATIO': 0.5}
    new_main = {**_base_main(), 'ORDER': 2, 'RULES': [[1, 1], [1, 2]],
                'EXPOSURE_RATIOS': [0.5, 0.5]}
    random.seed(0)
    torch.manual_seed(0)
    old = _prepare(old_main, 'mixed_ab')
    random.seed(0)
    torch.manual_seed(0)
    new = _prepare(new_main, 'mixed')
    assert len(old['train_dataset']) == len(new['train_dataset'])
    for (so, lo), (sn, ln) in zip(old['train_dataset'], new['train_dataset']):
        assert lo == ln and so.tolist() == sn.tolist()


def test_prepare_num_train_samples_override():
    main = {**_base_main(), 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]], 'NUM_TRAIN_SAMPLES': 10}
    random.seed(0)
    torch.manual_seed(0)
    ctx = _prepare(main, 'mixed_abc')
    assert len(ctx['train_dataset']) == 20
    main = {**_base_main(), 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]],
            'NUM_TRAIN_SAMPLES': [10, 30]}
    random.seed(0)
    torch.manual_seed(0)
    ctx = _prepare(main, 'mixed_abc')
    assert len(ctx['train_dataset']) == 40
    labels = [idx for _, idx in ctx['train_dataset']]
    assert labels.count(0) == 10 and labels.count(1) == 30
    main = {**_base_main(), 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]],
            'NUM_TRAIN_SAMPLES': [10]}
    random.seed(0)
    torch.manual_seed(0)
    try:
        _prepare(main, 'mixed_abc')
        assert False, "NUM_TRAIN_SAMPLES/rules length mismatch must raise"
    except AssertionError:
        pass


def test_ratios_length_mismatch_raises():
    main = {**_base_main(), 'AB_PAIRS': [[1, 1], [1, 2]],
            'MIXED_AB_MAX_UNIQUE_RATIOS': [0.5]}
    try:
        _prepare(main, 'mixed_ab')
        assert False, "ratios/rules length mismatch must raise"
    except ValueError:
        pass  # 长度校验已上移到归一化层（ValueError）


if __name__ == '__main__':
    test_prepare_mixed_ab_wiring()
    test_prepare_mixed_abc_wiring()
    test_prepare_new_format_equals_legacy()
    test_prepare_num_train_samples_override()
    test_ratios_length_mismatch_raises()
    print("ALL TESTS PASSED: test_prepare_mixed_recurrence.py")
```

注意：此文件同时修复了基线已红的 num_mask 断言（旧断言 2 是 tag 默认残留；新默认 order-1）。旧键 MAX_UNIQUE_RATIO 标量在 mixed 下广播为逐规则（Task 2 的 `_legacy_exposure`），新写法 EXPOSURE_RATIOS 列表——两路在 `_prepare_mixed_recurrence` 内都应展开成同样的逐规则样本数。

(c) `tests/test_action_trib.py` 的 `test_prepare_wiring`（:92-119）整函数替换：

```python
def test_prepare_wiring():
    from scenarios import normalize_task_config
    main = {
        'P': P, 'D_MODEL': 16, 'N_HEAD': 1, 'N_LAYER': 1, 'BATCH_SIZE': 16,
        'DROPOUT': 0.0, 'ABC_PAIRS': [list(p) for p in PAIRS],
        'NUM_TRAIN_SAMPLES': 20, 'NUM_TEST_SAMPLES': 10,
        'TRAIN_LEN': 8, 'OOD_LEN': 10,
    }
    cfg = {'main': main}
    random.seed(0)
    torch.manual_seed(0)
    ctx = _prepare_action(cfg, normalize_task_config('action_trib', main))
    assert ctx['model'].vocab_size == P + 1 + len(PAIRS)
    assert ctx['model'].pad_token_id == P
    assert ctx['save_config']['order'] == 3
    assert ctx['save_config']['scenario'] == 'action'
    # canonical（第一个系数乘最新值）= action 旧约定逐条 reverse
    assert ctx['save_config']['rules'] == [[1, 1, 1], [3, 2, 1]]
    assert ctx['save_config']['ab_pairs'] == [list(p) for p in PAIRS]  # action 约定照写
    # block_size covers 2*OOD_LEN - 3 = 17 -> 32
    assert ctx['save_config']['block_size'] == 32
    seq, mask = ctx['train_dataset'][0]
    assert len(seq) == 2 * 8 - 3
    assert len(ctx['train_dataset']) == 20
    assert len(ctx['test_dataset']) == 10

    # 系数条数与 order 不符必须报（校验在归一化层 _legacy_rules 的尾部统一
    # 长度检查，Task 2 已实现）
    try:
        normalize_task_config('action_trib', {'P': P, 'ABC_PAIRS': [[1, 1], [1, 2]]})
        assert False, "2-coefficient rules under order-3 must raise"
    except ValueError:
        pass
```

`tests/test_action_trib.py:18` 的 `from experiment import _prepare_action` 保留。

(d) `tests/test_action_missing.py` 的 `test_run_experiment_action_missing_smoke`（:108-150）里：

```python
    from experiment import _prepare_action
    prep = _prepare_action({'main': dict(main), '_BATCH_RUN_MERGED': True})
```

改为：

```python
    from experiment import _prepare_action
    from scenarios import normalize_task_config
    prep = _prepare_action({'main': dict(main), '_BATCH_RUN_MERGED': True},
                           normalize_task_config(main['TASK'], main))
```

并新增断言 `assert prep['save_config']['scenario'] == 'action'`。main dict 里的 `'ENTROPY_PENALTY_WEIGHT': 0.0` 和 `'USE_LEARNABLE_PE': False` 两行删除（键已写死，测试不再传）。

跑这四个文件确认失败（experiment.py 尚未适配）。

- [ ] **Step 2: experiment.py 改写（精确锚点 + 完整新代码）**

(a) import 块（:19-29）替换为：

```python
from rules import rules_from_config, single_rule_from_task, save_config_extra
from protocol import BATCH_RUN_MERGED_FLAG
from scenarios import normalize_task_config
from models import FibonacciTransformer
from datasets import (RecurrenceDataset, MixedRecurrenceDataset,
                      ActionDataset, BucketBatchSampler,
                      collate_fn, make_missing_collate,
                      mixed_ab_collate_fn, make_mixed_missing_collate,
                      action_collate_fn, make_action_missing_collate)
from training import run_training_engine, resume_checkpoint_path
```

模块 docstring（:1-9）更新：删 final_eval 提及，说明 run_experiment = parse config -> normalize -> scenario 分派 -> run_training_engine（stage-3 已删）。

(b) `_prepare_mixed_recurrence`（:111-274）整函数替换为：

```python
def _prepare_mixed_recurrence(config, device, norm):
    """Prepare dataset, model, loaders and training params for the mixed
    scenario (any order >= 2; canonical rules from the normalized config).

    Model is always FibonacciTransformer, vocab always P+1 (tag mode,
    cond_wte and rule_head were removed 2026-10-08). NUM_MASK default is
    order-1. Old spellings (mixed_ab/mixed_abc + AB_PAIRS/ABC_PAIRS +
    MAX_UNIQUE_RATIO/MIXED_AB_MAX_UNIQUE_RATIOS) arrive already translated
    by scenarios.normalize_task_config.
    """
    cfg = dict(config.get('main', {}))
    P = cfg['P']
    D_MODEL = cfg['D_MODEL']
    N_HEAD = cfg['N_HEAD']
    N_LAYER = cfg['N_LAYER']
    BATCH_SIZE = cfg['BATCH_SIZE']
    TRAIN_LEN = cfg['TRAIN_LEN']
    OOD_LEN = cfg['OOD_LEN']
    DROPOUT = cfg['DROPOUT']
    order = norm['order']
    rules = rules_from_config(norm['rules'], P)
    state_space_size = P ** order

    BLOCK_SIZE = _round_up_pow2(max(TRAIN_LEN, OOD_LEN))

    # Explicit NUM_TRAIN_SAMPLES overrides the ratio-derived counts: a scalar
    # gives every rule the same count, a list sets per-rule counts.
    nts = cfg.get('NUM_TRAIN_SAMPLES')
    if nts is not None:
        if isinstance(nts, (int, float)):
            nts = [nts] * len(rules)
        assert len(nts) == len(rules), \
            f"NUM_TRAIN_SAMPLES length ({len(nts)}) must equal number of rules ({len(rules)})"
        NUM_TRAIN_SAMPLES = [max(1, int(n)) for n in nts]
        # Keep `ratios` defined for save_config: actual exposure per rule.
        ratios = [n / state_space_size for n in NUM_TRAIN_SAMPLES]
    else:
        ratios = norm['exposure_ratios']
        NUM_TRAIN_SAMPLES = [max(1, int(state_space_size * r)) for r in ratios]

    # NUM_MASK unset (None) means: the first num_mask targets are initial
    # values and are not evaluated; default order-1.
    num_mask_cfg = cfg.get('NUM_MASK')
    num_mask = (order - 1) if num_mask_cfg is None else num_mask_cfg
    # An all-zero loss mask yields a grad-less constant loss and crashes
    # backward() far from the cause; require at least one evaluated position.
    assert num_mask < TRAIN_LEN - 1, \
        f"NUM_MASK ({num_mask}) must be < TRAIN_LEN - 1 ({TRAIN_LEN - 1})"

    missing_prob = cfg.get('MISSING_PROB', 0.0)
    ds = MixedRecurrenceDataset(rules=rules, num_samples=NUM_TRAIN_SAMPLES,
                                length=TRAIN_LEN, verbose=True)
    train_dataset = ds.train_data
    test_dataset = ds.test_data

    _print_task_banner(BLOCK_SIZE, TRAIN_LEN, f"Rules: {[r.name for r in rules]}")

    # Explicit collate selection: with MISSING_PROB, corruption is applied
    # on the fly in the collate (fresh randomness per batch, predict regime).
    if missing_prob > 0:
        mixed_collate = make_mixed_missing_collate(
            p=P, order=order, n_rules=len(rules),
            missing_prob=missing_prob, miss_len=cfg.get('MISS_LEN', 1))
    else:
        mixed_collate = mixed_ab_collate_fn           # (seq, label) -> MIXED_AB
    train_loader, test_loader = _make_loaders(
        train_dataset, test_dataset, BATCH_SIZE, mixed_collate,
        reshuffle_each_epoch=cfg.get('RESHUFFLE_EACH_EPOCH', False))

    # FRESH_TEST_PER_EVAL (mixed): at every eval, sample NUM_TEST_SAMPLES fresh
    # initial states PER RULE from each rule's full state space and roll out
    # OOD_LEN windows (num_samples=0 sends them all to the test split).
    test_loader_fn = None
    if cfg.get('FRESH_TEST_PER_EVAL', False):
        NUM_TEST_SAMPLES = cfg.get('NUM_TEST_SAMPLES', 256)
        fresh_rng = random.Random(cfg.get('RANDOM_SEED', 42) + 10 ** 6 + 7)

        def test_loader_fn():  # noqa: F811 (intentional closure name)
            fresh_seed = fresh_rng.randrange(2 ** 31)
            rng_state = random.getstate()
            random.seed(fresh_seed)
            fresh_samples, fresh_labels = [], []
            for idx, rule in enumerate(rules):
                fresh_ds = RecurrenceDataset(
                    p=P, recurrence_fn=rule.next_fn(), recurrence_name=rule.name,
                    init_len=rule.order, num_samples=0, length=OOD_LEN,
                    verbose=False, state_cap=NUM_TEST_SAMPLES)
                fresh_ds.run()
                fresh_samples += fresh_ds.test_samples
                fresh_labels += [idx] * len(fresh_ds.test_samples)
            random.setstate(rng_state)
            print(f"[FreshTest] rebuilt mixed test set: n={NUM_TEST_SAMPLES}/rule x {len(rules)}, len={OOD_LEN}, seed={fresh_seed}")
            return _make_test_loader(list(zip(fresh_samples, fresh_labels)),
                                     BATCH_SIZE, mixed_collate)

    model = FibonacciTransformer(p=P, d_model=D_MODEL, n_head=N_HEAD,
                                 n_layer=N_LAYER, block_size=BLOCK_SIZE,
                                 dropout=DROPOUT, mlp_ratio=cfg.get('MLP_RATIO', 4))
    save_config = {
        'p': P,
        'scenario': 'mixed',
        'order': order,
        'rules': [list(r.coeffs) for r in rules],
        # canonical == legacy mixed convention, safe for old analysis tools
        'ab_pairs': [list(r.coeffs) for r in rules],
        'exposure_ratios': ratios,
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'mlp_ratio': cfg.get('MLP_RATIO', 4),
        'vocab_size': P + 1,
        'pad_token_id': P,
        # for analyze_attention.py: in-dist length and corruption settings
        'train_len': TRAIN_LEN,
        'missing_prob': cfg.get('MISSING_PROB', 0.0),
        'miss_len': cfg.get('MISS_LEN', 1),
    }
    return {
        'model': model,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'test_loader_fn': test_loader_fn,
        'num_mask': num_mask,
        'save_config': save_config,
        'cfg': cfg,
        'p': P,
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
        'rules': rules,
        'order': order,
    }
```

(c) `_prepare_action`（:277-411）整函数替换为：

```python
def _prepare_action(config, norm):
    """Prepare dataset, model, loaders and training params for the action
    scenario (rule can change at every step), any order >= 2.

    norm['rules'] is canonical (first coefficient multiplies the NEWEST
    value). The ACTION data/layout convention is the opposite (first
    coefficient multiplies the OLDEST value), so the rules are reversed ONCE
    here — the only other reversal site is scenarios._legacy_rules (legacy
    AB_PAIRS/ABC_PAIRS -> canonical). Do not add a third one.
    """
    # batch_run already merges the action section into main with the correct
    # precedence (main -> task defaults -> experiment override). Do NOT
    # re-apply the section here.
    cfg = dict(config.get('main', {}))

    order = norm['order']
    AB_PAIRS = [tuple(reversed(r)) for r in norm['rules']]  # canonical -> action
    missing_prob = cfg.get('MISSING_PROB', 0.0)
    miss_len = cfg.get('MISS_LEN', 1)
    P = cfg['P']
    D_MODEL = cfg['D_MODEL']
    N_HEAD = cfg['N_HEAD']
    N_LAYER = cfg['N_LAYER']
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROPOUT = cfg['DROPOUT']
    NUM_TRAIN_SAMPLES = cfg.get('NUM_TRAIN_SAMPLES', 10000)
    NUM_TEST_SAMPLES = cfg.get('NUM_TEST_SAMPLES', 2000)  # default matches src/config.json
    TRAIN_LEN = cfg.get('TRAIN_LEN', 16)
    OOD_LEN = cfg.get('OOD_LEN', 16)

    # In action, each generated token is preceded by a flag token,
    # so the actual sequence length is 2*length - order.
    max_seq_len = 2 * max(TRAIN_LEN, OOD_LEN) - order
    BLOCK_SIZE = _round_up_pow2(max_seq_len)

    train_dataset = ActionDataset(
        p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TRAIN_SAMPLES,
        length=TRAIN_LEN, seed=cfg.get('RANDOM_SEED', 42), order=order
    )
    test_dataset = ActionDataset(
        p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TEST_SAMPLES,
        length=OOD_LEN, seed=cfg.get('RANDOM_SEED', 42) + 1, order=order
    )

    _print_task_banner(BLOCK_SIZE, TRAIN_LEN, f"Dynamic mixed rules: {AB_PAIRS}")

    # With MISSING_PROB, corruption is applied on the fly in the collate
    # (fresh randomness per batch; datasets store clean samples only).
    collate = (make_action_missing_collate(p=P, missing_prob=missing_prob,
                                           miss_len=miss_len, order=order)
               if missing_prob > 0 else action_collate_fn)
    train_loader, test_loader = _make_loaders(
        train_dataset, test_dataset, BATCH_SIZE, collate,
        reshuffle_each_epoch=cfg.get('RESHUFFLE_EACH_EPOCH', False))

    # FRESH_TEST_PER_EVAL: rebuild the test set at every eval with a fresh seed.
    test_loader_fn = None
    if cfg.get('FRESH_TEST_PER_EVAL', False):
        fresh_rng = random.Random(cfg.get('RANDOM_SEED', 42) + 10 ** 6 + 7)

        def test_loader_fn():  # noqa: F811 (intentional closure name)
            fresh_seed = fresh_rng.randrange(2 ** 31)
            ds = ActionDataset(
                p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TEST_SAMPLES,
                length=OOD_LEN, seed=fresh_seed, order=order)
            print(f"[FreshTest] rebuilt test set: n={NUM_TEST_SAMPLES}, len={OOD_LEN}, seed={fresh_seed}")
            return _make_test_loader(ds, BATCH_SIZE, collate)

    model = FibonacciTransformer(
        p=P, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER,
        block_size=BLOCK_SIZE, dropout=DROPOUT,
        mlp_ratio=cfg.get('MLP_RATIO', 4)
    )
    # vocab_size needs to include flag tokens
    model.vocab_size = P + 1 + len(AB_PAIRS)
    model.pad_token_id = P
    # Expand lm_head and transformer.wte to accommodate flags while preserving weight tying
    if model.lm_head.weight.size(0) < model.vocab_size:
        with torch.no_grad():
            old_head = model.lm_head
            old_wte = model.transformer.wte
            new_wte = nn.Embedding(model.vocab_size, old_wte.embedding_dim)
            new_wte.weight.data[:old_wte.weight.size(0)] = old_wte.weight.data
            model.transformer.wte = new_wte
            # Tie lm_head to wte, matching FibonacciTransformer's original design
            model.lm_head = nn.Linear(old_head.in_features, model.vocab_size, bias=False)
            model.lm_head.weight = model.transformer.wte.weight

    save_config = {
        'p': P,
        'scenario': 'action',
        'order': order,
        'rules': [list(r) for r in norm['rules']],  # canonical
        'ab_pairs': [list(pair) for pair in AB_PAIRS],  # action 旧约定，供旧分析工具
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'mlp_ratio': cfg.get('MLP_RATIO', 4),
        'vocab_size': model.vocab_size,
        'pad_token_id': model.pad_token_id,
        'recurrence': 'action',
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
        'missing_prob': missing_prob,
        'miss_len': miss_len,
    }
    return {
        'model': model,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'test_loader_fn': test_loader_fn,
        'num_mask': 0,  # unused; loss_mask comes from the dataset
        'save_config': save_config,
        'cfg': cfg,
        'p': P,
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
    }
```

注意 OOD_LEN 默认从 32 改 16（规格 §1.1）；config.json 的 action 段仍为旧 JSON 提供 32（Task 8）。

(d) `_prepare_single_recurrence`（:414-576）改签名与头部；数据模式/num_mask/数据集/loader 逻辑原样保留，差异如下。函数开头到 `save_extra_config` 替换为：

```python
def _prepare_single_recurrence(config, norm):
    """Prepare dataset, model, loaders and training params for the single
    scenario (linear, any order via LinearRecurrenceRule) and the nonlinear
    exception tasks (hardcoded recurrences)."""
    cfg = dict(config.get('main', {}))
    P = cfg['P']
    D_MODEL = cfg['D_MODEL']
    N_HEAD = cfg['N_HEAD']
    N_LAYER = cfg['N_LAYER']
    BATCH_SIZE = cfg['BATCH_SIZE']
    TRAIN_LEN = cfg['TRAIN_LEN']
    OOD_LEN = cfg['OOD_LEN']
    DROPOUT = cfg['DROPOUT']
    scenario = norm['scenario']

    BLOCK_SIZE = _round_up_pow2(max(TRAIN_LEN, OOD_LEN))

    if scenario == 'single':
        init_len, recurrence_fn, recurrence_name = single_rule_from_task(
            'single', {'P': P, 'RULES': norm['rules']})
        default_num_mask = norm['order'] - 1
        max_unique_ratio = norm['exposure_ratios']
    else:
        # Nonlinear exception tasks keep their hardcoded recurrences.
        init_len, recurrence_fn, recurrence_name = single_rule_from_task(scenario, cfg)
        default_num_mask = {'multiplication': 1, 'nonlinear': 1,
                            'nonlinear_mul': 1}[scenario]
        max_unique_ratio = cfg.get('MAX_UNIQUE_RATIO', 0.7)
    save_extra_config = save_config_extra(norm, cfg)
```

其后：`MAX_UNIQUE_RATIO` 引用（原 :426 定义与 :458 使用）改为用 `max_unique_ratio`；删 `predict_missing = cfg.get('PREDICT_MISSING', False)`（:480）；collate 选择改为：

```python
    if missing_prob > 0:
        collate = make_missing_collate(
            p=P, init_len=init_len, missing_prob=missing_prob,
            miss_len=cfg.get('MISS_LEN', 1))
    else:
        collate = collate_fn              # plain tensors -> PLAIN
```

模型构造删 `entropy_penalty_weight=ENTROPY_PENALTY_WEIGHT, use_learnable_pe=USE_LEARNABLE_PE` 两实参（及函数头两个变量的读取行 :425/:427）；save_config 删 `'use_learnable_pe'`、`'miss_second'`、`'predict_missing'` 三行；ctx 删 `'post_train_mode'`、`'extra_kwargs_fn'`、`'recurrence_fn'`、`'init_len'`、`'recurrence_name'`、`'data_mode'`、`'num_test_samples'` 键。

(e) `run_experiment`（:579-731）：
- TF32 块（:601-606）替换为写死：

```python
    # TF32 恒开（2026-10-08 写死；对训练数值影响可忽略，L20 上 ~1.3x）
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[Config] TF32 enabled (matmul + cudnn)")
```

- Stage 1 分派（:622-634）替换为：

```python
    # ========================================================================
    # Stage 1: Scenario dispatch -- prepare dataset, model, loaders
    # ========================================================================
    try:
        norm = normalize_task_config(TASK, cfg_main)
    except ValueError as exc:
        print(f"[Error] {exc}")
        sys.exit(2)  # non-zero so batch_run records failure instead of a silent "success"
    scenario = norm['scenario']
    if scenario == 'mixed':
        ctx = _prepare_mixed_recurrence(config, device, norm)
    elif scenario == 'action':
        ctx = _prepare_action(config, norm)
    else:  # 'single' or a nonlinear exception task name
        ctx = _prepare_single_recurrence(config, norm)
```

- ctx 解包块（:636-649）删 `extra_kwargs_fn = ctx['extra_kwargs_fn']` 与 `post_train_mode = ctx['post_train_mode']` 两行。
- `run_training_engine(...)` 调用（:686-708）删这些 kwarg 行：`extra_kwargs_fn=extra_kwargs_fn,`、`cond_fix=...`/`cond_fix_start=...`/`cond_fix_start_a1=...`/`cond_fix_start_a2=...` 四行、`use_amp=...`/`amp_dtype=...` 两行（`extra_kwargs_fn` 形参在引擎侧保留默认 None，调用处直接不传）。`extra_epochs_after_high_acc=cfg.get('EARLY_STOP_EXTRA_EPOCHS', 200)` 行**保留**，仅把默认值改为 `0`（用户决定：完整保留接线、默认 0）。

  注意：training.py Task 5 保留 `extra_epochs_after_high_acc` 形参（默认改 0）；`extra_kwargs_fn` 形参保留（默认 None），调用处直接省略。
- Stage 3 整块（:717-731）删除，包括 `SKIP_FINAL_GENERATION_TEST` 提示、`_run_mixed_ab_final_test` / `_run_single_recurrence_final_test` 调用。`run_training_engine` 返回后函数自然结束。

(f) 删除 `src/final_eval.py`（`git rm`）。
(g) `src/training.py`：整删 `_sample_seq`（:123-131）及其在模块 docstring :5 的提及。
(h) `src/core.py` 全文替换为：

```python
"""Compatibility shell for the recurrence experiment entry point.

Implementation lives in models.py / datasets.py / training.py / scenarios.py /
experiment.py. This module re-exports only the symbols the test suite still
references via `from core import ...` / `core.<name>`, and keeps the __main__
entry that batch_run.py spawns as `python src/core.py <merged_config.json>`.
"""
import sys

from datasets import (RecurrenceDataset, BatchTag, collate_fn,
                      make_missing_collate, make_mixed_missing_collate,
                      make_action_missing_collate,
                      mixed_ab_collate_fn)
from training import _unpack_batch
from experiment import _prepare_mixed_recurrence, run_experiment


if __name__ == '__main__':
    # Entry point for the training subprocess spawned by batch_run.py:
    #   python src/core.py <merged_config.json>
    if len(sys.argv) != 2:
        print("Usage: python src/core.py <merged_config.json>", file=sys.stderr)
        sys.exit(2)
    run_experiment(sys.argv[1])
```

- [ ] **Step 3: 跑端到端测试组（本 task 的验收门）**

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_training_smoke.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_prepare_mixed_recurrence.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_action_trib.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_action_missing.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_early_stop.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_chain_run.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_train_timeout_resume.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_sampler_reshuffle.py
```

预期：全部 PASS。已知例外：`tests/test_fresh_test_eval.py` 基线已红（single 分支正则过时，Task 12 修）；其中 action/mixed 两个用例在本 task 后应恢复 PASS（可单独跑确认，不强制）。

- [ ] **Step 4: commit**

```bash
git add src/experiment.py src/core.py src/training.py tests/test_prepare_mixed_recurrence.py tests/test_action_trib.py tests/test_action_missing.py tests/test_training_smoke.py
git rm src/final_eval.py
git commit -m "experiment: scenario routing via normalize_task_config; drop final generation test"
```

## Task 8: batch_run.py + src/config.json——场景默认段 / 别名挪移 / 默认值改动

**Files:**
- Modify: `src/batch_run.py`（:19 附近 import、:116-134 build_merged_config；attention 跳过行留到 Task 9 Step 6）
- Modify: `src/config.json`（全文重写）
- Test: `tests/test_config_mixed_abc.py`（全文重写）

**Interfaces:**
- Consumes: `scenarios.scenario_of`（Task 1）。
- Produces: `build_merged_config(exp, base_config, model_dir)` 签名不变；默认段查找顺序 = 场景段（`scenario_of(task)`）→ 旧 task 同名段（去重）。

- [ ] **Step 1: 重写 tests/test_config_mixed_abc.py（确认失败）**

全文替换为：

```python
"""Config tests: new scenario sections, deleted keys, batch_run merge order."""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

CONFIG_PATH = os.path.join(REPO_ROOT, 'src', 'config.json')

DELETED_MAIN_KEYS = ('A', 'B', 'AB_PAIRS', 'ABC_PAIRS', 'MAX_UNIQUE_RATIO',
                     'USE_LEARNABLE_PE', 'ENTROPY_PENALTY_WEIGHT', 'COND_FIX',
                     'COND_FIX_START', 'COND_FIX_START_A1', 'COND_FIX_START_A2',
                     'USE_AMP', 'AMP_DTYPE', 'ALLOW_TF32', 'PREDICT_MISSING',
                     'MISS_SECOND', 'USE_AB_TAG', 'USE_CONDITIONAL_WTE',
                     'COND_WTE_SHARED_RATIO', 'SKIP_FINAL_GENERATION_TEST',
                     'ORDER', 'RULES', 'EXPOSURE_RATIOS')


def test_config_json_main_rewrite():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    main = cfg['main']
    assert main['OOD_LEN'] == 16           # 默认从 32 改 16
    assert main['EARLY_STOP_EXTRA_EPOCHS'] == 0
    for key in DELETED_MAIN_KEYS:
        assert key not in main, f"{key} must not be in main"
    # main 不注入 ORDER/RULES/EXPOSURE_RATIOS：否则旧 action/mixed JSON 的
    # AB_PAIRS 会与 main 注入的 RULES 触发假冲突
    assert 'single' in cfg and 'mixed' in cfg and 'action' in cfg
    assert cfg['action']['NUM_TRAIN_SAMPLES'] == 10000
    assert cfg['action']['OOD_LEN'] == 32  # 旧 action JSON 的默认窗口保持
    assert cfg['tribonacci'] == {'P': 23, 'A': 1, 'B': 2, 'C': 3}
    assert cfg['mixed_ab'] == {'MIXED_AB_MAX_UNIQUE_RATIOS': [0.7, 0.7]}


def test_batch_run_merge_legacy_task():
    from batch_run import build_merged_config, BATCH_RUN_MERGED_FLAG
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        base = json.load(f)
    exp = {'name': 'smoke', 'task': 'mixed_abc',
           'config': {'P': 23, 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]]}}
    name, task, merged_main, merged = build_merged_config(exp, base, 'models/smoke')
    assert task == 'mixed_abc' and merged_main['TASK'] == 'mixed_abc'
    assert merged_main['MIXED_AB_MAX_UNIQUE_RATIOS'] == [0.7, 0.7]  # 旧 task 段生效
    assert merged_main['P'] == 23
    assert merged[BATCH_RUN_MERGED_FLAG] is True


def test_batch_run_merge_scenario_then_legacy_section():
    """查找顺序 = 场景段 -> 旧 task 段：task='action_trib' 应先吃到 action
    场景段（NUM_TRAIN_SAMPLES 等），再吃 action_trib 段。"""
    from batch_run import build_merged_config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        base = json.load(f)
    exp = {'name': 'smoke3', 'task': 'action_trib', 'config': {}}
    _, _, merged_main, _ = build_merged_config(exp, base, 'models/smoke3')
    assert merged_main['NUM_TRAIN_SAMPLES'] == 10000
    assert merged_main['TRAIN_LEN'] == 16


def test_batch_run_merge_new_action_task():
    from batch_run import build_merged_config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        base = json.load(f)
    exp = {'name': 'smoke2', 'task': 'action',
           'config': {'ORDER': 3, 'RULES': [[1, 1, 1], [3, 2, 1]]}}
    _, task, merged_main, _ = build_merged_config(exp, base, 'models/smoke2')
    assert task == 'action'
    assert merged_main['RULES'] == [[1, 1, 1], [3, 2, 1]]
    assert merged_main['NUM_TEST_SAMPLES'] == 2000  # action 场景段默认


if __name__ == '__main__':
    test_config_json_main_rewrite()
    test_batch_run_merge_legacy_task()
    test_batch_run_merge_scenario_then_legacy_section()
    test_batch_run_merge_new_action_task()
    print("ALL TESTS PASSED: test_config_mixed_abc.py")
```

跑确认失败（config.json 尚未改）。

- [ ] **Step 2: 重写 src/config.json**

全文替换为：

```json
{
  "main": {
    "P": 53,
    "TASK": "addition",
    "NUM_MASK": null,
    "D_MODEL": 64,
    "N_HEAD": 1,
    "N_LAYER": 1,
    "BATCH_SIZE": 512,
    "GRAD_ACCUM_STEPS": 1,
    "RESHUFFLE_EACH_EPOCH": false,
    "EPOCHS": 6000,
    "LR": 0.0003,
    "SAVE_PATH": "fibonacci_transformer.pth",
    "INIT_FROM": null,
    "MLP_RATIO": 4,
    "TRAIN_LEN": 16,
    "OOD_LEN": 16,
    "DROPOUT": 0.0,
    "WEIGHT_DECAY": 1,
    "GRAD_CLIP_NORM": 1.0,
    "OPT_DIAG_INTERVAL": 0,
    "FIRST_TASK_WEIGHT": 1.0,
    "RANDOM_SEED": 42,
    "EVAL_INTERVAL": 20,
    "EARLY_STOP_NO_IMPROVE": 3000,
    "EARLY_STOP_ACCURACY": 0.99,
    "EARLY_STOP_EXTRA_EPOCHS": 0,
    "MAX_TRAIN_HOURS": null,
    "RESUME_FROM": null,
    "DATA_MODE": null,
    "FRESH_TEST_PER_EVAL": false,
    "STATE_SPACE_CAP": null,
    "MISSING_PROB": 0.0,
    "MISS_LEN": 1,
    "NUM_TRAIN_SAMPLES": null,
    "NUM_TEST_SAMPLES": 256
  },
  "single": {},
  "mixed": {},
  "action": {
    "P": 53,
    "NUM_TRAIN_SAMPLES": 10000,
    "NUM_TEST_SAMPLES": 2000,
    "TRAIN_LEN": 16,
    "OOD_LEN": 32
  },
  "tribonacci": {
    "P": 23,
    "A": 1,
    "B": 2,
    "C": 3
  },
  "tetranacci": {
    "P": 23,
    "A": 1,
    "B": 1,
    "C": 1,
    "D": 1
  },
  "multiplication": {},
  "nonlinear": {},
  "nonlinear_mul": {},
  "action_trib": {},
  "mixed_ab": {
    "MIXED_AB_MAX_UNIQUE_RATIOS": [0.7, 0.7]
  },
  "mixed_abc": {
    "MIXED_AB_MAX_UNIQUE_RATIOS": [0.7, 0.7]
  }
}
```

设计说明（注释写不进 JSON，记录在本计划里）：
- `main.TASK` 保持 `"addition"` 而不是 `"single"`：main 不能注入 `ORDER`/`RULES`/`EXPOSURE_RATIOS`，否则旧 action/mixed JSON 的 `AB_PAIRS` 会与 main 注入的 `RULES` 触发归一化层的假冲突报错；`addition` 经兼容层等价于 `single` o2 `[[1,1]]`（A/B 缺省 1）。
- 系数类旧键默认值（A/B/C/D=1、AB_PAIRS=[[1,1],[1,2]]、ABC_PAIRS=[[1,1,1],[1,2,3]]、MAX_UNIQUE_RATIO=0.7）全部挪到 `scenarios.py` 的兼容默认，语义不变。
- `action_trib` 段清空：它的 P/样本数/窗口默认由查找顺序中的 action 场景段提供，系数默认在 `scenarios._DEFAULT_ABC_PAIRS`。

- [ ] **Step 3: batch_run.py 三处修改**

(a) import 区（:19 `from protocol import BATCH_RUN_MERGED_FLAG` 后）加 `from scenarios import scenario_of`。
(b) `build_merged_config`（:106-143）：删 :120-124 的 dynamic_mixed 别名次（4 行，`if task == 'dynamic_mixed':` 与 `if override.get('TASK') == 'dynamic_mixed':`，含 print）；`task_defaults = base_config.get(task, {})`（:133）替换为：

```python
    # Section lookup order: scenario section first, then the legacy same-name
    # section (dict.fromkeys dedupes when they coincide, e.g. task='action').
    task_defaults = {}
    for section in dict.fromkeys((scenario_of(task), task)):
        if section:
            task_defaults.update(base_config.get(section, {}))
```

(c) `run_single`（:219-227）的 attention 分析跳过行 `if returncode == 0 and task != 'action_trib':` **本 task 不动**（analyze_attention 的 order 参数化在 Task 9 完成）；Task 9 Step 6 负责删它。

跑：`C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_config_mixed_abc.py`
预期：4 个用例全 PASS（(c) 的跳过行保留与否不影响这些用例）。

- [ ] **Step 4: 回归 batch_run 相关测试**

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_chain_run.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_training_smoke.py
```

预期 PASS（这两个文件经 batch_run/chain_run 走 build_merged_config）。

- [ ] **Step 5: commit**

```bash
git add src/batch_run.py src/config.json tests/test_config_mixed_abc.py
git commit -m "batch_run+config: scenario default sections, alias moved to scenarios, main de-injected"
```

## Task 9: 分析/评估工具——analyze_attention order 参数化 + load_model 明确报错 + eval_clean_action / action_run_postprocess / verify_sample / error_breakdown / verify_circle / rule_fit

**Files:**
- Modify: `src/analyze_attention.py`
- Modify: `scripts/eval_clean_action.py`
- Modify: `scripts/action_run_postprocess.py`
- Modify: `src/verify_sample.py`
- Modify: `src/error_breakdown.py`
- Modify: `src/verify_circle.py`
- Modify: `src/rule_fit.py`
- Modify: `src/batch_run.py`（收尾 Task 8 留下的 action_trib 跳过行）
- Test: `tests/test_scenario_tools.py`（新建）

**Interfaces:**
- Consumes: 新 models/datasets/rules/scenarios 签名（Task 1-7）；checkpoint save_config 新块（scenario/order/rules，Task 3/7）。
- Produces:
  ```python
  # analyze_attention.py
  def load_model(pth_path, device='cpu')  # 旧 mixed(rule_head/cond_wte)/wpe checkpoint -> RuntimeError
  def _make_action_query_mask(length, order=2)
  def _resolve_recurrence(config)  # action 分支返回 dict 增加 'order' 键
  # eval_clean_action.py / action_run_postprocess.py：从 save_config 读 order（新块优先），无 order-2 硬编码
  ```

- [ ] **Step 1: 写 tests/test_scenario_tools.py（确认失败）**

完整文件：

```python
"""Tests for order-generalized analysis tools and removed-checkpoint errors."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

import analyze_attention
from models import FibonacciTransformer


def test_action_query_mask_any_order():
    # order=2 必须与旧实现逐点一致（回归钉死）
    assert analyze_attention._make_action_query_mask(5, 2) == \
        [0, 0, 0, 1, 0, 1, 0, 1]
    # order=3: 序列 [x0 x1 x2 f x3 f x4]，x3 在 index 4，x4 在 index 6
    assert analyze_attention._make_action_query_mask(5, 3) == \
        [0, 0, 0, 0, 1, 0, 1]


def _save_ckpt(state_dict, config):
    fd, path = tempfile.mkstemp(suffix='.pth')
    os.close(fd)
    torch.save({'model_state_dict': state_dict, 'config': config}, path)
    return path


def test_load_model_rejects_legacy_mixed_checkpoint():
    model = FibonacciTransformer(p=7, d_model=16, n_head=1, n_layer=1, block_size=16)
    state = model.state_dict()
    state['rule_head.0.weight'] = torch.zeros(8, 16)  # 伪造旧 mixed 指纹
    path = _save_ckpt(state, {'p': 7, 'd_model': 16, 'n_head': 1, 'n_layer': 1,
                              'block_size': 16, 'ab_pairs': [[1, 1]], 'order': 2})
    try:
        analyze_attention.load_model(path)
        assert False, "legacy mixed checkpoint must raise"
    except RuntimeError as exc:
        assert 'rule_head' in str(exc)
    finally:
        os.remove(path)


def test_load_model_rejects_learnable_pe_checkpoint():
    model = FibonacciTransformer(p=7, d_model=16, n_head=1, n_layer=1, block_size=16)
    state = model.state_dict()
    state['wpe.weight'] = torch.zeros(16, 16)  # 伪造 wpe 指纹
    path = _save_ckpt(state, {'p': 7, 'd_model': 16, 'n_head': 1, 'n_layer': 1,
                              'block_size': 16})
    try:
        analyze_attention.load_model(path)
        assert False, "wpe checkpoint must raise"
    except RuntimeError as exc:
        assert 'wpe' in str(exc)
    finally:
        os.remove(path)


def test_load_model_new_action_checkpoint():
    model = FibonacciTransformer(p=7, d_model=16, n_head=1, n_layer=1,
                                 block_size=16, vocab_size=7 + 1 + 2)
    path = _save_ckpt(model.state_dict(),
                      {'p': 7, 'd_model': 16, 'n_head': 1, 'n_layer': 1,
                       'block_size': 16, 'vocab_size': 10, 'pad_token_id': 7,
                       'scenario': 'action', 'order': 3, 'rules': [[1, 1, 1]],
                       'ab_pairs': [[1, 1, 1]], 'recurrence': 'action',
                       'train_len': 8})
    try:
        loaded, ckpt = analyze_attention.load_model(path)
        assert isinstance(loaded, FibonacciTransformer)
        rec = analyze_attention._resolve_recurrence(ckpt['config'])
        assert rec['is_action'] and rec['order'] == 3 and rec['init_len'] == 3
    finally:
        os.remove(path)


def test_resolve_recurrence_new_single_checkpoint():
    rec = analyze_attention._resolve_recurrence(
        {'p': 23, 'scenario': 'single', 'order': 3, 'rules': [[1, 2, 3]],
         'recurrence': 'tribonacci'})
    assert not rec['is_action'] and not rec.get('is_mixed')
    assert rec['init_len'] == 3
    assert rec['next_val']([1, 2, 3]) == (1 * 3 + 2 * 2 + 3 * 1) % 23


if __name__ == '__main__':
    test_action_query_mask_any_order()
    test_load_model_rejects_legacy_mixed_checkpoint()
    test_load_model_rejects_learnable_pe_checkpoint()
    test_load_model_new_action_checkpoint()
    test_resolve_recurrence_new_single_checkpoint()
    print("ALL TESTS PASSED: test_scenario_tools.py")
```

跑确认失败（`_make_action_query_mask` 尚无 order 参数）。

- [ ] **Step 2: analyze_attention.py 修改（精确锚点）**

(a) `load_model`（:35-78）：`valid_keys` 删 `'use_learnable_pe'` 和 `'cond_wte_shared_ratio'`；删 use_learnable_pe 推断块（:49-53）；MixedABTransformer 检测块（:64-72，`# Detect MixedABTransformer checkpoint...` 到 `model = models.MixedABTransformer(...)`）替换为：

```python
    # 旧 MixedABTransformer / 可学习 PE checkpoint：架构已删除，明确报错而不是
    # 让 load_state_dict 以 mismatch 崩在半截。
    if any('rule_head' in k or 'cond_wte' in k or 'ab_emb' in k for k in state_keys):
        raise RuntimeError(
            f"{pth_path} 是旧 MixedABTransformer checkpoint（state_dict 含 "
            f"rule_head/cond_wte/ab_emb 键）；该架构已在 2026-10-08 场景统一中删除。"
            f"请用重构前的代码分析，或用新代码重训。")
    if any(k.endswith('wpe.weight') for k in state_keys) or config.get('use_learnable_pe'):
        raise RuntimeError(
            f"{pth_path} 使用已删除的可学习位置编码（wpe/USE_LEARNABLE_PE）；"
            f"RoPE 现在是唯一位置编码。请用重构前的代码分析。")
```

（mlp_ratio 推断块 :55-62 保留。）
(b) `_build_input_embedding`（:81-117）：删 cond_wte 分支与 wpe 分支、`ab_label` 参数，函数体只剩 `return model.transformer.drop(model.transformer.wte(input_ids))`。
(c) `get_attention_weights`（:120-129）、`_forward_per_layer`（:132-188）、`extract_qk_raw_scores`（:191-202）：删 `ab_label=None` 参数与透传。
(d) `_resolve_recurrence`（:384-419）：action 分支替换为：

```python
    if recurrence in ('action', 'dynamic_mixed') or config.get('scenario') == 'action':
        order = config.get('order') or len(config['ab_pairs'][0])
        print(f"Model config: action order-{order}, ab_pairs={config['ab_pairs']}, p={p}")
        return {'init_len': order, 'is_action': True, 'next_val': None, 'order': order,
                'ab_pairs': config['ab_pairs'], 'flag_start_id': p + 1,
                'action_seq_len': config.get('train_len') or config.get('ood_len', 16)}
```

mixed 分支条件（:403）改为：

```python
    if ('ab_pairs' in config and 'order' in config) or config.get('scenario') == 'mixed':
```

分支内 `use_tag` 恒 False——删 `use_tag = config.get('use_ab_tag', False)`，字面量 `False` 填入返回 dict，打印行同步删 use_ab_tag 段。
(e) `_make_action_seq`（:422-425）：加 `order=2` 参数并透传 `generate_action_sample(..., order=order)`。
(f) `_make_action_query_mask`（:428-439）替换为：

```python
def _make_action_query_mask(length, order=2):
    # Attention matrix is over the input sequence, length 2*length - order.
    # x_k (k > order) sits at input index 2*(k-1) - order + 1; consistent with
    # the legacy order-2 layout (query 2*(k-1)-1), we analyze that row.
    total_len = 2 * length - order
    mask = [0] * total_len
    for k in range(order + 1, length + 1):
        mask[2 * (k - 1) - order + 1] = 1
    return mask
```

（验证 order=2：`2*(k-1)-1`，k 从 3 起——与旧实现逐点一致，测试钉死。）
(g) `_make_mixed_seq`（:442-464）：删 `miss_second=config.get('miss_second', False),` 行；`missing_token=datasets.missing_token_id(config, p, is_mixed=True)` 改为 `missing_token=p`（tag 已删，missing token 恒 p）。
(h) `analyze_model_attention`（:490-583）：action 段（:529-532）改为：

```python
    if rec['is_action']:
        test_seq = _make_action_seq(p, rec['ab_pairs'], rec['flag_start_id'],
                                    rec['action_seq_len'], seed=0, order=rec['order'])
        query_mask = _make_action_query_mask(rec['action_seq_len'], order=rec['order'])
```

single 污损段（:542-557）：删 `n_rules = ...`/`use_tag = ...`/`missing_token = ...` 三行，helper 构造的 `missing_token=missing_token` 改 `missing_token=p`；删打印行里的 `miss_second=...` 片段。
(i) 模块 import 区不动（datasets 仍需要）。

- [ ] **Step 3: scripts/eval_clean_action.py 修改**

(a) `build_model`（:65-89）替换为：

```python
def build_model(sc):
    """Rebuild the exact model via the repo's own prepare path (MISSING_PROB=0).

    order comes from the checkpoint's new block first ('order'), then from the
    legacy ab_pairs row length. New checkpoints carry canonical 'rules'; older
    ones only the action-convention 'ab_pairs'.
    """
    from scenarios import normalize_task_config
    base = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       'src', 'config.json')))
    mm = dict(base['main'])
    mm.update(base.get('action', {}))
    mm.update({
        'P': sc['p'],
        'D_MODEL': sc['d_model'],
        'N_HEAD': sc['n_head'],
        'N_LAYER': sc['n_layer'],
        'MLP_RATIO': sc.get('mlp_ratio', 4),
        'TRAIN_LEN': sc['train_len'],
        'OOD_LEN': sc['ood_len'],
        'MISSING_PROB': 0.0,
        'NUM_TRAIN_SAMPLES': 8,  # unused here; keep dataset construction cheap
        'NUM_TEST_SAMPLES': 8,
    })
    if sc.get('rules'):  # 新块优先（canonical）
        mm['ORDER'] = sc['order']
        mm['RULES'] = [list(r) for r in sc['rules']]
    else:                # 旧 checkpoint：action 约定的 ab_pairs，归一化层负责反转
        mm['AB_PAIRS'] = [list(pair) for pair in sc['ab_pairs']]
    cfg = dict(base)
    cfg['main'] = mm
    cfg['_BATCH_RUN_MERGED'] = True
    return experiment._prepare_action(cfg, normalize_task_config('action', mm))
```

(b) `main()`（:92-131）：`eval_ds = ActionDataset(...)` 与 `make_action_missing_collate(...)` 加 `order=order`，其中函数顶部加：

```python
    order = sc.get('order') or len(sc['ab_pairs'][0])
```

per-position 汇总三处 `x{t // 2 + 2}` / `x{worst[0] // 2 + 2}`（:116, :119, :129）改为 `x{(t + order) // 2 + 1}` 等（同公式：目标 t 对应值 x_j，j = (t + order)//2 + 1；order=2 时退化为旧式）。
(c) `evaluate(...)` 调用（:111-112）删 `num_mask=ctx['num_mask']`？——保留（ctx 仍有 num_mask 键，action 下为 0，evaluate 接受），不动。

- [ ] **Step 4: scripts/action_run_postprocess.py 修改**

(a) `load_full` 的规则一致性检查（:29-30）替换为：

```python
        saved_pairs = ([list(reversed(r)) for r in sc['rules']] if sc.get('rules')
                       else [list(p) for p in sc['ab_pairs']])
        if saved_pairs != [list(p) for p in cfg['AB_PAIRS']]:
            raise ValueError('Resume rules mismatch')
```

（`sc['rules']` 是 canonical，反转回 action 约定再与 cfg 的 AB_PAIRS 比。）
(b) clean 评估段（:95-110 区域）：`_prepare_action({'main': small, '_BATCH_RUN_MERGED': True})['model']` 改为：

```python
    from scenarios import normalize_task_config
    model = _prepare_action({'main': small, '_BATCH_RUN_MERGED': True},
                            normalize_task_config('action', small))['model']
```

`ActionDataset(...)` 调用加 `order=order`，`clean_per_position` 的 `x{k // 2 + 2}` 改 `x{(k + order) // 2 + 1}`；在该函数顶部加 `order = cfg.get('ORDER') or len(cfg['AB_PAIRS'][0])`。

(c) TF32 写死（:95）：`if device == 'cuda' and cfg['ALLOW_TF32']:` 改为 `if device == 'cuda':`（config.json 已删 ALLOW_TF32 键，不改会在 GPU 上 KeyError）。

- [ ] **Step 5: verify_sample.py / error_breakdown.py / verify_circle.py / rule_fit.py 修改**

(a) `src/verify_sample.py`：
- :32 import 删 `missing_token_id`。
- `resolve_experiment_cfg`（:113-129）：`is_mixed` 行替换为：

```python
    is_mixed = (task in ('mixed', 'mixed_ab', 'mixed_abc')
                or (cfg.get('ab_pairs') and 'recurrence' not in cfg
                    and cfg.get('scenario') not in ('action', 'single')))
```

- `run_one`（:132-223）：corrupt_window 调用（:169-173）删 `miss_second=...` 与 `missing_token=missing_token_id(...)` 两 kwargs（token 默认即 p）；删 `use_ab_tag`/`use_cond` 两行（:180-181）与 tag 前置（:182-184）及 cond 分支（:187-190），模型输入恒 `x = torch.tensor([list(view)], dtype=torch.long)`，`logits = model(x)[0]`；:205-206 的 tag 打印块删除；:212 的 `clean_targets` 改为 `seq[1:]`（无 tag）；:200-201 打印删 miss_second 片段。
(b) `src/error_breakdown.py`：`corrupt_view`（:44-51）删 `miss_second=...`、`missing_token=missing_token_id(cfg, p, is_mixed)` 两 kwargs，import（:35）删 `missing_token_id`；:134 打印删 miss_second 片段。
(c) `src/verify_circle.py:209-212`：`isinstance(model, models.MixedABTransformer)` 检查替换为：

```python
    from rules import task_from_save_config
    if task_from_save_config(checkpoint['config']) is None:
        print("[Warning] verify_circle targets single-rule checkpoints; "
              "mixed/action checkpoints are not meaningful here.")
```

（`models` import 若因此闲置则一并删。）
(d) `src/rule_fit.py`：
- `run_one`（:439-454）：task None 的回退块与 mixed 跳过替换为：

```python
    if task is None:
        # checkpoint fallback: task_from_save_config 新块优先；None = mixed/action
        task = task_from_save_config(checkpoint['config'])
        if task is None:
            task = checkpoint['config'].get('recurrence') or 'mixed_ab'
    if task in ('mixed', 'mixed_ab', 'mixed_abc', 'action', 'action_trib', 'dynamic_mixed'):
        print('[skip] rule_fit supports single-rule scenarios only')
        return
    if task == 'single' and 'RULES' not in cfg and 'rules' not in cfg:
        # 新 checkpoint 的系数在 save_config 的 rules 键里
        cfg['rules'] = checkpoint['config'].get('rules')
```

- `run_missing_categories`：:343 corrupt_window 调用删 `miss_second=cfg.get('MISS_SECOND', False)`；:360-361 打印删 `miss_second=...` 片段。

- [ ] **Step 6: batch_run.py 收尾——取消 action_trib 的 attention 分析跳过**

`src/batch_run.py` 的 `run_single`：`if returncode == 0 and task != 'action_trib':` 改 `if returncode == 0:`，并删其上方两行 TODO 注释（analyze_attention 已完成 order 参数化，action_trib 不再需要跳过）。

- [ ] **Step 7: 跑测试**

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_scenario_tools.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_rule_fit_missing.py
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_activation_patch.py
C:/Users/Chen/anaconda3/envs/torch/python.exe -c "import sys; sys.path.insert(0, 'src'); import analyze_attention, verify_sample, error_breakdown, verify_circle, rule_fit; print('imports OK')"
C:/Users/Chen/anaconda3/envs/torch/python.exe -c "import ast; ast.parse(open('scripts/eval_clean_action.py', encoding='utf-8').read()); ast.parse(open('scripts/action_run_postprocess.py', encoding='utf-8').read()); print('scripts parse OK')"
```

预期：全 PASS。

- [ ] **Step 8: commit**

```bash
git add src/analyze_attention.py src/verify_sample.py src/error_breakdown.py src/verify_circle.py src/rule_fit.py src/batch_run.py scripts/eval_clean_action.py scripts/action_run_postprocess.py tests/test_scenario_tools.py
git commit -m "tools: order-generalized analysis; clear errors for removed-architecture checkpoints"
```

## Task 10: scripts/experiment_pipeline.py + curriculum_chain_*.py ×3

**Files:**
- Modify: `scripts/experiment_pipeline.py`
- Modify: `src/curriculum_chain_Ntask.py`、`src/curriculum_chain_extend.py`、`src/curriculum_chain_reverse.py`
- Test: `tests/test_experiment_pipeline.py`

**Interfaces:**
- Consumes: `scenarios.normalize_task_config` / `scenario_of`；`experiment._prepare_mixed_recurrence(config, device, norm)` 新签名。
- Produces: `pipeline.normalize(task, cfg, origins)` 行为对新 task 名（single/mixed）与旧名一致地产出 effective 参数表；`SINGLE`/`TASKS` 表加新名。

- [ ] **Step 1: 改 tests/test_experiment_pipeline.py（确认失败）**

(a) `request()`（:29-41）：common 里删 `SKIP_FINAL_GENERATION_TEST=True, `（键已消失，pipeline 会以 unknown key 拒绝）。
(b) `test_effective_data_counts`（:213-245 区域）：
- 两处 `_prepare_mixed_recurrence({'main': e['config']}, 'cpu', order=2)` 改为：

```python
        from scenarios import normalize_task_config
        actual = _prepare_mixed_recurrence({'main': e['config']}, 'cpu',
                                           normalize_task_config('mixed_ab', e['config']))
```

- `USE_AB_TAG=True` 段（:229-236）替换为：

```python
    spec['experiments'][0]['config']['USE_AB_TAG'] = True
    expect_failure(lambda: pipeline.prepare(spec), 'USE_AB_TAG')  # 已删除功能必须报错
```

并删其后 `eval_sequence_tokens` / `'rule tag' in note` 相关断言（tag 模式已删）。
- 同函数后半的 `_prepare_single_recurrence({'main': e['config']}, 'addition')`（签名已变为 (config, norm)）改为：

```python
        actual = _prepare_single_recurrence({'main': e['config']},
                                            normalize_task_config('addition', e['config']))
```

(c) 其余用例不动。

跑确认失败（pipeline 尚无新逻辑）。注意本文件会起 CPU 冒烟训练，整体运行约 1-3 分钟。

- [ ] **Step 2: experiment_pipeline.py 修改（精确锚点）**

(a) `SINGLE`/`TASKS`（:21-23）改为：

```python
SINGLE = {'addition': 2, 'multiplication': 2, 'nonlinear': 2,
          'nonlinear_mul': 2, 'tribonacci': 3, 'tetranacci': 4}
TASKS = set(SINGLE) | {'mixed_ab', 'mixed_abc', 'action', 'action_trib',
                       'single', 'mixed'}
```

(b) `FALLBACKS`（:24-27）删 `MISS_SECOND`、`PREDICT_MISSING`、`SKIP_FINAL_GENERATION_TEST` 三键。
(c) `normalize()`（:79-211）：
- 函数开头（`notes = []` 之前）加：

```python
    sys.path.insert(0, str(REPO / 'src')) if str(REPO / 'src') not in sys.path else None
    from scenarios import normalize_task_config
    norm = normalize_task_config(task, cfg)
    scenario = norm['scenario']
    order = norm['order']
```

（`REPO` 在 :20 已定义；`normalize_task_config` 的报错（含已删功能）原样向上抛，正好是 prepare 的校验语义。）
- `if task in SINGLE:` 分支（:107-131）改条件为 `if scenario in SINGLE or scenario == 'single':`；分支内：
  - `order = SINGLE[task]` 行删除（用上面的 norm order；例外 task 的 order 仍需——`order = norm['order'] or SINGLE[scenario]`）；
  - `rules = [cfg[k] for k in 'ABCD'[:order]] if task in (...)` 行替换为：

```python
        rules = (norm['rules'][0] if scenario == 'single' else task)
```

  - `convention = ...` 行相应判断 `isinstance(rules, list)` 不变；
  - NUM_TRAIN_SAMPLES 推导行里的 `cfg['MAX_UNIQUE_RATIO']` 改 `norm['exposure_ratios']`（标量 float；例外 task 为 None，回退 `cfg.get('MAX_UNIQUE_RATIO', 0.7)`——写成 `exposure_scalar = norm['exposure_ratios'] if norm['exposure_ratios'] is not None else cfg.get('MAX_UNIQUE_RATIO', 0.7)`，推导处用 `exposure_scalar`）。
- `elif task.startswith('mixed_'):` 分支（:132-166）改条件为 `elif scenario == 'mixed':`；分支内：
  - 删 `order = 3 if ... else 2` 与 `key = ...` 两行；`rules = cfg.get(key)` 改 `rules = norm['rules']`；其后的 rules 校验与去重 require 保留（对 list 生效不变）；
  - 删 `cfg.setdefault('USE_AB_TAG', True)` 与 origins 行（:141-142）；`NUM_MASK` derive 改 `derived('NUM_MASK', order - 1)`；
  - `ratios = cfg.get('MIXED_AB_MAX_UNIQUE_RATIOS', cfg['MAX_UNIQUE_RATIO'])` 改 `ratios = norm['exposure_ratios']`；
  - `train_tokens = cfg['TRAIN_LEN'] + int(cfg['USE_AB_TAG'])` 改 `train_tokens = cfg['TRAIN_LEN']`；
  - notes 里的 mixed NUM_TRAIN_SAMPLES 说明行保留。
- `else:`（action）分支（:167-182）：
  - 删 `order = ...` / `key = ...` 两行；`rules = cfg.get(key)` 改 `rules = norm['rules']`（canonical）；校验保留；
  - 删 `require(not cfg['MISS_SECOND'], ...)` 行；
  - `convention = '第一个系数乘最老值；flag 在每一步选择规则'` 改为 `'RULES 为 canonical（第一个系数乘最新值）；旧 AB_PAIRS/ABC_PAIRS 约定相反，归一化层已反转；flag 在每一步选择规则'`。
- 通用部分：
  - 删 `if cfg['USE_AMP']:` notes 块（:190-191）；
  - `eval_len = ...`（:192）里 `task.startswith('action')` 改 `scenario == 'action'`；
  - 删 :193-198 的 tag 相关 `eval_tag`/`eval_tokens` 旧逻辑与 fresh-test tag 警告 notes，`eval_tokens = 2 * eval_len - order if scenario == 'action' else eval_len`；
  - `max_tokens = ...`（:199-200）改 `2 * max(...) - order if scenario == 'action' else max(cfg['TRAIN_LEN'], cfg['OOD_LEN'])`；
  - :201-202 两条 notes：第一条保留；第二条替换为 `'训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。'`；
  - `task.startswith('mixed_')`（:208 的 actual_num_mask 行）改 `scenario == 'action'` 的判断相应改为：`actual_num_mask='数据集 loss_mask' if scenario == 'action' else cfg['NUM_MASK']`；
  - 返回 dict 的 `rules=rules` 不变（现在是 canonical）。
(d) `prepare()`（:236-320）不变（`require(task in TASKS, ...)` 自动接受新名）。

- [ ] **Step 3: curriculum_chain_*.py ×3**

(a) `src/curriculum_chain_Ntask.py` 的 `run_stage` override（:127-135）替换为：

```python
    override = {
        'TASK': 'mixed', 'ORDER': 2, 'RULES': RULES[:n],
        'EXPOSURE_RATIOS': [0.7] * n,
        'RANDOM_SEED': donor_seed, 'P': P, 'D_MODEL': cfg['d'],
        'N_LAYER': cfg['l'], 'N_HEAD': cfg['h'], 'MLP_RATIO': 8,
        # 历史课程实验用 NUM_MASK=2（tag 时代的默认），显式保留以维持可比性
        'NUM_MASK': 2,
        'INIT_FROM': donor, 'SAVE_PATH': save_path,
    }
```

`merged_main.update(base.get('mixed_ab', {}))`（:139）改 `merged_main.update(base.get('mixed', {}))`。
(b) `src/curriculum_chain_extend.py` 的 override（:88-96）同样替换（RULES[:n]→pairs 变量、EXPOSURE_RATIOS [0.7]*len(pairs)、删 USE_AB_TAG/USE_CONDITIONAL_WTE、NUM_MASK 2 保留、MAX_TRAIN_HOURS 6 保留），`base.get('mixed_ab', {})`（:100）改 `base.get('mixed', {})`。
(c) `src/curriculum_chain_reverse.py` 的 override（:81-89）同样替换，`base.get('mixed_ab', {})`（:95）改 `base.get('mixed', {})`。
(d) 三文件的模块 docstring 里 `mixed_ab` 表述改为 `mixed`（TASK 名），其余不动。

注意：这些脚本的 donor 是旧 MixedABTransformer checkpoint——`_load_partial_checkpoint` 对 rule_head.\* 键名不匹配自动 skipped 并打印报告，无需代码改动。

- [ ] **Step 4: 跑测试**

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_experiment_pipeline.py
C:/Users/Chen/anaconda3/envs/torch/python.exe -c "import ast; [ast.parse(open(f'src/curriculum_chain_{n}.py', encoding='utf-8').read()) for n in ('Ntask', 'extend', 'reverse')]; print('chains parse OK')"
```

预期：PASS（pipeline 测试含 CPU 冒烟，耐心等待）。已知限制：本机 Windows 无 fcntl，`tests/test_remote_experiment_runner.py` 与 `tests/test_experiment_queue.py` 不可跑，本 task 不改它们，但需在 commit message 里注明 Linux 服务器回归要求。

- [ ] **Step 5: commit**

```bash
git add scripts/experiment_pipeline.py src/curriculum_chain_Ntask.py src/curriculum_chain_extend.py src/curriculum_chain_reverse.py tests/test_experiment_pipeline.py
git commit -m "pipeline+chains: scenario-aware review tables; chains on new mixed keys"
```

## Task 11: 等价性金标准测试

**Files:**
- Test: `tests/test_scenario_equivalence.py`（新建）

**Interfaces:**
- Consumes: `experiment.run_experiment` / `_prepare_mixed_recurrence`、`scenarios.normalize_task_config`（全部已冻结）。

- [ ] **Step 1: 写 tests/test_scenario_equivalence.py（直接应通过；若失败说明前面 task 引入了语义偏差，回去修）**

完整文件：

```python
"""Golden-master equivalence: legacy task spelling vs new scenario spelling.

Same seed, tiny CPU runs (2 epochs). single/action: the two spellings of the
same experiment must produce bitwise-identical model weights (normalization
is pure translation and consumes no RNG). mixed: the model differs by design
(rule_head removed), so only the datasets are compared sample-for-sample;
this doubles as the coefficient-reversal check for action_o3 vs action_trib.
"""
import contextlib
import io
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

from experiment import run_experiment, _prepare_mixed_recurrence
from scenarios import normalize_task_config


def _base_main():
    return {'P': 7, 'D_MODEL': 16, 'N_HEAD': 1, 'N_LAYER': 1, 'BATCH_SIZE': 8,
            'EPOCHS': 2, 'LR': 0.001, 'RANDOM_SEED': 42, 'TRAIN_LEN': 8,
            'OOD_LEN': 8, 'DROPOUT': 0.0, 'WEIGHT_DECAY': 0.1,
            'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0,
            'EARLY_STOP_NO_IMPROVE': 3000, 'MLP_RATIO': 2, 'NUM_MASK': None}


def _run(main, tmpdir, tag):
    main = dict(main)
    main['SAVE_PATH'] = os.path.join(tmpdir, f'{tag}.pth')
    path = os.path.join(tmpdir, f'{tag}.json')
    with open(path, 'w') as f:
        json.dump({'main': main, '_BATCH_RUN_MERGED': True}, f)
    with contextlib.redirect_stdout(io.StringIO()):
        run_experiment(path)
    return torch.load(main['SAVE_PATH'], map_location='cpu', weights_only=False)


def _assert_same_weights(a, b):
    sa, sb = a['model_state_dict'], b['model_state_dict']
    assert set(sa) == set(sb), f"state keys differ: {sorted(set(sa) ^ set(sb))}"
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"weights differ at {k}"


def test_single_o3_equals_tribonacci():
    with tempfile.TemporaryDirectory() as tmpdir:
        old = _run({**_base_main(), 'TASK': 'tribonacci', 'A': 1, 'B': 2, 'C': 3},
                   tmpdir, 'old')
        new = _run({**_base_main(), 'TASK': 'single', 'ORDER': 3,
                    'RULES': [[1, 2, 3]]}, tmpdir, 'new')
        _assert_same_weights(old, new)
        assert new['config']['scenario'] == 'single'
        assert new['config']['order'] == 3
        assert new['config']['rules'] == [[1, 2, 3]]
        # 旧字段照写，供旧分析工具
        assert (new['config']['a'], new['config']['b'], new['config']['c']) == (1, 2, 3)
        assert new['config']['recurrence'] == 'tribonacci'


def test_action_o3_equals_action_trib():
    # RULES 是 canonical（第一个系数乘最新值）；ABC_PAIRS 是 action 旧约定
    # （第一个系数乘最老值）。权重逐位相等即证明两处反转互为镜像、净效果为零。
    with tempfile.TemporaryDirectory() as tmpdir:
        old = _run({**_base_main(), 'TASK': 'action_trib',
                    'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]],
                    'NUM_TRAIN_SAMPLES': 40, 'NUM_TEST_SAMPLES': 16}, tmpdir, 'old')
        new = _run({**_base_main(), 'TASK': 'action', 'ORDER': 3,
                    'RULES': [[1, 1, 1], [3, 2, 1]],
                    'NUM_TRAIN_SAMPLES': 40, 'NUM_TEST_SAMPLES': 16}, tmpdir, 'new')
        _assert_same_weights(old, new)
        assert new['config']['scenario'] == 'action'
        assert new['config']['rules'] == [[1, 1, 1], [3, 2, 1]]      # canonical
        assert new['config']['ab_pairs'] == [[1, 1, 1], [1, 2, 3]]   # action 约定照写


def test_mixed_o2_dataset_equals_mixed_ab():
    # 模型结构不同（rule_head 已删），只比数据集逐点相同与样本数（规格 §5）。
    old_cfg = {'P': 7, 'D_MODEL': 16, 'N_HEAD': 1, 'N_LAYER': 1, 'BATCH_SIZE': 8,
               'TRAIN_LEN': 8, 'OOD_LEN': 8, 'DROPOUT': 0.0,
               'AB_PAIRS': [[1, 1], [2, 3]], 'MIXED_AB_MAX_UNIQUE_RATIOS': [0.4, 0.4]}
    new_cfg = {'P': 7, 'D_MODEL': 16, 'N_HEAD': 1, 'N_LAYER': 1, 'BATCH_SIZE': 8,
               'TRAIN_LEN': 8, 'OOD_LEN': 8, 'DROPOUT': 0.0,
               'ORDER': 2, 'RULES': [[1, 1], [2, 3]], 'EXPOSURE_RATIOS': [0.4, 0.4]}
    random.seed(0)
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        old_ctx = _prepare_mixed_recurrence({'main': old_cfg}, 'cpu',
                                            normalize_task_config('mixed_ab', old_cfg))
    random.seed(0)
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        new_ctx = _prepare_mixed_recurrence({'main': new_cfg}, 'cpu',
                                            normalize_task_config('mixed', new_cfg))
    assert len(old_ctx['train_dataset']) == len(new_ctx['train_dataset'])
    assert len(old_ctx['test_dataset']) == len(new_ctx['test_dataset'])
    for (so, lo), (sn, ln) in zip(old_ctx['train_dataset'], new_ctx['train_dataset']):
        assert lo == ln and so.tolist() == sn.tolist()
    for (so, lo), (sn, ln) in zip(old_ctx['test_dataset'], new_ctx['test_dataset']):
        assert lo == ln and so.tolist() == sn.tolist()


if __name__ == '__main__':
    test_single_o3_equals_tribonacci()
    test_action_o3_equals_action_trib()
    test_mixed_o2_dataset_equals_mixed_ab()
    print("ALL TESTS PASSED: test_scenario_equivalence.py")
```

- [ ] **Step 2: 跑并修偏差**

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_scenario_equivalence.py
```

预期：PASS。若 `_assert_same_weights` 失败：优先怀疑归一化层意外消耗 RNG 或 default 不一致（EXPOSURE_RATIOS/样本数/词表），逐一对齐两个 run 的 stdout 数据集行数定位。**不允许靠放宽断言过关。**

- [ ] **Step 3: commit**

```bash
git add tests/test_scenario_equivalence.py
git commit -m "tests: golden-master equivalence (single_o3=tribonacci, action_o3=action_trib, mixed dataset)"
```

## Task 12: 现有测试套件修复与全量回归

**Files:**
- Modify: `tests/test_fresh_test_eval.py`（修基线已红的 single 用例）
- Modify: `tests/test_sampler_reshuffle.py`（:54-55 删 `USE_AMP=False,`、`SKIP_FINAL_GENERATION_TEST=True,` 两个已消失键）

**Interfaces:**
- Consumes: 全部前序 task。
- Produces: 无新接口。

- [ ] **Step 1: 修 test_fresh_test_eval.py 的过时断言（基线已红，与本次重构无关但顺手修）**

`test_fresh_test_single_recurrence`（:60-69）：single 的 fresh 路径打印的是 `[FreshTest] sampled with replacement: n=32, len=8, seed=N`（experiment.py sampled_fresh_test 分支），不是 action 的 `rebuilt test set`。两处修改：
- 正则 `r'\[FreshTest\] rebuilt test set: n=32, len=8, seed=(\d+)'` 改 `r'\[FreshTest\] sampled with replacement: n=32, len=8, seed=(\d+)'`；
- `assert '200 + ' in out` 改 `assert '200 unique train samples' in out`（与 `[Dataset][sampled_fresh_test]` 打印对齐）。

跑：`C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_fresh_test_eval.py`
预期：4 个用例全 PASS（action 两个、single、mixed 各一）。

- [ ] **Step 2: 修 test_sampler_reshuffle.py 的已删键**

`run()` 的 main dict（:50-56）：删 `USE_LEARNABLE_PE=False,`、`USE_AMP=False,`、`SKIP_FINAL_GENERATION_TEST=True,` 三个已写死/删除的键（传 False 仅警告不报错，但测试不应再携带它们）。

跑：`C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_sampler_reshuffle.py`
预期 PASS。

- [ ] **Step 3: 全量回归（逐文件，预期输出全为 ALL TESTS PASSED / All tests passed / N/N tests passed）**

```bash
cd C:/Users/Chen/Documents/workspace/recursion
for t in test_scenarios test_rules test_scenario_tools test_scenario_equivalence \
         test_missing_values test_mixed_dataset test_mixed_ab_compat \
         test_prepare_mixed_recurrence test_action_trib test_action_missing \
         test_training_smoke test_early_stop test_fresh_test_eval \
         test_chain_run test_train_timeout_resume test_sampler_reshuffle \
         test_checkpoint_atomic test_data_generation_modes test_state_cap \
         test_activation_patch test_rule_fit_missing test_config_mixed_abc; do
  echo "== $t"
  C:/Users/Chen/anaconda3/envs/torch/python.exe tests/$t.py 2>&1 | tail -1
done
PYTHONPATH=src C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_sample_gen.py 2>&1 | tail -1
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_experiment_pipeline.py 2>&1 | tail -1
```

验收标准：24 个可跑文件全绿（循环 22 个 + test_sample_gen + test_experiment_pipeline）。Windows 本机不可跑的两个（`test_experiment_queue.py`、`test_remote_experiment_runner.py`，fcntl 依赖）做 import/编译级检查并记录：

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe -c "import ast; ast.parse(open('tests/test_experiment_queue.py', encoding='utf-8').read()); ast.parse(open('tests/test_remote_experiment_runner.py', encoding='utf-8').read()); print('parse OK')"
```

并在交付说明里注明：这两个文件需在 Linux 服务器回归（本 task 未改动它们涉及的逻辑面，但 `test_remote_experiment_runner.py` 的 common 含 `USE_AMP=False`，pipeline 端经 scenarios.py 的字符串扫描仍接受该键，预期不受影响）。

若有文件红：回到对应 task 修，不许在本 task 里改 src/ 逻辑（本 task 只允许动测试文件；src 有漏说明前序 task 有缺陷，应在前序 task 的代码上修并补该 task 的 focused test，然后重跑本回归——通过 amend 前序 commit 或新增 fix commit 均可，在 commit message 里说明）。

- [ ] **Step 4: commit**

```bash
git add tests/test_fresh_test_eval.py tests/test_sampler_reshuffle.py
git commit -m "tests: fix stale fresh-test assertions; drop removed keys from reshuffle test"
```

## Task 13: 文档同步

**Files:**
- Modify: `docs/recursion_model_and_training_settings.md`
- Modify: `AGENTS.md`
- Modify: `.agents/skills/recursion-experiment-runner/SKILL.md`
- Modify: `docs/AMP_PRECISION_NOTES.md`

**Interfaces:** 无代码接口；改完文档与代码现状必须一致（以 Task 1-12 落地后的代码为准）。

- [ ] **Step 1: docs/recursion_model_and_training_settings.md**

(a) §1（任务定义，:7-127）整体重写为三场景矩阵：`single`（ORDER≥2，RULES canonical 一条，旧名 addition/tribonacci/tetranacci 为兼容别名）、`mixed`（ORDER≥2，RULES≥1 条，NUM_MASK 默认 order-1，词表恒 P+1）、`action`（ORDER≥2，RULES≥1 条 canonical，flag 交错布局 `[x0..x_{o-1}, flag, x_o, ...]`，序列长 2L-order）；保留三个非线性例外 task 小节（原 1.2/1.4/1.5 的 multiplication/nonlinear/nonlinear_mul 内容平移）；污损维度统一 `MISSING_PROB`+`MISS_LEN`（0/不写=干净），并明确 **predict 是唯一口径**（污损位以原值为目标并计入损失，mask 口径 2026-10-08 删除）。
(b) §1.7 的 action 系数约定警告保留并改写：旧 `AB_PAIRS`/`ABC_PAIRS`（第一个系数乘最老值）仅作兼容输入，归一化层反转；新键 RULES 三场景同一方向（第一个系数乘最新值）；action 序列逐位置 loss mask 不变。
(c) §2 配置表按规格 §1.1 全量重写（保留键分类列出）；被删键（A/B/C/D、AB_PAIRS/ABC_PAIRS、MAX_UNIQUE_RATIO、MIXED_AB_MAX_UNIQUE_RATIOS、PREDICT_MISSING、MISS_SECOND、SKIP_FINAL_GENERATION_TEST、COND_FIX 系列、AMP_DTYPE、USE_AMP、USE_LEARNABLE_PE、ALLOW_TF32、ENTROPY_PENALTY_WEIGHT、USE_AB_TAG、USE_CONDITIONAL_WTE、COND_WTE_SHARED_RATIO）单列一节"已于 2026-10-08 删除的键"，注明归一化层对显式启用者报错、对与写死值一致者警告忽略；注明 OOD_LEN 默认 32→16、EARLY_STOP_EXTRA_EPOCHS 默认 200→0、写死五项（PREDICT_MISSING=true / USE_LEARNABLE_PE=false / ALLOW_TF32=true / USE_AMP=false / ENTROPY_PENALTY_WEIGHT=0）。
(d) §3.3（MixedABTransformer，:336-370）整节删除，§3.1 总述改为"唯一模型类 FibonacciTransformer"；§3.2 删 wpe/entropy penalty 相关描述；§3.4 action 词表扩展保留。
(e) §4.2 MixedRecurrenceDataset 小节删 tag 模式描述。

- [ ] **Step 2: AGENTS.md**

(a) §1 代码结构：`core.py` 行删 final_eval 提及；`experiment.py` 行改为"run_experiment 开头经 scenarios.normalize_task_config 归一化后按 scenario 分派（_prepare_single_recurrence / _prepare_mixed_recurrence / _prepare_action）"；新增 `scenarios.py` 行（归一化层：旧 task 名映射、action 反转、已删功能报错）；删 `final_eval.py` 相关表述；`rules.py` 行补 `rules_from_config(rules_raw, p)` 新签名。
(b) §3 语义改动清单顶部加 2026-10-08 条目：场景统一 + 删除功能清单 + 已知语义变化四条（predict 恒口径、mixed 无 rule_head/tag/cond_wte、无 final generation test、OOD_LEN 默认 16），指向 `docs/superpowers/specs/2026-10-08-scenario-unification-design.md`；§3 第 7 条 action_trib 的"attention 分析暂不支持 action_trib"一句删除（已参数化）。
(c) §4 misslen 评估 pipeline 条目不变（eval_clean_action.py 仍是口径工具，已 order 参数化）。

- [ ] **Step 3: .agents/skills/recursion-experiment-runner/SKILL.md**

§1 命名约定处补一行：新体系实验名 `{scenario}_o{order}_...`（如 `single_o3_d1024l2_P127_tr64_miss0.1len1_seed17996`），seed 后缀规则不变（课程链 `@prev` 配对依赖）；旧名（tribonacci/mixed_ab/action_trib 等）仍可跑但新实验一律用场景名。

- [ ] **Step 4: docs/AMP_PRECISION_NOTES.md**

文首加一段（不删历史内容）："**2026-10-08 更新**：AMP 代码路径已随场景统一移除（USE_AMP 写死 false、AMP_DTYPE 键删除、autocast/GradScaler 代码不存在）。本文档保留为历史教训：结论性实验必须 fp32。"

- [ ] **Step 5: 一致性自查并 commit**

逐一对照：`grep -n 'MixedABTransformer\|final_eval\|USE_AB_TAG\|COND_FIX' docs/recursion_model_and_training_settings.md AGENTS.md .agents/skills/recursion-experiment-runner/SKILL.md`——除"已删除"声明语境外不得再有现行语义描述。

```bash
git add docs/recursion_model_and_training_settings.md AGENTS.md .agents/skills/recursion-experiment-runner/SKILL.md docs/AMP_PRECISION_NOTES.md
git commit -m "docs: scenario-unification settings/AGENTS/SKILL sync; AMP notes marked historical"
```

## 收尾（所有 task 完成后）

- 全量回归重跑一次（Task 12 Step 3 的命令块）。
- 规格 §6 的四条语义变化写进给用户的交付说明；本仓库 `experiments/` 旧 JSON 未改动，依赖兼容层继续可跑。
