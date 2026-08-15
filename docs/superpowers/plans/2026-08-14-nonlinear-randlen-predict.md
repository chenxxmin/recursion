# 非线性规则 randlen predict 实验组 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `nonlinear_mul` 递推任务（X(k) = X(k-2)·X(k-1)² mod p）并生成 224-run 实验组 JSON（nonlinear + nonlinear_mul 的 randlen predict 污损网格 + prob=0 对照）。

**Architecture:** 复用现有单规则任务管线：rules.py 的三个查表函数各加分支 → experiment.py 路由 tuple + default_num_mask 各加一项 → config.json 加占位段。实验组 JSON 由一次性脚本生成并校验。数据集/模型/训练零改动。

**Tech Stack:** Python 3, PyTorch, 纯 JSON 网格文件。

**Spec:** `docs/superpowers/specs/2026-08-14-nonlinear-randlen-predict-design.md`（已批准，含 prob=0 对照增补）

## Global Constraints

- 测试必须用 torch 环境 python：`C:/Users/Chen/anaconda3/envs/torch/python.exe`（base 环境无 torch）。每个测试文件有 `__main__` runner，从 `tests/` 目录运行：`C:/Users/Chen/anaconda3/envs/torch/python.exe test_rules.py`。
- 不得改变任何现有 task 的行为；RNG 调用序不可变（只新增分支，不改既有路径）。
- 回归保障（Task 2 末尾必须跑）：黄金 stdout diff 为空——
  ```
  C:/Users/Chen/anaconda3/envs/torch/python.exe .superpowers/sdd/core-split/golden_run.py .superpowers/sdd/core-split/golden_after
  diff -q .superpowers/sdd/core-split/golden/addition.txt .superpowers/sdd/core-split/golden_after/addition.txt
  diff -q .superpowers/sdd/core-split/golden/mixed_ab.txt .superpowers/sdd/core-split/golden_after/mixed_ab.txt
  rm -rf .superpowers/sdd/core-split/golden_after
  ```
- 网格静态配置（逐键 verbatim）：`P:127, TRAIN_LEN:64, OOD_LEN:128, MAX_UNIQUE_RATIO:0.7, NUM_MASK:1, D_MODEL:256, N_HEAD:4, MLP_RATIO:4, BATCH_SIZE:128, LR:0.0003, WEIGHT_DECAY:1, EPOCHS:10000, EVAL_INTERVAL:20, EARLY_STOP_ACCURACY:0.99, EARLY_STOP_NO_IMPROVE:1000, PREDICT_MISSING:true`。**不写 A/B**（nonlinear 类规则忽略它们，写着会误导）。
- Seed 列表：`[17996, 18318, 34789, 44536, 62513, 64154, 72814, 82585]`。
- run 名模式：`{task}_d256l{L}r4h4_P127_tr64ood128_randmiss{prob}len{ml}_seed{seed}`（prob 用 `str()` 格式化：`0.1`/`0.3`/`0.5`/`0`）。
- 组文件为 new format：`{"concurrency": 8, "experiments": [...]}`；每条实验恰好三个字段 `name`/`task`/`config`。
- 本计划**不包含**启动 batch_run（用户在 GPU 机器上自行启动）；交付物 = 代码 + 测试 + 校验过的 JSON。

---

### Task 1: rules.py 支持 nonlinear_mul

**Files:**
- Modify: `src/rules.py`（`single_rule_from_task` :44-68、`save_config_extra` :71-83、`task_from_save_config` :86-104）
- Test: `tests/test_rules.py`（`test_single_rule_from_task` :70-87、`test_save_config_round_trip` :90-101、`__main__` :104-114 无需改——新断言加在现有函数内）

**Interfaces:**
- Consumes: 现有 `single_rule_from_task(task, cfg) -> (init_len, next_fn, name)` 契约。
- Produces: `single_rule_from_task('nonlinear_mul', cfg)` → `(2, fn, "X(k)=(X(k-2)*X(k-1)^2) mod <p>")`，其中 `fn(seq, p) == (seq[-2] * seq[-1] * seq[-1]) % p`；`save_config_extra('nonlinear_mul', cfg)` → `{'recurrence': 'nonlinear_mul'}`；`task_from_save_config({'recurrence': 'nonlinear_mul'})` → `'nonlinear_mul'`。Task 2 的路由和 Task 3 的网格都依赖这组行为。

- [ ] **Step 1: 写失败测试**

在 `tests/test_rules.py` 的 `test_single_rule_from_task` 中，`assert fn([4, 5], p) == (25 + 4) % p`（:82）之后、`try:`（:83）之前插入：

```python
    _, fn, name = single_rule_from_task('nonlinear_mul', {'P': p})
    assert fn([4, 5], p) == (4 * 5 * 5) % p and 'X(k-2)*X(k-1)^2' in name
    # x2=0 collapses to 0 regardless of x1 (non-bijective state map)
    assert fn([7, 0], p) == 0
```

在 `test_save_config_round_trip` 中把 :93 的 task tuple 改为：

```python
    for task in ('addition', 'multiplication', 'tribonacci', 'nonlinear', 'nonlinear_mul'):
```

- [ ] **Step 2: 跑测试确认失败**

```
cd tests && C:/Users/Chen/anaconda3/envs/torch/python.exe test_rules.py
```
预期：FAIL，`ValueError: unknown single-rule task: nonlinear_mul`。

- [ ] **Step 3: 实现 rules.py 三个分支**

`single_rule_from_task`：docstring :47 的 task 列表改为
`task: 'addition' | 'multiplication' | 'tribonacci' | 'nonlinear' | 'nonlinear_mul'.`，
并在 `if task == 'nonlinear':` 分支（:64-67）之后、`raise ValueError`（:68）之前插入：

```python
    if task == 'nonlinear_mul':
        # The state map (x,y) -> (y, x*y^2) is not bijective (any state with
        # y=0 flows into the (0,0) fixed point), so transient trajectories
        # exist; RecurrenceDataset handles them (truncated trajectories get
        # step-by-step extension in run(), see generate_cycle's comment).
        return 2, (lambda seq, p: (seq[-2] * seq[-1] * seq[-1]) % p), f"X(k)=(X(k-2)*X(k-1)^2) mod {p}"
```

`save_config_extra`：在 `if task == 'nonlinear':` 分支（:81-82）之后、`raise ValueError`（:83）之前插入：

```python
    if task == 'nonlinear_mul':
        return {'recurrence': 'nonlinear_mul'}
```

`task_from_save_config`：在 `if recurrence == 'nonlinear':` 分支（:102-103）之后、`return 'addition'`（:104）之前插入：

```python
    if recurrence == 'nonlinear_mul':
        return 'nonlinear_mul'
```

- [ ] **Step 4: 跑测试确认通过**

```
cd tests && C:/Users/Chen/anaconda3/envs/torch/python.exe test_rules.py
```
预期：`ALL TESTS PASSED: test_rules.py`

- [ ] **Step 5: 提交**

```bash
git add src/rules.py tests/test_rules.py
git commit -m "Add nonlinear_mul rule: X(k)=(X(k-2)*X(k-1)^2) mod p"
```

---

### Task 2: 任务路由 + config.json 占位段 + 训练冒烟

**Files:**
- Modify: `src/experiment.py`（docstring :282、`default_num_mask` :298、路由 tuple :418）
- Modify: `src/config.json`（:41 `"nonlinear": {},` 之后）
- Test: `tests/test_training_smoke.py`（新增测试函数 + `__main__` :67-70）

**Interfaces:**
- Consumes: Task 1 的 `single_rule_from_task('nonlinear_mul', ...)` / `save_config_extra('nonlinear_mul', ...)`。
- Produces: `run_experiment` 接受 `TASK: 'nonlinear_mul'` 的 merged config 并完整跑通（数据集→训练→stage-3）；Task 3 的 JSON 条目依赖该 task 名可路由。

- [ ] **Step 1: 写失败冒烟测试**

在 `tests/test_training_smoke.py` 的 `test_run_experiment_mixed_ab`（:58-64）之后插入：

```python
def test_run_experiment_nonlinear_mul():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run(tmpdir, {'TASK': 'nonlinear_mul'})
        _check_common(out, save_path)
        assert 'X(k)=(X(k-2)*X(k-1)^2) mod 7' in out
```

`__main__`（:67-70）改为：

```python
if __name__ == '__main__':
    test_run_experiment_addition()
    test_run_experiment_mixed_ab()
    test_run_experiment_nonlinear_mul()
    print("ALL TESTS PASSED: test_training_smoke.py")
```

- [ ] **Step 2: 跑测试确认失败**

```
cd tests && C:/Users/Chen/anaconda3/envs/torch/python.exe test_training_smoke.py
```
预期：FAIL——`run_experiment` 打印 `Unknown task: nonlinear_mul` 后 `sys.exit(2)`，测试以 `SystemExit: 2` 崩溃。

- [ ] **Step 3: 实现路由**

`src/experiment.py` 三处：

:282 docstring 改为：
```python
    """Prepare dataset, model, loaders and training params for addition/multiplication/tribonacci/nonlinear/nonlinear_mul."""
```

:298 改为：
```python
    default_num_mask = {'addition': 1, 'multiplication': 1, 'tribonacci': 2, 'nonlinear': 1, 'nonlinear_mul': 1}[task]
```

:418 改为：
```python
    elif TASK in ('addition', 'multiplication', 'tribonacci', 'nonlinear', 'nonlinear_mul'):
```

`src/config.json`：在 `"nonlinear": {},`（:41）之后插入一行：
```json
  "nonlinear_mul": {},
```

- [ ] **Step 4: 跑冒烟确认通过**

```
cd tests && C:/Users/Chen/anaconda3/envs/torch/python.exe test_training_smoke.py
```
预期：`ALL TESTS PASSED: test_training_smoke.py`

- [ ] **Step 5: 全套测试 + 黄金 diff 回归**

```
cd tests && for t in test_*.py; do C:/Users/Chen/anaconda3/envs/torch/python.exe "$t" > /tmp/t.txt 2>&1 && echo "PASS $t" || { echo "FAIL $t"; cat /tmp/t.txt; }; done
```
预期：全部 PASS（含新增 smoke）。

再跑 Global Constraints 里的黄金 diff 命令组，两个 diff 都必须为空。

- [ ] **Step 6: 提交**

```bash
git add src/experiment.py src/config.json tests/test_training_smoke.py
git commit -m "Route nonlinear_mul task (experiment.py + config.json + smoke test)"
```

---

### Task 3: 生成并校验 224-run 实验组 JSON

**Files:**
- Create: `.superpowers/sdd/core-split/gen_nonlinear_grid.py`（一次性脚本，scratch 不入库）
- Create: `experiments/nonlinear_p127_tr64_ood128_randmisslen_d256h4_predict.json`（生成产物，入库）

**Interfaces:**
- Consumes: Task 2 的路由（task 名 `nonlinear`/`nonlinear_mul` 可被 run_experiment 接受）；`batch_run.build_merged_config(exp, base_config, model_dir) -> (name, task, merged_main, merged)`（src/batch_run.py:106）。
- Produces: 组文件 `experiments/nonlinear_p127_tr64_ood128_randmisslen_d256h4_predict.json`，224 条（192 主网格 + 32 prob=0 对照），用户以 `python src/batch_run.py experiments/<file>` 启动。

- [ ] **Step 1: 写生成+校验脚本**

创建 `.superpowers/sdd/core-split/gen_nonlinear_grid.py`：

```python
"""One-off generator + validator for the nonlinear randlen-predict group.
Spec: docs/superpowers/specs/2026-08-14-nonlinear-randlen-predict-design.md
Scratch tool (lives under gitignored .superpowers/), not committed.
"""
import itertools
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
OUT = os.path.join(REPO_ROOT, 'experiments',
                   'nonlinear_p127_tr64_ood128_randmisslen_d256h4_predict.json')

SEEDS = [17996, 18318, 34789, 44536, 62513, 64154, 72814, 82585]
TASKS = ['nonlinear', 'nonlinear_mul']
STATIC = {
    'P': 127, 'TRAIN_LEN': 64, 'OOD_LEN': 128, 'MAX_UNIQUE_RATIO': 0.7,
    'NUM_MASK': 1, 'D_MODEL': 256, 'N_HEAD': 4, 'MLP_RATIO': 4,
    'BATCH_SIZE': 128, 'LR': 0.0003, 'WEIGHT_DECAY': 1, 'EPOCHS': 10000,
    'EVAL_INTERVAL': 20, 'EARLY_STOP_ACCURACY': 0.99,
    'EARLY_STOP_NO_IMPROVE': 1000, 'PREDICT_MISSING': True,
}


def entry(task, n_layer, miss_len, prob, seed):
    cfg = dict(STATIC, N_LAYER=n_layer, MISS_LEN=miss_len,
               MISSING_PROB=prob, RANDOM_SEED=seed)
    name = (f"{task}_d256l{n_layer}r4h4_P127_tr64ood128_"
            f"randmiss{prob}len{miss_len}_seed{seed}")
    return {'name': name, 'task': task, 'config': cfg}


experiments = []
for task, n_layer, miss_len, prob, seed in itertools.product(
        TASKS, [1, 2], [1, 2], [0.1, 0.3, 0.5], SEEDS):
    experiments.append(entry(task, n_layer, miss_len, prob, seed))
# prob=0 control: corruption is fully disabled at prob=0 (datasets.py gates on
# missing_prob > 0), so MISS_LEN is inert and the control spans no MISS_LEN axis.
for task, n_layer, seed in itertools.product(TASKS, [1, 2], SEEDS):
    experiments.append(entry(task, n_layer, 1, 0, seed))

group = {'concurrency': 8, 'experiments': experiments}

# ---- structural validation ----
assert len(experiments) == 224, len(experiments)
names = [e['name'] for e in experiments]
assert len(set(names)) == 224, "duplicate run names"
for e in experiments:
    assert set(e) == {'name', 'task', 'config'}, e['name']
    assert e['task'] in TASKS
    assert 'A' not in e['config'] and 'B' not in e['config']  # inert for nonlinear rules
main = experiments[:192]
assert len({(e['task'], e['config']['N_LAYER'], e['config']['MISS_LEN'],
             e['config']['MISSING_PROB']) for e in main}) == 2 * 2 * 2 * 3
ctrl = experiments[192:]
assert len(ctrl) == 32
assert all(e['config']['MISSING_PROB'] == 0 for e in ctrl)
assert len({(e['task'], e['config']['N_LAYER']) for e in ctrl}) == 4

# ---- merge-path validation through the real batch_run code ----
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
from batch_run import build_merged_config

base = json.load(open(os.path.join(REPO_ROOT, 'src', 'config.json')))
with tempfile.TemporaryDirectory() as d:
    for e in (experiments[0], experiments[96], experiments[223]):
        name, task, merged_main, merged = build_merged_config(e, base, d)
        assert name == e['name'] and task == e['task']
        assert merged_main['TASK'] == e['task']
        assert merged['_BATCH_RUN_MERGED'] is True
        assert merged_main['P'] == 127
        assert merged_main['MISSING_PROB'] == e['config']['MISSING_PROB']
        assert merged_main['N_LAYER'] == e['config']['N_LAYER']

with open(OUT, 'w') as f:
    json.dump(group, f, indent=1)
print(f"OK: wrote {len(experiments)} runs to {OUT}")
print("first:", experiments[0]['name'])
print("last: ", experiments[-1]['name'])
```

- [ ] **Step 2: 运行脚本**

```
C:/Users/Chen/anaconda3/envs/torch/python.exe .superpowers/sdd/core-split/gen_nonlinear_grid.py
```
预期：`OK: wrote 224 runs to ...`，first = `nonlinear_d256l1r4h4_P127_tr64ood128_randmiss0.1len1_seed17996`，last = `nonlinear_mul_d256l2r4h4_P127_tr64ood128_randmiss0len1_seed82585`。（batch_run 导入会连带 visualize/matplotlib，需用 torch 环境 python 运行。）

- [ ] **Step 3: 抽查生成产物**

```
C:/Users/Chen/anaconda3/envs/torch/python.exe -c "
import json
d = json.load(open('experiments/nonlinear_p127_tr64_ood128_randmisslen_d256h4_predict.json'))
exps = d['experiments']
print('concurrency:', d['concurrency'], 'runs:', len(exps))
import collections
print(collections.Counter((e['task'], e['config']['MISSING_PROB']) for e in exps))
"
```
预期：224 runs；主网格每 (task, prob) 组合 32 条（2 layers × 2 misslens × 8 seeds），prob=0 每 task 16 条（2 layers × 8 seeds，MISS_LEN 恒 1）。

- [ ] **Step 4: 提交（只提交 JSON，不提交脚本）**

```bash
git add experiments/nonlinear_p127_tr64_ood128_randmisslen_d256h4_predict.json
git commit -m "Add nonlinear randlen predict grid: nonlinear + nonlinear_mul x layers {1,2} x misslen {1,2} x probs {0.1,0.3,0.5} x 8 seeds + prob=0 control, 224 runs"
```

---

## Self-Review 记录

- Spec 覆盖：规则语义→Task 1；路由/配置→Task 2；网格+对照+命名+JSON→Task 3；测试与验证→各 Task 内嵌。§8 不做项均未出现在任何 Task 中。
- 占位符扫描：无 TBD/TODO；所有代码步骤含完整代码。
- 类型一致性：`nonlinear_mul` 在 rules.py 三函数、experiment.py 两处、smoke 测试、JSON task 字段中拼写一致；`build_merged_config` 签名与 src/batch_run.py:106 实测一致。
