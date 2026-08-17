# mixed_ab 暴露率补充实验组（P=251）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成并校验 `experiments/mixed_ab_exposure_p251_supplement.json`（30 runs：l2 0.1+0.1 补充、0.1+x 与 x+x 暴露率边界扫描）。

**Architecture:** 纯数据交付——一次性脚本生成 new-format 组 JSON 并做结构 + 合并路径校验；零代码改动。

**Tech Stack:** Python 3, JSON。

**Spec:** `docs/superpowers/specs/2026-08-14-mixed-ab-exposure-supplement-design.md`（已批准）

## Global Constraints

- 组文件为 new format：`{"concurrency": 8, "experiments": [...]}`；每条实验恰好三个字段 `name`/`task`/`config`，`task` 恒为 `"mixed_ab"`。
- 每条 run 的 config 键集合与原组 `experiments/mixed_ab_exposure_p251.json` 的条目完全一致（11 个键，取值除 N_LAYER / MIXED_AB_MAX_UNIQUE_RATIOS / RANDOM_SEED 外全部相同）：`RANDOM_SEED, P=251, D_MODEL=512, N_HEAD=4, N_LAYER, MLP_RATIO=8, AB_PAIRS=[[1,1],[1,2]], MIXED_AB_MAX_UNIQUE_RATIOS, USE_AB_TAG=false, USE_CONDITIONAL_WTE=false, NUM_MASK=2`。
- Seeds：`[0, 999, 12345]`。
- run 名模式：`mixed_basic_d512l{L}r8h4_P251_N2_e{r1}_{r2}_seed{seed}`（ratios 用 `str()` 格式化）；与原组 72 个 run 名不得冲突。
- 网格：A=[0.1,0.1]@l2 ×3 seeds；B=[0.1,x]@l1, x∈{0.2,0.3,0.4,0.5,0.6} ×3；C=[x,x]@l1, x∈{0.12,0.14,0.16,0.18} ×3。共 10 组合 × 3 = 30。
- 生成脚本放 scratch（`.superpowers/`，gitignored）不入库；只提交 JSON。
- torch 环境 python：`C:/Users/Chen/anaconda3/envs/torch/python.exe`（导入 batch_run 会连带 matplotlib）。
- 零代码改动；不包含启动 batch_run（用户在 GPU 机器执行）。

---

### Task 1: 生成并校验 30-run 补充组 JSON

**Files:**
- Create: `.superpowers/sdd/core-split/gen_mixed_ab_supplement.py`（一次性脚本，scratch 不入库）
- Create: `experiments/mixed_ab_exposure_p251_supplement.json`（生成产物，入库）

**Interfaces:**
- Consumes: `batch_run.build_merged_config(exp, base_config, model_dir) -> (name, task, merged_main, merged)`（src/batch_run.py:106）；原组文件 `experiments/mixed_ab_exposure_p251.json`（键集合与重名校验的基准）。
- Produces: `experiments/mixed_ab_exposure_p251_supplement.json`，30 条，用户以 `python src/batch_run.py experiments/mixed_ab_exposure_p251_supplement.json` 启动。

- [ ] **Step 1: 写生成+校验脚本**

创建 `.superpowers/sdd/core-split/gen_mixed_ab_supplement.py`：

```python
"""One-off generator + validator for the mixed_ab exposure supplement group.
Spec: docs/superpowers/specs/2026-08-14-mixed-ab-exposure-supplement-design.md
Scratch tool (lives under gitignored .superpowers/), not committed.
"""
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
OUT = os.path.join(REPO_ROOT, 'experiments', 'mixed_ab_exposure_p251_supplement.json')
ORIG = os.path.join(REPO_ROOT, 'experiments', 'mixed_ab_exposure_p251.json')

SEEDS = [0, 999, 12345]

# (n_layer, ratios) combos: A=l2 0.1+0.1, B=l1 0.1+x, C=l1 x+x
COMBOS = (
    [(2, [0.1, 0.1])]
    + [(1, [0.1, x]) for x in (0.2, 0.3, 0.4, 0.5, 0.6)]
    + [(1, [x, x]) for x in (0.12, 0.14, 0.16, 0.18)]
)


def entry(n_layer, ratios, seed):
    r1, r2 = ratios
    name = f"mixed_basic_d512l{n_layer}r8h4_P251_N2_e{r1}_{r2}_seed{seed}"
    # Key order matches the original p251 group's entries.
    config = {
        'RANDOM_SEED': seed,
        'P': 251,
        'D_MODEL': 512,
        'N_HEAD': 4,
        'N_LAYER': n_layer,
        'MLP_RATIO': 8,
        'AB_PAIRS': [[1, 1], [1, 2]],
        'MIXED_AB_MAX_UNIQUE_RATIOS': list(ratios),
        'USE_AB_TAG': False,
        'USE_CONDITIONAL_WTE': False,
        'NUM_MASK': 2,
    }
    return {'name': name, 'task': 'mixed_ab', 'config': config}


experiments = [entry(L, ratios, seed)
               for L, ratios in COMBOS for seed in SEEDS]
group = {'concurrency': 8, 'experiments': experiments}

# ---- structural validation ----
assert len(COMBOS) == 10, len(COMBOS)
assert len(experiments) == 30, len(experiments)
names = [e['name'] for e in experiments]
assert len(set(names)) == 30, "duplicate run names"
for e in experiments:
    assert set(e) == {'name', 'task', 'config'}, e['name']
    assert e['task'] == 'mixed_ab'

# key set identical to the original group's entries
orig = json.load(open(ORIG))
orig_keys = set(orig['experiments'][0]['config'].keys())
for e in experiments:
    assert set(e['config'].keys()) == orig_keys, \
        (e['name'], set(e['config'].keys()) ^ orig_keys)

# no name collision with the original 72 runs
orig_names = {e['name'] for e in orig['experiments']}
assert not (set(names) & orig_names), set(names) & orig_names

# grid coverage: each combo x 3 seeds
from collections import Counter
combo_count = Counter((e['config']['N_LAYER'],
                       tuple(e['config']['MIXED_AB_MAX_UNIQUE_RATIOS']))
                      for e in experiments)
assert len(combo_count) == 10 and set(combo_count.values()) == {3}

# ---- merge-path validation through the real batch_run code ----
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
from batch_run import build_merged_config

base = json.load(open(os.path.join(REPO_ROOT, 'src', 'config.json')))
with tempfile.TemporaryDirectory() as d:
    for e in (experiments[0], experiments[9], experiments[27]):  # A, B(x=0.4), C(x=0.18)
        name, task, merged_main, merged = build_merged_config(e, base, d)
        assert name == e['name'] and task == 'mixed_ab'
        assert merged_main['TASK'] == 'mixed_ab'
        assert merged['_BATCH_RUN_MERGED'] is True
        assert merged_main['P'] == 251
        assert merged_main['MIXED_AB_MAX_UNIQUE_RATIOS'] == e['config']['MIXED_AB_MAX_UNIQUE_RATIOS']
        assert merged_main['N_LAYER'] == e['config']['N_LAYER']
        # inherited from main defaults (not written in the entry)
        assert merged_main['TRAIN_LEN'] == 16 and merged_main['OOD_LEN'] == 32
        assert merged_main['EPOCHS'] == 6000 and merged_main['BATCH_SIZE'] == 512

with open(OUT, 'w') as f:
    json.dump(group, f, indent=1)
print(f"OK: wrote {len(experiments)} runs to {OUT}")
print("first:", experiments[0]['name'])
print("last: ", experiments[-1]['name'])
```

- [ ] **Step 2: 运行脚本**

```
C:/Users/Chen/anaconda3/envs/torch/python.exe .superpowers/sdd/core-split/gen_mixed_ab_supplement.py
```
预期：`OK: wrote 30 runs to ...`；first = `mixed_basic_d512l2r8h4_P251_N2_e0.1_0.1_seed0`；last = `mixed_basic_d512l1r8h4_P251_N2_e0.18_0.18_seed12345`。

- [ ] **Step 3: 抽查生成产物**

```
C:/Users/Chen/anaconda3/envs/torch/python.exe -c "
import json, collections
d = json.load(open('experiments/mixed_ab_exposure_p251_supplement.json'))
exps = d['experiments']
print('concurrency:', d['concurrency'], 'runs:', len(exps))
print(collections.Counter((e['config']['N_LAYER'], tuple(e['config']['MIXED_AB_MAX_UNIQUE_RATIOS'])) for e in exps))
"
```
预期：30 runs；10 个 (N_LAYER, ratios) 组合各 3 条——(2, (0.1,0.1))、(1, (0.1,x)) x∈{0.2..0.6}、(1, (x,x)) x∈{0.12,0.14,0.16,0.18}。

- [ ] **Step 4: 提交（只提交 JSON，不提交脚本）**

```bash
git add experiments/mixed_ab_exposure_p251_supplement.json
git commit -m "Add mixed_ab exposure supplement grid (P=251): l2 0.1+0.1, 0.1+x and x+x boundary scans, 30 runs"
```

---

## Self-Review 记录

- Spec 覆盖：网格/配置/命名/文件组织/生成校验全部落在 Task 1；§6 不做项均未出现。
- 占位符扫描：无 TBD/TODO；脚本完整可运行。
- 一致性：COMBOS（1+5+4=10）×3 seeds=30 与 spec 一致；抽查下标 0/9/27 按 COMBOS 顺序核算：A 占 0-2，B 占 3-17（0.2→3-5, 0.3→6-8, 0.4→9-11, 0.5→12-14, 0.6→15-17），C 占 18-29（0.12→18-20, 0.14→21-23, 0.16→24-26, 0.18→27-29）——0/9/27 分别落在 A(x=0.1@l2)、B(x=0.4)、C(x=0.18)。
