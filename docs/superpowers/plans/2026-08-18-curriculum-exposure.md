# 曝光课程实验（curriculum_exposure_p251）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 spec（docs/superpowers/specs/2026-08-18-curriculum-exposure-design.md）实现曝光课程实验：训练前冻结 WTE 的能力、chain 命名阶段引用、36 个实验的配置文件。

**Architecture:** 两个 ~10 行代码改动（training.py 从头冻结、chain_run.py `@<stage名>` 引用）+ 三个实验 JSON（baselines batch + l1/l2 两条 chain）。复用已有 INIT_FROM 部分加载与 chain gate（tests/test_chain_run.py 已覆盖）。

**Tech Stack:** Python 3.9, PyTorch（本机用 `C:/Users/Chen/anaconda3/envs/torch/python.exe` 运行一切测试；base 环境无 torch）。

## Global Constraints

- 测试一律用 `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_x.py` 直接跑（每个测试文件有 `__main__` runner；torch 环境无 pytest）。
- 配置镜像 supplement 网格：覆盖键仅 `P/D_MODEL/N_HEAD/N_LAYER/MLP_RATIO/NUM_MASK/USE_AB_TAG/USE_CONDITIONAL_WTE/AB_PAIRS/MAX_UNIQUE_RATIO(S)/RANDOM_SEED/EPOCHS/EARLY_STOP_ACCURACY/INIT_FROM/COND_FIX/COND_FIX_START`，其余吃 `src/config.json` 默认。
- seeds 固定为 0 / 12345 / 999；层数后缀 l1=N_LAYER:1、l2=N_LAYER:2。
- C1/P1（单规则）也必须显式 `NUM_MASK: 2`（与混合组及参照网格口径一致，spec §3.2 注）。
- 不改 `evaluate` 的 loss mask 口径（有意不带 first_task_weight）；不动 BucketBatchSampler 的 epoch 内固定 batch 语义。
- git 提交需用户逐次确认（仓库惯例：每个完整改动一个 commit）。

---

### Task 1: training.py 从头冻结（cond_fix_start == 0）

**Files:**
- Modify: `src/training.py:264-273`（run_training_engine 初始化区）
- Test: `tests/test_chain_run.py`（追加两个测试函数）

**Interfaces:**
- Consumes: `freeze_partial(model, cond_fix)`（training.py:17，已存在）；`run_training_engine(..., cond_fix=None, cond_fix_start=None, ...)` 签名不变。
- Produces: `run_training_engine` 新语义——`cond_fix_start == 0` 时在首个梯度步之前冻结；`COND_FIX_START=0` 的实验配置（Task 4 的 T2 组）依赖它。

- [ ] **Step 1: 写失败测试（追加到 tests/test_chain_run.py 末尾，`if __name__` 块之前）**

```python
def test_freeze_from_start():
    """COND_FIX_START=0 must freeze WTE before the first gradient step."""
    import experiment
    from training import run_training_engine
    cfg = {'main': {'P': 7, 'A': 1, 'B': 1, 'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
                    'BATCH_SIZE': 32, 'TRAIN_LEN': 8, 'OOD_LEN': 10, 'DROPOUT': 0.0,
                    'ENTROPY_PENALTY_WEIGHT': 0.0, 'WEIGHT_DECAY': 0.1,
                    'MAX_UNIQUE_RATIO': 0.7}}
    ctx = experiment._prepare_single_recurrence(cfg, 'addition')
    model = ctx['model']
    wte_before = model.transformer.wte.weight.detach().clone()
    attn_before = model.transformer.h[0].attn.c_attn.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        with contextlib.redirect_stdout(io.StringIO()):
            run_training_engine(
                model, ctx['train_loader'], ctx['test_loader'], optimizer, scheduler, 'cpu',
                epochs=2, eval_interval=1, early_stop_accuracy=2.0,
                early_stop_no_improve=3000,
                save_path=os.path.join(tmpdir, 'm.pth'), save_config=ctx['save_config'],
                num_mask=ctx['num_mask'], extra_kwargs_fn=ctx['extra_kwargs_fn'],
                first_task_weight=1.0,
                cond_fix='WTE', cond_fix_start=0)
    # WTE frozen from the start: unchanged (lm_head is tied, same tensor);
    # attention weights must still have trained
    assert torch.equal(model.transformer.wte.weight, wte_before)
    assert not torch.equal(model.transformer.h[0].attn.c_attn.weight, attn_before)
```

并在 `__main__` 块的调用列表加一行 `test_freeze_from_start()`。

- [ ] **Step 2: 跑测试确认失败**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_chain_run.py`
Expected: FAIL — `test_freeze_from_start` 中断言 `torch.equal(wte_after, wte_before)` 不成立（现有代码 `cond_fix_start == 0` 走旧阈值分支，在第 0 个 epoch 的 eval 处才冻结，WTE 已被训练一个 epoch；注意：若当前实现恰在 epoch 0 eval 时触发冻结，失败可能表现为 WTE 与 AdamW 第一步已有差异）。

- [ ] **Step 3: 实现（src/training.py，插在 `high_acc_epoch = None` 之后、`for epoch in range(epochs):` 之前）**

```python
    # cond_fix_start == 0: freeze BEFORE the first gradient step. (Any other
    # start value keeps the threshold semantics: freeze at the first eval
    # whose accuracy reaches it.)
    if cond_fix is not None and cond_fix_start == 0:
        frozen_param_states = freeze_partial(model, cond_fix)
        cond_fix_triggered = True
        print(f"[CondFix] frozen from start (cond_fix_start=0): {cond_fix}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_chain_run.py`
Expected: `ALL TESTS PASSED: test_chain_run.py`（含既有 3 个测试不回归）

- [ ] **Step 5: 全套测试 + 提交（先经用户确认）**

Run: `for f in tests/test_*.py; do C:/Users/Chen/anaconda3/envs/torch/python.exe "$f" > /dev/null 2>&1 && echo "PASS $f" || echo "FAIL $f"; done`
Expected: 9 个文件全 PASS
Commit: `git add src/training.py tests/test_chain_run.py && git commit -m "Freeze-from-start: COND_FIX_START=0 freezes before the first gradient step" -- src/training.py tests/test_chain_run.py`

---

### Task 2: chain_run.py 命名阶段引用（@<stage名>）

**Files:**
- Modify: `src/chain_run.py:52-78`（`_resolve_init_from` 与 `main()` 中的调用点、`prev_stage` 变量）
- Test: `tests/test_chain_run.py`（追加一个三阶段链测试）

**Interfaces:**
- Consumes: `_seed_suffix(name)`（chain_run.py 已存在）；chain JSON 的 `stages` 列表结构。
- Produces: `INIT_FROM` 引用语法——`"@prev"`（前一阶段，向后兼容）与 `"@<stage名>"`（指定已完成的任一阶段）；Task 4 的 chain JSON 依赖 `"@phase1"`。

- [ ] **Step 1: 写失败测试（追加到 tests/test_chain_run.py）**

```python
def test_chain_named_stage_ref():
    """INIT_FROM '@<stage>' resolves to the named earlier stage, not just the
    previous one (stage 3 referencing stage 1)."""
    main = {'P': 7, 'A': 1, 'B': 1, 'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
            'BATCH_SIZE': 32, 'EPOCHS': 2, 'LR': 0.001, 'TRAIN_LEN': 8,
            'OOD_LEN': 10, 'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0}
    mixed = {**main, 'AB_PAIRS': [[1, 1], [1, 2]], 'USE_AB_TAG': False,
             'MIXED_AB_MAX_UNIQUE_RATIOS': [0.7, 0.7]}
    chain = {'stages': [
        {'name': 's1', 'experiments': [{'name': 'c_A_seed0', 'task': 'addition',
                                        'config': dict(main)}]},
        {'name': 's2', 'experiments': [{'name': 'c_AB_seed0', 'task': 'mixed_ab',
                                        'config': {**mixed, 'INIT_FROM': '@s1'}}]},
        {'name': 's3', 'experiments': [{'name': 'c_ABF_seed0', 'task': 'mixed_ab',
                                        'config': {**mixed, 'INIT_FROM': '@s1'}}]},
    ]}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'named_chain.json')
        with open(path, 'w') as f:
            json.dump(chain, f)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _run_chain(path, tmpdir)
        out = buf.getvalue()
        stem = os.path.splitext(os.path.basename(path))[0]
        for stage, exp in (('s2', 'c_AB_seed0'), ('s3', 'c_ABF_seed0')):
            ckpt = os.path.join(tmpdir, 'models', f'{stem}_{stage}', f'{exp}.pth')
            assert os.path.exists(ckpt), f'{stage} checkpoint missing'
        # both downstream stages must reference the stage-1 checkpoint
        assert out.count('INIT_FROM <- s1/c_A_seed0') == 2 or \
               out.count("INIT_FROM <- c_A_seed0") == 2, out
```

并在 `__main__` 调用列表加 `test_chain_named_stage_ref()`。

- [ ] **Step 2: 跑测试确认失败**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_chain_run.py`
Expected: FAIL — `ValueError: c_AB_seed0: cannot resolve INIT_FROM '@s1'`（现有实现只认 `@prev`，`@s1` 不匹配任何分支；注意 s3 的 `@s1` 在旧实现里会被当成“前一阶段 = s2”解析或报无匹配，具体报错以实际输出为准）。

- [ ] **Step 3: 实现（src/chain_run.py，整体替换 `_resolve_init_from`，并改 main 调用点）**

```python
def _resolve_init_from(stage_exps, stages_so_far, chain_stem, model_base_dir):
    """Replace INIT_FROM='@...' with the same-seed checkpoint path from an
    earlier completed stage, in place. '@prev' = the immediately previous
    stage; '@<name>' = any earlier stage with that name."""
    refs = {e['name']: e['config']['INIT_FROM'][1:] for e in stage_exps
            if isinstance(e.get('config', {}).get('INIT_FROM'), str)
            and e['config']['INIT_FROM'].startswith('@')}
    for exp in stage_exps:
        if exp['name'] not in refs:
            continue
        ref = refs[exp['name']]
        if ref == 'prev':
            stage = stages_so_far[-1] if stages_so_far else None
        else:
            stage = next((s for s in stages_so_far if s['name'] == ref), None)
        if stage is None:
            raise ValueError(
                f"{exp['name']}: cannot resolve INIT_FROM '@{ref}' — "
                f"no such earlier stage")
        suffix = _seed_suffix(exp['name'])
        candidates = [e for e in stage['experiments']
                      if suffix is not None and e['name'].endswith(suffix)]
        if len(candidates) != 1:
            raise ValueError(
                f"{exp['name']}: cannot resolve INIT_FROM '@{ref}' — need exactly one "
                f"experiment ending with '{suffix}' in stage {stage['name']}")
        src = candidates[0]
        src_path = os.path.join(model_base_dir, f"{chain_stem}_{stage['name']}",
                                f"{src['name']}.pth")
        exp['config']['INIT_FROM'] = src_path
        print(f"[chain] {exp['name']}: INIT_FROM <- {stage['name']}/{src['name']} ({src_path})")
```

`main()` 中：
- 删除 `prev_stage = None` 和循环末尾的 `prev_stage = stage`；
- 调用点改为 `_resolve_init_from(stage['experiments'], stages[:idx], chain_stem, args.model_base_dir)`。

- [ ] **Step 4: 跑测试确认通过**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_chain_run.py`
Expected: `ALL TESTS PASSED: test_chain_run.py`（5 个测试：原有 4 个 + 新 1 个）

- [ ] **Step 5: 全套测试 + 提交（先经用户确认）**

Run: 同 Task 1 Step 5 的全套命令；Expected: 9 全 PASS
Commit: `git add src/chain_run.py tests/test_chain_run.py && git commit -m "chain_run: named-stage INIT_FROM references (@<stage>), keep @prev shorthand" -- src/chain_run.py tests/test_chain_run.py`

---

### Task 3: baselines 实验 JSON（C1/C2/C3 × l1/l2 × 3 seeds = 18 runs）

**Files:**
- Create: `experiments/curriculum_exposure_p251_baselines.json`

**Interfaces:**
- Consumes: `batch_run.py` 的合并语义（main → 任务段 → 实验覆盖）；无 chain 语法。
- Produces: C1/C2/C3 三组的训练日志与 checkpoint，供结果对比（不依赖其他 task）。

- [ ] **Step 1: 写 JSON（生成器脚本跑一次，避免手写 18 条出错）**

在仓库根跑：

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe - <<'EOF'
import json
COMMON = {'P': 251, 'D_MODEL': 512, 'N_HEAD': 4, 'MLP_RATIO': 8,
          'NUM_MASK': 2, 'USE_AB_TAG': False, 'USE_CONDITIONAL_WTE': False}
SEEDS = [0, 12345, 999]
GROUPS = []
# C1: single rule A @ 0.1
for layer in (1, 2):
    for seed in SEEDS:
        GROUPS.append({'name': f'c1_d512l{layer}r8h4_P251_singleA_e0.1_seed{seed}',
                       'task': 'addition',
                       'config': {**COMMON, 'N_LAYER': layer, 'A': 1, 'B': 1,
                                  'MAX_UNIQUE_RATIO': 0.1, 'RANDOM_SEED': seed}})
# C2: from-scratch mixed 0.1A+0.3B
for layer in (1, 2):
    for seed in SEEDS:
        GROUPS.append({'name': f'c2_d512l{layer}r8h4_P251_N2_e0.1_0.3_seed{seed}',
                       'task': 'mixed_ab',
                       'config': {**COMMON, 'N_LAYER': layer,
                                  'AB_PAIRS': [[1, 1], [1, 2]],
                                  'MIXED_AB_MAX_UNIQUE_RATIOS': [0.1, 0.3],
                                  'RANDOM_SEED': seed}})
# C3: from-scratch mixed 0.1A+0.1B
for layer in (1, 2):
    for seed in SEEDS:
        GROUPS.append({'name': f'c3_d512l{layer}r8h4_P251_N2_e0.1_0.1_seed{seed}',
                       'task': 'mixed_ab',
                       'config': {**COMMON, 'N_LAYER': layer,
                                  'AB_PAIRS': [[1, 1], [1, 2]],
                                  'MIXED_AB_MAX_UNIQUE_RATIOS': [0.1, 0.1],
                                  'RANDOM_SEED': seed}})
json.dump({'experiments': GROUPS},
          open('experiments/curriculum_exposure_p251_baselines.json', 'w'), indent=2)
print(len(GROUPS), 'experiments written')
EOF
```

Expected output: `18 experiments written`

- [ ] **Step 2: 校验 JSON 结构与合并结果**

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe -c "
import json, sys
sys.path.insert(0, 'src')
from batch_run import build_merged_config
base = json.load(open('src/config.json'))
exps = json.load(open('experiments/curriculum_exposure_p251_baselines.json'))['experiments']
assert len(exps) == 18
for e in exps:
    name, task, mm, _ = build_merged_config(e, base, 'models/x')
    assert mm['P'] == 251 and mm['D_MODEL'] == 512 and mm['MLP_RATIO'] == 8
    assert mm['NUM_MASK'] == 2 and mm['USE_AB_TAG'] is False
    if task == 'addition':
        assert mm['MAX_UNIQUE_RATIO'] == 0.1 and mm['A'] == 1 and mm['B'] == 1
    else:
        assert mm['AB_PAIRS'] == [[1, 1], [1, 2]]
        assert mm['MIXED_AB_MAX_UNIQUE_RATIOS'] in ([0.1, 0.3], [0.1, 0.1])
print('BASELINES_JSON_OK')
"
```

Expected: `BASELINES_JSON_OK`

- [ ] **Step 3: 提交（先经用户确认）**

Commit: `git add experiments/curriculum_exposure_p251_baselines.json && git commit -m "Add curriculum exposure baselines grid (C1/C2/C3 x l1/l2 x 3 seeds, 18 runs)" -- experiments/curriculum_exposure_p251_baselines.json`

---

### Task 4: curriculum chain JSON ×2（P1→T1→T2 × l1/l2，每层 9 runs）

**Files:**
- Create: `experiments/curriculum_exposure_p251_l1.json`
- Create: `experiments/curriculum_exposure_p251_l2.json`

**Interfaces:**
- Consumes: `chain_run.py` 的 `@<stage名>` 引用（Task 2）、`COND_FIX_START: 0` 的从头冻结（Task 1）、`INIT_FROM` 部分加载与 chain gate（既有）。
- Produces: P1/T1/T2 三组日志与 checkpoint；T1/T2 共享 P1 checkpoint（paired）。

- [ ] **Step 1: 写两个 JSON（生成器脚本）**

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe - <<'EOF'
import json
SEEDS = [0, 12345, 999]
def common(layer):
    return {'P': 251, 'D_MODEL': 512, 'N_HEAD': 4, 'N_LAYER': layer,
            'MLP_RATIO': 8, 'NUM_MASK': 2, 'USE_AB_TAG': False,
            'USE_CONDITIONAL_WTE': False}
for layer in (1, 2):
    tag = f'd512l{layer}r8h4_P251'
    phase1 = [{'name': f'p1_{tag}_singleB_e0.3_seed{s}', 'task': 'addition',
               'config': {**common(layer), 'A': 1, 'B': 2,
                          'MAX_UNIQUE_RATIO': 0.3, 'RANDOM_SEED': s,
                          'EARLY_STOP_ACCURACY': 0.99}}
              for s in SEEDS]
    mixed_cfg = lambda s: {**common(layer), 'AB_PAIRS': [[1, 1], [1, 2]],
                           'MIXED_AB_MAX_UNIQUE_RATIOS': [0.1, 0.3],
                           'RANDOM_SEED': s}
    t1 = [{'name': f't1_{tag}_N2_e0.1_0.3_seed{s}', 'task': 'mixed_ab',
           'config': {**mixed_cfg(s), 'INIT_FROM': '@phase1'}} for s in SEEDS]
    t2 = [{'name': f't2_{tag}_N2_e0.1_0.3_frzWTE_seed{s}', 'task': 'mixed_ab',
           'config': {**mixed_cfg(s), 'INIT_FROM': '@phase1',
                      'COND_FIX': 'WTE', 'COND_FIX_START': 0}} for s in SEEDS]
    chain = {'stages': [
        {'name': 'phase1', 'gate': {'min_best_accuracy': 0.99}, 'experiments': phase1},
        {'name': 'T1', 'experiments': t1},
        {'name': 'T2', 'experiments': t2}]}
    path = f'experiments/curriculum_exposure_p251_l{layer}.json'
    json.dump(chain, open(path, 'w'), indent=2)
    print(path, 'written')
EOF
```

Expected output: 两行 `... written`

- [ ] **Step 2: 校验 JSON 结构 + @phase1 可解析性（干跑 chain_run 的引用解析）**

```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe -c "
import json, sys
sys.path.insert(0, 'src')
import chain_run
for layer in (1, 2):
    chain = json.load(open(f'experiments/curriculum_exposure_p251_l{layer}.json'))
    stages = chain['stages']
    assert [s['name'] for s in stages] == ['phase1', 'T1', 'T2']
    assert stages[0]['gate']['min_best_accuracy'] == 0.99
    # 干跑引用解析（不训练）：T1/T2 的 @phase1 必须各解析出 3 条
    for i in (1, 2):
        chain_run._resolve_init_from(stages[i]['experiments'], stages[:i], f'curriculum_exposure_p251_l{layer}', '/data/cxm/models')
        for e in stages[i]['experiments']:
            p = e['config']['INIT_FROM']
            assert p.endswith(f\"p1_d512l{layer}r8h4_P251_singleB_e0.3_seed{e['config']['RANDOM_SEED']}.pth\"), p
    # T2 冻结键
    for e in stages[2]['experiments']:
        assert e['config']['COND_FIX'] == 'WTE' and e['config']['COND_FIX_START'] == 0
print('CHAIN_JSONS_OK')
"
```

Expected: `CHAIN_JSONS_OK`（注意：该干跑会就地改写 JSON 对象中的 INIT_FROM 为绝对路径，但只在内存中，不写回文件）

- [ ] **Step 3: 提交（先经用户确认）**

Commit: `git add experiments/curriculum_exposure_p251_l1.json experiments/curriculum_exposure_p251_l2.json && git commit -m "Add curriculum exposure chains (P1->T1->T2 x l1/l2, 9 runs each)" -- experiments/curriculum_exposure_p251_l1.json experiments/curriculum_exposure_p251_l2.json`

---

### Task 5: 文档同步与总验证

**Files:**
- Modify: `recursion_model_and_training_settings.md`（§2 表加 COND_FIX 行）
- Modify: `src/chain_run.py` 顶部 docstring（引用语法说明更新为 @prev/@<stage名>）

**Interfaces:**
- Consumes: Task 1/2 的最终语义。
- Produces: 文档与代码一致。

- [ ] **Step 1: settings 文档 §2 表加一行（紧跟 INIT_FROM 行之后）**

```markdown
| `COND_FIX` / `COND_FIX_START` | 部分参数冻结：`COND_FIX: "WTE"` 冻结 embedding 相关参数（含共享的 lm_head 权重；`"LINEAR"` 冻结 transformer 主干+rule_head）。`COND_FIX_START` 为准确率阈值（达到后于该 eval 处冻结）；**`0` 表示训练前就冻结**（首个梯度步之前生效） |
```

- [ ] **Step 2: chain_run.py docstring 更新引用语法说明**

将 docstring 中 `- "INIT_FROM": "@prev" in an experiment's config resolves to ...` 一段替换为：

```
- "INIT_FROM": "@<stage>" in an experiment's config resolves to the
  checkpoint of the experiment with the same trailing "_seed<N>" name
  suffix from the named earlier stage; "@prev" is shorthand for the
  immediately previous stage (error if there is no match).
```

- [ ] **Step 3: 全套测试 + py_compile + 提交（先经用户确认）**

Run:
```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe -m py_compile src/training.py src/chain_run.py && for f in tests/test_*.py; do C:/Users/Chen/anaconda3/envs/torch/python.exe "$f" > /dev/null 2>&1 && echo "PASS $f" || echo "FAIL $f"; done
```
Expected: 9 全 PASS
Commit: `git add recursion_model_and_training_settings.md src/chain_run.py && git commit -m "Docs: COND_FIX_START=0 freeze-from-start semantics and @<stage> chain reference syntax" -- recursion_model_and_training_settings.md src/chain_run.py`

---

## 执行后（本计划范围外，仅提示）

在服务器上按 spec §5 顺序运行：
`batch_run.py experiments/curriculum_exposure_p251_baselines.json`（可并行）+
`chain_run.py experiments/curriculum_exposure_p251_l1.json` 与 `_l2.json`。
