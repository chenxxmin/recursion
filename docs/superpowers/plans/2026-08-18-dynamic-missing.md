# dynamic_mixed + 污损（predict 模式）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 dynamic_mixed 任务增加缺失值污损（predict 模式：模型看污损视图，loss/准确率对干净真值），样本形如 `x1 x2 f3 x3 f4 M f5 x5 ...`（flag 完好，部分值位置被 M 替换）。

**Architecture:** `DynamicMixedDataset` 增加 `missing_prob/miss_len`，>0 时对值子序列 x3..x_L 做单规则同款 run 污损，样本变为 `(view, clean, loss_mask)`；新 `BatchTag.ACTION_MISS` + `dynamic_missing_collate_fn`；`_unpack_batch` 加分支同时返回 loss_mask 与 targets_override=clean；`_prepare_dynamic_mixed` 接线。设计依据：`docs/superpowers/specs/2026-08-18-dynamic-missing-design.md`（用户已批准）。

**Tech Stack:** Python 3 + torch。测试：`C:/Users/Chen/anaconda3/envs/torch/python.exe`（无 pytest，`__main__` runner）。

## Global Constraints

- 用户已授权**直接在 main 分支逐任务 git commit**（2026-08-18 确认）。
- **不得修改 `RecurrenceDataset._corrupt`**（tests/test_mixed_ab_compat.py 是其随机序列的 golden master）；dynamic 侧新写污损代码，用 `DynamicMixedDataset.self.rng`。
- 无污损路径（missing_prob=0）的样本格式 `(seq, loss_mask)`、随机序列、loss_mask 必须与现状**逐点一致**。
- M = p（值 < p、flag ≥ p+1、p 从不出现在 dynamic 序列中）；模型 vocab 不变。
- 只做 predict 模式；MISS_SECOND 不支持（设置了打 warning 忽略）；stage-3 终测无 dynamic 分支，不用动；rule_fit 等分析工具不涉及。
- 全项目 DYNAMIC→ACTION 重命名是后续事项，**不在本计划**；新 tag 直接叫 ACTION_MISS。
- 既有全部 tests/test_*.py 必须保持通过。

## 关键代码锚点（实现前必读）

- `src/datasets.py:313-319` BatchTag(IntEnum)：MIXED_AB=0 / DYNAMIC_MIXED=1 / PLAIN=2 / MIXED_AB_MASKED=3 / PLAIN_TARGET=4 / MIXED_AB_TARGET=5 → 新增 `ACTION_MISS = 6` 追加在末尾（不改已有编号）。
- `src/datasets.py:404-447` DynamicMixedDataset；`:375-400` generate_dynamic_sample（**不要改**，attention 分析共享它）。
- `src/datasets.py:450-453` dynamic_mixed_collate_fn（新 collate 放它后面）。
- `src/datasets.py:171-234` RecurrenceDataset._corrupt —— 语义参照（只读不改）：命中后 run 长 [1,miss_len]，run 后第一个位置保持干净，最长 run==miss_len 否则整样本重采（`max_run in (0, miss_len)` 接受全净样本），1000 次重试上限打 WARNING。
- `src/training.py:46-98` `_unpack_batch`：返回 `(x, loss_mask, kwargs, ab_labels, targets_override)`；docstring 的 tag 清单要同步加一行。
- `src/experiment.py:211-214` 现有两条 warning（删除）；`:237-244` 数据集构造；`:26` collate import 行；`:272-286` save_config。
- 序列布局：x1@0, x2@1, f_k@2k-4, x_k@2k-3（k≥3）；序列长 2L-2，loss_mask 长 2L-3，值目标位 `2*(k-1)`（k=2..L-1 对应 x3..x_L）。

---

### Task 1: 数据集污损 + ACTION_MISS tag + collate + 单测

**Files:**
- Modify: `src/datasets.py`（BatchTag :313-319、DynamicMixedDataset :404-447、collate :450 后）
- Test: `tests/test_dynamic_missing.py`（新建）

**Interfaces:**
- Produces（后续任务依赖）:
  - `BatchTag.ACTION_MISS = 6` — payload `(clean_seqs, loss_mask)`
  - `DynamicMixedDataset(p, ab_pairs, num_samples, length, seed=0, missing_prob=0.0, miss_len=1)`；missing_prob>0 时 `self.samples` 元素为 `(view, clean, loss_mask)`，否则仍为 `(seq, loss_mask)`
  - `dynamic_missing_collate_fn(batch) -> (x, BatchTag.ACTION_MISS, clean_stacked, mask_stacked)`

- [ ] **Step 1: 写失败测试**

新建 `tests/test_dynamic_missing.py`：

```python
"""Tests for DynamicMixedDataset missing-value corruption (predict mode)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from datasets import BatchTag, DynamicMixedDataset, dynamic_missing_collate_fn

P, PAIRS, L = 7, [(1, 1), (1, 2)], 12


def _make(missing_prob=0.5, miss_len=3, seed=42, n=200, length=L):
    return DynamicMixedDataset(p=P, ab_pairs=PAIRS, num_samples=n,
                               length=length, seed=seed,
                               missing_prob=missing_prob, miss_len=miss_len)


def test_missing_only_on_value_positions():
    ds = _make()
    n_masked = 0
    for view, clean, mask in ds.samples:
        v, c = view.tolist(), clean.tolist()
        for i, tok in enumerate(v):
            if i >= 2 and i % 2 == 0:
                assert tok >= P + 1, f"flag position {i} corrupted: {tok}"
            if tok == P:
                assert i >= 3 and i % 2 == 1, f"M at non-value position {i}"
                n_masked += 1
        # view differs from clean only at M positions; clean layout intact
        for a, b in zip(v, c):
            assert a == b or a == P
        assert all(tok < P for i, tok in enumerate(c) if i % 2 == 1 or i < 2)
        assert all(tok >= P + 1 for i, tok in enumerate(c) if i >= 2 and i % 2 == 0)
    assert n_masked > 0


def test_run_lengths_and_spacing():
    ds = _make(missing_prob=0.4, miss_len=3, n=300)
    seen_masked = 0
    for view, clean, mask in ds.samples:
        v = view.tolist()
        masked_ks = [k for k in range(3, L + 1) if v[2 * k - 3] == P]
        if not masked_ks:
            continue  # all-clean trials are accepted by design
        seen_masked += 1
        runs, start = [], masked_ks[0]
        for a, b in zip(masked_ks, masked_ks[1:]):
            if b != a + 1:
                runs.append((start, a))
                start = b
        runs.append((start, masked_ks[-1]))
        lengths = [b - a + 1 for a, b in runs]
        assert max(lengths) == 3, f"longest run {lengths} != miss_len 3"
        # runs are separated by at least one clean value
        for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
            assert a2 - b1 >= 2, f"adjacent runs merged: {runs}"
    assert seen_masked > 0


def test_seed_reproducibility():
    a = _make(seed=7, n=50)
    b = _make(seed=7, n=50)
    for (v1, c1, m1), (v2, c2, m2) in zip(a.samples, b.samples):
        assert torch.equal(v1, v2) and torch.equal(c1, c2) and torch.equal(m1, m2)


def test_no_missing_format_unchanged():
    ds = DynamicMixedDataset(p=P, ab_pairs=PAIRS, num_samples=20, length=L, seed=3)
    for item in ds.samples:
        assert isinstance(item, tuple) and len(item) == 2
        seq, mask = item
        assert all(tok != P for tok in seq.tolist())
        assert mask.sum().item() == L - 2  # x3..x_L targets only
        for k in range(2, L):
            assert mask[2 * (k - 1)] == 1.0


def test_collate_action_miss():
    ds = _make(n=4)
    x, tag, clean, mask = dynamic_missing_collate_fn(ds.samples[:4])
    assert tag == BatchTag.ACTION_MISS
    assert x.shape == clean.shape == (4, 2 * L - 2)
    assert mask.shape == (4, 2 * L - 3)


if __name__ == '__main__':
    test_missing_only_on_value_positions()
    test_run_lengths_and_spacing()
    test_seed_reproducibility()
    test_no_missing_format_unchanged()
    test_collate_action_miss()
    print("ALL TESTS PASSED: test_dynamic_missing.py")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_dynamic_missing.py`
Expected: ImportError（`dynamic_missing_collate_fn` 不存在）

- [ ] **Step 3: 实现**

`src/datasets.py` 三处：

(a) BatchTag 末尾追加（:319 之后）：
```python
    ACTION_MISS = 6    # payload: (clean_seqs, loss_mask) -- dynamic_mixed + missing (predict mode)
```

(b) DynamicMixedDataset 替换 __init__ 与 _generate_sample，并新增 _corrupt_values：
```python
    def __init__(self, p, ab_pairs, num_samples, length, seed=0,
                 missing_prob=0.0, miss_len=1):
        super().__init__()
        self.p = p
        self.ab_pairs = [tuple(pair) for pair in ab_pairs]
        self.num_ab_pairs = len(self.ab_pairs)
        self.num_samples = num_samples
        self.length = length
        self.pad_token_id = p
        self.flag_start_id = p + 1
        self.missing_prob = missing_prob
        self.miss_len = miss_len
        self.rng = random.Random(seed)
        self.samples = [self._generate_sample() for _ in range(num_samples)]

    def _generate_sample(self):
        """Generate one sequence with per-step random rules.

        Input format: [x1, x2, f3, x3, flag_4, x4, ..., f_L, x_L]
        where flag_k indicates which rule is used to generate x_k.
        Loss is computed only on x3..x_L; x1, x2 and all flags are masked.

        With missing_prob > 0 the item is (view, clean, loss_mask): view has
        some value positions replaced by the missing token p (predict mode —
        loss targets stay the clean values), clean is the untouched sequence.
        Otherwise the item is (seq, loss_mask) as before.
        """
        seq = generate_dynamic_sample(self.p, self.ab_pairs, self.flag_start_id,
                                      self.length, self.rng)

        # Build loss mask aligned to targets = seq[1:]
        # Input: [x1, x2, f3, x3, f4, x4, ..., f_L, x_L]
        # Target: [x2, f3, x3, f4, x4, ..., f_L, x_L]
        # Only x3, x4, ..., x_L should contribute to loss; x2 and all flags are masked.
        # Loop index k=2 generates x_3 (target index 2), k=3 generates x_4 (target index 4),
        # so x_{k+1} is at target index 2*(k-1).
        loss_mask = torch.zeros(len(seq) - 1, dtype=torch.float)
        for k in range(2, self.length):
            target_idx = 2 * (k - 1)
            loss_mask[target_idx] = 1.0

        seq_tensor = torch.tensor(seq, dtype=torch.long)
        if self.missing_prob <= 0:
            return seq_tensor, loss_mask
        view = seq_tensor.clone()
        self._corrupt_values(view)
        return view, seq_tensor, loss_mask

    def _corrupt_values(self, view):
        """Corrupt value positions x3..x_L (odd seq indices >= 3) in place with
        the missing token p. Mirrors RecurrenceDataset._corrupt's run model on
        the VALUE subsequence (flags are never corrupted): scan from x3; each
        value independently hits with probability missing_prob and corrupts a
        run of random length in [1, miss_len] consecutive values; the value
        right after a run stays clean; a run may be truncated at the end.
        The whole scan is redone until the longest run equals miss_len (an
        all-clean trial is also accepted, matching `max_run in (0, miss_len)`).
        Uses self.rng, so dataset seeding fully determines corruption.
        """
        L = self.length
        retries = 0
        while True:
            trial = view.clone()
            max_run = 0
            k = 3
            while k <= L:
                if self.rng.random() < self.missing_prob:
                    run = self.rng.randint(1, self.miss_len)
                    end = min(k + run, L + 1)  # exclusive value index
                    for q in range(k, end):
                        trial[2 * q - 3] = self.p
                    max_run = max(max_run, end - k)
                    k = end + 1  # the value right after a run stays clean
                else:
                    k += 1
            if max_run in (0, self.miss_len) or retries >= 1000:
                view.copy_(trial)
                if retries >= 1000:
                    print(f"WARNING: _corrupt_values gave up matching max run length "
                          f"{self.miss_len} after {retries} retries")
                break
            retries += 1
```

同时更新类 docstring（:405-409）末尾加一段：
```
    With missing_prob > 0, value positions x3..x_L are corrupted with the
    missing token p (runs of 1..miss_len consecutive values; flags stay clean)
    and items become (view, clean, loss_mask) triples routed through
    BatchTag.ACTION_MISS (predict mode: loss/accuracy targets are the clean
    values). train and test splits are corrupted alike.
```

(c) dynamic_mixed_collate_fn 之后（:453 后）新增：
```python
def dynamic_missing_collate_fn(batch):
    """(view, clean, loss_mask) samples (dynamic_mixed + MISSING_PROB) -> ACTION_MISS."""
    views = [item[0] for item in batch]
    cleans = [item[1] for item in batch]
    loss_masks = [item[2] for item in batch]
    return (torch.stack(views, dim=0), BatchTag.ACTION_MISS,
            torch.stack(cleans, dim=0), torch.stack(loss_masks, dim=0))
```

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run:
```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_dynamic_missing.py
for f in tests/test_*.py; do C:/Users/Chen/anaconda3/envs/torch/python.exe "$f" || exit 1; done
```
Expected: 全部 ALL TESTS PASSED

- [ ] **Step 5: Commit**

```bash
git add src/datasets.py tests/test_dynamic_missing.py
git commit -m "feat(datasets): dynamic_mixed missing-value corruption + ACTION_MISS tag (Task 1)"
```

---

### Task 2: `_unpack_batch` ACTION_MISS 分支

**Files:**
- Modify: `src/training.py:46-98`（一个分支 + docstring 清单）
- Test: `tests/test_dynamic_missing.py`（追加）

**Interfaces:**
- Consumes: `BatchTag.ACTION_MISS`、`dynamic_missing_collate_fn`（Task 1）
- Produces: `_unpack_batch([x, BatchTag.ACTION_MISS, clean, loss_mask], device, None)` → `(x, loss_mask, {}, None, clean)`

- [ ] **Step 1: 追加失败测试**

`tests/test_dynamic_missing.py` 追加（import 行加 `from training import _unpack_batch`）：

```python
def test_unpack_action_miss():
    ds = _make(n=4)
    batch = dynamic_missing_collate_fn(ds.samples[:4])
    x, loss_mask, kwargs, ab_labels, targets_override = _unpack_batch(list(batch), 'cpu', None)
    assert torch.equal(x, batch[0])
    assert torch.equal(loss_mask, batch[3])
    assert torch.equal(targets_override, batch[2])
    assert kwargs == {} and ab_labels is None
    try:
        _unpack_batch([batch[0], BatchTag.ACTION_MISS, batch[2]], 'cpu', None)
        assert False, "expected ValueError for malformed payload"
    except ValueError:
        pass
```

`__main__` runner 追加 `test_unpack_action_miss()`。

- [ ] **Step 2: 跑测试确认失败**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_dynamic_missing.py`
Expected: ValueError（Unknown batch tag: 6）

- [ ] **Step 3: 实现**

`src/training.py` `_unpack_batch` 中，PLAIN_TARGET 分支（:88-91）之后插入：

```python
    if tag == BatchTag.ACTION_MISS:
        if len(batch) != 4:
            raise ValueError(f"ACTION_MISS batch must carry exactly (clean_seqs, loss_mask), got {len(batch) - 2} payload item(s)")
        return x, batch[3].to(device), {}, None, batch[2].to(device)
```

docstring 的 tag 清单（:53-56 区域）在 `DYNAMIC_MIXED  -> (loss_mask,)` 后加一行：
```
      ACTION_MISS    -> (clean_seqs, loss_mask); targets_override = clean_seqs
```

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: 同 Task 1 Step 4 的两条命令。

- [ ] **Step 5: Commit**

```bash
git add src/training.py tests/test_dynamic_missing.py
git commit -m "feat(training): unpack ACTION_MISS batches (clean targets + loss mask) (Task 2)"
```

---

### Task 3: `_prepare_dynamic_mixed` 接线 + 示例实验 JSON + 冒烟

**Files:**
- Modify: `src/experiment.py`（:211-214 warning 块、:237-248 数据集与 loader、:26 import 行、:272-286 save_config）
- Create: `experiments/dynamic_missing.json`
- Test: `tests/test_dynamic_missing.py`（追加冒烟）

**Interfaces:**
- Consumes: Task 1/2 全部产物。
- Produces: dynamic_mixed + MISSING_PROB>0 时 run_experiment 全链路跑通；save_config 含 `missing_prob`/`miss_len` 键。

- [ ] **Step 1: 追加失败测试（端到端冒烟）**

`tests/test_dynamic_missing.py` 追加：

```python
def test_run_experiment_dynamic_missing_smoke():
    import contextlib
    import io
    import json
    import tempfile
    from core import run_experiment
    main = {
        'P': 7, 'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
        'BATCH_SIZE': 32, 'EPOCHS': 2, 'LR': 0.001, 'RANDOM_SEED': 42,
        'TRAIN_LEN': 8, 'OOD_LEN': 10, 'DROPOUT': 0.0,
        'WEIGHT_DECAY': 0.1, 'ENTROPY_PENALTY_WEIGHT': 0.0,
        'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0,
        'EARLY_STOP_NO_IMPROVE': 3000,
        'USE_LEARNABLE_PE': False, 'MLP_RATIO': 4,
        'TASK': 'dynamic_mixed', 'AB_PAIRS': [[1, 1], [1, 2]],
        'NUM_TRAIN_SAMPLES': 200, 'NUM_TEST_SAMPLES': 50,
        'MISSING_PROB': 0.3, 'MISS_LEN': 2,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        main['SAVE_PATH'] = os.path.join(tmpdir, 'model.pth')
        cfg = {'main': main, '_BATCH_RUN_MERGED': True}
        cfg_path = os.path.join(tmpdir, 'config.json')
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_experiment(cfg_path)
        out = buf.getvalue()
        assert os.path.exists(main['SAVE_PATH']), "checkpoint was not saved"
        assert 'Train: Loss=' in out, "training loop output missing"
        assert 'nan' not in out.lower(), "NaN appeared in training"
```

`__main__` runner 追加调用。

- [ ] **Step 2: 跑测试确认失败**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_dynamic_missing.py`
Expected: 冒烟失败（当前 MISSING_PROB 被 warning 忽略，样本是无污损二元组 → 不报错但测不到污损路径；实现者注意：若测试意外通过，说明污损路径未被真正接线——此时应在冒烟测试里加一条对训练输出或数据集的直接断言，例如构造同样配置的数据集检查样本为三元组，再确认失败）

- [ ] **Step 3: 实现**

`src/experiment.py`：

(a) :26 import 行把 `dynamic_missing_collate_fn` 加入 datasets import 列表。

(b) 删除 :211-214 两条 warning，替换为：
```python
    missing_prob = cfg.get('MISSING_PROB', 0.0)
    miss_len = cfg.get('MISS_LEN', 1)
    if missing_prob > 0 and cfg.get('MISS_SECOND', False):
        print("WARNING: MISS_SECOND is not supported for dynamic_mixed; ignoring it.")
```
（PREDICT_MISSING 不需要：dynamic + MISSING_PROB 隐含 predict 语义。）

(c) :237-244 数据集构造替换为：
```python
    train_dataset = DynamicMixedDataset(
        p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TRAIN_SAMPLES,
        length=TRAIN_LEN, seed=cfg.get('RANDOM_SEED', 42),
        missing_prob=missing_prob, miss_len=miss_len
    )
    test_dataset = DynamicMixedDataset(
        p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TEST_SAMPLES,
        length=OOD_LEN, seed=cfg.get('RANDOM_SEED', 42) + 1,
        missing_prob=missing_prob, miss_len=miss_len
    )
```

(d) :248 loader 行替换为：
```python
    collate = dynamic_missing_collate_fn if missing_prob > 0 else dynamic_mixed_collate_fn
    train_loader, test_loader = _make_loaders(train_dataset, test_dataset, BATCH_SIZE, collate)
```

(e) save_config（:272-286）`'ood_len': OOD_LEN,` 后加：
```python
        'missing_prob': missing_prob,
        'miss_len': miss_len,
```

(f) 新建 `experiments/dynamic_missing.json`（服务器上可直接批次跑的最小示例）：
```json
{
  "experiments": [
    {
      "name": "dynamic_a11a12_P53_miss0.1len3_seed42",
      "task": "dynamic_mixed",
      "config": {
        "AB_PAIRS": [[1, 1], [1, 2]],
        "P": 53,
        "MISSING_PROB": 0.1,
        "MISS_LEN": 3,
        "RANDOM_SEED": 42
      }
    }
  ]
}
```

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: 同 Task 1 Step 4 的两条命令。

- [ ] **Step 5: Commit**

```bash
git add src/experiment.py tests/test_dynamic_missing.py experiments/dynamic_missing.json
git commit -m "feat(experiment): wire MISSING_PROB into dynamic_mixed (predict mode) (Task 3)"
```

---

### Task 4: 文档同步

**Files:**
- Modify: `recursion_model_and_training_settings.md`（dynamic_mixed 相关段落补充 missing 支持说明——先 grep 定位 dynamic 段落）
- Modify: `REFACTOR_LOG.md`（追加 #54）

- [ ] **Step 1: recursion_model_and_training_settings.md**

在 dynamic_mixed 任务的描述处补充：MISSING_PROB>0 时按 predict 模式污损值位置（run 模型同单规则，连续值位置、最长 run==miss_len、flag 不动、M=p、train/test 都污损），样本为 (view, clean, loss_mask) 经 BatchTag.ACTION_MISS；MISS_SECOND 不支持。

- [ ] **Step 2: REFACTOR_LOG.md 追加**

```markdown
## 54. 【新功能】dynamic_mixed 支持缺失值污损（predict 模式）

**背景**：dynamic_mixed（每步规则可变、flag 交错布局）此前 MISSING_PROB 只打 warning 并忽略。实验需要"flag 完好、部分值位置被 M 替换"的样本。

**修改**：
- `DynamicMixedDataset` 新增 missing_prob/miss_len：>0 时对值子序列 x3..x_L 做单规则同款 run 污损（连续值位置、最长 run==miss_len 否则重采、全净接受），样本变为 (view, clean, loss_mask)；predict 语义隐含（loss 对干净真值，无 PREDICT_MISSING 开关）；flag 位永不污损；M=p（vocab 已含，模型零改动）。
- 新 `BatchTag.ACTION_MISS`（payload (clean, loss_mask)）+ `dynamic_missing_collate_fn`；`_unpack_batch` 新分支同时返回 loss_mask 与 targets_override=clean。
- `_prepare_dynamic_mixed` 接线（删除两条旧 warning；MISS_SECOND 打 warning 忽略）；save_config 增加 missing_prob/miss_len。
- 新增 `experiments/dynamic_missing.json` 示例。
- 设计文档：docs/superpowers/specs/2026-08-18-dynamic-missing-design.md。

**验证**：tests/test_dynamic_missing.py（污损位置 / run 约束与间隔 / seed 可复现 / 无污损格式不变 / collate / unpack / run_experiment 端到端冒烟）；全量测试回归通过。
```

- [ ] **Step 3: Commit**

```bash
git add recursion_model_and_training_settings.md REFACTOR_LOG.md
git commit -m "docs: dynamic_mixed missing-corruption notes + REFACTOR_LOG #54 (Task 4)"
```

---

## Self-Review 记录

- Spec 覆盖：predict-only ✓、连续值位置 run 污损 ✓、最长 run==miss_len ✓、train/test 都污损 ✓、ACTION_MISS 命名 ✓、MISS_SECOND warning ✓、save_config 字段 ✓、无污损路径不变（Global Constraints + test_no_missing_format_unchanged）✓、分析工具/重命名不做 ✓。
- 类型一致：collate 输出 `(x, tag, clean, mask)` ↔ `_unpack_batch` 返回 `(x, mask, {}, None, clean)`（batch[3]=mask、batch[2]=clean）✓；数据集三元组 `(view, clean, loss_mask)` 顺序与 collate 解包一致 ✓。
- 冒烟测试的 RED 步骤加了实现者注意事项（旧代码下可能"假通过"，需加三元组断言保证 RED 真实）。
- 占位符扫描：无 TBD；每步含完整代码/命令。
