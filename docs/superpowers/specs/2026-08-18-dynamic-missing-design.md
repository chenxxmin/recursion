# dynamic_mixed + 污损（predict 模式）设计

日期：2026-08-18
状态：已与用户确认

## 背景与目标

给 dynamic_mixed 任务（每步规则可变，布局 `[x1, x2, f3, x3, ..., f_L, x_L]`，值在奇数下标、flag 在偶数下标）增加缺失值污损能力，样本形如：

```
x0 x1 rule1 x2 rule2 M rule1 x4 rule1 x5 rule1 M ...
```

即 flag 原样、部分值位置被 M 替换。用于训练/评估链路（rule_fit 等分析工具本次不涉及）。

## 已确认的需求决策

| 决策点 | 结论 |
|---|---|
| 计量模式 | 只做 **predict 模式**（模型看污损视图，loss/准确率对干净真值）；`MISSING_PROB>0` 即隐含 predict 语义，不需要 PREDICT_MISSING 开关 |
| miss_len "连续" | 连续**值位置**（中间 flag 不动）；沿用单规则的"最长 run == miss_len 否则重采"约束 |
| MISS_SECOND | 不支持（设置了打 warning 忽略） |
| 污损范围 | train/test 都污损 |
| 分析工具 | 本次不做 |
| 新 BatchTag 名 | `ACTION_MISS`（用户指定）；全项目 DYNAMIC→ACTION 重命名为后续事项，不在本次范围 |

## 关键现状事实（实现依据）

- `DynamicMixedDataset`（src/datasets.py:404）：`pad_token_id = p` 从不出现在序列中（值 < p，flag >= p+1），**M 直接用 p，模型 vocab（p+1+N）已包含，模型零改动**。
- predict 语义机制现成：`_unpack_batch`（src/training.py:46）支持 `targets_override`（PLAIN_TARGET 分支在用）与 loss_mask；两者同时存在的分支需新增。
- dynamic_mixed 在 stage-3 无分支（src/experiment.py:507-514 仅 mixed_ab/single_recurrence）→ 终测不用动。
- `RecurrenceDataset` 的污损扫描逻辑有 golden-master 测试（tests/test_mixed_ab_compat.py）钉住随机序列 → **不得修改**，dynamic 侧新写污损代码。
- `_prepare_dynamic_mixed`（src/experiment.py:201）目前对 MISSING_PROB/PREDICT_MISSING 打 warning 并忽略（:211-214），本次替换为真正接线。

## 设计

### 1. 数据集（src/datasets.py，DynamicMixedDataset）

- 新增参数 `missing_prob=0.0, miss_len=1`（默认行为与现状逐点一致）。
- `missing_prob > 0` 时：
  - 对每个生成样本，在值子序列 x3..x_L 上扫描：每个值位置以 prob 命中 → 污损连续 `random [1, miss_len]` 个值位置（flag 不动）；整样本最长 run ≠ miss_len 则重采（同单规则约束）。
  - 随机数用已有的 `self.rng`（`random.Random(seed)`）推进，同 seed 可复现。
  - 样本变为 `(view, clean, loss_mask)` 三元组：view = 污损后序列（M = p），clean = 原始完整序列，loss_mask 与无污损版相同（x3..x_L 目标位为 1，无论该位是否被污损——干净位测正常预测、污损位测补缺，与单规则 predict 模式一致）。
- 无污损路径的样本格式、随机序列、loss_mask 完全不变。

### 2. BatchTag + collate（src/datasets.py / src/training.py）

- `BatchTag` 新增成员 `ACTION_MISS`。
- 新 collate `dynamic_missing_collate_fn(batch)` → `(x, BatchTag.ACTION_MISS, clean, loss_mask)`。
- `_unpack_batch` 新增分支：`ACTION_MISS -> (clean_seqs, loss_mask)` payload，返回 `x, loss_mask, {}, None, clean`。

### 3. 接线（src/experiment.py，_prepare_dynamic_mixed）

- 删除现有两条 warning（MISSING_PROB / PREDICT_MISSING "only supported for..."）。
- `MISSING_PROB > 0`：读 `MISS_LEN`（默认 1），带污损构造数据集，collate 用 `dynamic_missing_collate_fn`。
- `MISS_SECOND=True` 被设置 → warning 并忽略。
- `save_config` 增加 `missing_prob` / `miss_len` 字段（checkpoint 自描述）。
- 模型 vocab 不变。

### 4. 测试（新 tests/test_dynamic_missing.py，`__main__` runner，无 pytest）

- M 只出现在奇数下标 ≥ 3 的值位；flag 位永不污损；view 与 clean 仅 M 位不同
- run 长度 ∈ [1, miss_len]，且每样本最长 run == miss_len
- 同 seed 两数据集样本逐点一致；`missing_prob=0` 时样本与旧版逐点一致（回归保护）
- loss_mask 仅值目标位为 1；collate 输出形状/数值正确；`_unpack_batch` ACTION_MISS 分支返回正确五元组
- `run_experiment` 端到端冒烟：dynamic_mixed + MISSING_PROB=0.3 小配置 2 epoch 跑通（loss 正常、无 NaN）
- 回归：既有全部 tests/test_*.py 通过

### 明确不做（YAGNI）

MISS_SECOND；mask mode（loss 屏蔽污损位）；stage-3 终测；rule_fit / analyze_attention 对 dynamic+missing 的支持；全项目 DYNAMIC→ACTION 重命名（后续单独进行）。

## 测试环境

`C:/Users/Chen/anaconda3/envs/torch/python.exe`（torch 2.5.1，无 pytest，用 `__main__` runner）。
