# dynamic_mixed → action 改名设计

**日期**：2026-08-19
**目标**：把任务名 `dynamic_mixed` 全面改名为 `action`（代码、配置、文档、实验 JSON），对旧 checkpoint 与旧实验 JSON 保留兼容。不含任何行为变更。

## 1. 背景

`dynamic_mixed` 任务（每步规则可变，布局 `[x1, x2, f3, x3, ..., f_L, x_L]`）决定更名为 `action`。改名需覆盖：

- 代码符号（类名、函数名、枚举名、TASK 分支、save_config 字段值）
- `src/config.json` 的任务配置段名
- `experiments/*.json` 的 `task` 字段
- 活跃文档（`recursion_model_and_training_settings.md` 等）

兼容约束：

- 旧 checkpoint 的 save_config 里 `recurrence: 'dynamic_mixed'` 必须仍可被分析工具识别。
- 旧实验 JSON 里 `task: dynamic_mixed` 必须仍可跑（归一为 `action` + warning）。
- `BatchTag` 枚举只在运行时的 batch 元组里传递，不落盘，改名/改注释安全（数值保持不变，防止任何潜在序列化差异）。

## 2. 改名映射

### 2.1 任务标识（用户可见）

| 旧 | 新 |
|---|---|
| `TASK == 'dynamic_mixed'` | `TASK == 'action'` |
| config.json 段名 `"dynamic_mixed"` | `"action"` |
| save_config `recurrence: 'dynamic_mixed'` | `recurrence: 'action'`（新写入） |
| ctx `post_train_mode: 'dynamic_mixed'` | `'action'`（无消费者匹配该值，stage-3 分支只判 mixed_ab/single_recurrence，安全） |
| 实验 JSON `task: dynamic_mixed` | `task: action` |

### 2.2 代码符号

| 文件 | 旧 | 新 |
|---|---|---|
| datasets.py | `DynamicMixedDataset` | `ActionDataset` |
| datasets.py | `generate_dynamic_sample` | `generate_action_sample` |
| datasets.py | `dynamic_mixed_collate_fn` | `action_collate_fn` |
| datasets.py | `dynamic_missing_collate_fn` | `action_missing_collate_fn` |
| datasets.py | `BatchTag.DYNAMIC_MIXED` | `BatchTag.LOSS_MASK`（见 §4 说明） |
| experiment.py | `_prepare_dynamic_mixed` | `_prepare_action` |
| analyze_attention.py | `is_dynamic_mixed`（dict 键） | `is_action` |
| analyze_attention.py | `dynamic_seq_len`（dict 键） | `action_seq_len` |
| analyze_attention.py | `_make_dynamic_seq` | `_make_action_seq` |
| analyze_attention.py | `_make_dynamic_query_mask` | `_make_action_query_mask` |

注释/docstring 中的 `dynamic_mixed` 字样同步改为 `action`。

### 2.3 不改的部分

- `REFACTOR_LOG.md`、`CORE_SPLIT_PLAN.md`、`docs/superpowers/plans|specs/` 下的历史文档：它们是当时的历史记录，保持原样。
- `BatchTag` 枚举数值。
- `experiments/` 里 JSON 的 `name` 字段（实验结果目录名与其对应，改了会和已跑结果脱节）。

## 3. 兼容点（仅有的三处新逻辑）

1. **batch_run.py**：读实验 JSON 后，若 `task == 'dynamic_mixed'`，归一为 `'action'` 并打印 warning。config.json 段查找随之用 `'action'`（旧 JSON 经归一后自然命中新段名）。
2. **rules.py `task_from_save_config`**：`recurrence in ('action', 'dynamic_mixed')` 都返回 `None`。
3. **analyze_attention.py `_infer_recurrence`**：`recurrence in ('action', 'dynamic_mixed')` 都走 action 分支。

其余读 `recurrence` 的地方（rule_fit.py:449、verify_sample.py:127）对两个名字的行为一致（都走"非单规则"路径），无需改动。

## 4. BatchTag.DYNAMIC_MIXED 的命名说明

该 tag 的 payload 是 per-position `loss_mask`，除 action 任务外还被单规则 mask 模式（`collate_fn_masked`）复用。改成 `ACTION` 会误导（单规则路径与 action 无关），故按 payload 语义命名为 `LOSS_MASK`。`ACTION_MISS` 已是 action 语义，保留不动，仅更新注释。

## 5. 测试更新

- `tests/test_dynamic_missing.py`：TASK 改 `action`，import `_prepare_action`（文件名是否改为 test_action_missing.py 随实施决定，改更贴切）。
- `tests/test_missing_values.py:test_collate_routes_tuples_to_dynamic_mixed`：函数名与引用更新。
- `tests/test_rules.py:103`：断言改为同时覆盖 `'action'` 与旧名 `'dynamic_mixed'`。
- 全部测试跑通即验证完毕；另加一个端到端冒烟：action + MISSING_PROB=0.3 小配置 2 epoch（沿用 dynamic_missing 的既有冒烟套路）。

## 6. 错误处理 / 边界

- 旧实验 JSON 未改名直接跑：batch_run 归一化兜底（§3.1）。
- 旧 checkpoint 做 attention 分析 / rule_fit：§3.2、§3.3 兜底。
- config.json 里若有人残留 `"dynamic_mixed"` 段：不生效且无提示——在实施时把 base config 的段名直接改掉即可，仓库内无其他 base config。

## 7. 实施顺序（概要）

1. src/ 代码符号 + config.json 段名 + 三处兼容逻辑
2. tests/ 更新并跑全量 pytest
3. experiments/*.json 的 task 字段（dynamic_mixed.json / dynamic_d512.json / dynamic_missing.json 等）
4. 活跃文档（recursion_model_and_training_settings.md、RULE_FIT_NOTES.md 如需）
5. action + MISSING_PROB 冒烟
