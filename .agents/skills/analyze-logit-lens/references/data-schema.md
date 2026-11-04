# Logit Lens JSON 结构

现有 len5 文件：

- 最终答案：`reports/data/fillblank_len5_logit_lens_a25b56.json`
- 中间量：`reports/data/fillblank_len5_logit_lens_intermediates_a25b56.json`

## 1. config / criteria

`config` 保存 ckpt、base data、模型配置、batch/线程、seq_len、EOS 位置、基底/层数、`target_mode`、模系数和执行计划。`criteria` 保存聚合 primary/strict 两套阈值。

`config.plan` 按 blank 保存 `targets`。每个 target 包含：

```text
target_id, route, label, terms
```

`terms` 是 `[coefficient, clean_window_position]` 列表；目标真值为各系数乘对应干净窗口值之和对 P 取模。空格 `k` 与目标位置不再隐含绑定。

## 2. raw

每条记录对应一个基底、一个目标、一个层和一个位置：

```text
k, base_row, target_id, target_value, route,
layer_index, position, prob, logit, rank, margin
```

- `k`：BLANK 所在位置；
- `base_row`：对应该 blank 下 `base_clean_windows` 的 0-based 行号，不能跨 blank 当作配对样本；
- `target_id/target_value`：目标标识及从干净窗口计算的数值 token；
- `route`：`final` 或 F/M/R；
- `layer_index`：0-based block 索引；
- `position`：输入槽索引，EOS=5；
- `prob`：完整词表 softmax 下目标概率；
- `rank`：严格大于口径的目标排名；
- `margin`：目标 logit 减最大其他 logit。

数量：

| 模式 | 目标组合 | aggregate cells | raw |
|---|---:|---:|---:|
| final | 5 | 96 | 3,072 |
| intermediates | 24 | 492 | 15,744 |

中间量模式五个 blank 的目标数为 3/6/8/5/2。所有目标共享同一套模型 forward，raw 增多不意味着按目标重复 forward。

## 3. aggregates

每个 cell 聚合同一 `(k,target_id,layer,position)` 的 32 条 raw：

```text
k, target_id, route, layer_index, position, n,
mean_prob, median_prob, top1_rate, mean_margin, mean_rank
```

读取 EOS 层间趋势时筛选 `position=5`；分析非 EOS 时排除该值。没有 clear cell 时若另取最大 mean_prob，必须注明它不是达到阈值的 earliest。

## 4. earliest

最终答案模式：

```text
earliest.primary/strict.xK = {
  earliest,
  highest_mean_prob,
  clear_cells
}
```

中间量模式多一层 target：

```text
earliest.primary/strict.xK.TARGET_ID = {
  earliest,
  highest_mean_prob,
  clear_cells
}
```

`earliest=null` 表示没有 cell 达到阈值。不要把 `highest_mean_prob` 与层优先的 earliest 混用。

## 5. per_base_readability

```text
per_base_readability = {
  criteria,
  summary,
  bases
}
```

单基底 primary 为 `rank==1 && prob>=0.50`，strict 为 `rank==1 && prob>=0.90`。`bases` 保留每个 `(k,target,base_row)` 的：

```text
target_value, target_in_visible_inputs,
primary/strict.readable_anywhere,
primary/strict.readable_cells,
primary/strict.earliest,
primary/strict.best_cell
```

每个 readable cell 还带 `direct_copy_at_position` 和 `target_in_visible_inputs`。`summary` 按 `(k,target)` 汇总 readable base/cell 数、base rows 和最佳 cells，并提供：

```text
n_readable_bases_excluding_direct_copy_at_position
n_readable_cells_excluding_direct_copy_at_position
n_readable_bases_excluding_target_in_visible_inputs
n_readable_cells_excluding_target_in_visible_inputs
```

这些控制字段按非复制 readable cell 计数；如需“目标值出现在输入就排除整个 base-target”的更保守口径，应另行明确计算，不能与 cell-level 字段混称。

## 6. perturbation_route_support

仅 intermediates 模式包含。按 `by_blank.xK.F/M/R` 保存：

- 两个操作数位置及斜率；
- 每个 base 各自 126 个 d 的精确匹配计数/比例；
- 两个操作数在同一 d 均匹配的 `shared_d`、共同支持计数/比例；
- 按共同支持数、最小操作数匹配率排序的 `top3` 和对应 window。

这些数值来自已有扰动 raw，只用于选择/分层代表 input，不是 clean-circuit ground truth。

## 7. final_forward_validation

包括：

```text
rtol, atol, allclose, max_abs_diff,
n_batches, n_eos_logits_compared,
model_eos_predictions_match_base_outputs,
manual_eos_predictions_match_base_outputs,
base_output_mismatches
```

报告前要求 `allclose=true` 且 mismatch 列表为空。len5 当前 final 和 intermediates 均为 `max_abs_diff=0`；中间量模式核对 5 批、20,640 个 EOS logits。

独立审计文件 `reports/data/fillblank_len5_logit_lens_intermediates_a25b56_audit.json` 另存 raw key、目标真值、全部聚合指标、copy-control 和 forward 字段的复核结果。
