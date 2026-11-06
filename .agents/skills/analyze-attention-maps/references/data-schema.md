# Attention Map JSON 结构

当前 len5 文件：`reports/data/fillblank_len5_attention_a25b56.json`。

## 1. config

保存：

```text
ckpt, base_data, model_config,
checkpoint_final_epoch, checkpoint_best_accuracy,
device, threads, batch_size,
seq_len, input_len, eos_position,
n_bases_per_blank, n_layers, n_heads,
aggregation, attention_layout,
replay_rtol, replay_atol, causal_atol,
plot_vmin, plot_vmax, plot_cmap
```

关键固定值：

```text
input_len=6, eos_position=5
n_bases_per_blank=32, n_layers=4, n_heads=4
attention_layout=[layer][head][query][key]
plot_vmin=0, plot_vmax=1, plot_cmap=viridis
```

## 2. by_blank

`by_blank.x0` 至 `by_blank.x4` 各自包含：

```text
k, N, token_labels,
base_state_indices, base_clean_windows, base_outputs,
attention_layout, attention_mean, validation
```

- `N=32`；
- `token_labels` 长度为 6，实际 blank 槽为 `BLANK(xK)`，末尾为 `EOS`；
- `attention_mean` shape 为 `[4][4][6][6]`，只在 batch 维平均；
- `base_clean_windows/base_outputs` 用于核对样本和原 forward 预测；不同 blank 的相同行号不是配对样本。

读取一个单元时：

```text
attention_mean[layer_index][head_index][query_position][key_position]
```

索引均为 0-based；图中的层/头标题为 1-based L1..L4、H1..H4。

## 3. by_blank.validation

### hook_vs_independent_forward

```text
rtol, atol, allclose, max_abs_diff,
n_elements_compared, batches
```

`batches` 保存每批、每层独立重放的详细结果和 causal 检查。它只描述 attention 权重的对拍，不是最终 logits 对拍。

### eos_predictions

```text
match_base_outputs, n_compared, mismatches
```

来自原始 model forward 的 EOS 全词表 argmax 与已存 base outputs 的比较。

### mean_attention_causal

```text
shape, atol,
row_sum_max_abs_error, row_sum_passed,
upper_triangle_max_abs, upper_triangle_passed,
query0_self_max_abs_error, query0_self_passed,
all_passed
```

同样的 causal 检查也应用于每个原始 hook batch。

## 4. 顶层 validation / plots

顶层 `validation` 汇总：

```text
all_checks_passed,
hook_vs_independent_forward,
eos_predictions_match_base_outputs,
n_eos_predictions_compared,
causal_checks_passed, causal_atol,
n_blank_groups
```

`plots` 是五张 PNG 的绝对路径。当前正式结果：

- 5 个 blank 组；
- 20/20 层对拍；
- 92,160 个 attention 元素；
- `max_abs_diff=2.384185791015625e-07`；
- 160/160 个 EOS 预测匹配；
- 五组 `attention_mean` 均为 4×4×6×6。

## 5. 审计注意事项

- JSON 保存的是 batch 均值，不是每个 base 的 raw attention；需要逐 input 审计时应扩展保存 raw 或单独输出代表 input 文件。
- 独立复核形状、有限值、0~1范围、row-sum、未来 key 和 query0 self；不能只检查 `all_checks_passed=true`。
- 对拍数据本身由正式运行生成；若要重新验证 hook 与独立 forward，需要重新加载 checkpoint，不能仅凭均值 JSON 反推全部 raw。
- 移动 PNG 后同步更新 `plots`、脚本默认前缀、测试和报告路径，避免仓库链接失效。
