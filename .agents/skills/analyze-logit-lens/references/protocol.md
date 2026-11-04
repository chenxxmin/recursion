# Logit Lens 执行协议

## 1. 推荐流程

1. 固定任务布局、checkpoint 和目标；目标可为最终 `x_k`，也可为缩放项、模减项等中间量。
2. 复用已验证基底；没有基底时再生成 held-out 窗口并检查基线答案。
3. 分离空格索引 `k` 与目标定义，从原始干净窗口计算所有目标真值。
4. 枚举空格/目标与输入槽，普通输入位置升序，EOS 最后。
5. 对每个 block 后的隐藏态执行最终 `ln_f + lm_head`；多个目标共享同一套 hidden/logits。
6. 记录 full-vocab target probability/logit/rank/margin，raw 保留 `base_row/target_id/target_value`。
7. 运行原始 forward，核对最后一层 EOS logits 与 base outputs。
8. 按 `(blank,target,layer,position)` 聚合，应用清晰标准，生成最早位置、热图和报告。
9. 检查逐基底稀有命中、复制混淆和多个扰动支持代表 input，避免平均值掩盖 input-dependent 电路。
10. 独立重算并区分低置信 top1、高置信解码、真实新计算与输入复制。

## 2. 指标定义

对目标 token `v` 和完整词表 logits `z`：

```text
prob(v) = softmax(z)[v]
rank(v) = 1 + sum_j [z_j > z_v]
margin(v) = z_v - max_{j != v} z_j
```

rank 使用严格大于，因此并列第一时 rank=1。top1_rate 是基底中 rank=1 的比例，不等同于 mean_prob，也不要求 top1 时 margin 的平均值必须为正。多目标计算时不得原地污染后续目标使用的 logits。

## 3. 最早位置选择

对每个 `(空格 k, 目标 target)` 独立执行：

1. 按 `layer_index` 升序检查；
2. 同层按输入位置升序，EOS 最后；
3. 选择第一个满足阈值的 cell；
4. 保存所有 clear cells，并另报 mean_prob 最高的 clear cell；
5. 没有 clear cell 时明确写 None，不能填 EOS/L4 作为默认答案。

报告层号使用 1-based L1..L4；JSON 的 layer_index 保持 0-based。没有通过阈值的 cell 时，如需展示“最大平均概率”，必须明确它不是 earliest，也不表示已达到清晰标准。

## 4. 逐 input 与复制控制

聚合标准不能发现所有稀有电路。逐基底辅助分析使用：

- primary：`rank==1 && prob>=0.50`；
- strict：`rank==1 && prob>=0.90`。

保存每个 `(k,target,base_row)` 的 readable cells、最早和最佳 cell，并汇总 readable base 数。至少检查：

- `direct_copy_at_position`：目标值等于该解码位置的原有 token；
- `target_in_visible_inputs`：目标值出现在任一非 BLANK 数值输入槽。

目标值与输入 token 巧合相等时，简单复制比“新计算出中间量”更有解释力；排除后应重新报告命中数。逐点最优结果必须伴随全部基底计数，不能包装成稳定电路。

可按已有扰动 raw 为 F/M/R 计算逐 input 支持度并选择多个代表，但这些标签来自离干净流形后的响应，不是 clean 电路真值。不同 blank 的 base_row 通常不是配对 input，不能直接跨 blank 比较。

## 5. 解释限制

- 目标在 EOS 可读，说明答案表示集中到 EOS；不证明操作数没有参与早期计算。
- 非 EOS 槽不能解码，不排除该位置含操作数信息或参与注意力路由。
- 早期 top1率升高但 mean_prob 低，只能说明低置信选择，不能替代高置信结论。
- Logit Lens 使用最终 unembedding 做线性读出；没有信号不等于内部没有该计算，表示也可能是非线性或分布式的。
- 干净递推流形上 `uM=x[k+1]-b*x[k-1]=a*x[k]`、`uR=x[k+2]-a*x[k+1]=b*x[k]`，所以这些值本身不能区分“先算差”与“直接形成答案倍数”。
- 聚合指标描述所选基底；改变基底、扰动分布或 checkpoint 后必须重新评估可比性。

## 6. 新任务扩展

改变 len、层数或变体时，优先参数化现有脚本而非复制硬编码：

- `legal_positions` 控制目标邻域和 EOS；
- `targets_for_blank` / `compute_target_values` 控制目标集合和真值；
- `probe_batch` 控制共享逐层 forward；
- `target_metrics` 控制概率/rank/margin；
- `aggregate_records` 控制样本统计；
- `clear_summary` 控制最早清晰位置；
- `build_per_base_readability` 控制逐 input 与 copy-control。

先写 CPU 小测试验证 mask/位置偏移、模运算真值、多目标一致性和新任务 forward，再运行完整分析。
