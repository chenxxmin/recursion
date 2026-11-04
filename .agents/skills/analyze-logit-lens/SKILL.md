---
name: analyze-logit-lens
description: 在 recursion 模运算递推模型上执行 Logit Lens、逐层最终答案/中间量解码和表示位置分析。用于：对每个 Transformer block 的隐藏态应用最终 ln_f+lm_head，观察指定输入槽/EOS 上目标 token 的概率、rank、margin，确定最早清晰解码层和位置，检查 input-dependent 的逐基底信号并排除输入复制巧合，生成热图、JSON 和电路分析报告。触发词：Logit Lens、logit lens、逐层解码、中间结果、中间量、最早解码、清晰解码、第几层哪个位置、EOS 表示、target probability、rank、margin。
---

# Logit Lens 分析

## 1. 固定分析对象与基底

先明确任务、checkpoint、目标 token、输入位置集合和“清晰解码”标准。

- 优先使用对应课程的 `_best.pth`，不要用 resume/final 代替最佳电路。
- 与已有扰动分析比较时，读取同一组基底窗口和 base outputs，不重新抽样； len5 使用 `reports/data/fillblank_len5_all_blank_linear_a25b56.json` 中各空格的 32 个基底。
- 保持与任务相同的 BLANK/EOS 布局和目标位置。len5 填空为 `x[k-2]..x[k+2]` 加 EOS，EOS 索引为 5。
- 多目标分析必须区分空格索引 `k` 与目标定义；缩放项、模减项的真值从原始干净窗口计算，不能从含 BLANK 的输入读取。
- 如需更换模型、长度、位置或样本，先列实际路径、基底数量、层数、目标数和总 cell 数。

## 2. 逐层计算目标指标

不要只调用最终 forward。手动执行：

```text
hidden = wte(input) -> dropout
for block in transformer.h:
    hidden = block(hidden)
    logits = lm_head(ln_f(hidden))
```

多个目标共享同一套逐 block hidden/logits，在目标和位置循环中分别计算指标，不为每个目标重复模型 forward。每个 block 后、每个指定位置记录：

- full-vocab softmax 下的目标概率 `prob`；
- `logit`；
- `rank = 1 + count(other_logit > target_logit)`；
- `margin = target_logit - max_other_logit`。

默认使用完整词表，不在中途改成数值类归一化；最终 EOS 与训练指标对照时再明确两者是否一致。

## 3. 校验手工 forward

必须同时运行原始 `model(input)` 并验证：

- 最后一层 EOS logits 与手工逐 block 结果 `allclose`；
- 最大绝对差有明确记录，理想情况为 0；
- EOS 预测与已存 base outputs 一致；
- 任一校验失败时先停止并排查 dropout/eval、位置偏移、ln_f 或 lm_head，不继续解释结果。

## 4. 聚合与最早清晰解码

每个 `空格 × 目标 × 层 × 位置` 聚合全部基底，至少保存：

```text
N, mean_prob, median_prob, top1_rate, mean_margin, mean_rank
```

默认沿用 len5 双标准，也可按任务调整，但报告必须写清：

| 标准 | top1率 | 平均目标概率 |
|---|---:|---:|
| 主标准 | >=90% | >=0.50 |
| 严格标准 | >=95% | >=0.90 |

按层优先、输入位置顺序次之，EOS 最后，选择最早 clear cell；同时保留全部 clear cells 和其中 mean_prob 最高者。

区分：

- **开始出现低置信 top1**：某层 top1率已升高，但 mean_prob/margin 仍低；
- **稳定高置信解码**：top1率和平均概率达到标准，并结合 margin 检查置信度。

不要只凭 top1率宣称模型已学会目标值。

当电路可能随 input 改变时，不能只报告跨基底均值：

- raw 必须保留 `base_row` 和 `target_value`，逐基底保存最早/最佳可读 cell；
- len5 单基底辅助判据为 primary `rank==1 && prob>=0.50`、strict `rank==1 && prob>=0.90`，并与聚合判据明确区分；
- 检查目标值是否等于当前位置的原有 token，或出现在任何可见输入槽；直接复制巧合不能当作新计算的中间量；
- 可按既有扰动支持度选择多个代表 input 分层展示，但扰动标签不是 clean 电路真值，且必须同时保留全基底计数。

## 5. 输出与审计

1. 保存每样本 raw、每 cell 聚合、清晰标准、最早位置和 forward 校验。
2. 绘制每层×位置的 mean probability 热图；多目标/代表 input 图按 target 和 base 明确分面板，英文标题/坐标，生成后读图检查。
3. 报告最早层/位置表、EOS 逐层概率表和非 EOS 最大概率/top1率。
4. 独立重算 mean/median/top1_rate/mean_margin/mean_rank 与报告表格。
5. 将报告写入 `reports/circuit_analysis/`，JSON/日志写入 `reports/data/`，热图写入 `plots/`。
6. 不把“非 EOS 槽不能直接解码”解释为“该位置没有参与计算”；Logit Lens 与扰动/注意力分析回答的问题不同。
7. 公式使用 `$$...$$` 或行内代码，不使用反斜杠方括号定界符。只有用户明确要求时才提交或推送。

## 6. 现有实现

直接复用：

最终答案模式（默认，保持旧行为）：

```bash
/home/cxm/miniconda3/bin/python scripts/probe_fillblank_logit_lens.py \
  --device cpu --threads 16 --batch-size 32 \
  --out reports/data/fillblank_len5_logit_lens_a25b56.json \
  --plot plots/fillblank_len5_logit_lens_a25b56.png
```

len5 F/M/R 中间量模式：

```bash
/home/cxm/miniconda3/bin/python scripts/probe_fillblank_logit_lens.py \
  --target-mode intermediates --device cpu --threads 16 --batch-size 32
```

中间量模式覆盖 24 个 `(blank,target)`、492 个聚合 cell 和 15,744 条 raw，并输出逐基底 copy-control 与 F/M/R 代表 input 热图。

配套测试：

```bash
/home/cxm/miniconda3/bin/python tests/test_fillblank_logit_lens.py
```

协议细节见 `references/protocol.md`；JSON 字段和现有 len5 数量见 `references/data-schema.md`。结果示例：`reports/circuit_analysis/FILLBLANK_LEN5_LOGIT_LENS.md` 与 `reports/circuit_analysis/FILLBLANK_LEN5_INTERMEDIATE_LOGIT_LENS.md`。
