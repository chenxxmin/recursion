# Fill Blank 电路分析：len5 Logit Lens 的最早解码位置

- 日期：2026-11-03
- 模型：len3～5 训练的 a25b56 best checkpoint，四层 Transformer
- 基底：复用此前每个空格位置的同一组 32 个 held-out 正确窗口
- 范围：空格 `k=0..4`，每层 `L1..L4`，输入位置 `x[k-2]..x[k+2]` 及 EOS
- 中间量补充：`reports/circuit_analysis/FILLBLANK_LEN5_INTERMEDIATE_LOGIT_LENS.md`

## 结论摘要

按两套清晰解码标准，五个空格位置的最早清晰解码位置均为：

> **第 4 层（最后一个 Transformer block）的 EOS 槽。**

在 EOS 上，`x2/x3/x4` 最早于第 2 层已能达到约 50%～59% 的 top1率，`x0/x1` 于第 3 层达到约 72% top1率；但这些早期预测的平均概率仍很低，不能称为稳定高置信解码。第 4 层五个目标的平均概率均超过 0.95，top1率均为 100%。

非 EOS 输入位置没有清晰解码：平均目标概率最高只有 0.037，top1率最高只有 9.38%。这说明目标值的可读表示集中到 EOS，而不是在其他输入槽上直接形成。它不表示前层没有使用这些操作数。

## 1. 方法与清晰判据

对每个空格 `k`，构造：

```text
[x0,...,x(k-1),BLANK,x(k+1),...,x4,EOS]
BLANK=127, EOS=128
```

逐 block 执行 `wte -> dropout -> transformer block`，每层对隐藏态应用最终的 `ln_f + lm_head`，观察原始目标 `x_k` 的：

- full-vocab softmax probability；
- target logit；
- 严格大于口径的 rank；
- `target_logit - max_other_logit` margin。

每个单元聚合 32 个基底，记录 mean/median probability、top1 rate、mean margin 和 mean rank。共 96 个聚合单元、3072 条 raw 记录。

清晰解码采用两套标准：

| 标准 | top1率要求 | 平均目标概率要求 |
|---|---:|---:|
| 主标准 | >=90% | >=0.50 |
| 严格标准 | >=95% | >=0.90 |

层内按输入位置顺序查找，EOS 最后。最后一层手动 EOS logits 与模型原始 forward 逐元素一致：`allclose=True, max_abs_diff=0`，EOS 预测也与已存 base outputs 完全一致。

## 2. 最早清晰解码位置

主标准和严格标准得到完全相同的结果：

| 空格目标 | 最早层 | 最早位置 | 平均概率 | top1率 | 平均 logit margin |
|---|---:|---|---:|---:|---:|
| `x0` | L4 | EOS | 0.9536 | 100% | 5.486 |
| `x1` | L4 | EOS | 0.9858 | 100% | 6.499 |
| `x2` | L4 | EOS | 0.9898 | 100% | 7.138 |
| `x3` | L4 | EOS | 0.9862 | 100% | 6.811 |
| `x4` | L4 | EOS | 0.9729 | 100% | 5.925 |

所有通过主标准的 cell 也只有这些 L4/EOS 单元；严格标准下没有其他新增位置。

## 3. EOS 上目标概率如何逐层形成

每个单元格式为：**平均概率 / 中位概率 / top1率**。

| 目标 | L1 EOS | L2 EOS | L3 EOS | L4 EOS |
|---|---|---|---|---|
| `x0` | 0.008 / 0.008 / 0% | 0.010 / 0.009 / 0% | 0.053 / 0.054 / 72% | 0.954 / 0.976 / 100% |
| `x1` | 0.007 / 0.007 / 0% | 0.009 / 0.008 / 0% | 0.062 / 0.052 / 72% | 0.986 / 0.987 / 100% |
| `x2` | 0.008 / 0.007 / 0% | 0.036 / 0.034 / 50% | 0.121 / 0.117 / 97% | 0.990 / 0.990 / 100% |
| `x3` | 0.007 / 0.007 / 0% | 0.037 / 0.034 / 59% | 0.095 / 0.088 / 97% | 0.986 / 0.990 / 100% |
| `x4` | 0.007 / 0.008 / 0% | 0.038 / 0.035 / 50% | 0.083 / 0.080 / 81% | 0.973 / 0.983 / 100% |

可以区分两个阶段：

1. **开始能对答案进行低置信选择**：`x2/x3/x4` 在 L2，`x0/x1` 在 L3；
2. **稳定高置信解码**：五个目标都在 L4 才达到，平均概率大于 0.95、top1率 100%，margin 大于 5.4。

因此，仅用 top1率会把“清晰解码”判断得过早；平均概率和 margin 提供了必要的置信度约束。

## 4. 非 EOS 输入位置的结果

非 EOS 位置没有出现主标准或严格标准下的清晰解码：

| 空格目标 | 非 EOS 最大平均目标概率 | 非 EOS 最大 top1率 |
|---|---:|---:|
| `x0` | 0.010 | 3.12% |
| `x1` | 0.012 | 3.12% |
| `x2` | 0.031 | 9.38% |
| `x3` | 0.037 | 6.25% |
| `x4` | 0.007 | 0.00% |

少数非 EOS cell 会在个别基底上把目标排为第一名，但这些孤立事件的 top1率最高也只有 9.38%，平均 margin 仍为负，不能视为已经形成目标表示。

这与扰动分析是互补的：

- 扰动分析表明，EOS 的答案会按照局部递推斜率响应特定操作数；
- Logit Lens 表明，可直接读出的目标概率主要集中到 L4 的 EOS，而不是直接留在操作数所在槽位。

因此，更合理的解释是“操作数信息经网络传递和变换后在 EOS 形成答案”，而不是“操作数位置没有参与计算”。Logit Lens 本身不能证明前层没有使用 `x[k-2]/x[k-1]` 等操作数，也不能给出具体的注意力路由。

## 5. 可复现文件

- 分析脚本：`scripts/probe_fillblank_logit_lens.py`
- CPU 测试：`tests/test_fillblank_logit_lens.py`
- 全部 raw、聚合、最早解码和 forward 校验：`reports/data/fillblank_len5_logit_lens_a25b56.json`
- 运行日志：`reports/data/fillblank_len5_logit_lens_a25b56.log`
- 五位置概率热图：`plots/fillblank_len5_logit_lens_a25b56.png`
- 本报告：`reports/circuit_analysis/FILLBLANK_LEN5_LOGIT_LENS.md`

复现命令：

```bash
cd /home/cxm/recursion
/home/cxm/miniconda3/bin/python scripts/probe_fillblank_logit_lens.py \
  --device cpu --threads 16 --batch-size 32 \
  --out reports/data/fillblank_len5_logit_lens_a25b56.json \
  --plot plots/fillblank_len5_logit_lens_a25b56.png
```

基底直接读取 `reports/data/fillblank_len5_all_blank_linear_a25b56.json`，不重新抽样。最后一层 EOS logits 已与原始 forward 核对，热图也已读图核验。
