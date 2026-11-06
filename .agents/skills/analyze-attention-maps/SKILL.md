---
name: analyze-attention-maps
description: 在 recursion 模运算递推模型上提取并分析逐层、逐头注意力，生成完整 query-key 热图和可审计数据。用于：复用最佳 checkpoint 与既有基底，按挖空位置或输入分组绘制每层每头 attention，保留 BLANK/EOS，统一0~1渐变，执行 hook 与独立 forward 对拍、causal/row-sum 校验，并区分 batch 平均、单 input、attention 权重与 value norm。触发词：attention、注意力、每层每头、逐层逐头、注意力热图、attention map、query-key、BLANK attention、EOS attention、AttentionCapture。
---

# Attention Map 分析

## 1. 固定模型、基底和聚合口径

先明确任务、checkpoint、输入布局、层数、头数、样本集合和输出目录。

- 优先使用对应课程的 `_best.pth`，不要用 resume/final 代替最佳电路。
- 与已有 Logit Lens 或扰动分析比较时，复用同一组基底窗口和 base outputs，不重新抽样。len5 填空使用 `reports/data/fillblank_len5_all_blank_linear_a25b56.json` 中各 blank 的 32 个基底。
- 保持任务的 BLANK/EOS 布局。len5 fill_blank 输入为 `[x0,...,x4 with BLANK at k,EOS]`，所以矩阵是 6×6，不是 5×5 或 8×8。
- 先选择“batch 平均”或“逐 input/代表 input”口径。平均只能作用于明确指定的样本维，**不得平均层或头**；不同 blank 的 base_row 通常不是配对 input。
- 更换模型、长度或样本前，列出实际路径、N、层数、头数、矩阵尺寸和总图数。

## 2. 提取真实 attention 权重

优先复用现有工具，不修改生产模型接口：

```python
from probe_fillblank_attention import AttentionCapture
from analyze_attention import _forward_per_layer
```

原始 forward 时，`AttentionCapture` 在每个 attention block 的 hook 中按同一输入重算 QKV、RoPE、causal mask 和 softmax，返回每层 `(B,H,T,T)`。保留完整结果：

```text
attention[layer][head][query_position][key_position]
```

默认分析 **softmax 后、attention dropout 前的 probability**；在 eval/dropout=0 下即模型实际使用的权重。不要混成 raw QK score 或 `attention × value_norm`。

注意：`get_attention_weights` 等公开接口可能只返回第一个样本；批量分析使用保留 batch 维的 `_forward_per_layer`，不要误把 `[0]` 当作全 batch 平均。

## 3. 强制校验

每次正式分析都必须：

1. 用 `_forward_per_layer` 独立重放完整 residual/MLP 路径，与 hook attention 逐层逐元素比较；默认 `rtol=1e-5, atol=1e-6`。
2. 记录 `allclose`、`max_abs_diff` 和比较元素数，失败立即停止。
3. 用原始 forward 的 EOS 预测核对已存 base outputs，防止加载错 checkpoint 或基底。
4. 检查每个 query 行和为 1、未来 key 上三角为 0、首 query self-attention 为 1；对原始权重和序列化均值都检查。
5. 用 `try/finally` 关闭 hook，避免后续 forward 被污染。

对拍的是 attention 权重，EOS 对拍的是原 forward 预测；报告中不要混写两种校验范围。

## 4. 绘图要求

按用户指定的分组一图一组；len5 默认每个 blank 一张：

- 行是 L1..L4，列是 H1..H4，每格为完整 6×6 query-key 矩阵；
- 纵轴 query、横轴 key，不能转置；
- 统一 `vmin=0, vmax=1` 和 viridis 渐变，一张图共享 colorbar；
- 保留 self-attention、小权重、BLANK 和 EOS，不做阈值过滤；
- token 标签反映实际 BLANK 位置，标题、坐标轴和色条使用英文；
- 逐张读图检查层头布局、方向、标签、色阶和文件尺寸。

## 5. 输出与解释

- 热图统一写入 `reports/circuit_analysis/plots/`。
- 均值矩阵、token labels、样本信息和全部校验写入 `reports/data/`；报告写入 `reports/circuit_analysis/`。
- 保存复现脚本、纯 CPU 测试和运行日志；生成后重新读取 JSON 和 PNG。
- batch 均值可能掩盖 input-dependent 路由，必要时另画多个代表 input，不能只展示平均图。
- causal mask 下 BLANK query 不能直接看到右侧 token，但右侧信息可经其他位置和后续层传播；右侧零权重不等于模型没有使用右操作数。
- attention 权重不等于 value norm，也不单独证明计算电路；需要与扰动、Logit Lens 或激活干预联合解释。
- 只有用户明确要求时才提交或推送；工作区有其他实验文件时必须选择性暂存。

详细执行协议见 `references/protocol.md`；JSON 字段和 len5 数量见 `references/data-schema.md`。

## 6. 现有实现

直接复用：

```bash
cd /home/cxm/recursion
/home/cxm/miniconda3/bin/python scripts/probe_fillblank_attention_maps.py \
  --device cpu --threads 16 --batch-size 32
```

配套测试：

```bash
/home/cxm/miniconda3/bin/python tests/test_fillblank_attention_maps.py
```

当前输出：`reports/circuit_analysis/plots/fillblank_len5_attention_a25b56_blank_xK.png`（K=0..4）、`reports/data/fillblank_len5_attention_a25b56.json` 和 `reports/circuit_analysis/FILLBLANK_LEN5_ATTENTION.md`。
