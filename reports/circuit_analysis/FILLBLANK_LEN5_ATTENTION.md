# Fill Blank：len5 分挖空位置的逐层逐头注意力图

- 日期：2026-11-04
- 模型：与 Logit Lens 相同的 a25b56、len3～5 训练、四层四头 best checkpoint
- 基底：每个挖空位置复用同一组 32 个 held-out 正确窗口，不重新抽样

## 图文件

| 挖空位置 | PNG |
|---|---|
| `x0` | `reports/circuit_analysis/plots/fillblank_len5_attention_a25b56_blank_x0.png` |
| `x1` | `reports/circuit_analysis/plots/fillblank_len5_attention_a25b56_blank_x1.png` |
| `x2` | `reports/circuit_analysis/plots/fillblank_len5_attention_a25b56_blank_x2.png` |
| `x3` | `reports/circuit_analysis/plots/fillblank_len5_attention_a25b56_blank_x3.png` |
| `x4` | `reports/circuit_analysis/plots/fillblank_len5_attention_a25b56_blank_x4.png` |

每张图均为 **4 行层（L1～L4）×4 列头（H1～H4）**。每个面板保留完整的 **6×6** 注意力矩阵，颜色统一为 viridis 渐变，固定 `vmin=0, vmax=1`。

## 口径

输入为五个窗口 token 加一个 EOS：

```text
[x0, ..., BLANK at xk, ..., x4, EOS]
```

矩阵方向为：

```text
attention[layer][head][query_position][key_position]
```

即纵轴是当前 query，横轴是它读取的历史 key。图中数值是 softmax 后的 attention probability，并仅在 32 个基底的 batch 维取算术平均；**没有对头或层求平均，也没有过滤 self-attention 或小权重**。BLANK 和 EOS 都作为普通输入槽保留。

五组基底分别抽选，不是跨挖空位置的配对样本，因此这些图用于分位置查看，不应把相同 base row 当作同一 input 做逐位置差分。

## 校验

- hook 提取与 `src/analyze_attention.py::_forward_per_layer` 独立重放完成 20/20 层对拍，共比较 92,160 个 attention 元素。
- 全局最大绝对差：`2.384185791015625e-07`，满足 `rtol=1e-5, atol=1e-6`。
- 160/160 个 EOS 预测与已存 base outputs 完全一致。
- 原始权重及序列化均值均通过：query 行和为 1、未来 key 上三角为 0、首 query self-attention 为 1。
- 五张 PNG 均已读图核验层头布局、query/key 方向、BLANK/EOS 标签和统一色阶。

因果 mask 意味着 BLANK query 在同层只能看到自身和左侧 token，不能直接看到右侧操作数。右侧信息可以经后续位置及后续层传播，因此不能把 BLANK 行的右侧零权重解释为模型没有使用右侧信息。逐图展示的是权重，不是 value norm，也不单独证明某条计算电路。

## 可复现文件

- 脚本：`scripts/probe_fillblank_attention_maps.py`
- 测试：`tests/test_fillblank_attention_maps.py`
- 全部均值矩阵和校验：`reports/data/fillblank_len5_attention_a25b56.json`
- 运行日志：`reports/data/fillblank_len5_attention_a25b56.log`

```bash
cd /home/cxm/recursion
/home/cxm/miniconda3/bin/python tests/test_fillblank_attention_maps.py
/home/cxm/miniconda3/bin/python scripts/probe_fillblank_attention_maps.py \
  --device cpu --threads 16 --batch-size 32
```
