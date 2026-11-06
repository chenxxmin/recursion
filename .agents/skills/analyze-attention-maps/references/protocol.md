# Attention Map 执行协议

## 1. 推荐流程

1. 固定任务、best checkpoint、输入布局和样本集合。
2. 复用已验证基底；说明各组是否配对，禁止把不同 blank 的相同 base_row 默认当作同一 input。
3. 明确分析 batch 均值还是逐 input；均值只沿 batch 维计算，保留层、头、query、key。
4. 用原始 forward 和 `AttentionCapture` 提取每层 `(B,H,T,T)`，并用 `try/finally` 关闭 hook。
5. 用 `_forward_per_layer` 独立重放并逐元素对拍，核对 EOS/base outputs 和 causal 约束。
6. 按层×头绘制完整 query-key 热图，固定0~1色阶，保存矩阵和校验。
7. 重新读取 JSON/PNG，独立检查形状、均值、标签和报告数字，再解释结果。

## 2. Attention 口径

每个 attention block 的计算顺序为：

```text
LN -> QKV -> reshape heads -> RoPE(Q,K)
-> QK^T / sqrt(head_size) -> causal mask -> softmax
```

RoPE 直接改变 QK 权重，不能在提取时省略。causal mask 使 `key_position > query_position` 的概率为 0，使每个完整 query 行的概率和为 1。

`AttentionCapture` 是 hook 中按模块输入重算权重，不是直接读取 forward 的局部变量。独立 `_forward_per_layer` 会继续 residual 和 MLP，因此可以验证深层的实际输入及权重。二者使用不同 mask 填充写法时，以 softmax 后数值 allclose 为准，并记录容差和最大差。

## 3. 聚合与代表 input

batch 聚合：

```python
mean_att = att.mean(dim=0)  # (B,H,T,T) -> (H,T,T)
```

不要写成 `att.mean(dim=(0,1))`，后者会把头也平均掉。跨层聚合同样需要显式需求，默认禁止。

当出现以下情况时，补充逐 input 或代表 input 图：

- 已有扰动结果表明不同 input 使用不同电路；
- 平均后权重分散，疑似混合多个路由；
- 需要比较同一 input 在不同 blank、扰动或干预下的变化。

代表 input 的选择规则和数量必须固定并写入报告；不能只挑符合预期的个例。

## 4. 校验清单

### Hook 与独立 forward

- 每层 shape 都是 `(B,H,T,T)`；
- 逐元素 `allclose(rtol=1e-5, atol=1e-6)`；
- 保存每层和全局 `max_abs_diff`、`n_elements_compared`。

### 原模型与基底

- 原始 forward 的 EOS 预测与 base outputs 一致；
- base windows、state indices、N 和 token labels 与源数据一致；
- 不把数值类 argmax 与 full-vocab argmax 混为同一口径。

### Causal 与序列化

- 所有值有限且位于 `[0,1]`；
- `attention.sum(key)=1` 在容差内；
- 上三角未来 key 为 0；
- query 0 只能 attend 自身，所以权重为 1；
- 对原始权重和 JSON 中的均值矩阵分别检查。

## 5. 绘图与报告

len5 标准布局是每 blank 一张 4×4 面板图，共 5 张、16 面板/张、36 单元/面板。标题至少包含任务、blank、N；坐标标注 `Query token` 和 `Key token`；色条说明是否为 batch mean。

输出路径：

```text
reports/circuit_analysis/plots/fillblank_len5_attention_a25b56_blank_xK.png
reports/data/fillblank_len5_attention_a25b56.json
reports/data/fillblank_len5_attention_a25b56.log
reports/circuit_analysis/FILLBLANK_LEN5_ATTENTION.md
```

报告写清：checkpoint、基底、平均口径、矩阵方向、色阶、校验范围与解释限制。图移动目录时，必须同步脚本默认前缀、测试、JSON plots、日志和报告链接。

## 6. 解释边界

- attention 描述信息路由权重，不等同于 value 内容或计算结果。
- 平均权重高不证明因果必要性，平均权重低也不证明该路径没有被使用。
- BLANK query 看不到右侧是 causal mask 的结构结果，不是“模型忽略右操作数”的证据。
- 需要证明电路时，结合扰动、activation patching、value norm 或 Logit Lens，不能仅凭热图下结论。
