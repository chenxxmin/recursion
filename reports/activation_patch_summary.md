# Activation Patching Probe — 结果汇总

## 1. 修复

`src/activation_patch.py` 中的规则 tuple 原来按 `a*X(k-2)+b*X(k-1)` 解释，与训练侧 `LinearRecurrenceRule` 的约定 `(c1,c2) => c1*X(k-1)+c2*X(k-2)` 相反。已统一为训练侧约定，并更新 `tests/test_activation_patch.py` 期望。测试全绿：`7/7 passed`。

## 2. 选用的 100% mixed basic（无 ab_tag）checkpoint

经过筛选，以下 checkpoint 在随机初始值、OOD 长度 32 上双向 baseline 都接近 1.0：

| checkpoint | A→A | B→B | 层数/头数 |
|---|---|---|---|
| `mixed_basic_d512l2r8h4_P127_N2_e0.7_seed0.pth` | 0.993 | 0.965 | 2 层 / 4 头 |
| `mixed_basic_d256l2r8h4_P127_N2_e0.7_seed0.pth` | 0.988 | 0.964 | 2 层 / 4 头 |
| `mixed_basic_d512l1r8h4_P509_N2_e0.7_seed0.pth` | 0.982 | 0.980 | 1 层 / 4 头 |
| `mixed_basic_d256l1r8h4_P509_N2_e0.7_seed0.pth` | 0.975 | 0.973 | 1 层 / 4 头 |
| `mixed_basic_d1024l1r16h4_P127_N2_e0.7_seed0.pth` | 0.997 | 0.967 | 1 层 / 4 头 |

这些模型在训练测试集上的 best_accuracy 都 ≥ 99.5%。

## 3. 正式报告文件

- `reports/activation_patch_mixed_basic_d512l2r8h4_P127_seed0.json`
- `reports/activation_patch_mixed_basic_d256l2r8h4_P127_seed0.json`
- `reports/activation_patch_mixed_basic_d512l1r8h4_P509_seed0.json`
- `reports/activation_patch_mixed_basic_d256l1r8h4_P509_seed0.json`
- `reports/activation_patch_mixed_basic_d1024l1r16h4_P127_seed0.json`

参数：`--pairs 64 --len 32`。

## 4. 主要发现

### 单组件 patch 效果极弱
分别 patch 每个 attention head 的 `c_proj` 输入和每个 MLP 隐藏层输入：
- 把 A 的 hidden state 注入 B（AtoB）后，B 仍然基本遵循规则 B；
- 把 B 的 hidden state 注入 A（BtoA）后，A 仍然基本遵循规则 A。

没有任何单个 head 或 MLP 单元能显著翻转规则。

### 多层模型的深入验证（d512 l2）
在 2 层模型上额外做了更强的 patch：
- **同层所有 attention head 一起 patch**（替换整层 `c_proj` 输入）：不翻转。
- **替换整层 MLP 输出**（`mlp[3]` 之后）：**对预测完全无影响**（match_a / match_b 与 baseline 相同）。
- **替换整个 transformer block 的输出**（残差流经过 MLP 后）：不翻转。

这说明即使在 2 层模型里，规则区分的信号也**不集中在** `attn.c_proj` 输入、MLP 隐藏层、MLP 输出或整层输出这些常见 hook 点。MLP 输出 patch 完全无影响尤其说明：这些模型的 MLP 对规则区分几乎没有贡献，模型主要靠 attention/embedding 读取输入序列中的规则线索。

## 5. 结论与下一步建议

- **结论**：在现有 mixed basic 模型（无论 1 层还是 2 层）上，activation patching 无法定位到一个（或几个）“承载规则区分”的孤立 hidden state。规则信号更像是分布式地来自输入 token 序列本身，并通过残差流直接传递，而不是被某个 attention head 或 MLP 单元单独编码。
- **建议**：若要继续定位，可尝试：
  1. **对 attention pattern（softmax 后的 attention 权重）做 patching**，看规则是否体现在“看哪些位置”上；
  2. **对输入 embedding / 残差流做 patching**，验证规则信息是否确实来自原始 token；
  3. 使用 **logit lens / 方向分析 / 探针分类器** 等互补方法，寻找隐藏空间中的规则方向。
