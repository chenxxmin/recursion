# Activation Patching Probe — 结果汇总

> **2026-08-24 修正**：本文档第 4、5 节的两处结论已被后续实验修正
> （见文中【修正】标注），完整新结果见 `multi_pos_patch_summary.md`。
> 核心更正：① MLP 并非无贡献（l1 中为负载组件）；② 规则证据通道
> 在新口径下可定位（l1: lag2 头@query；l2: layer1 证据头）。

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
- **替换第 0 层 MLP 输出**（`mlp[3]` 之后）：**对预测完全无影响**（match_a / match_b 与 baseline 相同）。
- **替换第 0 层 transformer block 的输出**：不翻转。

【修正 2026-08-24】当时的推断“这些模型的 MLP 对规则区分几乎没有贡献”**过强**：
该实验 patch 的是 l2 模型的**第 0 层**，无影响只说明第 0 层 MLP 冗余、可被第 1 层补救。
后续在 l1 模型上的逐位置实验（见 `multi_pos_patch_summary.md`）表明 MLP 隐藏层
是递推计算的负载组件——patch 后预测完全崩坏（neither≈1.0）。
另外“替换 block 输出不翻转”指的是第 0 层输出；末层 block 输出（ln_f 输入）
在数学上必然转移整个状态（l2 重测 match_src≈0.998，单测已固化此不变量）。

## 5. 结论与下一步建议

- **结论（本文档口径下）**：单组件单位置 patch 无法显著翻转规则。但【修正 2026-08-24】
  这一 null result 主要是**方法论的**：① patch 位置 q 只影响预测位置 k=q+1，
  对全部预测位置取平均把真实翻转（~0.4）稀释到噪声水平（~0.01）；② 未区分
  “规则翻转”与“状态转移”。在新口径下规则证据通道**可以定位**（l1: lag2 头@
  query 位置，l2: layer 1 的对应证据头），详见 `multi_pos_patch_summary.md`。
- **仍然成立的部分**：规则信号不存在于单一 head/MLP 就能完全决定的“开关”
  （单点 patch 翻转率上限 ~0.65）；attention pattern 与规则无关；规则信息
  来自输入 token 序列本身。
