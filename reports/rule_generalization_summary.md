# Rule Generalization Probe — 结果汇总

**日期**：2026-08-24
**实验目的**：mixed_basic 训练规则为 `AB_PAIRS=[[1,1],[1,2]]`（a=1，b∈{1,2}）。
验证模型学到的是"从 x3 反推 b 的通用算法"（应泛化到任意 b），
还是"把 x3 匹配到背下来的两个模式"（只会在 b=1/2 之间选）。

**代码**：`src/rule_generalization.py`（复用 `analyze_attention.load_model`
和 `activation_patch.rule_targets`；规则约定 (c1,c2) => c1*X(k-1)+c2*X(k-2)）。

**Checkpoint**（3 seeds，best test acc ≈ 99.95%）：
`/data/cxm/models/mixed_ab_downscale_l1h4/mixed_basic_d256l1r8h4_P127_N2_e0.7_seed{0,999,12345}.pth`

**设置**：b ∈ {0,3,4,5,6,7,8,12,16,24,32,48,64,96,126}（+ b=1,2 对照），
每 b 256 个样本（x1,x2 ∈ 1..p-1，排除 0 退化），prefix 3 / 5 两种条件，
teacher forcing + 自由生成 16 步。报告：
`reports/rule_generalization_d256l1r8h4_P127_seed{0,999,12345}.json`。

## 主要发现

1. **对照成立**：b=1、b=2 时 match_true = 100%（3 seeds 全部），流程可信。

2. **所有新 b 的 match_true ≈ 0**（0~1.2%，与 argmax 噪声一致）。
   模型从未按真实规则续写哪怕一步。→ **否定"模型学会从 x3 算 b"假说。**

3. **模型把新样本吸入两条训练规则之一**：b1+b2 命中率合计 85~95%，
   neither 仅 5~15%（b=64 时最高 ~20%）。不是 OOD 崩溃，是自信地选错规则。

4. **长 prefix（5 token）不能挽救**（tf5 true 仍 ≈0）：
   模型不仅不能**识别**任意 b，也不能**执行**任意 b 的递推——
   它只会背下来的两个完整"程序"。

5. **自由生成显示规则锁定**：生成 16 步后 60~90% 的样本自洽地锁死在
   b1 或 b2（自己生成的历史不断强化所选规则）。

6. **吸引盆结构**（3 seeds 一致）：
   - b=0 → 多数吸入 b1（56~65%）
   - b≥3 → 多数吸入 b2（54~91%，b=3 最强 ~89%）
   - 这与"按 implied b 的普通整数距离取最近训练规则"完全吻合：
     b=0 离 1 最近；b≥3 时 |b−2|<|b−1| 恒成立（含 b=126≡−1 mod 127，
     模型仍选 b2，说明距离不是模 p 意义下的）。

## 结论

模型没有学到"a=1，从 x3 提取 b 再执行"的参数化算法，而是学到：
**两条固定的递推程序 + 一个把观察到的偏差模式归类到最近训练规则的分类器。**
结合 activation patching 的 null result（规则信号不集中、直接来自输入 token），
完整图景是：模型从输入 token 直接读出消歧线索，立即映射到两条背下来的
规则之一并执行——不存在可泛化的"规则变量"。

## 后续可选

- per-position 曲线（report JSON 里有逐位置数据）看吸入是否随位置变化；
- 在 P509 的 d256l1r8h4 上重复，检验结论的 p 依赖性；
- 用 logit/方向探针找残差流里的"b1 vs b2 分类方向"。
