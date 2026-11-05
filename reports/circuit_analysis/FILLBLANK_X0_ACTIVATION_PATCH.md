# Fill Blank x0：Clean→Corrupted 激活补丁定位因果计算节点

- 日期：2026-11-04
- 模型：a25b56、len3～5 训练、四层 best checkpoint（与 Logit Lens 和 attention 图相同）
- 基底：`by_blank.x0` 的同一组 32 个 held-out clean-correct 窗口
- 补丁范围：x1/x2 污损 × 5 个 residual 边界 × x1/x2 补丁位置

## 结论摘要

**可解释的有效恢复主要集中在 x2 位置的早、中期 residual。**

- x2 污损时，clean `after L1 / before L2 @ x2` 恢复 **4010/4032 = 99.4544%**，`after L2 / before L3 @ x2` 恢复 **3928/4032 = 97.4206%**。
- x1 污损时，同样 patch x2，分别恢复 **3510/4032 = 87.0536%** 和 **3056/4032 = 75.7937%**。
- 到 `after L3 / before L4 @ x2`，两种污损下仍分别有 **384/4032 = 9.5238%** 和 **355/4032 = 8.8046%** 恢复；均值虽低，但分布于 29/32、28/32 个 base，属于稀疏且 input-dependent 的部分恢复。
- x1 污损时直接 patch x1 只有 2 次和 4 次 top-1 恢复，但早层 target margin 普遍改善。它更像弱置信度效应，不是稳健精确恢复节点。

因此，在本协议下，**x2 是原始 x0 信息最强的局部充分性载体之一；直接 x1 状态的恢复能力弱得多。** 这不证明 x2 是必要的或唯一的计算节点，也不排除信息在更深层转移到其他位置、分散到多个节点，或需要联合补丁。

## 1. Clean、corrupted 与补丁网格

Clean 输入及答案读取方式为：

```text
[BLANK, x1, x2, x3, x4, EOS]
target = 原始合法窗口的 x0
read   = EOS 位置的 full-vocab logits
```

Corrupted 分别只修改一个来源 token：

```text
x1 <- x1 + d (mod 127)
或
x2 <- x2 + d (mod 127),  d = 1..126
```

x3/x4 保持原值，不根据递推重新计算。目标始终取 untouched `windows[:,0]`，不会从 corrupted 输入读取。局部逆解为：

$$
x0=b^{-1}(x2-a x1)\pmod {127},
$$

对应扰动斜率 x1→88、x2→93。已有 raw 中，x1/x2 的 8064 个 corrupted 输出全部改变且都是数值输出，因此没有“原本就正确”的样本抬高恢复率。

五个 residual 边界为：

| 标记 | cache site | 精确位置 |
|---|---|---|
| E | `emb` | L1 前 |
| R0 | `resid0` | L1 后 / L2 前 |
| R1 | `resid1` | L2 后 / L3 前 |
| R2 | `resid2` | L3 后 / L4 前 |
| R3 | `resid3` | L4 后 / `ln_f` 前 |

每个 corrupted 输入都将 clean run 的同层、同位置、同 base 行 residual 向量注入，补丁位置只扫描 x1、x2。总计 20 个补丁条件、80,640 条 patched 输出。此次是 x1/x2 的局部扫描，不覆盖 BLANK/EOS/x3/x4，也没有做多位置联合补丁。

## 2. 指标与基线校验

主指标为：

```text
Restore_full = patched EOS full-vocab argmax == 原始 x0
```

同时保存 numeric-only restore、corrupted answer persistence、其他输出、target probability/logit/rank/margin 及其相对 corrupted 的变化。`Restore_numeric` 仅是辅助上界，不能替代 full-vocab 恢复率。

基线与实现对拍全部通过：

- clean EOS 预测 **32/32** 等于原 x0；
- corrupted **8064/8064** 的 full-vocab、numeric、baseline output 和 D 与已有扰动 raw 完全一致；
- clean cache 只计算一次，按 base 行切片并沿 d batch 重复，不跨 base/d 串用；
- 全部 80,640 个 patched 输出均为数值类，因此每行可互斥分解为 Restore_full、corrupt persistence 或 other。

## 3. 必须排除解释的结构性对照

| 对照 | 覆盖 | 实测结果 | 正确解释 |
|---|---:|---|---|
| `emb @ 实际污损位置` | 8,064 | 8,064 恢复，clean logits 完全一致 | 恢复 clean 输入的平凡 ceiling，不是电路发现 |
| `emb @ 另一位置` | 8,064 | 0 恢复，logits 与 corrupted 完全一致 | 未改动 token 的 identity |
| x2 污损 → patch x1 | 20,160 | 0 恢复，所有 site 均为 no-op | x1 在 causal mask 下看不到未来的 x2，不能据此说 x1 不重要 |
| R3 @ x1/x2 → EOS | 16,128 | 0 恢复、EOS delta 全为 0 | 最后 block 后只剩逐位置 `ln_f+lm_head`，非 EOS patch 无路径影响 EOS |

最后一行 R3 的零恢复是**架构决定的结构性 no-op**，不能解释为“L4 后信息消失”或“最后一层不重要”。剩余可用于 EOS 因果定位的是 R0–R2 上的 9 个 cell。

## 4. 九个可解释 cell 的结果

每格 N=4032。`Δmargin` 是 patched 减 corrupted 的 full-vocab target-margin 均值，未归一化，不是恢复概率。`persistence / other` 表示未恢复输出的去向。

| 污损 | residual / patch | Restore n (%) | persistence / other | mean Δmargin |
|---|---|---:|---:|---:|
| x1 | R0 @ x1 | 2 (0.0496%) | 96.43% / 3.52% | +0.646175 |
| x1 | R1 @ x1 | 4 (0.0992%) | 96.78% / 3.13% | +0.548301 |
| x1 | R2 @ x1 | 0 (0%) | 99.85% / 0.15% | -0.000256 |
| x1 | R0 @ x2 | 3510 (87.0536%) | 0.35% / 12.60% | +11.944909 |
| x1 | R1 @ x2 | 3056 (75.7937%) | 0.82% / 23.39% | +10.929143 |
| x1 | R2 @ x2 | 384 (9.5238%) | 54.09% / 36.38% | +4.978326 |
| x2 | R0 @ x2 | 4010 (99.4544%) | 0.05% / 0.50% | +14.279435 |
| x2 | R1 @ x2 | 3928 (97.4206%) | 0.07% / 2.50% | +13.583003 |
| x2 | R2 @ x2 | 355 (8.8046%) | 60.86% / 30.33% | +5.039453 |

R0/R1 的 x2 patch 在两种污损下都最强，且 x2 污损直接恢复 clean x2 表示比 x1 污损经 x2 间接恢复更强。到 R2，两种条件下的恢复率都降至约 9%，说明单点 x2 patch 仅对少数输入仍充分。

直接 x1 patch 的结果需要单独表述：R0@x1 和 R1@x1 的 mean Δmargin 分别为 +0.646 和 +0.548，且每个 base 的平均 Δmargin 都为正，但 top-1 恢复只分别出现在 base18/21 和 base11/18/21。它们是弱 margin 改善与偶发阈值跨越，不是跨输入稳定恢复。R2@x1 没有恢复，只有 6 条 other，属于真实阴性结果而非结构性 no-op。

## 5. 逐 base 稳定性与 input 依赖

下表覆盖六个清晰 @x2 节点。每 base 分母为 126；full bases 指 126/126 全部恢复，zero bases 指 0/126。

| 节点 | 最低 base 恢复率 | per-base 中位数 | 最高 base 恢复率 | full bases | zero bases |
|---|---:|---:|---:|---:|---:|
| x1 / R0@x2 | 32.54%（base4） | 94.05% | 100% | 7 | 0 |
| x1 / R1@x2 | 4.76%（base4） | 82.14% | 100% | 3 | 0 |
| x1 / R2@x2 | 0% | 2.78% | 62.70%（base7） | 0 | 3 |
| x2 / R0@x2 | 82.54%（base4） | 100% | 100% | 31 | 0 |
| x2 / R1@x2 | 45.24%（base4） | 100% | 100% | 20 | 0 |
| x2 / R2@x2 | 0% | 7.14% | 27.78%（base18） | 0 | 4 |

- x2 R0@x2 最接近普遍规律：31/32 个 base 全部恢复，唯一例外 base4 为 104/126。
- x1 R0/R1@x2 虽所有 base 都有恢复，但最低仅 32.54% 和 4.76%，不能只报告总体均值。
- 两个 R2@x2 都是部分恢复且存在 zero-base，但分别仍有 29/32、28/32 个非零 base，不能当作纯数值噪声。
- **base4 在全部六个节点上都是最低恢复 base**，是系统性离群 input；当前数据没有进一步确定其原因。

已有 R 路由斜率之外的小样本也显示早期恢复，并非只挑选 route-exact 的结果：x1 R0/R1@x2 在非 route-exact 样本中分别恢复 50/53、43/53；x2 R0/R1@x2 分别恢复 33/34、32/34。非 route-exact 分母很小，深层比例不作强推断。

## 6. 解释边界

1. 这是**局部充分性**结果：某节点的 clean activation 能恢复答案，不等于该节点是必要的、唯一的或完整电路。
2. 未恢复也不等于没有信息，可能表示信息已移动、分散，或单点 patch 不足以修复下游全部 corrupted 状态。
3. 32 个 base 的 126 个 d 是聚簇样本，不是 4032 个独立随机试验；结论条件于该 seed42 checkpoint 和 clean-correct 基底，不直接推广到其他 seed 或模型。
4. 单 token 扰动后的输入通常不再满足原递推，是反事实输入；补丁回答的是“clean 信息能否在该反事实状态下恢复原答案”。
5. R3 行、embedding ceiling 和 x2-corrupt/x1 列必须从电路解释中排除，不能作为学习机制的零效应证据。

## 7. 可复现文件与审计

- 脚本：`scripts/probe_fillblank_x0_activation_patch.py`
- 测试：`tests/test_fillblank_x0_activation_patch.py`
- 全部 clean/corrupted/raw、总体及 per-base 聚合：`reports/data/fillblank_x0_activation_patch_a25b56.json.gz`
- 独立审计：`reports/data/fillblank_x0_activation_patch_a25b56_audit.json`
- 运行日志：`reports/data/fillblank_x0_activation_patch_a25b56.log`
- 热图：`reports/circuit_analysis/plots/fillblank_x0_activation_patch_a25b56_{x1,x2}_{restore_full,delta_margin}.png`

验证完成：新测试 10/10、通用 activation patch 测试 10/10、Logit Lens 回归 14/14 通过。独立重算全部 80,640 个唯一补丁键、20 个总体聚合和 640 个 per-base 聚合，指标差为 0；四张图均已读图核验。

```bash
cd /home/cxm/recursion
/home/cxm/miniconda3/bin/python tests/test_fillblank_x0_activation_patch.py
CUDA_VISIBLE_DEVICES='' /home/cxm/miniconda3/bin/python \
  scripts/probe_fillblank_x0_activation_patch.py \
  --device cpu --threads 16 --batch-size 126
```
