# Fill Blank 电路分析：len5 中间量的 Logit Lens

- 日期：2026-11-04
- 模型：与最终答案 Logit Lens 相同的 a25b56、len3～5 训练、四层 best checkpoint
- 基底：五个空格各自复用同一组 32 个 held-out 正确窗口，共 160 个 blank-input 组合
- 范围：24 个合法 `(blank, intermediate target)`，L1～L4，局部输入槽及 EOS
- 数据量：492 个聚合 cell、15,744 条逐基底 raw 记录

## 结论摘要

**这次没有收集到可信的 F/M/R 中间量线性解码信号。**

1. 24/24 个中间目标都没有达到 32 基底聚合的 primary 或 strict 清晰标准；所有聚合 earliest 均为 `None`。
2. 24/24 个目标在 EOS 上都没有单基底 primary 或 strict 命中，也没有单基底 strict 命中。
3. 未做复制控制时，只有 4 个目标、6 个 base-target、11 个 cell 达到单基底 primary。**这 11/11 个 cell 的目标值都恰好等于该解码位置原有的输入 token**，因此都能用直接复制解释；排除后直接归零。
4. 专门按已有扰动结果选取每条 F/M/R 路由支持度最高的三个 input，仍没有额外命中。行为扰动斜率符合某条公式，并不意味着对应中间项会被最终 `ln_f + lm_head` 直接线性读出。
5. 这与最终答案形成鲜明对照：原始 `x_k` 在五组中均于 L4/EOS 清晰解码，而所检查的缩放项和差项没有同类信号。

这里的否定结论仅限于：**在本模型、这些位置以及最终 unembedding 的 Logit Lens 下，没有检测到可信的目标 token 线性读出。** 它不等于模型没有计算这些量，也不排除它们以非线性、分布式或不可被最终 unembedding 直接读取的形式存在。

## 1. 目标公式与合法组合

递推和三个局部求解方向为：

$$
x_k=a x_{k-1}+b x_{k-2},\qquad
x_k=a'(x_{k+1}-b x_{k-1}),\qquad
x_k=b'(x_{k+2}-a x_{k+1}).
$$

所有计算均在模 127 下进行：

$$
a=25,\quad b=56,\quad a'=61,\quad b'=93,\quad -a'b=13,\quad -b'a=88.
$$

逐空格检查以下所有索引合法的目标：

- F：`b*x[k-2]`、`a*x[k-1]`；
- M：`-a'b*x[k-1]`、`a'*x[k+1]`、`uM=x[k+1]-b*x[k-1]`；
- R：`-b'a*x[k+1]`、`b'*x[k+2]`、`uR=x[k+2]-a*x[k+1]`。

空格 `k=0,1,2,3,4` 分别包含 3、6、8、5、2 个目标。目标真值总是从原始干净窗口计算，而不是从已经插入 BLANK 的模型输入读取。

## 2. 判据与复制控制

沿用原 Logit Lens 的 full-vocab probability、rank 和 margin。聚合清晰标准为：

| 标准 | top1率要求 | 平均目标概率要求 |
|---|---:|---:|
| primary | >=90% | >=0.50 |
| strict | >=95% | >=0.90 |

为了不漏掉 input-dependent 的少数电路，另外对单个基底使用：

- primary：`rank==1 && prob>=0.50`；
- strict：`rank==1 && prob>=0.90`。

单样本判据不等于聚合判据，它用于发现稀有命中，不能直接当作跨输入稳定电路的证据。每个命中再检查两级复制混淆：

- `direct_copy_at_position`：目标值等于当前解码位置原有的数值 token；
- `target_in_visible_inputs`：更保守地检查目标值是否出现在任一非 BLANK 输入槽。

## 3. 全部 24 个目标的结果

下表的“最大平均概率”是在该目标所有层和位置的聚合 cell 中取最大值，不表示它已通过清晰标准。`P bases` 是任一位置达到单样本 primary 的基底数，分母均为 32；`after copy control` 排除同位置直接复制后的数量。strict 在全部目标上均为 0。

| Blank | 中间目标 | 最大平均概率 @ 层/位置 | P bases | after copy control |
|---|---|---:|---:|---:|
| x0 | `-b'a*x1` | 0.009038 @ L3/EOS | 0 | 0 |
| x0 | `b'*x2` | 0.007740 @ L3/x0 | 0 | 0 |
| x0 | `uR=x2-a*x1` | 0.008562 @ L3/EOS | 0 | 0 |
| x1 | `-a'b*x0` | 0.008056 @ L1/EOS | 0 | 0 |
| x1 | `a'*x2` | 0.008521 @ L1/x1 | 0 | 0 |
| x1 | `uM=x2-b*x0` | 0.008426 @ L1/x1 | 0 | 0 |
| x1 | `-b'a*x2` | 0.010712 @ L3/EOS | 0 | 0 |
| x1 | `b'*x3` | 0.011318 @ L3/x0 | 0 | 0 |
| x1 | `uR=x3-a*x2` | 0.007797 @ L1/EOS | 0 | 0 |
| x2 | `b*x0` | 0.031643 @ L1/x3 | 2 | 0 |
| x2 | `a*x1` | 0.008715 @ L3/x2 | 0 | 0 |
| x2 | `-a'b*x1` | 0.017992 @ L1/x4 | 0 | 0 |
| x2 | `a'*x3` | 0.016355 @ L1/x4 | 0 | 0 |
| x2 | `uM=x3-b*x1` | 0.014551 @ L4/x3 | 0 | 0 |
| x2 | `-b'a*x3` | 0.008211 @ L4/x3 | 0 | 0 |
| x2 | `b'*x4` | 0.015265 @ L4/x2 | 0 | 0 |
| x2 | `uR=x4-a*x3` | 0.017305 @ L1/x3 | 0 | 0 |
| x3 | `b*x1` | 0.023609 @ L1/x2 | 1 | 0 |
| x3 | `a*x2` | 0.009272 @ L4/x3 | 0 | 0 |
| x3 | `-a'b*x2` | 0.018553 @ L1/x4 | 0 | 0 |
| x3 | `a'*x4` | 0.008454 @ L1/x3 | 0 | 0 |
| x3 | `uM=x4-b*x2` | 0.021160 @ L1/x1 | 1 | 0 |
| x4 | `b*x2` | 0.051855 @ L4/x3 | 2 | 0 |
| x4 | `a*x3` | 0.008255 @ L2/EOS | 0 | 0 |

全部聚合 cell 的最大 top1 率也只有 6.25%。例如 `x4 / b*x2` 在 L4/x3 的平均概率为 0.051855，但中位概率仅 0.002793，top1 率仅 2/32；其均值被少数复制巧合拉高，不是普遍存在的中等强度表示。

## 4. 六个稀疏命中全部是复制巧合

下表完整列出六组达到单样本 primary 的 base-target。所有 rank 均为 1；概率按达标 cell 列出。

| Blank / target | base row | 干净窗口 | 目标值与复制来源 | 达标 cell：层/位置，概率 |
|---|---:|---|---|---|
| x2 / `b*x0` | 18 | `[66,65,114,13,105]` | `13=x3` | L4/x3，0.651664 |
| x2 / `b*x0` | 23 | `[19,38,109,27,48]` | `48=x4` | L1/x4，0.770257；L2/x4，0.547146 |
| x3 / `b*x1` | 16 | `[46,79,106,89,33]` | `106=x2` | L1/x2，0.592614；L2/x2，0.545192；L3/x2，0.571182 |
| x3 / `uM=x4-b*x2` | 15 | `[69,90,18,29,82]` | `90=x1` | L1/x1，0.507234 |
| x4 / `b*x2` | 9 | `[72,36,106,94,31]` | `94=x3` | L1/x3，0.506654；L3/x3，0.660210；L4/x3，0.732077 |
| x4 / `b*x2` | 18 | `[68,34,86,117,121]` | `117=x3` | L4/x3，0.782923 |

因此，不能把 `F_b` 三类目标上的零星命中（共 5 个 base-target、10 个 cell）解释成“部分 input 上形成了 b 倍中间项”。更简单的解释是模型在这些位置读出了已有 token，而模乘结果恰好与该 token 同值。

采用更保守的控制——只要目标值出现在任何可见数值输入槽，就排除整个 base-target——剩余 raw 的最大目标概率也只有 0.183169，依然没有任何 primary 命中。

## 5. 针对 input-dependent 电路的多输入检查

本次不是只做 32 基底平均：全部 32 个逐 input raw 均已保存，并另外根据已有 80,640 次扰动的逐基底响应，为每条合法路由选取共同支持 `d` 最多的三个代表 input。共同支持是指两个相关操作数在同一个 `d` 下都精确满足该路由的斜率。

| Blank / route | 三个代表 base row（共同支持 d 数 /126） | 代表 input 上的 primary 命中 |
|---|---|---:|
| x0 / R | 0(126)、1(126)、3(126) | 0 |
| x1 / M | 15(54)、23(48)、12(36) | 0 |
| x1 / R | 11(42)、10(38)、24(38) | 0 |
| x2 / F | 20(61)、2(53)、17(49) | 0 |
| x2 / M | 29(19)、24(13)、5(12) | 0 |
| x2 / R | 31(8)、30(5)、14(3) | 0 |
| x3 / F | 13(121)、17(119)、25(117) | 0 |
| x3 / M | 2(1)、27(1)、9(1) | 0 |
| x4 / F | 0(126)、1(126)、2(126) | 0 |

其中 x0/R 和 x4/F 的代表 input 在全部 126 个非零扰动量上都同时支持对应双操作数斜率，但仍无中间量解码。x2/R 与 x3/M 的行为支持本身较稀有；全 32 基底扫描也没有找到被平均掩盖的可信命中。

这些扰动标签只是代表 input 的选择依据，不是干净输入上的电路真值。五个空格的 32 基底也不是跨 blank 的配对样本，不能把相同 `base_row` 当作同一 input 进行跨位置比较。

## 6. 复合差项的解释限制

在干净递推流形上，以下恒等式逐样本成立：

$$
uM=x_{k+1}-b x_{k-1}=a x_k,\qquad
uR=x_{k+2}-a x_{k+1}=b x_k.
$$

所以，即使干净输入上解码到了 `uM/uR`，也不能仅凭这一点区分“先计算差项”与“直接形成最终答案的 a/b 倍表示”。本次这些目标没有可信解码，因此更不支持具体的两阶段计算顺序。若要区分计算顺序，需要结合扰动输入、激活干预或更适合非线性表示的探针。

## 7. EOS 与最终答案的对照

- 中间目标在 EOS 上的最高聚合平均概率仅为 0.010712，出现在 `x1 / -b'a*x2` 的 L3。
- 中间目标在 EOS 上的最高单样本概率仅为 0.056075，远低于 primary 的 0.50。
- 同一批输入、同一手工 forward 下，最终答案 `x_k` 在原实验中全部于 L4/EOS 达到清晰解码，五组平均概率为 0.9536～0.9898、top1 率均为 100%。

因此，当前证据更支持“最终答案在 EOS 的末层形成可直接读出的表示”，而不是“F/M/R 的缩放项和差项也以同样方式显式形成”。不能反向推断这些中间计算不存在。

## 8. 校验与可复现文件

技术校验：

- 多目标共享同一套逐 block hidden/logits；合成模型测试确认其与逐目标独立 forward 完全一致。
- 最后一层 EOS logits 与原始 forward：`allclose=True`、`max_abs_diff=0`；共核对 20,640 个 logits。
- 所有 EOS 预测与已存 base outputs 一致，mismatch 为空。
- 独立从 raw 重算全部 15,744 个目标真值和 492 个聚合 cell，最大指标差为 0。
- 14 项纯 CPU 测试全部通过；五张总体热图和九张代表 input 热图均已读图检查。

文件：

- 探针：`scripts/probe_fillblank_logit_lens.py`
- 测试：`tests/test_fillblank_logit_lens.py`
- 完整 raw、聚合、逐 input、copy-control 和路由支持：`reports/data/fillblank_len5_logit_lens_intermediates_a25b56.json`
- 独立审计：`reports/data/fillblank_len5_logit_lens_intermediates_a25b56_audit.json`
- 运行日志：`reports/data/fillblank_len5_logit_lens_intermediates_a25b56.log`
- 总体热图：`plots/fillblank_len5_logit_lens_intermediates_a25b56_blank_xK.png`
- 代表 input 热图：`plots/fillblank_len5_logit_lens_intermediates_a25b56_route_xK_ROUTE.png`
- 最终答案对照报告：`reports/circuit_analysis/FILLBLANK_LEN5_LOGIT_LENS.md`

复现命令：

```bash
cd /home/cxm/recursion
/home/cxm/miniconda3/bin/python scripts/probe_fillblank_logit_lens.py \
  --target-mode intermediates --device cpu --threads 16 --batch-size 32
/home/cxm/miniconda3/bin/python tests/test_fillblank_logit_lens.py
```
