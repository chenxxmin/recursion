---
name: analyze-circuit-perturbations
description: 在 recursion 模运算递推仓库中对 fill_blank、corrupt_fix/value_only 做因果扰动和电路分析。用于：按挖空/错误位置做模 P 扰动，检验 D=k·d 与 b/a/逆元候选的关系，比较 len5 与 len8–64 的局部/远位置电路，区分 nonclean 与 clean-hit，分析未解释样本的输出/D/等效斜率分布，以及从已有 JSON 提取结果而无需重跑模型。触发词：电路分析、扰动分析、随机扰动、系数拟合、空格位置、错误位置、±2 之外、混合电路、其他样本分布。
---

# 电路扰动分析

## 1. 先判断提取还是重跑

先搜索 `reports/data/*.json`，确认是否已有相同模型、任务、长度、位置和扰动量的原始记录。

- **只提取**：已有记录包含 `d`、原输出、扰动后输出、`D` 和基底索引；做未解释样本分布、候选替换、计数复核时，不加载模型。
- **需要重跑**：更换 checkpoint/seed/任务/长度/位置集合，增加新扰动量，或缺少所需 raw 记录。
- **定向复核**：稀有异常先扩大样本或枚举全部非零 `d`，不要因一次命中就命名新电路。

提取分布时优先运行：

```bash
python .agents/skills/analyze-circuit-perturbations/scripts/extract_unmatched_distribution.py \
  --data reports/data/fillblank_len5_all_blank_linear_a25b56.json \
  --blank x1 --position x2 --slopes 61,88
```

多个排除斜率对应“其他”的定义。`x2` 挖空的三个例子见 `references/formulas-and-schemas.md`。

## 2. 固定实验语义

使用 canonical 二阶规则：

```text
x[t] = a*x[t-1] + b*x[t-2] (mod P)
```

保持以下口径一致：

- 填空：目标位置改为 `BLANK=P`，输入末尾接 `EOS=P+1`。
- 纠错 value_only：目标位置改为非原值的随机数值，没有 BLANK；仍只读 EOS 的正确值。
- 读取 EOS 槽数值 argmax，同时保留完整词表 argmax；非数值输出计入分母且不算拟合。
- 对位置 `j` 施加 `x[j] <- x[j]+d (mod P)`，定义 `D=(y1-y0) mod P`。
- 用 `D == k*d (mod P)` 的**精确命中率**拟合，不使用普通实数回归或 R²。
- 从基线答对的窗口中选基底；基线准确率与条件于答对后的拟合率分开报告。
- 聚合使用 `exact_count/N` 计数加权，不对各单元百分比做简单平均。

a25b56/P127 的候选为 `b=56, a=25, -a'b=13, a'=61, -b'a=88, b'=93, 0`。其他系数或模数必须重新计算逆元，不要复用这些数值。

## 3. 选择 checkpoint

默认使用 `_best.pth` 做电路分析；`resume` 用于精确续训，`final` 仅是训练终点权重且未必最佳。

已有 a25b56 路径特例：

- len5 填空：固定 len5 没有 ckpt，使用 `len3to5_wd2_ckpt` 的 best。
- len5 纠错：使用 `len5_initPos024` 的 best。
- len8 填空：使用 `len8_wd2_initLen35`，不要写成不存在的 `initLen7`。
- len16–64：使用标准的 `len(L-1) -> lenL` best。

先实测目标长度/位置的基线表现；不要仅凭训练日志中的全局 best 推断每个位置都超过 99%。

## 4. 按任务选择现成脚本

| 需求 | 入口 |
|---|---|
| len5 填空五位置，全部非零 d | `scripts/probe_fillblank_all_blank_linear.py` |
| len5 纠错五位置，含错误位自身和 clean-hit | `scripts/probe_corruptfix_len5_all_positions_linear.py` |
| len8–64 步长 8，±1/±2 随机扰动 | `scripts/probe_fillblank_len8to64_stride8_random.py` |
| len8–64 全部 `|r|>=3` 位置 | `scripts/probe_fillblank_len8to64_far_random.py` |
| 远位置稀有异常的全部 d 复核 | `scripts/confirm_fillblank_far_outliers.py` |
| 已有 len5 数据的“其他”分布 | 本技能 `scripts/extract_unmatched_distribution.py` |

具体参数、输出文件和计数见 `references/runbook.md`。长任务默认 `--device cpu --threads 16`，先检查 GPU 上是否有本课程或其他用户进程，不抢占、不停止他人任务。

## 5. 执行并审计

1. 打印实际模型、输入布局、候选值、基底数量、扰动集合、总样本数和拟合定义。
2. 用一条手动样例验证输入、EOS 输出和至少一个正/负扰动。
3. 保存基底、raw、汇总和日志；大扫描只保存异常 raw 时，必须保留所有单元的七候选精确计数。
4. 独立重算 `D=(y1-y0)%P`、每个候选的 exact_count、聚合计数和报告表格。
5. 区分三种结论：稳定单电路、多个局部公式混合、不匹配任何固定公式的非线性/门控响应。
6. 同时报告输出值、`D` 和 `D*d^{-1} mod P` 分布；检查唯一值数量、top1/top3/top20 覆盖率及基底集中度。
7. 将报告写入 `reports/circuit_analysis/`，原始数据写入 `reports/data/`，脚本写入 `scripts/`，测试写入 `tests/`。
8. 公式使用 `$$...$$` 或行内代码，不使用反斜杠方括号定界符。只有用户明确要求时才提交或推送。

## 6. 必须保留的边界条件

- **clean-hit**：扰动错误位恰好恢复正确值时，输入已无错误。分别统计 nonclean 与 clean-hit，不能混为 `k=0`。
- **双不一致**：纠错时扰动非错误位会保留原错误并制造第二处异常；拟合下降不能反推单错误输入也没有局部电路。
- **混合比例**：同一组局部公式的斜率并存不代表占比恒定；逐长度比较主导系数和完整拟合向量。
- **“其他”**：先明确排除了哪些局部响应（包括必要的 0），再分析分布；分散结果不能命名为第三个固定系数。
- **稀有异常**：区分输出变化与候选拟合。总体近似零影响时可以忽略对主结论的影响，但报告精确计数，不能写成绝对全零。
- **抽样外推**：随机扰动只支持所测 seed/基底；重要异常用相同基底扩大样本或全部 `d` 复核。

候选公式和 JSON 结构见 `references/formulas-and-schemas.md`；已有实验规模、命令和报告索引进 `references/runbook.md`。
