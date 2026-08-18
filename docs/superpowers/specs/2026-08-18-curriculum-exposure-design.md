# 曝光课程实验设计（curriculum_exposure_p251）

> 日期：2026-08-18
> 状态：已与用户逐节确认（基础配置、阶段终点、基线组成、组织方式）

## 1. 背景与动机

参照实验（`recursion_results/mixed_ab_exposure_p251_supplement`，P=251, d512, h4, r8, seed0）：

| 配置 | 层数 | best acc | 泛化（Unexposed ID/OOD） |
|---|---|---|---|
| 0.1+0.1 | l2 | 0.63% | 0.4% / 0.4%（纯背诵） |
| 0.12+0.12 | l1 | 0.87% | 0.8% / 0.9%（纯背诵） |
| 0.1+0.2 | l1 | 9.74% | 6.4%（部分） |
| 0.1+0.3 | l1 | 99.76% | 99.6%（完全） |

要点：对称低曝光混合失败（仅记住曝光状态、无泛化）；B 曝光提到 0.3 则 A(0.1) 也满血泛化。
注意参照数据本身混了层数（失败为 l2，成功为 l1），本实验 l1/l2 各做一套以消除混淆。

研究问题：样本学习的**先后**对结果的影响——先学好的规则 B 能否/如何帮助规则 A 在
低曝光混合中被学会；A 的学习是复用 B 的表征几何，还是必须重组 embedding。

## 2. 假设

- **H1（C1）**：单规则 0.1 曝光是否可学。若可学 → 0.1+0.1 混合失败源于"混合干扰"；
  若不可学 → 0.1 本身低于可学阈值。
- **H2（T1 vs C2/C3）**：先建立 B（0.3 曝光的解）是否改变混合学习的成功率与动力学
  （收敛速度、per-rule 准确率）。
- **H3（T2 vs T1）**：A 的学习是否依赖重组 embedding。冻结 WTE 后 A 学不会 → 需要重组；
  仍学会 → 复用 B 的几何结构。

## 3. 实验矩阵

36 runs 全部新跑，seeds = 0 / 12345 / 999。镜像 supplement 网格配置覆盖：
`P=251, D_MODEL=512, N_HEAD=4, MLP_RATIO=8, NUM_MASK=2, USE_AB_TAG=false,
USE_CONDITIONAL_WTE=false, AB_PAIRS=[[1,1],[1,2]]`，其余吃 `src/config.json` 默认。

### 3.1 baselines（普通 batch，`experiments/curriculum_exposure_p251_baselines.json`）

| 组 | task | 关键覆盖 | runs |
|---|---|---|---|
| C1 | addition | `A=1,B=1, MAX_UNIQUE_RATIO=0.1` | 3 seeds × 2 层 = 6 |
| C2 | mixed_ab | `MIXED_AB_MAX_UNIQUE_RATIOS=[0.1,0.3]` | 6 |
| C3 | mixed_ab | `MIXED_AB_MAX_UNIQUE_RATIOS=[0.1,0.1]` | 6 |

### 3.2 curriculum（chain ×2，`experiments/curriculum_exposure_p251_l1.json`、`_l2.json`）

- stage1 `phase1`：单规则 B，`A=1,B=2, MAX_UNIQUE_RATIO=0.3`，`EARLY_STOP_ACCURACY=0.99`，
  chain gate `min_best_accuracy=0.99`（3 seeds）
- stage2 `T1`：mixed_ab `[0.1,0.3]`，`INIT_FROM: "@phase1"`，不冻结（3 seeds）
- stage3 `T2`：配置与 T1 完全相同 + `COND_FIX: "WTE", COND_FIX_START: 0`（3 seeds）

NUM_MASK 口径：C1/P1（单规则）也显式设 `NUM_MASK=2`，与混合组及参照网格一致
（单规则默认是 1；统一为 2 保证"是否学会"的评估位置口径跨组可比）。

每层 3+3+3=9 runs，两层 18 runs。总计 18+18=36。

## 4. 代码改动（两处，各 ~10 行）

1. **`training.py` 从头冻结**：`cond_fix_start == 0` 时在 epoch 循环开始前调用
   `freeze_partial(model, cond_fix)` 并置 triggered（现有 `0.0` 语义是第 0 epoch 后才
   冻结，WTE 会白训一轮；新语义保证阶段 2 的任何梯度步之前 WTE 已冻结，
   weight-decay 回漂保护沿用现有 frozen_param_states 机制）。
2. **`chain_run.py` 命名阶段引用**：`INIT_FROM` 的值 `@<stage名>` 解析为指定阶段的
   同 seed 后缀 checkpoint；`@prev` 保留为前一阶段简写。T1/T2 都以 `@phase1` 引用
   （它们互为前后阶段，@prev 会指错）。

无其他代码改动：`INIT_FROM` 部分加载、chain gate 均已存在（tests/test_chain_run.py）。

## 5. 运行方式

```bash
# 无依赖，可并行：
python src/batch_run.py experiments/curriculum_exposure_p251_baselines.json
python src/chain_run.py experiments/curriculum_exposure_p251_l1.json
python src/chain_run.py experiments/curriculum_exposure_p251_l2.json
```

chain 阶段间严格串行；P1 某 seed 达不到 0.99 时该 seed 的链中止（设计行为：
阶段 2 前提不成立则不跑，结果中如实记录）。

## 6. 关键语义与解读注意

- **冻结范围**：`models.py:180` 将 `lm_head.weight` 绑定到 `transformer.wte.weight`
  （同一 Parameter），故 COND_FIX='WTE' 实际冻结 **WTE + 输出投影**；
  结果解读与写作必须按此口径，不能只写 "embedding"。
- **T1 与 T2 共享 phase-1 checkpoint**（同 seed 同一次 B@0.3 训练），两变体唯一
  差异是阶段 2 是否冻结——paired 对比成立。
- 阶段 2 的 optimizer/scheduler 全新开始（INIT_FROM 仅搬权重）。

## 7. 验证与指标

- 新增测试：从头冻结断言（epoch 0 前已冻结、AdamW step 后 WTE 不回漂）、
  `@<stage>` 引用解析（唯一匹配/多匹配报错/首阶段引用报错）；
  全套测试 + 训练冒烟保持绿。
- 核心读数（与参照日志同口径）：各组 stage-3 **per-rule exposed/unexposed 准确率**
  （泛化判据）、best test acc、收敛 epoch 数；T1/T2 与 C2 的逐 epoch 曲线对比。

## 8. 风险

- l2 上 P1（B@0.3）可能达不到 0.99 → gate 中止该 seed 的链（设计行为，见 §5）。
- l2 上 C2（0.1+0.3 from scratch）结果未知——这正是要测的；若 l2 全面更难，
  层数本身成为结果变量，如实报告。
- baselines 与 P1 若并行跑，注意 GPU 显存（d512 r8 l2 模型本身的训练显存；
  stage-3 全状态评估 63001 条序列按 EVAL_BATCH_SIZE=1024 分批，张量仅数十 MB，可承受）。
