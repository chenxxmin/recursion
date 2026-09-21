# mixed_ab 暴露率补充实验组（P=251）— 设计

> 日期：2026-08-14。状态：已获用户批准。
> 目标：补充两个暴露率边界问题——(1) 两层网络能否学会一层学不了的 0.1+0.1；
> (2) 在 l1 下定位 0.1+x 与 x+x 的可学习边界。

## 1. 背景

现有暴露率网格 `experiments/mixed_ab_exposure_p251.json` / `mixed_ab_exposure_p509.json`
（各 72 runs，d512/l1/r8/h4，seeds {0, 999, 12345}）覆盖了 {0.1,0.2,0.3}×{0.7,0.8,0.9}
双向交叉 + 同值对 {0.1,0.2,0.3,0.7,0.8,0.9}²。用户从该网格观察到（P=251）：

- **0.1+0.1 一层网络学不了**；0.2+0.2 可以学；0.1+0.7 可以学。
- 由此产生两个开放问题：
  1. 0.1+0.1 换**两层**网络能否学会？（容量假设）
  2. l1 下的边界位置：**0.1+x**（x 从 0.1 增大到何时可学）与 **x+x**（对称暴露
     在 (0.1, 0.2) 区间内的临界值）。x+x 已有数据点：0.1 不可学、0.2 可学；
     0.1+x 在 (0.1, 0.7) 区间内无数据（0.7 及以上已有）。

机制背景：`MIXED_AB_MAX_UNIQUE_RATIOS = [r1, r2]` 表示规则 i 的 p² 初始状态空间中
恰好 r_i 比例暴露给训练（experiment.py:78-87 → datasets.py:142 按状态精确配额），
每条规则独立划分 train/test。

## 2. 网格（30 runs，P=251）

| 子组 | 目的 | MIXED_AB_MAX_UNIQUE_RATIOS | N_LAYER | seeds | runs |
|------|------|---------------------------|---------|-------|------|
| A | 两层能否学 0.1+0.1 | [0.1, 0.1] | 2 | 3 | 3 |
| B | 0.1+x 边界（单向） | [0.1, x]，x ∈ {0.2, 0.3, 0.4, 0.5, 0.6} | 1 | 3 | 15 |
| C | x+x 边界 | [x, x]，x ∈ {0.12, 0.14, 0.16, 0.18} | 1 | 3 | 12 |

决策记录（用户确认）：
- **只 P=251**（"学不了/可以学"的观察来自该组；P=509 不补）。
- 0.1+x **单向**（[0.1, x]，不看 [x, 0.1] 的规则不对称性）。
- x+x **四点细扫** {0.12, 0.14, 0.16, 0.18}。
- 边界扫描在 **l1**（现象观察所在架构）；l2 只用于 0.1+0.1。
- seeds 沿用原组 {0, 999, 12345}。

## 3. 配置（与原暴露组逐键一致）

每条 run 的 config 只含以下键（与原 p251 组相同的键集合）：

```
RANDOM_SEED, P: 251, D_MODEL: 512, N_HEAD: 4, N_LAYER: <1|2>, MLP_RATIO: 8,
AB_PAIRS: [[1,1],[1,2]], MIXED_AB_MAX_UNIQUE_RATIOS: <子组比值>,
USE_AB_TAG: false, USE_CONDITIONAL_WTE: false, NUM_MASK: 2
```

其余继承 `src/config.json` main 默认：TRAIN_LEN=16, OOD_LEN=32, BATCH_SIZE=512,
EPOCHS=6000, LR=0.0003, WEIGHT_DECAY=1, DROPOUT=0.0, EVAL_INTERVAL=20,
EARLY_STOP_ACCURACY=0.99, EARLY_STOP_NO_IMPROVE=3000, MAX_UNIQUE_RATIO=0.7
（被 MIXED_AB_MAX_UNIQUE_RATIOS 覆盖）, USE_LEARNABLE_PE=false,
ENTROPY_PENALTY_WEIGHT=0.0, FIRST_TASK_WEIGHT=1.0。

## 4. 文件与命名

- 组文件：`experiments/mixed_ab_exposure_p251_supplement.json`，
  new format `{"concurrency": 8, "experiments": [...]}`（与原组一致）。
  独立补充文件（方案 A）：不修改已完成组的历史文件；日志落在独立目录
  `/mnt/workspace/hujiachen/recursion_results/mixed_ab_exposure_p251_supplement/`。
- run 名沿用原组模式：`mixed_basic_d512l{L}r8h4_P251_N2_e{r1}_{r2}_seed{seed}`
  例：`mixed_basic_d512l2r8h4_P251_N2_e0.1_0.1_seed0`（子组 A）、
  `mixed_basic_d512l1r8h4_P251_N2_e0.1_0.4_seed999`（子组 B）、
  `mixed_basic_d512l1r8h4_P251_N2_e0.14_0.14_seed12345`（子组 C）。
- 与原组无重名（原组 24 个 l1 组合均不在本组 10 个组合内；子组 A 的 l2 与原组 l1 区分）。

## 5. 生成与验证

- 一次性脚本生成（scratch 不入库，遵循"网格 JSON 直接提交"惯例）+ 校验：
  30 条；10 个 ratios/layer 组合（A 1 + B 5 + C 4）× 3 seeds 完整覆盖；名称唯一；每条恰好
  name/task/config 三字段；config 键集合与原组条目完全一致（除 N_LAYER 取值）；
  走真实 `batch_run.build_merged_config` 抽查 3 条（A/B/C 各一）。
- 零代码改动：不需要测试套件回归。mixed_ab 管线（数据集/模型/训练/stage-3）
  由现有测试覆盖；本任务产物是纯数据文件。
- 交付物 = 校验过的 JSON；不包含启动 batch_run（用户在 GPU 机器执行
  `python src/batch_run.py experiments/mixed_ab_exposure_p251_supplement.json`）。

## 6. 明确不做（YAGNI）

- 不改任何代码（rules/datasets/models/experiment 全部不动）。
- 不修改 `mixed_ab_exposure_p251.json` / `p509.json` 原文件。
- 不扫 [x, 0.1] 方向、不扫 P=509、不加更多 seeds（后续可按本组结果再加）。
- 不处理 0.5 同值对的架构缺口（mixed_ab_supplement.json 的 d4096 组是另一件事）。
