# 非线性规则 randlen predict 污损实验组 — 设计

> 日期：2026-08-14。状态：已获用户批准（含 prob=0 对照组增补）。
> 目标：在现有 randmisslen predict 污损实验框架下，把递推规则换成两个非线性规则，
> 新增一组 224-run 实验（192 主网格 + 32 个 prob=0 对照）。

## 1. 背景与目标

现有 `addition_p127_tr64_ood128_randmisslen_d256h4_predict.json`（216 runs）在
线性规则 `addition`（X(k)=(X(k-1)+X(k-2)) mod 127）上扫描污损参数。本组实验把规则
换成非线性递推，其余设置与 randmisslen predict 组一致，用于对比非线性规则下
transformer 对污损输入的重建（predict）能力。

## 2. 规则语义

| 规则 | task 名 | 递推式 | order | 说明 |
|------|---------|--------|-------|------|
| 规则 1 | `nonlinear`（现有，rules.py:64-67） | X(k) = (X(k-1)² + X(k-2)) mod 127 | 2 | 零改动复用 |
| 规则 2 | `nonlinear_mul`（新增） | X(k) = (X(k-2) · X(k-1)²) mod 127 | 2 | 本次新增 |

- 两个规则 init_len 均为 2，p = 127。
- 规则 2 的状态映射**非双射**（x2=0 时任意 x1 都汇入 (0,0) 不动点），存在暂态轨迹。
  数据集生成依赖已修复的暂态处理逻辑（REFACTOR_LOG #47 截断、#50 逐步续算延拓、
  #51 按状态精确配额），无需新代码；暂态起点的样本是截断轨迹的合法递推延拓。
- `default_num_mask` 新增 `'nonlinear_mul': 1`（与 nonlinear 一致；组内同时显式
  `NUM_MASK: 1`）。

## 3. 实验网格（224 runs，单个组文件）

**主网格（192 runs）**：task ∈ {nonlinear, nonlinear_mul} × N_LAYER ∈ {1,2}
× MISS_LEN ∈ {1,2} × MISSING_PROB ∈ {0.1, 0.3, 0.5} × 8 seeds
= 2×2×2×3×8 = **192 runs**。

**prob=0 对照（32 runs）**：task ∈ {nonlinear, nonlinear_mul} × N_LAYER ∈ {1,2}
× MISSING_PROB = 0 × 8 seeds = 2×2×8 = **32 runs**。

对照组**不带 MISS_LEN 轴**：prob=0 时污损整体关闭（datasets.py:145-153 的
`missing_prob > 0` 门控——`_corrupt` 不调用、零 RNG 消耗、样本为纯窗口，
experiment.py:333-338 走 collate_fn 纯序列路径），MISS_LEN/PREDICT_MISSING
均惰性，保留该轴只会产生逐字节重复的运行。对照 run 的 config 仍写
`MISS_LEN: 1, PREDICT_MISSING: true`（惰性占位，保持配置形状一致），
run 名段为 `randmiss0len1`。

Seed 列表沿用 randmisslen 组：17996, 18318, 34789, 44536, 62513, 64154, 72814, 82585。

## 4. 静态配置（与 randmisslen predict 组逐键一致）

```
P: 127, TRAIN_LEN: 64, OOD_LEN: 128, MAX_UNIQUE_RATIO: 0.7, NUM_MASK: 1,
D_MODEL: 256, N_HEAD: 4, MLP_RATIO: 4, BATCH_SIZE: 128, LR: 0.0003,
WEIGHT_DECAY: 1, EPOCHS: 10000, EVAL_INTERVAL: 20,
EARLY_STOP_ACCURACY: 0.99, EARLY_STOP_NO_IMPROVE: 1000, PREDICT_MISSING: true
```

**有意偏差**：randmisslen 组含 `A:1, B:1`，但 A/B 系数对 nonlinear 类规则被
`single_rule_from_task` 忽略（rules.py:52-53 只对线性任务读取）。本组省略 A/B，
避免误导。run 名相应不含 `a1b1` 段。

污损机制（随机 run 长度 [1,miss_len]、重抽至最长 run == miss_len）是 `_corrupt`
的无条件行为（datasets.py:207-233），自动适用，无配置键。

## 5. 文件与命名

- 组文件：`experiments/nonlinear_p127_tr64_ood128_randmisslen_d256h4_predict.json`
  - new format：`{"concurrency": 8, "experiments": [...]}`（与 randmisslen 组一致）
  - basename 决定输出目录（/mnt/workspace/hujiachen/recursion_results/<name>/{logs,plots} 等，batch_run.py:23-24）
- run 名：`{task}_d256l{L}r4h4_P127_tr64ood128_randmiss{prob}len{ml}_seed{seed}`
  例：`nonlinear_mul_d256l2r4h4_P127_tr64ood128_randmiss0.3len2_seed44536`，
  对照例：`nonlinear_d256l1r4h4_P127_tr64ood128_randmiss0len1_seed17996`
- JSON 由一次性脚本生成并校验（224 条、网格轴笛卡尔积完整、名称唯一、每条含
  name/task/config 三字段）；生成脚本留在 scratch 不入库，遵循"网格 JSON 直接
  提交、无生成器"的仓库惯例。

## 6. 代码改动清单（约 20 行 + 测试）

- `src/rules.py`
  - `single_rule_from_task`：新增 `nonlinear_mul` 分支
    （init_len=2，next_fn = `(seq[-2] * seq[-1]**2) % p`，name 说明递推式）
  - `save_config_extra`：新增分支，保存 `{'recurrence': 'nonlinear_mul'}`
  - `task_from_save_config`：新增反推分支（rule_fit/verify_sample/analyze_attention
    reload checkpoint 依赖）
- `src/experiment.py`
  - task 路由 tuple（:418 附近）加 `'nonlinear_mul'`
  - `default_num_mask` dict（:298 附近）加 `'nonlinear_mul': 1`
- `src/config.json`：加 `"nonlinear_mul": {}` 占位段（与 `"nonlinear": {}` 一致）
- 数据集/模型/训练/协议零改动。predict 模式（PREDICT_MISSING → collate_fn_predict）
  原样复用。

## 7. 测试与验证

- `tests/test_rules.py` 新增 nonlinear_mul 覆盖：
  - next_fn 手算值断言（如 (x1,x2)=(3,4), p=127：3·16 mod 127 = 48；(5,0) → 0）
  - save_config_extra / task_from_save_config roundtrip
  - 与 nonlinear 的区分（同输入不同输出）
- `tests/test_training_smoke.py` 新增 nonlinear_mul tiny run（p=7, 2 epochs），
  覆盖 task 路由 → 数据集生成 → 训练 → stage-3 最终测试全链路
- 全量验证：所有 tests/test_*.py PASS（torch 环境 python）
- 回归保障：黄金 stdout diff（addition/mixed_ab 2-epoch）必须为空——
  改动只新增分支，不触碰现有 task 路径；RNG 调用序约束天然满足
- JSON 校验脚本输出（条数/轴/唯一性）附在实施报告

## 8. 明确不做（YAGNI）

- 不参数化 nonlinear 任务（NONLINEAR_FORM），不引入通用多项式规则类
- 不改 rule head / mixed 管线（nonlinear 单规则走 FibonacciTransformer，无 rule loss）
- 不改 `rule_fit.py` 的线性高斯消元（它对 nonlinear checkpoint 本就不适用；
  task_from_save_config 能正确识别类型即可）
- 不更新 `run_queued.sh`（其引用已过时，与本组无关）
