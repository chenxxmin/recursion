# 配置参数去留决策表

> 用途：为训练场景统一（single/mixed/action × order × miss_len）重构确定每个配置键的去留。
> 填法：在"**决定**"栏填写 —— `保留`（继续作为可配置参数）/ `写死=值`（代码内固定，不再可配）/ `删除`（连代码逻辑一起移除）。
> 填好告诉助手读这个文件。
> "建议"栏是助手的初步意见，仅供参考，以你的决定为准。

## 1. 任务与规则

| 参数 | 当前默认 | 语义 | 建议 | 决定 |
|---|---|---|---|---|
| `TASK` | `addition` | 场景名（新体系 single/mixed/action） | 保留 | 保留 |
| `P` | 53 | 模数 | 保留 | 保留 |
| `A/B/C/D` | 1 | 单任务系数散键 | 被 RULES 取代，删（兼容层仍读旧键） | 删 |
| `AB_PAIRS` / `ABC_PAIRS` | [[1,1],[1,2]] 等 | 规则系数列表 | 被 RULES 取代，删（兼容层仍读旧键） | 删 |
| `NUM_MASK` | null（按场景+order 自动推导） | 前 N 位不计损失 | 保留 | 保留 |

## 2. 数据与样本

| 参数 | 当前默认 | 语义 | 建议 | 决定 |
|---|---|---|---|---|
| `TRAIN_LEN` | 16 | 训练窗口长 | 保留 | 保留 |
| `OOD_LEN` | 32 | 测试窗口长 | 保留 | 保留，默认改成16 |
| `MAX_UNIQUE_RATIO` | 0.7 | 单任务暴露率 | 保留 | 不保留，并入下个参数 |
| `MIXED_AB_MAX_UNIQUE_RATIOS` | [0.7,...] | mixed 逐规则暴露率 | 保留（可改名） | 保留，改名 |
| `NUM_TRAIN_SAMPLES` | null / action 10000 | 显式样本数覆盖 | 保留 | 保留 |
| `NUM_TEST_SAMPLES` | 256 / action 2000 | 测试集大小 | 保留 | 保留 |
| `STATE_SPACE_CAP` | null | 状态空间抽样上限（高阶必需） | 保留 | 保留 |
| `DATA_MODE` | null | 单任务数据模式：`full_split`（全枚举切分）/ `sampled_fresh_test`（抽样+新鲜测试，v2 制度用）；null 时按 FRESH_TEST_PER_EVAL/STATE_SPACE_CAP 兼容推导 | 保留（活的） | 保留 |
| `FRESH_TEST_PER_EVAL` | false | 每 eval 新鲜测试集 | 保留 | 保留 |
| `RESHUFFLE_EACH_EPOCH` | false | 每 epoch 重洗牌 | 保留 or 写死 | 保留 |

## 3. 污损

| 参数 | 当前默认 | 语义 | 建议 | 决定 |
|---|---|---|---|---|
| `MISSING_PROB` | 0 | 污损概率 | 保留 | 保留 |
| `MISS_LEN` | 1 | 污损段最大长度 | 保留 | 保留 |
| `PREDICT_MISSING` | false | 污损位是否计入损失（single/mixed；action 恒 true） | 写死？ | 写死true |
| `MISS_SECOND` | false | 第 2 项起必定缺失 MISS_LEN 项（action 不支持） | 删除 | 删 |

## 4. 模型结构

| 参数 | 当前默认 | 语义 | 建议 | 决定 |
|---|---|---|---|---|
| `D_MODEL` | 64 | 隐藏维度 | 保留 | 保留 |
| `N_LAYER` | 1 | 层数 | 保留 | 保留 |
| `N_HEAD` | 1 | 注意力头数 | 保留 | 保留 |
| `MLP_RATIO` | 4 | MLP 宽度倍数 | 保留 | 保留 |
| `DROPOUT` | 0.0 | Dropout | 写死 0？ | 保留 |
| `USE_LEARNABLE_PE` | false | 可学习 PE（false=RoPE） | 写死 false？ | 写死，用RoPE |

## 5. 优化器

| 参数 | 当前默认 | 语义 | 建议 | 决定 |
|---|---|---|---|---|
| `LR` | 3e-4 | AdamW 学习率 | 保留 | 保留 |
| `WEIGHT_DECAY` | 1 | AdamW weight decay（grokking 关键参数） | 保留 | 保留 |
| `BATCH_SIZE` | 512 | batch 大小 | 写死 512？ | 保留 |
| `GRAD_CLIP_NORM` | 1.0 | 梯度裁剪 | 保留 or 写死 | 保留 |
| `GRAD_ACCUM_STEPS` | 1 | 梯度累积（显存救火用） | 保留 or 写死 1 | 保留 |
| `EPOCHS` | 6000 | 最大 epoch | 保留 | 保留 |
| `EPOCH_CAP` | 无（代码可读） | 硬停止 epoch 上限，LR schedule 仍按完整 T_max 退火 | 保留（活的） or 删 | 保留 |

## 6. 训练控制与早停

| 参数 | 当前默认 | 语义 | 建议 | 决定 |
|---|---|---|---|---|
| `EVAL_INTERVAL` | 20 | 评估间隔（画曲线用 1） | 保留 | 保留 |
| `EARLY_STOP_ACCURACY` | 0.99 | 达标早停 | 保留 or 写死 | 保留 |
| `EARLY_STOP_NO_IMPROVE` | 3000 | 无提升早停 | 保留 or 写死 | 保留 |
| `EARLY_STOP_EXTRA_EPOCHS` | 200 | 文档说已废弃但代码还读 | 删除 | 保留，默认0        |
| `MAX_TRAIN_HOURS` | null | 墙钟定时（核心机制） | 保留 | 保留 |
| `SKIP_FINAL_GENERATION_TEST` | false | 跳过最终生成测试 | 保留 or 删 | 删除，不要最终测试 |

## 7. 精度

| 参数 | 当前默认 | 语义 | 建议 | 决定 |
|---|---|---|---|---|
| `ALLOW_TF32` | true | TF32 加速 | 写死 true | 写死 true |
| `USE_AMP` | false | 16-bit autocast（bf16 已被实证判决） | 写死 false？还是留作粗筛开关 | 写死 false |
| `AMP_DTYPE` | bfloat16 | autocast 精度 | 随 USE_AMP 去留 | 删 |

## 8. 初始化与续跑

| 参数 | 当前默认 | 语义 | 建议 | 决定 |
|---|---|---|---|---|
| `RANDOM_SEED` | 42 | 随机种子 | 保留 | 保留 |
| `INIT_FROM` | null | 暖启动 checkpoint | 保留 | 保留 |
| `RESUME_FROM` | null | 续跑 checkpoint | 保留 | 保留 |

## 9. 参数冻结（mixed 课程实验用）

| 参数 | 当前默认 | 语义 | 建议 | 决定 |
|---|---|---|---|---|
| `COND_FIX` | null | 冻结 WTE/LINEAR | 保留 or 删？近期还用吗 | 删 |
| `COND_FIX_START` | null | 达标后冻结阈值 | 同上 | 删 |
| `COND_FIX_START_A1` / `COND_FIX_START_A2` | null | 两阶段冻结变体 | 同上，键很碎 | 删 |

## 10. 损失附加项

| 参数 | 当前默认 | 语义 | 建议 | 决定 |
|---|---|---|---|---|
| `ENTROPY_PENALTY_WEIGHT` | 0.0 | 注意力熵惩罚 | 写死 0？还用在实验里吗 | 写死 0 |
| `FIRST_TASK_WEIGHT` | 1.0 | 首预测位损失加权 | 写死 1？ | 保留 |

## 11. mixed 模型变体

| 参数 | 当前默认 | 语义 | 建议 | 决定 |
|---|---|---|---|---|
| `USE_AB_TAG` | false（mixed 段） | rule token 前缀（basic/label 对照） | 保留 or 写死 | 删，去掉这个变体 |
| `USE_CONDITIONAL_WTE` | false | 条件嵌入模式 | 保留 or 删？近期还用吗 | 删 |
| `COND_WTE_SHARED_RATIO` | 0.0 | 条件嵌入共享比例 | 随上去留 | 删 |

## 12. 其他（不参与决策，仅列出）

| 参数 | 说明 |
|---|---|
| `SAVE_PATH` | batch_run 自动填充，不算用户参数 |
| `OPT_DIAG_INTERVAL` | 优化器诊断（默认 0），建议写死 0 或删——如需决策请填下栏 |
| 环境变量 `BATCH_RUN_GPUS` / `BATCH_RUN_IDLE_MEM_MB` / `BATCH_RUN_REQUIRE_GPU` | 运行层 GPU 调度，不在训练配置内，不动 |

| 参数 | 建议 | 决定 |
|---|---|---|
| `OPT_DIAG_INTERVAL` | 写死 0 或删 | 保留 |
