# 实验参数确认表

审阅版本：15bb5aa1958fc6923705
批次：fill_blank_o2_10coef_d512l2_P127_tr16；最大并发：2
此阶段只审阅参数，尚未生成可执行实验 JSON 或启动实验。

| 实验 | 任务 | 实际训练样本 | Seed | LR / WD / clip | 初始化 | 每条定时(h) |
| --- | --- | --- | --- | --- | --- | --- |
| single_o2_fillblank_a117b27_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |
| single_o2_fillblank_a25b56_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |
| single_o2_fillblank_a61b1_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |
| single_o2_fillblank_a104b105_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |
| single_o2_fillblank_a5b3_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |
| single_o2_fillblank_a78b24_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |
| single_o2_fillblank_a67b105_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |
| single_o2_fillblank_a54b113_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |
| single_o2_fillblank_a50b73_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |
| single_o2_fillblank_a83b88_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |

## single_o2_fillblank_a117b27_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | main 默认 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 6000 | main 默认 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | main 默认 |
| INIT_FROM | null | main 默认 |
| LR | 0.0003 | main 默认 |
| MAX_TRAIN_HOURS | null | main 默认 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | main 默认 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 1 | main 默认 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | main 默认 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| RANDOM_SEED | 42 | main 默认 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | main 默认 |
| RULES | [[117,27]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "fill_blank" | 用户公共参数 |
| TEST_ALL_POSITIONS | true | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | {"total":77424,"windows":4839} |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 17 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [117,27] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 17 |
| vocab_size | 129 |

- fill_blank：loss_mask 由 collate 给出（仅输出位），NUM_MASK 不参与。
- TEST_ALL_POSITIONS：测试集逐位置枚举挖空（确定性评估），test 样本数 = 窗口数 × 窗口长度；训练侧保持在线随机挖空。
- fill_blank：污损是任务内生的（恰好一位替换成空白 token id=P），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 正确值一个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_fillblank_a25b56_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | main 默认 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 6000 | main 默认 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | main 默认 |
| INIT_FROM | null | main 默认 |
| LR | 0.0003 | main 默认 |
| MAX_TRAIN_HOURS | null | main 默认 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | main 默认 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 1 | main 默认 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | main 默认 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| RANDOM_SEED | 42 | main 默认 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | main 默认 |
| RULES | [[25,56]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "fill_blank" | 用户公共参数 |
| TEST_ALL_POSITIONS | true | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | {"total":77424,"windows":4839} |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 17 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [25,56] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 17 |
| vocab_size | 129 |

- fill_blank：loss_mask 由 collate 给出（仅输出位），NUM_MASK 不参与。
- TEST_ALL_POSITIONS：测试集逐位置枚举挖空（确定性评估），test 样本数 = 窗口数 × 窗口长度；训练侧保持在线随机挖空。
- fill_blank：污损是任务内生的（恰好一位替换成空白 token id=P），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 正确值一个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_fillblank_a61b1_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | main 默认 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 6000 | main 默认 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | main 默认 |
| INIT_FROM | null | main 默认 |
| LR | 0.0003 | main 默认 |
| MAX_TRAIN_HOURS | null | main 默认 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | main 默认 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 1 | main 默认 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | main 默认 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| RANDOM_SEED | 42 | main 默认 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | main 默认 |
| RULES | [[61,1]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "fill_blank" | 用户公共参数 |
| TEST_ALL_POSITIONS | true | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | {"total":77424,"windows":4839} |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 17 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [61,1] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 17 |
| vocab_size | 129 |

- fill_blank：loss_mask 由 collate 给出（仅输出位），NUM_MASK 不参与。
- TEST_ALL_POSITIONS：测试集逐位置枚举挖空（确定性评估），test 样本数 = 窗口数 × 窗口长度；训练侧保持在线随机挖空。
- fill_blank：污损是任务内生的（恰好一位替换成空白 token id=P），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 正确值一个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_fillblank_a104b105_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | main 默认 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 6000 | main 默认 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | main 默认 |
| INIT_FROM | null | main 默认 |
| LR | 0.0003 | main 默认 |
| MAX_TRAIN_HOURS | null | main 默认 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | main 默认 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 1 | main 默认 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | main 默认 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| RANDOM_SEED | 42 | main 默认 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | main 默认 |
| RULES | [[104,105]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "fill_blank" | 用户公共参数 |
| TEST_ALL_POSITIONS | true | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | {"total":77424,"windows":4839} |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 17 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [104,105] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 17 |
| vocab_size | 129 |

- fill_blank：loss_mask 由 collate 给出（仅输出位），NUM_MASK 不参与。
- TEST_ALL_POSITIONS：测试集逐位置枚举挖空（确定性评估），test 样本数 = 窗口数 × 窗口长度；训练侧保持在线随机挖空。
- fill_blank：污损是任务内生的（恰好一位替换成空白 token id=P），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 正确值一个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_fillblank_a5b3_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | main 默认 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 6000 | main 默认 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | main 默认 |
| INIT_FROM | null | main 默认 |
| LR | 0.0003 | main 默认 |
| MAX_TRAIN_HOURS | null | main 默认 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | main 默认 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 1 | main 默认 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | main 默认 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| RANDOM_SEED | 42 | main 默认 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | main 默认 |
| RULES | [[5,3]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "fill_blank" | 用户公共参数 |
| TEST_ALL_POSITIONS | true | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | {"total":77424,"windows":4839} |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 17 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [5,3] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 17 |
| vocab_size | 129 |

- fill_blank：loss_mask 由 collate 给出（仅输出位），NUM_MASK 不参与。
- TEST_ALL_POSITIONS：测试集逐位置枚举挖空（确定性评估），test 样本数 = 窗口数 × 窗口长度；训练侧保持在线随机挖空。
- fill_blank：污损是任务内生的（恰好一位替换成空白 token id=P），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 正确值一个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_fillblank_a78b24_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | main 默认 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 6000 | main 默认 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | main 默认 |
| INIT_FROM | null | main 默认 |
| LR | 0.0003 | main 默认 |
| MAX_TRAIN_HOURS | null | main 默认 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | main 默认 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 1 | main 默认 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | main 默认 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| RANDOM_SEED | 42 | main 默认 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | main 默认 |
| RULES | [[78,24]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "fill_blank" | 用户公共参数 |
| TEST_ALL_POSITIONS | true | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | {"total":77424,"windows":4839} |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 17 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [78,24] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 17 |
| vocab_size | 129 |

- fill_blank：loss_mask 由 collate 给出（仅输出位），NUM_MASK 不参与。
- TEST_ALL_POSITIONS：测试集逐位置枚举挖空（确定性评估），test 样本数 = 窗口数 × 窗口长度；训练侧保持在线随机挖空。
- fill_blank：污损是任务内生的（恰好一位替换成空白 token id=P），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 正确值一个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_fillblank_a67b105_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | main 默认 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 6000 | main 默认 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | main 默认 |
| INIT_FROM | null | main 默认 |
| LR | 0.0003 | main 默认 |
| MAX_TRAIN_HOURS | null | main 默认 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | main 默认 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 1 | main 默认 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | main 默认 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| RANDOM_SEED | 42 | main 默认 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | main 默认 |
| RULES | [[67,105]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "fill_blank" | 用户公共参数 |
| TEST_ALL_POSITIONS | true | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | {"total":77424,"windows":4839} |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 17 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [67,105] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 17 |
| vocab_size | 129 |

- fill_blank：loss_mask 由 collate 给出（仅输出位），NUM_MASK 不参与。
- TEST_ALL_POSITIONS：测试集逐位置枚举挖空（确定性评估），test 样本数 = 窗口数 × 窗口长度；训练侧保持在线随机挖空。
- fill_blank：污损是任务内生的（恰好一位替换成空白 token id=P），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 正确值一个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_fillblank_a54b113_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | main 默认 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 6000 | main 默认 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | main 默认 |
| INIT_FROM | null | main 默认 |
| LR | 0.0003 | main 默认 |
| MAX_TRAIN_HOURS | null | main 默认 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | main 默认 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 1 | main 默认 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | main 默认 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| RANDOM_SEED | 42 | main 默认 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | main 默认 |
| RULES | [[54,113]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "fill_blank" | 用户公共参数 |
| TEST_ALL_POSITIONS | true | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | {"total":77424,"windows":4839} |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 17 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [54,113] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 17 |
| vocab_size | 129 |

- fill_blank：loss_mask 由 collate 给出（仅输出位），NUM_MASK 不参与。
- TEST_ALL_POSITIONS：测试集逐位置枚举挖空（确定性评估），test 样本数 = 窗口数 × 窗口长度；训练侧保持在线随机挖空。
- fill_blank：污损是任务内生的（恰好一位替换成空白 token id=P），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 正确值一个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_fillblank_a50b73_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | main 默认 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 6000 | main 默认 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | main 默认 |
| INIT_FROM | null | main 默认 |
| LR | 0.0003 | main 默认 |
| MAX_TRAIN_HOURS | null | main 默认 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | main 默认 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 1 | main 默认 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | main 默认 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| RANDOM_SEED | 42 | main 默认 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | main 默认 |
| RULES | [[50,73]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "fill_blank" | 用户公共参数 |
| TEST_ALL_POSITIONS | true | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | {"total":77424,"windows":4839} |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 17 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [50,73] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 17 |
| vocab_size | 129 |

- fill_blank：loss_mask 由 collate 给出（仅输出位），NUM_MASK 不参与。
- TEST_ALL_POSITIONS：测试集逐位置枚举挖空（确定性评估），test 样本数 = 窗口数 × 窗口长度；训练侧保持在线随机挖空。
- fill_blank：污损是任务内生的（恰好一位替换成空白 token id=P），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 正确值一个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_fillblank_a83b88_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | main 默认 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 6000 | main 默认 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | main 默认 |
| INIT_FROM | null | main 默认 |
| LR | 0.0003 | main 默认 |
| MAX_TRAIN_HOURS | null | main 默认 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | main 默认 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 1 | main 默认 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | main 默认 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| RANDOM_SEED | 42 | main 默认 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | main 默认 |
| RULES | [[83,88]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "fill_blank" | 用户公共参数 |
| TEST_ALL_POSITIONS | true | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | {"total":77424,"windows":4839} |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 17 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [83,88] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 17 |
| vocab_size | 129 |

- fill_blank：loss_mask 由 collate 给出（仅输出位），NUM_MASK 不参与。
- TEST_ALL_POSITIONS：测试集逐位置枚举挖空（确定性评估），test 样本数 = 窗口数 × 窗口长度；训练侧保持在线随机挖空。
- fill_blank：污损是任务内生的（恰好一位替换成空白 token id=P），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 正确值一个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

打包内容：已确认 JSON、定时写入 run.sh、训练代码快照、依赖清单、检查和运行工具。

请确认此版本；修改参数或训练代码后需重新出表。

<!-- PIPELINE_SNAPSHOT eyJhc3NldHMiOnt9LCJiYXRjaF9uYW1lIjoiZmlsbF9ibGFua19vMl8xMGNvZWZfZDUxMmwyX1AxMjdfdHIxNiIsImNvbmN1cnJlbmN5IjoyLCJleHBlcmltZW50cyI6W3siY29uZmlnIjp7IkJBVENIX1NJWkUiOjUxMiwiQ09OU1QiOjAsIkRBVEFfTU9ERSI6ImZ1bGxfc3BsaXQiLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjUxMiwiRUFSTFlfU1RPUF9BQ0NVUkFDWSI6MC45OSwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOjAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRVBPQ0hTIjo2MDAwLCJFVkFMX0lOVEVSVkFMIjoxLCJGSVJTVF9UQVNLX1dFSUdIVCI6MS4wLCJGUkVTSF9URVNUX1BFUl9FVkFMIjpmYWxzZSwiR1JBRF9BQ0NVTV9TVEVQUyI6MSwiR1JBRF9DTElQX05PUk0iOjEuMCwiSU5JVF9GUk9NIjpudWxsLCJMUiI6MC4wMDAzLCJNQVhfVFJBSU5fSE9VUlMiOm51bGwsIk1JU1NJTkdfUFJPQiI6MC4wLCJNSVNTX0xFTiI6MSwiTUxQX1JBVElPIjo4LCJOVU1fTUFTSyI6bnVsbCwiTlVNX1RFU1RfU0FNUExFUyI6MjU2LCJOVU1fVFJBSU5fU0FNUExFUyI6MTEyOTAsIk5fSEVBRCI6MSwiTl9MQVlFUiI6MiwiT09EX0xFTiI6MTYsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJPUkRFUiI6MiwiUCI6MTI3LCJSQU5ET01fU0VFRCI6NDIsIlJFU0hVRkZMRV9FQUNIX0VQT0NIIjpmYWxzZSwiUkVTVU1FX0ZST00iOm51bGwsIlJVTEVTIjpbWzExNywyN11dLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJzaW5nbGUiLCJUQVNLX1ZBUklBTlQiOiJmaWxsX2JsYW5rIiwiVEVTVF9BTExfUE9TSVRJT05TIjp0cnVlLCJUUkFJTl9MRU4iOjE2LCJXRUlHSFRfREVDQVkiOjF9LCJlZmZlY3RpdmUiOnsiYWN0dWFsX251bV9tYXNrIjoiY29sbGF0ZSBsb3NzX21hc2vvvIjkuI3lj4LkuI7vvIkiLCJhY3R1YWxfdGVzdF9zYW1wbGVzIjp7InRvdGFsIjo3NzQyNCwid2luZG93cyI6NDgzOX0sImFjdHVhbF90cmFpbl9zYW1wbGVzIjoxMTI5MCwiYmxvY2tfc2l6ZSI6MzIsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDmlrDlgLwiLCJlZmZlY3RpdmVfZGF0YV9tb2RlIjoiZnVsbF9zcGxpdCIsImV2YWxfYmFzZV9sZW5ndGgiOjE2LCJldmFsX3NlcXVlbmNlX3Rva2VucyI6MTcsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJmaWxsX2JsYW5r77yabG9zc19tYXNrIOeUsSBjb2xsYXRlIOe7meWHuu+8iOS7hei+k+WHuuS9je+8ie+8jE5VTV9NQVNLIOS4jeWPguS4juOAgiIsIlRFU1RfQUxMX1BPU0lUSU9OU++8mua1i+ivlembhumAkOS9jee9ruaemuS4vuaMluepuu+8iOehruWumuaAp+ivhOS8sO+8ie+8jHRlc3Qg5qC35pys5pWwID0g56qX5Y+j5pWwIMOXIOeql+WPo+mVv+W6pu+8m+iuree7g+S+p+S/neaMgeWcqOe6v+maj+acuuaMluepuuOAgiIsImZpbGxfYmxhbmvvvJrmsaHmjZ/mmK/ku7vliqHlhoXnlJ/nmoTvvIjmgbDlpb3kuIDkvY3mm7/mjaLmiJDnqbrnmb0gdG9rZW4gaWQ9UO+8ie+8jOaXoCBjbGVhbi90b3RhbCDkuYvliIbvvIznu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2PvvJvovpPlh7ogPSDmraPnoa7lgLzkuIDkuKogdG9rZW7jgIIiLCLpnZnmgIHmtYvor5XlnKjovr7liLAgRUFSTFlfU1RPUF9BQ0NVUkFDWSDml7blgZzmraLvvJtmcmVzaCB0ZXN0IOWcqOacgOi/kSAxMCDmrKHor4TkvLDlnYflgLzotoXov4fpmIjlgLzml7blgZzmraLjgIJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyDlt7Llup/lvIPvvIzkuI3nlJ/mlYjjgIIiLCLorq3nu4Pnu5PmnZ/kuI3lho3ot5EgZmluYWwgZ2VuZXJhdGlvbiB0ZXN077yIMjAyNi0xMC0wOCDliKDpmaTvvInvvJvnu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2MgKyBldmFsX2NsZWFuX2FjdGlvbi5weeOAguato+W4uOWujOaIkOWQjui3kSBhdHRlbnRpb24g5YiG5p6Q77yb5a2Y5qGj6YCA5Ye656CBIDQyIOi3s+i/h+WQjuWkhOeQhuOAgiJdLCJvcHRpbWl6ZXIiOiJBZGFtVyIsIm91dHB1dF9wYXRoIjoiPG91dHB1dD4vbW9kZWxzLzxleHBlcmltZW50Pi88ZXhwZXJpbWVudD4ucHRoIiwicmVjdXJyZW5jZV9vcmRlciI6MiwicnVsZXMiOlsxMTcsMjddLCJzY2hlZHVsZXIiOiJDb3NpbmVBbm5lYWxpbmdMUiIsInNjaGVkdWxlcl9UX21heCI6NjAwMCwidHJhaW5fZXhwb3N1cmUiOjAuNjk5OTgxMzk5OTYyOCwidHJhaW5fc2VxdWVuY2VfdG9rZW5zIjoxNywidm9jYWJfc2l6ZSI6MTI5fSwibmFtZSI6InNpbmdsZV9vMl9maWxsYmxhbmtfYTExN2IyN19kNTEybDJyOGgxX1AxMjdfdHIxNl9zZWVkNDIiLCJvcmlnaW5zIjp7IkJBVENIX1NJWkUiOiJtYWluIOm7mOiupCIsIkNPTlNUIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoibWFpbiDpu5jorqQiLCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6Im1haW4g6buY6K6kIiwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjoibWFpbiDpu5jorqQiLCJFUE9DSFMiOiJtYWluIOm7mOiupCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6Im1haW4g6buY6K6kIiwiR1JBRF9BQ0NVTV9TVEVQUyI6Im1haW4g6buY6K6kIiwiR1JBRF9DTElQX05PUk0iOiJtYWluIOm7mOiupCIsIklOSVRfRlJPTSI6Im1haW4g6buY6K6kIiwiTFIiOiJtYWluIOm7mOiupCIsIk1BWF9UUkFJTl9IT1VSUyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoibWFpbiDpu5jorqQiLCJOVU1fTUFTSyI6Im1haW4g6buY6K6kIiwiTlVNX1RFU1RfU0FNUExFUyI6Im1haW4g6buY6K6kIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5fSEVBRCI6Im1haW4g6buY6K6kIiwiTl9MQVlFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9PRF9MRU4iOiJtYWluIOm7mOiupCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSQU5ET01fU0VFRCI6Im1haW4g6buY6K6kIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoibWFpbiDpu5jorqQiLCJSVUxFUyI6IueUqOaIt+mAkOWunumqjOWPguaVsCIsIlNUQVRFX1NQQUNFX0NBUCI6Im1haW4g6buY6K6kIiwiVEFTSyI6IuWunumqjOS7u+WKoSIsIlRBU0tfVkFSSUFOVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlRFU1RfQUxMX1BPU0lUSU9OUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6Im1haW4g6buY6K6kIn0sInRhc2siOiJzaW5nbGUifSx7ImNvbmZpZyI6eyJCQVRDSF9TSVpFIjo1MTIsIkNPTlNUIjowLCJEQVRBX01PREUiOiJmdWxsX3NwbGl0IiwiRFJPUE9VVCI6MC4wLCJEX01PREVMIjo1MTIsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjowLCJFQVJMWV9TVE9QX05PX0lNUFJPVkUiOjMwMDAsIkVQT0NIUyI6NjAwMCwiRVZBTF9JTlRFUlZBTCI6MSwiRklSU1RfVEFTS19XRUlHSFQiOjEuMCwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6ZmFsc2UsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjpudWxsLCJNSVNTSU5HX1BST0IiOjAuMCwiTUlTU19MRU4iOjEsIk1MUF9SQVRJTyI6OCwiTlVNX01BU0siOm51bGwsIk5VTV9URVNUX1NBTVBMRVMiOjI1NiwiTlVNX1RSQUlOX1NBTVBMRVMiOjExMjkwLCJOX0hFQUQiOjEsIk5fTEFZRVIiOjIsIk9PRF9MRU4iOjE2LCJPUFRfRElBR19JTlRFUlZBTCI6MCwiT1JERVIiOjIsIlAiOjEyNywiUkFORE9NX1NFRUQiOjQyLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6ZmFsc2UsIlJFU1VNRV9GUk9NIjpudWxsLCJSVUxFUyI6W1syNSw1Nl1dLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJzaW5nbGUiLCJUQVNLX1ZBUklBTlQiOiJmaWxsX2JsYW5rIiwiVEVTVF9BTExfUE9TSVRJT05TIjp0cnVlLCJUUkFJTl9MRU4iOjE2LCJXRUlHSFRfREVDQVkiOjF9LCJlZmZlY3RpdmUiOnsiYWN0dWFsX251bV9tYXNrIjoiY29sbGF0ZSBsb3NzX21hc2vvvIjkuI3lj4LkuI7vvIkiLCJhY3R1YWxfdGVzdF9zYW1wbGVzIjp7InRvdGFsIjo3NzQyNCwid2luZG93cyI6NDgzOX0sImFjdHVhbF90cmFpbl9zYW1wbGVzIjoxMTI5MCwiYmxvY2tfc2l6ZSI6MzIsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDmlrDlgLwiLCJlZmZlY3RpdmVfZGF0YV9tb2RlIjoiZnVsbF9zcGxpdCIsImV2YWxfYmFzZV9sZW5ndGgiOjE2LCJldmFsX3NlcXVlbmNlX3Rva2VucyI6MTcsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJmaWxsX2JsYW5r77yabG9zc19tYXNrIOeUsSBjb2xsYXRlIOe7meWHuu+8iOS7hei+k+WHuuS9je+8ie+8jE5VTV9NQVNLIOS4jeWPguS4juOAgiIsIlRFU1RfQUxMX1BPU0lUSU9OU++8mua1i+ivlembhumAkOS9jee9ruaemuS4vuaMluepuu+8iOehruWumuaAp+ivhOS8sO+8ie+8jHRlc3Qg5qC35pys5pWwID0g56qX5Y+j5pWwIMOXIOeql+WPo+mVv+W6pu+8m+iuree7g+S+p+S/neaMgeWcqOe6v+maj+acuuaMluepuuOAgiIsImZpbGxfYmxhbmvvvJrmsaHmjZ/mmK/ku7vliqHlhoXnlJ/nmoTvvIjmgbDlpb3kuIDkvY3mm7/mjaLmiJDnqbrnmb0gdG9rZW4gaWQ9UO+8ie+8jOaXoCBjbGVhbi90b3RhbCDkuYvliIbvvIznu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2PvvJvovpPlh7ogPSDmraPnoa7lgLzkuIDkuKogdG9rZW7jgIIiLCLpnZnmgIHmtYvor5XlnKjovr7liLAgRUFSTFlfU1RPUF9BQ0NVUkFDWSDml7blgZzmraLvvJtmcmVzaCB0ZXN0IOWcqOacgOi/kSAxMCDmrKHor4TkvLDlnYflgLzotoXov4fpmIjlgLzml7blgZzmraLjgIJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyDlt7Llup/lvIPvvIzkuI3nlJ/mlYjjgIIiLCLorq3nu4Pnu5PmnZ/kuI3lho3ot5EgZmluYWwgZ2VuZXJhdGlvbiB0ZXN077yIMjAyNi0xMC0wOCDliKDpmaTvvInvvJvnu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2MgKyBldmFsX2NsZWFuX2FjdGlvbi5weeOAguato+W4uOWujOaIkOWQjui3kSBhdHRlbnRpb24g5YiG5p6Q77yb5a2Y5qGj6YCA5Ye656CBIDQyIOi3s+i/h+WQjuWkhOeQhuOAgiJdLCJvcHRpbWl6ZXIiOiJBZGFtVyIsIm91dHB1dF9wYXRoIjoiPG91dHB1dD4vbW9kZWxzLzxleHBlcmltZW50Pi88ZXhwZXJpbWVudD4ucHRoIiwicmVjdXJyZW5jZV9vcmRlciI6MiwicnVsZXMiOlsyNSw1Nl0sInNjaGVkdWxlciI6IkNvc2luZUFubmVhbGluZ0xSIiwic2NoZWR1bGVyX1RfbWF4Ijo2MDAwLCJ0cmFpbl9leHBvc3VyZSI6MC42OTk5ODEzOTk5NjI4LCJ0cmFpbl9zZXF1ZW5jZV90b2tlbnMiOjE3LCJ2b2NhYl9zaXplIjoxMjl9LCJuYW1lIjoic2luZ2xlX28yX2ZpbGxibGFua19hMjViNTZfZDUxMmwycjhoMV9QMTI3X3RyMTZfc2VlZDQyIiwib3JpZ2lucyI6eyJCQVRDSF9TSVpFIjoibWFpbiDpu5jorqQiLCJDT05TVCI6Im1haW4g6buY6K6kIiwiREFUQV9NT0RFIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJEUk9QT1VUIjoibWFpbiDpu5jorqQiLCJEX01PREVMIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9BQ0NVUkFDWSI6Im1haW4g6buY6K6kIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoibWFpbiDpu5jorqQiLCJFVkFMX0lOVEVSVkFMIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRklSU1RfVEFTS19XRUlHSFQiOiJtYWluIOm7mOiupCIsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOiJtYWluIOm7mOiupCIsIkdSQURfQUNDVU1fU1RFUFMiOiJtYWluIOm7mOiupCIsIkdSQURfQ0xJUF9OT1JNIjoibWFpbiDpu5jorqQiLCJJTklUX0ZST00iOiJtYWluIOm7mOiupCIsIkxSIjoibWFpbiDpu5jorqQiLCJNQVhfVFJBSU5fSE9VUlMiOiJtYWluIOm7mOiupCIsIk1JU1NJTkdfUFJPQiI6Im1haW4g6buY6K6kIiwiTUlTU19MRU4iOiJtYWluIOm7mOiupCIsIk1MUF9SQVRJTyI6Im1haW4g6buY6K6kIiwiTlVNX01BU0siOiJtYWluIOm7mOiupCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiJtYWluIOm7mOiupCIsIk5fTEFZRVIiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJPT0RfTEVOIjoibWFpbiDpu5jorqQiLCJPUFRfRElBR19JTlRFUlZBTCI6Im1haW4g6buY6K6kIiwiT1JERVIiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUkFORE9NX1NFRUQiOiJtYWluIOm7mOiupCIsIlJFU0hVRkZMRV9FQUNIX0VQT0NIIjoibWFpbiDpu5jorqQiLCJSRVNVTUVfRlJPTSI6Im1haW4g6buY6K6kIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJURVNUX0FMTF9QT1NJVElPTlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJXRUlHSFRfREVDQVkiOiJtYWluIOm7mOiupCJ9LCJ0YXNrIjoic2luZ2xlIn0seyJjb25maWciOnsiQkFUQ0hfU0laRSI6NTEyLCJDT05TVCI6MCwiREFUQV9NT0RFIjoiZnVsbF9zcGxpdCIsIkRST1BPVVQiOjAuMCwiRF9NT0RFTCI6NTEyLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjowLjk5LCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6MCwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjozMDAwLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOmZhbHNlLCJHUkFEX0FDQ1VNX1NURVBTIjoxLCJHUkFEX0NMSVBfTk9STSI6MS4wLCJJTklUX0ZST00iOm51bGwsIkxSIjowLjAwMDMsIk1BWF9UUkFJTl9IT1VSUyI6bnVsbCwiTUlTU0lOR19QUk9CIjowLjAsIk1JU1NfTEVOIjoxLCJNTFBfUkFUSU8iOjgsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjoxMTI5MCwiTl9IRUFEIjoxLCJOX0xBWUVSIjoyLCJPT0RfTEVOIjoxNiwiT1BUX0RJQUdfSU5URVJWQUwiOjAsIk9SREVSIjoyLCJQIjoxMjcsIlJBTkRPTV9TRUVEIjo0MiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiUlVMRVMiOltbNjEsMV1dLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJzaW5nbGUiLCJUQVNLX1ZBUklBTlQiOiJmaWxsX2JsYW5rIiwiVEVTVF9BTExfUE9TSVRJT05TIjp0cnVlLCJUUkFJTl9MRU4iOjE2LCJXRUlHSFRfREVDQVkiOjF9LCJlZmZlY3RpdmUiOnsiYWN0dWFsX251bV9tYXNrIjoiY29sbGF0ZSBsb3NzX21hc2vvvIjkuI3lj4LkuI7vvIkiLCJhY3R1YWxfdGVzdF9zYW1wbGVzIjp7InRvdGFsIjo3NzQyNCwid2luZG93cyI6NDgzOX0sImFjdHVhbF90cmFpbl9zYW1wbGVzIjoxMTI5MCwiYmxvY2tfc2l6ZSI6MzIsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDmlrDlgLwiLCJlZmZlY3RpdmVfZGF0YV9tb2RlIjoiZnVsbF9zcGxpdCIsImV2YWxfYmFzZV9sZW5ndGgiOjE2LCJldmFsX3NlcXVlbmNlX3Rva2VucyI6MTcsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJmaWxsX2JsYW5r77yabG9zc19tYXNrIOeUsSBjb2xsYXRlIOe7meWHuu+8iOS7hei+k+WHuuS9je+8ie+8jE5VTV9NQVNLIOS4jeWPguS4juOAgiIsIlRFU1RfQUxMX1BPU0lUSU9OU++8mua1i+ivlembhumAkOS9jee9ruaemuS4vuaMluepuu+8iOehruWumuaAp+ivhOS8sO+8ie+8jHRlc3Qg5qC35pys5pWwID0g56qX5Y+j5pWwIMOXIOeql+WPo+mVv+W6pu+8m+iuree7g+S+p+S/neaMgeWcqOe6v+maj+acuuaMluepuuOAgiIsImZpbGxfYmxhbmvvvJrmsaHmjZ/mmK/ku7vliqHlhoXnlJ/nmoTvvIjmgbDlpb3kuIDkvY3mm7/mjaLmiJDnqbrnmb0gdG9rZW4gaWQ9UO+8ie+8jOaXoCBjbGVhbi90b3RhbCDkuYvliIbvvIznu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2PvvJvovpPlh7ogPSDmraPnoa7lgLzkuIDkuKogdG9rZW7jgIIiLCLpnZnmgIHmtYvor5XlnKjovr7liLAgRUFSTFlfU1RPUF9BQ0NVUkFDWSDml7blgZzmraLvvJtmcmVzaCB0ZXN0IOWcqOacgOi/kSAxMCDmrKHor4TkvLDlnYflgLzotoXov4fpmIjlgLzml7blgZzmraLjgIJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyDlt7Llup/lvIPvvIzkuI3nlJ/mlYjjgIIiLCLorq3nu4Pnu5PmnZ/kuI3lho3ot5EgZmluYWwgZ2VuZXJhdGlvbiB0ZXN077yIMjAyNi0xMC0wOCDliKDpmaTvvInvvJvnu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2MgKyBldmFsX2NsZWFuX2FjdGlvbi5weeOAguato+W4uOWujOaIkOWQjui3kSBhdHRlbnRpb24g5YiG5p6Q77yb5a2Y5qGj6YCA5Ye656CBIDQyIOi3s+i/h+WQjuWkhOeQhuOAgiJdLCJvcHRpbWl6ZXIiOiJBZGFtVyIsIm91dHB1dF9wYXRoIjoiPG91dHB1dD4vbW9kZWxzLzxleHBlcmltZW50Pi88ZXhwZXJpbWVudD4ucHRoIiwicmVjdXJyZW5jZV9vcmRlciI6MiwicnVsZXMiOls2MSwxXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjowLjY5OTk4MTM5OTk2MjgsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTcsInZvY2FiX3NpemUiOjEyOX0sIm5hbWUiOiJzaW5nbGVfbzJfZmlsbGJsYW5rX2E2MWIxX2Q1MTJsMnI4aDFfUDEyN190cjE2X3NlZWQ0MiIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6Im1haW4g6buY6K6kIiwiQ09OU1QiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6IuS7o+eggeivreS5ieaOqOWvvO+8m+WOn+WAvD1udWxsIiwiRFJPUE9VVCI6Im1haW4g6buY6K6kIiwiRF9NT0RFTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoibWFpbiDpu5jorqQiLCJFQVJMWV9TVE9QX05PX0lNUFJPVkUiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6Im1haW4g6buY6K6kIiwiRVZBTF9JTlRFUlZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkZJUlNUX1RBU0tfV0VJR0hUIjoibWFpbiDpu5jorqQiLCJGUkVTSF9URVNUX1BFUl9FVkFMIjoibWFpbiDpu5jorqQiLCJHUkFEX0FDQ1VNX1NURVBTIjoibWFpbiDpu5jorqQiLCJHUkFEX0NMSVBfTk9STSI6Im1haW4g6buY6K6kIiwiSU5JVF9GUk9NIjoibWFpbiDpu5jorqQiLCJMUiI6Im1haW4g6buY6K6kIiwiTUFYX1RSQUlOX0hPVVJTIjoibWFpbiDpu5jorqQiLCJNSVNTSU5HX1BST0IiOiJtYWluIOm7mOiupCIsIk1JU1NfTEVOIjoibWFpbiDpu5jorqQiLCJNTFBfUkFUSU8iOiJtYWluIOm7mOiupCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoibWFpbiDpu5jorqQiLCJOVU1fVFJBSU5fU0FNUExFUyI6IuS7o+eggeivreS5ieaOqOWvvO+8m+WOn+WAvD1udWxsIiwiTl9IRUFEIjoibWFpbiDpu5jorqQiLCJOX0xBWUVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiT09EX0xFTiI6Im1haW4g6buY6K6kIiwiT1BUX0RJQUdfSU5URVJWQUwiOiJtYWluIOm7mOiupCIsIk9SREVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJBTkRPTV9TRUVEIjoibWFpbiDpu5jorqQiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6Im1haW4g6buY6K6kIiwiUkVTVU1FX0ZST00iOiJtYWluIOm7mOiupCIsIlJVTEVTIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiU1RBVEVfU1BBQ0VfQ0FQIjoibWFpbiDpu5jorqQiLCJUQVNLIjoi5a6e6aqM5Lu75YqhIiwiVEFTS19WQVJJQU5UIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVEVTVF9BTExfUE9TSVRJT05TIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVFJBSU5fTEVOIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiV0VJR0hUX0RFQ0FZIjoibWFpbiDpu5jorqQifSwidGFzayI6InNpbmdsZSJ9LHsiY29uZmlnIjp7IkJBVENIX1NJWkUiOjUxMiwiQ09OU1QiOjAsIkRBVEFfTU9ERSI6ImZ1bGxfc3BsaXQiLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjUxMiwiRUFSTFlfU1RPUF9BQ0NVUkFDWSI6MC45OSwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOjAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRVBPQ0hTIjo2MDAwLCJFVkFMX0lOVEVSVkFMIjoxLCJGSVJTVF9UQVNLX1dFSUdIVCI6MS4wLCJGUkVTSF9URVNUX1BFUl9FVkFMIjpmYWxzZSwiR1JBRF9BQ0NVTV9TVEVQUyI6MSwiR1JBRF9DTElQX05PUk0iOjEuMCwiSU5JVF9GUk9NIjpudWxsLCJMUiI6MC4wMDAzLCJNQVhfVFJBSU5fSE9VUlMiOm51bGwsIk1JU1NJTkdfUFJPQiI6MC4wLCJNSVNTX0xFTiI6MSwiTUxQX1JBVElPIjo4LCJOVU1fTUFTSyI6bnVsbCwiTlVNX1RFU1RfU0FNUExFUyI6MjU2LCJOVU1fVFJBSU5fU0FNUExFUyI6MTEyOTAsIk5fSEVBRCI6MSwiTl9MQVlFUiI6MiwiT09EX0xFTiI6MTYsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJPUkRFUiI6MiwiUCI6MTI3LCJSQU5ET01fU0VFRCI6NDIsIlJFU0hVRkZMRV9FQUNIX0VQT0NIIjpmYWxzZSwiUkVTVU1FX0ZST00iOm51bGwsIlJVTEVTIjpbWzEwNCwxMDVdXSwiU1RBVEVfU1BBQ0VfQ0FQIjpudWxsLCJUQVNLIjoic2luZ2xlIiwiVEFTS19WQVJJQU5UIjoiZmlsbF9ibGFuayIsIlRFU1RfQUxMX1BPU0lUSU9OUyI6dHJ1ZSwiVFJBSU5fTEVOIjoxNiwiV0VJR0hUX0RFQ0FZIjoxfSwiZWZmZWN0aXZlIjp7ImFjdHVhbF9udW1fbWFzayI6ImNvbGxhdGUgbG9zc19tYXNr77yI5LiN5Y+C5LiO77yJIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6eyJ0b3RhbCI6Nzc0MjQsIndpbmRvd3MiOjQ4Mzl9LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MTEyOTAsImJsb2NrX3NpemUiOjMyLCJjb2VmZmljaWVudF9jb252ZW50aW9uIjoi56ys5LiA5Liq57O75pWw5LmY5pyA5paw5YC8IiwiZWZmZWN0aXZlX2RhdGFfbW9kZSI6ImZ1bGxfc3BsaXQiLCJldmFsX2Jhc2VfbGVuZ3RoIjoxNiwiZXZhbF9zZXF1ZW5jZV90b2tlbnMiOjE3LCJpbml0aWFsaXphdGlvbiI6ImZyZXNoIiwibm90ZXMiOlsiZmlsbF9ibGFua++8mmxvc3NfbWFzayDnlLEgY29sbGF0ZSDnu5nlh7rvvIjku4XovpPlh7rkvY3vvInvvIxOVU1fTUFTSyDkuI3lj4LkuI7jgIIiLCJURVNUX0FMTF9QT1NJVElPTlPvvJrmtYvor5Xpm4bpgJDkvY3nva7mnprkuL7mjJbnqbrvvIjnoa7lrprmgKfor4TkvLDvvInvvIx0ZXN0IOagt+acrOaVsCA9IOeql+WPo+aVsCDDlyDnqpflj6Pplb/luqbvvJvorq3nu4Pkvqfkv53mjIHlnKjnur/pmo/mnLrmjJbnqbrjgIIiLCJmaWxsX2JsYW5r77ya5rGh5o2f5piv5Lu75Yqh5YaF55Sf55qE77yI5oGw5aW95LiA5L2N5pu/5o2i5oiQ56m655m9IHRva2VuIGlkPVDvvInvvIzml6AgY2xlYW4vdG90YWwg5LmL5YiG77yM57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNj77yb6L6T5Ye6ID0g5q2j56Gu5YC85LiA5LiqIHRva2Vu44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5LiN5YaN6LeRIGZpbmFsIGdlbmVyYXRpb24gdGVzdO+8iDIwMjYtMTAtMDgg5Yig6Zmk77yJ77yb57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNjICsgZXZhbF9jbGVhbl9hY3Rpb24ucHnjgILmraPluLjlrozmiJDlkI7ot5EgYXR0ZW50aW9uIOWIhuaekO+8m+WtmOaho+mAgOWHuueggSA0MiDot7Pov4flkI7lpITnkIbjgIIiXSwib3B0aW1pemVyIjoiQWRhbVciLCJvdXRwdXRfcGF0aCI6IjxvdXRwdXQ+L21vZGVscy88ZXhwZXJpbWVudD4vPGV4cGVyaW1lbnQ+LnB0aCIsInJlY3VycmVuY2Vfb3JkZXIiOjIsInJ1bGVzIjpbMTA0LDEwNV0sInNjaGVkdWxlciI6IkNvc2luZUFubmVhbGluZ0xSIiwic2NoZWR1bGVyX1RfbWF4Ijo2MDAwLCJ0cmFpbl9leHBvc3VyZSI6MC42OTk5ODEzOTk5NjI4LCJ0cmFpbl9zZXF1ZW5jZV90b2tlbnMiOjE3LCJ2b2NhYl9zaXplIjoxMjl9LCJuYW1lIjoic2luZ2xlX28yX2ZpbGxibGFua19hMTA0YjEwNV9kNTEybDJyOGgxX1AxMjdfdHIxNl9zZWVkNDIiLCJvcmlnaW5zIjp7IkJBVENIX1NJWkUiOiJtYWluIOm7mOiupCIsIkNPTlNUIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoibWFpbiDpu5jorqQiLCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6Im1haW4g6buY6K6kIiwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjoibWFpbiDpu5jorqQiLCJFUE9DSFMiOiJtYWluIOm7mOiupCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6Im1haW4g6buY6K6kIiwiR1JBRF9BQ0NVTV9TVEVQUyI6Im1haW4g6buY6K6kIiwiR1JBRF9DTElQX05PUk0iOiJtYWluIOm7mOiupCIsIklOSVRfRlJPTSI6Im1haW4g6buY6K6kIiwiTFIiOiJtYWluIOm7mOiupCIsIk1BWF9UUkFJTl9IT1VSUyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoibWFpbiDpu5jorqQiLCJOVU1fTUFTSyI6Im1haW4g6buY6K6kIiwiTlVNX1RFU1RfU0FNUExFUyI6Im1haW4g6buY6K6kIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5fSEVBRCI6Im1haW4g6buY6K6kIiwiTl9MQVlFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9PRF9MRU4iOiJtYWluIOm7mOiupCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSQU5ET01fU0VFRCI6Im1haW4g6buY6K6kIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoibWFpbiDpu5jorqQiLCJSVUxFUyI6IueUqOaIt+mAkOWunumqjOWPguaVsCIsIlNUQVRFX1NQQUNFX0NBUCI6Im1haW4g6buY6K6kIiwiVEFTSyI6IuWunumqjOS7u+WKoSIsIlRBU0tfVkFSSUFOVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlRFU1RfQUxMX1BPU0lUSU9OUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6Im1haW4g6buY6K6kIn0sInRhc2siOiJzaW5nbGUifSx7ImNvbmZpZyI6eyJCQVRDSF9TSVpFIjo1MTIsIkNPTlNUIjowLCJEQVRBX01PREUiOiJmdWxsX3NwbGl0IiwiRFJPUE9VVCI6MC4wLCJEX01PREVMIjo1MTIsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjowLCJFQVJMWV9TVE9QX05PX0lNUFJPVkUiOjMwMDAsIkVQT0NIUyI6NjAwMCwiRVZBTF9JTlRFUlZBTCI6MSwiRklSU1RfVEFTS19XRUlHSFQiOjEuMCwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6ZmFsc2UsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjpudWxsLCJNSVNTSU5HX1BST0IiOjAuMCwiTUlTU19MRU4iOjEsIk1MUF9SQVRJTyI6OCwiTlVNX01BU0siOm51bGwsIk5VTV9URVNUX1NBTVBMRVMiOjI1NiwiTlVNX1RSQUlOX1NBTVBMRVMiOjExMjkwLCJOX0hFQUQiOjEsIk5fTEFZRVIiOjIsIk9PRF9MRU4iOjE2LCJPUFRfRElBR19JTlRFUlZBTCI6MCwiT1JERVIiOjIsIlAiOjEyNywiUkFORE9NX1NFRUQiOjQyLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6ZmFsc2UsIlJFU1VNRV9GUk9NIjpudWxsLCJSVUxFUyI6W1s1LDNdXSwiU1RBVEVfU1BBQ0VfQ0FQIjpudWxsLCJUQVNLIjoic2luZ2xlIiwiVEFTS19WQVJJQU5UIjoiZmlsbF9ibGFuayIsIlRFU1RfQUxMX1BPU0lUSU9OUyI6dHJ1ZSwiVFJBSU5fTEVOIjoxNiwiV0VJR0hUX0RFQ0FZIjoxfSwiZWZmZWN0aXZlIjp7ImFjdHVhbF9udW1fbWFzayI6ImNvbGxhdGUgbG9zc19tYXNr77yI5LiN5Y+C5LiO77yJIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6eyJ0b3RhbCI6Nzc0MjQsIndpbmRvd3MiOjQ4Mzl9LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MTEyOTAsImJsb2NrX3NpemUiOjMyLCJjb2VmZmljaWVudF9jb252ZW50aW9uIjoi56ys5LiA5Liq57O75pWw5LmY5pyA5paw5YC8IiwiZWZmZWN0aXZlX2RhdGFfbW9kZSI6ImZ1bGxfc3BsaXQiLCJldmFsX2Jhc2VfbGVuZ3RoIjoxNiwiZXZhbF9zZXF1ZW5jZV90b2tlbnMiOjE3LCJpbml0aWFsaXphdGlvbiI6ImZyZXNoIiwibm90ZXMiOlsiZmlsbF9ibGFua++8mmxvc3NfbWFzayDnlLEgY29sbGF0ZSDnu5nlh7rvvIjku4XovpPlh7rkvY3vvInvvIxOVU1fTUFTSyDkuI3lj4LkuI7jgIIiLCJURVNUX0FMTF9QT1NJVElPTlPvvJrmtYvor5Xpm4bpgJDkvY3nva7mnprkuL7mjJbnqbrvvIjnoa7lrprmgKfor4TkvLDvvInvvIx0ZXN0IOagt+acrOaVsCA9IOeql+WPo+aVsCDDlyDnqpflj6Pplb/luqbvvJvorq3nu4Pkvqfkv53mjIHlnKjnur/pmo/mnLrmjJbnqbrjgIIiLCJmaWxsX2JsYW5r77ya5rGh5o2f5piv5Lu75Yqh5YaF55Sf55qE77yI5oGw5aW95LiA5L2N5pu/5o2i5oiQ56m655m9IHRva2VuIGlkPVDvvInvvIzml6AgY2xlYW4vdG90YWwg5LmL5YiG77yM57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNj77yb6L6T5Ye6ID0g5q2j56Gu5YC85LiA5LiqIHRva2Vu44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5LiN5YaN6LeRIGZpbmFsIGdlbmVyYXRpb24gdGVzdO+8iDIwMjYtMTAtMDgg5Yig6Zmk77yJ77yb57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNjICsgZXZhbF9jbGVhbl9hY3Rpb24ucHnjgILmraPluLjlrozmiJDlkI7ot5EgYXR0ZW50aW9uIOWIhuaekO+8m+WtmOaho+mAgOWHuueggSA0MiDot7Pov4flkI7lpITnkIbjgIIiXSwib3B0aW1pemVyIjoiQWRhbVciLCJvdXRwdXRfcGF0aCI6IjxvdXRwdXQ+L21vZGVscy88ZXhwZXJpbWVudD4vPGV4cGVyaW1lbnQ+LnB0aCIsInJlY3VycmVuY2Vfb3JkZXIiOjIsInJ1bGVzIjpbNSwzXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjowLjY5OTk4MTM5OTk2MjgsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTcsInZvY2FiX3NpemUiOjEyOX0sIm5hbWUiOiJzaW5nbGVfbzJfZmlsbGJsYW5rX2E1YjNfZDUxMmwycjhoMV9QMTI3X3RyMTZfc2VlZDQyIiwib3JpZ2lucyI6eyJCQVRDSF9TSVpFIjoibWFpbiDpu5jorqQiLCJDT05TVCI6Im1haW4g6buY6K6kIiwiREFUQV9NT0RFIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJEUk9QT1VUIjoibWFpbiDpu5jorqQiLCJEX01PREVMIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9BQ0NVUkFDWSI6Im1haW4g6buY6K6kIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoibWFpbiDpu5jorqQiLCJFVkFMX0lOVEVSVkFMIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRklSU1RfVEFTS19XRUlHSFQiOiJtYWluIOm7mOiupCIsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOiJtYWluIOm7mOiupCIsIkdSQURfQUNDVU1fU1RFUFMiOiJtYWluIOm7mOiupCIsIkdSQURfQ0xJUF9OT1JNIjoibWFpbiDpu5jorqQiLCJJTklUX0ZST00iOiJtYWluIOm7mOiupCIsIkxSIjoibWFpbiDpu5jorqQiLCJNQVhfVFJBSU5fSE9VUlMiOiJtYWluIOm7mOiupCIsIk1JU1NJTkdfUFJPQiI6Im1haW4g6buY6K6kIiwiTUlTU19MRU4iOiJtYWluIOm7mOiupCIsIk1MUF9SQVRJTyI6Im1haW4g6buY6K6kIiwiTlVNX01BU0siOiJtYWluIOm7mOiupCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiJtYWluIOm7mOiupCIsIk5fTEFZRVIiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJPT0RfTEVOIjoibWFpbiDpu5jorqQiLCJPUFRfRElBR19JTlRFUlZBTCI6Im1haW4g6buY6K6kIiwiT1JERVIiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUkFORE9NX1NFRUQiOiJtYWluIOm7mOiupCIsIlJFU0hVRkZMRV9FQUNIX0VQT0NIIjoibWFpbiDpu5jorqQiLCJSRVNVTUVfRlJPTSI6Im1haW4g6buY6K6kIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJURVNUX0FMTF9QT1NJVElPTlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJXRUlHSFRfREVDQVkiOiJtYWluIOm7mOiupCJ9LCJ0YXNrIjoic2luZ2xlIn0seyJjb25maWciOnsiQkFUQ0hfU0laRSI6NTEyLCJDT05TVCI6MCwiREFUQV9NT0RFIjoiZnVsbF9zcGxpdCIsIkRST1BPVVQiOjAuMCwiRF9NT0RFTCI6NTEyLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjowLjk5LCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6MCwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjozMDAwLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOmZhbHNlLCJHUkFEX0FDQ1VNX1NURVBTIjoxLCJHUkFEX0NMSVBfTk9STSI6MS4wLCJJTklUX0ZST00iOm51bGwsIkxSIjowLjAwMDMsIk1BWF9UUkFJTl9IT1VSUyI6bnVsbCwiTUlTU0lOR19QUk9CIjowLjAsIk1JU1NfTEVOIjoxLCJNTFBfUkFUSU8iOjgsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjoxMTI5MCwiTl9IRUFEIjoxLCJOX0xBWUVSIjoyLCJPT0RfTEVOIjoxNiwiT1BUX0RJQUdfSU5URVJWQUwiOjAsIk9SREVSIjoyLCJQIjoxMjcsIlJBTkRPTV9TRUVEIjo0MiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiUlVMRVMiOltbNzgsMjRdXSwiU1RBVEVfU1BBQ0VfQ0FQIjpudWxsLCJUQVNLIjoic2luZ2xlIiwiVEFTS19WQVJJQU5UIjoiZmlsbF9ibGFuayIsIlRFU1RfQUxMX1BPU0lUSU9OUyI6dHJ1ZSwiVFJBSU5fTEVOIjoxNiwiV0VJR0hUX0RFQ0FZIjoxfSwiZWZmZWN0aXZlIjp7ImFjdHVhbF9udW1fbWFzayI6ImNvbGxhdGUgbG9zc19tYXNr77yI5LiN5Y+C5LiO77yJIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6eyJ0b3RhbCI6Nzc0MjQsIndpbmRvd3MiOjQ4Mzl9LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MTEyOTAsImJsb2NrX3NpemUiOjMyLCJjb2VmZmljaWVudF9jb252ZW50aW9uIjoi56ys5LiA5Liq57O75pWw5LmY5pyA5paw5YC8IiwiZWZmZWN0aXZlX2RhdGFfbW9kZSI6ImZ1bGxfc3BsaXQiLCJldmFsX2Jhc2VfbGVuZ3RoIjoxNiwiZXZhbF9zZXF1ZW5jZV90b2tlbnMiOjE3LCJpbml0aWFsaXphdGlvbiI6ImZyZXNoIiwibm90ZXMiOlsiZmlsbF9ibGFua++8mmxvc3NfbWFzayDnlLEgY29sbGF0ZSDnu5nlh7rvvIjku4XovpPlh7rkvY3vvInvvIxOVU1fTUFTSyDkuI3lj4LkuI7jgIIiLCJURVNUX0FMTF9QT1NJVElPTlPvvJrmtYvor5Xpm4bpgJDkvY3nva7mnprkuL7mjJbnqbrvvIjnoa7lrprmgKfor4TkvLDvvInvvIx0ZXN0IOagt+acrOaVsCA9IOeql+WPo+aVsCDDlyDnqpflj6Pplb/luqbvvJvorq3nu4Pkvqfkv53mjIHlnKjnur/pmo/mnLrmjJbnqbrjgIIiLCJmaWxsX2JsYW5r77ya5rGh5o2f5piv5Lu75Yqh5YaF55Sf55qE77yI5oGw5aW95LiA5L2N5pu/5o2i5oiQ56m655m9IHRva2VuIGlkPVDvvInvvIzml6AgY2xlYW4vdG90YWwg5LmL5YiG77yM57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNj77yb6L6T5Ye6ID0g5q2j56Gu5YC85LiA5LiqIHRva2Vu44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5LiN5YaN6LeRIGZpbmFsIGdlbmVyYXRpb24gdGVzdO+8iDIwMjYtMTAtMDgg5Yig6Zmk77yJ77yb57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNjICsgZXZhbF9jbGVhbl9hY3Rpb24ucHnjgILmraPluLjlrozmiJDlkI7ot5EgYXR0ZW50aW9uIOWIhuaekO+8m+WtmOaho+mAgOWHuueggSA0MiDot7Pov4flkI7lpITnkIbjgIIiXSwib3B0aW1pemVyIjoiQWRhbVciLCJvdXRwdXRfcGF0aCI6IjxvdXRwdXQ+L21vZGVscy88ZXhwZXJpbWVudD4vPGV4cGVyaW1lbnQ+LnB0aCIsInJlY3VycmVuY2Vfb3JkZXIiOjIsInJ1bGVzIjpbNzgsMjRdLCJzY2hlZHVsZXIiOiJDb3NpbmVBbm5lYWxpbmdMUiIsInNjaGVkdWxlcl9UX21heCI6NjAwMCwidHJhaW5fZXhwb3N1cmUiOjAuNjk5OTgxMzk5OTYyOCwidHJhaW5fc2VxdWVuY2VfdG9rZW5zIjoxNywidm9jYWJfc2l6ZSI6MTI5fSwibmFtZSI6InNpbmdsZV9vMl9maWxsYmxhbmtfYTc4YjI0X2Q1MTJsMnI4aDFfUDEyN190cjE2X3NlZWQ0MiIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6Im1haW4g6buY6K6kIiwiQ09OU1QiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6IuS7o+eggeivreS5ieaOqOWvvO+8m+WOn+WAvD1udWxsIiwiRFJPUE9VVCI6Im1haW4g6buY6K6kIiwiRF9NT0RFTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoibWFpbiDpu5jorqQiLCJFQVJMWV9TVE9QX05PX0lNUFJPVkUiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6Im1haW4g6buY6K6kIiwiRVZBTF9JTlRFUlZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkZJUlNUX1RBU0tfV0VJR0hUIjoibWFpbiDpu5jorqQiLCJGUkVTSF9URVNUX1BFUl9FVkFMIjoibWFpbiDpu5jorqQiLCJHUkFEX0FDQ1VNX1NURVBTIjoibWFpbiDpu5jorqQiLCJHUkFEX0NMSVBfTk9STSI6Im1haW4g6buY6K6kIiwiSU5JVF9GUk9NIjoibWFpbiDpu5jorqQiLCJMUiI6Im1haW4g6buY6K6kIiwiTUFYX1RSQUlOX0hPVVJTIjoibWFpbiDpu5jorqQiLCJNSVNTSU5HX1BST0IiOiJtYWluIOm7mOiupCIsIk1JU1NfTEVOIjoibWFpbiDpu5jorqQiLCJNTFBfUkFUSU8iOiJtYWluIOm7mOiupCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoibWFpbiDpu5jorqQiLCJOVU1fVFJBSU5fU0FNUExFUyI6IuS7o+eggeivreS5ieaOqOWvvO+8m+WOn+WAvD1udWxsIiwiTl9IRUFEIjoibWFpbiDpu5jorqQiLCJOX0xBWUVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiT09EX0xFTiI6Im1haW4g6buY6K6kIiwiT1BUX0RJQUdfSU5URVJWQUwiOiJtYWluIOm7mOiupCIsIk9SREVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJBTkRPTV9TRUVEIjoibWFpbiDpu5jorqQiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6Im1haW4g6buY6K6kIiwiUkVTVU1FX0ZST00iOiJtYWluIOm7mOiupCIsIlJVTEVTIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiU1RBVEVfU1BBQ0VfQ0FQIjoibWFpbiDpu5jorqQiLCJUQVNLIjoi5a6e6aqM5Lu75YqhIiwiVEFTS19WQVJJQU5UIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVEVTVF9BTExfUE9TSVRJT05TIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVFJBSU5fTEVOIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiV0VJR0hUX0RFQ0FZIjoibWFpbiDpu5jorqQifSwidGFzayI6InNpbmdsZSJ9LHsiY29uZmlnIjp7IkJBVENIX1NJWkUiOjUxMiwiQ09OU1QiOjAsIkRBVEFfTU9ERSI6ImZ1bGxfc3BsaXQiLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjUxMiwiRUFSTFlfU1RPUF9BQ0NVUkFDWSI6MC45OSwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOjAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRVBPQ0hTIjo2MDAwLCJFVkFMX0lOVEVSVkFMIjoxLCJGSVJTVF9UQVNLX1dFSUdIVCI6MS4wLCJGUkVTSF9URVNUX1BFUl9FVkFMIjpmYWxzZSwiR1JBRF9BQ0NVTV9TVEVQUyI6MSwiR1JBRF9DTElQX05PUk0iOjEuMCwiSU5JVF9GUk9NIjpudWxsLCJMUiI6MC4wMDAzLCJNQVhfVFJBSU5fSE9VUlMiOm51bGwsIk1JU1NJTkdfUFJPQiI6MC4wLCJNSVNTX0xFTiI6MSwiTUxQX1JBVElPIjo4LCJOVU1fTUFTSyI6bnVsbCwiTlVNX1RFU1RfU0FNUExFUyI6MjU2LCJOVU1fVFJBSU5fU0FNUExFUyI6MTEyOTAsIk5fSEVBRCI6MSwiTl9MQVlFUiI6MiwiT09EX0xFTiI6MTYsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJPUkRFUiI6MiwiUCI6MTI3LCJSQU5ET01fU0VFRCI6NDIsIlJFU0hVRkZMRV9FQUNIX0VQT0NIIjpmYWxzZSwiUkVTVU1FX0ZST00iOm51bGwsIlJVTEVTIjpbWzY3LDEwNV1dLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJzaW5nbGUiLCJUQVNLX1ZBUklBTlQiOiJmaWxsX2JsYW5rIiwiVEVTVF9BTExfUE9TSVRJT05TIjp0cnVlLCJUUkFJTl9MRU4iOjE2LCJXRUlHSFRfREVDQVkiOjF9LCJlZmZlY3RpdmUiOnsiYWN0dWFsX251bV9tYXNrIjoiY29sbGF0ZSBsb3NzX21hc2vvvIjkuI3lj4LkuI7vvIkiLCJhY3R1YWxfdGVzdF9zYW1wbGVzIjp7InRvdGFsIjo3NzQyNCwid2luZG93cyI6NDgzOX0sImFjdHVhbF90cmFpbl9zYW1wbGVzIjoxMTI5MCwiYmxvY2tfc2l6ZSI6MzIsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDmlrDlgLwiLCJlZmZlY3RpdmVfZGF0YV9tb2RlIjoiZnVsbF9zcGxpdCIsImV2YWxfYmFzZV9sZW5ndGgiOjE2LCJldmFsX3NlcXVlbmNlX3Rva2VucyI6MTcsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJmaWxsX2JsYW5r77yabG9zc19tYXNrIOeUsSBjb2xsYXRlIOe7meWHuu+8iOS7hei+k+WHuuS9je+8ie+8jE5VTV9NQVNLIOS4jeWPguS4juOAgiIsIlRFU1RfQUxMX1BPU0lUSU9OU++8mua1i+ivlembhumAkOS9jee9ruaemuS4vuaMluepuu+8iOehruWumuaAp+ivhOS8sO+8ie+8jHRlc3Qg5qC35pys5pWwID0g56qX5Y+j5pWwIMOXIOeql+WPo+mVv+W6pu+8m+iuree7g+S+p+S/neaMgeWcqOe6v+maj+acuuaMluepuuOAgiIsImZpbGxfYmxhbmvvvJrmsaHmjZ/mmK/ku7vliqHlhoXnlJ/nmoTvvIjmgbDlpb3kuIDkvY3mm7/mjaLmiJDnqbrnmb0gdG9rZW4gaWQ9UO+8ie+8jOaXoCBjbGVhbi90b3RhbCDkuYvliIbvvIznu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2PvvJvovpPlh7ogPSDmraPnoa7lgLzkuIDkuKogdG9rZW7jgIIiLCLpnZnmgIHmtYvor5XlnKjovr7liLAgRUFSTFlfU1RPUF9BQ0NVUkFDWSDml7blgZzmraLvvJtmcmVzaCB0ZXN0IOWcqOacgOi/kSAxMCDmrKHor4TkvLDlnYflgLzotoXov4fpmIjlgLzml7blgZzmraLjgIJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyDlt7Llup/lvIPvvIzkuI3nlJ/mlYjjgIIiLCLorq3nu4Pnu5PmnZ/kuI3lho3ot5EgZmluYWwgZ2VuZXJhdGlvbiB0ZXN077yIMjAyNi0xMC0wOCDliKDpmaTvvInvvJvnu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2MgKyBldmFsX2NsZWFuX2FjdGlvbi5weeOAguato+W4uOWujOaIkOWQjui3kSBhdHRlbnRpb24g5YiG5p6Q77yb5a2Y5qGj6YCA5Ye656CBIDQyIOi3s+i/h+WQjuWkhOeQhuOAgiJdLCJvcHRpbWl6ZXIiOiJBZGFtVyIsIm91dHB1dF9wYXRoIjoiPG91dHB1dD4vbW9kZWxzLzxleHBlcmltZW50Pi88ZXhwZXJpbWVudD4ucHRoIiwicmVjdXJyZW5jZV9vcmRlciI6MiwicnVsZXMiOls2NywxMDVdLCJzY2hlZHVsZXIiOiJDb3NpbmVBbm5lYWxpbmdMUiIsInNjaGVkdWxlcl9UX21heCI6NjAwMCwidHJhaW5fZXhwb3N1cmUiOjAuNjk5OTgxMzk5OTYyOCwidHJhaW5fc2VxdWVuY2VfdG9rZW5zIjoxNywidm9jYWJfc2l6ZSI6MTI5fSwibmFtZSI6InNpbmdsZV9vMl9maWxsYmxhbmtfYTY3YjEwNV9kNTEybDJyOGgxX1AxMjdfdHIxNl9zZWVkNDIiLCJvcmlnaW5zIjp7IkJBVENIX1NJWkUiOiJtYWluIOm7mOiupCIsIkNPTlNUIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoibWFpbiDpu5jorqQiLCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6Im1haW4g6buY6K6kIiwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjoibWFpbiDpu5jorqQiLCJFUE9DSFMiOiJtYWluIOm7mOiupCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6Im1haW4g6buY6K6kIiwiR1JBRF9BQ0NVTV9TVEVQUyI6Im1haW4g6buY6K6kIiwiR1JBRF9DTElQX05PUk0iOiJtYWluIOm7mOiupCIsIklOSVRfRlJPTSI6Im1haW4g6buY6K6kIiwiTFIiOiJtYWluIOm7mOiupCIsIk1BWF9UUkFJTl9IT1VSUyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoibWFpbiDpu5jorqQiLCJOVU1fTUFTSyI6Im1haW4g6buY6K6kIiwiTlVNX1RFU1RfU0FNUExFUyI6Im1haW4g6buY6K6kIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5fSEVBRCI6Im1haW4g6buY6K6kIiwiTl9MQVlFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9PRF9MRU4iOiJtYWluIOm7mOiupCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSQU5ET01fU0VFRCI6Im1haW4g6buY6K6kIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoibWFpbiDpu5jorqQiLCJSVUxFUyI6IueUqOaIt+mAkOWunumqjOWPguaVsCIsIlNUQVRFX1NQQUNFX0NBUCI6Im1haW4g6buY6K6kIiwiVEFTSyI6IuWunumqjOS7u+WKoSIsIlRBU0tfVkFSSUFOVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlRFU1RfQUxMX1BPU0lUSU9OUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6Im1haW4g6buY6K6kIn0sInRhc2siOiJzaW5nbGUifSx7ImNvbmZpZyI6eyJCQVRDSF9TSVpFIjo1MTIsIkNPTlNUIjowLCJEQVRBX01PREUiOiJmdWxsX3NwbGl0IiwiRFJPUE9VVCI6MC4wLCJEX01PREVMIjo1MTIsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjowLCJFQVJMWV9TVE9QX05PX0lNUFJPVkUiOjMwMDAsIkVQT0NIUyI6NjAwMCwiRVZBTF9JTlRFUlZBTCI6MSwiRklSU1RfVEFTS19XRUlHSFQiOjEuMCwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6ZmFsc2UsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjpudWxsLCJNSVNTSU5HX1BST0IiOjAuMCwiTUlTU19MRU4iOjEsIk1MUF9SQVRJTyI6OCwiTlVNX01BU0siOm51bGwsIk5VTV9URVNUX1NBTVBMRVMiOjI1NiwiTlVNX1RSQUlOX1NBTVBMRVMiOjExMjkwLCJOX0hFQUQiOjEsIk5fTEFZRVIiOjIsIk9PRF9MRU4iOjE2LCJPUFRfRElBR19JTlRFUlZBTCI6MCwiT1JERVIiOjIsIlAiOjEyNywiUkFORE9NX1NFRUQiOjQyLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6ZmFsc2UsIlJFU1VNRV9GUk9NIjpudWxsLCJSVUxFUyI6W1s1NCwxMTNdXSwiU1RBVEVfU1BBQ0VfQ0FQIjpudWxsLCJUQVNLIjoic2luZ2xlIiwiVEFTS19WQVJJQU5UIjoiZmlsbF9ibGFuayIsIlRFU1RfQUxMX1BPU0lUSU9OUyI6dHJ1ZSwiVFJBSU5fTEVOIjoxNiwiV0VJR0hUX0RFQ0FZIjoxfSwiZWZmZWN0aXZlIjp7ImFjdHVhbF9udW1fbWFzayI6ImNvbGxhdGUgbG9zc19tYXNr77yI5LiN5Y+C5LiO77yJIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6eyJ0b3RhbCI6Nzc0MjQsIndpbmRvd3MiOjQ4Mzl9LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MTEyOTAsImJsb2NrX3NpemUiOjMyLCJjb2VmZmljaWVudF9jb252ZW50aW9uIjoi56ys5LiA5Liq57O75pWw5LmY5pyA5paw5YC8IiwiZWZmZWN0aXZlX2RhdGFfbW9kZSI6ImZ1bGxfc3BsaXQiLCJldmFsX2Jhc2VfbGVuZ3RoIjoxNiwiZXZhbF9zZXF1ZW5jZV90b2tlbnMiOjE3LCJpbml0aWFsaXphdGlvbiI6ImZyZXNoIiwibm90ZXMiOlsiZmlsbF9ibGFua++8mmxvc3NfbWFzayDnlLEgY29sbGF0ZSDnu5nlh7rvvIjku4XovpPlh7rkvY3vvInvvIxOVU1fTUFTSyDkuI3lj4LkuI7jgIIiLCJURVNUX0FMTF9QT1NJVElPTlPvvJrmtYvor5Xpm4bpgJDkvY3nva7mnprkuL7mjJbnqbrvvIjnoa7lrprmgKfor4TkvLDvvInvvIx0ZXN0IOagt+acrOaVsCA9IOeql+WPo+aVsCDDlyDnqpflj6Pplb/luqbvvJvorq3nu4Pkvqfkv53mjIHlnKjnur/pmo/mnLrmjJbnqbrjgIIiLCJmaWxsX2JsYW5r77ya5rGh5o2f5piv5Lu75Yqh5YaF55Sf55qE77yI5oGw5aW95LiA5L2N5pu/5o2i5oiQ56m655m9IHRva2VuIGlkPVDvvInvvIzml6AgY2xlYW4vdG90YWwg5LmL5YiG77yM57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNj77yb6L6T5Ye6ID0g5q2j56Gu5YC85LiA5LiqIHRva2Vu44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5LiN5YaN6LeRIGZpbmFsIGdlbmVyYXRpb24gdGVzdO+8iDIwMjYtMTAtMDgg5Yig6Zmk77yJ77yb57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNjICsgZXZhbF9jbGVhbl9hY3Rpb24ucHnjgILmraPluLjlrozmiJDlkI7ot5EgYXR0ZW50aW9uIOWIhuaekO+8m+WtmOaho+mAgOWHuueggSA0MiDot7Pov4flkI7lpITnkIbjgIIiXSwib3B0aW1pemVyIjoiQWRhbVciLCJvdXRwdXRfcGF0aCI6IjxvdXRwdXQ+L21vZGVscy88ZXhwZXJpbWVudD4vPGV4cGVyaW1lbnQ+LnB0aCIsInJlY3VycmVuY2Vfb3JkZXIiOjIsInJ1bGVzIjpbNTQsMTEzXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjowLjY5OTk4MTM5OTk2MjgsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTcsInZvY2FiX3NpemUiOjEyOX0sIm5hbWUiOiJzaW5nbGVfbzJfZmlsbGJsYW5rX2E1NGIxMTNfZDUxMmwycjhoMV9QMTI3X3RyMTZfc2VlZDQyIiwib3JpZ2lucyI6eyJCQVRDSF9TSVpFIjoibWFpbiDpu5jorqQiLCJDT05TVCI6Im1haW4g6buY6K6kIiwiREFUQV9NT0RFIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJEUk9QT1VUIjoibWFpbiDpu5jorqQiLCJEX01PREVMIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9BQ0NVUkFDWSI6Im1haW4g6buY6K6kIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoibWFpbiDpu5jorqQiLCJFVkFMX0lOVEVSVkFMIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRklSU1RfVEFTS19XRUlHSFQiOiJtYWluIOm7mOiupCIsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOiJtYWluIOm7mOiupCIsIkdSQURfQUNDVU1fU1RFUFMiOiJtYWluIOm7mOiupCIsIkdSQURfQ0xJUF9OT1JNIjoibWFpbiDpu5jorqQiLCJJTklUX0ZST00iOiJtYWluIOm7mOiupCIsIkxSIjoibWFpbiDpu5jorqQiLCJNQVhfVFJBSU5fSE9VUlMiOiJtYWluIOm7mOiupCIsIk1JU1NJTkdfUFJPQiI6Im1haW4g6buY6K6kIiwiTUlTU19MRU4iOiJtYWluIOm7mOiupCIsIk1MUF9SQVRJTyI6Im1haW4g6buY6K6kIiwiTlVNX01BU0siOiJtYWluIOm7mOiupCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiJtYWluIOm7mOiupCIsIk5fTEFZRVIiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJPT0RfTEVOIjoibWFpbiDpu5jorqQiLCJPUFRfRElBR19JTlRFUlZBTCI6Im1haW4g6buY6K6kIiwiT1JERVIiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUkFORE9NX1NFRUQiOiJtYWluIOm7mOiupCIsIlJFU0hVRkZMRV9FQUNIX0VQT0NIIjoibWFpbiDpu5jorqQiLCJSRVNVTUVfRlJPTSI6Im1haW4g6buY6K6kIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJURVNUX0FMTF9QT1NJVElPTlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJXRUlHSFRfREVDQVkiOiJtYWluIOm7mOiupCJ9LCJ0YXNrIjoic2luZ2xlIn0seyJjb25maWciOnsiQkFUQ0hfU0laRSI6NTEyLCJDT05TVCI6MCwiREFUQV9NT0RFIjoiZnVsbF9zcGxpdCIsIkRST1BPVVQiOjAuMCwiRF9NT0RFTCI6NTEyLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjowLjk5LCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6MCwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjozMDAwLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOmZhbHNlLCJHUkFEX0FDQ1VNX1NURVBTIjoxLCJHUkFEX0NMSVBfTk9STSI6MS4wLCJJTklUX0ZST00iOm51bGwsIkxSIjowLjAwMDMsIk1BWF9UUkFJTl9IT1VSUyI6bnVsbCwiTUlTU0lOR19QUk9CIjowLjAsIk1JU1NfTEVOIjoxLCJNTFBfUkFUSU8iOjgsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjoxMTI5MCwiTl9IRUFEIjoxLCJOX0xBWUVSIjoyLCJPT0RfTEVOIjoxNiwiT1BUX0RJQUdfSU5URVJWQUwiOjAsIk9SREVSIjoyLCJQIjoxMjcsIlJBTkRPTV9TRUVEIjo0MiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiUlVMRVMiOltbNTAsNzNdXSwiU1RBVEVfU1BBQ0VfQ0FQIjpudWxsLCJUQVNLIjoic2luZ2xlIiwiVEFTS19WQVJJQU5UIjoiZmlsbF9ibGFuayIsIlRFU1RfQUxMX1BPU0lUSU9OUyI6dHJ1ZSwiVFJBSU5fTEVOIjoxNiwiV0VJR0hUX0RFQ0FZIjoxfSwiZWZmZWN0aXZlIjp7ImFjdHVhbF9udW1fbWFzayI6ImNvbGxhdGUgbG9zc19tYXNr77yI5LiN5Y+C5LiO77yJIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6eyJ0b3RhbCI6Nzc0MjQsIndpbmRvd3MiOjQ4Mzl9LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MTEyOTAsImJsb2NrX3NpemUiOjMyLCJjb2VmZmljaWVudF9jb252ZW50aW9uIjoi56ys5LiA5Liq57O75pWw5LmY5pyA5paw5YC8IiwiZWZmZWN0aXZlX2RhdGFfbW9kZSI6ImZ1bGxfc3BsaXQiLCJldmFsX2Jhc2VfbGVuZ3RoIjoxNiwiZXZhbF9zZXF1ZW5jZV90b2tlbnMiOjE3LCJpbml0aWFsaXphdGlvbiI6ImZyZXNoIiwibm90ZXMiOlsiZmlsbF9ibGFua++8mmxvc3NfbWFzayDnlLEgY29sbGF0ZSDnu5nlh7rvvIjku4XovpPlh7rkvY3vvInvvIxOVU1fTUFTSyDkuI3lj4LkuI7jgIIiLCJURVNUX0FMTF9QT1NJVElPTlPvvJrmtYvor5Xpm4bpgJDkvY3nva7mnprkuL7mjJbnqbrvvIjnoa7lrprmgKfor4TkvLDvvInvvIx0ZXN0IOagt+acrOaVsCA9IOeql+WPo+aVsCDDlyDnqpflj6Pplb/luqbvvJvorq3nu4Pkvqfkv53mjIHlnKjnur/pmo/mnLrmjJbnqbrjgIIiLCJmaWxsX2JsYW5r77ya5rGh5o2f5piv5Lu75Yqh5YaF55Sf55qE77yI5oGw5aW95LiA5L2N5pu/5o2i5oiQ56m655m9IHRva2VuIGlkPVDvvInvvIzml6AgY2xlYW4vdG90YWwg5LmL5YiG77yM57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNj77yb6L6T5Ye6ID0g5q2j56Gu5YC85LiA5LiqIHRva2Vu44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5LiN5YaN6LeRIGZpbmFsIGdlbmVyYXRpb24gdGVzdO+8iDIwMjYtMTAtMDgg5Yig6Zmk77yJ77yb57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNjICsgZXZhbF9jbGVhbl9hY3Rpb24ucHnjgILmraPluLjlrozmiJDlkI7ot5EgYXR0ZW50aW9uIOWIhuaekO+8m+WtmOaho+mAgOWHuueggSA0MiDot7Pov4flkI7lpITnkIbjgIIiXSwib3B0aW1pemVyIjoiQWRhbVciLCJvdXRwdXRfcGF0aCI6IjxvdXRwdXQ+L21vZGVscy88ZXhwZXJpbWVudD4vPGV4cGVyaW1lbnQ+LnB0aCIsInJlY3VycmVuY2Vfb3JkZXIiOjIsInJ1bGVzIjpbNTAsNzNdLCJzY2hlZHVsZXIiOiJDb3NpbmVBbm5lYWxpbmdMUiIsInNjaGVkdWxlcl9UX21heCI6NjAwMCwidHJhaW5fZXhwb3N1cmUiOjAuNjk5OTgxMzk5OTYyOCwidHJhaW5fc2VxdWVuY2VfdG9rZW5zIjoxNywidm9jYWJfc2l6ZSI6MTI5fSwibmFtZSI6InNpbmdsZV9vMl9maWxsYmxhbmtfYTUwYjczX2Q1MTJsMnI4aDFfUDEyN190cjE2X3NlZWQ0MiIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6Im1haW4g6buY6K6kIiwiQ09OU1QiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6IuS7o+eggeivreS5ieaOqOWvvO+8m+WOn+WAvD1udWxsIiwiRFJPUE9VVCI6Im1haW4g6buY6K6kIiwiRF9NT0RFTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoibWFpbiDpu5jorqQiLCJFQVJMWV9TVE9QX05PX0lNUFJPVkUiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6Im1haW4g6buY6K6kIiwiRVZBTF9JTlRFUlZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkZJUlNUX1RBU0tfV0VJR0hUIjoibWFpbiDpu5jorqQiLCJGUkVTSF9URVNUX1BFUl9FVkFMIjoibWFpbiDpu5jorqQiLCJHUkFEX0FDQ1VNX1NURVBTIjoibWFpbiDpu5jorqQiLCJHUkFEX0NMSVBfTk9STSI6Im1haW4g6buY6K6kIiwiSU5JVF9GUk9NIjoibWFpbiDpu5jorqQiLCJMUiI6Im1haW4g6buY6K6kIiwiTUFYX1RSQUlOX0hPVVJTIjoibWFpbiDpu5jorqQiLCJNSVNTSU5HX1BST0IiOiJtYWluIOm7mOiupCIsIk1JU1NfTEVOIjoibWFpbiDpu5jorqQiLCJNTFBfUkFUSU8iOiJtYWluIOm7mOiupCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoibWFpbiDpu5jorqQiLCJOVU1fVFJBSU5fU0FNUExFUyI6IuS7o+eggeivreS5ieaOqOWvvO+8m+WOn+WAvD1udWxsIiwiTl9IRUFEIjoibWFpbiDpu5jorqQiLCJOX0xBWUVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiT09EX0xFTiI6Im1haW4g6buY6K6kIiwiT1BUX0RJQUdfSU5URVJWQUwiOiJtYWluIOm7mOiupCIsIk9SREVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJBTkRPTV9TRUVEIjoibWFpbiDpu5jorqQiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6Im1haW4g6buY6K6kIiwiUkVTVU1FX0ZST00iOiJtYWluIOm7mOiupCIsIlJVTEVTIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiU1RBVEVfU1BBQ0VfQ0FQIjoibWFpbiDpu5jorqQiLCJUQVNLIjoi5a6e6aqM5Lu75YqhIiwiVEFTS19WQVJJQU5UIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVEVTVF9BTExfUE9TSVRJT05TIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVFJBSU5fTEVOIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiV0VJR0hUX0RFQ0FZIjoibWFpbiDpu5jorqQifSwidGFzayI6InNpbmdsZSJ9LHsiY29uZmlnIjp7IkJBVENIX1NJWkUiOjUxMiwiQ09OU1QiOjAsIkRBVEFfTU9ERSI6ImZ1bGxfc3BsaXQiLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjUxMiwiRUFSTFlfU1RPUF9BQ0NVUkFDWSI6MC45OSwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOjAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRVBPQ0hTIjo2MDAwLCJFVkFMX0lOVEVSVkFMIjoxLCJGSVJTVF9UQVNLX1dFSUdIVCI6MS4wLCJGUkVTSF9URVNUX1BFUl9FVkFMIjpmYWxzZSwiR1JBRF9BQ0NVTV9TVEVQUyI6MSwiR1JBRF9DTElQX05PUk0iOjEuMCwiSU5JVF9GUk9NIjpudWxsLCJMUiI6MC4wMDAzLCJNQVhfVFJBSU5fSE9VUlMiOm51bGwsIk1JU1NJTkdfUFJPQiI6MC4wLCJNSVNTX0xFTiI6MSwiTUxQX1JBVElPIjo4LCJOVU1fTUFTSyI6bnVsbCwiTlVNX1RFU1RfU0FNUExFUyI6MjU2LCJOVU1fVFJBSU5fU0FNUExFUyI6MTEyOTAsIk5fSEVBRCI6MSwiTl9MQVlFUiI6MiwiT09EX0xFTiI6MTYsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJPUkRFUiI6MiwiUCI6MTI3LCJSQU5ET01fU0VFRCI6NDIsIlJFU0hVRkZMRV9FQUNIX0VQT0NIIjpmYWxzZSwiUkVTVU1FX0ZST00iOm51bGwsIlJVTEVTIjpbWzgzLDg4XV0sIlNUQVRFX1NQQUNFX0NBUCI6bnVsbCwiVEFTSyI6InNpbmdsZSIsIlRBU0tfVkFSSUFOVCI6ImZpbGxfYmxhbmsiLCJURVNUX0FMTF9QT1NJVElPTlMiOnRydWUsIlRSQUlOX0xFTiI6MTYsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiJjb2xsYXRlIGxvc3NfbWFza++8iOS4jeWPguS4ju+8iSIsImFjdHVhbF90ZXN0X3NhbXBsZXMiOnsidG90YWwiOjc3NDI0LCJ3aW5kb3dzIjo0ODM5fSwiYWN0dWFsX3RyYWluX3NhbXBsZXMiOjExMjkwLCJibG9ja19zaXplIjozMiwiY29lZmZpY2llbnRfY29udmVudGlvbiI6IuesrOS4gOS4quezu+aVsOS5mOacgOaWsOWAvCIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJmdWxsX3NwbGl0IiwiZXZhbF9iYXNlX2xlbmd0aCI6MTYsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxNywiaW5pdGlhbGl6YXRpb24iOiJmcmVzaCIsIm5vdGVzIjpbImZpbGxfYmxhbmvvvJpsb3NzX21hc2sg55SxIGNvbGxhdGUg57uZ5Ye677yI5LuF6L6T5Ye65L2N77yJ77yMTlVNX01BU0sg5LiN5Y+C5LiO44CCIiwiVEVTVF9BTExfUE9TSVRJT05T77ya5rWL6K+V6ZuG6YCQ5L2N572u5p6a5Li+5oyW56m677yI56Gu5a6a5oCn6K+E5Lyw77yJ77yMdGVzdCDmoLfmnKzmlbAgPSDnqpflj6PmlbAgw5cg56qX5Y+j6ZW/5bqm77yb6K6t57uD5L6n5L+d5oyB5Zyo57q/6ZqP5py65oyW56m644CCIiwiZmlsbF9ibGFua++8muaxoeaNn+aYr+S7u+WKoeWGheeUn+eahO+8iOaBsOWlveS4gOS9jeabv+aNouaIkOepuueZvSB0b2tlbiBpZD1Q77yJ77yM5pegIGNsZWFuL3RvdGFsIOS5i+WIhu+8jOe7k+aenOWPo+W+hCA9IOiuree7g+aXpeW/lyBiZXN0IGFjY++8m+i+k+WHuiA9IOato+ehruWAvOS4gOS4qiB0b2tlbuOAgiIsIumdmeaAgea1i+ivleWcqOi+vuWIsCBFQVJMWV9TVE9QX0FDQ1VSQUNZIOaXtuWBnOatou+8m2ZyZXNoIHRlc3Qg5Zyo5pyA6L+RIDEwIOasoeivhOS8sOWdh+WAvOi2hei/h+mYiOWAvOaXtuWBnOatouOAgkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIOW3suW6n+W8g++8jOS4jeeUn+aViOOAgiIsIuiuree7g+e7k+adn+S4jeWGjei3kSBmaW5hbCBnZW5lcmF0aW9uIHRlc3TvvIgyMDI2LTEwLTA4IOWIoOmZpO+8ie+8m+e7k+aenOWPo+W+hCA9IOiuree7g+aXpeW/lyBiZXN0IGFjYyArIGV2YWxfY2xlYW5fYWN0aW9uLnB544CC5q2j5bi45a6M5oiQ5ZCO6LeRIGF0dGVudGlvbiDliIbmnpDvvJvlrZjmoaPpgIDlh7rnoIEgNDIg6Lez6L+H5ZCO5aSE55CG44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiI8b3V0cHV0Pi9tb2RlbHMvPGV4cGVyaW1lbnQ+LzxleHBlcmltZW50Pi5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6WzgzLDg4XSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjowLjY5OTk4MTM5OTk2MjgsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTcsInZvY2FiX3NpemUiOjEyOX0sIm5hbWUiOiJzaW5nbGVfbzJfZmlsbGJsYW5rX2E4M2I4OF9kNTEybDJyOGgxX1AxMjdfdHIxNl9zZWVkNDIiLCJvcmlnaW5zIjp7IkJBVENIX1NJWkUiOiJtYWluIOm7mOiupCIsIkNPTlNUIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoibWFpbiDpu5jorqQiLCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6Im1haW4g6buY6K6kIiwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjoibWFpbiDpu5jorqQiLCJFUE9DSFMiOiJtYWluIOm7mOiupCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6Im1haW4g6buY6K6kIiwiR1JBRF9BQ0NVTV9TVEVQUyI6Im1haW4g6buY6K6kIiwiR1JBRF9DTElQX05PUk0iOiJtYWluIOm7mOiupCIsIklOSVRfRlJPTSI6Im1haW4g6buY6K6kIiwiTFIiOiJtYWluIOm7mOiupCIsIk1BWF9UUkFJTl9IT1VSUyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoibWFpbiDpu5jorqQiLCJOVU1fTUFTSyI6Im1haW4g6buY6K6kIiwiTlVNX1RFU1RfU0FNUExFUyI6Im1haW4g6buY6K6kIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5fSEVBRCI6Im1haW4g6buY6K6kIiwiTl9MQVlFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9PRF9MRU4iOiJtYWluIOm7mOiupCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSQU5ET01fU0VFRCI6Im1haW4g6buY6K6kIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoibWFpbiDpu5jorqQiLCJSVUxFUyI6IueUqOaIt+mAkOWunumqjOWPguaVsCIsIlNUQVRFX1NQQUNFX0NBUCI6Im1haW4g6buY6K6kIiwiVEFTSyI6IuWunumqjOS7u+WKoSIsIlRBU0tfVkFSSUFOVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlRFU1RfQUxMX1BPU0lUSU9OUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6Im1haW4g6buY6K6kIn0sInRhc2siOiJzaW5nbGUifV0sImxvY2FsX2dwdV9hbGxvd2xpc3QiOlswLDEsMiwzXSwicGFja2FnaW5nX2hvc3QiOiJ0ZXN0Iiwic291cmNlX2ZpbGVzIjp7InJlcXVpcmVtZW50cy50eHQiOiI5YzFjMDE4M2ZhMWFkYWIxMTdmODJhZGEwMWYwYWEzNGFmYWFkNjMyOWVmYzUyZGE0ZDI4Y2IyNGI1NWIwODc2Iiwic2NyaXB0cy9leHBlcmltZW50X3BhY2thZ2VfcnVubmVyLnB5IjoiMGJhMTQ3ZjhhNzY3NjU4NzkwNWM0MzEzMDkyZjFmOTllMWUyYjQ2ZTllOWZjODc4NzEzMWY4ZGJmYTcyM2YzYiIsInNyYy9hY3RpdmF0aW9uX3BhdGNoLnB5IjoiMWE2YTU5ODAxNTkyODI4N2Q1MjU5ZTI0NzQwN2Q0OTRkMDI5NWRmNjZjYjczN2M3YzU0YzFjNGNiZjhiYzdiMiIsInNyYy9hbmFseXplX2F0dGVudGlvbi5weSI6IjJkMDcyMzE4MzIwYjAwNjU0YWM3OTUwMzY3YTEzOTg0MmI2N2Q4ZjczYWU4YWU3YTA1OTgwMWMwYTE4MWE3MTgiLCJzcmMvYmF0Y2hfcnVuLnB5IjoiMGRmNjdhN2JlNTE3Y2RhYzc2YmIwNTQ3ZjA0OGVkMTM1ODNiNzJkYzcyZmE5NDA1MTM5MjI4ZDczNjY2YTU2OSIsInNyYy9jaGFpbl9ydW4ucHkiOiIwNTk2NWFlMzM1OGY2MWY1MGNhOTk1ZjJjNmU3ZDhhNThjNTAzZWRhZDgyMmZmMmUwN2MzZTZmNjFkOTAwNGUzIiwic3JjL2NvbmZpZy5qc29uIjoiMGMyODkwOThkZjk3MzFhMDllZmZlYzkzZTM3NTliODMzNDVhOWEzZWFjZjZmNGFjNjVhMjI3ZmMwYzEwOGYwMyIsInNyYy9jb3JlLnB5IjoiMTUyMmEzMGJhZGE4NDNmMDA5MDIzYTJhNzA1NzVkNzcxNGEzZTg2NjNmMDEyMTA3N2FiNjA0NjUwMGU3YzBhYiIsInNyYy9jdXJyaWN1bHVtX2NoYWluX050YXNrLnB5IjoiNDdlYmRiYWJlY2IxODYxNDM1Nzk0NDUzYTBjNDQxZDBiMDlmOGMxODFhYzg0MDRiYTE3ZmY2ZWEwZGE3YzU5MyIsInNyYy9jdXJyaWN1bHVtX2NoYWluX2V4dGVuZC5weSI6ImIyNzM1MzU0MjE2MTBiMzQ2MzRlZWIzMDc2OGE1OTE4NWMxODcwZGIxNWMwMDhkYzk4ODk5NTZmYWNkYTNmZmYiLCJzcmMvY3VycmljdWx1bV9jaGFpbl9yZXZlcnNlLnB5IjoiNWMzMTg5MjEwY2U2MTJlNDQ4NzIxOThmNmEyMmUwYjI3ZjVlMjNhMDA4ODA2MjViMTQ1Y2UxYTc4MDZlNmVhNyIsInNyYy9kYXRhc2V0cy5weSI6IjZhOTA0ZDY1ZTRkODhkM2Q5MTAyMzFjNzI3NmQxZWNhZDUwNDIxMGU1OWVmOTVjOWI0YzJkZGFmZDEzMjg2YTEiLCJzcmMvZXJyb3JfYnJlYWtkb3duLnB5IjoiYTVkZjJlNjI0M2IyNjUwZTQ4NGQ0OTQzMDdkOTMwZDNlMTBhYmQ4MGViZWYxMTFiNmU3YjUzMzQyOTliNjBlMSIsInNyYy9ldmlkZW5jZV9zd2VlcC5weSI6IjM1NTAyNTczMWE4NzQ0MjNiZTcxMGVlNGIyYTVmNDRkNDAwODhhYzRlZmIxNGI5YjI0YzBkMjNmYjhiZTFiNDkiLCJzcmMvZXhwZXJpbWVudC5weSI6IjI0MDBjNjRiZjg2MjAzOGQzOWRhNDA1MThiZDJhMDkxMmExOTZiYjYwOTA0MDdmNDg0ZDgyMzM3ZmRmMGQxMzAiLCJzcmMvZml4X2xvZ3MucHkiOiI4MzgzZjljMTdmMmZhZDkzYTRhZjhlMzA0NTY5ZjBjMGVjY2VkYTM3Y2U5NWNhZDM3N2FhYjhiYWYwZDIwOGNhIiwic3JjL2hlYWRfYWJsYXRpb24ucHkiOiJkMDM4NDYxMDBkM2Q5YTk3NjcxYjkyYzUxY2MyNzYxZDY4MjExYzk0ZjAwYzVkNmE1Yzg3YTQxYmQxNjE2NDVmIiwic3JjL21vZGVscy5weSI6ImEyNDJlZTljZDg4MWQ3MDcwZTA5MjE4ZmY2Mjg2MGUyZjZiMDRmYmE2YWEyNzIxYzdkYzMwYWUwZWQ4NWZiOTAiLCJzcmMvbXVsdGlfcG9zX3BhdGNoLnB5IjoiZTU0MzM0MTEzM2JiZjczOTk4ODIwN2UxMDg1Y2FkZWUxOTdhNjZlZWVmNTM4MDYxNzRlNDUwOTZkMTNhY2U5YyIsInNyYy9wcmVwYXJlX3JlcnVuLnB5IjoiN2ZjYWQwNzVhOGE5OTY0YTgzYzc3YTQzZTFmZmEyOTJhMjA2NGM3ZjE1NzZiOGUxMjQ1ZGVmZTkwNWE5N2NmNCIsInNyYy9wcm90b2NvbC5weSI6ImRhZmE1OGNkYmFmYTk0NmRmNWU5ZGRhYjc4YTM0Mjc3NDM1NzVkMmVjNjI5YjUzZGM4YzhjZjU0YjgyMjNlNzIiLCJzcmMvcmVwb3J0X2Zyb250X2JhY2tfYXR0ZW50aW9uLnB5IjoiNWM3M2ZmMjYyZDJkOGQ5ZThkZWY3Y2MwMWMxZjgyOGY5NDMxNTJjZWI4NmYzY2I3YWYzNGI4ZWMzOTIxNWQ2NSIsInNyYy9ydWxlX2ZpdC5weSI6ImFjMmM1MmEwNzY0ZTFmMTliNWQxNGVhOTBiMTg4OWJiNjRlYmU2YTlkZWU5OWRkYzlkMzNkY2E0NDMzNmM5Y2MiLCJzcmMvcnVsZV9nZW5lcmFsaXphdGlvbi5weSI6IjRiMTkyN2Y4NDI1ZWM3Zjg4MGY0N2ZiMzE0NDE0MWIxZmRiNmI3YTM3MTZlZjkwZjc4YTE5NjJjMmRiMzM5ZTEiLCJzcmMvcnVsZXMucHkiOiJkYTZjOWM1OGM4NDhiNDJkZWRlY2M0YmRiM2ZhZmU5YzVhYTEzZDZlOWI4MzFhZDhjNzRmNjhlZDkxNTdjMGRlIiwic3JjL3NhbXBsZV9nZW4ucHkiOiI4MzkzYjMzZjkwYWViNzYwZmY2YTU2YmY4Y2ZhODU4MjM2Y2IyZDEzZWI1NDhhOThlZmQwZjY4NTJiZTI1M2QxIiwic3JjL3NjZW5hcmlvcy5weSI6IjA5NWU4MmVjNWJmNDVmY2E5MjhhMDE4Nzc2MjRmNGRhMDJlYmE4ODU0YTQwMTUwNWMxYmVmNTM1ZjgxMzc3YTMiLCJzcmMvdHJhaW5pbmcucHkiOiJmZGZhZjc1MGEyOWE0MTUxNGNmNmZlMDYzNDhhMDhlMjMwZDYxNmYwMjJjOGE1NDQxMzNlYzdkMzE5NDE4NDY5Iiwic3JjL3ZlcmlmeV9jaXJjbGUucHkiOiJlMzFjNWFkYjY3ZTBiODNiMTliN2U0YjkyNzBkOWZiNWFjOGY1ZWZkMWQxZjNhMWUxYWVlNjFkMjQ1ZjQ0ZmYzIiwic3JjL3ZlcmlmeV9zYW1wbGUucHkiOiI2NzkwYzNmMGVjYTJhOTNhMDNiOTUxODFlZmIzNThkOWEyNWE3NjE5NzFjNGE2MmJjYjQzYjliMDE3YzUxODRlIiwic3JjL3Zpc3VhbGl6ZS5weSI6IjEyYTQ1M2FlOTU3NjE1NTUxZmFjYWI1MTZhYWIyN2NhNWViZWEyZDRkMzY5NzMzZGE4NzlhYzcyM2U3Y2FhMjcifSwidW5yZXNvbHZlZCI6W10sInZlcnNpb24iOjF9 -->
