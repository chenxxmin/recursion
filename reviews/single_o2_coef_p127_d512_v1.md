# 实验参数确认表

审阅版本：bc9479f62a7c08b35343
批次：single_o2_coef_p127_d512；最大并发：2
此阶段只审阅参数，尚未生成可执行实验 JSON 或启动实验。

| 实验 | 任务 | 实际训练样本 | Seed | LR / WD / clip | 初始化 | 每条定时(h) |
| --- | --- | --- | --- | --- | --- | --- |
| single_o2_a5b3_d512l2r8h4_P127_seed17996 | single | 11290 | 17996 | 0.0003 / 1 / 1.0 | fresh | 1 |
| single_o2_a25b56_d512l2r8h4_P127_seed17996 | single | 11290 | 17996 | 0.0003 / 1 / 1.0 | fresh | 1 |
| single_o2_a62b59_d512l2r8h4_P127_seed17996 | single | 11290 | 17996 | 0.0003 / 1 / 1.0 | fresh | 1 |
| single_o2_a67b65_d512l2r8h4_P127_seed17996 | single | 11290 | 17996 | 0.0003 / 1 / 1.0 | fresh | 1 |
| single_o2_a5b3_d512l2r8h4_P127_seed18318 | single | 11290 | 18318 | 0.0003 / 1 / 1.0 | fresh | 1 |
| single_o2_a25b56_d512l2r8h4_P127_seed18318 | single | 11290 | 18318 | 0.0003 / 1 / 1.0 | fresh | 1 |
| single_o2_a62b59_d512l2r8h4_P127_seed18318 | single | 11290 | 18318 | 0.0003 / 1 / 1.0 | fresh | 1 |
| single_o2_a67b65_d512l2r8h4_P127_seed18318 | single | 11290 | 18318 | 0.0003 / 1 / 1.0 | fresh | 1 |

## single_o2_a5b3_d512l2r8h4_P127_seed17996

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | 用户公共参数 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 1000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 1 | 用户公共参数 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | 用户公共参数 |
| NUM_MASK | 1 | 代码语义推导；原值=null |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| POS_PROBE | true | main 默认 |
| RANDOM_SEED | 17996 | 用户逐实验参数 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | 用户公共参数 |
| RULES | [[5,3]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | null | main 默认 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | 1 |
| actual_test_samples | 4839 |
| actual_train_samples | 11290 |
| block_size | 16 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 16 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [5,3] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 1000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 16 |

- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_a25b56_d512l2r8h4_P127_seed17996

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | 用户公共参数 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 1000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 1 | 用户公共参数 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | 用户公共参数 |
| NUM_MASK | 1 | 代码语义推导；原值=null |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| POS_PROBE | true | main 默认 |
| RANDOM_SEED | 17996 | 用户逐实验参数 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | 用户公共参数 |
| RULES | [[25,56]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | null | main 默认 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | 1 |
| actual_test_samples | 4839 |
| actual_train_samples | 11290 |
| block_size | 16 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 16 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [25,56] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 1000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 16 |

- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_a62b59_d512l2r8h4_P127_seed17996

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | 用户公共参数 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 1000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 1 | 用户公共参数 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | 用户公共参数 |
| NUM_MASK | 1 | 代码语义推导；原值=null |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| POS_PROBE | true | main 默认 |
| RANDOM_SEED | 17996 | 用户逐实验参数 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | 用户公共参数 |
| RULES | [[62,59]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | null | main 默认 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | 1 |
| actual_test_samples | 4839 |
| actual_train_samples | 11290 |
| block_size | 16 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 16 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [62,59] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 1000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 16 |

- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_a67b65_d512l2r8h4_P127_seed17996

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | 用户公共参数 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 1000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 1 | 用户公共参数 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | 用户公共参数 |
| NUM_MASK | 1 | 代码语义推导；原值=null |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| POS_PROBE | true | main 默认 |
| RANDOM_SEED | 17996 | 用户逐实验参数 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | 用户公共参数 |
| RULES | [[67,65]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | null | main 默认 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | 1 |
| actual_test_samples | 4839 |
| actual_train_samples | 11290 |
| block_size | 16 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 16 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [67,65] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 1000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 16 |

- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_a5b3_d512l2r8h4_P127_seed18318

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | 用户公共参数 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 1000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 1 | 用户公共参数 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | 用户公共参数 |
| NUM_MASK | 1 | 代码语义推导；原值=null |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| POS_PROBE | true | main 默认 |
| RANDOM_SEED | 18318 | 用户逐实验参数 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | 用户公共参数 |
| RULES | [[5,3]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | null | main 默认 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | 1 |
| actual_test_samples | 4839 |
| actual_train_samples | 11290 |
| block_size | 16 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 16 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [5,3] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 1000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 16 |

- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_a25b56_d512l2r8h4_P127_seed18318

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | 用户公共参数 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 1000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 1 | 用户公共参数 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | 用户公共参数 |
| NUM_MASK | 1 | 代码语义推导；原值=null |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| POS_PROBE | true | main 默认 |
| RANDOM_SEED | 18318 | 用户逐实验参数 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | 用户公共参数 |
| RULES | [[25,56]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | null | main 默认 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | 1 |
| actual_test_samples | 4839 |
| actual_train_samples | 11290 |
| block_size | 16 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 16 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [25,56] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 1000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 16 |

- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_a62b59_d512l2r8h4_P127_seed18318

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | 用户公共参数 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 1000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 1 | 用户公共参数 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | 用户公共参数 |
| NUM_MASK | 1 | 代码语义推导；原值=null |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| POS_PROBE | true | main 默认 |
| RANDOM_SEED | 18318 | 用户逐实验参数 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | 用户公共参数 |
| RULES | [[62,59]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | null | main 默认 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | 1 |
| actual_test_samples | 4839 |
| actual_train_samples | 11290 |
| block_size | 16 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 16 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [62,59] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 1000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 16 |

- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_a67b65_d512l2r8h4_P127_seed18318

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | 用户公共参数 |
| CONST | 0 | main 默认 |
| DATA_MODE | "full_split" | 代码语义推导；原值=null |
| DROPOUT | 0.0 | main 默认 |
| D_MODEL | 512 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 0 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | main 默认 |
| EPOCHS | 1000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | false | main 默认 |
| GRAD_ACCUM_STEPS | 1 | main 默认 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 1 | 用户公共参数 |
| MISSING_PROB | 0.0 | main 默认 |
| MISS_LEN | 1 | main 默认 |
| MLP_RATIO | 8 | 用户公共参数 |
| NUM_MASK | 1 | 代码语义推导；原值=null |
| NUM_TEST_SAMPLES | 256 | main 默认 |
| NUM_TRAIN_SAMPLES | 11290 | 代码语义推导；原值=null |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户公共参数 |
| OOD_LEN | 16 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| ORDER | 2 | 用户公共参数 |
| P | 127 | 用户公共参数 |
| POS_PROBE | true | main 默认 |
| RANDOM_SEED | 18318 | 用户逐实验参数 |
| RESHUFFLE_EACH_EPOCH | false | main 默认 |
| RESUME_FROM | null | 用户公共参数 |
| RULES | [[67,65]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | null | main 默认 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | 1 |
| actual_test_samples | 4839 |
| actual_train_samples | 11290 |
| block_size | 16 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 16 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [67,65] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 1000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 16 |

- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

打包内容：已确认 JSON、定时写入 run.sh、训练代码快照、依赖清单、检查和运行工具。

请确认此版本；修改参数或训练代码后需重新出表。

<!-- PIPELINE_SNAPSHOT eyJhc3NldHMiOnt9LCJiYXRjaF9uYW1lIjoic2luZ2xlX28yX2NvZWZfcDEyN19kNTEyIiwiY29uY3VycmVuY3kiOjIsImV4cGVyaW1lbnRzIjpbeyJjb25maWciOnsiQkFUQ0hfU0laRSI6NTEyLCJDT05TVCI6MCwiREFUQV9NT0RFIjoiZnVsbF9zcGxpdCIsIkRST1BPVVQiOjAuMCwiRF9NT0RFTCI6NTEyLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjowLjk5LCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6MCwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjozMDAwLCJFUE9DSFMiOjEwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOmZhbHNlLCJHUkFEX0FDQ1VNX1NURVBTIjoxLCJHUkFEX0NMSVBfTk9STSI6MS4wLCJJTklUX0ZST00iOm51bGwsIkxSIjowLjAwMDMsIk1BWF9UUkFJTl9IT1VSUyI6MSwiTUlTU0lOR19QUk9CIjowLjAsIk1JU1NfTEVOIjoxLCJNTFBfUkFUSU8iOjgsIk5VTV9NQVNLIjoxLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjoxMTI5MCwiTl9IRUFEIjo0LCJOX0xBWUVSIjoyLCJPT0RfTEVOIjoxNiwiT1BUX0RJQUdfSU5URVJWQUwiOjAsIk9SREVSIjoyLCJQIjoxMjcsIlBPU19QUk9CRSI6dHJ1ZSwiUkFORE9NX1NFRUQiOjE3OTk2LCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6ZmFsc2UsIlJFU1VNRV9GUk9NIjpudWxsLCJSVUxFUyI6W1s1LDNdXSwiU1RBVEVfU1BBQ0VfQ0FQIjpudWxsLCJUQVNLIjoic2luZ2xlIiwiVEFTS19WQVJJQU5UIjpudWxsLCJUUkFJTl9MRU4iOjE2LCJXRUlHSFRfREVDQVkiOjF9LCJlZmZlY3RpdmUiOnsiYWN0dWFsX251bV9tYXNrIjoxLCJhY3R1YWxfdGVzdF9zYW1wbGVzIjo0ODM5LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MTEyOTAsImJsb2NrX3NpemUiOjE2LCJjb2VmZmljaWVudF9jb252ZW50aW9uIjoi56ys5LiA5Liq57O75pWw5LmY5pyA5paw5YC8IiwiZWZmZWN0aXZlX2RhdGFfbW9kZSI6ImZ1bGxfc3BsaXQiLCJldmFsX2Jhc2VfbGVuZ3RoIjoxNiwiZXZhbF9zZXF1ZW5jZV90b2tlbnMiOjE2LCJpbml0aWFsaXphdGlvbiI6ImZyZXNoIiwibm90ZXMiOlsi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5LiN5YaN6LeRIGZpbmFsIGdlbmVyYXRpb24gdGVzdO+8iDIwMjYtMTAtMDgg5Yig6Zmk77yJ77yb57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNjICsgZXZhbF9jbGVhbl9hY3Rpb24ucHnjgILmraPluLjlrozmiJDlkI7ot5EgYXR0ZW50aW9uIOWIhuaekO+8m+WtmOaho+mAgOWHuueggSA0MiDot7Pov4flkI7lpITnkIbjgIIiXSwib3B0aW1pemVyIjoiQWRhbVciLCJvdXRwdXRfcGF0aCI6IjxvdXRwdXQ+L21vZGVscy88ZXhwZXJpbWVudD4vPGV4cGVyaW1lbnQ+LnB0aCIsInJlY3VycmVuY2Vfb3JkZXIiOjIsInJ1bGVzIjpbNSwzXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjEwMDAsInRyYWluX2V4cG9zdXJlIjowLjY5OTk4MTM5OTk2MjgsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTZ9LCJuYW1lIjoic2luZ2xlX28yX2E1YjNfZDUxMmwycjhoNF9QMTI3X3NlZWQxNzk5NiIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkNPTlNUIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRVZBTF9JTlRFUlZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkZJUlNUX1RBU0tfV0VJR0hUIjoibWFpbiDpu5jorqQiLCJGUkVTSF9URVNUX1BFUl9FVkFMIjoibWFpbiDpu5jorqQiLCJHUkFEX0FDQ1VNX1NURVBTIjoibWFpbiDpu5jorqQiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX01BU0siOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQT1NfUFJPQkUiOiJtYWluIOm7mOiupCIsIlJBTkRPTV9TRUVEIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiJtYWluIOm7mOiupCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoic2luZ2xlIn0seyJjb25maWciOnsiQkFUQ0hfU0laRSI6NTEyLCJDT05TVCI6MCwiREFUQV9NT0RFIjoiZnVsbF9zcGxpdCIsIkRST1BPVVQiOjAuMCwiRF9NT0RFTCI6NTEyLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjowLjk5LCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6MCwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjozMDAwLCJFUE9DSFMiOjEwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOmZhbHNlLCJHUkFEX0FDQ1VNX1NURVBTIjoxLCJHUkFEX0NMSVBfTk9STSI6MS4wLCJJTklUX0ZST00iOm51bGwsIkxSIjowLjAwMDMsIk1BWF9UUkFJTl9IT1VSUyI6MSwiTUlTU0lOR19QUk9CIjowLjAsIk1JU1NfTEVOIjoxLCJNTFBfUkFUSU8iOjgsIk5VTV9NQVNLIjoxLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjoxMTI5MCwiTl9IRUFEIjo0LCJOX0xBWUVSIjoyLCJPT0RfTEVOIjoxNiwiT1BUX0RJQUdfSU5URVJWQUwiOjAsIk9SREVSIjoyLCJQIjoxMjcsIlBPU19QUk9CRSI6dHJ1ZSwiUkFORE9NX1NFRUQiOjE3OTk2LCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6ZmFsc2UsIlJFU1VNRV9GUk9NIjpudWxsLCJSVUxFUyI6W1syNSw1Nl1dLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJzaW5nbGUiLCJUQVNLX1ZBUklBTlQiOm51bGwsIlRSQUlOX0xFTiI6MTYsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOjEsImFjdHVhbF90ZXN0X3NhbXBsZXMiOjQ4MzksImFjdHVhbF90cmFpbl9zYW1wbGVzIjoxMTI5MCwiYmxvY2tfc2l6ZSI6MTYsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDmlrDlgLwiLCJlZmZlY3RpdmVfZGF0YV9tb2RlIjoiZnVsbF9zcGxpdCIsImV2YWxfYmFzZV9sZW5ndGgiOjE2LCJldmFsX3NlcXVlbmNlX3Rva2VucyI6MTYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyLmr4/mnaHlrp7pqozni6znq4vorqHml7bvvIzku47orq3nu4Plvqrnjq/lvIDlp4vvvJvmr48gMSBlcG9jaCDnmoTor4TkvLDngrnmo4Dmn6XvvIzliLDml7bkv53lrZjlrozmlbTlrZjmoaPpgIDlh7rvvIzlj6/og73otoXov4flkI3kuYnml7bpmZDvvJvkuI3mlLnlj5ggY29zaW5lIGhvcml6b27jgIIiLCLpnZnmgIHmtYvor5XlnKjovr7liLAgRUFSTFlfU1RPUF9BQ0NVUkFDWSDml7blgZzmraLvvJtmcmVzaCB0ZXN0IOWcqOacgOi/kSAxMCDmrKHor4TkvLDlnYflgLzotoXov4fpmIjlgLzml7blgZzmraLjgIJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyDlt7Llup/lvIPvvIzkuI3nlJ/mlYjjgIIiLCLorq3nu4Pnu5PmnZ/kuI3lho3ot5EgZmluYWwgZ2VuZXJhdGlvbiB0ZXN077yIMjAyNi0xMC0wOCDliKDpmaTvvInvvJvnu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2MgKyBldmFsX2NsZWFuX2FjdGlvbi5weeOAguato+W4uOWujOaIkOWQjui3kSBhdHRlbnRpb24g5YiG5p6Q77yb5a2Y5qGj6YCA5Ye656CBIDQyIOi3s+i/h+WQjuWkhOeQhuOAgiJdLCJvcHRpbWl6ZXIiOiJBZGFtVyIsIm91dHB1dF9wYXRoIjoiPG91dHB1dD4vbW9kZWxzLzxleHBlcmltZW50Pi88ZXhwZXJpbWVudD4ucHRoIiwicmVjdXJyZW5jZV9vcmRlciI6MiwicnVsZXMiOlsyNSw1Nl0sInNjaGVkdWxlciI6IkNvc2luZUFubmVhbGluZ0xSIiwic2NoZWR1bGVyX1RfbWF4IjoxMDAwLCJ0cmFpbl9leHBvc3VyZSI6MC42OTk5ODEzOTk5NjI4LCJ0cmFpbl9zZXF1ZW5jZV90b2tlbnMiOjE2fSwibmFtZSI6InNpbmdsZV9vMl9hMjViNTZfZDUxMmwycjhoNF9QMTI3X3NlZWQxNzk5NiIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkNPTlNUIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRVZBTF9JTlRFUlZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkZJUlNUX1RBU0tfV0VJR0hUIjoibWFpbiDpu5jorqQiLCJGUkVTSF9URVNUX1BFUl9FVkFMIjoibWFpbiDpu5jorqQiLCJHUkFEX0FDQ1VNX1NURVBTIjoibWFpbiDpu5jorqQiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX01BU0siOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQT1NfUFJPQkUiOiJtYWluIOm7mOiupCIsIlJBTkRPTV9TRUVEIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiJtYWluIOm7mOiupCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoic2luZ2xlIn0seyJjb25maWciOnsiQkFUQ0hfU0laRSI6NTEyLCJDT05TVCI6MCwiREFUQV9NT0RFIjoiZnVsbF9zcGxpdCIsIkRST1BPVVQiOjAuMCwiRF9NT0RFTCI6NTEyLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjowLjk5LCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6MCwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjozMDAwLCJFUE9DSFMiOjEwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOmZhbHNlLCJHUkFEX0FDQ1VNX1NURVBTIjoxLCJHUkFEX0NMSVBfTk9STSI6MS4wLCJJTklUX0ZST00iOm51bGwsIkxSIjowLjAwMDMsIk1BWF9UUkFJTl9IT1VSUyI6MSwiTUlTU0lOR19QUk9CIjowLjAsIk1JU1NfTEVOIjoxLCJNTFBfUkFUSU8iOjgsIk5VTV9NQVNLIjoxLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjoxMTI5MCwiTl9IRUFEIjo0LCJOX0xBWUVSIjoyLCJPT0RfTEVOIjoxNiwiT1BUX0RJQUdfSU5URVJWQUwiOjAsIk9SREVSIjoyLCJQIjoxMjcsIlBPU19QUk9CRSI6dHJ1ZSwiUkFORE9NX1NFRUQiOjE3OTk2LCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6ZmFsc2UsIlJFU1VNRV9GUk9NIjpudWxsLCJSVUxFUyI6W1s2Miw1OV1dLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJzaW5nbGUiLCJUQVNLX1ZBUklBTlQiOm51bGwsIlRSQUlOX0xFTiI6MTYsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOjEsImFjdHVhbF90ZXN0X3NhbXBsZXMiOjQ4MzksImFjdHVhbF90cmFpbl9zYW1wbGVzIjoxMTI5MCwiYmxvY2tfc2l6ZSI6MTYsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDmlrDlgLwiLCJlZmZlY3RpdmVfZGF0YV9tb2RlIjoiZnVsbF9zcGxpdCIsImV2YWxfYmFzZV9sZW5ndGgiOjE2LCJldmFsX3NlcXVlbmNlX3Rva2VucyI6MTYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyLmr4/mnaHlrp7pqozni6znq4vorqHml7bvvIzku47orq3nu4Plvqrnjq/lvIDlp4vvvJvmr48gMSBlcG9jaCDnmoTor4TkvLDngrnmo4Dmn6XvvIzliLDml7bkv53lrZjlrozmlbTlrZjmoaPpgIDlh7rvvIzlj6/og73otoXov4flkI3kuYnml7bpmZDvvJvkuI3mlLnlj5ggY29zaW5lIGhvcml6b27jgIIiLCLpnZnmgIHmtYvor5XlnKjovr7liLAgRUFSTFlfU1RPUF9BQ0NVUkFDWSDml7blgZzmraLvvJtmcmVzaCB0ZXN0IOWcqOacgOi/kSAxMCDmrKHor4TkvLDlnYflgLzotoXov4fpmIjlgLzml7blgZzmraLjgIJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyDlt7Llup/lvIPvvIzkuI3nlJ/mlYjjgIIiLCLorq3nu4Pnu5PmnZ/kuI3lho3ot5EgZmluYWwgZ2VuZXJhdGlvbiB0ZXN077yIMjAyNi0xMC0wOCDliKDpmaTvvInvvJvnu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2MgKyBldmFsX2NsZWFuX2FjdGlvbi5weeOAguato+W4uOWujOaIkOWQjui3kSBhdHRlbnRpb24g5YiG5p6Q77yb5a2Y5qGj6YCA5Ye656CBIDQyIOi3s+i/h+WQjuWkhOeQhuOAgiJdLCJvcHRpbWl6ZXIiOiJBZGFtVyIsIm91dHB1dF9wYXRoIjoiPG91dHB1dD4vbW9kZWxzLzxleHBlcmltZW50Pi88ZXhwZXJpbWVudD4ucHRoIiwicmVjdXJyZW5jZV9vcmRlciI6MiwicnVsZXMiOls2Miw1OV0sInNjaGVkdWxlciI6IkNvc2luZUFubmVhbGluZ0xSIiwic2NoZWR1bGVyX1RfbWF4IjoxMDAwLCJ0cmFpbl9leHBvc3VyZSI6MC42OTk5ODEzOTk5NjI4LCJ0cmFpbl9zZXF1ZW5jZV90b2tlbnMiOjE2fSwibmFtZSI6InNpbmdsZV9vMl9hNjJiNTlfZDUxMmwycjhoNF9QMTI3X3NlZWQxNzk5NiIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkNPTlNUIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRVZBTF9JTlRFUlZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkZJUlNUX1RBU0tfV0VJR0hUIjoibWFpbiDpu5jorqQiLCJGUkVTSF9URVNUX1BFUl9FVkFMIjoibWFpbiDpu5jorqQiLCJHUkFEX0FDQ1VNX1NURVBTIjoibWFpbiDpu5jorqQiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX01BU0siOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQT1NfUFJPQkUiOiJtYWluIOm7mOiupCIsIlJBTkRPTV9TRUVEIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiJtYWluIOm7mOiupCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoic2luZ2xlIn0seyJjb25maWciOnsiQkFUQ0hfU0laRSI6NTEyLCJDT05TVCI6MCwiREFUQV9NT0RFIjoiZnVsbF9zcGxpdCIsIkRST1BPVVQiOjAuMCwiRF9NT0RFTCI6NTEyLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjowLjk5LCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6MCwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjozMDAwLCJFUE9DSFMiOjEwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOmZhbHNlLCJHUkFEX0FDQ1VNX1NURVBTIjoxLCJHUkFEX0NMSVBfTk9STSI6MS4wLCJJTklUX0ZST00iOm51bGwsIkxSIjowLjAwMDMsIk1BWF9UUkFJTl9IT1VSUyI6MSwiTUlTU0lOR19QUk9CIjowLjAsIk1JU1NfTEVOIjoxLCJNTFBfUkFUSU8iOjgsIk5VTV9NQVNLIjoxLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjoxMTI5MCwiTl9IRUFEIjo0LCJOX0xBWUVSIjoyLCJPT0RfTEVOIjoxNiwiT1BUX0RJQUdfSU5URVJWQUwiOjAsIk9SREVSIjoyLCJQIjoxMjcsIlBPU19QUk9CRSI6dHJ1ZSwiUkFORE9NX1NFRUQiOjE3OTk2LCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6ZmFsc2UsIlJFU1VNRV9GUk9NIjpudWxsLCJSVUxFUyI6W1s2Nyw2NV1dLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJzaW5nbGUiLCJUQVNLX1ZBUklBTlQiOm51bGwsIlRSQUlOX0xFTiI6MTYsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOjEsImFjdHVhbF90ZXN0X3NhbXBsZXMiOjQ4MzksImFjdHVhbF90cmFpbl9zYW1wbGVzIjoxMTI5MCwiYmxvY2tfc2l6ZSI6MTYsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDmlrDlgLwiLCJlZmZlY3RpdmVfZGF0YV9tb2RlIjoiZnVsbF9zcGxpdCIsImV2YWxfYmFzZV9sZW5ndGgiOjE2LCJldmFsX3NlcXVlbmNlX3Rva2VucyI6MTYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyLmr4/mnaHlrp7pqozni6znq4vorqHml7bvvIzku47orq3nu4Plvqrnjq/lvIDlp4vvvJvmr48gMSBlcG9jaCDnmoTor4TkvLDngrnmo4Dmn6XvvIzliLDml7bkv53lrZjlrozmlbTlrZjmoaPpgIDlh7rvvIzlj6/og73otoXov4flkI3kuYnml7bpmZDvvJvkuI3mlLnlj5ggY29zaW5lIGhvcml6b27jgIIiLCLpnZnmgIHmtYvor5XlnKjovr7liLAgRUFSTFlfU1RPUF9BQ0NVUkFDWSDml7blgZzmraLvvJtmcmVzaCB0ZXN0IOWcqOacgOi/kSAxMCDmrKHor4TkvLDlnYflgLzotoXov4fpmIjlgLzml7blgZzmraLjgIJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyDlt7Llup/lvIPvvIzkuI3nlJ/mlYjjgIIiLCLorq3nu4Pnu5PmnZ/kuI3lho3ot5EgZmluYWwgZ2VuZXJhdGlvbiB0ZXN077yIMjAyNi0xMC0wOCDliKDpmaTvvInvvJvnu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2MgKyBldmFsX2NsZWFuX2FjdGlvbi5weeOAguato+W4uOWujOaIkOWQjui3kSBhdHRlbnRpb24g5YiG5p6Q77yb5a2Y5qGj6YCA5Ye656CBIDQyIOi3s+i/h+WQjuWkhOeQhuOAgiJdLCJvcHRpbWl6ZXIiOiJBZGFtVyIsIm91dHB1dF9wYXRoIjoiPG91dHB1dD4vbW9kZWxzLzxleHBlcmltZW50Pi88ZXhwZXJpbWVudD4ucHRoIiwicmVjdXJyZW5jZV9vcmRlciI6MiwicnVsZXMiOls2Nyw2NV0sInNjaGVkdWxlciI6IkNvc2luZUFubmVhbGluZ0xSIiwic2NoZWR1bGVyX1RfbWF4IjoxMDAwLCJ0cmFpbl9leHBvc3VyZSI6MC42OTk5ODEzOTk5NjI4LCJ0cmFpbl9zZXF1ZW5jZV90b2tlbnMiOjE2fSwibmFtZSI6InNpbmdsZV9vMl9hNjdiNjVfZDUxMmwycjhoNF9QMTI3X3NlZWQxNzk5NiIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkNPTlNUIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRVZBTF9JTlRFUlZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkZJUlNUX1RBU0tfV0VJR0hUIjoibWFpbiDpu5jorqQiLCJGUkVTSF9URVNUX1BFUl9FVkFMIjoibWFpbiDpu5jorqQiLCJHUkFEX0FDQ1VNX1NURVBTIjoibWFpbiDpu5jorqQiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX01BU0siOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQT1NfUFJPQkUiOiJtYWluIOm7mOiupCIsIlJBTkRPTV9TRUVEIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiJtYWluIOm7mOiupCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoic2luZ2xlIn0seyJjb25maWciOnsiQkFUQ0hfU0laRSI6NTEyLCJDT05TVCI6MCwiREFUQV9NT0RFIjoiZnVsbF9zcGxpdCIsIkRST1BPVVQiOjAuMCwiRF9NT0RFTCI6NTEyLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjowLjk5LCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6MCwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjozMDAwLCJFUE9DSFMiOjEwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOmZhbHNlLCJHUkFEX0FDQ1VNX1NURVBTIjoxLCJHUkFEX0NMSVBfTk9STSI6MS4wLCJJTklUX0ZST00iOm51bGwsIkxSIjowLjAwMDMsIk1BWF9UUkFJTl9IT1VSUyI6MSwiTUlTU0lOR19QUk9CIjowLjAsIk1JU1NfTEVOIjoxLCJNTFBfUkFUSU8iOjgsIk5VTV9NQVNLIjoxLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjoxMTI5MCwiTl9IRUFEIjo0LCJOX0xBWUVSIjoyLCJPT0RfTEVOIjoxNiwiT1BUX0RJQUdfSU5URVJWQUwiOjAsIk9SREVSIjoyLCJQIjoxMjcsIlBPU19QUk9CRSI6dHJ1ZSwiUkFORE9NX1NFRUQiOjE4MzE4LCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6ZmFsc2UsIlJFU1VNRV9GUk9NIjpudWxsLCJSVUxFUyI6W1s1LDNdXSwiU1RBVEVfU1BBQ0VfQ0FQIjpudWxsLCJUQVNLIjoic2luZ2xlIiwiVEFTS19WQVJJQU5UIjpudWxsLCJUUkFJTl9MRU4iOjE2LCJXRUlHSFRfREVDQVkiOjF9LCJlZmZlY3RpdmUiOnsiYWN0dWFsX251bV9tYXNrIjoxLCJhY3R1YWxfdGVzdF9zYW1wbGVzIjo0ODM5LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MTEyOTAsImJsb2NrX3NpemUiOjE2LCJjb2VmZmljaWVudF9jb252ZW50aW9uIjoi56ys5LiA5Liq57O75pWw5LmY5pyA5paw5YC8IiwiZWZmZWN0aXZlX2RhdGFfbW9kZSI6ImZ1bGxfc3BsaXQiLCJldmFsX2Jhc2VfbGVuZ3RoIjoxNiwiZXZhbF9zZXF1ZW5jZV90b2tlbnMiOjE2LCJpbml0aWFsaXphdGlvbiI6ImZyZXNoIiwibm90ZXMiOlsi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5LiN5YaN6LeRIGZpbmFsIGdlbmVyYXRpb24gdGVzdO+8iDIwMjYtMTAtMDgg5Yig6Zmk77yJ77yb57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNjICsgZXZhbF9jbGVhbl9hY3Rpb24ucHnjgILmraPluLjlrozmiJDlkI7ot5EgYXR0ZW50aW9uIOWIhuaekO+8m+WtmOaho+mAgOWHuueggSA0MiDot7Pov4flkI7lpITnkIbjgIIiXSwib3B0aW1pemVyIjoiQWRhbVciLCJvdXRwdXRfcGF0aCI6IjxvdXRwdXQ+L21vZGVscy88ZXhwZXJpbWVudD4vPGV4cGVyaW1lbnQ+LnB0aCIsInJlY3VycmVuY2Vfb3JkZXIiOjIsInJ1bGVzIjpbNSwzXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjEwMDAsInRyYWluX2V4cG9zdXJlIjowLjY5OTk4MTM5OTk2MjgsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTZ9LCJuYW1lIjoic2luZ2xlX28yX2E1YjNfZDUxMmwycjhoNF9QMTI3X3NlZWQxODMxOCIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkNPTlNUIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRVZBTF9JTlRFUlZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkZJUlNUX1RBU0tfV0VJR0hUIjoibWFpbiDpu5jorqQiLCJGUkVTSF9URVNUX1BFUl9FVkFMIjoibWFpbiDpu5jorqQiLCJHUkFEX0FDQ1VNX1NURVBTIjoibWFpbiDpu5jorqQiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX01BU0siOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQT1NfUFJPQkUiOiJtYWluIOm7mOiupCIsIlJBTkRPTV9TRUVEIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiJtYWluIOm7mOiupCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoic2luZ2xlIn0seyJjb25maWciOnsiQkFUQ0hfU0laRSI6NTEyLCJDT05TVCI6MCwiREFUQV9NT0RFIjoiZnVsbF9zcGxpdCIsIkRST1BPVVQiOjAuMCwiRF9NT0RFTCI6NTEyLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjowLjk5LCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6MCwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjozMDAwLCJFUE9DSFMiOjEwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOmZhbHNlLCJHUkFEX0FDQ1VNX1NURVBTIjoxLCJHUkFEX0NMSVBfTk9STSI6MS4wLCJJTklUX0ZST00iOm51bGwsIkxSIjowLjAwMDMsIk1BWF9UUkFJTl9IT1VSUyI6MSwiTUlTU0lOR19QUk9CIjowLjAsIk1JU1NfTEVOIjoxLCJNTFBfUkFUSU8iOjgsIk5VTV9NQVNLIjoxLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjoxMTI5MCwiTl9IRUFEIjo0LCJOX0xBWUVSIjoyLCJPT0RfTEVOIjoxNiwiT1BUX0RJQUdfSU5URVJWQUwiOjAsIk9SREVSIjoyLCJQIjoxMjcsIlBPU19QUk9CRSI6dHJ1ZSwiUkFORE9NX1NFRUQiOjE4MzE4LCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6ZmFsc2UsIlJFU1VNRV9GUk9NIjpudWxsLCJSVUxFUyI6W1syNSw1Nl1dLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJzaW5nbGUiLCJUQVNLX1ZBUklBTlQiOm51bGwsIlRSQUlOX0xFTiI6MTYsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOjEsImFjdHVhbF90ZXN0X3NhbXBsZXMiOjQ4MzksImFjdHVhbF90cmFpbl9zYW1wbGVzIjoxMTI5MCwiYmxvY2tfc2l6ZSI6MTYsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDmlrDlgLwiLCJlZmZlY3RpdmVfZGF0YV9tb2RlIjoiZnVsbF9zcGxpdCIsImV2YWxfYmFzZV9sZW5ndGgiOjE2LCJldmFsX3NlcXVlbmNlX3Rva2VucyI6MTYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyLmr4/mnaHlrp7pqozni6znq4vorqHml7bvvIzku47orq3nu4Plvqrnjq/lvIDlp4vvvJvmr48gMSBlcG9jaCDnmoTor4TkvLDngrnmo4Dmn6XvvIzliLDml7bkv53lrZjlrozmlbTlrZjmoaPpgIDlh7rvvIzlj6/og73otoXov4flkI3kuYnml7bpmZDvvJvkuI3mlLnlj5ggY29zaW5lIGhvcml6b27jgIIiLCLpnZnmgIHmtYvor5XlnKjovr7liLAgRUFSTFlfU1RPUF9BQ0NVUkFDWSDml7blgZzmraLvvJtmcmVzaCB0ZXN0IOWcqOacgOi/kSAxMCDmrKHor4TkvLDlnYflgLzotoXov4fpmIjlgLzml7blgZzmraLjgIJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyDlt7Llup/lvIPvvIzkuI3nlJ/mlYjjgIIiLCLorq3nu4Pnu5PmnZ/kuI3lho3ot5EgZmluYWwgZ2VuZXJhdGlvbiB0ZXN077yIMjAyNi0xMC0wOCDliKDpmaTvvInvvJvnu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2MgKyBldmFsX2NsZWFuX2FjdGlvbi5weeOAguato+W4uOWujOaIkOWQjui3kSBhdHRlbnRpb24g5YiG5p6Q77yb5a2Y5qGj6YCA5Ye656CBIDQyIOi3s+i/h+WQjuWkhOeQhuOAgiJdLCJvcHRpbWl6ZXIiOiJBZGFtVyIsIm91dHB1dF9wYXRoIjoiPG91dHB1dD4vbW9kZWxzLzxleHBlcmltZW50Pi88ZXhwZXJpbWVudD4ucHRoIiwicmVjdXJyZW5jZV9vcmRlciI6MiwicnVsZXMiOlsyNSw1Nl0sInNjaGVkdWxlciI6IkNvc2luZUFubmVhbGluZ0xSIiwic2NoZWR1bGVyX1RfbWF4IjoxMDAwLCJ0cmFpbl9leHBvc3VyZSI6MC42OTk5ODEzOTk5NjI4LCJ0cmFpbl9zZXF1ZW5jZV90b2tlbnMiOjE2fSwibmFtZSI6InNpbmdsZV9vMl9hMjViNTZfZDUxMmwycjhoNF9QMTI3X3NlZWQxODMxOCIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkNPTlNUIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRVZBTF9JTlRFUlZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkZJUlNUX1RBU0tfV0VJR0hUIjoibWFpbiDpu5jorqQiLCJGUkVTSF9URVNUX1BFUl9FVkFMIjoibWFpbiDpu5jorqQiLCJHUkFEX0FDQ1VNX1NURVBTIjoibWFpbiDpu5jorqQiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX01BU0siOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQT1NfUFJPQkUiOiJtYWluIOm7mOiupCIsIlJBTkRPTV9TRUVEIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiJtYWluIOm7mOiupCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoic2luZ2xlIn0seyJjb25maWciOnsiQkFUQ0hfU0laRSI6NTEyLCJDT05TVCI6MCwiREFUQV9NT0RFIjoiZnVsbF9zcGxpdCIsIkRST1BPVVQiOjAuMCwiRF9NT0RFTCI6NTEyLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjowLjk5LCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6MCwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjozMDAwLCJFUE9DSFMiOjEwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOmZhbHNlLCJHUkFEX0FDQ1VNX1NURVBTIjoxLCJHUkFEX0NMSVBfTk9STSI6MS4wLCJJTklUX0ZST00iOm51bGwsIkxSIjowLjAwMDMsIk1BWF9UUkFJTl9IT1VSUyI6MSwiTUlTU0lOR19QUk9CIjowLjAsIk1JU1NfTEVOIjoxLCJNTFBfUkFUSU8iOjgsIk5VTV9NQVNLIjoxLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjoxMTI5MCwiTl9IRUFEIjo0LCJOX0xBWUVSIjoyLCJPT0RfTEVOIjoxNiwiT1BUX0RJQUdfSU5URVJWQUwiOjAsIk9SREVSIjoyLCJQIjoxMjcsIlBPU19QUk9CRSI6dHJ1ZSwiUkFORE9NX1NFRUQiOjE4MzE4LCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6ZmFsc2UsIlJFU1VNRV9GUk9NIjpudWxsLCJSVUxFUyI6W1s2Miw1OV1dLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJzaW5nbGUiLCJUQVNLX1ZBUklBTlQiOm51bGwsIlRSQUlOX0xFTiI6MTYsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOjEsImFjdHVhbF90ZXN0X3NhbXBsZXMiOjQ4MzksImFjdHVhbF90cmFpbl9zYW1wbGVzIjoxMTI5MCwiYmxvY2tfc2l6ZSI6MTYsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDmlrDlgLwiLCJlZmZlY3RpdmVfZGF0YV9tb2RlIjoiZnVsbF9zcGxpdCIsImV2YWxfYmFzZV9sZW5ndGgiOjE2LCJldmFsX3NlcXVlbmNlX3Rva2VucyI6MTYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyLmr4/mnaHlrp7pqozni6znq4vorqHml7bvvIzku47orq3nu4Plvqrnjq/lvIDlp4vvvJvmr48gMSBlcG9jaCDnmoTor4TkvLDngrnmo4Dmn6XvvIzliLDml7bkv53lrZjlrozmlbTlrZjmoaPpgIDlh7rvvIzlj6/og73otoXov4flkI3kuYnml7bpmZDvvJvkuI3mlLnlj5ggY29zaW5lIGhvcml6b27jgIIiLCLpnZnmgIHmtYvor5XlnKjovr7liLAgRUFSTFlfU1RPUF9BQ0NVUkFDWSDml7blgZzmraLvvJtmcmVzaCB0ZXN0IOWcqOacgOi/kSAxMCDmrKHor4TkvLDlnYflgLzotoXov4fpmIjlgLzml7blgZzmraLjgIJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyDlt7Llup/lvIPvvIzkuI3nlJ/mlYjjgIIiLCLorq3nu4Pnu5PmnZ/kuI3lho3ot5EgZmluYWwgZ2VuZXJhdGlvbiB0ZXN077yIMjAyNi0xMC0wOCDliKDpmaTvvInvvJvnu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2MgKyBldmFsX2NsZWFuX2FjdGlvbi5weeOAguato+W4uOWujOaIkOWQjui3kSBhdHRlbnRpb24g5YiG5p6Q77yb5a2Y5qGj6YCA5Ye656CBIDQyIOi3s+i/h+WQjuWkhOeQhuOAgiJdLCJvcHRpbWl6ZXIiOiJBZGFtVyIsIm91dHB1dF9wYXRoIjoiPG91dHB1dD4vbW9kZWxzLzxleHBlcmltZW50Pi88ZXhwZXJpbWVudD4ucHRoIiwicmVjdXJyZW5jZV9vcmRlciI6MiwicnVsZXMiOls2Miw1OV0sInNjaGVkdWxlciI6IkNvc2luZUFubmVhbGluZ0xSIiwic2NoZWR1bGVyX1RfbWF4IjoxMDAwLCJ0cmFpbl9leHBvc3VyZSI6MC42OTk5ODEzOTk5NjI4LCJ0cmFpbl9zZXF1ZW5jZV90b2tlbnMiOjE2fSwibmFtZSI6InNpbmdsZV9vMl9hNjJiNTlfZDUxMmwycjhoNF9QMTI3X3NlZWQxODMxOCIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkNPTlNUIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRVZBTF9JTlRFUlZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkZJUlNUX1RBU0tfV0VJR0hUIjoibWFpbiDpu5jorqQiLCJGUkVTSF9URVNUX1BFUl9FVkFMIjoibWFpbiDpu5jorqQiLCJHUkFEX0FDQ1VNX1NURVBTIjoibWFpbiDpu5jorqQiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX01BU0siOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQT1NfUFJPQkUiOiJtYWluIOm7mOiupCIsIlJBTkRPTV9TRUVEIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiJtYWluIOm7mOiupCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoic2luZ2xlIn0seyJjb25maWciOnsiQkFUQ0hfU0laRSI6NTEyLCJDT05TVCI6MCwiREFUQV9NT0RFIjoiZnVsbF9zcGxpdCIsIkRST1BPVVQiOjAuMCwiRF9NT0RFTCI6NTEyLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjowLjk5LCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6MCwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjozMDAwLCJFUE9DSFMiOjEwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOmZhbHNlLCJHUkFEX0FDQ1VNX1NURVBTIjoxLCJHUkFEX0NMSVBfTk9STSI6MS4wLCJJTklUX0ZST00iOm51bGwsIkxSIjowLjAwMDMsIk1BWF9UUkFJTl9IT1VSUyI6MSwiTUlTU0lOR19QUk9CIjowLjAsIk1JU1NfTEVOIjoxLCJNTFBfUkFUSU8iOjgsIk5VTV9NQVNLIjoxLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjoxMTI5MCwiTl9IRUFEIjo0LCJOX0xBWUVSIjoyLCJPT0RfTEVOIjoxNiwiT1BUX0RJQUdfSU5URVJWQUwiOjAsIk9SREVSIjoyLCJQIjoxMjcsIlBPU19QUk9CRSI6dHJ1ZSwiUkFORE9NX1NFRUQiOjE4MzE4LCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6ZmFsc2UsIlJFU1VNRV9GUk9NIjpudWxsLCJSVUxFUyI6W1s2Nyw2NV1dLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJzaW5nbGUiLCJUQVNLX1ZBUklBTlQiOm51bGwsIlRSQUlOX0xFTiI6MTYsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOjEsImFjdHVhbF90ZXN0X3NhbXBsZXMiOjQ4MzksImFjdHVhbF90cmFpbl9zYW1wbGVzIjoxMTI5MCwiYmxvY2tfc2l6ZSI6MTYsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDmlrDlgLwiLCJlZmZlY3RpdmVfZGF0YV9tb2RlIjoiZnVsbF9zcGxpdCIsImV2YWxfYmFzZV9sZW5ndGgiOjE2LCJldmFsX3NlcXVlbmNlX3Rva2VucyI6MTYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyLmr4/mnaHlrp7pqozni6znq4vorqHml7bvvIzku47orq3nu4Plvqrnjq/lvIDlp4vvvJvmr48gMSBlcG9jaCDnmoTor4TkvLDngrnmo4Dmn6XvvIzliLDml7bkv53lrZjlrozmlbTlrZjmoaPpgIDlh7rvvIzlj6/og73otoXov4flkI3kuYnml7bpmZDvvJvkuI3mlLnlj5ggY29zaW5lIGhvcml6b27jgIIiLCLpnZnmgIHmtYvor5XlnKjovr7liLAgRUFSTFlfU1RPUF9BQ0NVUkFDWSDml7blgZzmraLvvJtmcmVzaCB0ZXN0IOWcqOacgOi/kSAxMCDmrKHor4TkvLDlnYflgLzotoXov4fpmIjlgLzml7blgZzmraLjgIJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyDlt7Llup/lvIPvvIzkuI3nlJ/mlYjjgIIiLCLorq3nu4Pnu5PmnZ/kuI3lho3ot5EgZmluYWwgZ2VuZXJhdGlvbiB0ZXN077yIMjAyNi0xMC0wOCDliKDpmaTvvInvvJvnu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2MgKyBldmFsX2NsZWFuX2FjdGlvbi5weeOAguato+W4uOWujOaIkOWQjui3kSBhdHRlbnRpb24g5YiG5p6Q77yb5a2Y5qGj6YCA5Ye656CBIDQyIOi3s+i/h+WQjuWkhOeQhuOAgiJdLCJvcHRpbWl6ZXIiOiJBZGFtVyIsIm91dHB1dF9wYXRoIjoiPG91dHB1dD4vbW9kZWxzLzxleHBlcmltZW50Pi88ZXhwZXJpbWVudD4ucHRoIiwicmVjdXJyZW5jZV9vcmRlciI6MiwicnVsZXMiOls2Nyw2NV0sInNjaGVkdWxlciI6IkNvc2luZUFubmVhbGluZ0xSIiwic2NoZWR1bGVyX1RfbWF4IjoxMDAwLCJ0cmFpbl9leHBvc3VyZSI6MC42OTk5ODEzOTk5NjI4LCJ0cmFpbl9zZXF1ZW5jZV90b2tlbnMiOjE2fSwibmFtZSI6InNpbmdsZV9vMl9hNjdiNjVfZDUxMmwycjhoNF9QMTI3X3NlZWQxODMxOCIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkNPTlNUIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRVZBTF9JTlRFUlZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkZJUlNUX1RBU0tfV0VJR0hUIjoibWFpbiDpu5jorqQiLCJGUkVTSF9URVNUX1BFUl9FVkFMIjoibWFpbiDpu5jorqQiLCJHUkFEX0FDQ1VNX1NURVBTIjoibWFpbiDpu5jorqQiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX01BU0siOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQT1NfUFJPQkUiOiJtYWluIOm7mOiupCIsIlJBTkRPTV9TRUVEIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiJtYWluIOm7mOiupCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoic2luZ2xlIn1dLCJsb2NhbF9ncHVfYWxsb3dsaXN0IjpbMCwxLDIsM10sInBhY2thZ2luZ19ob3N0IjoidGVzdCIsInNvdXJjZV9maWxlcyI6eyJyZXF1aXJlbWVudHMudHh0IjoiOWMxYzAxODNmYTFhZGFiMTE3ZjgyYWRhMDFmMGFhMzRhZmFhZDYzMjllZmM1MmRhNGQyOGNiMjRiNTViMDg3NiIsInNjcmlwdHMvZXhwZXJpbWVudF9wYWNrYWdlX3J1bm5lci5weSI6IjBiYTE0N2Y4YTc2NzY1ODc5MDVjNDMxMzA5MmYxZjk5ZTFlMmI0NmU5ZTlmYzg3ODcxMzFmOGRiZmE3MjNmM2IiLCJzcmMvYWN0aXZhdGlvbl9wYXRjaC5weSI6IjFhNmE1OTgwMTU5MjgyODdkNTI1OWUyNDc0MDdkNDk0ZDAyOTVkZjY2Y2I3MzdjN2M1NGMxYzRjYmY4YmM3YjIiLCJzcmMvYW5hbHl6ZV9hdHRlbnRpb24ucHkiOiIyZDA3MjMxODMyMGIwMDY1NGFjNzk1MDM2N2ExMzk4NDJiNjdkOGY3M2FlOGFlN2EwNTk4MDFjMGExODFhNzE4Iiwic3JjL2JhdGNoX3J1bi5weSI6IjBkZjY3YTdiZTUxN2NkYWM3NmJiMDU0N2YwNDhlZDEzNTgzYjcyZGM3MmZhOTQwNTEzOTIyOGQ3MzY2NmE1NjkiLCJzcmMvYmluYXJ5X2FyaXRobWV0aWMucHkiOiI1OWZiMGI4OTFjNmRjNWQ5YzZjYTBkZmVkNmI1Y2QxNTE0OWM1NDk2YjkwMTIyZWRlZGU2ZWFkYjllNjllODYwIiwic3JjL2NoYWluX3J1bi5weSI6IjA1OTY1YWUzMzU4ZjYxZjUwY2E5OTVmMmM2ZTdkOGE1OGM1MDNlZGFkODIyZmYyZTA3YzNlNmY2MWQ5MDA0ZTMiLCJzcmMvY29uZmlnLmpzb24iOiIwYTMxMTk5MTI2MThmNmQ1M2ZiNzMxMTA0ZDNkMmYyNzQ3N2RkNmVhYjRhMWQxMjFhYzgxZWI1ZThiNjY3OWM5Iiwic3JjL2NvcmUucHkiOiIxNTIyYTMwYmFkYTg0M2YwMDkwMjNhMmE3MDU3NWQ3NzE0YTNlODY2M2YwMTIxMDc3YWI2MDQ2NTAwZTdjMGFiIiwic3JjL2N1cnJpY3VsdW1fY2hhaW5fTnRhc2sucHkiOiI0N2ViZGJhYmVjYjE4NjE0MzU3OTQ0NTNhMGM0NDFkMGIwOWY4YzE4MWFjODQwNGJhMTdmZjZlYTBkYTdjNTkzIiwic3JjL2N1cnJpY3VsdW1fY2hhaW5fZXh0ZW5kLnB5IjoiYjI3MzUzNTQyMTYxMGIzNDYzNGVlYjMwNzY4YTU5MTg1YzE4NzBkYjE1YzAwOGRjOTg4OTk1NmZhY2RhM2ZmZiIsInNyYy9jdXJyaWN1bHVtX2NoYWluX3JldmVyc2UucHkiOiI1YzMxODkyMTBjZTYxMmU0NDg3MjE5OGY2YTIyZTBiMjdmNWUyM2EwMDg4MDYyNWIxNDVjZTFhNzgwNmU2ZWE3Iiwic3JjL2RhdGFzZXRzLnB5IjoiYWY0ODM1MmVmMzFmZDg0NGRmZDIzN2M5YzJhNjEyY2FmYjI0ZjI0MzdjMzI3OGRjYTQyNzM1NmRlYWI5OThjZiIsInNyYy9lcnJvcl9icmVha2Rvd24ucHkiOiJhNWRmMmU2MjQzYjI2NTBlNDg0ZDQ5NDMwN2Q5MzBkM2UxMGFiZDgwZWJlZjExMWI2ZTdiNTMzNDI5OWI2MGUxIiwic3JjL2V2aWRlbmNlX3N3ZWVwLnB5IjoiMzU1MDI1NzMxYTg3NDQyM2JlNzEwZWU0YjJhNWY0NGQ0MDA4OGFjNGVmYjE0YjliMjRjMGQyM2ZiOGJlMWI0OSIsInNyYy9leHBlcmltZW50LnB5IjoiZWI3MTliMzM1OGRlMGJmODAwOWE5NmEwN2RhZjc5ZmM1Y2MzNmIyYjVhYTY0OWRkN2MzNDk0OWY2MjQ1Y2E1NCIsInNyYy9maXhfbG9ncy5weSI6IjgzODNmOWMxN2YyZmFkOTNhNGFmOGUzMDQ1NjlmMGMwZWNjZWRhMzdjZTk1Y2FkMzc3YWFiOGJhZjBkMjA4Y2EiLCJzcmMvaGVhZF9hYmxhdGlvbi5weSI6ImQwMzg0NjEwMGQzZDlhOTc2NzFiOTJjNTFjYzI3NjFkNjgyMTFjOTRmMDBjNWQ2YTVjODdhNDFiZDE2MTY0NWYiLCJzcmMvbW9kZWxzLnB5IjoiYWExMTYxNzEzODcxMTk0NDAyMTYzYzgzMTY2ODQ1MzM3MmM4MGFlMDZmMTg2NTY4YmRkMWU3MzE3NTk4ODM1ZCIsInNyYy9tdWx0aV9wb3NfcGF0Y2gucHkiOiJlNTQzMzQxMTMzYmJmNzM5OTg4MjA3ZTEwODVjYWRlZTE5N2E2NmVlZWY1MzgwNjE3NGU0NTA5NmQxM2FjZTljIiwic3JjL3ByZXBhcmVfcmVydW4ucHkiOiI3ZmNhZDA3NWE4YTk5NjRhODNjNzdhNDNlMWZmYTI5MmEyMDY0YzdmMTU3NmI4ZTEyNDVkZWZlOTA1YTk3Y2Y0Iiwic3JjL3Byb3RvY29sLnB5IjoiZGFmYTU4Y2RiYWZhOTQ2ZGY1ZTlkZGFiNzhhMzQyNzc0MzU3NWQyZWM2MjliNTNkYzhjOGNmNTRiODIyM2U3MiIsInNyYy9yZXBvcnRfZnJvbnRfYmFja19hdHRlbnRpb24ucHkiOiI1YzczZmYyNjJkMmQ4ZDllOGRlZjdjYzAxYzFmODI4Zjk0MzE1MmNlYjg2ZjNjYjdhZjM0YjhlYzM5MjE1ZDY1Iiwic3JjL3J1bGVfZml0LnB5IjoiYWMyYzUyYTA3NjRlMWYxOWI1ZDE0ZWE5MGIxODg5YmI2NGViZTZhOWRlZTk5ZGRjOWQzM2RjYTQ0MzM2YzljYyIsInNyYy9ydWxlX2dlbmVyYWxpemF0aW9uLnB5IjoiNGIxOTI3Zjg0MjVlYzdmODgwZjQ3ZmIzMTQ0MTQxYjFmZGI2YjdhMzcxNmVmOTBmNzhhMTk2MmMyZGIzMzllMSIsInNyYy9ydWxlcy5weSI6ImRhNmM5YzU4Yzg0OGI0MmRlZGVjYzRiZGIzZmFmZTljNWFhMTNkNmU5YjgzMWFkOGM3NGY2OGVkOTE1N2MwZGUiLCJzcmMvc2FtcGxlX2dlbi5weSI6IjgzOTNiMzNmOTBhZWI3NjBmZjZhNTZiZjhjZmE4NTgyMzZjYjJkMTNlYjU0OGE5OGVmZDBmNjg1MmJlMjUzZDEiLCJzcmMvc2NlbmFyaW9zLnB5IjoiNmNjMWJhZWZkNmYyM2I0YWI2NjUxYmNlZTcyMWU5ZDQyMGQzOWM5N2ZhZmU1YzY1ZThhY2I2ZTY2YzNjOTRkMyIsInNyYy90cmFpbmluZy5weSI6IjYwY2VmN2Y0YTdkM2E5NGMzZGJiMTYwMmRmNWUyZWRmZmRmZmUwNDUyMWM1YjIzZDE3N2RkZjY1ODRiZGVjNWUiLCJzcmMvdmVyaWZ5X2NpcmNsZS5weSI6ImUzMWM1YWRiNjdlMGI4M2IxOWI3ZTRiOTI3MGQ5ZmI1YWM4ZjVlZmQxZDFmM2ExZTFhZWU2MWQyNDVmNDRmZjMiLCJzcmMvdmVyaWZ5X3NhbXBsZS5weSI6IjY3OTBjM2YwZWNhMmE5M2EwM2I5NTE4MWVmYjM1OGQ5YTI1YTc2MTk3MWM0YTYyYmNiNDNiOWIwMTdjNTE4NGUiLCJzcmMvdmlzdWFsaXplLnB5IjoiMTJhNDUzYWU5NTc2MTU1NTFmYWNhYjUxNmFhYjI3Y2E1ZWJlYTJkNGQzNjk3MzNkYTg3OWFjNzIzZTdjYWEyNyJ9LCJ1bnJlc29sdmVkIjpbXSwidmVyc2lvbiI6MX0= -->
