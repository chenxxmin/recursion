# 实验参数确认表

审阅版本：687505114f37bae8fc5c
批次：action_online_300k_11runs_3gpu_6h_20260922；最大并发：3
此阶段只审阅参数，尚未生成可执行实验 JSON 或启动实验。

| 实验 | 任务 | 实际训练样本 | Seed | LR / WD / clip | 初始化 | 每条定时(h) |
| --- | --- | --- | --- | --- | --- | --- |
| action_d1024l2r4h4_P127_N2_tr64_miss01len1_online300k_seed17996_6h | action | 300000 | 17996 | 0.0003 / 1 / 1.0 | fresh | 6 |
| action_d1024l4r4h4_P127_N2_tr64_miss01len1_online300k_seed17996_6h | action | 300000 | 17996 | 0.0003 / 1 / 1.0 | fresh | 6 |
| action_d1024l6r4h4_P127_N2_tr64_miss01len1_online300k_seed17996_6h | action | 300000 | 17996 | 0.0003 / 1 / 1.0 | fresh | 6 |
| action_d1024l2r4h4_P127_N2_tr64_miss01len2_online300k_seed17996_6h | action | 300000 | 17996 | 0.0003 / 1 / 1.0 | fresh | 6 |
| action_d1024l6r4h4_P127_N2_tr64_miss01len2_online300k_seed17996_6h | action | 300000 | 17996 | 0.0003 / 1 / 1.0 | fresh | 6 |
| action_d1024l2r4h4_P127_N2_tr64_miss01len3_online300k_seed17996_6h | action | 300000 | 17996 | 0.0003 / 1 / 1.0 | fresh | 6 |
| action_d1024l6r4h4_P127_N2_tr64_miss01len3_online300k_seed17996_6h | action | 300000 | 17996 | 0.0003 / 1 / 1.0 | fresh | 6 |
| action_d1024l2r4h4_P127_N2_tr64_miss01len5_online300k_seed17996_6h | action | 300000 | 17996 | 0.0003 / 1 / 1.0 | fresh | 6 |
| action_d1024l2r4h4_P127_N2_tr64_miss01len8_online300k_seed17996_6h | action | 300000 | 17996 | 0.0003 / 1 / 1.0 | fresh | 6 |
| action_d1024l4r4h4_P127_N2_tr64_miss01len8_online300k_seed17996_6h | action | 300000 | 17996 | 0.0003 / 1 / 1.0 | fresh | 6 |
| action_d1024l6r4h4_P127_N2_tr64_miss01len8_online300k_seed17996_6h | action | 300000 | 17996 | 0.0003 / 1 / 1.0 | fresh | 6 |

跨机运行规则：启动前必须取得全部预设空闲 GPU，否则整批退出；启动后 FIFO 接续。
每个实验在 output_root 下单独建目录，包含 configs/logs/checkpoints/plots/metrics。
每组独立计时；后台运行继承 GPU 运行锁；resume 保持优化器与 scheduler 状态。

| 运行参数 | 确认值 |
| --- | --- |
| clean_eval_samples | 1000 |
| clean_eval_seed | 123 |
| output_root | "/data/cxm/final" |
| required_gpus | 3 |

## action_d1024l2r4h4_P127_N2_tr64_miss01len1_online300k_seed17996_6h

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| A | 1 | main 默认 |
| AB_PAIRS | [[1,1],[2,3]] | 用户公共参数 |
| ALLOW_TF32 | true | 用户公共参数 |
| AMP_DTYPE | "bfloat16" | main 默认 |
| B | 1 | main 默认 |
| BATCH_SIZE | 512 | 用户公共参数 |
| COND_FIX | null | main 默认 |
| COND_FIX_START | null | main 默认 |
| COND_FIX_START_A1 | null | main 默认 |
| COND_FIX_START_A2 | null | main 默认 |
| DATA_MODE | null | main 默认 |
| DROPOUT | 0.0 | 用户公共参数 |
| D_MODEL | 1024 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 200 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | 用户公共参数 |
| ENTROPY_PENALTY_WEIGHT | 0.0 | main 默认 |
| EPOCHS | 6000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | true | 用户公共参数 |
| GRAD_ACCUM_STEPS | 1 | 用户公共参数 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 6 | 用户公共参数 |
| MAX_UNIQUE_RATIO | 0.7 | main 默认 |
| MISSING_PROB | 0.1 | 用户公共参数 |
| MISS_LEN | 1 | 用户逐实验参数 |
| MISS_SECOND | false | 用户公共参数 |
| MLP_RATIO | 4 | 用户公共参数 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | 用户公共参数 |
| NUM_TRAIN_SAMPLES | 300000 | 用户公共参数 |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户逐实验参数 |
| OOD_LEN | 64 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| P | 127 | 用户公共参数 |
| PREDICT_MISSING | false | 代码默认 |
| RANDOM_SEED | 17996 | 用户公共参数 |
| RESHUFFLE_EACH_EPOCH | false | 用户公共参数 |
| RESUME_FROM | null | 用户公共参数 |
| SKIP_FINAL_GENERATION_TEST | false | 代码默认 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "action" | 实验任务 |
| TRAIN_LEN | 64 | 用户公共参数 |
| USE_AMP | false | 用户公共参数 |
| USE_LEARNABLE_PE | false | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "数据集 loss_mask" |
| actual_test_samples | 256 |
| actual_train_samples | 300000 |
| block_size | 128 |
| coefficient_convention | "第一个系数乘最老值；flag 在每一步选择规则" |
| effective_data_mode | "generated + fresh_test" |
| eval_base_length | 64 |
| eval_sequence_tokens | 126 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "/data/cxm/final/<experiment>__<timestamp>/checkpoints/model.pth" |
| recurrence_order | 2 |
| rules | [[1,1],[2,3]] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | "action 动态生成；规则逐步随机选择，无固定每规则样本配额" |
| train_sequence_tokens | 126 |

- action 使用数据集的逐位置 loss_mask；NUM_MASK 配置不参与 loss。
- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束后保存完整 resume，执行同长度 clean 评估并导出曲线；评估使用原 GPU，不计入训练时限。

## action_d1024l4r4h4_P127_N2_tr64_miss01len1_online300k_seed17996_6h

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| A | 1 | main 默认 |
| AB_PAIRS | [[1,1],[2,3]] | 用户公共参数 |
| ALLOW_TF32 | true | 用户公共参数 |
| AMP_DTYPE | "bfloat16" | main 默认 |
| B | 1 | main 默认 |
| BATCH_SIZE | 512 | 用户公共参数 |
| COND_FIX | null | main 默认 |
| COND_FIX_START | null | main 默认 |
| COND_FIX_START_A1 | null | main 默认 |
| COND_FIX_START_A2 | null | main 默认 |
| DATA_MODE | null | main 默认 |
| DROPOUT | 0.0 | 用户公共参数 |
| D_MODEL | 1024 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 200 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | 用户公共参数 |
| ENTROPY_PENALTY_WEIGHT | 0.0 | main 默认 |
| EPOCHS | 6000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | true | 用户公共参数 |
| GRAD_ACCUM_STEPS | 1 | 用户公共参数 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 6 | 用户公共参数 |
| MAX_UNIQUE_RATIO | 0.7 | main 默认 |
| MISSING_PROB | 0.1 | 用户公共参数 |
| MISS_LEN | 1 | 用户逐实验参数 |
| MISS_SECOND | false | 用户公共参数 |
| MLP_RATIO | 4 | 用户公共参数 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | 用户公共参数 |
| NUM_TRAIN_SAMPLES | 300000 | 用户公共参数 |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 4 | 用户逐实验参数 |
| OOD_LEN | 64 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| P | 127 | 用户公共参数 |
| PREDICT_MISSING | false | 代码默认 |
| RANDOM_SEED | 17996 | 用户公共参数 |
| RESHUFFLE_EACH_EPOCH | false | 用户公共参数 |
| RESUME_FROM | null | 用户公共参数 |
| SKIP_FINAL_GENERATION_TEST | false | 代码默认 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "action" | 实验任务 |
| TRAIN_LEN | 64 | 用户公共参数 |
| USE_AMP | false | 用户公共参数 |
| USE_LEARNABLE_PE | false | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "数据集 loss_mask" |
| actual_test_samples | 256 |
| actual_train_samples | 300000 |
| block_size | 128 |
| coefficient_convention | "第一个系数乘最老值；flag 在每一步选择规则" |
| effective_data_mode | "generated + fresh_test" |
| eval_base_length | 64 |
| eval_sequence_tokens | 126 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "/data/cxm/final/<experiment>__<timestamp>/checkpoints/model.pth" |
| recurrence_order | 2 |
| rules | [[1,1],[2,3]] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | "action 动态生成；规则逐步随机选择，无固定每规则样本配额" |
| train_sequence_tokens | 126 |

- action 使用数据集的逐位置 loss_mask；NUM_MASK 配置不参与 loss。
- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束后保存完整 resume，执行同长度 clean 评估并导出曲线；评估使用原 GPU，不计入训练时限。

## action_d1024l6r4h4_P127_N2_tr64_miss01len1_online300k_seed17996_6h

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| A | 1 | main 默认 |
| AB_PAIRS | [[1,1],[2,3]] | 用户公共参数 |
| ALLOW_TF32 | true | 用户公共参数 |
| AMP_DTYPE | "bfloat16" | main 默认 |
| B | 1 | main 默认 |
| BATCH_SIZE | 512 | 用户公共参数 |
| COND_FIX | null | main 默认 |
| COND_FIX_START | null | main 默认 |
| COND_FIX_START_A1 | null | main 默认 |
| COND_FIX_START_A2 | null | main 默认 |
| DATA_MODE | null | main 默认 |
| DROPOUT | 0.0 | 用户公共参数 |
| D_MODEL | 1024 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 200 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | 用户公共参数 |
| ENTROPY_PENALTY_WEIGHT | 0.0 | main 默认 |
| EPOCHS | 6000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | true | 用户公共参数 |
| GRAD_ACCUM_STEPS | 1 | 用户公共参数 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 6 | 用户公共参数 |
| MAX_UNIQUE_RATIO | 0.7 | main 默认 |
| MISSING_PROB | 0.1 | 用户公共参数 |
| MISS_LEN | 1 | 用户逐实验参数 |
| MISS_SECOND | false | 用户公共参数 |
| MLP_RATIO | 4 | 用户公共参数 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | 用户公共参数 |
| NUM_TRAIN_SAMPLES | 300000 | 用户公共参数 |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 6 | 用户逐实验参数 |
| OOD_LEN | 64 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| P | 127 | 用户公共参数 |
| PREDICT_MISSING | false | 代码默认 |
| RANDOM_SEED | 17996 | 用户公共参数 |
| RESHUFFLE_EACH_EPOCH | false | 用户公共参数 |
| RESUME_FROM | null | 用户公共参数 |
| SKIP_FINAL_GENERATION_TEST | false | 代码默认 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "action" | 实验任务 |
| TRAIN_LEN | 64 | 用户公共参数 |
| USE_AMP | false | 用户公共参数 |
| USE_LEARNABLE_PE | false | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "数据集 loss_mask" |
| actual_test_samples | 256 |
| actual_train_samples | 300000 |
| block_size | 128 |
| coefficient_convention | "第一个系数乘最老值；flag 在每一步选择规则" |
| effective_data_mode | "generated + fresh_test" |
| eval_base_length | 64 |
| eval_sequence_tokens | 126 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "/data/cxm/final/<experiment>__<timestamp>/checkpoints/model.pth" |
| recurrence_order | 2 |
| rules | [[1,1],[2,3]] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | "action 动态生成；规则逐步随机选择，无固定每规则样本配额" |
| train_sequence_tokens | 126 |

- action 使用数据集的逐位置 loss_mask；NUM_MASK 配置不参与 loss。
- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束后保存完整 resume，执行同长度 clean 评估并导出曲线；评估使用原 GPU，不计入训练时限。

## action_d1024l2r4h4_P127_N2_tr64_miss01len2_online300k_seed17996_6h

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| A | 1 | main 默认 |
| AB_PAIRS | [[1,1],[2,3]] | 用户公共参数 |
| ALLOW_TF32 | true | 用户公共参数 |
| AMP_DTYPE | "bfloat16" | main 默认 |
| B | 1 | main 默认 |
| BATCH_SIZE | 512 | 用户公共参数 |
| COND_FIX | null | main 默认 |
| COND_FIX_START | null | main 默认 |
| COND_FIX_START_A1 | null | main 默认 |
| COND_FIX_START_A2 | null | main 默认 |
| DATA_MODE | null | main 默认 |
| DROPOUT | 0.0 | 用户公共参数 |
| D_MODEL | 1024 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 200 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | 用户公共参数 |
| ENTROPY_PENALTY_WEIGHT | 0.0 | main 默认 |
| EPOCHS | 6000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | true | 用户公共参数 |
| GRAD_ACCUM_STEPS | 1 | 用户公共参数 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 6 | 用户公共参数 |
| MAX_UNIQUE_RATIO | 0.7 | main 默认 |
| MISSING_PROB | 0.1 | 用户公共参数 |
| MISS_LEN | 2 | 用户逐实验参数 |
| MISS_SECOND | false | 用户公共参数 |
| MLP_RATIO | 4 | 用户公共参数 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | 用户公共参数 |
| NUM_TRAIN_SAMPLES | 300000 | 用户公共参数 |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户逐实验参数 |
| OOD_LEN | 64 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| P | 127 | 用户公共参数 |
| PREDICT_MISSING | false | 代码默认 |
| RANDOM_SEED | 17996 | 用户公共参数 |
| RESHUFFLE_EACH_EPOCH | false | 用户公共参数 |
| RESUME_FROM | null | 用户公共参数 |
| SKIP_FINAL_GENERATION_TEST | false | 代码默认 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "action" | 实验任务 |
| TRAIN_LEN | 64 | 用户公共参数 |
| USE_AMP | false | 用户公共参数 |
| USE_LEARNABLE_PE | false | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "数据集 loss_mask" |
| actual_test_samples | 256 |
| actual_train_samples | 300000 |
| block_size | 128 |
| coefficient_convention | "第一个系数乘最老值；flag 在每一步选择规则" |
| effective_data_mode | "generated + fresh_test" |
| eval_base_length | 64 |
| eval_sequence_tokens | 126 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "/data/cxm/final/<experiment>__<timestamp>/checkpoints/model.pth" |
| recurrence_order | 2 |
| rules | [[1,1],[2,3]] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | "action 动态生成；规则逐步随机选择，无固定每规则样本配额" |
| train_sequence_tokens | 126 |

- action 使用数据集的逐位置 loss_mask；NUM_MASK 配置不参与 loss。
- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束后保存完整 resume，执行同长度 clean 评估并导出曲线；评估使用原 GPU，不计入训练时限。

## action_d1024l6r4h4_P127_N2_tr64_miss01len2_online300k_seed17996_6h

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| A | 1 | main 默认 |
| AB_PAIRS | [[1,1],[2,3]] | 用户公共参数 |
| ALLOW_TF32 | true | 用户公共参数 |
| AMP_DTYPE | "bfloat16" | main 默认 |
| B | 1 | main 默认 |
| BATCH_SIZE | 512 | 用户公共参数 |
| COND_FIX | null | main 默认 |
| COND_FIX_START | null | main 默认 |
| COND_FIX_START_A1 | null | main 默认 |
| COND_FIX_START_A2 | null | main 默认 |
| DATA_MODE | null | main 默认 |
| DROPOUT | 0.0 | 用户公共参数 |
| D_MODEL | 1024 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 200 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | 用户公共参数 |
| ENTROPY_PENALTY_WEIGHT | 0.0 | main 默认 |
| EPOCHS | 6000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | true | 用户公共参数 |
| GRAD_ACCUM_STEPS | 1 | 用户公共参数 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 6 | 用户公共参数 |
| MAX_UNIQUE_RATIO | 0.7 | main 默认 |
| MISSING_PROB | 0.1 | 用户公共参数 |
| MISS_LEN | 2 | 用户逐实验参数 |
| MISS_SECOND | false | 用户公共参数 |
| MLP_RATIO | 4 | 用户公共参数 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | 用户公共参数 |
| NUM_TRAIN_SAMPLES | 300000 | 用户公共参数 |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 6 | 用户逐实验参数 |
| OOD_LEN | 64 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| P | 127 | 用户公共参数 |
| PREDICT_MISSING | false | 代码默认 |
| RANDOM_SEED | 17996 | 用户公共参数 |
| RESHUFFLE_EACH_EPOCH | false | 用户公共参数 |
| RESUME_FROM | null | 用户公共参数 |
| SKIP_FINAL_GENERATION_TEST | false | 代码默认 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "action" | 实验任务 |
| TRAIN_LEN | 64 | 用户公共参数 |
| USE_AMP | false | 用户公共参数 |
| USE_LEARNABLE_PE | false | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "数据集 loss_mask" |
| actual_test_samples | 256 |
| actual_train_samples | 300000 |
| block_size | 128 |
| coefficient_convention | "第一个系数乘最老值；flag 在每一步选择规则" |
| effective_data_mode | "generated + fresh_test" |
| eval_base_length | 64 |
| eval_sequence_tokens | 126 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "/data/cxm/final/<experiment>__<timestamp>/checkpoints/model.pth" |
| recurrence_order | 2 |
| rules | [[1,1],[2,3]] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | "action 动态生成；规则逐步随机选择，无固定每规则样本配额" |
| train_sequence_tokens | 126 |

- action 使用数据集的逐位置 loss_mask；NUM_MASK 配置不参与 loss。
- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束后保存完整 resume，执行同长度 clean 评估并导出曲线；评估使用原 GPU，不计入训练时限。

## action_d1024l2r4h4_P127_N2_tr64_miss01len3_online300k_seed17996_6h

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| A | 1 | main 默认 |
| AB_PAIRS | [[1,1],[2,3]] | 用户公共参数 |
| ALLOW_TF32 | true | 用户公共参数 |
| AMP_DTYPE | "bfloat16" | main 默认 |
| B | 1 | main 默认 |
| BATCH_SIZE | 512 | 用户公共参数 |
| COND_FIX | null | main 默认 |
| COND_FIX_START | null | main 默认 |
| COND_FIX_START_A1 | null | main 默认 |
| COND_FIX_START_A2 | null | main 默认 |
| DATA_MODE | null | main 默认 |
| DROPOUT | 0.0 | 用户公共参数 |
| D_MODEL | 1024 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 200 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | 用户公共参数 |
| ENTROPY_PENALTY_WEIGHT | 0.0 | main 默认 |
| EPOCHS | 6000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | true | 用户公共参数 |
| GRAD_ACCUM_STEPS | 1 | 用户公共参数 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 6 | 用户公共参数 |
| MAX_UNIQUE_RATIO | 0.7 | main 默认 |
| MISSING_PROB | 0.1 | 用户公共参数 |
| MISS_LEN | 3 | 用户逐实验参数 |
| MISS_SECOND | false | 用户公共参数 |
| MLP_RATIO | 4 | 用户公共参数 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | 用户公共参数 |
| NUM_TRAIN_SAMPLES | 300000 | 用户公共参数 |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户逐实验参数 |
| OOD_LEN | 64 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| P | 127 | 用户公共参数 |
| PREDICT_MISSING | false | 代码默认 |
| RANDOM_SEED | 17996 | 用户公共参数 |
| RESHUFFLE_EACH_EPOCH | false | 用户公共参数 |
| RESUME_FROM | null | 用户公共参数 |
| SKIP_FINAL_GENERATION_TEST | false | 代码默认 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "action" | 实验任务 |
| TRAIN_LEN | 64 | 用户公共参数 |
| USE_AMP | false | 用户公共参数 |
| USE_LEARNABLE_PE | false | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "数据集 loss_mask" |
| actual_test_samples | 256 |
| actual_train_samples | 300000 |
| block_size | 128 |
| coefficient_convention | "第一个系数乘最老值；flag 在每一步选择规则" |
| effective_data_mode | "generated + fresh_test" |
| eval_base_length | 64 |
| eval_sequence_tokens | 126 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "/data/cxm/final/<experiment>__<timestamp>/checkpoints/model.pth" |
| recurrence_order | 2 |
| rules | [[1,1],[2,3]] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | "action 动态生成；规则逐步随机选择，无固定每规则样本配额" |
| train_sequence_tokens | 126 |

- action 使用数据集的逐位置 loss_mask；NUM_MASK 配置不参与 loss。
- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束后保存完整 resume，执行同长度 clean 评估并导出曲线；评估使用原 GPU，不计入训练时限。

## action_d1024l6r4h4_P127_N2_tr64_miss01len3_online300k_seed17996_6h

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| A | 1 | main 默认 |
| AB_PAIRS | [[1,1],[2,3]] | 用户公共参数 |
| ALLOW_TF32 | true | 用户公共参数 |
| AMP_DTYPE | "bfloat16" | main 默认 |
| B | 1 | main 默认 |
| BATCH_SIZE | 512 | 用户公共参数 |
| COND_FIX | null | main 默认 |
| COND_FIX_START | null | main 默认 |
| COND_FIX_START_A1 | null | main 默认 |
| COND_FIX_START_A2 | null | main 默认 |
| DATA_MODE | null | main 默认 |
| DROPOUT | 0.0 | 用户公共参数 |
| D_MODEL | 1024 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 200 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | 用户公共参数 |
| ENTROPY_PENALTY_WEIGHT | 0.0 | main 默认 |
| EPOCHS | 6000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | true | 用户公共参数 |
| GRAD_ACCUM_STEPS | 1 | 用户公共参数 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 6 | 用户公共参数 |
| MAX_UNIQUE_RATIO | 0.7 | main 默认 |
| MISSING_PROB | 0.1 | 用户公共参数 |
| MISS_LEN | 3 | 用户逐实验参数 |
| MISS_SECOND | false | 用户公共参数 |
| MLP_RATIO | 4 | 用户公共参数 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | 用户公共参数 |
| NUM_TRAIN_SAMPLES | 300000 | 用户公共参数 |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 6 | 用户逐实验参数 |
| OOD_LEN | 64 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| P | 127 | 用户公共参数 |
| PREDICT_MISSING | false | 代码默认 |
| RANDOM_SEED | 17996 | 用户公共参数 |
| RESHUFFLE_EACH_EPOCH | false | 用户公共参数 |
| RESUME_FROM | null | 用户公共参数 |
| SKIP_FINAL_GENERATION_TEST | false | 代码默认 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "action" | 实验任务 |
| TRAIN_LEN | 64 | 用户公共参数 |
| USE_AMP | false | 用户公共参数 |
| USE_LEARNABLE_PE | false | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "数据集 loss_mask" |
| actual_test_samples | 256 |
| actual_train_samples | 300000 |
| block_size | 128 |
| coefficient_convention | "第一个系数乘最老值；flag 在每一步选择规则" |
| effective_data_mode | "generated + fresh_test" |
| eval_base_length | 64 |
| eval_sequence_tokens | 126 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "/data/cxm/final/<experiment>__<timestamp>/checkpoints/model.pth" |
| recurrence_order | 2 |
| rules | [[1,1],[2,3]] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | "action 动态生成；规则逐步随机选择，无固定每规则样本配额" |
| train_sequence_tokens | 126 |

- action 使用数据集的逐位置 loss_mask；NUM_MASK 配置不参与 loss。
- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束后保存完整 resume，执行同长度 clean 评估并导出曲线；评估使用原 GPU，不计入训练时限。

## action_d1024l2r4h4_P127_N2_tr64_miss01len5_online300k_seed17996_6h

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| A | 1 | main 默认 |
| AB_PAIRS | [[1,1],[2,3]] | 用户公共参数 |
| ALLOW_TF32 | true | 用户公共参数 |
| AMP_DTYPE | "bfloat16" | main 默认 |
| B | 1 | main 默认 |
| BATCH_SIZE | 512 | 用户公共参数 |
| COND_FIX | null | main 默认 |
| COND_FIX_START | null | main 默认 |
| COND_FIX_START_A1 | null | main 默认 |
| COND_FIX_START_A2 | null | main 默认 |
| DATA_MODE | null | main 默认 |
| DROPOUT | 0.0 | 用户公共参数 |
| D_MODEL | 1024 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 200 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | 用户公共参数 |
| ENTROPY_PENALTY_WEIGHT | 0.0 | main 默认 |
| EPOCHS | 6000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | true | 用户公共参数 |
| GRAD_ACCUM_STEPS | 1 | 用户公共参数 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 6 | 用户公共参数 |
| MAX_UNIQUE_RATIO | 0.7 | main 默认 |
| MISSING_PROB | 0.1 | 用户公共参数 |
| MISS_LEN | 5 | 用户逐实验参数 |
| MISS_SECOND | false | 用户公共参数 |
| MLP_RATIO | 4 | 用户公共参数 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | 用户公共参数 |
| NUM_TRAIN_SAMPLES | 300000 | 用户公共参数 |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户逐实验参数 |
| OOD_LEN | 64 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| P | 127 | 用户公共参数 |
| PREDICT_MISSING | false | 代码默认 |
| RANDOM_SEED | 17996 | 用户公共参数 |
| RESHUFFLE_EACH_EPOCH | false | 用户公共参数 |
| RESUME_FROM | null | 用户公共参数 |
| SKIP_FINAL_GENERATION_TEST | false | 代码默认 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "action" | 实验任务 |
| TRAIN_LEN | 64 | 用户公共参数 |
| USE_AMP | false | 用户公共参数 |
| USE_LEARNABLE_PE | false | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "数据集 loss_mask" |
| actual_test_samples | 256 |
| actual_train_samples | 300000 |
| block_size | 128 |
| coefficient_convention | "第一个系数乘最老值；flag 在每一步选择规则" |
| effective_data_mode | "generated + fresh_test" |
| eval_base_length | 64 |
| eval_sequence_tokens | 126 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "/data/cxm/final/<experiment>__<timestamp>/checkpoints/model.pth" |
| recurrence_order | 2 |
| rules | [[1,1],[2,3]] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | "action 动态生成；规则逐步随机选择，无固定每规则样本配额" |
| train_sequence_tokens | 126 |

- action 使用数据集的逐位置 loss_mask；NUM_MASK 配置不参与 loss。
- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束后保存完整 resume，执行同长度 clean 评估并导出曲线；评估使用原 GPU，不计入训练时限。

## action_d1024l2r4h4_P127_N2_tr64_miss01len8_online300k_seed17996_6h

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| A | 1 | main 默认 |
| AB_PAIRS | [[1,1],[2,3]] | 用户公共参数 |
| ALLOW_TF32 | true | 用户公共参数 |
| AMP_DTYPE | "bfloat16" | main 默认 |
| B | 1 | main 默认 |
| BATCH_SIZE | 512 | 用户公共参数 |
| COND_FIX | null | main 默认 |
| COND_FIX_START | null | main 默认 |
| COND_FIX_START_A1 | null | main 默认 |
| COND_FIX_START_A2 | null | main 默认 |
| DATA_MODE | null | main 默认 |
| DROPOUT | 0.0 | 用户公共参数 |
| D_MODEL | 1024 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 200 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | 用户公共参数 |
| ENTROPY_PENALTY_WEIGHT | 0.0 | main 默认 |
| EPOCHS | 6000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | true | 用户公共参数 |
| GRAD_ACCUM_STEPS | 1 | 用户公共参数 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 6 | 用户公共参数 |
| MAX_UNIQUE_RATIO | 0.7 | main 默认 |
| MISSING_PROB | 0.1 | 用户公共参数 |
| MISS_LEN | 8 | 用户逐实验参数 |
| MISS_SECOND | false | 用户公共参数 |
| MLP_RATIO | 4 | 用户公共参数 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | 用户公共参数 |
| NUM_TRAIN_SAMPLES | 300000 | 用户公共参数 |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 2 | 用户逐实验参数 |
| OOD_LEN | 64 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| P | 127 | 用户公共参数 |
| PREDICT_MISSING | false | 代码默认 |
| RANDOM_SEED | 17996 | 用户公共参数 |
| RESHUFFLE_EACH_EPOCH | false | 用户公共参数 |
| RESUME_FROM | null | 用户公共参数 |
| SKIP_FINAL_GENERATION_TEST | false | 代码默认 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "action" | 实验任务 |
| TRAIN_LEN | 64 | 用户公共参数 |
| USE_AMP | false | 用户公共参数 |
| USE_LEARNABLE_PE | false | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "数据集 loss_mask" |
| actual_test_samples | 256 |
| actual_train_samples | 300000 |
| block_size | 128 |
| coefficient_convention | "第一个系数乘最老值；flag 在每一步选择规则" |
| effective_data_mode | "generated + fresh_test" |
| eval_base_length | 64 |
| eval_sequence_tokens | 126 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "/data/cxm/final/<experiment>__<timestamp>/checkpoints/model.pth" |
| recurrence_order | 2 |
| rules | [[1,1],[2,3]] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | "action 动态生成；规则逐步随机选择，无固定每规则样本配额" |
| train_sequence_tokens | 126 |

- action 使用数据集的逐位置 loss_mask；NUM_MASK 配置不参与 loss。
- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束后保存完整 resume，执行同长度 clean 评估并导出曲线；评估使用原 GPU，不计入训练时限。

## action_d1024l4r4h4_P127_N2_tr64_miss01len8_online300k_seed17996_6h

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| A | 1 | main 默认 |
| AB_PAIRS | [[1,1],[2,3]] | 用户公共参数 |
| ALLOW_TF32 | true | 用户公共参数 |
| AMP_DTYPE | "bfloat16" | main 默认 |
| B | 1 | main 默认 |
| BATCH_SIZE | 512 | 用户公共参数 |
| COND_FIX | null | main 默认 |
| COND_FIX_START | null | main 默认 |
| COND_FIX_START_A1 | null | main 默认 |
| COND_FIX_START_A2 | null | main 默认 |
| DATA_MODE | null | main 默认 |
| DROPOUT | 0.0 | 用户公共参数 |
| D_MODEL | 1024 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 200 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | 用户公共参数 |
| ENTROPY_PENALTY_WEIGHT | 0.0 | main 默认 |
| EPOCHS | 6000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | true | 用户公共参数 |
| GRAD_ACCUM_STEPS | 1 | 用户公共参数 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 6 | 用户公共参数 |
| MAX_UNIQUE_RATIO | 0.7 | main 默认 |
| MISSING_PROB | 0.1 | 用户公共参数 |
| MISS_LEN | 8 | 用户逐实验参数 |
| MISS_SECOND | false | 用户公共参数 |
| MLP_RATIO | 4 | 用户公共参数 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | 用户公共参数 |
| NUM_TRAIN_SAMPLES | 300000 | 用户公共参数 |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 4 | 用户逐实验参数 |
| OOD_LEN | 64 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| P | 127 | 用户公共参数 |
| PREDICT_MISSING | false | 代码默认 |
| RANDOM_SEED | 17996 | 用户公共参数 |
| RESHUFFLE_EACH_EPOCH | false | 用户公共参数 |
| RESUME_FROM | null | 用户公共参数 |
| SKIP_FINAL_GENERATION_TEST | false | 代码默认 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "action" | 实验任务 |
| TRAIN_LEN | 64 | 用户公共参数 |
| USE_AMP | false | 用户公共参数 |
| USE_LEARNABLE_PE | false | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "数据集 loss_mask" |
| actual_test_samples | 256 |
| actual_train_samples | 300000 |
| block_size | 128 |
| coefficient_convention | "第一个系数乘最老值；flag 在每一步选择规则" |
| effective_data_mode | "generated + fresh_test" |
| eval_base_length | 64 |
| eval_sequence_tokens | 126 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "/data/cxm/final/<experiment>__<timestamp>/checkpoints/model.pth" |
| recurrence_order | 2 |
| rules | [[1,1],[2,3]] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | "action 动态生成；规则逐步随机选择，无固定每规则样本配额" |
| train_sequence_tokens | 126 |

- action 使用数据集的逐位置 loss_mask；NUM_MASK 配置不参与 loss。
- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束后保存完整 resume，执行同长度 clean 评估并导出曲线；评估使用原 GPU，不计入训练时限。

## action_d1024l6r4h4_P127_N2_tr64_miss01len8_online300k_seed17996_6h

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| A | 1 | main 默认 |
| AB_PAIRS | [[1,1],[2,3]] | 用户公共参数 |
| ALLOW_TF32 | true | 用户公共参数 |
| AMP_DTYPE | "bfloat16" | main 默认 |
| B | 1 | main 默认 |
| BATCH_SIZE | 512 | 用户公共参数 |
| COND_FIX | null | main 默认 |
| COND_FIX_START | null | main 默认 |
| COND_FIX_START_A1 | null | main 默认 |
| COND_FIX_START_A2 | null | main 默认 |
| DATA_MODE | null | main 默认 |
| DROPOUT | 0.0 | 用户公共参数 |
| D_MODEL | 1024 | 用户公共参数 |
| EARLY_STOP_ACCURACY | 0.99 | 用户公共参数 |
| EARLY_STOP_EXTRA_EPOCHS | 200 | main 默认 |
| EARLY_STOP_NO_IMPROVE | 3000 | 用户公共参数 |
| ENTROPY_PENALTY_WEIGHT | 0.0 | main 默认 |
| EPOCHS | 6000 | 用户公共参数 |
| EVAL_INTERVAL | 1 | 用户公共参数 |
| FIRST_TASK_WEIGHT | 1.0 | main 默认 |
| FRESH_TEST_PER_EVAL | true | 用户公共参数 |
| GRAD_ACCUM_STEPS | 1 | 用户公共参数 |
| GRAD_CLIP_NORM | 1.0 | 用户公共参数 |
| INIT_FROM | null | 用户公共参数 |
| LR | 0.0003 | 用户公共参数 |
| MAX_TRAIN_HOURS | 6 | 用户公共参数 |
| MAX_UNIQUE_RATIO | 0.7 | main 默认 |
| MISSING_PROB | 0.1 | 用户公共参数 |
| MISS_LEN | 8 | 用户逐实验参数 |
| MISS_SECOND | false | 用户公共参数 |
| MLP_RATIO | 4 | 用户公共参数 |
| NUM_MASK | null | main 默认 |
| NUM_TEST_SAMPLES | 256 | 用户公共参数 |
| NUM_TRAIN_SAMPLES | 300000 | 用户公共参数 |
| N_HEAD | 4 | 用户公共参数 |
| N_LAYER | 6 | 用户逐实验参数 |
| OOD_LEN | 64 | 用户公共参数 |
| OPT_DIAG_INTERVAL | 0 | main 默认 |
| P | 127 | 用户公共参数 |
| PREDICT_MISSING | false | 代码默认 |
| RANDOM_SEED | 17996 | 用户公共参数 |
| RESHUFFLE_EACH_EPOCH | false | 用户公共参数 |
| RESUME_FROM | null | 用户公共参数 |
| SKIP_FINAL_GENERATION_TEST | false | 代码默认 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "action" | 实验任务 |
| TRAIN_LEN | 64 | 用户公共参数 |
| USE_AMP | false | 用户公共参数 |
| USE_LEARNABLE_PE | false | 用户公共参数 |
| WEIGHT_DECAY | 1 | 用户公共参数 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "数据集 loss_mask" |
| actual_test_samples | 256 |
| actual_train_samples | 300000 |
| block_size | 128 |
| coefficient_convention | "第一个系数乘最老值；flag 在每一步选择规则" |
| effective_data_mode | "generated + fresh_test" |
| eval_base_length | 64 |
| eval_sequence_tokens | 126 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "/data/cxm/final/<experiment>__<timestamp>/checkpoints/model.pth" |
| recurrence_order | 2 |
| rules | [[1,1],[2,3]] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | "action 动态生成；规则逐步随机选择，无固定每规则样本配额" |
| train_sequence_tokens | 126 |

- action 使用数据集的逐位置 loss_mask；NUM_MASK 配置不参与 loss。
- 每条实验独立计时，从训练循环开始；每 1 epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束后保存完整 resume，执行同长度 clean 评估并导出曲线；评估使用原 GPU，不计入训练时限。

打包内容：已确认 JSON、定时写入 run.sh、训练代码快照、依赖清单、检查和运行工具。

请确认此版本；修改参数或训练代码后需重新出表。

<!-- PIPELINE_SNAPSHOT eyJhc3NldHMiOnt9LCJiYXRjaF9uYW1lIjoiYWN0aW9uX29ubGluZV8zMDBrXzExcnVuc18zZ3B1XzZoXzIwMjYwOTIyIiwiY29uY3VycmVuY3kiOjMsImRlbGl2ZXJ5Ijp7ImNsZWFuX2V2YWxfc2FtcGxlcyI6MTAwMCwiY2xlYW5fZXZhbF9zZWVkIjoxMjMsIm91dHB1dF9yb290IjoiL2RhdGEvY3htL2ZpbmFsIiwicmVxdWlyZWRfZ3B1cyI6M30sImV4cGVyaW1lbnRzIjpbeyJjb25maWciOnsiQSI6MSwiQUJfUEFJUlMiOltbMSwxXSxbMiwzXV0sIkFMTE9XX1RGMzIiOnRydWUsIkFNUF9EVFlQRSI6ImJmbG9hdDE2IiwiQiI6MSwiQkFUQ0hfU0laRSI6NTEyLCJDT05EX0ZJWCI6bnVsbCwiQ09ORF9GSVhfU1RBUlQiOm51bGwsIkNPTkRfRklYX1NUQVJUX0ExIjpudWxsLCJDT05EX0ZJWF9TVEFSVF9BMiI6bnVsbCwiREFUQV9NT0RFIjpudWxsLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjEwMjQsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoyMDAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRU5UUk9QWV9QRU5BTFRZX1dFSUdIVCI6MC4wLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOnRydWUsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjo2LCJNQVhfVU5JUVVFX1JBVElPIjowLjcsIk1JU1NJTkdfUFJPQiI6MC4xLCJNSVNTX0xFTiI6MSwiTUlTU19TRUNPTkQiOmZhbHNlLCJNTFBfUkFUSU8iOjQsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjozMDAwMDAsIk5fSEVBRCI6NCwiTl9MQVlFUiI6MiwiT09EX0xFTiI6NjQsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJQIjoxMjcsIlBSRURJQ1RfTUlTU0lORyI6ZmFsc2UsIlJBTkRPTV9TRUVEIjoxNzk5NiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOmZhbHNlLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJhY3Rpb24iLCJUUkFJTl9MRU4iOjY0LCJVU0VfQU1QIjpmYWxzZSwiVVNFX0xFQVJOQUJMRV9QRSI6ZmFsc2UsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiLmlbDmja7pm4YgbG9zc19tYXNrIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6MjU2LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MzAwMDAwLCJibG9ja19zaXplIjoxMjgsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDogIHlgLzvvJtmbGFnIOWcqOavj+S4gOatpemAieaLqeinhOWImSIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJnZW5lcmF0ZWQgKyBmcmVzaF90ZXN0IiwiZXZhbF9iYXNlX2xlbmd0aCI6NjQsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxMjYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJhY3Rpb24g5L2/55So5pWw5o2u6ZuG55qE6YCQ5L2N572uIGxvc3NfbWFza++8m05VTV9NQVNLIOmFjee9ruS4jeWPguS4jiBsb3Nz44CCIiwi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5ZCO5L+d5a2Y5a6M5pW0IHJlc3VtZe+8jOaJp+ihjOWQjOmVv+W6piBjbGVhbiDor4TkvLDlubblr7zlh7rmm7Lnur/vvJvor4TkvLDkvb/nlKjljp8gR1BV77yM5LiN6K6h5YWl6K6t57uD5pe26ZmQ44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiIvZGF0YS9jeG0vZmluYWwvPGV4cGVyaW1lbnQ+X188dGltZXN0YW1wPi9jaGVja3BvaW50cy9tb2RlbC5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6W1sxLDFdLFsyLDNdXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjoiYWN0aW9uIOWKqOaAgeeUn+aIkO+8m+inhOWImemAkOatpemaj+acuumAieaLqe+8jOaXoOWbuuWumuavj+inhOWImeagt+acrOmFjeminSIsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTI2fSwibmFtZSI6ImFjdGlvbl9kMTAyNGwycjRoNF9QMTI3X04yX3RyNjRfbWlzczAxbGVuMV9vbmxpbmUzMDBrX3NlZWQxNzk5Nl82aCIsIm9yaWdpbnMiOnsiQSI6Im1haW4g6buY6K6kIiwiQUJfUEFJUlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJBTExPV19URjMyIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQU1QX0RUWVBFIjoibWFpbiDpu5jorqQiLCJCIjoibWFpbiDpu5jorqQiLCJCQVRDSF9TSVpFIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQ09ORF9GSVgiOiJtYWluIOm7mOiupCIsIkNPTkRfRklYX1NUQVJUIjoibWFpbiDpu5jorqQiLCJDT05EX0ZJWF9TVEFSVF9BMSI6Im1haW4g6buY6K6kIiwiQ09ORF9GSVhfU1RBUlRfQTIiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6Im1haW4g6buY6K6kIiwiRFJPUE9VVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVOVFJPUFlfUEVOQUxUWV9XRUlHSFQiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkdSQURfQUNDVU1fU1RFUFMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1VOSVFVRV9SQVRJTyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU19MRU4iOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJNSVNTX1NFQ09ORCI6IueUqOaIt+WFrOWFseWPguaVsCIsIk1MUF9SQVRJTyI6IueUqOaIt+WFrOWFseWPguaVsCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUFJFRElDVF9NSVNTSU5HIjoi5Luj56CB6buY6K6kIiwiUkFORE9NX1NFRUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOiLku6PnoIHpu5jorqQiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJVU0VfQU1QIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVVNFX0xFQVJOQUJMRV9QRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoiYWN0aW9uIn0seyJjb25maWciOnsiQSI6MSwiQUJfUEFJUlMiOltbMSwxXSxbMiwzXV0sIkFMTE9XX1RGMzIiOnRydWUsIkFNUF9EVFlQRSI6ImJmbG9hdDE2IiwiQiI6MSwiQkFUQ0hfU0laRSI6NTEyLCJDT05EX0ZJWCI6bnVsbCwiQ09ORF9GSVhfU1RBUlQiOm51bGwsIkNPTkRfRklYX1NUQVJUX0ExIjpudWxsLCJDT05EX0ZJWF9TVEFSVF9BMiI6bnVsbCwiREFUQV9NT0RFIjpudWxsLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjEwMjQsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoyMDAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRU5UUk9QWV9QRU5BTFRZX1dFSUdIVCI6MC4wLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOnRydWUsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjo2LCJNQVhfVU5JUVVFX1JBVElPIjowLjcsIk1JU1NJTkdfUFJPQiI6MC4xLCJNSVNTX0xFTiI6MSwiTUlTU19TRUNPTkQiOmZhbHNlLCJNTFBfUkFUSU8iOjQsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjozMDAwMDAsIk5fSEVBRCI6NCwiTl9MQVlFUiI6NCwiT09EX0xFTiI6NjQsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJQIjoxMjcsIlBSRURJQ1RfTUlTU0lORyI6ZmFsc2UsIlJBTkRPTV9TRUVEIjoxNzk5NiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOmZhbHNlLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJhY3Rpb24iLCJUUkFJTl9MRU4iOjY0LCJVU0VfQU1QIjpmYWxzZSwiVVNFX0xFQVJOQUJMRV9QRSI6ZmFsc2UsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiLmlbDmja7pm4YgbG9zc19tYXNrIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6MjU2LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MzAwMDAwLCJibG9ja19zaXplIjoxMjgsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDogIHlgLzvvJtmbGFnIOWcqOavj+S4gOatpemAieaLqeinhOWImSIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJnZW5lcmF0ZWQgKyBmcmVzaF90ZXN0IiwiZXZhbF9iYXNlX2xlbmd0aCI6NjQsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxMjYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJhY3Rpb24g5L2/55So5pWw5o2u6ZuG55qE6YCQ5L2N572uIGxvc3NfbWFza++8m05VTV9NQVNLIOmFjee9ruS4jeWPguS4jiBsb3Nz44CCIiwi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5ZCO5L+d5a2Y5a6M5pW0IHJlc3VtZe+8jOaJp+ihjOWQjOmVv+W6piBjbGVhbiDor4TkvLDlubblr7zlh7rmm7Lnur/vvJvor4TkvLDkvb/nlKjljp8gR1BV77yM5LiN6K6h5YWl6K6t57uD5pe26ZmQ44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiIvZGF0YS9jeG0vZmluYWwvPGV4cGVyaW1lbnQ+X188dGltZXN0YW1wPi9jaGVja3BvaW50cy9tb2RlbC5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6W1sxLDFdLFsyLDNdXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjoiYWN0aW9uIOWKqOaAgeeUn+aIkO+8m+inhOWImemAkOatpemaj+acuumAieaLqe+8jOaXoOWbuuWumuavj+inhOWImeagt+acrOmFjeminSIsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTI2fSwibmFtZSI6ImFjdGlvbl9kMTAyNGw0cjRoNF9QMTI3X04yX3RyNjRfbWlzczAxbGVuMV9vbmxpbmUzMDBrX3NlZWQxNzk5Nl82aCIsIm9yaWdpbnMiOnsiQSI6Im1haW4g6buY6K6kIiwiQUJfUEFJUlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJBTExPV19URjMyIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQU1QX0RUWVBFIjoibWFpbiDpu5jorqQiLCJCIjoibWFpbiDpu5jorqQiLCJCQVRDSF9TSVpFIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQ09ORF9GSVgiOiJtYWluIOm7mOiupCIsIkNPTkRfRklYX1NUQVJUIjoibWFpbiDpu5jorqQiLCJDT05EX0ZJWF9TVEFSVF9BMSI6Im1haW4g6buY6K6kIiwiQ09ORF9GSVhfU1RBUlRfQTIiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6Im1haW4g6buY6K6kIiwiRFJPUE9VVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVOVFJPUFlfUEVOQUxUWV9XRUlHSFQiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkdSQURfQUNDVU1fU1RFUFMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1VOSVFVRV9SQVRJTyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU19MRU4iOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJNSVNTX1NFQ09ORCI6IueUqOaIt+WFrOWFseWPguaVsCIsIk1MUF9SQVRJTyI6IueUqOaIt+WFrOWFseWPguaVsCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUFJFRElDVF9NSVNTSU5HIjoi5Luj56CB6buY6K6kIiwiUkFORE9NX1NFRUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOiLku6PnoIHpu5jorqQiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJVU0VfQU1QIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVVNFX0xFQVJOQUJMRV9QRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoiYWN0aW9uIn0seyJjb25maWciOnsiQSI6MSwiQUJfUEFJUlMiOltbMSwxXSxbMiwzXV0sIkFMTE9XX1RGMzIiOnRydWUsIkFNUF9EVFlQRSI6ImJmbG9hdDE2IiwiQiI6MSwiQkFUQ0hfU0laRSI6NTEyLCJDT05EX0ZJWCI6bnVsbCwiQ09ORF9GSVhfU1RBUlQiOm51bGwsIkNPTkRfRklYX1NUQVJUX0ExIjpudWxsLCJDT05EX0ZJWF9TVEFSVF9BMiI6bnVsbCwiREFUQV9NT0RFIjpudWxsLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjEwMjQsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoyMDAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRU5UUk9QWV9QRU5BTFRZX1dFSUdIVCI6MC4wLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOnRydWUsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjo2LCJNQVhfVU5JUVVFX1JBVElPIjowLjcsIk1JU1NJTkdfUFJPQiI6MC4xLCJNSVNTX0xFTiI6MSwiTUlTU19TRUNPTkQiOmZhbHNlLCJNTFBfUkFUSU8iOjQsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjozMDAwMDAsIk5fSEVBRCI6NCwiTl9MQVlFUiI6NiwiT09EX0xFTiI6NjQsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJQIjoxMjcsIlBSRURJQ1RfTUlTU0lORyI6ZmFsc2UsIlJBTkRPTV9TRUVEIjoxNzk5NiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOmZhbHNlLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJhY3Rpb24iLCJUUkFJTl9MRU4iOjY0LCJVU0VfQU1QIjpmYWxzZSwiVVNFX0xFQVJOQUJMRV9QRSI6ZmFsc2UsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiLmlbDmja7pm4YgbG9zc19tYXNrIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6MjU2LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MzAwMDAwLCJibG9ja19zaXplIjoxMjgsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDogIHlgLzvvJtmbGFnIOWcqOavj+S4gOatpemAieaLqeinhOWImSIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJnZW5lcmF0ZWQgKyBmcmVzaF90ZXN0IiwiZXZhbF9iYXNlX2xlbmd0aCI6NjQsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxMjYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJhY3Rpb24g5L2/55So5pWw5o2u6ZuG55qE6YCQ5L2N572uIGxvc3NfbWFza++8m05VTV9NQVNLIOmFjee9ruS4jeWPguS4jiBsb3Nz44CCIiwi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5ZCO5L+d5a2Y5a6M5pW0IHJlc3VtZe+8jOaJp+ihjOWQjOmVv+W6piBjbGVhbiDor4TkvLDlubblr7zlh7rmm7Lnur/vvJvor4TkvLDkvb/nlKjljp8gR1BV77yM5LiN6K6h5YWl6K6t57uD5pe26ZmQ44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiIvZGF0YS9jeG0vZmluYWwvPGV4cGVyaW1lbnQ+X188dGltZXN0YW1wPi9jaGVja3BvaW50cy9tb2RlbC5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6W1sxLDFdLFsyLDNdXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjoiYWN0aW9uIOWKqOaAgeeUn+aIkO+8m+inhOWImemAkOatpemaj+acuumAieaLqe+8jOaXoOWbuuWumuavj+inhOWImeagt+acrOmFjeminSIsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTI2fSwibmFtZSI6ImFjdGlvbl9kMTAyNGw2cjRoNF9QMTI3X04yX3RyNjRfbWlzczAxbGVuMV9vbmxpbmUzMDBrX3NlZWQxNzk5Nl82aCIsIm9yaWdpbnMiOnsiQSI6Im1haW4g6buY6K6kIiwiQUJfUEFJUlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJBTExPV19URjMyIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQU1QX0RUWVBFIjoibWFpbiDpu5jorqQiLCJCIjoibWFpbiDpu5jorqQiLCJCQVRDSF9TSVpFIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQ09ORF9GSVgiOiJtYWluIOm7mOiupCIsIkNPTkRfRklYX1NUQVJUIjoibWFpbiDpu5jorqQiLCJDT05EX0ZJWF9TVEFSVF9BMSI6Im1haW4g6buY6K6kIiwiQ09ORF9GSVhfU1RBUlRfQTIiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6Im1haW4g6buY6K6kIiwiRFJPUE9VVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVOVFJPUFlfUEVOQUxUWV9XRUlHSFQiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkdSQURfQUNDVU1fU1RFUFMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1VOSVFVRV9SQVRJTyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU19MRU4iOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJNSVNTX1NFQ09ORCI6IueUqOaIt+WFrOWFseWPguaVsCIsIk1MUF9SQVRJTyI6IueUqOaIt+WFrOWFseWPguaVsCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUFJFRElDVF9NSVNTSU5HIjoi5Luj56CB6buY6K6kIiwiUkFORE9NX1NFRUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOiLku6PnoIHpu5jorqQiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJVU0VfQU1QIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVVNFX0xFQVJOQUJMRV9QRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoiYWN0aW9uIn0seyJjb25maWciOnsiQSI6MSwiQUJfUEFJUlMiOltbMSwxXSxbMiwzXV0sIkFMTE9XX1RGMzIiOnRydWUsIkFNUF9EVFlQRSI6ImJmbG9hdDE2IiwiQiI6MSwiQkFUQ0hfU0laRSI6NTEyLCJDT05EX0ZJWCI6bnVsbCwiQ09ORF9GSVhfU1RBUlQiOm51bGwsIkNPTkRfRklYX1NUQVJUX0ExIjpudWxsLCJDT05EX0ZJWF9TVEFSVF9BMiI6bnVsbCwiREFUQV9NT0RFIjpudWxsLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjEwMjQsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoyMDAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRU5UUk9QWV9QRU5BTFRZX1dFSUdIVCI6MC4wLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOnRydWUsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjo2LCJNQVhfVU5JUVVFX1JBVElPIjowLjcsIk1JU1NJTkdfUFJPQiI6MC4xLCJNSVNTX0xFTiI6MiwiTUlTU19TRUNPTkQiOmZhbHNlLCJNTFBfUkFUSU8iOjQsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjozMDAwMDAsIk5fSEVBRCI6NCwiTl9MQVlFUiI6MiwiT09EX0xFTiI6NjQsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJQIjoxMjcsIlBSRURJQ1RfTUlTU0lORyI6ZmFsc2UsIlJBTkRPTV9TRUVEIjoxNzk5NiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOmZhbHNlLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJhY3Rpb24iLCJUUkFJTl9MRU4iOjY0LCJVU0VfQU1QIjpmYWxzZSwiVVNFX0xFQVJOQUJMRV9QRSI6ZmFsc2UsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiLmlbDmja7pm4YgbG9zc19tYXNrIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6MjU2LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MzAwMDAwLCJibG9ja19zaXplIjoxMjgsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDogIHlgLzvvJtmbGFnIOWcqOavj+S4gOatpemAieaLqeinhOWImSIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJnZW5lcmF0ZWQgKyBmcmVzaF90ZXN0IiwiZXZhbF9iYXNlX2xlbmd0aCI6NjQsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxMjYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJhY3Rpb24g5L2/55So5pWw5o2u6ZuG55qE6YCQ5L2N572uIGxvc3NfbWFza++8m05VTV9NQVNLIOmFjee9ruS4jeWPguS4jiBsb3Nz44CCIiwi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5ZCO5L+d5a2Y5a6M5pW0IHJlc3VtZe+8jOaJp+ihjOWQjOmVv+W6piBjbGVhbiDor4TkvLDlubblr7zlh7rmm7Lnur/vvJvor4TkvLDkvb/nlKjljp8gR1BV77yM5LiN6K6h5YWl6K6t57uD5pe26ZmQ44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiIvZGF0YS9jeG0vZmluYWwvPGV4cGVyaW1lbnQ+X188dGltZXN0YW1wPi9jaGVja3BvaW50cy9tb2RlbC5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6W1sxLDFdLFsyLDNdXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjoiYWN0aW9uIOWKqOaAgeeUn+aIkO+8m+inhOWImemAkOatpemaj+acuumAieaLqe+8jOaXoOWbuuWumuavj+inhOWImeagt+acrOmFjeminSIsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTI2fSwibmFtZSI6ImFjdGlvbl9kMTAyNGwycjRoNF9QMTI3X04yX3RyNjRfbWlzczAxbGVuMl9vbmxpbmUzMDBrX3NlZWQxNzk5Nl82aCIsIm9yaWdpbnMiOnsiQSI6Im1haW4g6buY6K6kIiwiQUJfUEFJUlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJBTExPV19URjMyIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQU1QX0RUWVBFIjoibWFpbiDpu5jorqQiLCJCIjoibWFpbiDpu5jorqQiLCJCQVRDSF9TSVpFIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQ09ORF9GSVgiOiJtYWluIOm7mOiupCIsIkNPTkRfRklYX1NUQVJUIjoibWFpbiDpu5jorqQiLCJDT05EX0ZJWF9TVEFSVF9BMSI6Im1haW4g6buY6K6kIiwiQ09ORF9GSVhfU1RBUlRfQTIiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6Im1haW4g6buY6K6kIiwiRFJPUE9VVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVOVFJPUFlfUEVOQUxUWV9XRUlHSFQiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkdSQURfQUNDVU1fU1RFUFMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1VOSVFVRV9SQVRJTyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU19MRU4iOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJNSVNTX1NFQ09ORCI6IueUqOaIt+WFrOWFseWPguaVsCIsIk1MUF9SQVRJTyI6IueUqOaIt+WFrOWFseWPguaVsCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUFJFRElDVF9NSVNTSU5HIjoi5Luj56CB6buY6K6kIiwiUkFORE9NX1NFRUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOiLku6PnoIHpu5jorqQiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJVU0VfQU1QIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVVNFX0xFQVJOQUJMRV9QRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoiYWN0aW9uIn0seyJjb25maWciOnsiQSI6MSwiQUJfUEFJUlMiOltbMSwxXSxbMiwzXV0sIkFMTE9XX1RGMzIiOnRydWUsIkFNUF9EVFlQRSI6ImJmbG9hdDE2IiwiQiI6MSwiQkFUQ0hfU0laRSI6NTEyLCJDT05EX0ZJWCI6bnVsbCwiQ09ORF9GSVhfU1RBUlQiOm51bGwsIkNPTkRfRklYX1NUQVJUX0ExIjpudWxsLCJDT05EX0ZJWF9TVEFSVF9BMiI6bnVsbCwiREFUQV9NT0RFIjpudWxsLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjEwMjQsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoyMDAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRU5UUk9QWV9QRU5BTFRZX1dFSUdIVCI6MC4wLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOnRydWUsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjo2LCJNQVhfVU5JUVVFX1JBVElPIjowLjcsIk1JU1NJTkdfUFJPQiI6MC4xLCJNSVNTX0xFTiI6MiwiTUlTU19TRUNPTkQiOmZhbHNlLCJNTFBfUkFUSU8iOjQsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjozMDAwMDAsIk5fSEVBRCI6NCwiTl9MQVlFUiI6NiwiT09EX0xFTiI6NjQsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJQIjoxMjcsIlBSRURJQ1RfTUlTU0lORyI6ZmFsc2UsIlJBTkRPTV9TRUVEIjoxNzk5NiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOmZhbHNlLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJhY3Rpb24iLCJUUkFJTl9MRU4iOjY0LCJVU0VfQU1QIjpmYWxzZSwiVVNFX0xFQVJOQUJMRV9QRSI6ZmFsc2UsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiLmlbDmja7pm4YgbG9zc19tYXNrIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6MjU2LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MzAwMDAwLCJibG9ja19zaXplIjoxMjgsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDogIHlgLzvvJtmbGFnIOWcqOavj+S4gOatpemAieaLqeinhOWImSIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJnZW5lcmF0ZWQgKyBmcmVzaF90ZXN0IiwiZXZhbF9iYXNlX2xlbmd0aCI6NjQsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxMjYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJhY3Rpb24g5L2/55So5pWw5o2u6ZuG55qE6YCQ5L2N572uIGxvc3NfbWFza++8m05VTV9NQVNLIOmFjee9ruS4jeWPguS4jiBsb3Nz44CCIiwi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5ZCO5L+d5a2Y5a6M5pW0IHJlc3VtZe+8jOaJp+ihjOWQjOmVv+W6piBjbGVhbiDor4TkvLDlubblr7zlh7rmm7Lnur/vvJvor4TkvLDkvb/nlKjljp8gR1BV77yM5LiN6K6h5YWl6K6t57uD5pe26ZmQ44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiIvZGF0YS9jeG0vZmluYWwvPGV4cGVyaW1lbnQ+X188dGltZXN0YW1wPi9jaGVja3BvaW50cy9tb2RlbC5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6W1sxLDFdLFsyLDNdXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjoiYWN0aW9uIOWKqOaAgeeUn+aIkO+8m+inhOWImemAkOatpemaj+acuumAieaLqe+8jOaXoOWbuuWumuavj+inhOWImeagt+acrOmFjeminSIsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTI2fSwibmFtZSI6ImFjdGlvbl9kMTAyNGw2cjRoNF9QMTI3X04yX3RyNjRfbWlzczAxbGVuMl9vbmxpbmUzMDBrX3NlZWQxNzk5Nl82aCIsIm9yaWdpbnMiOnsiQSI6Im1haW4g6buY6K6kIiwiQUJfUEFJUlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJBTExPV19URjMyIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQU1QX0RUWVBFIjoibWFpbiDpu5jorqQiLCJCIjoibWFpbiDpu5jorqQiLCJCQVRDSF9TSVpFIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQ09ORF9GSVgiOiJtYWluIOm7mOiupCIsIkNPTkRfRklYX1NUQVJUIjoibWFpbiDpu5jorqQiLCJDT05EX0ZJWF9TVEFSVF9BMSI6Im1haW4g6buY6K6kIiwiQ09ORF9GSVhfU1RBUlRfQTIiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6Im1haW4g6buY6K6kIiwiRFJPUE9VVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVOVFJPUFlfUEVOQUxUWV9XRUlHSFQiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkdSQURfQUNDVU1fU1RFUFMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1VOSVFVRV9SQVRJTyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU19MRU4iOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJNSVNTX1NFQ09ORCI6IueUqOaIt+WFrOWFseWPguaVsCIsIk1MUF9SQVRJTyI6IueUqOaIt+WFrOWFseWPguaVsCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUFJFRElDVF9NSVNTSU5HIjoi5Luj56CB6buY6K6kIiwiUkFORE9NX1NFRUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOiLku6PnoIHpu5jorqQiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJVU0VfQU1QIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVVNFX0xFQVJOQUJMRV9QRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoiYWN0aW9uIn0seyJjb25maWciOnsiQSI6MSwiQUJfUEFJUlMiOltbMSwxXSxbMiwzXV0sIkFMTE9XX1RGMzIiOnRydWUsIkFNUF9EVFlQRSI6ImJmbG9hdDE2IiwiQiI6MSwiQkFUQ0hfU0laRSI6NTEyLCJDT05EX0ZJWCI6bnVsbCwiQ09ORF9GSVhfU1RBUlQiOm51bGwsIkNPTkRfRklYX1NUQVJUX0ExIjpudWxsLCJDT05EX0ZJWF9TVEFSVF9BMiI6bnVsbCwiREFUQV9NT0RFIjpudWxsLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjEwMjQsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoyMDAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRU5UUk9QWV9QRU5BTFRZX1dFSUdIVCI6MC4wLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOnRydWUsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjo2LCJNQVhfVU5JUVVFX1JBVElPIjowLjcsIk1JU1NJTkdfUFJPQiI6MC4xLCJNSVNTX0xFTiI6MywiTUlTU19TRUNPTkQiOmZhbHNlLCJNTFBfUkFUSU8iOjQsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjozMDAwMDAsIk5fSEVBRCI6NCwiTl9MQVlFUiI6MiwiT09EX0xFTiI6NjQsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJQIjoxMjcsIlBSRURJQ1RfTUlTU0lORyI6ZmFsc2UsIlJBTkRPTV9TRUVEIjoxNzk5NiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOmZhbHNlLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJhY3Rpb24iLCJUUkFJTl9MRU4iOjY0LCJVU0VfQU1QIjpmYWxzZSwiVVNFX0xFQVJOQUJMRV9QRSI6ZmFsc2UsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiLmlbDmja7pm4YgbG9zc19tYXNrIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6MjU2LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MzAwMDAwLCJibG9ja19zaXplIjoxMjgsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDogIHlgLzvvJtmbGFnIOWcqOavj+S4gOatpemAieaLqeinhOWImSIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJnZW5lcmF0ZWQgKyBmcmVzaF90ZXN0IiwiZXZhbF9iYXNlX2xlbmd0aCI6NjQsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxMjYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJhY3Rpb24g5L2/55So5pWw5o2u6ZuG55qE6YCQ5L2N572uIGxvc3NfbWFza++8m05VTV9NQVNLIOmFjee9ruS4jeWPguS4jiBsb3Nz44CCIiwi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5ZCO5L+d5a2Y5a6M5pW0IHJlc3VtZe+8jOaJp+ihjOWQjOmVv+W6piBjbGVhbiDor4TkvLDlubblr7zlh7rmm7Lnur/vvJvor4TkvLDkvb/nlKjljp8gR1BV77yM5LiN6K6h5YWl6K6t57uD5pe26ZmQ44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiIvZGF0YS9jeG0vZmluYWwvPGV4cGVyaW1lbnQ+X188dGltZXN0YW1wPi9jaGVja3BvaW50cy9tb2RlbC5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6W1sxLDFdLFsyLDNdXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjoiYWN0aW9uIOWKqOaAgeeUn+aIkO+8m+inhOWImemAkOatpemaj+acuumAieaLqe+8jOaXoOWbuuWumuavj+inhOWImeagt+acrOmFjeminSIsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTI2fSwibmFtZSI6ImFjdGlvbl9kMTAyNGwycjRoNF9QMTI3X04yX3RyNjRfbWlzczAxbGVuM19vbmxpbmUzMDBrX3NlZWQxNzk5Nl82aCIsIm9yaWdpbnMiOnsiQSI6Im1haW4g6buY6K6kIiwiQUJfUEFJUlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJBTExPV19URjMyIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQU1QX0RUWVBFIjoibWFpbiDpu5jorqQiLCJCIjoibWFpbiDpu5jorqQiLCJCQVRDSF9TSVpFIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQ09ORF9GSVgiOiJtYWluIOm7mOiupCIsIkNPTkRfRklYX1NUQVJUIjoibWFpbiDpu5jorqQiLCJDT05EX0ZJWF9TVEFSVF9BMSI6Im1haW4g6buY6K6kIiwiQ09ORF9GSVhfU1RBUlRfQTIiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6Im1haW4g6buY6K6kIiwiRFJPUE9VVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVOVFJPUFlfUEVOQUxUWV9XRUlHSFQiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkdSQURfQUNDVU1fU1RFUFMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1VOSVFVRV9SQVRJTyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU19MRU4iOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJNSVNTX1NFQ09ORCI6IueUqOaIt+WFrOWFseWPguaVsCIsIk1MUF9SQVRJTyI6IueUqOaIt+WFrOWFseWPguaVsCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUFJFRElDVF9NSVNTSU5HIjoi5Luj56CB6buY6K6kIiwiUkFORE9NX1NFRUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOiLku6PnoIHpu5jorqQiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJVU0VfQU1QIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVVNFX0xFQVJOQUJMRV9QRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoiYWN0aW9uIn0seyJjb25maWciOnsiQSI6MSwiQUJfUEFJUlMiOltbMSwxXSxbMiwzXV0sIkFMTE9XX1RGMzIiOnRydWUsIkFNUF9EVFlQRSI6ImJmbG9hdDE2IiwiQiI6MSwiQkFUQ0hfU0laRSI6NTEyLCJDT05EX0ZJWCI6bnVsbCwiQ09ORF9GSVhfU1RBUlQiOm51bGwsIkNPTkRfRklYX1NUQVJUX0ExIjpudWxsLCJDT05EX0ZJWF9TVEFSVF9BMiI6bnVsbCwiREFUQV9NT0RFIjpudWxsLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjEwMjQsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoyMDAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRU5UUk9QWV9QRU5BTFRZX1dFSUdIVCI6MC4wLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOnRydWUsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjo2LCJNQVhfVU5JUVVFX1JBVElPIjowLjcsIk1JU1NJTkdfUFJPQiI6MC4xLCJNSVNTX0xFTiI6MywiTUlTU19TRUNPTkQiOmZhbHNlLCJNTFBfUkFUSU8iOjQsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjozMDAwMDAsIk5fSEVBRCI6NCwiTl9MQVlFUiI6NiwiT09EX0xFTiI6NjQsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJQIjoxMjcsIlBSRURJQ1RfTUlTU0lORyI6ZmFsc2UsIlJBTkRPTV9TRUVEIjoxNzk5NiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOmZhbHNlLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJhY3Rpb24iLCJUUkFJTl9MRU4iOjY0LCJVU0VfQU1QIjpmYWxzZSwiVVNFX0xFQVJOQUJMRV9QRSI6ZmFsc2UsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiLmlbDmja7pm4YgbG9zc19tYXNrIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6MjU2LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MzAwMDAwLCJibG9ja19zaXplIjoxMjgsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDogIHlgLzvvJtmbGFnIOWcqOavj+S4gOatpemAieaLqeinhOWImSIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJnZW5lcmF0ZWQgKyBmcmVzaF90ZXN0IiwiZXZhbF9iYXNlX2xlbmd0aCI6NjQsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxMjYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJhY3Rpb24g5L2/55So5pWw5o2u6ZuG55qE6YCQ5L2N572uIGxvc3NfbWFza++8m05VTV9NQVNLIOmFjee9ruS4jeWPguS4jiBsb3Nz44CCIiwi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5ZCO5L+d5a2Y5a6M5pW0IHJlc3VtZe+8jOaJp+ihjOWQjOmVv+W6piBjbGVhbiDor4TkvLDlubblr7zlh7rmm7Lnur/vvJvor4TkvLDkvb/nlKjljp8gR1BV77yM5LiN6K6h5YWl6K6t57uD5pe26ZmQ44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiIvZGF0YS9jeG0vZmluYWwvPGV4cGVyaW1lbnQ+X188dGltZXN0YW1wPi9jaGVja3BvaW50cy9tb2RlbC5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6W1sxLDFdLFsyLDNdXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjoiYWN0aW9uIOWKqOaAgeeUn+aIkO+8m+inhOWImemAkOatpemaj+acuumAieaLqe+8jOaXoOWbuuWumuavj+inhOWImeagt+acrOmFjeminSIsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTI2fSwibmFtZSI6ImFjdGlvbl9kMTAyNGw2cjRoNF9QMTI3X04yX3RyNjRfbWlzczAxbGVuM19vbmxpbmUzMDBrX3NlZWQxNzk5Nl82aCIsIm9yaWdpbnMiOnsiQSI6Im1haW4g6buY6K6kIiwiQUJfUEFJUlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJBTExPV19URjMyIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQU1QX0RUWVBFIjoibWFpbiDpu5jorqQiLCJCIjoibWFpbiDpu5jorqQiLCJCQVRDSF9TSVpFIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQ09ORF9GSVgiOiJtYWluIOm7mOiupCIsIkNPTkRfRklYX1NUQVJUIjoibWFpbiDpu5jorqQiLCJDT05EX0ZJWF9TVEFSVF9BMSI6Im1haW4g6buY6K6kIiwiQ09ORF9GSVhfU1RBUlRfQTIiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6Im1haW4g6buY6K6kIiwiRFJPUE9VVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVOVFJPUFlfUEVOQUxUWV9XRUlHSFQiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkdSQURfQUNDVU1fU1RFUFMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1VOSVFVRV9SQVRJTyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU19MRU4iOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJNSVNTX1NFQ09ORCI6IueUqOaIt+WFrOWFseWPguaVsCIsIk1MUF9SQVRJTyI6IueUqOaIt+WFrOWFseWPguaVsCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUFJFRElDVF9NSVNTSU5HIjoi5Luj56CB6buY6K6kIiwiUkFORE9NX1NFRUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOiLku6PnoIHpu5jorqQiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJVU0VfQU1QIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVVNFX0xFQVJOQUJMRV9QRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoiYWN0aW9uIn0seyJjb25maWciOnsiQSI6MSwiQUJfUEFJUlMiOltbMSwxXSxbMiwzXV0sIkFMTE9XX1RGMzIiOnRydWUsIkFNUF9EVFlQRSI6ImJmbG9hdDE2IiwiQiI6MSwiQkFUQ0hfU0laRSI6NTEyLCJDT05EX0ZJWCI6bnVsbCwiQ09ORF9GSVhfU1RBUlQiOm51bGwsIkNPTkRfRklYX1NUQVJUX0ExIjpudWxsLCJDT05EX0ZJWF9TVEFSVF9BMiI6bnVsbCwiREFUQV9NT0RFIjpudWxsLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjEwMjQsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoyMDAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRU5UUk9QWV9QRU5BTFRZX1dFSUdIVCI6MC4wLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOnRydWUsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjo2LCJNQVhfVU5JUVVFX1JBVElPIjowLjcsIk1JU1NJTkdfUFJPQiI6MC4xLCJNSVNTX0xFTiI6NSwiTUlTU19TRUNPTkQiOmZhbHNlLCJNTFBfUkFUSU8iOjQsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjozMDAwMDAsIk5fSEVBRCI6NCwiTl9MQVlFUiI6MiwiT09EX0xFTiI6NjQsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJQIjoxMjcsIlBSRURJQ1RfTUlTU0lORyI6ZmFsc2UsIlJBTkRPTV9TRUVEIjoxNzk5NiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOmZhbHNlLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJhY3Rpb24iLCJUUkFJTl9MRU4iOjY0LCJVU0VfQU1QIjpmYWxzZSwiVVNFX0xFQVJOQUJMRV9QRSI6ZmFsc2UsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiLmlbDmja7pm4YgbG9zc19tYXNrIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6MjU2LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MzAwMDAwLCJibG9ja19zaXplIjoxMjgsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDogIHlgLzvvJtmbGFnIOWcqOavj+S4gOatpemAieaLqeinhOWImSIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJnZW5lcmF0ZWQgKyBmcmVzaF90ZXN0IiwiZXZhbF9iYXNlX2xlbmd0aCI6NjQsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxMjYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJhY3Rpb24g5L2/55So5pWw5o2u6ZuG55qE6YCQ5L2N572uIGxvc3NfbWFza++8m05VTV9NQVNLIOmFjee9ruS4jeWPguS4jiBsb3Nz44CCIiwi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5ZCO5L+d5a2Y5a6M5pW0IHJlc3VtZe+8jOaJp+ihjOWQjOmVv+W6piBjbGVhbiDor4TkvLDlubblr7zlh7rmm7Lnur/vvJvor4TkvLDkvb/nlKjljp8gR1BV77yM5LiN6K6h5YWl6K6t57uD5pe26ZmQ44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiIvZGF0YS9jeG0vZmluYWwvPGV4cGVyaW1lbnQ+X188dGltZXN0YW1wPi9jaGVja3BvaW50cy9tb2RlbC5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6W1sxLDFdLFsyLDNdXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjoiYWN0aW9uIOWKqOaAgeeUn+aIkO+8m+inhOWImemAkOatpemaj+acuumAieaLqe+8jOaXoOWbuuWumuavj+inhOWImeagt+acrOmFjeminSIsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTI2fSwibmFtZSI6ImFjdGlvbl9kMTAyNGwycjRoNF9QMTI3X04yX3RyNjRfbWlzczAxbGVuNV9vbmxpbmUzMDBrX3NlZWQxNzk5Nl82aCIsIm9yaWdpbnMiOnsiQSI6Im1haW4g6buY6K6kIiwiQUJfUEFJUlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJBTExPV19URjMyIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQU1QX0RUWVBFIjoibWFpbiDpu5jorqQiLCJCIjoibWFpbiDpu5jorqQiLCJCQVRDSF9TSVpFIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQ09ORF9GSVgiOiJtYWluIOm7mOiupCIsIkNPTkRfRklYX1NUQVJUIjoibWFpbiDpu5jorqQiLCJDT05EX0ZJWF9TVEFSVF9BMSI6Im1haW4g6buY6K6kIiwiQ09ORF9GSVhfU1RBUlRfQTIiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6Im1haW4g6buY6K6kIiwiRFJPUE9VVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVOVFJPUFlfUEVOQUxUWV9XRUlHSFQiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkdSQURfQUNDVU1fU1RFUFMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1VOSVFVRV9SQVRJTyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU19MRU4iOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJNSVNTX1NFQ09ORCI6IueUqOaIt+WFrOWFseWPguaVsCIsIk1MUF9SQVRJTyI6IueUqOaIt+WFrOWFseWPguaVsCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUFJFRElDVF9NSVNTSU5HIjoi5Luj56CB6buY6K6kIiwiUkFORE9NX1NFRUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOiLku6PnoIHpu5jorqQiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJVU0VfQU1QIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVVNFX0xFQVJOQUJMRV9QRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoiYWN0aW9uIn0seyJjb25maWciOnsiQSI6MSwiQUJfUEFJUlMiOltbMSwxXSxbMiwzXV0sIkFMTE9XX1RGMzIiOnRydWUsIkFNUF9EVFlQRSI6ImJmbG9hdDE2IiwiQiI6MSwiQkFUQ0hfU0laRSI6NTEyLCJDT05EX0ZJWCI6bnVsbCwiQ09ORF9GSVhfU1RBUlQiOm51bGwsIkNPTkRfRklYX1NUQVJUX0ExIjpudWxsLCJDT05EX0ZJWF9TVEFSVF9BMiI6bnVsbCwiREFUQV9NT0RFIjpudWxsLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjEwMjQsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoyMDAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRU5UUk9QWV9QRU5BTFRZX1dFSUdIVCI6MC4wLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOnRydWUsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjo2LCJNQVhfVU5JUVVFX1JBVElPIjowLjcsIk1JU1NJTkdfUFJPQiI6MC4xLCJNSVNTX0xFTiI6OCwiTUlTU19TRUNPTkQiOmZhbHNlLCJNTFBfUkFUSU8iOjQsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjozMDAwMDAsIk5fSEVBRCI6NCwiTl9MQVlFUiI6MiwiT09EX0xFTiI6NjQsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJQIjoxMjcsIlBSRURJQ1RfTUlTU0lORyI6ZmFsc2UsIlJBTkRPTV9TRUVEIjoxNzk5NiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOmZhbHNlLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJhY3Rpb24iLCJUUkFJTl9MRU4iOjY0LCJVU0VfQU1QIjpmYWxzZSwiVVNFX0xFQVJOQUJMRV9QRSI6ZmFsc2UsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiLmlbDmja7pm4YgbG9zc19tYXNrIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6MjU2LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MzAwMDAwLCJibG9ja19zaXplIjoxMjgsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDogIHlgLzvvJtmbGFnIOWcqOavj+S4gOatpemAieaLqeinhOWImSIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJnZW5lcmF0ZWQgKyBmcmVzaF90ZXN0IiwiZXZhbF9iYXNlX2xlbmd0aCI6NjQsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxMjYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJhY3Rpb24g5L2/55So5pWw5o2u6ZuG55qE6YCQ5L2N572uIGxvc3NfbWFza++8m05VTV9NQVNLIOmFjee9ruS4jeWPguS4jiBsb3Nz44CCIiwi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5ZCO5L+d5a2Y5a6M5pW0IHJlc3VtZe+8jOaJp+ihjOWQjOmVv+W6piBjbGVhbiDor4TkvLDlubblr7zlh7rmm7Lnur/vvJvor4TkvLDkvb/nlKjljp8gR1BV77yM5LiN6K6h5YWl6K6t57uD5pe26ZmQ44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiIvZGF0YS9jeG0vZmluYWwvPGV4cGVyaW1lbnQ+X188dGltZXN0YW1wPi9jaGVja3BvaW50cy9tb2RlbC5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6W1sxLDFdLFsyLDNdXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjoiYWN0aW9uIOWKqOaAgeeUn+aIkO+8m+inhOWImemAkOatpemaj+acuumAieaLqe+8jOaXoOWbuuWumuavj+inhOWImeagt+acrOmFjeminSIsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTI2fSwibmFtZSI6ImFjdGlvbl9kMTAyNGwycjRoNF9QMTI3X04yX3RyNjRfbWlzczAxbGVuOF9vbmxpbmUzMDBrX3NlZWQxNzk5Nl82aCIsIm9yaWdpbnMiOnsiQSI6Im1haW4g6buY6K6kIiwiQUJfUEFJUlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJBTExPV19URjMyIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQU1QX0RUWVBFIjoibWFpbiDpu5jorqQiLCJCIjoibWFpbiDpu5jorqQiLCJCQVRDSF9TSVpFIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQ09ORF9GSVgiOiJtYWluIOm7mOiupCIsIkNPTkRfRklYX1NUQVJUIjoibWFpbiDpu5jorqQiLCJDT05EX0ZJWF9TVEFSVF9BMSI6Im1haW4g6buY6K6kIiwiQ09ORF9GSVhfU1RBUlRfQTIiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6Im1haW4g6buY6K6kIiwiRFJPUE9VVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVOVFJPUFlfUEVOQUxUWV9XRUlHSFQiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkdSQURfQUNDVU1fU1RFUFMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1VOSVFVRV9SQVRJTyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU19MRU4iOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJNSVNTX1NFQ09ORCI6IueUqOaIt+WFrOWFseWPguaVsCIsIk1MUF9SQVRJTyI6IueUqOaIt+WFrOWFseWPguaVsCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUFJFRElDVF9NSVNTSU5HIjoi5Luj56CB6buY6K6kIiwiUkFORE9NX1NFRUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOiLku6PnoIHpu5jorqQiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJVU0VfQU1QIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVVNFX0xFQVJOQUJMRV9QRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoiYWN0aW9uIn0seyJjb25maWciOnsiQSI6MSwiQUJfUEFJUlMiOltbMSwxXSxbMiwzXV0sIkFMTE9XX1RGMzIiOnRydWUsIkFNUF9EVFlQRSI6ImJmbG9hdDE2IiwiQiI6MSwiQkFUQ0hfU0laRSI6NTEyLCJDT05EX0ZJWCI6bnVsbCwiQ09ORF9GSVhfU1RBUlQiOm51bGwsIkNPTkRfRklYX1NUQVJUX0ExIjpudWxsLCJDT05EX0ZJWF9TVEFSVF9BMiI6bnVsbCwiREFUQV9NT0RFIjpudWxsLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjEwMjQsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoyMDAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRU5UUk9QWV9QRU5BTFRZX1dFSUdIVCI6MC4wLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOnRydWUsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjo2LCJNQVhfVU5JUVVFX1JBVElPIjowLjcsIk1JU1NJTkdfUFJPQiI6MC4xLCJNSVNTX0xFTiI6OCwiTUlTU19TRUNPTkQiOmZhbHNlLCJNTFBfUkFUSU8iOjQsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjozMDAwMDAsIk5fSEVBRCI6NCwiTl9MQVlFUiI6NCwiT09EX0xFTiI6NjQsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJQIjoxMjcsIlBSRURJQ1RfTUlTU0lORyI6ZmFsc2UsIlJBTkRPTV9TRUVEIjoxNzk5NiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOmZhbHNlLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJhY3Rpb24iLCJUUkFJTl9MRU4iOjY0LCJVU0VfQU1QIjpmYWxzZSwiVVNFX0xFQVJOQUJMRV9QRSI6ZmFsc2UsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiLmlbDmja7pm4YgbG9zc19tYXNrIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6MjU2LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MzAwMDAwLCJibG9ja19zaXplIjoxMjgsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDogIHlgLzvvJtmbGFnIOWcqOavj+S4gOatpemAieaLqeinhOWImSIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJnZW5lcmF0ZWQgKyBmcmVzaF90ZXN0IiwiZXZhbF9iYXNlX2xlbmd0aCI6NjQsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxMjYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJhY3Rpb24g5L2/55So5pWw5o2u6ZuG55qE6YCQ5L2N572uIGxvc3NfbWFza++8m05VTV9NQVNLIOmFjee9ruS4jeWPguS4jiBsb3Nz44CCIiwi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5ZCO5L+d5a2Y5a6M5pW0IHJlc3VtZe+8jOaJp+ihjOWQjOmVv+W6piBjbGVhbiDor4TkvLDlubblr7zlh7rmm7Lnur/vvJvor4TkvLDkvb/nlKjljp8gR1BV77yM5LiN6K6h5YWl6K6t57uD5pe26ZmQ44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiIvZGF0YS9jeG0vZmluYWwvPGV4cGVyaW1lbnQ+X188dGltZXN0YW1wPi9jaGVja3BvaW50cy9tb2RlbC5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6W1sxLDFdLFsyLDNdXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjoiYWN0aW9uIOWKqOaAgeeUn+aIkO+8m+inhOWImemAkOatpemaj+acuumAieaLqe+8jOaXoOWbuuWumuavj+inhOWImeagt+acrOmFjeminSIsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTI2fSwibmFtZSI6ImFjdGlvbl9kMTAyNGw0cjRoNF9QMTI3X04yX3RyNjRfbWlzczAxbGVuOF9vbmxpbmUzMDBrX3NlZWQxNzk5Nl82aCIsIm9yaWdpbnMiOnsiQSI6Im1haW4g6buY6K6kIiwiQUJfUEFJUlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJBTExPV19URjMyIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQU1QX0RUWVBFIjoibWFpbiDpu5jorqQiLCJCIjoibWFpbiDpu5jorqQiLCJCQVRDSF9TSVpFIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQ09ORF9GSVgiOiJtYWluIOm7mOiupCIsIkNPTkRfRklYX1NUQVJUIjoibWFpbiDpu5jorqQiLCJDT05EX0ZJWF9TVEFSVF9BMSI6Im1haW4g6buY6K6kIiwiQ09ORF9GSVhfU1RBUlRfQTIiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6Im1haW4g6buY6K6kIiwiRFJPUE9VVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVOVFJPUFlfUEVOQUxUWV9XRUlHSFQiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkdSQURfQUNDVU1fU1RFUFMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1VOSVFVRV9SQVRJTyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU19MRU4iOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJNSVNTX1NFQ09ORCI6IueUqOaIt+WFrOWFseWPguaVsCIsIk1MUF9SQVRJTyI6IueUqOaIt+WFrOWFseWPguaVsCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUFJFRElDVF9NSVNTSU5HIjoi5Luj56CB6buY6K6kIiwiUkFORE9NX1NFRUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOiLku6PnoIHpu5jorqQiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJVU0VfQU1QIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVVNFX0xFQVJOQUJMRV9QRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoiYWN0aW9uIn0seyJjb25maWciOnsiQSI6MSwiQUJfUEFJUlMiOltbMSwxXSxbMiwzXV0sIkFMTE9XX1RGMzIiOnRydWUsIkFNUF9EVFlQRSI6ImJmbG9hdDE2IiwiQiI6MSwiQkFUQ0hfU0laRSI6NTEyLCJDT05EX0ZJWCI6bnVsbCwiQ09ORF9GSVhfU1RBUlQiOm51bGwsIkNPTkRfRklYX1NUQVJUX0ExIjpudWxsLCJDT05EX0ZJWF9TVEFSVF9BMiI6bnVsbCwiREFUQV9NT0RFIjpudWxsLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjEwMjQsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoyMDAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRU5UUk9QWV9QRU5BTFRZX1dFSUdIVCI6MC4wLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOnRydWUsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjo2LCJNQVhfVU5JUVVFX1JBVElPIjowLjcsIk1JU1NJTkdfUFJPQiI6MC4xLCJNSVNTX0xFTiI6OCwiTUlTU19TRUNPTkQiOmZhbHNlLCJNTFBfUkFUSU8iOjQsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjozMDAwMDAsIk5fSEVBRCI6NCwiTl9MQVlFUiI6NiwiT09EX0xFTiI6NjQsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJQIjoxMjcsIlBSRURJQ1RfTUlTU0lORyI6ZmFsc2UsIlJBTkRPTV9TRUVEIjoxNzk5NiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOmZhbHNlLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJhY3Rpb24iLCJUUkFJTl9MRU4iOjY0LCJVU0VfQU1QIjpmYWxzZSwiVVNFX0xFQVJOQUJMRV9QRSI6ZmFsc2UsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiLmlbDmja7pm4YgbG9zc19tYXNrIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6MjU2LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MzAwMDAwLCJibG9ja19zaXplIjoxMjgsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDogIHlgLzvvJtmbGFnIOWcqOavj+S4gOatpemAieaLqeinhOWImSIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJnZW5lcmF0ZWQgKyBmcmVzaF90ZXN0IiwiZXZhbF9iYXNlX2xlbmd0aCI6NjQsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxMjYsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJhY3Rpb24g5L2/55So5pWw5o2u6ZuG55qE6YCQ5L2N572uIGxvc3NfbWFza++8m05VTV9NQVNLIOmFjee9ruS4jeWPguS4jiBsb3Nz44CCIiwi5q+P5p2h5a6e6aqM54us56uL6K6h5pe277yM5LuO6K6t57uD5b6q546v5byA5aeL77yb5q+PIDEgZXBvY2gg55qE6K+E5Lyw54K55qOA5p+l77yM5Yiw5pe25L+d5a2Y5a6M5pW05a2Y5qGj6YCA5Ye677yM5Y+v6IO96LaF6L+H5ZCN5LmJ5pe26ZmQ77yb5LiN5pS55Y+YIGNvc2luZSBob3Jpem9u44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5ZCO5L+d5a2Y5a6M5pW0IHJlc3VtZe+8jOaJp+ihjOWQjOmVv+W6piBjbGVhbiDor4TkvLDlubblr7zlh7rmm7Lnur/vvJvor4TkvLDkvb/nlKjljp8gR1BV77yM5LiN6K6h5YWl6K6t57uD5pe26ZmQ44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiIvZGF0YS9jeG0vZmluYWwvPGV4cGVyaW1lbnQ+X188dGltZXN0YW1wPi9jaGVja3BvaW50cy9tb2RlbC5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6W1sxLDFdLFsyLDNdXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjoiYWN0aW9uIOWKqOaAgeeUn+aIkO+8m+inhOWImemAkOatpemaj+acuumAieaLqe+8jOaXoOWbuuWumuavj+inhOWImeagt+acrOmFjeminSIsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTI2fSwibmFtZSI6ImFjdGlvbl9kMTAyNGw2cjRoNF9QMTI3X04yX3RyNjRfbWlzczAxbGVuOF9vbmxpbmUzMDBrX3NlZWQxNzk5Nl82aCIsIm9yaWdpbnMiOnsiQSI6Im1haW4g6buY6K6kIiwiQUJfUEFJUlMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJBTExPV19URjMyIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQU1QX0RUWVBFIjoibWFpbiDpu5jorqQiLCJCIjoibWFpbiDpu5jorqQiLCJCQVRDSF9TSVpFIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiQ09ORF9GSVgiOiJtYWluIOm7mOiupCIsIkNPTkRfRklYX1NUQVJUIjoibWFpbiDpu5jorqQiLCJDT05EX0ZJWF9TVEFSVF9BMSI6Im1haW4g6buY6K6kIiwiQ09ORF9GSVhfU1RBUlRfQTIiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6Im1haW4g6buY6K6kIiwiRFJPUE9VVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVOVFJPUFlfUEVOQUxUWV9XRUlHSFQiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkdSQURfQUNDVU1fU1RFUFMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJHUkFEX0NMSVBfTk9STSI6IueUqOaIt+WFrOWFseWPguaVsCIsIklOSVRfRlJPTSI6IueUqOaIt+WFrOWFseWPguaVsCIsIkxSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1RSQUlOX0hPVVJTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUFYX1VOSVFVRV9SQVRJTyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTUlTU19MRU4iOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJNSVNTX1NFQ09ORCI6IueUqOaIt+WFrOWFseWPguaVsCIsIk1MUF9SQVRJTyI6IueUqOaIt+WFrOWFseWPguaVsCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0hFQUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJOX0xBWUVSIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiT09EX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUFJFRElDVF9NSVNTSU5HIjoi5Luj56CB6buY6K6kIiwiUkFORE9NX1NFRUQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJFU1VNRV9GUk9NIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiU0tJUF9GSU5BTF9HRU5FUkFUSU9OX1RFU1QiOiLku6PnoIHpu5jorqQiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJVU0VfQU1QIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVVNFX0xFQVJOQUJMRV9QRSI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6IueUqOaIt+WFrOWFseWPguaVsCJ9LCJ0YXNrIjoiYWN0aW9uIn1dLCJsb2NhbF9ncHVfYWxsb3dsaXN0IjpbMCwxLDIsM10sInBhY2thZ2luZ19ob3N0IjoicHJvajI2MDMtdHIyLTAiLCJzb3VyY2VfZmlsZXMiOnsicmVxdWlyZW1lbnRzLnR4dCI6IjljMWMwMTgzZmExYWRhYjExN2Y4MmFkYTAxZjBhYTM0YWZhYWQ2MzI5ZWZjNTJkYTRkMjhjYjI0YjU1YjA4NzYiLCJzY3JpcHRzL2FjdGlvbl9ydW5fcG9zdHByb2Nlc3MucHkiOiJkNTQyZTA2ZGU5ZjE2NTRmZjNhYjVhYjE0NGJhMmRiYmQwMWMwMWRkNGQ4NTk2OWM4OGFmMjBlZGIxNGE3ZWZhIiwic2NyaXB0cy9leHBlcmltZW50X3BhY2thZ2VfcnVubmVyLnB5IjoiMGJhMTQ3ZjhhNzY3NjU4NzkwNWM0MzEzMDkyZjFmOTllMWUyYjQ2ZTllOWZjODc4NzEzMWY4ZGJmYTcyM2YzYiIsInNjcmlwdHMvcmVtb3RlX2V4cGVyaW1lbnRfcnVubmVyLnB5IjoiMWY5NjI3NWI5MGQyNTM3OTUxZmNkNjMxNmQzNmU0OThiNmExNDVhMGQ5ODI5MGRjYzI5ZDhmZWRiMzc4NTY0YSIsInNyYy9hY3RpdmF0aW9uX3BhdGNoLnB5IjoiMWE2YTU5ODAxNTkyODI4N2Q1MjU5ZTI0NzQwN2Q0OTRkMDI5NWRmNjZjYjczN2M3YzU0YzFjNGNiZjhiYzdiMiIsInNyYy9hbmFseXplX2F0dGVudGlvbi5weSI6IjBhYzg1YjNhNjk3MzQ3NDhiMzBmZmY1YzYxNTA4MzU1OWI1ZGUxMTdhYzQ5ODg2MWFlZWU2ZTE1MzQ3NmU0M2MiLCJzcmMvYmF0Y2hfcnVuLnB5IjoiM2I2ZWQzMzRjOWE1ZjRkMDFlYWVjOTcwYmY3NGM4ZGEwODVjMTA3M2Y3YzE2YmVkYWQ1ZjVjYWIzZTczNTllMyIsInNyYy9jaGFpbl9ydW4ucHkiOiIwNTk2NWFlMzM1OGY2MWY1MGNhOTk1ZjJjNmU3ZDhhNThjNTAzZWRhZDgyMmZmMmUwN2MzZTZmNjFkOTAwNGUzIiwic3JjL2NvbmZpZy5qc29uIjoiM2Q0MjUwNTFlNTEzODhiOWMxNmY3OWNjNjAwZjQwN2I1NzA2M2ZiZDUzMTNkNGVlYTdkMTYxODM5YjlmNjhjMSIsInNyYy9jb3JlLnB5IjoiNzBhYTJiY2M3ZmQ2NTBlOWI3YjZiM2IwOGI5MTk5OTg0Yjg4YzdhNzNiOTYwOWQ1YTFjYWNlMzQwYjI0MTA2NSIsInNyYy9jdXJyaWN1bHVtX2NoYWluX050YXNrLnB5IjoiMDBhNDRjNjYwMTY1ZThhM2FmYjk0MjE3ZWY3YzcxMTMwNTgzMGM2NjY2MzkwY2IwMzFhMzFmYzk2MTQxMTk4OCIsInNyYy9jdXJyaWN1bHVtX2NoYWluX3JldmVyc2UucHkiOiI4YmQ4ODIzZmU3ZWVlNTU4ZjQzNDgwN2NkZmRmYTdkNGRkNWE1MmE1Nzc2NTUzYjE2NDlkYmE2ZTc0YWNhN2VjIiwic3JjL2RhdGFzZXRzLnB5IjoiZGI3NTY3MmVjZWQ2NGU4ZWE5ODhmZjk4MjZiZjFhOGQ1MmQ1YTNhZTUzNzY1N2Y5MWMwYmQ4YjJlZDFkZThhNCIsInNyYy9lcnJvcl9icmVha2Rvd24ucHkiOiI3N2NhNzJjMzlhYWQ0NmUwOWVjYjVlOGQwYjMwZDZmYWZkYWFlYzE4NWZiN2ZjYTA4MWIzMTJlMWVmZDg4Zjk3Iiwic3JjL2V2aWRlbmNlX3N3ZWVwLnB5IjoiMzU1MDI1NzMxYTg3NDQyM2JlNzEwZWU0YjJhNWY0NGQ0MDA4OGFjNGVmYjE0YjliMjRjMGQyM2ZiOGJlMWI0OSIsInNyYy9leHBlcmltZW50LnB5IjoiNGYwZTlkNjQ2NDAxMzM3NWU5YzNhNmM3NzhkZTI3Y2NlZWY0OGM3YzBkNzQxMzkxOGQ2NjYwY2EwYmNmNzkwZiIsInNyYy9maW5hbF9ldmFsLnB5IjoiNzJiZGNmMWYzNjMzOTViZTA1YmEyMTcyNGRlM2Q5MjZjOWQxMWM4N2VmNDIwYWFkMjIzNDc2NGZlZTAxZTA5NCIsInNyYy9maXhfbG9ncy5weSI6IjgzODNmOWMxN2YyZmFkOTNhNGFmOGUzMDQ1NjlmMGMwZWNjZWRhMzdjZTk1Y2FkMzc3YWFiOGJhZjBkMjA4Y2EiLCJzcmMvaGVhZF9hYmxhdGlvbi5weSI6ImQwMzg0NjEwMGQzZDlhOTc2NzFiOTJjNTFjYzI3NjFkNjgyMTFjOTRmMDBjNWQ2YTVjODdhNDFiZDE2MTY0NWYiLCJzcmMvbW9kZWxzLnB5IjoiZTEzMTM1ZTdmOGFhZjYzM2FlY2JlZjQ1ZmU4ODViZmFlYmNmMGI1Y2MxMThjYTBiMDgxNDE5ZjVhYjczYmE3YyIsInNyYy9tdWx0aV9wb3NfcGF0Y2gucHkiOiJlNTQzMzQxMTMzYmJmNzM5OTg4MjA3ZTEwODVjYWRlZTE5N2E2NmVlZWY1MzgwNjE3NGU0NTA5NmQxM2FjZTljIiwic3JjL3ByZXBhcmVfcmVydW4ucHkiOiI3ZmNhZDA3NWE4YTk5NjRhODNjNzdhNDNlMWZmYTI5MmEyMDY0YzdmMTU3NmI4ZTEyNDVkZWZlOTA1YTk3Y2Y0Iiwic3JjL3Byb3RvY29sLnB5IjoiZGFmYTU4Y2RiYWZhOTQ2ZGY1ZTlkZGFiNzhhMzQyNzc0MzU3NWQyZWM2MjliNTNkYzhjOGNmNTRiODIyM2U3MiIsInNyYy9yZXBvcnRfZnJvbnRfYmFja19hdHRlbnRpb24ucHkiOiI1YzczZmYyNjJkMmQ4ZDllOGRlZjdjYzAxYzFmODI4Zjk0MzE1MmNlYjg2ZjNjYjdhZjM0YjhlYzM5MjE1ZDY1Iiwic3JjL3J1bGVfZml0LnB5IjoiNzIyMTZkNjcwY2ZmNzI2NjkyZTczYTE0NGNiYzViZTMyMjM5Mjg5OTdmNTZiMzc4ZGMxMTJhYTViYTc3MDIxMCIsInNyYy9ydWxlX2dlbmVyYWxpemF0aW9uLnB5IjoiNGIxOTI3Zjg0MjVlYzdmODgwZjQ3ZmIzMTQ0MTQxYjFmZGI2YjdhMzcxNmVmOTBmNzhhMTk2MmMyZGIzMzllMSIsInNyYy9ydWxlcy5weSI6IjE5NGEwMmVjY2MwMTI1OWQ2NTNhNDllZjY2NDE0NTlmYThhODJjMmNjZWQ1MmIwNWU0YmQ2ZDZhNjIzNGQ5ZDkiLCJzcmMvc2FtcGxlX2dlbi5weSI6IjgzOTNiMzNmOTBhZWI3NjBmZjZhNTZiZjhjZmE4NTgyMzZjYjJkMTNlYjU0OGE5OGVmZDBmNjg1MmJlMjUzZDEiLCJzcmMvdHJhaW5pbmcucHkiOiI3ODE0NzVmM2VjYmE1YmIwODVmMTBhNDAwZmM1ZjFmNDNkZDdkZDk3OGFhODIzOGJiNWQ4NTMxZGJmYmExZjQ3Iiwic3JjL3ZlcmlmeV9jaXJjbGUucHkiOiIzOTAzZmM2MWI0MDMyYWM4NDEyMTY2YzBmMGNiM2JiOWRhODg2NDVkNWY1OGM4YTM2MzY4NzNjNDcyMWVjODAxIiwic3JjL3ZlcmlmeV9zYW1wbGUucHkiOiJlYmM3YjIyYTViNjZhMDVhNTJhNzZjZWRhNTE3NWQ2YzJkZDY0ZTViN2I1NGY5ZmQxZDRlMjNlMDBhMWJlZjkyIiwic3JjL3Zpc3VhbGl6ZS5weSI6IjEyYTQ1M2FlOTU3NjE1NTUxZmFjYWI1MTZhYWIyN2NhNWViZWEyZDRkMzY5NzMzZGE4NzlhYzcyM2U3Y2FhMjcifSwidW5yZXNvbHZlZCI6W10sInZlcnNpb24iOjF9 -->
