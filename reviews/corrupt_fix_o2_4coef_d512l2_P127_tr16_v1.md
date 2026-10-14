# 实验参数确认表

审阅版本：5e479e62b44fc255d69f
批次：corrupt_fix_single_o2_4coef_d512l2_P127_tr16；最大并发：4
此阶段只审阅参数，尚未生成可执行实验 JSON 或启动实验。

| 实验 | 任务 | 实际训练样本 | Seed | LR / WD / clip | 初始化 | 每条定时(h) |
| --- | --- | --- | --- | --- | --- | --- |
| single_o2_corruptfix_a93b69_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |
| single_o2_corruptfix_a22b86_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |
| single_o2_corruptfix_a98b19_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |
| single_o2_corruptfix_a103b61_d512l2r8h1_P127_tr16_seed42 | single | 11290 | 42 | 0.0003 / 1 / 1.0 | fresh | 不限 |

## single_o2_corruptfix_a93b69_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
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
| RULES | [[93,69]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "corrupt_fix" | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | 4839 |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 18 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [93,69] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 18 |
| vocab_size | 145 |

- corrupt_fix：loss_mask 由 collate 给出（仅位置/值两个输出位），NUM_MASK 不参与。
- corrupt_fix：污损是任务内生的（恰好一位改成随机错误值），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 位置 token + 正确值两个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_corruptfix_a22b86_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
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
| RULES | [[22,86]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "corrupt_fix" | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | 4839 |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 18 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [22,86] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 18 |
| vocab_size | 145 |

- corrupt_fix：loss_mask 由 collate 给出（仅位置/值两个输出位），NUM_MASK 不参与。
- corrupt_fix：污损是任务内生的（恰好一位改成随机错误值），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 位置 token + 正确值两个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_corruptfix_a98b19_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
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
| RULES | [[98,19]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "corrupt_fix" | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | 4839 |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 18 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [98,19] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 18 |
| vocab_size | 145 |

- corrupt_fix：loss_mask 由 collate 给出（仅位置/值两个输出位），NUM_MASK 不参与。
- corrupt_fix：污损是任务内生的（恰好一位改成随机错误值），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 位置 token + 正确值两个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

## single_o2_corruptfix_a103b61_d512l2r8h1_P127_tr16_seed42

| 参数 | 实际值 | 来源 |
| --- | --- | --- |
| BATCH_SIZE | 512 | main 默认 |
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
| RULES | [[103,61]] | 用户逐实验参数 |
| STATE_SPACE_CAP | null | main 默认 |
| TASK | "single" | 实验任务 |
| TASK_VARIANT | "corrupt_fix" | 用户公共参数 |
| TRAIN_LEN | 16 | 用户公共参数 |
| WEIGHT_DECAY | 1 | main 默认 |

| 推导项 | 实际值 |
| --- | --- |
| actual_num_mask | "collate loss_mask（不参与）" |
| actual_test_samples | 4839 |
| actual_train_samples | 11290 |
| block_size | 32 |
| coefficient_convention | "第一个系数乘最新值" |
| effective_data_mode | "full_split" |
| eval_base_length | 16 |
| eval_sequence_tokens | 18 |
| initialization | "fresh" |
| optimizer | "AdamW" |
| output_path | "<output>/models/<experiment>/<experiment>.pth" |
| recurrence_order | 2 |
| rules | [103,61] |
| scheduler | "CosineAnnealingLR" |
| scheduler_T_max | 6000 |
| train_exposure | 0.6999813999628 |
| train_sequence_tokens | 18 |
| vocab_size | 145 |

- corrupt_fix：loss_mask 由 collate 给出（仅位置/值两个输出位），NUM_MASK 不参与。
- corrupt_fix：污损是任务内生的（恰好一位改成随机错误值），无 clean/total 之分，结果口径 = 训练日志 best acc；输出 = 位置 token + 正确值两个 token。
- 静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。
- 训练结束不再跑 final generation test（2026-10-08 删除）；结果口径 = 训练日志 best acc + eval_clean_action.py。正常完成后跑 attention 分析；存档退出码 42 跳过后处理。

打包内容：已确认 JSON、定时写入 run.sh、训练代码快照、依赖清单、检查和运行工具。

请确认此版本；修改参数或训练代码后需重新出表。

<!-- PIPELINE_SNAPSHOT eyJhc3NldHMiOnt9LCJiYXRjaF9uYW1lIjoiY29ycnVwdF9maXhfc2luZ2xlX28yXzRjb2VmX2Q1MTJsMl9QMTI3X3RyMTYiLCJjb25jdXJyZW5jeSI6NCwiZXhwZXJpbWVudHMiOlt7ImNvbmZpZyI6eyJCQVRDSF9TSVpFIjo1MTIsIkRBVEFfTU9ERSI6ImZ1bGxfc3BsaXQiLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjUxMiwiRUFSTFlfU1RPUF9BQ0NVUkFDWSI6MC45OSwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOjAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRVBPQ0hTIjo2MDAwLCJFVkFMX0lOVEVSVkFMIjoxLCJGSVJTVF9UQVNLX1dFSUdIVCI6MS4wLCJGUkVTSF9URVNUX1BFUl9FVkFMIjpmYWxzZSwiR1JBRF9BQ0NVTV9TVEVQUyI6MSwiR1JBRF9DTElQX05PUk0iOjEuMCwiSU5JVF9GUk9NIjpudWxsLCJMUiI6MC4wMDAzLCJNQVhfVFJBSU5fSE9VUlMiOm51bGwsIk1JU1NJTkdfUFJPQiI6MC4wLCJNSVNTX0xFTiI6MSwiTUxQX1JBVElPIjo4LCJOVU1fTUFTSyI6bnVsbCwiTlVNX1RFU1RfU0FNUExFUyI6MjU2LCJOVU1fVFJBSU5fU0FNUExFUyI6MTEyOTAsIk5fSEVBRCI6MSwiTl9MQVlFUiI6MiwiT09EX0xFTiI6MTYsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJPUkRFUiI6MiwiUCI6MTI3LCJSQU5ET01fU0VFRCI6NDIsIlJFU0hVRkZMRV9FQUNIX0VQT0NIIjpmYWxzZSwiUkVTVU1FX0ZST00iOm51bGwsIlJVTEVTIjpbWzkzLDY5XV0sIlNUQVRFX1NQQUNFX0NBUCI6bnVsbCwiVEFTSyI6InNpbmdsZSIsIlRBU0tfVkFSSUFOVCI6ImNvcnJ1cHRfZml4IiwiVFJBSU5fTEVOIjoxNiwiV0VJR0hUX0RFQ0FZIjoxfSwiZWZmZWN0aXZlIjp7ImFjdHVhbF9udW1fbWFzayI6ImNvbGxhdGUgbG9zc19tYXNr77yI5LiN5Y+C5LiO77yJIiwiYWN0dWFsX3Rlc3Rfc2FtcGxlcyI6NDgzOSwiYWN0dWFsX3RyYWluX3NhbXBsZXMiOjExMjkwLCJibG9ja19zaXplIjozMiwiY29lZmZpY2llbnRfY29udmVudGlvbiI6IuesrOS4gOS4quezu+aVsOS5mOacgOaWsOWAvCIsImVmZmVjdGl2ZV9kYXRhX21vZGUiOiJmdWxsX3NwbGl0IiwiZXZhbF9iYXNlX2xlbmd0aCI6MTYsImV2YWxfc2VxdWVuY2VfdG9rZW5zIjoxOCwiaW5pdGlhbGl6YXRpb24iOiJmcmVzaCIsIm5vdGVzIjpbImNvcnJ1cHRfZml477yabG9zc19tYXNrIOeUsSBjb2xsYXRlIOe7meWHuu+8iOS7heS9jee9ri/lgLzkuKTkuKrovpPlh7rkvY3vvInvvIxOVU1fTUFTSyDkuI3lj4LkuI7jgIIiLCJjb3JydXB0X2ZpeO+8muaxoeaNn+aYr+S7u+WKoeWGheeUn+eahO+8iOaBsOWlveS4gOS9jeaUueaIkOmaj+acuumUmeivr+WAvO+8ie+8jOaXoCBjbGVhbi90b3RhbCDkuYvliIbvvIznu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2PvvJvovpPlh7ogPSDkvY3nva4gdG9rZW4gKyDmraPnoa7lgLzkuKTkuKogdG9rZW7jgIIiLCLpnZnmgIHmtYvor5XlnKjovr7liLAgRUFSTFlfU1RPUF9BQ0NVUkFDWSDml7blgZzmraLvvJtmcmVzaCB0ZXN0IOWcqOacgOi/kSAxMCDmrKHor4TkvLDlnYflgLzotoXov4fpmIjlgLzml7blgZzmraLjgIJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyDlt7Llup/lvIPvvIzkuI3nlJ/mlYjjgIIiLCLorq3nu4Pnu5PmnZ/kuI3lho3ot5EgZmluYWwgZ2VuZXJhdGlvbiB0ZXN077yIMjAyNi0xMC0wOCDliKDpmaTvvInvvJvnu5Pmnpzlj6PlvoQgPSDorq3nu4Pml6Xlv5cgYmVzdCBhY2MgKyBldmFsX2NsZWFuX2FjdGlvbi5weeOAguato+W4uOWujOaIkOWQjui3kSBhdHRlbnRpb24g5YiG5p6Q77yb5a2Y5qGj6YCA5Ye656CBIDQyIOi3s+i/h+WQjuWkhOeQhuOAgiJdLCJvcHRpbWl6ZXIiOiJBZGFtVyIsIm91dHB1dF9wYXRoIjoiPG91dHB1dD4vbW9kZWxzLzxleHBlcmltZW50Pi88ZXhwZXJpbWVudD4ucHRoIiwicmVjdXJyZW5jZV9vcmRlciI6MiwicnVsZXMiOls5Myw2OV0sInNjaGVkdWxlciI6IkNvc2luZUFubmVhbGluZ0xSIiwic2NoZWR1bGVyX1RfbWF4Ijo2MDAwLCJ0cmFpbl9leHBvc3VyZSI6MC42OTk5ODEzOTk5NjI4LCJ0cmFpbl9zZXF1ZW5jZV90b2tlbnMiOjE4LCJ2b2NhYl9zaXplIjoxNDV9LCJuYW1lIjoic2luZ2xlX28yX2NvcnJ1cHRmaXhfYTkzYjY5X2Q1MTJsMnI4aDFfUDEyN190cjE2X3NlZWQ0MiIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6Im1haW4g6buY6K6kIiwiREFUQV9NT0RFIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJEUk9QT1VUIjoibWFpbiDpu5jorqQiLCJEX01PREVMIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9BQ0NVUkFDWSI6Im1haW4g6buY6K6kIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoibWFpbiDpu5jorqQiLCJFVkFMX0lOVEVSVkFMIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRklSU1RfVEFTS19XRUlHSFQiOiJtYWluIOm7mOiupCIsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOiJtYWluIOm7mOiupCIsIkdSQURfQUNDVU1fU1RFUFMiOiJtYWluIOm7mOiupCIsIkdSQURfQ0xJUF9OT1JNIjoibWFpbiDpu5jorqQiLCJJTklUX0ZST00iOiJtYWluIOm7mOiupCIsIkxSIjoibWFpbiDpu5jorqQiLCJNQVhfVFJBSU5fSE9VUlMiOiJtYWluIOm7mOiupCIsIk1JU1NJTkdfUFJPQiI6Im1haW4g6buY6K6kIiwiTUlTU19MRU4iOiJtYWluIOm7mOiupCIsIk1MUF9SQVRJTyI6Im1haW4g6buY6K6kIiwiTlVNX01BU0siOiJtYWluIOm7mOiupCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiJtYWluIOm7mOiupCIsIk5fTEFZRVIiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJPT0RfTEVOIjoibWFpbiDpu5jorqQiLCJPUFRfRElBR19JTlRFUlZBTCI6Im1haW4g6buY6K6kIiwiT1JERVIiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUkFORE9NX1NFRUQiOiJtYWluIOm7mOiupCIsIlJFU0hVRkZMRV9FQUNIX0VQT0NIIjoibWFpbiDpu5jorqQiLCJSRVNVTUVfRlJPTSI6Im1haW4g6buY6K6kIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJXRUlHSFRfREVDQVkiOiJtYWluIOm7mOiupCJ9LCJ0YXNrIjoic2luZ2xlIn0seyJjb25maWciOnsiQkFUQ0hfU0laRSI6NTEyLCJEQVRBX01PREUiOiJmdWxsX3NwbGl0IiwiRFJPUE9VVCI6MC4wLCJEX01PREVMIjo1MTIsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOjAuOTksIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjowLCJFQVJMWV9TVE9QX05PX0lNUFJPVkUiOjMwMDAsIkVQT0NIUyI6NjAwMCwiRVZBTF9JTlRFUlZBTCI6MSwiRklSU1RfVEFTS19XRUlHSFQiOjEuMCwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6ZmFsc2UsIkdSQURfQUNDVU1fU1RFUFMiOjEsIkdSQURfQ0xJUF9OT1JNIjoxLjAsIklOSVRfRlJPTSI6bnVsbCwiTFIiOjAuMDAwMywiTUFYX1RSQUlOX0hPVVJTIjpudWxsLCJNSVNTSU5HX1BST0IiOjAuMCwiTUlTU19MRU4iOjEsIk1MUF9SQVRJTyI6OCwiTlVNX01BU0siOm51bGwsIk5VTV9URVNUX1NBTVBMRVMiOjI1NiwiTlVNX1RSQUlOX1NBTVBMRVMiOjExMjkwLCJOX0hFQUQiOjEsIk5fTEFZRVIiOjIsIk9PRF9MRU4iOjE2LCJPUFRfRElBR19JTlRFUlZBTCI6MCwiT1JERVIiOjIsIlAiOjEyNywiUkFORE9NX1NFRUQiOjQyLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6ZmFsc2UsIlJFU1VNRV9GUk9NIjpudWxsLCJSVUxFUyI6W1syMiw4Nl1dLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJzaW5nbGUiLCJUQVNLX1ZBUklBTlQiOiJjb3JydXB0X2ZpeCIsIlRSQUlOX0xFTiI6MTYsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiJjb2xsYXRlIGxvc3NfbWFza++8iOS4jeWPguS4ju+8iSIsImFjdHVhbF90ZXN0X3NhbXBsZXMiOjQ4MzksImFjdHVhbF90cmFpbl9zYW1wbGVzIjoxMTI5MCwiYmxvY2tfc2l6ZSI6MzIsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDmlrDlgLwiLCJlZmZlY3RpdmVfZGF0YV9tb2RlIjoiZnVsbF9zcGxpdCIsImV2YWxfYmFzZV9sZW5ndGgiOjE2LCJldmFsX3NlcXVlbmNlX3Rva2VucyI6MTgsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJjb3JydXB0X2ZpeO+8mmxvc3NfbWFzayDnlLEgY29sbGF0ZSDnu5nlh7rvvIjku4XkvY3nva4v5YC85Lik5Liq6L6T5Ye65L2N77yJ77yMTlVNX01BU0sg5LiN5Y+C5LiO44CCIiwiY29ycnVwdF9maXjvvJrmsaHmjZ/mmK/ku7vliqHlhoXnlJ/nmoTvvIjmgbDlpb3kuIDkvY3mlLnmiJDpmo/mnLrplJnor6/lgLzvvInvvIzml6AgY2xlYW4vdG90YWwg5LmL5YiG77yM57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNj77yb6L6T5Ye6ID0g5L2N572uIHRva2VuICsg5q2j56Gu5YC85Lik5LiqIHRva2Vu44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5LiN5YaN6LeRIGZpbmFsIGdlbmVyYXRpb24gdGVzdO+8iDIwMjYtMTAtMDgg5Yig6Zmk77yJ77yb57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNjICsgZXZhbF9jbGVhbl9hY3Rpb24ucHnjgILmraPluLjlrozmiJDlkI7ot5EgYXR0ZW50aW9uIOWIhuaekO+8m+WtmOaho+mAgOWHuueggSA0MiDot7Pov4flkI7lpITnkIbjgIIiXSwib3B0aW1pemVyIjoiQWRhbVciLCJvdXRwdXRfcGF0aCI6IjxvdXRwdXQ+L21vZGVscy88ZXhwZXJpbWVudD4vPGV4cGVyaW1lbnQ+LnB0aCIsInJlY3VycmVuY2Vfb3JkZXIiOjIsInJ1bGVzIjpbMjIsODZdLCJzY2hlZHVsZXIiOiJDb3NpbmVBbm5lYWxpbmdMUiIsInNjaGVkdWxlcl9UX21heCI6NjAwMCwidHJhaW5fZXhwb3N1cmUiOjAuNjk5OTgxMzk5OTYyOCwidHJhaW5fc2VxdWVuY2VfdG9rZW5zIjoxOCwidm9jYWJfc2l6ZSI6MTQ1fSwibmFtZSI6InNpbmdsZV9vMl9jb3JydXB0Zml4X2EyMmI4Nl9kNTEybDJyOGgxX1AxMjdfdHIxNl9zZWVkNDIiLCJvcmlnaW5zIjp7IkJBVENIX1NJWkUiOiJtYWluIOm7mOiupCIsIkRBVEFfTU9ERSI6IuS7o+eggeivreS5ieaOqOWvvO+8m+WOn+WAvD1udWxsIiwiRFJPUE9VVCI6Im1haW4g6buY6K6kIiwiRF9NT0RFTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkVBUkxZX1NUT1BfQUNDVVJBQ1kiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIjoibWFpbiDpu5jorqQiLCJFQVJMWV9TVE9QX05PX0lNUFJPVkUiOiJtYWluIOm7mOiupCIsIkVQT0NIUyI6Im1haW4g6buY6K6kIiwiRVZBTF9JTlRFUlZBTCI6IueUqOaIt+WFrOWFseWPguaVsCIsIkZJUlNUX1RBU0tfV0VJR0hUIjoibWFpbiDpu5jorqQiLCJGUkVTSF9URVNUX1BFUl9FVkFMIjoibWFpbiDpu5jorqQiLCJHUkFEX0FDQ1VNX1NURVBTIjoibWFpbiDpu5jorqQiLCJHUkFEX0NMSVBfTk9STSI6Im1haW4g6buY6K6kIiwiSU5JVF9GUk9NIjoibWFpbiDpu5jorqQiLCJMUiI6Im1haW4g6buY6K6kIiwiTUFYX1RSQUlOX0hPVVJTIjoibWFpbiDpu5jorqQiLCJNSVNTSU5HX1BST0IiOiJtYWluIOm7mOiupCIsIk1JU1NfTEVOIjoibWFpbiDpu5jorqQiLCJNTFBfUkFUSU8iOiJtYWluIOm7mOiupCIsIk5VTV9NQVNLIjoibWFpbiDpu5jorqQiLCJOVU1fVEVTVF9TQU1QTEVTIjoibWFpbiDpu5jorqQiLCJOVU1fVFJBSU5fU0FNUExFUyI6IuS7o+eggeivreS5ieaOqOWvvO+8m+WOn+WAvD1udWxsIiwiTl9IRUFEIjoibWFpbiDpu5jorqQiLCJOX0xBWUVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiT09EX0xFTiI6Im1haW4g6buY6K6kIiwiT1BUX0RJQUdfSU5URVJWQUwiOiJtYWluIOm7mOiupCIsIk9SREVSIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlJBTkRPTV9TRUVEIjoibWFpbiDpu5jorqQiLCJSRVNIVUZGTEVfRUFDSF9FUE9DSCI6Im1haW4g6buY6K6kIiwiUkVTVU1FX0ZST00iOiJtYWluIOm7mOiupCIsIlJVTEVTIjoi55So5oi36YCQ5a6e6aqM5Y+C5pWwIiwiU1RBVEVfU1BBQ0VfQ0FQIjoibWFpbiDpu5jorqQiLCJUQVNLIjoi5a6e6aqM5Lu75YqhIiwiVEFTS19WQVJJQU5UIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiVFJBSU5fTEVOIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiV0VJR0hUX0RFQ0FZIjoibWFpbiDpu5jorqQifSwidGFzayI6InNpbmdsZSJ9LHsiY29uZmlnIjp7IkJBVENIX1NJWkUiOjUxMiwiREFUQV9NT0RFIjoiZnVsbF9zcGxpdCIsIkRST1BPVVQiOjAuMCwiRF9NT0RFTCI6NTEyLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjowLjk5LCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6MCwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjozMDAwLCJFUE9DSFMiOjYwMDAsIkVWQUxfSU5URVJWQUwiOjEsIkZJUlNUX1RBU0tfV0VJR0hUIjoxLjAsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOmZhbHNlLCJHUkFEX0FDQ1VNX1NURVBTIjoxLCJHUkFEX0NMSVBfTk9STSI6MS4wLCJJTklUX0ZST00iOm51bGwsIkxSIjowLjAwMDMsIk1BWF9UUkFJTl9IT1VSUyI6bnVsbCwiTUlTU0lOR19QUk9CIjowLjAsIk1JU1NfTEVOIjoxLCJNTFBfUkFUSU8iOjgsIk5VTV9NQVNLIjpudWxsLCJOVU1fVEVTVF9TQU1QTEVTIjoyNTYsIk5VTV9UUkFJTl9TQU1QTEVTIjoxMTI5MCwiTl9IRUFEIjoxLCJOX0xBWUVSIjoyLCJPT0RfTEVOIjoxNiwiT1BUX0RJQUdfSU5URVJWQUwiOjAsIk9SREVSIjoyLCJQIjoxMjcsIlJBTkRPTV9TRUVEIjo0MiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOmZhbHNlLCJSRVNVTUVfRlJPTSI6bnVsbCwiUlVMRVMiOltbOTgsMTldXSwiU1RBVEVfU1BBQ0VfQ0FQIjpudWxsLCJUQVNLIjoic2luZ2xlIiwiVEFTS19WQVJJQU5UIjoiY29ycnVwdF9maXgiLCJUUkFJTl9MRU4iOjE2LCJXRUlHSFRfREVDQVkiOjF9LCJlZmZlY3RpdmUiOnsiYWN0dWFsX251bV9tYXNrIjoiY29sbGF0ZSBsb3NzX21hc2vvvIjkuI3lj4LkuI7vvIkiLCJhY3R1YWxfdGVzdF9zYW1wbGVzIjo0ODM5LCJhY3R1YWxfdHJhaW5fc2FtcGxlcyI6MTEyOTAsImJsb2NrX3NpemUiOjMyLCJjb2VmZmljaWVudF9jb252ZW50aW9uIjoi56ys5LiA5Liq57O75pWw5LmY5pyA5paw5YC8IiwiZWZmZWN0aXZlX2RhdGFfbW9kZSI6ImZ1bGxfc3BsaXQiLCJldmFsX2Jhc2VfbGVuZ3RoIjoxNiwiZXZhbF9zZXF1ZW5jZV90b2tlbnMiOjE4LCJpbml0aWFsaXphdGlvbiI6ImZyZXNoIiwibm90ZXMiOlsiY29ycnVwdF9maXjvvJpsb3NzX21hc2sg55SxIGNvbGxhdGUg57uZ5Ye677yI5LuF5L2N572uL+WAvOS4pOS4qui+k+WHuuS9je+8ie+8jE5VTV9NQVNLIOS4jeWPguS4juOAgiIsImNvcnJ1cHRfZml477ya5rGh5o2f5piv5Lu75Yqh5YaF55Sf55qE77yI5oGw5aW95LiA5L2N5pS55oiQ6ZqP5py66ZSZ6K+v5YC877yJ77yM5pegIGNsZWFuL3RvdGFsIOS5i+WIhu+8jOe7k+aenOWPo+W+hCA9IOiuree7g+aXpeW/lyBiZXN0IGFjY++8m+i+k+WHuiA9IOS9jee9riB0b2tlbiArIOato+ehruWAvOS4pOS4qiB0b2tlbuOAgiIsIumdmeaAgea1i+ivleWcqOi+vuWIsCBFQVJMWV9TVE9QX0FDQ1VSQUNZIOaXtuWBnOatou+8m2ZyZXNoIHRlc3Qg5Zyo5pyA6L+RIDEwIOasoeivhOS8sOWdh+WAvOi2hei/h+mYiOWAvOaXtuWBnOatouOAgkVBUkxZX1NUT1BfRVhUUkFfRVBPQ0hTIOW3suW6n+W8g++8jOS4jeeUn+aViOOAgiIsIuiuree7g+e7k+adn+S4jeWGjei3kSBmaW5hbCBnZW5lcmF0aW9uIHRlc3TvvIgyMDI2LTEwLTA4IOWIoOmZpO+8ie+8m+e7k+aenOWPo+W+hCA9IOiuree7g+aXpeW/lyBiZXN0IGFjYyArIGV2YWxfY2xlYW5fYWN0aW9uLnB544CC5q2j5bi45a6M5oiQ5ZCO6LeRIGF0dGVudGlvbiDliIbmnpDvvJvlrZjmoaPpgIDlh7rnoIEgNDIg6Lez6L+H5ZCO5aSE55CG44CCIl0sIm9wdGltaXplciI6IkFkYW1XIiwib3V0cHV0X3BhdGgiOiI8b3V0cHV0Pi9tb2RlbHMvPGV4cGVyaW1lbnQ+LzxleHBlcmltZW50Pi5wdGgiLCJyZWN1cnJlbmNlX29yZGVyIjoyLCJydWxlcyI6Wzk4LDE5XSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjowLjY5OTk4MTM5OTk2MjgsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTgsInZvY2FiX3NpemUiOjE0NX0sIm5hbWUiOiJzaW5nbGVfbzJfY29ycnVwdGZpeF9hOThiMTlfZDUxMmwycjhoMV9QMTI3X3RyMTZfc2VlZDQyIiwib3JpZ2lucyI6eyJCQVRDSF9TSVpFIjoibWFpbiDpu5jorqQiLCJEQVRBX01PREUiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIkRST1BPVVQiOiJtYWluIOm7mOiupCIsIkRfTU9ERUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJFQVJMWV9TVE9QX0FDQ1VSQUNZIjoibWFpbiDpu5jorqQiLCJFQVJMWV9TVE9QX0VYVFJBX0VQT0NIUyI6Im1haW4g6buY6K6kIiwiRUFSTFlfU1RPUF9OT19JTVBST1ZFIjoibWFpbiDpu5jorqQiLCJFUE9DSFMiOiJtYWluIOm7mOiupCIsIkVWQUxfSU5URVJWQUwiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJGSVJTVF9UQVNLX1dFSUdIVCI6Im1haW4g6buY6K6kIiwiRlJFU0hfVEVTVF9QRVJfRVZBTCI6Im1haW4g6buY6K6kIiwiR1JBRF9BQ0NVTV9TVEVQUyI6Im1haW4g6buY6K6kIiwiR1JBRF9DTElQX05PUk0iOiJtYWluIOm7mOiupCIsIklOSVRfRlJPTSI6Im1haW4g6buY6K6kIiwiTFIiOiJtYWluIOm7mOiupCIsIk1BWF9UUkFJTl9IT1VSUyI6Im1haW4g6buY6K6kIiwiTUlTU0lOR19QUk9CIjoibWFpbiDpu5jorqQiLCJNSVNTX0xFTiI6Im1haW4g6buY6K6kIiwiTUxQX1JBVElPIjoibWFpbiDpu5jorqQiLCJOVU1fTUFTSyI6Im1haW4g6buY6K6kIiwiTlVNX1RFU1RfU0FNUExFUyI6Im1haW4g6buY6K6kIiwiTlVNX1RSQUlOX1NBTVBMRVMiOiLku6PnoIHor63kuYnmjqjlr7zvvJvljp/lgLw9bnVsbCIsIk5fSEVBRCI6Im1haW4g6buY6K6kIiwiTl9MQVlFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIk9PRF9MRU4iOiJtYWluIOm7mOiupCIsIk9QVF9ESUFHX0lOVEVSVkFMIjoibWFpbiDpu5jorqQiLCJPUkRFUiI6IueUqOaIt+WFrOWFseWPguaVsCIsIlAiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJSQU5ET01fU0VFRCI6Im1haW4g6buY6K6kIiwiUkVTSFVGRkxFX0VBQ0hfRVBPQ0giOiJtYWluIOm7mOiupCIsIlJFU1VNRV9GUk9NIjoibWFpbiDpu5jorqQiLCJSVUxFUyI6IueUqOaIt+mAkOWunumqjOWPguaVsCIsIlNUQVRFX1NQQUNFX0NBUCI6Im1haW4g6buY6K6kIiwiVEFTSyI6IuWunumqjOS7u+WKoSIsIlRBU0tfVkFSSUFOVCI6IueUqOaIt+WFrOWFseWPguaVsCIsIlRSQUlOX0xFTiI6IueUqOaIt+WFrOWFseWPguaVsCIsIldFSUdIVF9ERUNBWSI6Im1haW4g6buY6K6kIn0sInRhc2siOiJzaW5nbGUifSx7ImNvbmZpZyI6eyJCQVRDSF9TSVpFIjo1MTIsIkRBVEFfTU9ERSI6ImZ1bGxfc3BsaXQiLCJEUk9QT1VUIjowLjAsIkRfTU9ERUwiOjUxMiwiRUFSTFlfU1RPUF9BQ0NVUkFDWSI6MC45OSwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOjAsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6MzAwMCwiRVBPQ0hTIjo2MDAwLCJFVkFMX0lOVEVSVkFMIjoxLCJGSVJTVF9UQVNLX1dFSUdIVCI6MS4wLCJGUkVTSF9URVNUX1BFUl9FVkFMIjpmYWxzZSwiR1JBRF9BQ0NVTV9TVEVQUyI6MSwiR1JBRF9DTElQX05PUk0iOjEuMCwiSU5JVF9GUk9NIjpudWxsLCJMUiI6MC4wMDAzLCJNQVhfVFJBSU5fSE9VUlMiOm51bGwsIk1JU1NJTkdfUFJPQiI6MC4wLCJNSVNTX0xFTiI6MSwiTUxQX1JBVElPIjo4LCJOVU1fTUFTSyI6bnVsbCwiTlVNX1RFU1RfU0FNUExFUyI6MjU2LCJOVU1fVFJBSU5fU0FNUExFUyI6MTEyOTAsIk5fSEVBRCI6MSwiTl9MQVlFUiI6MiwiT09EX0xFTiI6MTYsIk9QVF9ESUFHX0lOVEVSVkFMIjowLCJPUkRFUiI6MiwiUCI6MTI3LCJSQU5ET01fU0VFRCI6NDIsIlJFU0hVRkZMRV9FQUNIX0VQT0NIIjpmYWxzZSwiUkVTVU1FX0ZST00iOm51bGwsIlJVTEVTIjpbWzEwMyw2MV1dLCJTVEFURV9TUEFDRV9DQVAiOm51bGwsIlRBU0siOiJzaW5nbGUiLCJUQVNLX1ZBUklBTlQiOiJjb3JydXB0X2ZpeCIsIlRSQUlOX0xFTiI6MTYsIldFSUdIVF9ERUNBWSI6MX0sImVmZmVjdGl2ZSI6eyJhY3R1YWxfbnVtX21hc2siOiJjb2xsYXRlIGxvc3NfbWFza++8iOS4jeWPguS4ju+8iSIsImFjdHVhbF90ZXN0X3NhbXBsZXMiOjQ4MzksImFjdHVhbF90cmFpbl9zYW1wbGVzIjoxMTI5MCwiYmxvY2tfc2l6ZSI6MzIsImNvZWZmaWNpZW50X2NvbnZlbnRpb24iOiLnrKzkuIDkuKrns7vmlbDkuZjmnIDmlrDlgLwiLCJlZmZlY3RpdmVfZGF0YV9tb2RlIjoiZnVsbF9zcGxpdCIsImV2YWxfYmFzZV9sZW5ndGgiOjE2LCJldmFsX3NlcXVlbmNlX3Rva2VucyI6MTgsImluaXRpYWxpemF0aW9uIjoiZnJlc2giLCJub3RlcyI6WyJjb3JydXB0X2ZpeO+8mmxvc3NfbWFzayDnlLEgY29sbGF0ZSDnu5nlh7rvvIjku4XkvY3nva4v5YC85Lik5Liq6L6T5Ye65L2N77yJ77yMTlVNX01BU0sg5LiN5Y+C5LiO44CCIiwiY29ycnVwdF9maXjvvJrmsaHmjZ/mmK/ku7vliqHlhoXnlJ/nmoTvvIjmgbDlpb3kuIDkvY3mlLnmiJDpmo/mnLrplJnor6/lgLzvvInvvIzml6AgY2xlYW4vdG90YWwg5LmL5YiG77yM57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNj77yb6L6T5Ye6ID0g5L2N572uIHRva2VuICsg5q2j56Gu5YC85Lik5LiqIHRva2Vu44CCIiwi6Z2Z5oCB5rWL6K+V5Zyo6L6+5YiwIEVBUkxZX1NUT1BfQUNDVVJBQ1kg5pe25YGc5q2i77ybZnJlc2ggdGVzdCDlnKjmnIDov5EgMTAg5qyh6K+E5Lyw5Z2H5YC86LaF6L+H6ZiI5YC85pe25YGc5q2i44CCRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMg5bey5bqf5byD77yM5LiN55Sf5pWI44CCIiwi6K6t57uD57uT5p2f5LiN5YaN6LeRIGZpbmFsIGdlbmVyYXRpb24gdGVzdO+8iDIwMjYtMTAtMDgg5Yig6Zmk77yJ77yb57uT5p6c5Y+j5b6EID0g6K6t57uD5pel5b+XIGJlc3QgYWNjICsgZXZhbF9jbGVhbl9hY3Rpb24ucHnjgILmraPluLjlrozmiJDlkI7ot5EgYXR0ZW50aW9uIOWIhuaekO+8m+WtmOaho+mAgOWHuueggSA0MiDot7Pov4flkI7lpITnkIbjgIIiXSwib3B0aW1pemVyIjoiQWRhbVciLCJvdXRwdXRfcGF0aCI6IjxvdXRwdXQ+L21vZGVscy88ZXhwZXJpbWVudD4vPGV4cGVyaW1lbnQ+LnB0aCIsInJlY3VycmVuY2Vfb3JkZXIiOjIsInJ1bGVzIjpbMTAzLDYxXSwic2NoZWR1bGVyIjoiQ29zaW5lQW5uZWFsaW5nTFIiLCJzY2hlZHVsZXJfVF9tYXgiOjYwMDAsInRyYWluX2V4cG9zdXJlIjowLjY5OTk4MTM5OTk2MjgsInRyYWluX3NlcXVlbmNlX3Rva2VucyI6MTgsInZvY2FiX3NpemUiOjE0NX0sIm5hbWUiOiJzaW5nbGVfbzJfY29ycnVwdGZpeF9hMTAzYjYxX2Q1MTJsMnI4aDFfUDEyN190cjE2X3NlZWQ0MiIsIm9yaWdpbnMiOnsiQkFUQ0hfU0laRSI6Im1haW4g6buY6K6kIiwiREFUQV9NT0RFIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJEUk9QT1VUIjoibWFpbiDpu5jorqQiLCJEX01PREVMIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRUFSTFlfU1RPUF9BQ0NVUkFDWSI6Im1haW4g6buY6K6kIiwiRUFSTFlfU1RPUF9FWFRSQV9FUE9DSFMiOiJtYWluIOm7mOiupCIsIkVBUkxZX1NUT1BfTk9fSU1QUk9WRSI6Im1haW4g6buY6K6kIiwiRVBPQ0hTIjoibWFpbiDpu5jorqQiLCJFVkFMX0lOVEVSVkFMIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiRklSU1RfVEFTS19XRUlHSFQiOiJtYWluIOm7mOiupCIsIkZSRVNIX1RFU1RfUEVSX0VWQUwiOiJtYWluIOm7mOiupCIsIkdSQURfQUNDVU1fU1RFUFMiOiJtYWluIOm7mOiupCIsIkdSQURfQ0xJUF9OT1JNIjoibWFpbiDpu5jorqQiLCJJTklUX0ZST00iOiJtYWluIOm7mOiupCIsIkxSIjoibWFpbiDpu5jorqQiLCJNQVhfVFJBSU5fSE9VUlMiOiJtYWluIOm7mOiupCIsIk1JU1NJTkdfUFJPQiI6Im1haW4g6buY6K6kIiwiTUlTU19MRU4iOiJtYWluIOm7mOiupCIsIk1MUF9SQVRJTyI6Im1haW4g6buY6K6kIiwiTlVNX01BU0siOiJtYWluIOm7mOiupCIsIk5VTV9URVNUX1NBTVBMRVMiOiJtYWluIOm7mOiupCIsIk5VTV9UUkFJTl9TQU1QTEVTIjoi5Luj56CB6K+t5LmJ5o6o5a+877yb5Y6f5YC8PW51bGwiLCJOX0hFQUQiOiJtYWluIOm7mOiupCIsIk5fTEFZRVIiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJPT0RfTEVOIjoibWFpbiDpu5jorqQiLCJPUFRfRElBR19JTlRFUlZBTCI6Im1haW4g6buY6K6kIiwiT1JERVIiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJQIjoi55So5oi35YWs5YWx5Y+C5pWwIiwiUkFORE9NX1NFRUQiOiJtYWluIOm7mOiupCIsIlJFU0hVRkZMRV9FQUNIX0VQT0NIIjoibWFpbiDpu5jorqQiLCJSRVNVTUVfRlJPTSI6Im1haW4g6buY6K6kIiwiUlVMRVMiOiLnlKjmiLfpgJDlrp7pqozlj4LmlbAiLCJTVEFURV9TUEFDRV9DQVAiOiJtYWluIOm7mOiupCIsIlRBU0siOiLlrp7pqozku7vliqEiLCJUQVNLX1ZBUklBTlQiOiLnlKjmiLflhazlhbHlj4LmlbAiLCJUUkFJTl9MRU4iOiLnlKjmiLflhazlhbHlj4LmlbAiLCJXRUlHSFRfREVDQVkiOiJtYWluIOm7mOiupCJ9LCJ0YXNrIjoic2luZ2xlIn1dLCJsb2NhbF9ncHVfYWxsb3dsaXN0IjpbMCwxLDIsM10sInBhY2thZ2luZ19ob3N0IjoidGVzdCIsInNvdXJjZV9maWxlcyI6eyJyZXF1aXJlbWVudHMudHh0IjoiOWMxYzAxODNmYTFhZGFiMTE3ZjgyYWRhMDFmMGFhMzRhZmFhZDYzMjllZmM1MmRhNGQyOGNiMjRiNTViMDg3NiIsInNjcmlwdHMvZXhwZXJpbWVudF9wYWNrYWdlX3J1bm5lci5weSI6IjBiYTE0N2Y4YTc2NzY1ODc5MDVjNDMxMzA5MmYxZjk5ZTFlMmI0NmU5ZTlmYzg3ODcxMzFmOGRiZmE3MjNmM2IiLCJzcmMvYWN0aXZhdGlvbl9wYXRjaC5weSI6IjFhNmE1OTgwMTU5MjgyODdkNTI1OWUyNDc0MDdkNDk0ZDAyOTVkZjY2Y2I3MzdjN2M1NGMxYzRjYmY4YmM3YjIiLCJzcmMvYW5hbHl6ZV9hdHRlbnRpb24ucHkiOiIyZDA3MjMxODMyMGIwMDY1NGFjNzk1MDM2N2ExMzk4NDJiNjdkOGY3M2FlOGFlN2EwNTk4MDFjMGExODFhNzE4Iiwic3JjL2JhdGNoX3J1bi5weSI6IjBkZjY3YTdiZTUxN2NkYWM3NmJiMDU0N2YwNDhlZDEzNTgzYjcyZGM3MmZhOTQwNTEzOTIyOGQ3MzY2NmE1NjkiLCJzcmMvY2hhaW5fcnVuLnB5IjoiMDU5NjVhZTMzNThmNjFmNTBjYTk5NWYyYzZlN2Q4YTU4YzUwM2VkYWQ4MjJmZjJlMDdjM2U2ZjYxZDkwMDRlMyIsInNyYy9jb25maWcuanNvbiI6IjVmMTNjNTAwZjA5YzhjNTQ2OGFlMzBlNWEwZTQyMDJmMWI4MzYxNGVjNGI3NGViYjZhOWE2NDU0ZGZhNDliODQiLCJzcmMvY29yZS5weSI6IjE1MjJhMzBiYWRhODQzZjAwOTAyM2EyYTcwNTc1ZDc3MTRhM2U4NjYzZjAxMjEwNzdhYjYwNDY1MDBlN2MwYWIiLCJzcmMvY3VycmljdWx1bV9jaGFpbl9OdGFzay5weSI6IjQ3ZWJkYmFiZWNiMTg2MTQzNTc5NDQ1M2EwYzQ0MWQwYjA5ZjhjMTgxYWM4NDA0YmExN2ZmNmVhMGRhN2M1OTMiLCJzcmMvY3VycmljdWx1bV9jaGFpbl9leHRlbmQucHkiOiJiMjczNTM1NDIxNjEwYjM0NjM0ZWViMzA3NjhhNTkxODVjMTg3MGRiMTVjMDA4ZGM5ODg5OTU2ZmFjZGEzZmZmIiwic3JjL2N1cnJpY3VsdW1fY2hhaW5fcmV2ZXJzZS5weSI6IjVjMzE4OTIxMGNlNjEyZTQ0ODcyMTk4ZjZhMjJlMGIyN2Y1ZTIzYTAwODgwNjI1YjE0NWNlMWE3ODA2ZTZlYTciLCJzcmMvZGF0YXNldHMucHkiOiI5OWJjZmMxMWRkNzI5ZjkxZmM2ODZhNzc3ODYyMDI4MTExM2M4YzFiMzNiNWJlN2MwODMzZjJlMTI5ZWEzZTUxIiwic3JjL2Vycm9yX2JyZWFrZG93bi5weSI6ImE1ZGYyZTYyNDNiMjY1MGU0ODRkNDk0MzA3ZDkzMGQzZTEwYWJkODBlYmVmMTExYjZlN2I1MzM0Mjk5YjYwZTEiLCJzcmMvZXZpZGVuY2Vfc3dlZXAucHkiOiIzNTUwMjU3MzFhODc0NDIzYmU3MTBlZTRiMmE1ZjQ0ZDQwMDg4YWM0ZWZiMTRiOWIyNGMwZDIzZmI4YmUxYjQ5Iiwic3JjL2V4cGVyaW1lbnQucHkiOiI3NmIxNjRlZWI5ODRkNTEyNzYxZmI1OTI1NzZjZWU4ODlkZDkzZTcwMTU3NGExZjJhMzE3NjM5NWMzMjdkYTU1Iiwic3JjL2ZpeF9sb2dzLnB5IjoiODM4M2Y5YzE3ZjJmYWQ5M2E0YWY4ZTMwNDU2OWYwYzBlY2NlZGEzN2NlOTVjYWQzNzdhYWI4YmFmMGQyMDhjYSIsInNyYy9oZWFkX2FibGF0aW9uLnB5IjoiZDAzODQ2MTAwZDNkOWE5NzY3MWI5MmM1MWNjMjc2MWQ2ODIxMWM5NGYwMGM1ZDZhNWM4N2E0MWJkMTYxNjQ1ZiIsInNyYy9tb2RlbHMucHkiOiJhMjQyZWU5Y2Q4ODFkNzA3MGUwOTIxOGZmNjI4NjBlMmY2YjA0ZmJhNmFhMjcyMWM3ZGMzMGFlMGVkODVmYjkwIiwic3JjL211bHRpX3Bvc19wYXRjaC5weSI6ImU1NDMzNDExMzNiYmY3Mzk5ODgyMDdlMTA4NWNhZGVlMTk3YTY2ZWVlZjUzODA2MTc0ZTQ1MDk2ZDEzYWNlOWMiLCJzcmMvcHJlcGFyZV9yZXJ1bi5weSI6IjdmY2FkMDc1YThhOTk2NGE4M2M3N2E0M2UxZmZhMjkyYTIwNjRjN2YxNTc2YjhlMTI0NWRlZmU5MDVhOTdjZjQiLCJzcmMvcHJvdG9jb2wucHkiOiJkYWZhNThjZGJhZmE5NDZkZjVlOWRkYWI3OGEzNDI3NzQzNTc1ZDJlYzYyOWI1M2RjOGM4Y2Y1NGI4MjIzZTcyIiwic3JjL3JlcG9ydF9mcm9udF9iYWNrX2F0dGVudGlvbi5weSI6IjVjNzNmZjI2MmQyZDhkOWU4ZGVmN2NjMDFjMWY4MjhmOTQzMTUyY2ViODZmM2NiN2FmMzRiOGVjMzkyMTVkNjUiLCJzcmMvcnVsZV9maXQucHkiOiJhYzJjNTJhMDc2NGUxZjE5YjVkMTRlYTkwYjE4ODliYjY0ZWJlNmE5ZGVlOTlkZGM5ZDMzZGNhNDQzMzZjOWNjIiwic3JjL3J1bGVfZ2VuZXJhbGl6YXRpb24ucHkiOiI0YjE5MjdmODQyNWVjN2Y4ODBmNDdmYjMxNDQxNDFiMWZkYjZiN2EzNzE2ZWY5MGY3OGExOTYyYzJkYjMzOWUxIiwic3JjL3J1bGVzLnB5IjoiNjdjYmYwMmU3ODQ1YjhlZDUwMmZjMzRlNWQ4MTg4NzNiOTJhZDg3NmU2ODQ3NzNjYjJmNzAyZDljMjY2OTUyMCIsInNyYy9zYW1wbGVfZ2VuLnB5IjoiODM5M2IzM2Y5MGFlYjc2MGZmNmE1NmJmOGNmYTg1ODIzNmNiMmQxM2ViNTQ4YTk4ZWZkMGY2ODUyYmUyNTNkMSIsInNyYy9zY2VuYXJpb3MucHkiOiJiYmNmNmRmYWFmMjlhZGIwMDA5Y2E1NDVjYzAzNWIyNjNhZTgyMmQ3ZTQ3Njg0YzJlNzRjOGNkOWNmNTFiMDI3Iiwic3JjL3RyYWluaW5nLnB5IjoiOTFjMzMwMzdmYTcwNWNkN2EyYjIxOGQ4ZWMwNDgzOTIwMTg1ZmZmNDYxMmZlMDliNWJmYjY3NDk1ZTFlOTRiOSIsInNyYy92ZXJpZnlfY2lyY2xlLnB5IjoiZTMxYzVhZGI2N2UwYjgzYjE5YjdlNGI5MjcwZDlmYjVhYzhmNWVmZDFkMWYzYTFlMWFlZTYxZDI0NWY0NGZmMyIsInNyYy92ZXJpZnlfc2FtcGxlLnB5IjoiNjc5MGMzZjBlY2EyYTkzYTAzYjk1MTgxZWZiMzU4ZDlhMjVhNzYxOTcxYzRhNjJiY2I0M2I5YjAxN2M1MTg0ZSIsInNyYy92aXN1YWxpemUucHkiOiIxMmE0NTNhZTk1NzYxNTU1MWZhY2FiNTE2YWFiMjdjYTVlYmVhMmQ0ZDM2OTczM2RhODc5YWM3MjNlN2NhYTI3In0sInVucmVzb2x2ZWQiOltdLCJ2ZXJzaW9uIjoxfQ== -->
