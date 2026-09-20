# Curriculum / Task-Level Sequential Learning 实验报告

> 实验日期: 2025-09-15 ~ 2025-09-15
> 实验脚本: `src/curriculum_sequential.py`
> 对标 baseline: `recursion_results/mixed_basic_d1024_p127_missLen/mixed_basic_d1024l4r8h4_p127_rules12_miss01_len64_online/` (seed=17996, MISS_LEN=1)

## 1. 实验目标

在 mixed basic setting (P=127, rules=[[1,1],[2,3]]) 上做 **task-level sequential learning**：
- Stage 1: 在单条规则的 70% 训练集上训练，目标 test acc ≥ 98% (早停) 或 6h 超时
- Stage 2: 用 Stage 1 best checkpoint 暖启动，开始双规则混合训练 (6h 超时)

对比两种课程顺序：
- **Model A**: 先学 [1,1] (简单规则) → 混合 [1,1]+[2,3]
- **Model B**: 先学 [2,3] (难规则) → 混合 [1,1]+[2,3]

## 2. 超参数 (严格对齐 baseline)

| 项 | 值 |
|---|---|
| P | 127 |
| Rules | [[1,1], [2,3]] |
| D_MODEL / N_LAYER / N_HEAD / MLP_RATIO | 1024 / 4 / 4 / 8 |
| TRAIN_LEN / OOD_LEN | 64 / 128 |
| BATCH_SIZE / LR / WD | 512 / 3e-4 / 1.0 |
| USE_AB_TAG / USE_CONDITIONAL_WTE | False / False |
| MIXED_AB_MAX_UNIQUE_RATIOS | [0.7, 0.7] (state_space=16129 → train=11290/rule) |
| MISSING_PROB / MISS_LEN / PREDICT_MISSING | 0.1 / 1 / False (online missing, 两阶段都用 `make_mixed_missing_collate`) |
| ALLOW_TF32 / USE_AMP | True / False |
| EPOCHS / EVAL_INTERVAL | 6000 / 20 |
| Stage1 early_stop_accuracy | 0.98 |
| Stage1/Stage2 MAX_TRAIN_HOURS | 6.0 / 6.0 |
| Seed | 17996 |
| 模型 | `MixedABTransformer(num_ab_pairs=2, use_ab_tag=False)` (含 rule_head) |

> 与 baseline 唯一差异：训练过程采用两阶段序列学习，而非从一开始就双规则混合。

## 3. Checkpoint 布局

每个模型在 `recursion_results/curriculum_sequential/<exp_name>/checkpoints/` 下保存：

| 文件 | 含义 |
|---|---|
| `<exp>_init.pth` | Stage 1 训练前随机初始化权重 |
| `<exp>_stage1_best.pth` | Stage 1 rolling best (Stage 2 暖启动源) |
| `<exp>_stage1_latest.pth` | Stage 1 rolling latest (含 optimizer state) |
| `<exp>_stage1.pth` 或 `_stage1_resume.pth` | Stage 1 终态 |
| `<exp>_stage2_best.pth` | Stage 2 rolling best |
| `<exp>_stage2_latest.pth` / `_stage2_resume.pth` | Stage 2 终态 |
| `<exp>_stage2.pth` | Stage 2 最终权重 (脚本末尾 `torch.save(model.state_dict())`) |

实验名：
- Model A: `curriculum_seq_d1024l4r8h4_P127_rules12_stage1[a11]_seed17996_gpu4`
- Model B: `curriculum_seq_d1024l4r8h4_P127_rules12_stage1[a23]_seed17996_gpu6`

## 4. 实验结果

### 4.1 Model A (先 [1,1] → 混合) — 成功

| 阶段 | 时长 | 结束 epoch | Best Test Acc | timed_out | per-rule acc |
|---|---|---|---|---|---|
| Stage 1 | 3h06m | 500 | **99.29%** @500 | False (达到 98% 早停) | — (单规则) |
| Stage 2 | 6h00m | 540 | **99.76%** @540 | True | **1.00 / 1.00** |

关键轨迹 (Stage 2 Test Acc):
- Epoch 0 (暖启动初测): 56.0% — rule [1,1] 已知，rule [2,3] 待学
- 单调爬升 → Epoch 460: 99.8% (per-rule 1.00/1.00)
- Epoch 500 短暂 spike (Loss=5.17, Acc=1.1%) — 训练不稳定性
- Epoch 520 立即恢复: 99.5% → Epoch 540: 99.8% (per-rule 1.00/1.00)
- 6h 超时退出

**结论**: Stage 1 学到的 rule [1,1] 表征完全保留，暖启动为 Stage 2 提供良好初始化，rule [2,3] 在混合训练中也被学到 100%。

### 4.2 Model B (先 [2,3] → 混合) — 失败

| 阶段 | 时长 | 结束 epoch | Best Test Acc | timed_out | per-rule acc |
|---|---|---|---|---|---|
| Stage 1 | 6h00m | 1100 | **17.07%** @1100 | True (远未达 98%) | — (单规则) |
| Stage 2 | 6h00m | 460 | **40.34%** @0 | True | 0.05 / 0.26 |

Stage 1 轨迹 (Test Acc):
- Epoch 0-400: 长期在 1.1-1.4% 徘徊
- Epoch 980 起开始爬升: 8% → 1100: 17.07% (末期加速但远未达 98%)
- 6h 超时退出，best 仅 17.07% → 暖启动源质量极差

Stage 2 轨迹 (Test Acc):
- Epoch 0 (暖启动初测): 40.3% — 即暖启动后双规则总体表现
- 一路退化: 40.3%@0 → 14.0%@260 → 15.3%@460
- **best_acc 始终停留在 epoch 0**，意味着整个 Stage 2 没有任何进步
- 6h 超时退出，per-rule=0.05/0.26 (rule [1,1] 部分学到 ~26%，rule [2,3] 几乎为 0)

**结论**: Stage 1 学不会 rule [2,3] → 暖启动无意义 → Stage 2 出现灾难性遗忘 + 混合训练困难，最终 best 即为暖启动初始状态。

## 5. 与 baseline (joint training from scratch) 对比

baseline: `mixed_basic_d1024l4r8h4_P127_rules12_miss01_len64_online`, seed=17996, MISS_LEN=1

| 方案 | Stage1 Acc | 最终 Test Acc | per-rule [1,1] / [2,3] | 总耗时 |
|---|---|---|---|---|
| baseline (joint) | — | **4.7%** @640 (6h 超时) | 0.08 / 0.02 | 6h |
| Curriculum A (先 [1,1]) | 99.29% @500 | **99.76%** @540 | 1.00 / 1.00 | ~9h |
| Curriculum B (先 [2,3]) | 17.07% @1100 | **40.34%** @0 | 0.26 / 0.05 | ~12h |

- **Curriculum A 完胜**: 双规则都达 100% per-rule，整体 99.76%，远超 baseline 的 4.7%
- **Curriculum B 仍失败**: 40.3% 略好于 baseline 的 4.7%，但 best@0 说明纯靠暖启动初测，Stage 2 没有任何学习进展

## 6. 核心结论

1. **课程顺序影响巨大**: 先学简单规则 [1,1] 能为混合训练提供良好基础，使难规则 [2,3] 也得以学会；先学难规则 [2,3] 则 Stage 1 都学不会，Stage 2 暖启动几乎无意义。
2. **Stage 1 质量 = Stage 2 上限**: Model A Stage 1 达 99.29% → Stage 2 最终 99.76%；Model B Stage 1 仅 17.07% → Stage 2 最佳仅 40.34% (即暖启动初始值)。
3. **难规则单独学也学不会**: rule [2,3] 在 70% 数据、6h 时长、单规则训练条件下仍仅达 17% (远低于 98% 目标)。rule [2,3] 的难度不是"数据少"或"被简单规则干扰"造成的，而是该规则本身在当前模型/优化设置下难以被学到。
4. **暖启动 + 简单规则前置是有效的课程学习策略**: 在 mixed_basic + online missing + MISS_LEN=1 这个 baseline 几乎学不会的 setting 下，先学简单规则再混合训练能让模型完全掌握双规则。

## 7. 复现命令

```bash
cd /data/lly/recursion

# Model A: 先学 [1,1] (GPU 4)
nohup python -u src/curriculum_sequential.py \
    --seed 17996 --gpu 4 --stage1-rule "1,1" \
    > recursion_results/curriculum_sequential/modelA.out 2>&1 &

# Model B: 先学 [2,3] (GPU 6)
nohup python -u src/curriculum_sequential.py \
    --seed 17996 --gpu 6 --stage1-rule "2,3" \
    > recursion_results/curriculum_sequential/modelB.out 2>&1 &
```

## 8. 文件位置

- 实验脚本: `src/curriculum_sequential.py`
- stdout 日志:
  - `recursion_results/curriculum_sequential/modelA.out`
  - `recursion_results/curriculum_sequential/modelB.out`
- Checkpoints:
  - `recursion_results/curriculum_sequential/curriculum_seq_d1024l4r8h4_P127_rules12_stage1[a11]_seed17996_gpu4/checkpoints/`
  - `recursion_results/curriculum_sequential/curriculum_seq_d1024l4r8h4_P127_rules12_stage1[a23]_seed17996_gpu6/checkpoints/`
- 实验内 `.log`:
  - `recursion_results/curriculum_sequential/<exp_name>/logs/<exp_name>.log`
