# core.py 拆分计划（待执行，另开会话）

> 背景：2026-08-14 大审查后唯一遗留的架构项。core.py 目前 ~1750 行、7 类职责，
> 且与 mixed_dataset.py 存在靠函数级 import 硬扛的循环依赖。
> 本文档是交给执行会话的完整交接。

## 拆分目标

```
src/
  rules.py            （已有，不动）
  protocol.py         （已有，不动）
  models.py           ← RotaryEmbedding, apply_rotary_emb, ATTN_MASK_NEG,
                        CausalSelfAttention, TransformerBlock, _lm_loss,
                        FibonacciTransformer, MixedABTransformer, RULE_LOSS_WEIGHT
  datasets.py         ← RecurrenceDataset, BucketBatchSampler, BatchTag + 全部 collate,
                        DynamicMixedDataset, generate_dynamic_sample,
                        corrupt_window, missing_token_id,
                        并入 mixed_dataset.py 的 MixedRecurrenceDataset
  training.py         ← freeze_partial, _unpack_batch, _sample_seq, _print_nan_diagnostics,
                        _accumulate_accuracy, _default_loss_mask,
                        train_epoch, evaluate, run_training_engine
  final_eval.py       ← EVAL_BATCH_SIZE, _safe_div, _split_exposure_stats, _print_*,
                        _teacher_forced_correct，以及 run_experiment 里两段 stage-3
  experiment.py       ← _round_up_pow2, _make_loaders, _print_task_banner,
                        三个 _prepare_*，run_experiment（瘦身为编排函数）
  core.py             ← 过渡期只留 re-export + __main__ 入口（见约束 1）
```

## 硬性约束（违反即破坏生产/分析链路）

1. **入口不可断**：`batch_run.py` 以 `python src/core.py <config>` 派生训练子进程；
   `run_queued.sh` 用 pgrep 匹配 `core\.py`。过渡期 core.py 必须保留
   `if __name__ == '__main__': run_experiment(sys.argv[1])` 且 `from core import run_experiment` 可用。
2. **分析脚本的 import**：analyze_attention / verify_circle / verify_sample / rule_fit /
   error_breakdown 直接引用 `core.FibonacciTransformer`、`core.MixedABTransformer`、
   `core.apply_rotary_emb`、`core.RecurrenceDataset`、`core.corrupt_window` 等。
   要么逐一改 import，要么 core.py re-export 全部公开符号（推荐后者，逐步收敛）。
3. **RNG 调用序不可变**：数据集生成依赖全局 random 的调用次序
   （tests/test_mixed_ab_compat.py 钉住了 MixedRecurrenceDataset 与 legacy 实现的
   逐样本等价）。移动代码时不得增删任何 random 调用。
4. **行为等价验证**：每步完成后跑：
   - 全套 `tests/test_*.py`（每个文件都有 `__main__` runner；本机用
     `C:/Users/Chen/anaconda3/envs/torch/python.exe`，base 环境无 torch）
   - 黄金 stdout 比对：tests/test_training_smoke.py 已是常驻冒烟；如需更细，
     可用 2-epoch 小 config（addition + mixed_ab）改动前后 stdout 逐行 diff。

## 推荐执行顺序（每步独立验证）

1. **models.py**（风险最低）：纯 nn 代码，无内部依赖。同步把分析脚本里
   `core.FibonacciTransformer` 等引用改为 models（或先靠 core re-export）。
2. **datasets.py**（结构收益最大）：并入 mixed_dataset.py，**消除 core↔mixed_dataset
   循环依赖**；compat 测试直接验证等价性。
3. **training.py**：依赖 models + datasets。
4. **final_eval.py / experiment.py**：最后做；run_experiment 瘦身时保持 core.py 壳可用。
5. 收尾：删除 core.py 壳里已无人用的 re-export；更新
   `recursion_model_and_training_settings.md` §7 源码位置表。

## 已知的小坑

- `MixedABTransformer.__init__` 会改写 kwargs 里的 vocab_size/pad_token_id（use_ab_tag 时），
  `analyze_attention.load_model` 依赖这个行为从 checkpoint 重建模型。
- `protocol.py` 故意无依赖：不要把 BATCH_RUN_MERGED_FLAG 挪进任何带 torch/matplotlib 的模块。
- `core._prepare_mixed_recurrence` 里的函数级 `from mixed_dataset import ...`
  注释（循环依赖说明）在第 2 步完成后应删除。

## 当时审查发现但决定不动的事项（供参考，勿顺手"改进"）

- `evaluate` 的默认 loss mask 有意不带 first_task_weight（保持 eval loss 历史可比）。
- `BucketBatchSampler` 每个 epoch 的 batch 组成固定（构造时 shuffle 一次）——是有意的
  可复现性选择还是疏漏，尚未定论；改动属于行为变更，需单独评估。
- dynamic_mixed 的 stage-3 最终测试不存在（设计如此）。
