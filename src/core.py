"""Compatibility shell for the recurrence experiment entry point.

Implementation lives in models.py / datasets.py / training.py / final_eval.py /
experiment.py (core.py split, steps 1-4). This module only re-exports their
symbols so existing `from core import ...` / `core.<name>` users (analysis
scripts, tests) keep working unchanged, and keeps the __main__ entry that
batch_run.py spawns as `python src/core.py <merged_config.json>`.
"""
import sys

from models import (RotaryEmbedding, apply_rotary_emb, ATTN_MASK_NEG,
                    CausalSelfAttention, TransformerBlock, _lm_loss,
                    FibonacciTransformer, MixedABTransformer, RULE_LOSS_WEIGHT)
from datasets import (RecurrenceDataset, MixedRecurrenceDataset,
                      DynamicMixedDataset, BucketBatchSampler, BatchTag,
                      collate_fn, collate_fn_masked, collate_fn_predict,
                      mixed_ab_collate_fn, mixed_ab_collate_fn_masked,
                      mixed_ab_collate_fn_predict, dynamic_mixed_collate_fn,
                      generate_dynamic_sample, corrupt_window, missing_token_id)
from training import (freeze_partial, _unpack_batch, _sample_seq,
                      _print_nan_diagnostics, _accumulate_accuracy,
                      _default_loss_mask, train_epoch, evaluate,
                      run_training_engine)
from final_eval import (EVAL_BATCH_SIZE, _teacher_forced_correct, _safe_div,
                        _split_exposure_stats, _print_exposure_stats,
                        _print_per_position_exposure,
                        _run_mixed_ab_final_test,
                        _run_single_recurrence_final_test)
from experiment import (_round_up_pow2, _make_loaders, _print_task_banner,
                        _prepare_mixed_recurrence, _prepare_dynamic_mixed,
                        _prepare_single_recurrence, run_experiment)


if __name__ == '__main__':
    # Entry point for the training subprocess spawned by batch_run.py:
    #   python src/core.py <merged_config.json>
    if len(sys.argv) != 2:
        print("Usage: python src/core.py <merged_config.json>", file=sys.stderr)
        sys.exit(2)
    run_experiment(sys.argv[1])
