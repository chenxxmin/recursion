"""Compatibility shell for the recurrence experiment entry point.

Implementation lives in models.py / datasets.py / training.py / final_eval.py /
experiment.py (core.py split, steps 1-4). This module re-exports only the
symbols that external `from core import ...` / `core.<name>` users (analysis
scripts, tests) actually reference (core.py split, step 5), and keeps the
__main__ entry that batch_run.py spawns as
`python src/core.py <merged_config.json>`.
"""
import sys

from models import MixedABTransformer, FibonacciTransformer, apply_rotary_emb
from datasets import (RecurrenceDataset, BatchTag, collate_fn,
                      collate_fn_masked, collate_fn_predict,
                      mixed_ab_collate_fn, mixed_ab_collate_fn_masked,
                      mixed_ab_collate_fn_predict, generate_dynamic_sample,
                      corrupt_window, missing_token_id)
from training import _unpack_batch, _sample_seq
from experiment import _prepare_mixed_recurrence, run_experiment


if __name__ == '__main__':
    # Entry point for the training subprocess spawned by batch_run.py:
    #   python src/core.py <merged_config.json>
    if len(sys.argv) != 2:
        print("Usage: python src/core.py <merged_config.json>", file=sys.stderr)
        sys.exit(2)
    run_experiment(sys.argv[1])
