"""Compatibility shell for the recurrence experiment entry point.

Implementation lives in models.py / datasets.py / training.py / scenarios.py /
experiment.py. This module re-exports only the symbols the test suite still
references via `from core import ...` / `core.<name>`, and keeps the __main__
entry that batch_run.py spawns as `python src/core.py <merged_config.json>`.
"""
import sys

from datasets import (RecurrenceDataset, BatchTag, collate_fn,
                      make_missing_collate, make_mixed_missing_collate,
                      make_action_missing_collate,
                      mixed_ab_collate_fn)
from training import _unpack_batch
from experiment import _prepare_mixed_recurrence, run_experiment


if __name__ == '__main__':
    # Entry point for the training subprocess spawned by batch_run.py:
    #   python src/core.py <merged_config.json>
    if len(sys.argv) != 2:
        print("Usage: python src/core.py <merged_config.json>", file=sys.stderr)
        sys.exit(2)
    run_experiment(sys.argv[1])
