"""Cross-process protocol constants shared by batch_run.py and its subprocesses.

Single source for values that must agree across process boundaries. This
module deliberately has no dependencies: it cannot live in batch_run.py
(which pulls in visualize -> matplotlib) or core.py (which pulls in torch),
or the other side could not import it cheaply.
"""

# Marker written into merged configs by batch_run.build_merged_config;
# core.run_experiment refuses to run on an unmerged config (raw config.json
# lacks per-experiment fields).
BATCH_RUN_MERGED_FLAG = '_BATCH_RUN_MERGED'
