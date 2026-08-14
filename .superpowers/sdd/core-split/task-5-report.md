# Task 5 report — trim core.py re-exports + update docs §7 source map

Status: DONE. Commit on main (created after this report): "Split core.py
step 5: trim core.py re-exports, update docs source map".

core.py went from 40 lines / 48 re-exported symbols to 28 lines / 18
re-exported symbols. Only the shell's import list and module docstrings were
touched — zero running logic changed.

## Deleted re-exports (30 symbols) — each justified by grep

Repo-wide grep for `core.<symbol>` (pattern covering all 30 names, run over
the whole repo incl. docs): the ONLY hit was `REFACTOR_LOG.md:674`
(`core._prepare_dynamic_mixed` in historical changelog prose — not code).
No live code reference exists for any deleted symbol. `getattr(core, ...)` /
`importlib` / `vars(core)` / `core.__dict__` dynamic access: grep — zero
hits repo-wide. No analysis script or test import was modified (brief rule
respected — trim achieved purely by deleting from core.py's list).

- from models (6): `RotaryEmbedding`, `ATTN_MASK_NEG`,
  `CausalSelfAttention`, `TransformerBlock`, `_lm_loss`, `RULE_LOSS_WEIGHT`
- from datasets (4): `MixedRecurrenceDataset`, `DynamicMixedDataset`,
  `BucketBatchSampler`, `dynamic_mixed_collate_fn`
- from training (7): `freeze_partial`, `_print_nan_diagnostics`,
  `_accumulate_accuracy`, `_default_loss_mask`, `train_epoch`, `evaluate`,
  `run_training_engine`
- from final_eval (8, entire block removed): `EVAL_BATCH_SIZE`,
  `_teacher_forced_correct`, `_safe_div`, `_split_exposure_stats`,
  `_print_exposure_stats`, `_print_per_position_exposure`,
  `_run_mixed_ab_final_test`, `_run_single_recurrence_final_test`
- from experiment (5): `_round_up_pow2`, `_make_loaders`,
  `_print_task_banner`, `_prepare_dynamic_mixed`, `_prepare_single_recurrence`

`rotate_half` remains NOT re-exported (task-1 decision, zero consumers).

## Kept re-exports (18 symbols) — each with its external reference point

- `core.MixedABTransformer` — src/analyze_attention.py:71,
  src/verify_circle.py:209, tests/test_rule_start_offset.py:17
- `core.FibonacciTransformer` — src/analyze_attention.py:73
- `core.apply_rotary_emb` — src/analyze_attention.py:161-162
- `core.RecurrenceDataset` — src/analyze_attention.py:544,
  tests/test_mixed_ab_compat.py:49, tests/test_missing_values.py:20
- `core.generate_dynamic_sample` — src/analyze_attention.py:422
- `core.corrupt_window` — src/analyze_attention.py:454, src/rule_fit.py:46,
  src/error_breakdown.py:35, src/verify_sample.py:32
- `core.missing_token_id` — src/analyze_attention.py:458,
  src/error_breakdown.py:35, src/verify_sample.py:32
- `from core import BatchTag` — tests/test_missing_values.py:20
- `from core import collate_fn` — tests/test_missing_values.py:20
- `from core import collate_fn_masked` — tests/test_missing_values.py:172
- `from core import mixed_ab_collate_fn` — tests/test_missing_values.py:213
- `from core import mixed_ab_collate_fn_masked` — tests/test_missing_values.py:221
- `from core import collate_fn_predict` — tests/test_missing_values.py:321
- `from core import mixed_ab_collate_fn_predict` — tests/test_missing_values.py:344
- `from core import _unpack_batch` — tests/test_missing_values.py:221/321/344
- `from core import _sample_seq` — tests/test_missing_values.py:280/293
- `from core import run_experiment` — tests/test_training_smoke.py:17; also
  core.py's own `__main__` guard (hard constraint 1)
- `core._prepare_mixed_recurrence` — tests/test_prepare_mixed_recurrence.py:26/44/63

Not counted as references (per task-4 review facts, re-verified):
`core.MixedABDataset` in tests/test_mixed_ab_compat.py:5 (docstring text;
class no longer exists), `core._sample_seq` in test_missing_values.py:279
(docstring), all `core.` mentions in src/*.py docstrings, batch_run.py:52/186
(spawn target, not attribute access), protocol.py comments, run_queued.sh
(pgrep pattern `core\.py` — still matches the file), docs/superpowers/*
(historical 2026-07-22 plan/spec docs), REFACTOR_LOG.md (changelog).

External reference surface check (brief item 1):
- src/batch_run.py: no `import core` / `from core import` — only spawns
  `core.py` as subprocess (line 186). Confirmed.
- New modules importing core: `import core|from core` in
  models/datasets/training/final_eval/experiment/rules/protocol — zero
  (only docstring prose mentions). Confirmed 0.

## Docs change: recursion_model_and_training_settings.md §7

Updated the source-location table (rows 数据生成 through 配置入口) to point
at the actual post-split modules, verified by grep against the real
definitions (models.py:8/31/42/149/224, datasets.py:17/404/468,
training.py:17/162/210/258, experiment.py:384):

- 数据生成: core.py/mixed_dataset.py → all three datasets in `src/datasets.py`
  (the `src/mixed_dataset.py` entry was stale since step 2's merge — fixed
  as part of making the table accurate)
- 模型定义 / 注意力层 / RoPE 实现: core.py → `src/models.py`
- 训练流程 / 参数冻结: core.py → `src/training.py`
- 新增一行 训练后最终生成测试 → `src/final_eval.py::_run_mixed_ab_final_test`,
  `_run_single_recurrence_final_test` (final_eval previously had no row;
  added so every split module appears in the map)
- 配置入口: `src/core.py::run_experiment` → `src/experiment.py::run_experiment`,
  with the row note updated to say core.py is the entry shell re-exporting
  it and batch_run.py still spawns `python src/core.py <config>`

Table format unchanged (same 功能/文件与位置 two-column style).

## Docstring sweep (no logic touched)

The step 1-4 module docstrings claimed "core.py re-exports every symbol
defined here" — now false after the trim. Updated one sentence each in
src/datasets.py, src/training.py, src/final_eval.py (now states core.py does
not re-export its symbols), src/experiment.py (now names the two re-exported
symbols). core.py's own docstring updated to describe the trimmed policy.

## Verification (all commands from the brief; torch python =
C:/Users/Chen/anaconda3/envs/torch/python.exe)

Test suite (`cd tests; for t in test_*.py; ...`):

```
PASS test_config_mixed_abc.py
PASS test_missing_values.py
PASS test_mixed_ab_compat.py
PASS test_mixed_dataset.py
PASS test_prepare_mixed_recurrence.py
PASS test_rule_start_offset.py
PASS test_rules.py
PASS test_training_smoke.py
```

Golden stdout (golden_run.py spawns `python src/core.py <cfg>`, the
production entry path):

```
addition: captured 44 lines; mixed_ab: captured 62 lines
diff -u golden/addition.txt golden_after/addition.txt   → EMPTY
diff -u golden/mixed_ab.txt golden_after/mixed_ab.txt   → EMPTY
rm -rf golden_after → done
```

Analysis-script import check:

```
python -c "import sys; sys.path.insert(0,'src'); import analyze_attention,
verify_circle, rule_fit, error_breakdown, verify_sample, batch_run"
→ imports OK
```

Entry-point constraints:

- `cd src && python -c "from core import run_experiment"` → OK
- `python src/core.py` (no args) → usage line on stderr, exit=2 (unchanged)
- `__main__` guard verbatim: `run_experiment(sys.argv[1])` (hard constraint 1)
- protocol.py untouched; BATCH_RUN_MERGED_FLAG not moved (hard constraint 3)
- No running logic touched anywhere (hard constraint 2) — confirmed by empty
  golden diffs on both tasks.

## Self-review

- Every deleted re-export: zero external references (single combined grep +
  dynamic-access grep, results above).
- Every kept symbol: at least one named external reference point (list above).
- Docs table verified against actual `class`/`def` locations via grep.

## Concerns / follow-ups (non-blocking)

1. §3/§4 body text of recursion_model_and_training_settings.md still has
   stale pointers (`src/core.py::FibonacciTransformer` line 225,
   `::MixedABTransformer` 286, `::RecurrenceDataset` 341,
   `src/mixed_dataset.py::MixedRecurrenceDataset` 350,
   `::DynamicMixedDataset` 359). The brief scoped this task to the §7 table
   only; flagged for a future docs pass.
2. Historical docs (docs/superpowers/plans|specs/2026-07-22-*.md,
   REFACTOR_LOG.md) reference `core.<symbol>` names including now-deleted
   re-exports — historical documents describing pre-split state, left
   untouched by design.
