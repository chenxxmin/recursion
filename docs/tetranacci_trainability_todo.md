# Tetranacci trainability TODO

Current symptom: tetranacci train loss itself does not decrease adequately, so
the immediate problem is optimization / learnability rather than held-out
generalization.

## Verified in code

- Dataset recurrence for tetranacci is
  `A*x[t-1] + B*x[t-2] + C*x[t-3] + D*x[t-4] (mod P)`.
- With `NUM_MASK=3` and `targets = x[:, 1:]`, the first supervised target is
  `x4`; `x1,x2,x3` are excluded from the loss.
- Current tetranacci experiments use clean inputs (`MISSING_PROB=0`), so the
  collate path does not alter values or target alignment.
- Added `tests/test_tetranacci_pipeline.py` to assert the post-collate
  sequences, shifted targets, and effective loss positions satisfy the exact
  order-4 recurrence.

## In progress

- Tiny fixed-training-set memorization test: determine whether the model can
  drive training loss near zero on 32 / 128 / 1024 examples.

## Next diagnostics, in order

1. Instrument pre-clip global gradient norm and clipping frequency for d1024
   versus d4096. The current trainer always applies global
   `clip_grad_norm_(..., 1.0)`.
2. If clipping is persistent/severe, compare clip thresholds 1, 5, 10, and no
   clipping while keeping all other hyperparameters fixed.
3. Sweep learning rate around the current baseline after the clipping regime
   is understood.
4. Measure optimizer-update norm versus AdamW decay-update norm before changing
   `WEIGHT_DECAY=1`; then test 0.1 / 0.01 only if decay is comparable to the
   gradient-driven update.
5. Diagnostic objective: train only the first valid transition `x4`. If this
   succeeds but supervising `x4..x15` fails, investigate multi-position
   gradient interference.
6. If optimization still fails, compare more depth at controlled width
   (e.g. l4/l6) rather than continuing to increase width/heads.
7. Warm-start tetranacci from a successfully trained tribonacci checkpoint to
   distinguish an optimization barrier from representational incapacity.
