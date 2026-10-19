# H1 binary modular-function controls

The standalone driver src/binary_arithmetic.py compares two architectures on
the same supervised function:

    q2 = (a*q1 + b*q0) % p,  (q0,q1) in F_p^2.

This distinguishes binary-function learnability from the additional context
and multi-position supervision of the ordinary recurrence experiments.

## Run from repository root

Single two-token Transformer:

~~~bash
python src/binary_arithmetic.py experiments/binary_arithmetic_h1_p127.json \
  --model transformer --rule 62,59 --seed 17996
~~~

Same split, tied-embedding MLP:

~~~bash
python src/binary_arithmetic.py experiments/binary_arithmetic_h1_p127.json \
  --model mlp --rule 62,59 --seed 17996
~~~

All four H1 coefficients x two models x three seeds:

~~~bash
python src/binary_arithmetic.py experiments/binary_arithmetic_h1_p127.json
~~~

Coefficient-independent partition of the state pairs (important control):

~~~bash
python src/binary_arithmetic.py experiments/binary_arithmetic_h1_p127.json \
  --split fixed_states --model both
~~~

One-epoch smoke test:

~~~bash
python src/binary_arithmetic.py experiments/binary_arithmetic_h1_p127.json \
  --model mlp --rule 62,59 --seed 17996 --epochs 1 \
  --out-dir /tmp/binary_arithmetic_smoke
~~~

## Design choices

- Input token order is [q0,q1] (oldest to newest). Prediction is
  (a*q1+b*q0)%p. This exactly matches the order-2 single-task prefix.
- Transformer is the unchanged FibonacciTransformer with causal RoPE.
  It predicts q2 at token position 1. The input [q0,q1,q2] is passed
  through the original next-token engine, but only the position 1 output
  contributes to loss (num_mask=1). By causality, it never sees q2.
- MLP applies one shared embedding table to q0 and q1 and concatenates
  the embeddings; output weights are tied to that same table. It
  ignores the q2 token in its input.
- Optimizer/training reuse BucketBatchSampler, run_training_engine,
  AdamW, cosine LR decay, gradient clipping, early stopping, checkpoints.
  Both models are width 128 by default but NOT parameter-count matched.
- full_split (default): exactly the existing orbit-traversal split over
  2D states, with the same split_seed. A boundary orbit may straddle
  train/test. Across coefficients, train state membership may differ.
- fixed_states: a single random shuffle of all p^2 starting pairs
  using split_seed. All coefficients use identical train/test pairs;
  unlike full_split, an orbit may appear on both sides.
- In both modes, no input pair appears in both train and test. The
  held-out set is exhaustively evaluated; random baseline is 1/p.
- The experiment writes checkpoints and resolved configurations to
  binary_arithmetic_results/model_a.../ and a summary.csv in the root.
  Evaluation curves are in stdout; redirect to a log file if needed.

## Interpretation cautions

Cross-model differences do not isolate attention alone: the architectures
differ in residual pathways, normalization and parameter count.
Compare multiple random seeds and optimizer steps before claiming a
function-specific inductive bias. The 2-token experiment does not by
itself identify an internal computational circuit.
