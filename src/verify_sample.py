"""Random-sample prediction check for manual verification.

Load trained models, generate ONE random sample from the task's recurrence,
and print each model's next-token predictions aligned with the true values
position by position. If the experiment config enables MISSING_PROB, the
sample is corrupted with the same rule (RecurrenceDataset._corrupt) before
being shown to the model, so gap-filling can be checked by eye.

Usage (repo root):
    python src/verify_sample.py <name> [--seed N] [--length L] [--clean]

<name> selects experiments/<name>.json; every experiment's model is read from
/mnt/workspace/hujiachen/models/<name>/<exp>.pth. Missing/unreadable models are skipped and
reported at the end. All output is teed to verify_sample_output.log.
--clean forces an uncorrupted sample even when MISSING_PROB is configured.
"""
import argparse
import contextlib
import json
import os
import random
import sys

import torch

# Allow running from repo root as: python src/verify_sample.py <pth>
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from analyze_attention import load_model
from datasets import corrupt_window, missing_token_id
from rules import LinearRecurrenceRule, single_rule_from_task, task_from_save_config

COL_W = 5  # display width per position column
MODEL_BASE = '/mnt/workspace/hujiachen/models'


def build_single_rule(task, cfg):
    """Return (init_len, next_fn, description) for a single-rule task.

    Thin wrapper over rules.single_rule_from_task; next_fn closes over p and
    takes only the sequence.
    """
    p = cfg.get('P', cfg.get('p'))
    init_len, next_fn, desc = single_rule_from_task(task, cfg)
    return init_len, (lambda s: next_fn(s, p)), desc


def find_exp_config(pth_path, json_path):
    """Locate the experiment config in the JSON by model filename stem.

    Returns (task, exp_config or None).
    """
    stem = os.path.splitext(os.path.basename(pth_path))[0]
    if not json_path:
        return None, None
    import json
    with open(json_path, encoding='utf-8') as f:
        exps = json.load(f)['experiments']
    for e in exps:
        if e['name'] == stem:
            return e.get('task'), e['config']
    print(f"[warn] {stem} not found in {json_path}; falling back to checkpoint config")
    return None, None


class _Tee:
    """Minimal tee: write to multiple text streams at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def dispatch_and_log(args, script_stem, run_one):
    """Shared CLI dispatch: `args.target` may be a .pth file or a directory of
    .pth files (each is analyzed in turn); all printed output is teed to
    {script_stem}_output.log in the current directory."""
    import contextlib
    import glob

    if os.path.isdir(args.target):
        targets = sorted(glob.glob(os.path.join(args.target, '*.pth')))
        if not targets:
            print(f"No .pth files found in {args.target}")
            return
    else:
        targets = [args.target]

    out_path = f"{script_stem}_output.log"
    with open(out_path, 'w', encoding='utf-8') as f:
        tee = _Tee(sys.stdout, f)
        with contextlib.redirect_stdout(tee):
            for pth in targets:
                print('\n' + '#' * 70)
                print(f"# {pth}")
                print('#' * 70)
                try:
                    run_one(args, pth)
                except Exception as e:  # keep iterating over the directory
                    print(f"[error] {pth}: {type(e).__name__}: {e}")
    print(f"\n[output saved to {out_path}]")


def resolve_experiment_cfg(ckpt_cfg, exp_cfg, task):
    """Merge checkpoint + experiment configs and resolve the task name.

    Returns (cfg, task, is_mixed). task may be None: the checkpoint fallback
    applies (task_from_save_config normalizes the checkpoint vocabulary, e.g.
    'multiplicative' -> 'multiplication').
    """
    cfg = dict(ckpt_cfg)
    if exp_cfg:
        cfg.update(exp_cfg)
    cfg['P'] = cfg.get('P', cfg.get('p'))  # checkpoints save lowercase 'p'
    if task is None:
        task = task_from_save_config(cfg)
        if task is None:  # mixed / dynamic checkpoint
            task = cfg.get('recurrence') or ('mixed_ab' if cfg.get('ab_pairs') else 'addition')
    is_mixed = task in ('mixed_ab', 'mixed_abc') or (cfg.get('ab_pairs') and 'recurrence' not in cfg)
    return cfg, task, is_mixed


def run_one(args, exp_cfg, task, exp_name, pth_path):
    model, checkpoint = load_model(pth_path, device='cpu')
    cfg, task, is_mixed = resolve_experiment_cfg(checkpoint['config'], exp_cfg, task)

    # seed first so it also governs the mixed-rule pick
    if args.seed is not None:
        random.seed(args.seed)
    used_seed = args.seed if args.seed is not None else random.randrange(2**31)
    if args.seed is None:
        random.seed(used_seed)

    if is_mixed:
        pairs = cfg.get('ab_pairs') or cfg.get('AB_PAIRS') or cfg.get('ABC_PAIRS')
        p = cfg.get('P', cfg.get('p'))
        order = cfg.get('order', len(pairs[0]))
        rule_idx = random.randrange(len(pairs))
        coeffs = pairs[rule_idx]
        _next = LinearRecurrenceRule(coeffs=tuple(coeffs), p=p).next_fn()
        next_fn = lambda s: _next(s, p)
        init_len = order
        desc = f"{task} rule {rule_idx}: coeffs={coeffs} mod {p}"
    else:
        init_len, next_fn, desc = build_single_rule(task, cfg)
        rule_idx = None
        p = cfg['P']

    length = args.length or cfg.get('TRAIN_LEN', cfg.get('train_len', 16))

    # clean sample
    seq = [random.randrange(p) for _ in range(init_len)]
    while len(seq) < length:
        seq.append(next_fn(seq))

    # missing corruption (same rule as the dataset)
    missing_prob = cfg.get('MISSING_PROB', 0.0)
    corrupted = bool(missing_prob > 0 and not args.clean)
    if corrupted:
        window = torch.tensor(seq, dtype=torch.long)
        missing_token = corrupt_window(
            window, True, p=p, init_len=init_len, missing_prob=missing_prob,
            miss_len=cfg.get('MISS_LEN', 1), miss_second=cfg.get('MISS_SECOND', False),
            missing_token=missing_token_id(cfg, p, is_mixed))
        view = window.tolist()
    else:
        view = list(seq)
        missing_token = None

    # model input: mixed tag mode prepends the rule flag token
    use_ab_tag = bool(is_mixed and cfg.get('use_ab_tag', cfg.get('USE_AB_TAG', False)))
    use_cond = bool(is_mixed and cfg.get('use_conditional_wte', cfg.get('USE_CONDITIONAL_WTE', False)))
    model_in = list(view)
    if use_ab_tag:
        model_in = [p + rule_idx] + model_in
    x = torch.tensor([model_in], dtype=torch.long)
    with torch.no_grad():
        if use_cond:
            logits = model(x, ab_labels=torch.tensor([rule_idx]))[0]
        else:
            logits = model(x)[0]
    preds = logits[0].argmax(dim=-1).tolist()  # preds[t] = prediction for token t+1

    def fmt(v):
        return 'M' if v == missing_token and missing_token is not None else str(v)

    print('=' * 70)
    print(f"Model : {os.path.basename(pth_path)}")
    print(f"Task  : {desc}")
    print(f"Seed  : {used_seed}  | length {length}  | corrupted: {corrupted}"
          + (f" (prob={missing_prob}, miss_len={cfg.get('MISS_LEN', 1)}, "
             f"miss_second={cfg.get('MISS_SECOND', False)})" if corrupted else ""))
    print('=' * 70)

    n = len(model_in) - 1  # predictions cover targets 1..len-1
    if use_ab_tag:
        print(f"rule flag token: {model_in[0]} (= p + rule_idx)")
    print('true seq :' + ''.join(f'{fmt(v):>{COL_W}}' for v in seq))
    print('model in :' + ''.join(f'{fmt(v):>{COL_W}}' for v in model_in))
    print()
    # display true values from the CLEAN sequence (tag mode prepends the flag,
    # whose first target is x0)
    clean_targets = (([seq[0]] if use_ab_tag else []) + seq[1:])[:n]
    print('t        :' + ''.join(f'{t:>{COL_W}}' for t in range(n)))
    print('true next:' + ''.join(f'{v:>{COL_W}}' for v in clean_targets))
    print('pred     :' + ''.join(f'{fmt(v):>{COL_W}}' for v in preds[:n]))
    # compare prediction against the CLEAN value: the point is whether the
    # model recovers the true token, including through missing gaps
    marks = ['.' if a == b else 'x' for a, b in zip(preds[:n], clean_targets)]
    print('correct  :' + ''.join(f'{m:>{COL_W}}' for m in marks))
    n_ok = sum(1 for a, b in zip(preds[:n], clean_targets) if a == b)
    print(f"\naccuracy on this sample: {n_ok}/{n} = {n_ok / n:.1%}"
          " (positions whose input was corrupted count as failures if the clean"
          " value is not recovered)")


def main():
    ap = argparse.ArgumentParser(description='Print random sample(s) with aligned model predictions.')
    ap.add_argument('name', help='experiment batch name (experiments/<name>.json; '
                                 'models read from /mnt/workspace/hujiachen/models/<name>/<exp>.pth)')
    ap.add_argument('--seed', type=int, default=None, help='seed for sample generation (default: random)')
    ap.add_argument('--length', type=int, default=None, help='sample length (default: TRAIN_LEN from config)')
    ap.add_argument('--clean', action='store_true', help='do not corrupt the sample even if MISSING_PROB is set')
    args = ap.parse_args()

    exp_path = f'experiments/{args.name}.json'
    with open(exp_path, encoding='utf-8') as f:
        experiments = json.load(f)['experiments']

    out_path = 'verify_sample_output.log'
    missing = []
    with open(out_path, 'w', encoding='utf-8') as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            for e in experiments:
                exp_name = e['name']
                print('\n' + '#' * 70)
                print(f'# {exp_name}')
                print('#' * 70)
                pth = os.path.join(MODEL_BASE, args.name, f'{exp_name}.pth')
                if not os.path.exists(pth):
                    missing.append(exp_name)
                    print(f'[skip] model not found: {pth}')
                    continue
                try:
                    run_one(args, e['config'], e.get('task'), exp_name, pth)
                except Exception as exc:  # keep iterating over the batch
                    missing.append(f'{exp_name} (error: {type(exc).__name__}: {exc})')
                    print(f'[error] {type(exc).__name__}: {exc}')

            print('\n' + '=' * 70)
            print(f'MISSING REPORT ({len(missing)}/{len(experiments)} skipped):')
            for m in missing:
                print(f'  {m}')
            if not missing:
                print('  none — all experiments processed')
    print(f'\n[output saved to {out_path}]')


if __name__ == '__main__':
    main()
