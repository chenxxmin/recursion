"""Error breakdown by gap structure of the prediction context.

Loads a trained model, generates random samples (corrupted with the experiment's
missing-value rule when configured), and classifies EVERY prediction by whether
the last two input tokens of its context are missing:

    (..., 0, 0)   both missing
    (..., 0, x)   second-to-last missing, last present
    (..., x, 0)   second-to-last present, last missing (the post-gap position)
    other (x, x)  both present

Prints correct/wrong counts per category. Here "0" means the missing token;
if the experiment has no missing configured, "0" falls back to the literal
token 0.

Usage (repo root):
    python src/error_breakdown.py <model.pth> [--json experiments/xxx.json]
           [--seed N] [--samples N] [--length L] [--clean]
"""
import argparse
import os
import random
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from analyze_attention import load_model
from core import RecurrenceDataset
from verify_sample import build_single_rule, find_exp_config

CATEGORIES = ['(...,0,0)', '(...,0,x)', '(...,x,0)', 'other(x,x)']


def corrupt_view(seq, cfg, p, init_len):
    """Corrupt a clean sample with the experiment's missing rule (dataset logic)."""
    n_rules = len(cfg.get('ab_pairs') or cfg.get('AB_PAIRS') or cfg.get('ABC_PAIRS') or [1])
    use_tag = cfg.get('use_ab_tag', cfg.get('USE_AB_TAG', False))
    missing_token = p + n_rules if (cfg.get('_is_mixed') and use_tag) else p
    helper = RecurrenceDataset(p=p, init_len=init_len, length=len(seq), verbose=False,
                               missing_prob=cfg['MISSING_PROB'],
                               miss_len=cfg.get('MISS_LEN', 1),
                               miss_second=cfg.get('MISS_SECOND', False),
                               missing_token=missing_token,
                               num_mask=cfg.get('NUM_MASK') or 0)
    window = torch.tensor(seq, dtype=torch.long)
    helper._corrupt(window, True)
    return window.tolist(), missing_token


def main():
    ap = argparse.ArgumentParser(description='Error counts by last-two-context gap structure.')
    ap.add_argument('pth', help='model checkpoint path')
    ap.add_argument('--json', default=None, help='experiments JSON that generated the model')
    ap.add_argument('--seed', type=int, default=0, help='seed for sample generation')
    ap.add_argument('--samples', type=int, default=200, help='number of random samples')
    ap.add_argument('--length', type=int, default=None, help='sample length (default: TRAIN_LEN)')
    ap.add_argument('--clean', action='store_true', help='do not corrupt samples')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model(args.pth, device=device)
    ckpt_cfg = checkpoint['config']

    task, exp_cfg = find_exp_config(args, ckpt_cfg)
    cfg = dict(ckpt_cfg)
    if exp_cfg:
        cfg.update(exp_cfg)
    cfg['P'] = cfg.get('P', cfg.get('p'))
    if task is None:
        task = cfg.get('recurrence') or ('mixed_ab' if cfg.get('ab_pairs') else 'addition')
    is_mixed = task in ('mixed_ab', 'mixed_abc') or (cfg.get('ab_pairs') and 'recurrence' not in cfg)
    cfg['_is_mixed'] = is_mixed

    pairs = None
    if is_mixed:
        pairs = cfg.get('ab_pairs') or cfg.get('AB_PAIRS') or cfg.get('ABC_PAIRS')
        p = cfg['P']
        order = cfg.get('order', len(pairs[0]))
        init_len = order
        def make_next(coeffs):
            return lambda s: sum(ci * si for ci, si in zip(coeffs, reversed(s[-len(coeffs):]))) % p
    else:
        init_len, next_fn, desc = build_single_rule(task, cfg)
        p = cfg['P']

    length = args.length or cfg.get('TRAIN_LEN', cfg.get('train_len', 16))
    num_mask = cfg.get('NUM_MASK') or 0
    missing_prob = cfg.get('MISSING_PROB', 0.0)
    corrupted = bool(missing_prob > 0 and not args.clean)
    random.seed(args.seed)

    use_ab_tag = bool(is_mixed and cfg.get('use_ab_tag', cfg.get('USE_AB_TAG', False)))
    use_cond = bool(is_mixed and cfg.get('use_conditional_wte', cfg.get('USE_CONDITIONAL_WTE', False)))

    # counts[category] = [correct, wrong]
    counts = {c: [0, 0] for c in CATEGORIES}
    missing_token_used = None

    for _ in range(args.samples):
        if is_mixed:
            rule_idx = random.randrange(len(pairs))
            next_fn = make_next(pairs[rule_idx])
        seq = [random.randrange(p) for _ in range(init_len)]
        while len(seq) < length:
            seq.append(next_fn(seq))

        if corrupted:
            view, missing_token_used = corrupt_view(seq, cfg, p, init_len)
        else:
            view = list(seq)
            missing_token_used = 0  # fallback: literal token 0

        model_in = ([p + rule_idx] + view) if use_ab_tag else list(view)
        x = torch.tensor([model_in], dtype=torch.long, device=device)
        with torch.no_grad():
            if use_cond:
                logits = model(x, ab_labels=torch.tensor([rule_idx], device=device))[0]
            else:
                logits = model(x)[0]
        preds = logits[0].argmax(dim=-1).tolist()

        # tag mode: input has the leading flag; true targets shift accordingly
        clean_targets = (([seq[0]] if use_ab_tag else []) + seq[1:])
        n = len(model_in) - 1
        for t in range(max(1, num_mask), n):  # t>=1: need two context tokens
            last2_missing = (model_in[t - 1] == missing_token_used)
            last1_missing = (model_in[t] == missing_token_used)
            if last2_missing and last1_missing:
                cat = CATEGORIES[0]
            elif last2_missing:
                cat = CATEGORIES[1]
            elif last1_missing:
                cat = CATEGORIES[2]
            else:
                cat = CATEGORIES[3]
            ok = preds[t] == clean_targets[t]
            counts[cat][0 if ok else 1] += 1

    miss_desc = (f"missing token id {missing_token_used}" if corrupted
                 else "no corruption: '0' = literal token 0")
    print('=' * 64)
    print(f"Model: {os.path.basename(args.pth)}")
    print(f"samples={args.samples}, length={length}, corrupted={corrupted} ({miss_desc})")
    if corrupted:
        print(f"missing rule: prob={missing_prob}, miss_len={cfg.get('MISS_LEN', 1)}, "
              f"miss_second={cfg.get('MISS_SECOND', False)}")
    print('=' * 64)
    print(f"{'category':<14}{'correct':>10}{'wrong':>10}{'total':>10}{'acc':>10}")
    tot_c = tot_w = 0
    for c in CATEGORIES:
        ok, bad = counts[c]
        tot = ok + bad
        tot_c += ok
        tot_w += bad
        acc = ok / tot if tot else float('nan')
        print(f"{c:<14}{ok:>10}{bad:>10}{tot:>10}{acc:>10.3f}")
    print('-' * 64)
    tot = tot_c + tot_w
    print(f"{'TOTAL':<14}{tot_c:>10}{tot_w:>10}{tot:>10}{tot_c / tot if tot else float('nan'):>10.3f}")


if __name__ == '__main__':
    main()
