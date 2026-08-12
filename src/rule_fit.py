"""Fit linear recurrence formulas to a trained model's predictions, over F_p.

For each prediction position, the model's argmax prediction is regressed
(exactly, mod p) onto candidate feature sets built from the visible context:

  set A: the true-rule positions (d0..d{order-1}, i.e. x_i, x_{i-1}, ...)
  set C: A + extra distances observed in attention (default d7)

An exact solution (agreement 100%) means the model's predictions ARE that
linear formula on those positions; a RANSAC best-effort fit is reported
otherwise. With --log, positions are split into FRONT/BACK segments by the
shortcut split rule (see analyze_attention.compute_front_split).

Usage (repo root):
    python src/rule_fit.py <model.pth|dir> [experiments/xxx.json]
           [--log path] [--samples N] [--seed N] [--length L] [--extra-dists 7 8]

Output is teed to rule_fit_output.log.
"""
import argparse
import os
import random
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from analyze_attention import compute_front_split, load_model
from core import RecurrenceDataset
from verify_sample import build_single_rule, dispatch_and_log, find_exp_config


def solve_mod_p(rows, p):
    """Exact linear solve over F_p via Gaussian elimination.

    rows: list of [x1..xk, y]. Returns the coefficient list when the system
    has a unique exact solution, else None (inconsistent or underdetermined).
    """
    A = [[v % p for v in r] for r in rows]
    k = len(A[0]) - 1
    piv_row = 0
    for col in range(k):
        piv = next((r for r in range(piv_row, len(A)) if A[r][col] != 0), None)
        if piv is None:
            return None
        A[piv_row], A[piv] = A[piv], A[piv_row]
        inv = pow(A[piv_row][col], p - 2, p)
        A[piv_row] = [v * inv % p for v in A[piv_row]]
        for r in range(len(A)):
            if r != piv_row and A[r][col] != 0:
                f = A[r][col]
                A[r] = [(a - f * b) % p for a, b in zip(A[r], A[piv_row])]
        piv_row += 1
        if piv_row == len(A):
            break
    if piv_row < k:
        return None
    for r in A:
        if all(v == 0 for v in r[:k]) and r[k] != 0:
            return None
    return [A[r][k] for r in range(k)]


def fit_and_score(X, y, p, rounds=200):
    """Return (coeffs, agreement, exact). Exact solve first; RANSAC fallback."""
    k = len(X[0])
    coeffs = solve_mod_p([xi + [yi] for xi, yi in zip(X, y)], p)
    if coeffs is not None:
        return coeffs, 1.0, True
    best_c, best_acc = None, -1.0
    for _ in range(rounds):
        # RANSAC: solve exactly from k independent rows, score on all data
        idx = random.sample(range(len(X)), k)
        c = solve_mod_p([[X[i][j] for j in range(k)] + [y[i]] for i in idx], p)
        if c is None:
            continue
        acc = sum(1 for xi, yi in zip(X, y)
                  if sum(ci * vi for ci, vi in zip(c, xi)) % p == yi) / len(X)
        if acc > best_acc:
            best_c, best_acc = c, acc
    return best_c, best_acc, False


def fmt_equation(dists, coeffs, p):
    terms = ' + '.join(f'{c}*x_{{t-{d}}}' if d else f'{c}*x_t'
                       for d, c in zip(dists, coeffs))
    return f'x_{{t+1}} = {terms}  (mod {p})'


def run_one(args, pth_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model(pth_path, device=device)
    ckpt_cfg = checkpoint['config']

    task, exp_cfg = find_exp_config(pth_path, args.json)
    cfg = dict(ckpt_cfg)
    if exp_cfg:
        cfg.update(exp_cfg)
    cfg['P'] = cfg.get('P', cfg.get('p'))
    if task is None:
        task = cfg.get('recurrence') or ('mixed_ab' if cfg.get('ab_pairs') else 'addition')
    if task in ('mixed_ab', 'mixed_abc') or (cfg.get('ab_pairs') and 'recurrence' not in cfg):
        print('[skip] mixed tasks are not supported by rule_fit (single-rule only)')
        return

    init_len, next_fn, desc = build_single_rule(task, cfg)
    p = cfg['P']
    length = args.length or cfg.get('TRAIN_LEN', cfg.get('train_len', 16))
    num_mask = cfg.get('NUM_MASK') or 0
    missing_prob = cfg.get('MISSING_PROB', 0.0)
    random.seed(args.seed)

    # position segments (query position t predicts x_{t+1})
    lo = max(num_mask, init_len - 1)
    segments = [('ALL', None)]
    if args.log and os.path.exists(args.log):
        start_x, k = compute_front_split(args.log)
        if k:
            segments = [('FRONT', (start_x - 1, start_x - 1 + k)),
                        ('BACK', (start_x - 1 + k, length - 1))]
            print(f'split from log: front k={k} (start_x={start_x})')
        else:
            print(f'split from log: no shortcut (k={k}); fitting ALL positions')

    # candidate feature sets as attention distances
    set_A = list(range(init_len))                       # d0..d{order-1}
    extra = [d for d in args.extra_dists if d < length - 1]
    sets = [('A (true-rule positions)', set_A)]
    if extra:
        sets.append(('C (A + attention dists)', set_A + extra))

    # collect features/predictions per segment
    buckets = {(seg, name): ([], []) for seg, _ in segments for name, _ in sets}
    true_counts = {(seg): [0, 0] for seg, _ in segments}  # [match, total] vs true rule

    for _ in range(args.samples):
        seq = [random.randrange(p) for _ in range(init_len)]
        while len(seq) < length:
            seq.append(next_fn(seq))
        view = list(seq)
        miss_tok = None
        if missing_prob > 0 and not args.clean:
            helper = RecurrenceDataset(p=p, init_len=init_len, length=length, verbose=False,
                                       missing_prob=missing_prob,
                                       miss_len=cfg.get('MISS_LEN', 1),
                                       miss_second=cfg.get('MISS_SECOND', False),
                                       num_mask=num_mask)
            window = torch.tensor(view, dtype=torch.long)
            helper._corrupt(window, True)
            view = window.tolist()
            miss_tok = p
        x = torch.tensor([view], dtype=torch.long, device=device)
        with torch.no_grad():
            preds = model(x)[0][0].argmax(dim=-1).tolist()

        for seg, rng in segments:
            if rng is None:
                positions = range(lo, length - 1)
            else:
                positions = range(max(lo, rng[0]), rng[1])
            for t in positions:
                pred = preds[t]
                true_next = seq[t + 1]
                key = seg
                true_counts[key][1] += 1
                if pred == true_next:
                    true_counts[key][0] += 1
                for name, dists in sets:
                    feats = []
                    ok = True
                    for d in dists:
                        v = view[t - d]
                        if miss_tok is not None and v == miss_tok:
                            ok = False
                            break
                        feats.append(v)
                    if ok:
                        buckets[(seg, name)][0].append(feats)
                        buckets[(seg, name)][1].append(pred)

    print(f'Model: {os.path.basename(pth_path)}')
    print(f'Rule : {desc} | samples={args.samples}, length={length}, '
          f'missing_prob={missing_prob}')
    for seg, _ in segments:
        m, tot = true_counts[seg]
        print(f'\n== segment {seg}: model-vs-truth agreement {m}/{tot} '
              f'= {m / tot:.1%}' if tot else f'\n== segment {seg}: no positions')
        for name, dists in sets:
            X, y = buckets[(seg, name)]
            if not X:
                print(f'  set {name}: no usable positions (all involve missing tokens)')
                continue
            coeffs, acc, exact = fit_and_score(X, y, p)
            if coeffs is None:
                print(f'  set {name}: no linear fit found (best agreement {acc:.1%})')
            else:
                tag = 'EXACT' if exact else f'best-effort, agreement {acc:.1%}'
                print(f'  set {name} d{dists}: {fmt_equation(dists, coeffs, p)}  [{tag}]')


def main():
    ap = argparse.ArgumentParser(description='Fit linear formulas to model predictions over F_p.')
    ap.add_argument('target', help='model .pth file, or a directory of .pth files')
    ap.add_argument('json', nargs='?', default=None, help='experiments JSON (matched by filename stem)')
    ap.add_argument('--log', default=None, help='training log for front/back split (optional)')
    ap.add_argument('--samples', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--length', type=int, default=None)
    ap.add_argument('--extra-dists', type=int, nargs='*', default=[7],
                    help='extra attention distances for candidate set C (default: 7)')
    ap.add_argument('--clean', action='store_true')
    args = ap.parse_args()
    stem = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    dispatch_and_log(args, stem, run_one)


if __name__ == '__main__':
    main()
