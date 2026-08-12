"""Fit linear recurrence formulas to trained models' predictions, over F_p.

Given an experiment batch NAME, this script:
  1. reads experiments/<name>.json and iterates its experiments
  2. for each experiment, finds the training log (attention analysis section)
     and the model checkpoint
  3. dumps the len x len attention matrix as a symbol grid
     (　<0.05, · 0.05~0.1, ○ 0.1~0.25, × 0.25~0.5, ※ 0.5~1)
  4. auto-builds candidate feature set C = the distances whose mean attention
     is >= 0.05 (union over layers/heads, per front/back segment)
  5. fits the model's argmax predictions on set A (true-rule positions, from
     the a,b[,c] config) and set C, exactly over F_p (RANSAC fallback)
  6. reports experiments with missing log/model at the end

Paths:
  model: /data/cxm/models/<name>/<exp>.pth
  log:   /data/cxm/models/<name>/logs/<exp>.log  or
         /data/cxm/recursion/<name>/logs/<exp>.log

Usage (repo root):  python src/rule_fit.py <name> [--samples N] [--seed N] [--length L] [--clean]
Output is teed to rule_fit_output.log.
"""
import argparse
import contextlib
import json
import os
import random
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from analyze_attention import load_model
from core import RecurrenceDataset
from report_front_back_attention import dump_matrices, focus_by_distance, parse_log, split_k
from verify_sample import _Tee, build_single_rule

MODEL_BASE = '/data/cxm/models'
LOG_BASE_CANDIDATES = ['/data/cxm/models/{name}/logs', '/data/cxm/recursion/{name}/logs']
ATTN_MIN = 0.05  # distances with mean attention >= this go into candidate set C


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


def significant_dists(attn, qpos):
    """Union over layers/heads of distances with mean attention >= ATTN_MIN."""
    sig = set()
    for layer in attn.values():
        for mat in layer.values():
            fbd = focus_by_distance(mat, qpos)
            sig |= {d for d, v in fbd.items() if v >= ATTN_MIN}
    return sorted(sig)


def run_one(args, exp_cfg, task, exp_name, pth_path, log_path):
    model, checkpoint = load_model(pth_path, device='cpu')
    cfg = dict(exp_cfg)
    cfg['P'] = cfg.get('P', cfg.get('p'))
    if task in ('mixed_ab', 'mixed_abc'):
        print('[skip] mixed tasks are not supported by rule_fit (single-rule only)')
        return

    init_len, next_fn, desc = build_single_rule(task, cfg)
    p = cfg['P']
    length = args.length or cfg.get('TRAIN_LEN', 16)
    num_mask = cfg.get('NUM_MASK') or 0
    missing_prob = cfg.get('MISSING_PROB', 0.0)
    random.seed(args.seed)

    # --- log: split + attention matrices
    start_x, series, attn = parse_log(log_path)
    k = split_k(series)
    T_att = max((i for layer in attn.values() for head in layer.values() for i in head),
                default=-1) + 1

    print(f'Model: {exp_name}')
    print(f'Rule : {desc} | length={length}, num_mask={num_mask}, missing_prob={missing_prob}')
    print(f'Log  : front k={k}, attention matrix {"present" if attn else "MISSING"} (T={T_att})')

    # attention symbol grids
    if attn:
        dump_matrices(attn, print, exp_name)

    # --- segments (query position t predicts x_{t+1})
    lo = max(num_mask, init_len - 1)
    if k:
        seg_ranges = [('FRONT', (start_x - 1, start_x - 1 + k)),
                      ('BACK', (start_x - 1 + k, length - 1))]
    else:
        seg_ranges = [('ALL', (lo, length - 1))]

    set_A = list(range(init_len))
    # set C per segment: from attention over that segment's queries
    seg_sets = []
    for seg, (a, b) in seg_ranges:
        if attn and T_att > 0:
            q_lo = max(a, 0)
            q_hi = min(b, T_att - 1)  # matrix coords; last row has no target
            qpos = list(range(q_lo, q_hi))
            cset = sorted(set(set_A) | set(significant_dists(attn, qpos)))
        else:
            cset = list(set_A)
        seg_sets.append((seg, (a, b), set_A, cset))

    # --- forward samples, collect features/preds
    buckets = {}
    agree = {}
    for seg, rng, _, _ in seg_sets:
        agree[seg] = [0, 0]
        buckets[seg] = {'A': ([], []), 'C': ([], [])}

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
        x = torch.tensor([view], dtype=torch.long)
        with torch.no_grad():
            preds = model(x)[0][0].argmax(dim=-1).tolist()

        for seg, (a, b), _, _ in seg_sets:
            for t in range(max(lo, a), b):
                pred = preds[t]
                agree[seg][1] += 1
                if pred == seq[t + 1]:
                    agree[seg][0] += 1
                for label, dists in (('A', [s for s in seg_sets if s[0] == seg][0][2]),
                                     ('C', [s for s in seg_sets if s[0] == seg][0][3])):
                    feats, ok = [], True
                    for d in dists:
                        v = view[t - d]
                        if miss_tok is not None and v == miss_tok:
                            ok = False
                            break
                        feats.append(v)
                    if ok:
                        buckets[seg][label][0].append(feats)
                        buckets[seg][label][1].append(pred)

    # --- fit and report
    for seg, (a, b), dists_A, dists_C in seg_sets:
        m, tot = agree[seg]
        print(f'\n== segment {seg} (queries {max(lo, a)}..{b - 1}): '
              f'model-vs-truth {m}/{tot} = {m / tot:.1%}' if tot else f'\n== segment {seg}: empty')
        for label, dists in (('A', dists_A), ('C', dists_C)):
            X, y = buckets[seg][label]
            if not X:
                print(f'  set {label} {dists}: no usable positions')
                continue
            coeffs, acc, exact = fit_and_score(X, y, p)
            if coeffs is None:
                print(f'  set {label} {dists}: no linear fit (best agreement {acc:.1%})')
            else:
                tag = 'EXACT' if exact else f'best-effort, agreement {acc:.1%}'
                print(f'  set {label} {dists}: {fmt_equation(dists, coeffs, p)}  [{tag}]')
    print()


def main():
    ap = argparse.ArgumentParser(description='Fit linear formulas to model predictions over F_p.')
    ap.add_argument('name', help='experiment batch name (experiments/<name>.json)')
    ap.add_argument('--samples', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--length', type=int, default=None)
    ap.add_argument('--clean', action='store_true')
    args = ap.parse_args()

    exp_path = f'experiments/{args.name}.json'
    with open(exp_path, encoding='utf-8') as f:
        experiments = json.load(f)['experiments']

    out_path = 'rule_fit_output.log'
    missing = []
    with open(out_path, 'w', encoding='utf-8') as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            for e in experiments:
                exp_name, task, exp_cfg = e['name'], e.get('task'), e['config']
                print('=' * 78)
                print(f'# {exp_name}')
                print('=' * 78)
                pth = os.path.join(MODEL_BASE, args.name, f'{exp_name}.pth')
                log = next((cand for cand in
                            (os.path.join(base.format(name=args.name), f'{exp_name}.log')
                             for base in LOG_BASE_CANDIDATES)
                            if os.path.exists(cand)), None)
                if not os.path.exists(pth):
                    missing.append((exp_name, 'model'))
                    print(f'[skip] model not found: {pth}')
                    continue
                if log is None:
                    missing.append((exp_name, 'log'))
                    print(f'[skip] log not found for {exp_name}')
                    continue
                try:
                    run_one(args, exp_cfg, task, exp_name, pth, log)
                except Exception as exc:
                    missing.append((exp_name, f'error: {type(exc).__name__}: {exc}'))
                    print(f'[error] {type(exc).__name__}: {exc}')

            print('=' * 78)
            print(f'MISSING REPORT ({len(missing)}/{len(experiments)} skipped):')
            for exp_name, why in missing:
                print(f'  {exp_name}: {why}')
            if not missing:
                print('  none — all experiments processed')
    print(f'\n[output saved to {out_path}]')


if __name__ == '__main__':
    main()
