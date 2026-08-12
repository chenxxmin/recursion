"""Fit linear recurrence formulas to trained models' predictions, over F_p.

Given an experiment batch NAME, this script:
  1. reads experiments/<name>.json and iterates its experiments
  2. finds the training log and the model checkpoint for each experiment
  3. dumps the attention matrix from the log as a symbol grid
     (　<0.05, · 0.05~0.1, ○ 0.1~0.25, × 0.25~0.5, ※ 0.5~1)
  4. segments prediction positions by ATTENTION SIGNATURE (set of significant
     distances, union over heads): positions j >= num_mask-1 are scanned in
     order, and a new segment starts whenever the signature changes
  5. fits the model's predictions on two candidate feature sets per segment:
     A = true-rule positions (d0..d{order-1}), C = A ∪ segment signature
  6. PROBE DATA: probe sequences are fully random (every position iid uniform
     over F_p, NOT recurrence-following); every segment position contributes
     the same number of equations (balanced by construction), and feature
     values marginally cover F_p uniformly
  7. reports experiments with missing log/model at the end

Paths:
  model: /data/cxm/models/<name>/<exp>.pth
  log:   /data/cxm/models/<name>/logs/<exp>.log  or
         /data/cxm/recursion/<name>/logs/<exp>.log

Usage (repo root):
    python src/rule_fit.py <name> [--samples N] [--seed N] [--length L] [--min-seg N]

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
from report_front_back_attention import (dump_matrices, parse_log,
                                         segment_by_attention)
from verify_sample import _Tee, build_single_rule

MODEL_BASE = '/data/cxm/models'
LOG_BASE_CANDIDATES = ['/data/cxm/models/{name}/logs', '/data/cxm/recursion/{name}/logs']
SIG_THRESHOLD = 0.10  # attention values below this are ignored in signatures


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


def fit_and_score(X, y, p, rounds=300):
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


def run_one(args, exp_cfg, task, exp_name, pth_path, log_path):
    model, checkpoint = load_model(pth_path, device='cpu')
    cfg = dict(exp_cfg)
    cfg['P'] = cfg.get('P', cfg.get('p'))
    if task in ('mixed_ab', 'mixed_abc'):
        print('[skip] mixed tasks are not supported by rule_fit (single-rule only)')
        return

    init_len, next_fn, desc = build_single_rule(task, cfg)
    p = cfg['P']
    num_mask = cfg.get('NUM_MASK') or 0
    random.seed(args.seed)

    # --- log: attention matrices + signature segmentation
    start_x, series, attn = parse_log(log_path)
    if not attn:
        print('[skip] no attention section in log')
        return
    T_att = max(i for layer in attn.values() for head in layer.values() for i in head) + 1
    # probe length must cover the attention matrix coordinates, otherwise
    # segments beyond TRAIN_LEN would have no data
    length = args.length or max(cfg.get('TRAIN_LEN', 16), T_att)
    segments = segment_by_attention(attn, start_pos=num_mask - 1,
                                    threshold=SIG_THRESHOLD)
    if args.min_seg > 1:
        segments = _merge_short_segments(segments, args.min_seg)

    print(f'Model: {exp_name}')
    print(f'Rule : {desc} | probe length={length}, num_mask={num_mask}')
    print(f'Segments (signature = significant attention dists, union over heads):')
    for a, b, sig in segments:
        print(f'  positions {a:2d}..{b:2d}  dists={sorted(sig)}')

    # attention symbol grids
    dump_matrices(attn, print, exp_name)

    # --- probe: fully random sequences (off-manifold by design)
    # every segment position contributes exactly args.samples equations
    set_A = list(range(init_len))
    buckets = {}   # (seg_idx, 'A'|'C') -> (X, y)
    agree = {}     # seg_idx -> [match, total]  (model pred vs random-seq "truth" is
                   # meaningless here; instead report pred == recurrence continuation)
    for i, (a, b, sig) in enumerate(segments):
        buckets[(i, 'A')] = ([], [])
        buckets[(i, 'C')] = ([], [])
        agree[i] = [0, 0]

    for _ in range(args.samples):
        seq = [random.randrange(p) for _ in range(length)]
        x = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            preds = model(x)[0][0].argmax(dim=-1).tolist()
        for i, (a, b, sig) in enumerate(segments):
            dists_C = sorted(set(set_A) | set(sig))
            for t in range(max(a, 0), min(b, length - 2) + 1):
                pred = preds[t]
                # reference: what the TRUE rule would predict from the last
                # init_len tokens of the random probe
                agree[i][1] += 1
                if t + 1 >= init_len and pred == next_fn(seq[:t + 1]):
                    agree[i][0] += 1
                for label, dists in (('A', set_A), ('C', dists_C)):
                    feats = [seq[t - d] for d in dists if t - d >= 0]
                    if len(feats) < len(dists):
                        continue
                    buckets[(i, label)][0].append(feats)
                    buckets[(i, label)][1].append(pred)

    # --- fit and report per segment
    seg_results = []  # (a, b, sig, {label: (dists, coeffs, acc, exact)})
    for i, (a, b, sig) in enumerate(segments):
        m, tot = agree[i]
        line = f'\n== segment {i} (positions {a}..{b}), signature {sorted(sig)}'
        if tot:
            line += f' | pred == true-rule-on-random-input {m}/{tot} = {m / tot:.1%}'
        print(line)
        fits = {}
        for label in ('A', 'C'):
            dists = set_A if label == 'A' else sorted(set(set_A) | set(sig))
            X, y = buckets[(i, label)]
            if not X:
                fits[label] = (dists, None, 0.0, False)
                print(f'  set {label} {dists}: no usable positions')
                continue
            coeffs, acc, exact = fit_and_score(X, y, p)
            fits[label] = (dists, coeffs, acc, exact)
            if coeffs is None:
                print(f'  set {label} {dists}: no linear fit (best agreement {acc:.1%})')
            else:
                tag = 'EXACT' if exact else f'best-effort, agreement {acc:.1%}'
                print(f'  set {label} {dists}: {fmt_equation(dists, coeffs, p)}  [{tag}]')
        seg_results.append((a, b, sig, fits))

    # --- merge adjacent segments with identical fitted formula (set C).
    # Normalization DROPS zero coefficients, so e.g. signature [0,1,6] with
    # coeffs (c0, 0, c6) merges with [0,6] with coeffs (c0, c6).
    def norm_C(fits):
        dists, coeffs, acc, exact = fits['C']
        if coeffs is None:
            return None
        return {d: c % p for d, c in zip(dists, coeffs) if c % p != 0}

    print('== merged formula groups (adjacent segments, identical formula):')
    groups = []  # each: [start, end, formula_dict, member segment idxs, min_acc, all_exact]
    for i, (a, b, sig, fits) in enumerate(seg_results):
        f = norm_C(fits)
        acc = fits['C'][2]
        exact = fits['C'][3]
        if groups and f is not None and groups[-1][2] == f:
            g = groups[-1]
            g[1] = b
            g[3].append(i)
            g[4] = min(g[4], acc)
            g[5] = g[5] and exact
        else:
            groups.append([a, b, f, [i], acc, exact])
    for ga, gb, gf, members, min_acc, all_exact in groups:
        if gf is None:
            print(f'  positions {ga:2d}..{gb:2d}  (segments {members}): no linear fit')
            continue
        dists = sorted(gf)
        coeffs = [gf[d] for d in dists]
        tag = 'EXACT' if all_exact else f'min agreement {min_acc:.1%}'
        print(f'  positions {ga:2d}..{gb:2d}  (segments {members}): '
              f'{fmt_equation(dists, coeffs, p)}  [{tag}]')
    print()


def _merge_short_segments(segments, min_len):
    """Merge runs shorter than min_len into the previous segment (or the next
    one when there is no previous)."""
    segs = [[a, b, sig] for a, b, sig in segments]
    i = 0
    while i < len(segs):
        if segs[i][1] - segs[i][0] + 1 < min_len and len(segs) > 1:
            if i == 0:
                segs[1][0] = segs[0][0]
            else:
                segs[i - 1][1] = segs[i][1]
            del segs[i]
            i = max(i - 1, 0)
        else:
            i += 1
    return [(a, b, sig) for a, b, sig in segs]


def main():
    ap = argparse.ArgumentParser(description='Fit linear formulas to model predictions over F_p.')
    ap.add_argument('name', help='experiment batch name (experiments/<name>.json)')
    ap.add_argument('--samples', type=int, default=500,
                    help='random probe sequences per model (each position contributes one equation per sequence)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--length', type=int, default=None, help='probe length (default: TRAIN_LEN)')
    ap.add_argument('--min-seg', type=int, default=1,
                    help='merge segments shorter than this into neighbours (default 1 = literal rule)')
    ap.add_argument('--clean', action='store_true', help='ignored (probes are never corrupted)')
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
