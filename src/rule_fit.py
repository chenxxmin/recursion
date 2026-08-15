"""Fit linear recurrence formulas to trained models' predictions, over F_p.

Given an experiment batch NAME, this script:
  1. reads experiments/<name>.json and iterates its experiments
  2. finds the training log and the model checkpoint for each experiment
  3. dumps the attention matrix from the log as a symbol grid
     (　<0.05, · 0.05~0.1, ○ 0.1~0.25, × 0.25~0.5, ※ 0.5~1)
  4. MISSING-MODE (experiments with MISSING_PROB > 0): positions are grouped
     by the corruption pattern of the last two context tokens —
     (x,x) / (M,x) / (M,M) / (x,M) — and fitted per category; sequences
     follow the recurrence and are corrupted with the experiment's own rule
  5. otherwise positions are segmented by ATTENTION SIGNATURE (set of
     significant distances >= 0.1, union over heads) and fitted per segment
  6. PROBE DATA for attention mode: fully random sequences (every position
     iid uniform over F_p); every segment position contributes the same
     number of equations (balanced by construction)
  7. reports experiments with missing log/model at the end

Paths:
  model: /data/cxm/models/<name>/<exp>.pth
  log:   /data/cxm/models/<name>/logs/<exp>.log  or
         /data/cxm/recursion/<name>/logs/<exp>.log

Usage (repo root):
    python src/rule_fit.py <name>                 # batch mode: iterate experiments/<name>.json
    python src/rule_fit.py <model.pth> <cfg.json> # single-model mode: match the
                                                  # experiment by pth filename stem
    [--samples N] [--seed N] [--length L] [--min-seg N] [--depth N] [--min-n N]

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

from analyze_attention import extract_qk_raw_scores, load_model
from datasets import corrupt_window
from report_front_back_attention import (dump_matrices, parse_log,
                                         segment_by_attention)
from rules import task_from_save_config
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


def run_missing_categories(args, model, cfg, next_fn, init_len, p, exp_name, desc):
    """Missing-experiment mode, recursive edition: group prediction positions
    by the corruption pattern of the last D tokens (D = miss_len + 2 by
    default; --depth overrides). This unrolls the recursion the user cares
    about — e.g. within (M,M), positions where x_{t-3} is itself missing show
    up as their own pattern (M,x,M,M). Per pattern we report model-vs-truth
    agreement, the mean attention per distance per head (computed live on the
    corrupted samples), and coefficient fits on visible-position features.
    """
    length = args.length or cfg.get('TRAIN_LEN', 16)
    num_mask = cfg.get('NUM_MASK') or 0
    missing_prob = cfg['MISSING_PROB']
    miss_len = cfg.get('MISS_LEN', 1)
    D = args.depth or (miss_len + 2)
    min_n = args.min_n
    CATS = ['(x,x)', '(M,x)', '(M,M)', '(x,M)']  # by (view[t-1], view[t])
    set_A = list(range(init_len))
    dists_C_full = list(range(init_len + miss_len + 1))
    lo = max(num_mask, 1)
    start = max(lo, D - 1)

    cat4 = {c: [0, 0] for c in CATS}
    P = {}  # pattern tuple -> record

    for _ in range(args.samples):
        seq = [random.randrange(p) for _ in range(init_len)]
        while len(seq) < length:
            seq.append(next_fn(seq))
        window = torch.tensor(seq, dtype=torch.long)
        # single-rule only here, so the missing token is plain p
        corrupt_window(window, True, p=p, init_len=init_len, missing_prob=missing_prob,
                       miss_len=miss_len, miss_second=cfg.get('MISS_SECOND', False))
        view = window.tolist()
        x = torch.tensor([view], dtype=torch.long)
        with torch.no_grad():
            preds = model(x)[0][0].argmax(dim=-1).tolist()
            attn_layers = [lo_['attn_weights'] for lo_ in extract_qk_raw_scores(model, x)]

        for t in range(start, length - 1):
            patt = tuple(view[t - D + 1 + k] == p for k in range(D))
            rec = P.get(patt)
            if rec is None:
                rec = P[patt] = {'n': 0, 'agree': 0, 'fits': {}, 'attn': {}}
            rec['n'] += 1
            s1, s0 = patt[-2], patt[-1]
            cat = '(M,M)' if s1 and s0 else '(M,x)' if s1 else '(x,M)' if s0 else '(x,x)'
            cat4[cat][1] += 1
            pred = preds[t]
            if pred == seq[t + 1]:
                cat4[cat][0] += 1
                rec['agree'] += 1
            for label, full_dists in (('A', set_A), ('C', dists_C_full)):
                dists = [d for d in full_dists if t - d >= 0 and view[t - d] != p]
                if not dists:
                    continue
                fb = rec['fits'].setdefault(label, {}).setdefault(tuple(dists), ([], []))
                fb[0].append([view[t - d] for d in dists])
                fb[1].append(pred)
            for li, att in enumerate(attn_layers):
                row_all = att[:, t, :]  # (H, T)
                for h in range(att.shape[0]):
                    acc = rec['attn'].setdefault((li, h), {})
                    row = row_all[h]
                    for d in range(t + 1):
                        v = row[t - d].item()
                        if d in acc:
                            acc[d][0] += v
                            acc[d][1] += 1
                        else:
                            acc[d] = [v, 1]

    print(f'Model: {exp_name}')
    print(f'Rule : {desc} | length={length}, missing prob={missing_prob}, '
          f'miss_len={miss_len}, miss_second={cfg.get("MISS_SECOND", False)}, depth={D}')
    print('Top-level grouping (last two context tokens):')
    for cat in CATS:
        m, tot = cat4[cat]
        print(f'  {cat}: model-vs-truth {m}/{tot}'
              + (f' = {m / tot:.1%}' if tot else ' (no positions)'))

    print(f'\nDeep patterns (last {D} tokens, oldest first; n >= {min_n}):')
    for patt, rec in sorted(P.items(), key=lambda kv: -kv[1]['n']):
        if rec['n'] < min_n:
            continue
        label = '(' + ','.join('M' if m else 'x' for m in patt) + ')'
        print(f'\n== pattern {label}  n={rec["n"]}, '
              f'model-vs-truth {rec["agree"] / rec["n"]:.1%}')
        attn_parts = []
        for (li, h), acc in sorted(rec['attn'].items()):
            sig = [(d, s / c) for d, (s, c) in sorted(acc.items()) if s / c >= SIG_THRESHOLD]
            if sig:
                attn_parts.append(f'L{li}H{h} ' + ' '.join(f'd{d}:{v:.2f}' for d, v in sig))
        if attn_parts:
            print('  attn: ' + ' | '.join(attn_parts))
        for label in ('A', 'C'):
            groups = rec['fits'].get(label, {})
            if not groups:
                print(f'  set {label}: no usable positions')
                continue
            for dists_t, (X, y) in sorted(groups.items(), key=lambda kv: -len(kv[1][0])):
                coeffs, acc, exact = fit_and_score(X, y, p)
                if coeffs is None:
                    print(f'  set {label} {list(dists_t)}: no linear fit '
                          f'(best agreement {acc:.1%}, n={len(X)})')
                else:
                    tag = 'EXACT' if exact else f'best-effort, agreement {acc:.1%}'
                    print(f'  set {label} {list(dists_t)}: '
                          f'{fmt_equation(list(dists_t), coeffs, p)}  [{tag}, n={len(X)}]')
    print()


def run_one(args, exp_cfg, task, exp_name, pth_path, log_path):
    model, checkpoint = load_model(pth_path, device='cpu')
    cfg = dict(exp_cfg)
    cfg['P'] = cfg.get('P', cfg.get('p'))
    if task is None:
        # checkpoint fallback: task_from_save_config normalizes the checkpoint
        # vocabulary ('multiplicative' -> 'multiplication'); None means a
        # mixed/dynamic checkpoint
        task = task_from_save_config(checkpoint['config'])
        if task is None:
            task = checkpoint['config'].get('recurrence') or 'mixed_ab'
    if task in ('mixed_ab', 'mixed_abc'):
        print('[skip] mixed tasks are not supported by rule_fit (single-rule only)')
        return

    init_len, next_fn, desc = build_single_rule(task, cfg)
    p = cfg['P']
    num_mask = cfg.get('NUM_MASK') or 0
    random.seed(args.seed)

    # missing experiments: group by corruption pattern of the last two context
    # tokens instead of attention signatures
    if cfg.get('MISSING_PROB', 0.0) > 0 and not args.clean:
        run_missing_categories(args, model, cfg, next_fn, init_len, p, exp_name, desc)
        return

    # --- log: attention matrices + signature segmentation
    if log_path is None:
        print('[skip] attention mode needs the training log (none found for this model)')
        return
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
    ap.add_argument('name', help='experiment batch name (experiments/<name>.json), '
                                 'or a .pth path for single-model mode')
    ap.add_argument('json', nargs='?', default=None,
                    help='single-model mode: experiments JSON to match the pth filename stem')
    ap.add_argument('--samples', type=int, default=500,
                    help='random probe sequences per model (each position contributes one equation per sequence)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--length', type=int, default=None, help='probe length (default: TRAIN_LEN)')
    ap.add_argument('--min-seg', type=int, default=1,
                    help='merge segments shorter than this into neighbours (default 1 = literal rule)')
    ap.add_argument('--depth', type=int, default=None,
                    help='missing mode: pattern depth D (default miss_len + 2)')
    ap.add_argument('--min-n', type=int, default=20,
                    help='missing mode: only print patterns with at least this many positions')
    ap.add_argument('--clean', action='store_true',
                    help='attention mode: do not corrupt; also forces attention mode for missing experiments')
    args = ap.parse_args()

    out_path = 'rule_fit_output.log'

    # ---- single-model mode: rule_fit model.pth config.json
    if args.name.endswith('.pth'):
        pth = args.name
        if not args.json:
            ap.error('single-model mode needs the experiments JSON as second argument')
        with open(args.json, encoding='utf-8') as f:
            experiments = json.load(f)['experiments']
        stem = os.path.splitext(os.path.basename(pth))[0]
        match = next((e for e in experiments if e['name'] == stem), None)
        if match is None:
            print(f'[error] {stem} not found in {args.json}')
            return
        batch = os.path.splitext(os.path.basename(args.json))[0]
        log = next((cand for cand in
                    (os.path.join(base.format(name=batch), f'{stem}.log')
                     for base in LOG_BASE_CANDIDATES)
                    if os.path.exists(cand)), None)
        with open(out_path, 'w', encoding='utf-8') as f:
            with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
                try:
                    run_one(args, match['config'], match.get('task'), stem, pth, log)
                except Exception as exc:
                    print(f'[error] {type(exc).__name__}: {exc}')
        print(f'\n[output saved to {out_path}]')
        return

    # ---- batch mode
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
