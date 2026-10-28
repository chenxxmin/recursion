"""Multi-position simultaneous patching ("flip hunt") for mixed_ab ckpts.

Motivation: attention patterns are rule-independent (each head reads a
fixed lag; the model collectively covers the last 3 tokens + the skip
connection). Single-position patches therefore cannot flip the rule:
the rule signal is spread across the last few positions' residual
streams. This script patches SEVERAL positions at once, over a ladder
of site types, to find the minimal patch that flips the prediction
from one rule's continuation to the other's.

Ladder (per target prediction position k, query q = k-1, window
W = [q-2, q-1, q] clipped to >= 0):

  attn_head@q      -- each single head at the query position (null ctrl)
  attn_all@q       -- whole c_proj input (all heads) at q
  attn_all@W       -- all heads at the whole attended window
  mlp@q            -- MLP hidden at q
  emb@q            -- embedding output at q (current token only)
  emb@W            -- embedding output at the whole window
  resid<i>@q       -- block-i output at q (i = each layer)
  resid<i>@W       -- block-i output at the whole window
  resid_last@all   -- last block output at ALL positions (ceiling)

Directions: AtoB (input B, patched FROM A; flip shows as match_a rising)
and BtoA (symmetric). Reports per-position match_a / match_b / neither.

Usage:
    python src/multi_pos_patch.py model.pth [--pairs 128] [--len 12] \
        [--max-k 9] [--seed 0] [--out report.json]
"""
import argparse
import json
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_attention import load_model  # noqa: E402
from activation_patch import (  # noqa: E402
    generate_paired_samples, rule_targets, capture_hidden, patched_logits)


def build_conditions(n_layer, n_head, q):
    """Ladder of (label, site, positions) for query position q."""
    W = [p for p in (q - 2, q - 1, q) if p >= 0]
    conds = []
    for li in range(n_layer):
        for h in range(n_head):
            conds.append((f'attn_l{li}h{h}@q', ('attn', li, h), [q]))
        conds.append((f'attn_all_l{li}@q', ('attn_all', li), [q]))
    conds.append(('attn_all_l0@W', ('attn_all', 0), W))
    for li in range(n_layer):
        conds.append((f'mlp_l{li}@q', ('mlp', li), [q]))
    conds.append(('emb@q', ('emb',), [q]))
    conds.append(('emb@W', ('emb',), W))
    for li in range(n_layer):
        conds.append((f'resid{li}@q', ('resid', li), [q]))
        conds.append((f'resid{li}@W', ('resid', li), W))
    return conds


def run(model, config, num_pairs=128, length=12, max_k=9, seed=0,
        device=None, verbose=True):
    if device is None:
        device = next(model.parameters()).device
    p = config['p']
    rule_a, rule_b = [tuple(x) for x in config['ab_pairs']][:2]
    n_layer = len(model.transformer.h)
    n_head = model.transformer.h[0].attn.n_head

    seqs_a, seqs_b = generate_paired_samples(p, rule_a, rule_b, num_pairs,
                                             length, seed)
    seqs_a, seqs_b = seqs_a.to(device), seqs_b.to(device)
    cache_a, logits_a = capture_hidden(model, seqs_a)
    cache_b, logits_b = capture_hidden(model, seqs_b)

    def targets(seqs):
        return rule_targets(seqs, rule_a, p), rule_targets(seqs, rule_b, p)

    va_a, vb_a = targets(seqs_a)  # rule-A / rule-B continuation on A history
    va_b, vb_b = targets(seqs_b)  # ... on B history

    def hybrid_targets(idx_seq, src_seq):
        """For single-position patch at query q=k-1: the model's effective
        hybrid operands are the SOURCE token at q (patched position) and
        the INPUT token at q-1 (unpatched). v_hX[:, k] = rule X applied to
        that hybrid pair. Valid only for @q (single-position) conditions."""
        (c1a, c2a), (c1b, c2b) = rule_a, rule_b
        vha = torch.full_like(idx_seq, -1)
        vhb = torch.full_like(idx_seq, -1)
        vha[:, 2:] = (c1a * src_seq[:, 1:-1] + c2a * idx_seq[:, :-2]) % p
        vhb[:, 2:] = (c1b * src_seq[:, 1:-1] + c2b * idx_seq[:, :-2]) % p
        return vha, vhb

    def rates(logits, va_own, vb_own, va_src, vha, vhb, k):
        """Five-target classification at prediction position k.

        va_own/vb_own: rule-A / rule-B continuation of the INPUT's own
            history (rule flip = matching va_own after AtoB patch);
        va_src: the SOURCE run's actual continuation (state transfer);
        vha/vhb: rule-A / rule-B applied to the HYBRID operand pair
            (source token at q, input token at q-1) -- detects consistent
            computation on a mixed history rather than true breakdown.
        """
        pred = logits[:, :-1, :].argmax(dim=-1)[:, k - 1]  # predicts x_k
        ma = (pred == va_own[:, k]).float().mean().item()
        mb = (pred == vb_own[:, k]).float().mean().item()
        ms = (pred == va_src[:, k]).float().mean().item()
        mha = (pred == vha[:, k]).float().mean().item()
        mhb = (pred == vhb[:, k]).float().mean().item()
        both = (va_own[:, k] == vb_own[:, k]).float().mean().item()
        nei = 1 - ma - mb + both
        return {'match_a': ma, 'match_b': mb, 'match_src': ms,
                'match_hyb_a': mha, 'match_hyb_b': mhb, 'neither': nei}

    # k=2 (predicting x3) is skipped: rule is not identifiable there yet.
    ks = list(range(3, min(max_k, length - 1) + 1))
    report = {'config_summary': {
        'p': p, 'rule_a': rule_a, 'rule_b': rule_b, 'num_pairs': num_pairs,
        'length': length, 'n_layer': n_layer, 'n_head': n_head,
        'seed': seed, 'ks': ks}}

    for direction, idx, src, src_seq, va_own, vb_own, va_src, logits_base in (
            ('AtoB', seqs_b, cache_a, seqs_a, va_b, vb_b, va_a, logits_b),
            ('BtoA', seqs_a, cache_b, seqs_b, va_a, vb_a, vb_b, logits_a)):
        # va_own/vb_own: rule-A/rule-B continuation of the INPUT's history;
        # va_src: the SOURCE run's continuation of ITS OWN history
        # (AtoB: rule A on A history = va_a; BtoA: rule B on B history = vb_b)
        vha, vhb = hybrid_targets(idx, src_seq)
        base = {k: rates(logits_base, va_own, vb_own, va_src, vha, vhb, k)
                for k in ks}
        entries = []
        for k in ks:
            q = k - 1
            for label, site, positions in build_conditions(n_layer, n_head, q):
                logits = patched_logits(model, idx, site, positions, src)
                r = rates(logits, va_own, vb_own, va_src, vha, vhb, k)
                entries.append({'k': k, 'cond': label,
                                'positions': positions, **r})
            if verbose:
                print(f'  [{direction}] k={k} done', flush=True)
        report[direction] = {'baseline': base, 'results': entries}
    return report


def print_summary(report):
    ks = report['config_summary']['ks']
    # flip targets: AtoB wants match_a (rule A on B's history) rising;
    #               BtoA wants match_b (rule B on A's history) rising.
    flip_key = {'AtoB': 'match_a', 'BtoA': 'match_b'}
    for direction in ('AtoB', 'BtoA'):
        fk = flip_key[direction]
        base = report[direction]['baseline']
        print(f'\n=== {direction}: rule-flip = {fk} ABOVE baseline '
              f'(baseline {fk}: ' +
              ' '.join(f'{base[k][fk]:.2f}' for k in ks) + ') ===')
        agg = {}
        for e in report[direction]['results']:
            inc = e[fk] - base[e['k']][fk]
            agg.setdefault(e['cond'], []).append((inc, e['match_src']))
        rows = []
        for cond, vals in agg.items():
            incs = [v[0] for v in vals]
            srcs = [v[1] for v in vals]
            rows.append((sum(incs) / len(incs), max(incs),
                         sum(srcs) / len(srcs), cond))
        rows.sort(reverse=True)
        print(f"{'condition':16} {'mean_dflip':>10} {'max_dflip':>9} {'mean_src':>9}")
        for mean_i, max_i, mean_s, cond in rows:
            print(f'{cond:16} {mean_i:10.3f} {max_i:9.3f} {mean_s:9.3f}')


def main():
    ap = argparse.ArgumentParser(description="Multi-position patch flip hunt")
    ap.add_argument('model', help="path to .pth checkpoint")
    ap.add_argument('--pairs', type=int, default=128)
    ap.add_argument('--len', type=int, default=12)
    ap.add_argument('--max-k', type=int, default=9,
                    help="evaluate prediction positions 2..max_k")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model(args.model, device=device)
    report = run(model, checkpoint['config'], num_pairs=args.pairs,
                 length=args.len, max_k=args.max_k, seed=args.seed,
                 device=device)
    print_summary(report)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=1)
        print(f'\nReport written to {args.out}')


if __name__ == '__main__':
    main()
