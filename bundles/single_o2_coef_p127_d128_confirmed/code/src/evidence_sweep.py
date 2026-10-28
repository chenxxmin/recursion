"""Evidence-value sweep: map the rule-decision function.

Idea (synthetic V-content patch): a head's c_proj-input slice at query q is
approximately V(x) of the token it attends. Instead of patching with values
captured from a real run (which only gives one arbitrary evidence value per
sample pair), we SYNTHESIZE the V-content of any token value x~ directly from
the model's weights:

    v(x~) = c_attn(ln_1(wte(x~)))[2d + h*hs : 2d + (h+1)*hs]

and sweep x~ over all p values, recording which rule the model follows.

Fidelity checks (run first on any new model): x~ = truthful evidence value
should reproduce baseline behavior; x~ = actual source-run value should
reproduce the natural patch flip rate.

Output report: per-x~ match rate to each rule's continuation of the input
history, plus basin capture rates at the ideal evidence values
x~* = (x_(q+1) - c1*x_q) / c2  (mod p)  for each candidate rule (c1,c2).

Usage:
    python src/evidence_sweep.py model.pth [--q 4] [--pairs 512] \
        [--input-rule 0] [--layer 0] [--head -1(auto=lag2-dominant)] \
        [--out report.json]
"""
import argparse
import json
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_attention import load_model, get_attention_weights  # noqa: E402
from activation_patch import (  # noqa: E402
    generate_paired_samples, rule_targets, capture_hidden, patched_logits)


def find_lag2_head(model, seqs):
    """Head (at layer 0) with the largest attention mass on lag 2."""
    w = get_attention_weights(model, seqs)[0]  # (H, T, T)
    H, T, _ = w.shape
    q = torch.arange(3, T - 1)
    lag2 = w[:, q, q - 2].mean(dim=1)  # (H,)
    return int(lag2.argmax())


def main():
    ap = argparse.ArgumentParser(description="Evidence-value sweep probe")
    ap.add_argument('model')
    ap.add_argument('--q', type=int, default=4, help="query position (predict token q+1)")
    ap.add_argument('--pairs', type=int, default=512)
    ap.add_argument('--input-rule', type=int, default=0,
                    help="index into ab_pairs: which rule the input follows")
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--head', type=int, default=-1, help="-1 = auto lag2-dominant")
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model(args.model, device=device)
    config = checkpoint['config']
    p = config['p']
    rules = [tuple(x) for x in config['ab_pairs']]
    in_rule = rules[args.input_rule]
    block = model.transformer.h[args.layer]
    hs = block.attn.head_size
    d = block.attn.n_embd

    seqs, _ = generate_paired_samples(p, in_rule, rules[-1], args.pairs,
                                      args.q + 8, args.seed)
    seqs = seqs.to(device)
    head = args.head if args.head >= 0 else find_lag2_head(model, seqs)
    print(f'rules={rules} input_rule={in_rule} head=L{args.layer}h{head} '
          f'q={args.q}')

    # V-content of every token value
    vsl = slice(2 * d + head * hs, 2 * d + (head + 1) * hs)
    toks = torch.arange(p, device=device)
    with torch.no_grad():
        v_all = block.attn.c_attn(block.ln_1(model.transformer.wte(toks)))[:, vsl]

    cache_in, logits_base = capture_hidden(model, seqs)
    attv_in = cache_in[('attn', args.layer)]
    k = args.q + 1
    v_tgts = [rule_targets(seqs, r, p).to(device)[:, k] for r in rules]
    N = seqs.shape[0]

    match = torch.zeros(N, p, len(rules))
    for xt in range(p):
        fake = attv_in.clone()
        fake[:, args.q, head * hs:(head + 1) * hs] = v_all[xt]
        pred = patched_logits(model, seqs, ('attn', args.layer, head), args.q,
                              {('attn', args.layer): fake})[:, args.q].argmax(-1)
        for i in range(len(rules)):
            match[:, xt, i] = (pred == v_tgts[i]).cpu()

    # baseline sanity: unpatched should follow input rule ~100%
    pred_base = logits_base[:, args.q].argmax(-1)
    base_rate = (pred_base == v_tgts[args.input_rule]).float().mean().item()

    # basin capture at ideal evidence x~* = (x_(q+1) - c1*x_q) / c2
    basins = {}
    for i, (c1, c2) in enumerate(rules):
        xstar = (((seqs[:, args.q] - c1 * seqs[:, args.q - 1]) % p)
                 * pow(c2, -1, p)) % p
        ok = (xstar > 0).cpu()
        cap = match.cpu()[torch.arange(N), xstar.cpu(), i][ok].float().mean()
        basins[str((c1, c2))] = float(cap)

    # fidelity: x~ = truthful evidence (input rule's actual token at q-2 pos)
    truthful = seqs[:, args.q - 2].cpu()
    fid = match.cpu()[torch.arange(N), truthful, args.input_rule].float().mean()

    report = {
        'config_summary': {'p': p, 'rules': rules, 'input_rule': in_rule,
                           'layer': args.layer, 'head': head, 'q': args.q,
                           'pairs': args.pairs, 'seed': args.seed},
        'baseline_own_rule_rate': base_rate,
        'basin_capture': basins,
        'fidelity_truthful_evidence_match': float(fid),
        'per_xtilde_match': {
            str(rules[i]): match[:, :, i].mean(0).tolist()
            for i in range(len(rules))},
    }
    print(f'baseline own-rule={base_rate:.3f}  '
          f'fidelity(truthful evidence)={fid:.3f}')
    for r, v in basins.items():
        print(f'basin capture rule {r}: {v:.3f}')
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=1)
        print(f'Report written to {args.out}')


if __name__ == '__main__':
    main()
