"""Head ablation probe: is each attention head NEEDED (not just read)?

Patching (replace with a contradictory value) only proves a channel is read;
ablation (zero the head's c_proj input slice at ALL positions) tests whether
the model can do without it. Reports per-head, per-rule teacher-forcing
accuracy with the head's output zeroed.

Interpretation guide (from the mixed_ab line of experiments):
  - ablate operand head (lag0/lag1)  -> collapse (load-bearing)
  - ablate evidence head (lag2)      -> partial collapse, asymmetric across
    rules (model defaults to one rule without evidence)
  - ablate mixed/inert head          -> no change (truly redundant)

Usage:
    python src/head_ablation.py model.pth [--pairs 256] [--len 16] \
        [--out report.json]
"""
import argparse
import json
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_attention import load_model  # noqa: E402
from activation_patch import generate_paired_samples, rule_targets  # noqa: E402


@torch.no_grad()
def accuracy(model, seqs, rule, p, layer, ablate_head=None):
    """Teacher-forcing match to the sample's own rule, positions k>=3."""
    block = model.transformer.h[layer]
    handle = None
    if ablate_head is not None:
        hs = block.attn.head_size
        sl = slice(ablate_head * hs, (ablate_head + 1) * hs)

        def hook(module, args):
            y = args[0].clone()
            y[:, :, sl] = 0.0
            return (y,)
        handle = block.attn.c_proj.register_forward_pre_hook(hook)
    logits = model(seqs)[0]
    if handle:
        handle.remove()
    pred = logits[:, :-1].argmax(-1)
    v = rule_targets(seqs, rule, p).to(seqs.device)
    return (pred[:, 2:] == v[:, 3:]).float().mean().item()


def main():
    ap = argparse.ArgumentParser(description="Head ablation probe")
    ap.add_argument('model')
    ap.add_argument('--pairs', type=int, default=256)
    ap.add_argument('--len', type=int, default=16)
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model(args.model, device=device)
    config = checkpoint['config']
    p = config['p']
    rules = [tuple(x) for x in config['ab_pairs']]
    n_head = model.transformer.h[args.layer].attn.n_head

    g = torch.Generator().manual_seed(args.seed)
    inits = torch.randint(1, p, (args.pairs, 2), generator=g)
    seqs = {}
    for r in rules:
        seq = inits.clone()
        for _ in range(2, args.len):
            seq = torch.cat([seq, ((r[0] * seq[:, -1] + r[1] * seq[:, -2]) % p)
                             .unsqueeze(1)], 1)
        seqs[r] = seq.to(device)

    report = {'config_summary': {'p': p, 'rules': rules, 'layer': args.layer,
                                 'n_head': n_head, 'pairs': args.pairs},
              'baseline': {}, 'ablated': {}}
    for r in rules:
        report['baseline'][str(r)] = accuracy(model, seqs[r], r, p, args.layer)
    print(f'baseline: {report["baseline"]}')
    for h in range(n_head):
        for r in rules:
            report['ablated'].setdefault(f'head{h}', {})[str(r)] = \
                accuracy(model, seqs[r], r, p, args.layer, ablate_head=h)
        print(f'ablate head{h}: {report["ablated"][f"head{h}"]}')
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=1)
        print(f'Report written to {args.out}')


if __name__ == '__main__':
    main()
