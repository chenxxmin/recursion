"""Activation patching probe for mixed-rule (mixed_ab) checkpoints.

Question: which hidden states carry the "which rule am I on" signal?

Method: generate paired samples that share the same initial values x1, x2 --
one continued by rule A, one by rule B. Cache the hidden states of both runs
(HS_A, HS_B) at two kinds of sites, per layer:

  - attention head outputs: the per-head (B, T, head_size) slice of `att @ v`
    BEFORE c_proj (head h occupies columns h*hs:(h+1)*hs of the c_proj input);
  - MLP hidden activations: the post-GELU (B, T, mlp_ratio*d_model) input of
    the second MLP linear.

Rule tuples follow the same convention as LinearRecurrenceRule: for order-2,
rule (c1, c2) means X(k) = (c1*X(k-1) + c2*X(k-2)) % p.

Then, for each site x each token position, patch one direction's activation
into the other run (both directions: A->B input and B->A input) and check how
the model's next-token predictions change.

Classification per predicted position k (using the input sample's own
history): v_A = rule-A continuation, v_B = rule-B continuation. Every
position is recorded (nothing is skipped): match_A = (argmax == v_A),
match_B = (argmax == v_B). When v_A == v_B both can be true; a prediction
matching neither is still counted (category 'neither').

Usage:
    python src/activation_patch.py model.pth [--pairs 64] [--out report.json]
"""
import argparse
import json
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_attention import load_model  # noqa: E402


def generate_paired_samples(p, rule_a, rule_b, num_pairs, length, seed=0):
    """Paired sequences sharing initial values; A follows rule_a, B rule_b.

    Rule tuples are interpreted like LinearRecurrenceRule.coeffs: for order-2,
    (c1, c2) means X(k) = (c1*X(k-1) + c2*X(k-2)) % p.

    Returns (seqs_a, seqs_b): LongTensors (num_pairs, length).
    """
    gen = torch.Generator().manual_seed(seed)
    inits = torch.randint(0, p, (num_pairs, 2), generator=gen)

    def continue_rule(rule, x):
        c1, c2 = rule
        seq = x.clone()
        for _ in range(2, length):
            nxt = (c1 * seq[:, -1] + c2 * seq[:, -2]) % p
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        return seq

    return continue_rule(rule_a, inits), continue_rule(rule_b, inits)


def rule_targets(seqs, rule, p):
    """v[k] = rule continuation at position k (k >= 2), from the sample's own
    history. Rule tuple (c1, c2) means c1*X(k-1) + c2*X(k-2).
    Returns LongTensor (N, length) with v[:2] = -1 (unused)."""
    c1, c2 = rule
    v = torch.full_like(seqs, -1)
    v[:, 2:] = (c1 * seqs[:, 1:-1] + c2 * seqs[:, :-2]) % p
    return v


def _forward_logits(model, idx):
    out = model(idx)
    return out[0]  # both model classes return logits first


def capture_hidden(model, idx):
    """One forward pass; cache per-layer attention c_proj input and MLP
    post-GELU activation, plus the residual stream: ('emb',) is the input
    of the first block (embedding output), ('resid', li) is the output of
    block li (input of block li+1, or of ln_f for the last block).
    Returns dict and the logits."""
    cache = {}
    handles = []

    def make_hook(key):
        def hook(module, args):
            cache[key] = args[0].detach().clone()
        return hook

    h = model.transformer.h
    handles.append(h[0].register_forward_pre_hook(make_hook(('emb',))))
    for li, block in enumerate(h):
        handles.append(block.attn.c_proj.register_forward_pre_hook(make_hook(('attn', li))))
        handles.append(block.mlp[2].register_forward_pre_hook(make_hook(('mlp', li))))
        # output of block li == input of block li+1 (or of ln_f if last)
        nxt = h[li + 1] if li + 1 < len(h) else model.transformer.ln_f
        handles.append(nxt.register_forward_pre_hook(make_hook(('resid', li))))
    with torch.no_grad():
        logits = _forward_logits(model, idx)
    for hd in handles:
        hd.remove()
    return cache, logits


def patched_logits(model, idx, site, pos, source_cache):
    """Forward `idx`, overwriting the activation at `site` at token
    position(s) `pos` (int or list of ints) with values from source_cache.

    Sites:
      ('attn', layer, head) -- that head's columns of the c_proj input
      ('attn_all', layer)   -- the whole c_proj input (all heads)
      ('mlp', layer)        -- the whole MLP hidden (post-GELU) vector
      ('emb',)              -- embedding output (input of block 0)
      ('resid', layer)      -- residual stream at the output of block `layer`
    """
    kind = site[0]
    positions = [pos] if isinstance(pos, int) else list(pos)
    h = model.transformer.h

    def make_patch_hook(module_hook_target, src, sl=None):
        def hook(module, args):
            y = args[0].clone()
            if sl is None:
                y[:, positions, :] = src[:, positions, :]
            else:
                y[:, positions, sl] = src[:, positions, sl]
            return (y,)
        return module_hook_target.register_forward_pre_hook(hook)

    if kind == 'emb':
        handle = make_patch_hook(h[0], source_cache[('emb',)])
    elif kind == 'resid':
        layer = site[1]
        nxt = h[layer + 1] if layer + 1 < len(h) else model.transformer.ln_f
        handle = make_patch_hook(nxt, source_cache[('resid', layer)])
    elif kind == 'attn_all':
        layer = site[1]
        handle = make_patch_hook(h[layer].attn.c_proj,
                                 source_cache[('attn', layer)])
    elif kind == 'attn':
        layer, head = site[1], site[2]
        block = h[layer]
        hs = block.attn.head_size
        sl = slice(head * hs, (head + 1) * hs)
        handle = make_patch_hook(block.attn.c_proj,
                                 source_cache[('attn', layer)], sl)
    elif kind == 'mlp':
        layer = site[1]
        handle = make_patch_hook(h[layer].mlp[2], source_cache[('mlp', layer)])
    else:
        raise ValueError(f"unknown site {site}")

    with torch.no_grad():
        logits = _forward_logits(model, idx)
    handle.remove()
    return logits


def classify(logits, target_seqs, v_a, v_b):
    """Per prediction position k (2..T-1): argmax at input pos k-1 vs v_A/v_B.

    Returns dict of count tensors (N, T): match_a, match_b (booleans).
    """
    pred = logits[:, :-1, :].argmax(dim=-1)  # (N, T-1), pred[:, k-1] predicts x_k
    N, T = target_seqs.shape
    match_a = torch.zeros(N, T, dtype=torch.bool)
    match_b = torch.zeros(N, T, dtype=torch.bool)
    match_a[:, 2:] = pred[:, 1:] == v_a[:, 2:]
    match_b[:, 2:] = pred[:, 1:] == v_b[:, 2:]
    return {'match_a': match_a, 'match_b': match_b}


def run_probe(model, config, num_pairs=64, length=None, seed=0,
              rule_a_idx=0, rule_b_idx=1, device=None, verbose=True):
    """Full bidirectional per-position sweep. Returns the report dict."""
    if device is None:
        device = next(model.parameters()).device
    p = config['p']
    pairs = [tuple(x) for x in config['ab_pairs']]
    rule_a, rule_b = pairs[rule_a_idx], pairs[rule_b_idx]
    if length is None:
        length = config.get('train_len') or config.get('block_size', 20)
    n_layer = len(model.transformer.h)
    n_head = model.transformer.h[0].attn.n_head

    seqs_a, seqs_b = generate_paired_samples(p, rule_a, rule_b, num_pairs, length, seed)
    seqs_a, seqs_b = seqs_a.to(device), seqs_b.to(device)
    v_a_on_a = rule_targets(seqs_a, rule_a, p)  # rule-A continuation, A history
    v_b_on_a = rule_targets(seqs_a, rule_b, p)  # rule-B continuation, A history
    v_a_on_b = rule_targets(seqs_b, rule_a, p)
    v_b_on_b = rule_targets(seqs_b, rule_b, p)

    cache_a, logits_a = capture_hidden(model, seqs_a)
    cache_b, logits_b = capture_hidden(model, seqs_b)

    # Baselines (no patch): input A should follow A, input B should follow B.
    baseline = {
        'input_A': classify(logits_a, seqs_a, v_a_on_a, v_b_on_a),
        'input_B': classify(logits_b, seqs_b, v_a_on_b, v_b_on_b),
    }

    # sites: per layer, heads 0..n_head-1 plus the MLP
    sites = []
    for li in range(n_layer):
        for h in range(n_head):
            sites.append(('attn', li, h))
        sites.append(('mlp', li))

    results = []
    # direction 'AtoB': input is the B sample, patched FROM cache_a (HS_A).
    # direction 'BtoA': input is the A sample, patched FROM cache_b (HS_B).
    for direction, idx, src, va, vb in (
            ('AtoB', seqs_b, cache_a, v_a_on_b, v_b_on_b),
            ('BtoA', seqs_a, cache_b, v_a_on_a, v_b_on_a)):
        for site in sites:
            for pos in range(length):
                logits = patched_logits(model, idx, site, pos, src)
                res = classify(logits, idx, va, vb)
                ma = res['match_a'][:, 2:].float().mean().item()
                mb = res['match_b'][:, 2:].float().mean().item()
                entry = {
                    'direction': direction,
                    'site': list(site),
                    'pos': pos,
                    'match_a_rate': ma,
                    'match_b_rate': mb,
                }
                results.append(entry)
            if verbose:
                kind = f"attn L{site[1]}H{site[2]}" if site[0] == 'attn' else f"mlp L{site[1]}"
                print(f"  [{direction}] {kind} done", flush=True)

    report = {
        'config_summary': {'p': p, 'rule_a': rule_a, 'rule_b': rule_b,
                           'num_pairs': num_pairs, 'length': length,
                           'n_layer': n_layer, 'n_head': n_head},
        'baseline': {
            'input_A': {k: baseline['input_A'][k][:, 2:].float().mean().item()
                        for k in ('match_a', 'match_b')},
            'input_B': {k: baseline['input_B'][k][:, 2:].float().mean().item()
                        for k in ('match_a', 'match_b')},
        },
        'results': results,
    }
    return report


def print_summary(report):
    """Per-site table: patch effect averaged over positions, both directions."""
    bl = report['baseline']
    print(f"\nBaseline: input A -> match A {bl['input_A']['match_a']:.3f} / "
          f"match B {bl['input_A']['match_b']:.3f}; "
          f"input B -> match B {bl['input_B']['match_b']:.3f} / "
          f"match A {bl['input_B']['match_a']:.3f}")
    cs = report['config_summary']
    print(f"p={cs['p']} rule_a={cs['rule_a']} rule_b={cs['rule_b']} "
          f"pairs={cs['num_pairs']} len={cs['length']}\n")

    # aggregate per site
    agg = {}
    for r in report['results']:
        key = (r['direction'], tuple(r['site']))
        agg.setdefault(key, []).append((r['match_a_rate'], r['match_b_rate']))
    header = f"{'direction':8} {'site':12} {'matchA':>7} {'matchB':>7}"
    print(header)
    print('-' * len(header))
    for (direction, site), vals in sorted(agg.items()):
        ma = sum(v[0] for v in vals) / len(vals)
        mb = sum(v[1] for v in vals) / len(vals)
        name = f"L{site[1]}H{site[2]}" if site[0] == 'attn' else f"L{site[1]} MLP"
        print(f"{direction:8} {name:12} {ma:7.3f} {mb:7.3f}")


def main():
    ap = argparse.ArgumentParser(description="Activation patching probe (mixed_ab)")
    ap.add_argument('model', help="path to .pth checkpoint")
    ap.add_argument('--pairs', type=int, default=64, help="number of paired samples")
    ap.add_argument('--len', type=int, default=None, help="sequence length (default: train_len)")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rule-a', type=int, default=0, help="index into ab_pairs for rule A")
    ap.add_argument('--rule-b', type=int, default=1, help="index into ab_pairs for rule B")
    ap.add_argument('--out', default=None, help="write JSON report here")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model(args.model, device=device)
    config = checkpoint['config']

    report = run_probe(model, config, num_pairs=args.pairs, length=args.len,
                       seed=args.seed, rule_a_idx=args.rule_a,
                       rule_b_idx=args.rule_b, device=device)
    print_summary(report)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=1)
        print(f"\nReport written to {args.out}")


if __name__ == '__main__':
    main()
