"""Rule generalization probe for mixed_ab checkpoints (no ab_tag).

Question: for models trained on AB_PAIRS = [[1,1],[1,2]] (a=1, b in {1,2}),
has the model learned the *algorithm* "a=1, infer b from x3" (which would
generalize to any b), or has it only learned to *match* x3 against the two
memorized patterns (classification into two slots)?

Method: build sequences that follow rule (1, b) for NEW b values (not in
training), feed them to the model, and classify each prediction against
three candidate continuations:

  - v_b1:   continuation under the trained rule (1,1)
  - v_b2:   continuation under the trained rule (1,2)
  - v_true: continuation under the sample's true rule (1,b)

All three matches are recorded simultaneously (they are not mutually
exclusive when candidate values collide); 'neither' = matches none.

Three conditions per b:

  1. teacher forcing (prefix 3): feed the true sequence, classify the
     argmax at every prediction position k>=2.
  2. free generation (prefix 3): feed x1..x3, greedily generate, and at
     each step check which candidate rules stay 'alive' (consistent with
     every token the model has produced so far, computed against the
     model's OWN generated history). Detects rule lock-in.
  3. long prefix (prefix 5): same as (1) but with x1..x5 given.
     Separates "cannot identify b" from "cannot execute rule b".

Trained b values (1, 2) are always included as sanity controls -- these
must come out ~100% match_true, otherwise the setup is broken.

x1 and x2 are sampled from 1..p-1 (0 excluded: x1=0 makes b unidentifiable
from x3, x2=0 makes x4 identical for all b).

Usage:
    python src/rule_generalization.py model.pth \
        [--b-values 0,3,4,5,8,16,32,64] [--samples 256] [--gen-len 16] \
        [--seed 0] [--out report.json]
"""
import argparse
import json
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_attention import load_model  # noqa: E402
from activation_patch import rule_targets  # noqa: E402


def gen_inits(p, n, seed):
    """(n, 2) initial values, both in 1..p-1 (0 excluded, see docstring)."""
    g = torch.Generator().manual_seed(seed)
    x1 = torch.randint(1, p, (n,), generator=g)
    x2 = torch.randint(1, p, (n,), generator=g)
    return torch.stack([x1, x2], dim=1)


def continue_rule(rule, inits, length, p):
    """Continue (n, 2) inits under rule (c1, c2) up to `length` tokens."""
    c1, c2 = rule
    seq = inits.clone()
    for _ in range(2, length):
        nxt = (c1 * seq[:, -1] + c2 * seq[:, -2]) % p
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
    return seq


@torch.no_grad()
def forward_argmax(model, idx):
    out = model(idx)
    logits = out[0]
    return logits[:, :-1, :].argmax(dim=-1)  # pred[:, k-1] predicts x_k


def teacher_forcing_eval(model, seqs, cand_rules, p):
    """Classify per-position argmax against each candidate continuation.

    Returns {name: per-position match rate list (len T, positions 0,1 = None)}
    plus 'neither'."""
    pred = forward_argmax(model, seqs)
    N, T = seqs.shape
    out = {}
    matches = []
    for name, rule in cand_rules.items():
        v = rule_targets(seqs, rule, p).to(seqs.device)
        m = torch.zeros(N, T, dtype=torch.bool, device=seqs.device)
        m[:, 2:] = pred[:, 1:] == v[:, 2:]
        matches.append(m)
        out[name] = [None, None] + m[:, 2:].float().mean(dim=0).tolist()
    stacked = torch.stack(matches).any(dim=0)  # (N, T) matched any candidate
    neither = torch.zeros(N, T, dtype=torch.bool, device=seqs.device)
    neither[:, 2:] = ~stacked[:, 2:]
    out['neither'] = [None, None] + neither[:, 2:].float().mean(dim=0).tolist()
    return out


@torch.no_grad()
def free_gen_eval(model, prefix, cand_rules, p, gen_len):
    """Greedily continue `prefix` (N, P0) for gen_len steps.

    A candidate rule is 'alive' for a sample while every token the model
    generated (from position 3 on, i.e. once a 2-token history exists)
    equals that rule's continuation computed from the model's OWN tokens.

    Returns {'per_step_alive': {name: [rates per generated step]},
             'final_alive_set': {frozenset-as-string: count}}."""
    seq = prefix.clone()
    N, P0 = prefix.shape
    device = prefix.device
    alive = {name: torch.ones(N, dtype=torch.bool, device=device)
             for name in cand_rules}
    per_step = {name: [] for name in cand_rules}

    for t in range(gen_len):
        pos = seq.shape[1]  # predicting token at index pos
        nxt = model(seq)[0][:, -1, :].argmax(dim=-1)  # (N,)
        if pos >= 2:  # two-token history available -> check consistency
            for name, rule in cand_rules.items():
                c1, c2 = rule
                cand = (c1 * seq[:, -1] + c2 * seq[:, -2]) % p
                alive[name] &= nxt == cand
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        for name in cand_rules:
            per_step[name].append(alive[name].float().mean().item())

    final = {}
    for i in range(N):
        key = ','.join(sorted(n for n in cand_rules if alive[n][i])) or 'none'
        final[key] = final.get(key, 0) + 1
    return {'per_step_alive': per_step, 'final_alive_set': final,
            'gen_len': gen_len, 'prefix_len': P0}


def run(model, config, b_values, num_samples, gen_len, seed, device,
        prefix_lens=(3, 5), verbose=True):
    p = config['p']
    trained = [tuple(x) for x in config['ab_pairs']]
    assert all(r[0] == 1 for r in trained), \
        f"probe assumes a=1 trained rules, got {trained}"

    # candidate rule set: trained b's (controls) + requested new b's
    all_b = sorted(set([r[1] for r in trained]) | set(b_values))
    max_len = max(prefix_lens) + gen_len

    report = {'config_summary': {
        'p': p, 'trained_rules': trained, 'b_values': all_b,
        'num_samples': num_samples, 'gen_len': gen_len,
        'prefix_lens': list(prefix_lens), 'seed': seed}}

    for b in all_b:
        rule = (1, b)
        cand = {'b1': trained[0], 'b2': trained[1], 'true': rule}
        # dedupe display names if true coincides with a trained rule
        entry = {'rule': rule, 'is_trained': rule in trained}

        inits = gen_inits(p, num_samples, seed + 1000 * b)
        # sanity: implied b from x3 must equal b (x1 != 0 guaranteed)
        seq_chk = continue_rule(rule, inits, 3, p)
        inv = torch.tensor([pow(int(v), -1, p) for v in seq_chk[:, 0]])
        implied_all = ((seq_chk[:, 2] - seq_chk[:, 1]) * inv) % p
        assert (implied_all == b % p).all(), "implied-b sanity check failed"

        for P0 in prefix_lens:
            seqs = continue_rule(rule, inits, max_len, p).to(device)
            cond = f'prefix{P0}'
            entry[f'teacher_{cond}'] = teacher_forcing_eval(
                model, seqs, cand, p)
            entry[f'freegen_{cond}'] = free_gen_eval(
                model, seqs[:, :P0].contiguous(), cand, p, gen_len)
            if verbose:
                tf = entry[f'teacher_{cond}']
                fg = entry[f'freegen_{cond}']['final_alive_set']
                # teacher-forcing match at the first predicted position
                # after the prefix (index P0 in the match lists)
                mt = tf['true'][P0]
                m1 = tf['b1'][P0]
                m2 = tf['b2'][P0]
                mn = tf['neither'][P0]
                print(f"b={b:4d} {cond} | first-pos true={mt:.3f} "
                      f"b1={m1:.3f} b2={m2:.3f} neither={mn:.3f} | "
                      f"lock-in: {fg}", flush=True)
        report.setdefault('results', []).append(entry)
    return report


def main():
    ap = argparse.ArgumentParser(
        description="Rule generalization probe (mixed_ab, a=1)")
    ap.add_argument('model', help="path to .pth checkpoint")
    ap.add_argument('--b-values', default='0,3,4,5,8,16,32,64',
                    help="comma-separated new b values (trained b's are "
                         "always added as controls)")
    ap.add_argument('--samples', type=int, default=256)
    ap.add_argument('--gen-len', type=int, default=16,
                    help="tokens generated in free-generation condition")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help="write JSON report here")
    args = ap.parse_args()

    b_values = [int(x) for x in args.b_values.split(',') if x.strip()]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model(args.model, device=device)
    config = checkpoint['config']

    report = run(model, config, b_values, args.samples, args.gen_len,
                 args.seed, device)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=1)
        print(f"\nReport written to {args.out}")


if __name__ == '__main__':
    main()
