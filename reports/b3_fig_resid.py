"""Sweep figure with RESIDUAL x-axis: flip rate vs target-rule residual
r_X = x_(q+1) - c1_X * x_q - c2_X * x~   (mod p, centered at 0).

Basins should align at r=0 for both directions.
"""
import sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')
from analyze_attention import load_model
from activation_patch import (generate_paired_samples, rule_targets,
                              capture_hidden, patched_logits)

DEV = 'cuda'
P = 127
Q = 4
RUNS = [
    ('h2 (d256 l1 h2)', 1,
     '/data/cxm/models/mixed_ab_downscale_l1h4/mixed_basic_d256l1r8h2_P127_N2_e0.7_seed0.pth'),
    ('h4 (d256 l1 h4)', 0,
     '/data/cxm/models/mixed_ab_downscale_l1h4/mixed_basic_d256l1r8h4_P127_N2_e0.7_seed0.pth'),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, (title, lag2h, path) in zip(axes, RUNS):
    model, ck = load_model(path, device=DEV)
    block = model.transformer.h[0]
    hs = block.attn.head_size
    d = block.attn.n_embd
    vsl = slice(2 * d + lag2h * hs, 2 * d + (lag2h + 1) * hs)
    toks = torch.arange(P, device=DEV)
    with torch.no_grad():
        v_all = block.attn.c_attn(block.ln_1(model.transformer.wte(toks)))[:, vsl]

    a, b = generate_paired_samples(P, (1, 1), (1, 2), 512, 12, seed=11)
    a, b = a.to(DEV), b.to(DEV)

    for direc, idx, tgt_rule, mk in (
            ('AtoB (resid of rule A)', b, (1, 1), 'o'),
            ('BtoA (resid of rule B)', a, (1, 2), 's')):
        attv_in = capture_hidden(model, idx)[0][('attn', 0)]
        v_flip = rule_targets(idx, tgt_rule, P).to(DEV)[:, Q + 1]
        c1t, c2t = tgt_rule
        N = idx.shape[0]
        flip_mat = torch.zeros(N, P, dtype=torch.bool)
        for xt in range(P):
            fake = attv_in.clone()
            fake[:, Q, lag2h * hs:(lag2h + 1) * hs] = v_all[xt]
            pred = patched_logits(model, idx, ('attn', 0, lag2h), Q,
                                  {('attn', 0): fake})[:, Q].argmax(-1)
            flip_mat[:, xt] = (pred == v_flip).cpu()
        # 目标规则残差（逐样本）：r = (x_(q+1) - c1*x_q) - c2*x~  (mod p)
        xt_all = torch.arange(P).unsqueeze(0)          # (1,P)
        resid = ((idx[:, Q].cpu() - c1t * idx[:, Q - 1].cpu()).unsqueeze(1)
                 - c2t * xt_all) % P                    # (N,P)
        rates = []
        xs = []
        for rv in range(P):
            m = resid == rv
            if m.sum() == 0:
                continue
            x = rv if rv <= P // 2 else rv - P
            xs.append(x)
            rates.append(float((flip_mat.float() * m.float()).sum() / m.sum()))
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        rates = [rates[i] for i in order]
        ax.scatter(xs, rates, marker=mk, s=22, alpha=0.8, label=direc)
    ax.axvline(0, color='gray', alpha=0.4, lw=0.8)
    ax.set_xlabel('target-rule residual  r = (x_(q+1) - c1*x_q) - c2*x~  (mod 127, centered)')
    ax.set_ylabel('flip rate')
    ax.set_title(f'{title}, q={Q}')
    ax.legend()
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
plt.savefig('.chain_tmp/b3_sweep_fig_resid.png', dpi=110)
print('saved .chain_tmp/b3_sweep_fig_resid.png')
