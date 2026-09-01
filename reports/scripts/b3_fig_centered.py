"""B3 sweep figure: flip rate vs implied ratio c2_hat, both directions,
both l1 models. Saves PNG."""
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

    for direc, idx, src, in_c2, tgt_c2 in (
            ('AtoB (flip to rule A)', b, a, 2, 1),
            ('BtoA (flip to rule B)', a, b, 1, 2)):
        cache_src, _ = capture_hidden(model, src)
        attv_in = capture_hidden(model, idx)[0][('attn', 0)]
        # 翻转目标 = 对方规则 × 输入历史
        v_flip = rule_targets(idx, (1, tgt_c2), P).to(DEV)
        k = Q + 1
        delta = (idx[:, Q] - idx[:, Q - 1]) % P
        N = idx.shape[0]
        flip_mat = torch.zeros(N, P, dtype=torch.bool)
        for xt in range(P):
            fake = attv_in.clone()
            fake[:, Q, lag2h * hs:(lag2h + 1) * hs] = v_all[xt]
            pred = patched_logits(model, idx, ('attn', 0, lag2h), Q,
                                  {('attn', 0): fake})[:, Q].argmax(-1)
            flip_mat[:, xt] = (pred == v_flip[:, k]).cpu()
        inv = torch.tensor([0] + [pow(int(t), -1, P) for t in range(1, P)])
        ratio = (delta.cpu().unsqueeze(1) % P) * inv.unsqueeze(0) % P
        valid = torch.arange(P).unsqueeze(0).expand(N, P) > 0
        rates = []
        for c2v in range(P):
            m = (ratio == c2v) & valid
            rates.append(float((flip_mat.float() * m).sum() / m.sum())
                         if m.sum() else float('nan'))
        # 居中横轴：c2v > 63 映射为负数（模 p 环绕）
        xs = [c2v if c2v <= 63 else c2v - P for c2v in range(P)]
        order = sorted(range(P), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        rates = [rates[i] for i in order]
        style = '-' if direc.startswith('AtoB') else '--'
        ax.plot(xs, rates, style, label=direc, lw=1.5)
    ax.axvline(0, color='gray', alpha=0.3, lw=0.8)
    ax.axvline(1, color='gray', alpha=0.3, lw=0.8)
    ax.axvline(2, color='gray', alpha=0.3, lw=0.8)
    ax.set_xlabel('implied ratio  c2_hat = delta / x_tilde  (centered mod 127)')
    ax.set_ylabel('flip rate')
    ax.set_title(f'{title}, q={Q}')
    ax.legend()
    ax.set_ylim(-0.02, 1.05)

plt.tight_layout()
plt.savefig('.chain_tmp/b3_sweep_fig_centered.png', dpi=110)
print('saved .chain_tmp/b3_sweep_fig_centered.png')
