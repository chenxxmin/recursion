"""Rules (1,1)/(2,3): decision function over residual space.

For input rule A samples, patch lag2 head with synthetic V(x~) for all x~,
compute both rules' residuals per sample:
    r_A = (x_(q+1) - x_q) - x~        (rule A=(1,1))
    r_B = (x_(q+1) - 2x_q) - 3x~      (rule B=(2,3))
Outputs: 1D curve (flip rate vs r_B) + 2D heatmap over (r_A, r_B).
"""
import sys
import torch
import numpy as np
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
RULES = [(1, 1), (2, 3)]
LAG2H = 1  # seed17996 的 lag2 头

path = ('/mnt/workspace/hujiachen/models/mixed_basic_d256l1r8h2_p127_rules12345/'
        'mixed_basic_d256l1r8h2_P127_N2_e0.7_seed17996.pth')
model, ck = load_model(path, device=DEV)
block = model.transformer.h[0]
hs = block.attn.head_size
d = block.attn.n_embd
vsl = slice(2 * d + LAG2H * hs, 2 * d + (LAG2H + 1) * hs)
toks = torch.arange(P, device=DEV)
with torch.no_grad():
    v_all = block.attn.c_attn(block.ln_1(model.transformer.wte(toks)))[:, vsl]

a, _ = generate_paired_samples(P, (1, 1), (2, 3), 512, 12, seed=11)
a = a.to(DEV)
cache_a, _ = capture_hidden(model, a)
attv_a = cache_a[('attn', 0)]
v_flip = rule_targets(a, (2, 3), P).to(DEV)[:, Q + 1]  # 规则B × A历史
v_stay = rule_targets(a, (1, 1), P).to(DEV)[:, Q + 1]

N = a.shape[0]
flip_mat = torch.zeros(N, P, dtype=torch.bool)
stay_mat = torch.zeros(N, P, dtype=torch.bool)
for xt in range(P):
    fake = attv_a.clone()
    fake[:, Q, LAG2H * hs:(LAG2H + 1) * hs] = v_all[xt]
    pred = patched_logits(model, a, ('attn', 0, LAG2H), Q,
                          {('attn', 0): fake})[:, Q].argmax(-1)
    flip_mat[:, xt] = (pred == v_flip).cpu()
    stay_mat[:, xt] = (pred == v_stay).cpu()

# 逐样本逐 x~ 的两条残差
xt_all = torch.arange(P).unsqueeze(0).cpu()
D1 = (a[:, Q] - a[:, Q - 1]).cpu()              # x_(q+1) - x_q
D2 = (a[:, Q] - 2 * a[:, Q - 1]).cpu()          # x_(q+1) - 2x_q
rA = (D1.unsqueeze(1) - xt_all) % P             # (N, P)
rB = (D2.unsqueeze(1) - 3 * xt_all) % P

# ---- 1D: flip vs r_B ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
for mat, mk, lbl in ((flip_mat, 'o', 'flip to B=(2,3)'),
                     (stay_mat, '^', 'stay A=(1,1)')):
    xs, ys = [], []
    for rv in range(P):
        m = rB == rv
        if m.sum() == 0:
            continue
        xs.append(rv if rv <= P // 2 else rv - P)
        ys.append(float((mat.float() * m.float()).sum() / m.sum()))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ax.scatter([xs[i] for i in order], [ys[i] for i in order],
               marker=mk, s=22, alpha=0.8, label=lbl)
ax.axvline(0, color='gray', alpha=0.4)
ax.set_xlabel('r_B = (x_(q+1) - 2 x_q) - 3 x~  (mod 127, centered)')
ax.set_ylabel('rate')
ax.set_title('BtoA: flip rate vs rule-B residual (input rule A)')
ax.legend()
ax.grid(alpha=0.25)

# ---- 2D heatmap over (r_A, r_B)，居中坐标 ----
ax = axes[1]
H = np.full((P, P), np.nan)
f = flip_mat.float().numpy()
rAc = np.where(rA.numpy() > P // 2, rA.numpy() - P, rA.numpy())  # 居中
rBc = np.where(rB.numpy() > P // 2, rB.numpy() - P, rB.numpy())
for iA in range(P):
    for iB in range(P):
        m = (rAc == iA - 63) & (rBc == iB - 63)
        if m.sum() > 0:
            H[iB, iA] = f[m].mean()
im = ax.imshow(H, origin='lower', cmap='viridis',
               extent=(-63.5, 63.5, -63.5, 63.5))
ax.set_xlabel('r_A (centered)')
ax.set_ylabel('r_B (centered)')
ax.set_title('flip-to-B rate over residual plane')

plt.tight_layout()
plt.savefig('.chain_tmp/rules23_resid.png', dpi=110)
print('saved .chain_tmp/rules23_resid.png')
