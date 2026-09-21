"""Lag1-head patch: is 'breakdown' actually coherent hybrid computation?

After patching the lag1 (operand-2) head at query q with the source run's
value, the model receives operands (x_(q+1)^input, x_q^src) with UNCHANGED
evidence (x_(q-1)^input still says the input rule). If the model still applies
a rule, the prediction should equal one of:
  hyb_in  = x_(q+1)^in  + c2_in * x_q^src   (input rule's coeffs, hybrid ops)
  hyb_src = x_(q+1)^in  + c2_src * x_q^src  (source rule's coeffs, hybrid ops)
Targets also include own-rule continuation, source continuation, neither.
"""
import sys
import torch

sys.path.insert(0, 'src')
from analyze_attention import load_model
from activation_patch import (generate_paired_samples, rule_targets,
                              capture_hidden, patched_logits)

DEV = 'cuda'
P = 127
RUNS = [
    ('h2_s0', 0,  # lag1 头 = head0
     '/mnt/workspace/hujiachen/models/mixed_ab_downscale_l1h4/mixed_basic_d256l1r8h2_P127_N2_e0.7_seed0.pth'),
    ('h4_s0', 3,  # lag1 头 = head3
     '/mnt/workspace/hujiachen/models/mixed_ab_downscale_l1h4/mixed_basic_d256l1r8h4_P127_N2_e0.7_seed0.pth'),
]
RA, RB = (1, 1), (1, 2)

for tag, lag1h, path in RUNS:
    model, ck = load_model(path, device=DEV)
    a, b = generate_paired_samples(P, RA, RB, 512, 12, seed=11)
    a, b = a.to(DEV), b.to(DEV)
    cache_a, _ = capture_hidden(model, a)
    cache_b, _ = capture_hidden(model, b)
    va_b = rule_targets(b, RA, P).to(DEV)
    vb_b = rule_targets(b, RB, P).to(DEV)
    va_a = rule_targets(a, RA, P).to(DEV)
    vb_a = rule_targets(a, RB, P).to(DEV)

    print(f'===== {tag} (lag1=head{lag1h}) =====')
    print(f'{"dir":>5} {"k":>3} {"hyb_in":>7} {"hyb_src":>8} {"own":>6} '
          f'{"src":>6} {"neither":>8}')
    for direc, idx, src in (('AtoB', b, a), ('BtoA', a, b)):
        cache_src = cache_a if direc == 'AtoB' else cache_b
        c2_in = RB[1] if direc == 'AtoB' else RA[1]
        c2_src = RA[1] if direc == 'AtoB' else RB[1]
        for k in range(3, 10):
            q = k - 1
            logits = patched_logits(model, idx, ('attn', 0, lag1h), q, cache_src)
            pred = logits[:, q].argmax(-1)
            hyb_in = (idx[:, q] + c2_in * src[:, q - 1]) % P
            hyb_src = (idx[:, q] + c2_src * src[:, q - 1]) % P
            v_own = (vb_b if direc == 'AtoB' else va_a)[:, k]
            v_src = (va_a if direc == 'AtoB' else vb_b)[:, k]
            m_hin = (pred == hyb_in).float().mean().item()
            m_hsrc = (pred == hyb_src).float().mean().item()
            m_own = (pred == v_own).float().mean().item()
            m_src = (pred == v_src).float().mean().item()
            nei = max(0.0, 1 - m_hin - m_hsrc - m_own - m_src)
            print(f'{direc:>5} {k:>3} {m_hin:7.2f} {m_hsrc:8.2f} '
                  f'{m_own:6.2f} {m_src:6.2f} {nei:8.2f}')
