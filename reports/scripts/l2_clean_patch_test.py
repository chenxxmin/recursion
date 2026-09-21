"""l2 clean-patch recipes test on d512l2r8h4 (bigP, rules (1,1)/(1,2)).

Token x_(q-1) (the lag2 token from query q) appears in layer-0 outputs at:
  index q-2 (skip/emb), index q-1 (L0 lag1 head h0), index q (L0 lag2 head h3).
Test which manipulation of 'the lag2 token as seen by layer 1' gives clean
rule flips (match_a up, src~0, neither~0), for input B patched from A:
  1. emb@(q-2)               — token at its own position (with K/V ripples)
  2. L0h0@(q-1)              — the lag1-fetched copy only
  3. L0h3@q                  — the lag2-fetched copy at the query position
  4. coordinated: all three  — full replacement of the token everywhere
"""
import sys
import torch

sys.path.insert(0, 'src')
from analyze_attention import load_model
from activation_patch import (generate_paired_samples, rule_targets,
                              capture_hidden, patched_logits)

DEV = 'cuda'
P = 127
RA, RB = (1, 1), (1, 2)
Q = 4  # k=5

path = '/mnt/workspace/hujiachen/models/mixed_ab_basic_bigP/mixed_basic_d512l2r8h4_P127_N2_e0.7_seed0.pth'
model, ck = load_model(path, device=DEV)
a, b = generate_paired_samples(P, RA, RB, 256, 12, seed=11)
a, b = a.to(DEV), b.to(DEV)
cache_a, _ = capture_hidden(model, a)
va_b = rule_targets(b, RA, P).to(DEV)  # 翻转目标：规则A × B历史
vb_b = rule_targets(b, RB, P).to(DEV)
va_a = rule_targets(a, RA, P).to(DEV)  # 源续写

k = Q + 1
conds = [
    ('emb@(q-2)          ', ('emb',), [Q - 2]),
    ('L0h0(lag1)@(q-1)   ', ('attn', 0, 0), [Q - 1]),
    ('L0h3(lag2)@q       ', ('attn', 0, 3), [Q]),
    ('coordinated 三点   ', None, None),  # 三次组合 patch
]
print(f'{"condition":>20} {"match_a":>8} {"match_b":>8} {"src":>6} {"neither":>8}')
for label, site, positions in conds[:3]:
    logits = patched_logits(model, b, site, positions, cache_a)
    pred = logits[:, Q].argmax(-1)
    ma = (pred == va_b[:, k]).float().mean().item()
    mb = (pred == vb_b[:, k]).float().mean().item()
    ms = (pred == va_a[:, k]).float().mean().item()
    nei = max(0.0, 1 - ma - mb - ms)
    print(f'{label:>20} {ma:8.2f} {mb:8.2f} {ms:6.2f} {nei:8.2f}')

# 组合 patch：手写 hook 依次应用三个 patch
# patched_logits 一次只能挂一个 site——连续调用三次（每次重算前向）
logits = patched_logits(model, b, ('emb',), [Q - 2], cache_a)
# 注意：连续调用 patched_logits 不会累积 patch（每次重新前向原始 idx）。
# 需要一个多 site 同时 patch 的版本——用 activation_patch 的原语手动组合：
from activation_patch import _forward_logits  # noqa: E402


def multi_patch_logits(model, idx, site_pos_list, source_cache):
    handles = []
    for site, positions in site_pos_list:
        kind = site[0]
        positions = [positions] if isinstance(positions, int) else list(positions)
        if kind == 'emb':
            def make_hook(src, pos):
                def hook(module, args):
                    y = args[0].clone()
                    y[:, pos, :] = src[:, pos, :]
                    return (y,)
                return hook
            h = model.transformer.h[0].register_forward_pre_hook(
                make_hook(source_cache[('emb',)], positions))
            handles.append(h)
        elif kind == 'attn':
            li, hh = site[1], site[2]
            block = model.transformer.h[li]
            hs = block.attn.head_size
            sl = slice(hh * hs, (hh + 1) * hs)

            def make_hook(src, pos, sl=sl, key=('attn', li)):
                def hook(module, args):
                    y = args[0].clone()
                    y[:, pos, sl] = src[:, pos, sl]
                    return (y,)
                return hook
            h = block.attn.c_proj.register_forward_pre_hook(
                make_hook(source_cache[('attn', li)], positions))
            handles.append(h)
    with torch.no_grad():
        logits = _forward_logits(model, idx)
    for h in handles:
        h.remove()
    return logits


logits = multi_patch_logits(model, b, [
    (('emb',), Q - 2),
    (('attn', 0, 0), Q - 1),
    (('attn', 0, 3), Q),
], cache_a)
pred = logits[:, Q].argmax(-1)
ma = (pred == va_b[:, k]).float().mean().item()
mb = (pred == vb_b[:, k]).float().mean().item()
ms = (pred == va_a[:, k]).float().mean().item()
nei = max(0.0, 1 - ma - mb - ms)
print(f'{"coordinated 三点":>20} {ma:8.2f} {mb:8.2f} {ms:6.2f} {nei:8.2f}')

# 对照：baseline 不 patch
from activation_patch import _forward_logits as _fl  # noqa: E402
with torch.no_grad():
    pred = _fl(model, b)[:, Q].argmax(-1)
print(f'{"baseline 不 patch":>20} {(pred == va_b[:, k]).float().mean().item():8.2f} '
      f'{(pred == vb_b[:, k]).float().mean().item():8.2f}')
