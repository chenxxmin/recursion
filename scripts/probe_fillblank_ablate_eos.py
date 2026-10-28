#!/usr/bin/env python3
"""末位/中段电路的 EOS 通路消融：检验"最简电路"是否够用。

模式：
  baseline      不干预
  only_blank    EOS 的 query 只准 attend BLANK 位（和 EOS 自身），其余全部置零
  no_blank      EOS 的 query 禁止 attend BLANK 位
  only_operands EOS 只准 attend 前向操作数位 x_{k-1},x_{k-2}（和 EOS 自身）
对 att 在 softmax 后做 mask 并重新归一化，比较 EOS 答案准确率。
"""
import argparse, os, re, sys
import torch
import torch.nn.functional as F
import math

ROOT = '/home/cxm/recursion'
sys.path.insert(0, os.path.join(ROOT, 'src'))
from models import apply_rotary_emb
from probe_fillblank_operands import load_model, test_windows, P, BLANK, EOS


def patched_forward(module, mask_fn):
    def forward(x):
        B, T, C = x.size()
        qkv = module.c_attn(x)
        q, k, v = qkv.split(module.n_embd, dim=2)
        q = q.view(B, T, module.n_head, module.head_size).transpose(1, 2)
        k = k.view(B, T, module.n_head, module.head_size).transpose(1, 2)
        v = v.view(B, T, module.n_head, module.head_size).transpose(1, 2)
        if module.rope is not None:
            cos, sin = module.rope(q, seq_len=T)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(module.head_size)
        att = att.masked_fill(module.causal_mask[:, :, :T, :T] == 0, -1e9)
        att = F.softmax(att, dim=-1)
        att = mask_fn(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return module.resid_dropout(module.c_proj(y))
    return forward


def mask_for(mode, L, k):
    def fn(att):
        if mode == 'baseline':
            return att
        keep = torch.ones_like(att)
        if mode == 'only_blank':
            keep[:, :, L, :] = 0.0
            keep[:, :, L, k] = 1.0
            keep[:, :, L, L] = 1.0
        elif mode == 'no_blank':
            keep[:, :, L, k] = 0.0
        elif mode == 'only_operands':
            keep[:, :, L, :] = 0.0
            for j in (k - 1, k - 2):
                if j >= 0:
                    keep[:, :, L, j] = 1.0
            keep[:, :, L, L] = 1.0
        att = att * keep
        return att / att.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return fn


@torch.no_grad()
def accuracy(model, wins, L, k, device, mode):
    for block in model.transformer.h:
        block.attn.forward = patched_forward(block.attn, mask_for(mode, L, k))
    correct, total = 0, 0
    for w in wins:
        seq = w[:L].clone()
        v = int(seq[k])
        seq[k] = BLANK
        x = torch.cat([seq, torch.tensor([EOS])]).unsqueeze(0).to(device)
        logits, _ = model(x)
        pred = int(logits[0, -1, :P].argmax())
        correct += (pred == v)
        total += 1
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cuda:7')
    ap.add_argument('--L', type=int, default=5)
    ap.add_argument('--ks', type=int, nargs='+', default=[0, 2, 4])
    ap.add_argument('--n-windows', type=int, default=128)
    args = ap.parse_args()
    m = re.search(r'a(\d+)b(\d+)', os.path.basename(args.ckpt))
    a, b = int(m.group(1)), int(m.group(2))
    wins = test_windows(a, b)[:args.n_windows]
    model, _ = load_model(args.ckpt, args.device)
    print(f'a{a}b{b} EOS-path ablation, L={args.L}, n={len(wins)}')
    for k in args.ks:
        if k >= args.L:
            continue
        row = []
        for mode in ('baseline', 'only_blank', 'no_blank', 'only_operands'):
            acc = accuracy(model, wins, args.L, k, args.device, mode)
            row.append(f'{mode}={acc*100:.1f}%')
        print(f'  k={k}: ' + '  '.join(row))


if __name__ == '__main__':
    main()
