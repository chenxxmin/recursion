#!/usr/bin/env python3
"""阶段3：fill_blank 模型的 attention 路由分析。

回答两个问题：
  1) EOS（唯一有 loss 的输出位）是均匀 attend 全窗，还是尖锐指向 BLANK/操作数？
  2) BLANK 位置自己向前（k-1,k-2）还是向后（k+1,k+2）attend？
     按 k 类别分桶：首位 k=0 / 末位 k=L-1 / 中段（其余）。
同时输出 attention 权重与 权重×value范数 两种口径，防止"权重均匀但 value 特异"。
"""
import argparse, os, re, sys
from collections import defaultdict

import torch

ROOT = '/home/cxm/recursion'
sys.path.insert(0, os.path.join(ROOT, 'src'))
from models import apply_rotary_emb
from probe_fillblank_operands import load_model, test_windows, P, BLANK, EOS
import math


class AttentionCapture:
    """hook 进每个 CausalSelfAttention，重算 att 权重与 value 范数。"""

    def __init__(self, model):
        self.records = []  # 每层 (att, vnorm)：att (B,H,T,T)，vnorm (B,H,T)
        self.hooks = []
        for block in model.transformer.h:
            h = block.attn.register_forward_hook(self._hook)
            self.hooks.append(h)

    def _hook(self, module, args, output):
        (x,) = args
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
        att = torch.softmax(att, dim=-1)
        vnorm = v.norm(dim=-1)  # (B,H,T)
        self.records.append((att.detach().cpu(), vnorm.detach().cpu()))

    def close(self):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def run_model(path, a, b, device, n_windows=48):
    model, _ = load_model(path, device)
    cap = AttentionCapture(model)
    wins = test_windows(a, b)
    print(f'\n########## a{a}b{b} attention ({os.path.basename(path)[:55]}...)')
    for L in (3, 4, 5):
        print(f'  == L={L} ==')
        for k in range(L):
            seqs = []
            for w in wins[:n_windows]:
                seq = w[:L].clone()
                seq[k] = BLANK
                seqs.append(torch.cat([seq, torch.tensor([EOS])]))
            x = torch.stack(seqs).to(device)
            cap.records = []
            model(x)
            cat = 'first' if k == 0 else ('last' if k == L - 1 else 'mid')
            print(f'    -- k={k} ({cat}) --')
            for layer, (att, vnorm) in enumerate(cap.records):
                eos = att[:, :, L, :].mean(dim=(0, 1))       # (T,) 头平均
                eosv = (att[:, :, L, :] * vnorm).mean(dim=(0, 1))
                blk = att[:, :, k, :].mean(dim=(0, 1))
                blv = (att[:, :, k, :] * vnorm).mean(dim=(0, 1))
                t = att.shape[-1]
                fmt = lambda v: ' '.join(f'{v[j]:.2f}' for j in range(t))
                print(f'      l{layer} EOS→: {fmt(eos)} | ×val: {fmt(eosv)}')
                print(f'      l{layer} BLK→: {fmt(blk)} | ×val: {fmt(blv)}')
                # 每个 head 的 top-2 指向（BLANK query）
                tops = []
                for h in range(att.shape[1]):
                    w = att[:, h, k, :].mean(0)
                    idx = w.topk(min(2, t)).indices.tolist()
                    tops.append(f'h{h}[' + ','.join(f'x{j}:{w[j]:.2f}' for j in idx) + ']')
                print(f'      l{layer} BLK heads: ' + ' '.join(tops))
                # 右邻位 query（逆向电路的候选计算位）：看它是否同时读 BLANK 与 x_{k-1}
                if k + 1 < L:
                    rn = att[:, :, k + 1, :].mean(dim=(0, 1))
                    rnv = (att[:, :, k + 1, :] * vnorm).mean(dim=(0, 1))
                    print(f'      l{layer} R(x{k+1})→: {fmt(rn)} | ×val: {fmt(rnv)}')
    cap.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--device', default='cuda:7')
    ap.add_argument('--n-windows', type=int, default=48)
    args = ap.parse_args()
    for path in args.ckpt:
        m = re.search(r'a(\d+)b(\d+)', os.path.basename(path))
        run_model(path, int(m.group(1)), int(m.group(2)), args.device, args.n_windows)


if __name__ == '__main__':
    main()
