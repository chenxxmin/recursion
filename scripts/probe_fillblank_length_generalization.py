#!/usr/bin/env python3
"""零样本长度泛化：len3~5 收敛模型直接评 len6/7/8（不训练）。
回答"电路是否可移植到更长窗口"：逐 (L,k) 报 EOS 答案准确率。
"""
import argparse, os, re, sys
import torch

ROOT = '/home/cxm/recursion'
sys.path.insert(0, os.path.join(ROOT, 'src'))
from probe_fillblank_operands import load_model, test_windows, predict, P, BLANK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--device', default='cuda:7')
    ap.add_argument('--n-windows', type=int, default=128)
    ap.add_argument('--lengths', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8])
    args = ap.parse_args()
    for path in args.ckpt:
        m = re.search(r'a(\d+)b(\d+)', os.path.basename(path))
        a, b = int(m.group(1)), int(m.group(2))
        model, _ = load_model(path, args.device)
        # ckpt 的 block_size 只有 8，长序列需要扩大 causal_mask（RoPE 本身不受限）
        n = max(args.lengths) + 2
        for block in model.transformer.h:
            block.attn.causal_mask = torch.tril(
                torch.ones(n, n)).view(1, 1, n, n).to(args.device)
        print(f'\n### a{a}b{b} zero-shot length generalization')
        for L in args.lengths:
            wins = test_windows(a, b, length=L)[:args.n_windows]
            accs = []
            for k in range(L):
                seqs, vs = [], []
                for w in wins:
                    seq = w[:L].clone()
                    vs.append(int(seq[k]))
                    seq[k] = BLANK
                    seqs.append(seq)
                preds = predict(model, seqs, args.device)
                accs.append(sum(p == v for p, v in zip(preds, vs)) / len(vs))
            mean = sum(accs) / len(accs)
            cells = ' '.join(f'k{k}={acc*100:.0f}' for k, acc in enumerate(accs))
            print(f'  L={L}: mean={mean*100:5.1f}% | {cells}')


if __name__ == '__main__':
    main()
