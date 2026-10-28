#!/usr/bin/env python3
"""value_only 纠错模型的逐位置等效系数分析（len5，全位置暖启动 ckpt）。

对每个出错位 k∈{0..4}：
  - 基线：污损窗（k 位为错误值）下 EOS 的答案与准确率；
  - 对每个位置 j（含错误位本身）做 ±δ 扰动，测答案偏移是否满足 c·δ（mod 127）；
  - 与前向/逆向公式对照（见 probe_fillblank_shifts.candidates）。
错误位 j=k 若系数≈0，说明模型学会忽略错值；若非零，说明错值仍参与计算。
"""
import argparse, os, re, sys
from collections import Counter, defaultdict

import torch

ROOT = '/home/cxm/recursion'
sys.path.insert(0, os.path.join(ROOT, 'src'))
from probe_fillblank_operands import load_model, test_windows, predict, P
from probe_fillblank_shifts import candidates


@torch.no_grad()
def run_model(path, a, b, device, n_windows=64, deltas=(1, -1, 7, -7)):
    model, _ = load_model(path, device)
    wins = test_windows(a, b)
    print(f'\n########## a{a}b{b} value_only shift analysis (n={n_windows})')
    for k in range(5):
        base_seqs, vs = [], []
        for w in wins[:n_windows]:
            seq = w[:5].clone()
            vs.append(int(seq[k]))
            seq[k] = (int(seq[k]) + 37) % P   # 与训练一致的污损
            base_seqs.append(seq)
        preds = predict(model, base_seqs, device)
        correct = [i for i in range(len(vs)) if preds[i] == vs[i]]
        base_acc = len(correct) / len(vs)
        cand = candidates(k, a, b)
        cells = []
        for j in range(5):
            shifts = defaultdict(list)
            for delta in deltas:
                pseqs = []
                for i in correct:
                    seq = base_seqs[i].clone()
                    seq[j] = (int(seq[j]) + delta) % P
                    pseqs.append(seq)
                if not pseqs:
                    continue
                pp = predict(model, pseqs, device)
                for ii, i in enumerate(correct):
                    shifts[delta].append((pp[ii] - preds[i]) % P)
            if not shifts:
                continue
            c1 = shifts[1]
            c_est, _ = Counter(c1).most_common(1)[0]
            total, hit = 0, 0
            for delta, vals in shifts.items():
                for v in vals:
                    total += 1
                    hit += (v == (c_est * delta) % P)
            consist = hit / total if total else 0.0
            changed = sum(1 for v in c1 if v != 0) / len(c1)
            if changed < 0.3:
                continue
            matches = [fname for fname, fmap in cand.items()
                       if j in fmap and fmap[j] == c_est and consist > 0.8]
            tag = j == k and '（错值）' or ''
            cells.append(f'x{j}{tag}:c={c_est}({consist*100:.0f}%|{changed*100:.0f}%变){"/".join(matches) if matches else "?"}')
        print(f'  k={k} base={base_acc*100:.1f}% | ' + ('; '.join(cells) if cells else '扰动无显著影响'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n-windows', type=int, default=64)
    args = ap.parse_args()
    for path in args.ckpt:
        m = re.search(r'a(\d+)b(\d+)', os.path.basename(path))
        run_model(path, int(m.group(1)), int(m.group(2)), args.device, args.n_windows)


if __name__ == '__main__':
    main()
