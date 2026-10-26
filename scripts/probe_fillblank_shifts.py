#!/usr/bin/env python3
"""阶段1b：等效系数估计。对每个挖空位 k 和扰动位 j，测量
    pred(x_j+δ) - pred(x) 是否为 c·δ（mod 127）的线性关系，
读出模型在该位置的等效系数 c，再与前向/逆向候选公式对照。

候选公式（二阶 x_k = a*x_{k-1} + b*x_{k-2}）：
  前向 F : j=k-1 -> a ; j=k-2 -> b
  右逆 B1: x_k = (x_{k+1} - b*x_{k-1})/a
           j=k+1 -> a^{-1} ; j=k-1 -> -b*a^{-1}
  右逆 B2: x_k = (x_{k+2} - a*x_{k+1})/b
           j=k+2 -> b^{-1} ; j=k+1 -> -a*b^{-1}
  k=0   : x0 = (x2 - a*x1)/b : j=1 -> -a*b^{-1} ; j=2 -> b^{-1}
  k=1   : x1 = (x2 - b*x0)/a : j=0 -> -b*a^{-1} ; j=2 -> a^{-1}
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

import torch

ROOT = '/home/cxm/recursion'
sys.path.insert(0, os.path.join(ROOT, 'src'))
from probe_fillblank_operands import (load_model, test_windows, predict, P, BLANK)


def candidates(k, a, b):
    """该 (k,j) 各候选公式对应的等效系数；返回 {公式名: {j: coef}}。"""
    ai, bi = pow(a, -1, P), pow(b, -1, P)
    out = {}
    if k >= 2:
        out['F'] = {k - 1: a, k - 2: b}
    if k >= 1:
        out['B1'] = {k + 1: ai, k - 1: (-b * ai) % P}
    out['B2'] = {k + 2: bi, k + 1: (-a * bi) % P}
    if k == 0:
        out['inv0'] = {1: (-a * bi) % P, 2: bi}
    if k == 1:
        out['inv1'] = {0: (-b * ai) % P, 2: ai}
    return out


def run_model(path, a, b, device, n_windows=64, deltas=(1, -1, 7, -7)):
    model, _ = load_model(path, device)
    wins = test_windows(a, b)
    print(f'\n########## a{a}b{b} ({os.path.basename(path)[:60]}...)')
    for L in (3, 4, 5):
        print(f'  == L={L} ==')
        for k in range(L):
            base_seqs, vs = [], []
            for w in wins[:n_windows]:
                seq = w[:L].clone()
                vs.append(int(seq[k]))
                seq[k] = BLANK
                base_seqs.append(seq)
            preds = predict(model, base_seqs, device)
            correct = [i for i in range(len(vs)) if preds[i] == vs[i]]
            base_acc = len(correct) / len(vs)
            cand = candidates(k, a, b)
            cells = []
            for j in range(L):
                if j == k:
                    continue
                shifts = defaultdict(list)   # delta -> list of (pred2-pred)%P
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
                # 估计 c：用 delta=1 的众数；再算全体 delta 的线性一致率
                c1 = shifts[1]
                if not c1:
                    continue
                c_est, cnt = Counter(c1).most_common(1)[0]
                total, hit = 0, 0
                for delta, vals in shifts.items():
                    for v in vals:
                        total += 1
                        hit += (v == (c_est * delta) % P)
                consist = hit / total if total else 0.0
                changed = sum(1 for v in c1 if v != 0) / len(c1)
                if changed < 0.3:
                    continue
                # 候选公式匹配
                matches = []
                for fname, fmap in cand.items():
                    if j in fmap and fmap[j] == c_est and consist > 0.8:
                        matches.append(fname)
                cells.append(f'x{j}:c={c_est}({consist*100:.0f}%|{changed*100:.0f}%变){"/".join(matches) if matches else "?"}')
            print(f'    k={k} base={base_acc*100:.0f}% | ' + ('; '.join(cells) if cells else '无显著影响位'))
            # 打印各候选公式的理论系数备查（只打印一次 k>=2 的）
        if L == 5:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--device', default='cuda:7')
    ap.add_argument('--n-windows', type=int, default=64)
    args = ap.parse_args()
    for path in args.ckpt:
        m = re.search(r'a(\d+)b(\d+)', os.path.basename(path))
        run_model(path, int(m.group(1)), int(m.group(2)), args.device, args.n_windows)


if __name__ == '__main__':
    main()
