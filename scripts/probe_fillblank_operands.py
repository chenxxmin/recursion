#!/usr/bin/env python3
"""阶段1：fill_blank 模型的操作数因果扰动探针。

对成功 ckpt（len3~5, wd=2）做受控扰动：
  - 基线：新鲜测试窗 + 挖空位 k，读 EOS 槽 argmax 作为答案；
  - 扰动：把其他位置 x_j 加减 δ（mod 127），BLANK 不动；
  - 记录 change_rate（答案改变比例）与 rule_match（答案等于局部递推公式
    预言的新值的比例）。k>=2 用前向公式 v'=v+coef*delta（j=k-1 -> a, j=k-2 -> b），
    k=0/1 用最小逆向公式（x0=(x2-a*x1)/b，x1=(x2-b*x0)/a），其余位置预期不变。
若 change_rate 集中在操作数位且 rule_match 接近 100%，即证明模型执行的是
局部递推规则而非全窗查找表。
"""
import argparse, json, os, sys
from collections import defaultdict

import torch

ROOT = '/home/cxm/recursion'
sys.path.insert(0, os.path.join(ROOT, 'src'))
from datasets import RecurrenceDataset
from models import FibonacciTransformer
from rules import LinearRecurrenceRule

P = 127
BLANK = P
EOS = P + 1


def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = FibonacciTransformer(
        p=cfg['p'], d_model=cfg['d_model'], n_head=cfg['n_head'],
        n_layer=cfg['n_layer'], block_size=cfg['block_size'],
        mlp_ratio=cfg['mlp_ratio'], vocab_size=cfg['vocab_size'])
    model.load_state_dict(ck['model_state_dict'])
    model.to(device).eval()
    return model, cfg


def test_windows(a, b, length=5):
    rule = LinearRecurrenceRule([a, b], P)
    ds = RecurrenceDataset(p=P, recurrence_fn=rule.next_fn(),
                           recurrence_name=rule.name, init_len=2,
                           num_samples=int(P * P * 0.7), length=length,
                           verbose=False)
    ds.run_full_split()
    out = []
    for item in ds.test_samples:
        seq = item[0] if isinstance(item, (list, tuple)) else item
        out.append(seq.clone())
    return out


@torch.no_grad()
def predict(model, xs, device, bs=256):
    """xs: list of 1-D LongTensor（含 BLANK，长度 L），末尾补 EOS；返回答案 int 列表。"""
    ans = []
    for i in range(0, len(xs), bs):
        chunk = xs[i:i + bs]
        x = torch.stack([torch.cat([s, torch.tensor([EOS])]) for s in chunk]).to(device)
        logits, _ = model(x)
        last = logits[:, -1, :P]  # EOS 槽，限制在 127 个数值类上
        ans.extend(last.argmax(dim=-1).tolist())
    return ans


def shift_coef(k, j, a, b):
    """局部递推公式下，扰动 x_j 对 x_k 的系数（mod 127）；0 表示无关。"""
    ai, bi = pow(a, -1, P), pow(b, -1, P)
    if k >= 2:
        if j == k - 1: return a
        if j == k - 2: return b
        return 0
    if k == 0:
        if j == 1: return (-a * bi) % P
        if j == 2: return bi
        return 0
    if k == 1:
        if j == 0: return (-b * ai) % P
        if j == 2: return ai
        return 0
    return 0


def run_model(path, a, b, device, n_windows=48, deltas=(1, -1, 7, -7)):
    model, cfg = load_model(path, device)
    name = os.path.basename(path).replace('.pth', '')
    wins = test_windows(a, b)
    print(f'\n########## a{a}b{b} | {name} | test windows={len(wins)}')
    report = {'coef': [a, b], 'ckpt': name, 'by_L': {}}
    for L in (3, 4, 5):
        per_k = {}
        for k in range(L):
            base_seqs, vs = [], []
            for w in wins[:n_windows]:
                seq = w[:L].clone()
                v = int(seq[k])
                seq[k] = BLANK
                base_seqs.append(seq)
                vs.append(v)
            preds = predict(model, base_seqs, device)
            correct = [i for i in range(len(vs)) if preds[i] == vs[i]]
            base_acc = len(correct) / len(vs)
            # 扰动（只用基线答对的窗，隔离规则行为）
            change = defaultdict(list)
            match = defaultdict(list)
            for j in range(L):
                if j == k:
                    continue
                for delta in deltas:
                    pseqs, expects = [], []
                    for i in correct:
                        seq = base_seqs[i].clone()
                        seq[j] = (int(seq[j]) + delta) % P
                        pseqs.append(seq)
                        coef = shift_coef(k, j, a, b)
                        expects.append((vs[i] + coef * delta) % P)
                    if not pseqs:
                        continue
                    pp = predict(model, pseqs, device)
                    for ii, i in enumerate(correct):
                        change[j].append(pp[ii] != preds[i])
                        match[j].append(pp[ii] == expects[ii])
            per_k[k] = {
                'base_acc': base_acc,
                'influence': {j: sum(v) / len(v) for j, v in change.items()},
                'rule_match': {j: sum(v) / len(v) for j, v in match.items()},
            }
        report['by_L'][L] = per_k
        print(f'  == L={L} ==')
        for k in range(L):
            d = per_k[k]
            infl = sorted(d['influence'].items(), key=lambda kv: -kv[1])
            cells = []
            for j, c in infl:
                if c < 0.05:
                    continue
                rm = d['rule_match'][j]
                cells.append(f'x{j}:Δ{c*100:.0f}%/规则命中{rm*100:.0f}%')
            n_irrel = sum(1 for j, c in d['influence'].items() if c < 0.05)
            print(f'    k={k} base={d["base_acc"]*100:.1f}% | ' +
                  ('; '.join(cells) if cells else '（扰动几乎无影响）') +
                  f' | 无影响位×{n_irrel}')
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--device', default='cuda:7')
    ap.add_argument('--n-windows', type=int, default=48)
    ap.add_argument('--out', default='reports/fillblank_operand_probe.json')
    args = ap.parse_args()
    reports = []
    for path in args.ckpt:
        m = re.search(r'a(\d+)b(\d+)', os.path.basename(path))
        a, b = int(m.group(1)), int(m.group(2))
        reports.append(run_model(path, a, b, args.device, args.n_windows))
    with open(args.out, 'w') as f:
        json.dump(reports, f, indent=1)
    print('\nsaved', args.out)


if __name__ == '__main__':
    import re
    main()
