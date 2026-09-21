"""Attention probe: hole-lag decomposition for action+misslen=1 models.

For every predicted value x_j on the test set, categorize by hole context:
  lag0  : x_j itself is corrupted (predict the hole)
  lag1  : x_{j-1} is corrupted (predict right after a hole)
  clean : neither
Then report, per category and per head, where the query position (which holds
the rule flag f_j) puts its attention mass:
  x_{j-1} / x_{j-2} / x_{j-3} (recent values), x1+x2 (initial conditions),
  hole tokens, other flags, other earlier values, self.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python scripts/probe_attention_hole_lag.py [pth]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch

import experiment
from analyze_attention import get_attention_weights, load_model

DEFAULT_PTH = ('/mnt/workspace/hujiachen/models/action_p127_tr64_ood128_misslen/'
               'action_d1024l1r4h4_P127_tr64ood128_N2_randmiss0.1len1_seed17996.pth')
N_SAMPLES = 500  # 测试集前 N 条足够收敛统计


def main():
    pth = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PTH
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model(pth, device=device)
    config = checkpoint['config']
    P = config['p']
    ab_pairs = [[1, 1], [2, 3]]

    from datasets import ActionDataset
    test_ds = ActionDataset(p=P, ab_pairs=ab_pairs, num_samples=N_SAMPLES,
                            length=128, seed=17996 + 1, missing_prob=0.1, miss_len=1)
    print(f"model: {os.path.basename(pth)}")
    print(f"test samples: {len(test_ds)}, p={P}")

    # 类别 -> [n, correct, per-head 分组质量累计]
    groups = ['x_j-1', 'x_j-2', 'x_j-3', 'x1x2', 'holes', 'flags', 'other_vals', 'self']
    cats = ('lag0', 'lag1', 'clean')
    stats = {c: {'n': 0, 'ok': 0, 'mass': None} for c in cats}
    n_layers = None

    with torch.no_grad():
        for idx in range(len(test_ds)):
            view, clean, _ = test_ds[idx]
            L = test_ds.length
            att_layers = get_attention_weights(model, view.unsqueeze(0).to(device))
            logits = model(view.unsqueeze(0).to(device))[0][0]  # (T-1? no: (T, vocab)) — logits[t] ~ seq[t+1]
            if n_layers is None:
                n_layers = len(att_layers)
                n_head = att_layers[0].shape[0]
                for c in cats:
                    stats[c]['mass'] = torch.zeros(n_layers, n_head, len(groups))

            for j in range(3, L + 1):
                vi = 2 * j - 3          # x_j 的 seq 位置
                t = 2 * j - 4           # query 位置（持有 flag f_j）
                is_miss_j = bool(view[vi] == P)
                is_miss_prev = j >= 4 and bool(view[vi - 2] == P)
                cat = 'lag0' if is_miss_j else ('lag1' if is_miss_prev else 'clean')
                ok = bool(logits[t].argmax().item() == clean[vi].item())

                mass = torch.zeros(n_layers, n_head, len(groups))
                for li, att in enumerate(att_layers):
                    # att: (n_head, T, T) on cpu
                    row = att[:, t, :t + 1].float()  # (n_head, t+1)
                    def put(gi, idxs):
                        if idxs:
                            mass[li, :, gi] += row[:, idxs].sum(dim=-1)
                    put(0, [2 * j - 5] if j >= 4 else [])          # x_{j-1}
                    put(1, [2 * j - 7] if j >= 5 else [])          # x_{j-2}
                    put(2, [2 * j - 9] if j >= 6 else [])          # x_{j-3}
                    put(3, [0, 1] if t >= 1 else [])               # x1, x2
                    holes = [q for q in range(t + 1) if view[q].item() == P]
                    put(4, holes)
                    flags = [q for q in range(2, t) if q % 2 == 0]  # f_3..f_{j-1}
                    put(5, flags)
                    vals = [q for q in range(3, t + 1) if q % 2 == 1
                            and q not in (2 * j - 5, 2 * j - 7, 2 * j - 9)]
                    put(6, vals)
                    put(7, [t])                                     # self (f_j)

                stats[cat]['n'] += 1
                stats[cat]['ok'] += ok
                stats[cat]['mass'] += mass

    print(f"\n=== 分类统计（样本内 value 位置数）===")
    for c in cats:
        s = stats[c]
        print(f"{c:>6}: n={s['n']}, acc={s['ok'] / s['n'] * 100:.2f}%")

    for li in range(n_layers):
        print(f"\n=== Layer {li} 各 head 的 attention 质量分布（按类别）===")
        print(f"{'cat':>6} {'head':>4} " + ' '.join(f"{g:>10}" for g in groups))
        for c in cats:
            m = stats[c]['mass'][li] / stats[c]['n']  # (n_head, groups)
            for h in range(m.shape[0]):
                row = ' '.join(f"{m[h, gi].item() * 100:>9.1f}%" for gi in range(len(groups)))
                print(f"{c:>6} h{h:<3} {row}")
            print()


if __name__ == '__main__':
    main()
