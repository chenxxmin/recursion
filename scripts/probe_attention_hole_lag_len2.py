"""Attention probe for misslen=2: four hole-context patterns.

For every predicted value x_j on the test set:
  hole1  : x_j 是洞且 x_{j-1} 干净（独立单空 或 连二空的第一空）
  hole2  : x_j 是洞且 x_{j-1} 也是洞（连二空的第二空）
  lag1   : x_j 干净但 x_{j-1} 是洞（洞后一位，两跳推理）
  clean  : x_j 与 x_{j-1} 都干净

Usage: CUDA_VISIBLE_DEVICES=<gpu> python scripts/probe_attention_hole_lag_len2.py [pth]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch

from analyze_attention import get_attention_weights, load_model

DEFAULT_PTH = ('/data/cxm/models/action_p127_tr64_ood128_misslen/'
               'action_d1024l2r4h4_P127_tr64ood128_N2_randmiss0.1len2_seed17996.pth')
N_SAMPLES = 500


def main():
    pth = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PTH
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model(pth, device=device)
    P = checkpoint['config']['p']

    from datasets import ActionDataset
    test_ds = ActionDataset(p=P, ab_pairs=[[1, 1], [2, 3]], num_samples=N_SAMPLES,
                            length=128, seed=17996 + 1, missing_prob=0.1, miss_len=2)
    print(f"model: {os.path.basename(pth)} (best={checkpoint.get('best_accuracy'):.4f})")
    print(f"test samples: {len(test_ds)}, p={P}")

    groups = ['x_{j-1}', 'x_{j-2}', 'x_{j-3}', 'x1x2', '洞位', 'flags', '其他value', 'self']
    cats = ('hole1', 'hole2', 'lag1', 'clean')
    stats = {c: {'n': 0, 'ok': 0, 'mass': None} for c in cats}
    n_layers = None

    with torch.no_grad():
        for idx in range(len(test_ds)):
            view, clean, _ = test_ds[idx]
            L = test_ds.length
            att_layers = get_attention_weights(model, view.unsqueeze(0).to(device))
            logits = model(view.unsqueeze(0).to(device))[0][0]
            if n_layers is None:
                n_layers = len(att_layers)
                n_head = att_layers[0].shape[0]
                for c in cats:
                    stats[c]['mass'] = torch.zeros(n_layers, n_head, len(groups))

            for j in range(3, L + 1):
                vi = 2 * j - 3
                t = 2 * j - 4
                miss_j = bool(view[vi] == P)
                miss_prev = j >= 4 and bool(view[vi - 2] == P)
                if miss_j and not miss_prev:
                    cat = 'hole1'
                elif miss_j and miss_prev:
                    cat = 'hole2'
                elif miss_prev:
                    cat = 'lag1'
                else:
                    cat = 'clean'
                ok = bool(logits[t].argmax().item() == clean[vi].item())

                mass = torch.zeros(n_layers, att_layers[0].shape[0], len(groups))
                for li, att in enumerate(att_layers):
                    row = att[:, t, :t + 1].float()
                    def put(gi, idxs, row=row, mass=mass, li=li):
                        if idxs:
                            mass[li, :, gi] += row[:, idxs].sum(dim=-1)
                    put(0, [2 * j - 5] if j >= 4 else [])
                    put(1, [2 * j - 7] if j >= 5 else [])
                    put(2, [2 * j - 9] if j >= 6 else [])
                    put(3, [0, 1])
                    put(4, [q for q in range(t + 1) if view[q].item() == P])
                    put(5, [q for q in range(2, t) if q % 2 == 0])
                    put(6, [q for q in range(3, t + 1) if q % 2 == 1
                            and q not in (2 * j - 5, 2 * j - 7, 2 * j - 9)])
                    put(7, [t])

                stats[cat]['n'] += 1
                stats[cat]['ok'] += ok
                stats[cat]['mass'] += mass

    print(f"\n=== 四类正确率 ===")
    for c in cats:
        s = stats[c]
        print(f"{c:>6}: n={s['n']}, acc={s['ok'] / s['n'] * 100:.2f}%")

    for li in range(n_layers):
        print(f"\n### Layer {li}")
        for c in cats:
            print(f"**{c}**")
            m = stats[c]['mass'][li] / stats[c]['n']
            for h in range(m.shape[0]):
                parts = [f"{g}({m[h, gi].item() * 100:.1f}%)" for gi, g in enumerate(groups)
                         if m[h, gi].item() * 100 > 5]
                print(f"- h{h}：" + " + ".join(parts))
            print()


if __name__ == '__main__':
    main()
