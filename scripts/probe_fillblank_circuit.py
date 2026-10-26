#!/usr/bin/env python3
"""画 L=5、k=2 的四层电路图：节点 = x0,x1,BLANK,x3,x4,EOS，
边 = attention 信息流向（key -> query），只画均值权重 >= 阈值的边。
"""
import argparse, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import torch

ROOT = '/home/cxm/recursion'
sys.path.insert(0, os.path.join(ROOT, 'src'))
from probe_fillblank_attention import AttentionCapture
from probe_fillblank_operands import load_model, test_windows, P, BLANK, EOS


@torch.no_grad()
def full_attention(path, a, b, device, L, k, n=64):
    model, _ = load_model(path, device)
    cap = AttentionCapture(model)
    wins = test_windows(a, b)
    seqs = []
    for w in wins[:n]:
        seq = w[:L].clone()
        seq[k] = BLANK
        seqs.append(torch.cat([seq, torch.tensor([EOS])]))
    x = torch.stack(seqs).to(device)
    cap.records = []
    model(x)
    cap.close()
    # 每层 (T,T)：query i <- key j，按 batch 和 head 求平均
    mats = []
    for att, vnorm in cap.records:
        mats.append(att.mean(dim=(0, 1)).numpy())
    return mats


def draw(mats, a, b, L, k, out, threshold=0.15):
    labels = [f'x{i}' if i != k else 'BLANK' for i in range(L)] + ['EOS']
    colors = ['#4c78a8' if i != k else '#e45756' for i in range(L)] + ['#54a24b']
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    for layer, ax in enumerate(axes):
        mat = mats[layer]
        for i, lab in enumerate(labels):
            ax.scatter([i], [0], s=1300, color=colors[i], zorder=3,
                       edgecolor='white', linewidth=1.5)
            ax.text(i, 0, lab, ha='center', va='center', fontsize=10,
                    color='white', weight='bold', zorder=4)
        for q in range(L + 1):
            edges = [(j, mat[q, j]) for j in range(6)
                     if mat[q, j] >= threshold and q != j]
            for ei, (j, w) in enumerate(edges):
                # 信息流向：key j -> query q，用二次贝塞尔弧，按距离给高度
                dist = abs(q - j)
                apex = 0.35 + 0.28 * dist + 0.12 * ei
                mx = (j + q) / 2
                # 控制点让弧线从源节点顶部起到目标节点顶部
                cx = mx
                cy = 0.2 + 2 * (apex - 0.2)
                ts = [i / 40 for i in range(41)]
                xs = [(1 - t) ** 2 * j + 2 * (1 - t) * t * cx + t ** 2 * q for t in ts]
                ys = [(1 - t) ** 2 * 0.2 + 2 * (1 - t) * t * cy + t ** 2 * 0.2 for t in ts]
                ax.plot(xs, ys, color='#333333', alpha=min(1.0, 0.35 + w),
                        linewidth=1.0 + 3 * w, zorder=2)
                ax.annotate('', xy=(q, 0.22), xytext=(xs[-2], ys[-2]),
                            arrowprops=dict(arrowstyle='-|>', color='#333333',
                                            lw=1.0 + 3 * w,
                                            alpha=min(1.0, 0.35 + w)))
                ax.text(mx, apex + 0.03, f'{w:.2f}', fontsize=8.5,
                        ha='center', color='#111111')
        ax.set_xlim(-0.6, 5.6)
        ax.set_ylim(-0.35, 2.6)
        ax.set_title(f'Layer {layer}', fontsize=13, pad=4)
        ax.axis('off')
    fig.suptitle(f'fill_blank circuit: a{a}b{b}, L={L}, BLANK@k={k} '
                 f'(attention mean over 64 windows x 4 heads, edges >= {threshold})',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    print('saved', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--L', type=int, default=5)
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--device', default='cuda:7')
    ap.add_argument('--n-windows', type=int, default=64)
    ap.add_argument('--threshold', type=float, default=0.15)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    import re
    m = re.search(r'a(\d+)b(\d+)', os.path.basename(args.ckpt))
    a, b = int(m.group(1)), int(m.group(2))
    out = args.out or f'plots/fillblank_circuit_L{args.L}_k{args.k}.png'
    mats = full_attention(args.ckpt, a, b, args.device, L=args.L, k=args.k, n=args.n_windows)
    for layer, mat in enumerate(mats):
        print(f'--- layer {layer} (rows=query, cols=key) ---')
        for q in range(6):
            print('  ', ' '.join(f'{mat[q, j]:.2f}' for j in range(6)))
    draw(mats, a, b, args.L, args.k, out, args.threshold)


if __name__ == '__main__':
    main()
