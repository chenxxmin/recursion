#!/usr/bin/env python3
"""overnight_h123 600 实验按分类标准 1（特征根类型）画学习曲线。

布局：4 行 x 4 列（列 = double / root1 / rootm1 / other）。
  行1: train acc，每实验一根曲线
  行2: test  acc，每实验一根曲线
  行3: train 均值（深色）+ min~max 包络（浅色）
  行4: test  均值（深色）+ min~max 包络（浅色）
均值口径：早停实验的最终值向前填充（carry-forward）到列内最大 epoch。
输出：plots/overnight_h123_rootclass_learning_curves.{png,pdf}
"""
import glob
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = '/data/cxm/recursion/overnight_h123_p127_d512_600_nockpt/logs'
CLS_JSON = 'reports/data/overnight_h123_coef_classification.json'
OUT_BASE = 'plots/overnight_h123_rootclass_learning_curves'

CLASSES = ['double', 'root1', 'rootm1', 'other']
CLASS_TITLE = {
    'double': 'double root ($a^2+4b=0$)',
    'root1': 'root $\\lambda=1$ ($a+b=1$)',
    'rootm1': 'root $\\lambda=-1$ ($b=a+1$)',
    'other': 'other',
}
CLASS_COLOR = {'double': '#d62728', 'root1': '#1f77b4', 'rootm1': '#2ca02c', 'other': '#9467bd'}

EPOCH_RE = re.compile(r'Epoch\s+(\d+) \| Train: Loss=[\d.]+ Acc=([\d.]+)% \| Test: Acc=([\d.]+)%')
NAME_RE = re.compile(r'_a(\d+)b(\d+)_')


def main():
    cls = json.load(open(CLS_JSON))
    root_class = {(r['a'], r['b']): r['root_class'] for r in cls['records']}

    # curves[cls] = list of (epochs_array, train_acc_array, test_acc_array)
    curves = {c: [] for c in CLASSES}
    for f in sorted(glob.glob(f'{LOG_DIR}/*.log')):
        m = NAME_RE.search(f)
        assert m, f
        rc = root_class[(int(m.group(1)), int(m.group(2)))]
        rows = EPOCH_RE.findall(open(f).read())
        assert rows, f
        ep = np.array([int(r[0]) for r in rows])
        tr = np.array([float(r[1]) for r in rows])
        te = np.array([float(r[2]) for r in rows])
        curves[rc].append((ep, tr, te))

    fig, axes = plt.subplots(4, 4, figsize=(22, 15), sharex=True)
    row_labels = ['train acc (per run)', 'test acc (per run)',
                  'train acc (mean + min/max)', 'test acc (mean + min/max)']

    for col, c in enumerate(CLASSES):
        color = CLASS_COLOR[c]
        runs = curves[c]
        max_ep = max(int(ep[-1]) for ep, _, _ in runs)
        grid = np.arange(max_ep + 1)

        # carry-forward 填充到统一网格
        tr_full = np.empty((len(runs), max_ep + 1))
        te_full = np.empty((len(runs), max_ep + 1))
        for i, (ep, tr, te) in enumerate(runs):
            idx = np.clip(grid, 0, int(ep[-1]))
            # epoch 从 0 连续（EVAL_INTERVAL=1），直接索引
            tr_full[i] = tr[idx]
            te_full[i] = te[idx]

        for row, (mat, raw) in enumerate([(tr_full, 'tr'), (te_full, 'te'),
                                          (tr_full, 'tr'), (te_full, 'te')]):
            ax = axes[row][col]
            if row < 2:  # 每实验一根曲线
                for ep, tr, te in runs:
                    ax.plot(ep, tr if raw == 'tr' else te, color=color, alpha=0.18, lw=0.8)
            else:  # 均值 + min/max 包络
                mean = mat.mean(axis=0)
                lo = mat.min(axis=0)
                hi = mat.max(axis=0)
                ax.fill_between(grid, lo, hi, color=color, alpha=0.2, lw=0)
                ax.plot(grid, mean, color=color, lw=2)
            ax.set_ylim(-2, 102)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f'{CLASS_TITLE[c]}  (n={len(runs)})', fontsize=12)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11)
            if row == 3:
                ax.set_xlabel('epoch')

    fig.suptitle('overnight_h123_p127_d512_600: learning curves by root type '
                 '(150 coefficient pairs x 4 seeds, all runs converged to >=99%)', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_BASE + '.png', dpi=150)
    fig.savefig(OUT_BASE + '.pdf')
    print('saved:', OUT_BASE + '.png', OUT_BASE + '.pdf')

    # 终端速览：各类收敛 epoch（test acc 首次 >=99%）统计
    print('\nfirst epoch with test acc >= 99%:')
    for c in CLASSES:
        conv = []
        for ep, _, te in curves[c]:
            hit = np.nonzero(te >= 99)[0]
            conv.append(int(ep[hit[0]]) if len(hit) else None)
        conv = [x for x in conv if x is not None]
        print(f'  {c:7s}: median={np.median(conv):6.1f}  mean={np.mean(conv):6.1f}  '
              f'min={min(conv):3d}  max={max(conv):3d}  n={len(conv)}')


if __name__ == '__main__':
    main()
