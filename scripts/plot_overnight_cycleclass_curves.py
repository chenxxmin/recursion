#!/usr/bin/env python3
"""overnight_h123 600 实验按分类标准 2（循环轨道结构，15 类）画学习曲线。

布局：4 行 x 15 列（列 = 循环类，按均值收敛 epoch 从快到慢排序）。
  行1: train acc，每实验一根曲线
  行2: test  acc，每实验一根曲线
  行3: train 均值 + min~max 包络
  行4: test  均值 + min~max 包络
列颜色按所属 root class（double/root1/rootm1/other）着色，与分类 1 图可比。
输出：plots/overnight_h123_cycleclass_learning_curves.{png,pdf}
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
OUT_BASE = 'plots/overnight_h123_cycleclass_learning_curves'

ROOT_COLOR = {'double': '#d62728', 'root1': '#1f77b4', 'rootm1': '#2ca02c', 'other': '#9467bd'}

EPOCH_RE = re.compile(r'Epoch\s+(\d+) \| Train: Loss=[\d.]+ Acc=([\d.]+)% \| Test: Acc=([\d.]+)%')
NAME_RE = re.compile(r'_a(\d+)b(\d+)_')


def main():
    cls = json.load(open(CLS_JSON))
    info = {(r['a'], r['b']): (r['cycle_class'], r['cycle_period'], r['root_class'])
            for r in cls['records']}

    curves = {}
    for f in sorted(glob.glob(f'{LOG_DIR}/*.log')):
        m = NAME_RE.search(f)
        cc, period, rc = info[(int(m.group(1)), int(m.group(2)))]
        rows = EPOCH_RE.findall(open(f).read())
        ep = np.array([int(r[0]) for r in rows])
        tr = np.array([float(r[1]) for r in rows])
        te = np.array([float(r[2]) for r in rows])
        curves.setdefault(cc, {'period': period, 'root_class': rc, 'runs': []})
        curves[cc]['runs'].append((ep, tr, te))

    # 按均值收敛 epoch（test 首次 >=99%）从快到慢排序
    def mean_conv(d):
        ts = []
        for ep, _, te in d['runs']:
            hit = np.nonzero(te >= 99)[0]
            ts.append(int(ep[hit[0]]) if len(hit) else int(ep[-1]))
        return np.mean(ts)

    classes = sorted(curves, key=lambda c: mean_conv(curves[c]))

    fig, axes = plt.subplots(4, len(classes), figsize=(3.0 * len(classes), 15), sharex=True)
    row_labels = ['train acc (per run)', 'test acc (per run)',
                  'train acc (mean + min/max)', 'test acc (mean + min/max)']

    for col, c in enumerate(classes):
        d = curves[c]
        color = ROOT_COLOR[d['root_class']]
        runs = d['runs']
        max_ep = max(int(ep[-1]) for ep, _, _ in runs)
        grid = np.arange(max_ep + 1)
        tr_full = np.empty((len(runs), max_ep + 1))
        te_full = np.empty((len(runs), max_ep + 1))
        for i, (ep, tr, te) in enumerate(runs):
            idx = np.clip(grid, 0, int(ep[-1]))
            tr_full[i] = tr[idx]
            te_full[i] = te[idx]

        for row, mat in enumerate([tr_full, te_full, tr_full, te_full]):
            ax = axes[row][col]
            if row < 2:
                for ep, tr, te in runs:
                    ax.plot(ep, tr if row == 0 else te, color=color, alpha=0.25, lw=0.8)
            else:
                ax.fill_between(grid, mat.min(axis=0), mat.max(axis=0), color=color, alpha=0.2, lw=0)
                ax.plot(grid, mat.mean(axis=0), color=color, lw=1.8)
            ax.set_ylim(-2, 102)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f'{c}\nT={d["period"]}, n={len(runs)}', fontsize=9)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11)
            if row == 3:
                ax.set_xlabel('epoch', fontsize=9)

    handles = [plt.Line2D([], [], color=v, lw=2, label=k) for k, v in ROOT_COLOR.items()]
    fig.legend(handles=handles, loc='upper right', fontsize=10, title='root class')
    fig.suptitle('overnight_h123_p127_d512_600: learning curves by cycle structure '
                 '(sorted by mean convergence epoch, fastest first)', fontsize=15, x=0.45)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_BASE + '.png', dpi=110)
    fig.savefig(OUT_BASE + '.pdf')
    print('saved:', OUT_BASE + '.png', OUT_BASE + '.pdf')
    print('column order:', ', '.join(classes))


if __name__ == '__main__':
    main()
