#!/usr/bin/env python3
"""Plot curriculum-stage epochs solely from the standalone JSON data file."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = 'reports/data/length_curriculum_epochs.json'
COEFS = ['a25b56', 'a62b59', 'a56b93', 'a5b3', 'a67b65', 'a44b59']
COLORS = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2', '#b279a2']


def make_fig(records, task, out):
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True, sharey=True)
    for idx, (ax, coef) in enumerate(zip(axes.flat, COEFS)):
        selected = [r for r in records
                    if r['task'] == task and r['coef'] == coef
                    and 17 <= r['length'] <= 64 and r['status'] == 'success']
        xs = [r['length'] for r in selected]
        ys = [r['epochs'] for r in selected]
        ax.bar(xs, ys, color=COLORS[idx], width=0.85)
        ax.set_yscale('log')
        ax.set_ylim(0.7, 4000)
        ax.set_xlim(16.5, 64.5)
        ax.set_title(coef, fontsize=12)
        ax.grid(alpha=0.3, axis='y')
        if not selected:
            ax.text(0.5, 0.5, 'No valid curriculum data', ha='center', va='center',
                    transform=ax.transAxes, color='#666666')
        if idx % 3 == 0:
            ax.set_ylabel('epochs to converge (log)')
        if idx >= 3:
            ax.set_xlabel('window length')
    title = 'corrupt_fix value_only' if task == 'corrupt_fix' else 'fill_blank'
    fig.suptitle(f'{title}: epochs per curriculum stage (len17–64; failed/missing stages omitted)',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('saved', out)


def main():
    with open(DATA, encoding='utf-8') as f:
        records = json.load(f)['records']
    make_fig(records, 'corrupt_fix', 'plots/corruptfix_len17to64_epochs.png')
    make_fig(records, 'fill_blank', 'plots/fillblank_len17to64_epochs.png')


if __name__ == '__main__':
    main()
