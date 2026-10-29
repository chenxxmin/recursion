#!/usr/bin/env python3
"""5 组填空实验的 test acc 曲线（每组 6 系数），合为一张 PNG 并上传用。
子图：len5~10失败 / len3~5 wd1 / len3~5 wd2 / len5~8失败 / len5~8暖启动。
"""
import glob, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r'Epoch\s+(\d+) \| Train:.*Test: Acc=([\d.]+)%')
COEFS = ['a25b56', 'a62b59', 'a56b93', 'a5b3', 'a67b65', 'a44b59']
COLORS = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2', '#b279a2']

GROUPS = [
    ('len5~10 noaux (failed)', 'variant_o2_fillblank_noaux_d512l4_P127_len5to10'),
    ('len3~5 noaux wd=1', 'variant_o2_fillblank_noaux_randpos_d512l4_P127_len3to5'),
    ('len3~5 noaux wd=2', 'tune_len3to5_l4_d512l4_lr0.0003_wd2_cap3000'),
    ('len5~8 noaux wd=2 (failed)', 'variant_o2_fillblank_noaux_randpos_d512l4_P127_len5to8_wd2'),
    ('len5~8 warm-start from len3~5',
     ['variant_o2_fillblank_noaux_randpos_d512l4_P127_len8_wd2_initLen35',
      'variant_o2_fillblank_noaux_randpos_d512l4_P127_len8_wd2_initLen35_rest']),
]


def series(logdir, coef):
    dirs = [logdir] if not isinstance(logdir, (list, tuple)) else logdir
    eps, accs = [], []
    for d in dirs:
        for f in glob.glob(f'/data/cxm/recursion/{d}/logs/*{coef}_*.log'):
            for line in open(f):
                m = EPOCH_RE.search(line)
                if m:
                    eps.append(int(m.group(1)))
                    accs.append(float(m.group(2)))
    return eps, accs


fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharex=False, sharey=True)
axes = axes.flatten()
for i, (title, logdir) in enumerate(GROUPS):
    ax = axes[i]
    for coef, color in zip(COEFS, COLORS):
        eps, accs = series(logdir, coef)
        ax.plot(eps, accs, color=color, lw=1.1, marker='.', ms=3, label=coef)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.set_ylabel('test acc (%)')
    ax.set_xlabel('epoch')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', fontsize=10, title='coefficients')
fig.suptitle('fill_blank: test accuracy by coefficient (6 groups)', fontsize=14)
fig.tight_layout(rect=(0, 0, 0.93, 0.96))
out = 'plots/fillblank_5groups_testacc_v2.png'
fig.savefig(out, dpi=150)
print('saved', out)
