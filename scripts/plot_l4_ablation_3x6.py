"""3x6 grid: rows = l4 baseline / -dynmask / -allpos, cols = 6 coef groups.
Each subplot: 3 test accuracy curves over epochs (body / pos head / val head).
"""
import glob, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r'Epoch\s+(\d+) \| Train:.*Test: Acc=([\d.]+)%')
X2_RE = re.compile(r'from x2: (.*)')
PP_RE = re.compile(r'PosProbe (pos|val)\| (.*)')
KV_RE = re.compile(r'k(\d+)=([\d.]+)')

ROWS = [
    # (label, dirname, suffix, allpos_on)  allpos_on -> x 轴乘以每窗口展开倍数
    # （一个 allpos epoch = 每个窗口在全部 ~7.47 个位置各出现一次 = 7.47 个 allpos-off epoch 的呈现量）
    ('l4 + dynmask + allpos (baseline)', 'variant_o2_corruptfix_allpos_dynmask_d512l4_P127_len5to10', 'allpos_dynmask', True),
    ('l4 + allpos (dynmask OFF)', 'variant_o2_corruptfix_allpos_d512l4_P127_len5to10', 'allpos', True),
    ('l4 + dynmask (allpos OFF)', 'variant_o2_corruptfix_dynmask_d512l4_P127_len5to10', 'dynmask', False),
]
# 84307 (allpos 展开) / 11290 (原始窗数) = 平均窗口长 ≈ 7.47
ALLPOS_FACTOR = 84307 / 11290
COEFS = [(25,56),(62,59),(56,93),(5,3),(67,65),(44,59)]


def parse(path):
    """-> (epochs, body%, pos%, val%) series aligned by eval point."""
    epochs, test, body, pos, val = [], {}, {}, {}, {}
    cur = None
    for line in open(path):
        m = EPOCH_RE.search(line)
        if m:
            cur = int(m.group(1))
            epochs.append(cur)
            continue
        if cur is None:
            continue
        m = X2_RE.search(line)
        if m:
            vals = [float(x) for x in m.group(1).split()]
            body[cur] = 100.0 * sum(vals) / len(vals)
            continue
        m = PP_RE.search(line)
        if m:
            kind, rest = m.group(1), m.group(2)
            vals = [float(v) for _, v in KV_RE.findall(rest)]
            (pos if kind == 'pos' else val)[cur] = sum(vals) / len(vals)
    n = len(epochs)
    def get(d):
        return [d.get(e, float('nan')) for e in epochs]
    return epochs, get(body), get(pos), get(val)


fig, axes = plt.subplots(3, 6, figsize=(22, 9), sharex=True, sharey=True)
for r, (rowlabel, dirname, suffix, allpos_on) in enumerate(ROWS):
    for c, (a, b) in enumerate(COEFS):
        ax = axes[r][c]
        pat = f'/data/cxm/recursion/{dirname}/logs/*a{a}b{b}_d512l4r8h4_P127_len5to10_{suffix}_seed42.log'
        files = glob.glob(pat)
        if not files:
            ax.set_title(f'a{a}b{b} MISSING', fontsize=9)
            continue
        ep, body, pos, val = parse(files[0])
        if allpos_on:
            ep = [e * ALLPOS_FACTOR for e in ep]
        ax.plot(ep, body, color='tab:blue', lw=1.2, label='body')
        ax.plot(ep, pos, color='tab:orange', lw=1.2, label='pos')
        ax.plot(ep, val, color='tab:green', lw=1.2, label='val')
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.3)
        if r == 0:
            ax.set_title(f'[{a},{b}]', fontsize=11)
        if c == 0:
            ax.set_ylabel(rowlabel, fontsize=9)
        if r == 2:
            ax.set_xlabel('effective epochs (sample presentations / 11290)')
axes[0][5].legend(loc='lower right', fontsize=9)
fig.suptitle('l4 corrupt_fix ablation: body / pos / val test accuracy (seed42, P127, len5to10);\n'
             'x-axis = sample presentations / 11290 (allpos rows x 7.47)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = 'plots/variant_o2_l4_ablation_body_pos_val_3x6.png'
fig.savefig(out, dpi=150)
print('saved', out)
