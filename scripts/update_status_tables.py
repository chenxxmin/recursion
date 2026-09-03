#!/usr/bin/env python3
"""Regenerate the result tables in reports/ACTION_MISSLEN_STATUS.md.

Scans all fp32-valid log dirs under /data/cxm/recursion, and rewrites the
'## 结果总表' section (A line by N, B line len2-N2). Format: bare acc = run
completed; 'acc @ epoch' = in progress (cumulative); 未开始 = not scheduled.

Usage: python scripts/update_status_tables.py
"""
import os
import re
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = '/data/cxm/recursion'


def collect(dirs, key_fn, skip=()):
    data = {}
    for D in dirs:
        if D.split('/')[-2] in skip or not os.path.isdir(D):
            continue
        for f in os.listdir(D):
            if not f.endswith('.log'):
                continue
            key = key_fn(f)
            if key is None:
                continue
            txt = open(os.path.join(D, f), errors='replace').read()
            curb = re.findall(r'Best=([\d.]+)%', txt)
            eps = re.findall(r'^Epoch (\d+) ', txt, re.M)
            if not curb:
                continue
            done = 'Best test accuracy' in txt
            best, ep = float(curb[-1]), int(eps[-1])
            if key not in data or ep >= data[key][1]:
                data[key] = (best, ep, done)
    return data


def cell(v):
    if v is None:
        return '未开始'
    best, ep, done = v
    return f"{best:.1f}" if done else f"{best:.1f} @ {ep}"


HDR = "| 模型 | 0.1 · s17996 | 0.1 · s18318 | 0.3 · s17996 | 0.3 · s18318 |\n|---|---|---|---|---|"


def table(data, sizes, n=None):
    lines = []
    for d, l in sizes:
        cells = []
        for prob in ('0.1', '0.3'):
            for seed in (17996, 18318):
                k = (d, l, prob, seed) if n is None else (d, l, n, prob, seed)
                cells.append(cell(data.get(k)))
        lines.append(f"| d{d}l{l} | " + " | ".join(cells) + " |")
    return lines


def main():
    # B 线：len2 N2（bf16 摧毁轮目录跳过）
    b_dirs = [f'{DATA}/action_p127_tr64_ood128_misslen/logs',
              f'{DATA}/action_misslen2_n2_scaleup/logs',
              f'{DATA}/action_misslen2_n2_d2048_resume/logs',
              f'{DATA}/action_misslen2_n2_d2048_resume2/logs',
              f'{DATA}/chain_b_d2048l4_01/logs',
              f'{DATA}/chain_b_d2048l2/logs',
              f'{DATA}/chain_b_d2048l4/logs',
              f'{DATA}/chain_b_d1024l4/logs']

    def bkey(f):
        m = re.search(r'd(\d+)l(\d+)r4h4_.*_N2_randmiss(0\.\d)len2_seed(\d+)\.log', f)
        if not m or int(m.group(4)) not in (17996, 18318):
            return None
        return (int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)))

    b_data = collect(b_dirs, bkey, skip={'action_misslen2_n2_d1024l4_resume'})
    b_sizes = sorted({(d, l) for d, l, *_ in b_data})

    # A 线：len1，按 N 分组
    a_dirs = [f'{DATA}/action_p127_tr64_ood128_misslen/logs',
              f'{DATA}/action_misslen1_n345_baseline/logs',
              f'{DATA}/action_misslen1_n3_depth_fp32/logs',
              f'{DATA}/action_misslen1_n3_depth_fp32_l4r2/logs',
              f'{DATA}/action_misslen1_n3_depth_fp32_l6r2/logs',
              f'{DATA}/chain_a_d1024l2/logs',
              f'{DATA}/chain_a_d1024l4/logs',
              f'{DATA}/chain_a_d512l2_fp32/logs']

    def akey(f):
        m = re.search(r'd(\d+)l(\d+)r4h4_.*_(N\d+)_randmiss(0\.\d)len1_seed(\d+)\.log', f)
        if not m or int(m.group(5)) not in (17996, 18318):
            return None
        return (int(m.group(1)), int(m.group(2)), m.group(3), m.group(4), int(m.group(5)))

    a_data = collect(a_dirs, akey)

    now = datetime.now().strftime('%m-%d %H:%M')
    out = [f"## 结果总表（{now} 自动更新）",
           "",
           "裸数字 = 完整跑完（6000ep 或早停）；`acc @ epoch` = 跑过但未完（累计 best @ 累计 epoch）；未开始 = 未安排。仅 fp32/TF32 有效数据。",
           "", "### B 线：misslen=2，N2", "", HDR]
    out += table(b_data, b_sizes)
    for n in ('N2', 'N3', 'N4', 'N5'):
        keys = {k for k in a_data if k[2] == n}
        if not keys:
            continue
        sizes = sorted({(d, l) for d, l, *_ in keys})
        out += ["", f"### A 线：misslen=1，{n}", "", HDR]
        out += table(a_data, sizes, n=n)

    p = os.path.join(REPO, 'reports/ACTION_MISSLEN_STATUS.md')
    txt = open(p).read()
    txt = re.sub(r'\n## 结果总表（[^）]*）.*?(?=\n## [^#]|\Z)', '', txt, flags=re.S)
    txt = txt.replace('\n## 一、已有确定答案', '\n' + '\n'.join(out) + '\n\n## 一、已有确定答案')
    open(p, 'w').write(txt)
    print(f"tables updated in {p}")


if __name__ == '__main__':
    main()
