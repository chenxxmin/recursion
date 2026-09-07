#!/usr/bin/env python3
"""Regenerate the v2 full-matrix table in reports/v2/ACTION_V2_STATUS.md.

Scans /data/cxm/recursion_v2/*/logs and rewrites the '## 全矩阵结果表' section.
Format: bare acc = completed; 'acc @ epoch' = in progress / interrupted at
checkpoint (cumulative); 未开始 = not scheduled.

Usage: python scripts/update_v2_status.py
"""
import os
import re
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = '/data/cxm/recursion_v2'


def collect():
    data = {}
    for b in os.listdir(DATA):
        D = os.path.join(DATA, b, 'logs')
        if not os.path.isdir(D):
            continue
        for f in os.listdir(D):
            if not f.endswith('.log'):
                continue
            m = re.search(r'action_v2_d1024(l\d)r4h4_P127_tr127_(N\d)_miss(0\.\d)len1_seed(\d+)\.log', f)
            if not m:
                continue
            txt = open(os.path.join(D, f), errors='replace').read()
            curb = re.findall(r'Best=([\d.]+)%', txt)
            eps = re.findall(r'^Epoch\s+(\d+)', txt, re.M)
            if not curb:
                continue
            key = (m.group(2), m.group(3), m.group(1), int(m.group(4)))
            best, ep = float(curb[-1]), int(eps[-1]) if eps else 0
            done = 'Best test accuracy' in txt
            if key not in data or ep >= data[key][1]:
                data[key] = (best, ep, done)
    return data


def cell(data, n, prob, l, seed):
    v = data.get((n, prob, l, seed))
    if v is None:
        return '未开始'
    best, ep, done = v
    return f"{best:.1f}" if done else f"{best:.1f} @ {ep}"


def main():
    data = collect()
    lines = ["| 组 | 模型 | seed17996 | seed18318 |", "|---|---|---|---|"]
    for n in ('N2', 'N3', 'N4'):
        for prob in ('0.1', '0.3'):
            for l in ('l1', 'l2', 'l4'):
                lines.append(f"| {n} × {prob} | {l} | {cell(data, n, prob, l, 17996)} | {cell(data, n, prob, l, 18318)} |")
    now = datetime.now().strftime('%m-%d %H:%M')
    sec = (f"## 全矩阵结果表（{now} 自动更新）\n\n"
           "裸数字 = 已完成；`acc @ epoch` = 在跑/中断续中（累计 best @ 累计 epoch）；未开始 = 未排上。\n"
           "注：N2×0.1 的 l2/l4 四格是\"破 99 + 超 20ep 缓冲后政策关停\"，结果按关停时 best 记录，等同完成。\n\n"
           + '\n'.join(lines))

    p = os.path.join(REPO, 'reports/v2/ACTION_V2_STATUS.md')
    txt = open(p).read()
    txt = re.sub(r'\n## 全矩阵结果表（[^）]*）.*?(?=\n## |\Z)', '', txt, flags=re.S)
    txt = txt.replace('\n## 明细', '\n' + sec + '\n\n## 明细')
    open(p, 'w').write(txt)
    print(f"v2 matrix table updated ({now})")


if __name__ == '__main__':
    main()
