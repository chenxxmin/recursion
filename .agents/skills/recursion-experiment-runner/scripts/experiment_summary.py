#!/usr/bin/env python3
"""Summarize all experiments: epochs trained (cumulative across resume
rounds), best test acc, seconds per epoch, status (RUN/DONE/STOP).

Status: RUN = trainer process alive and log fresh; DONE = finished normally
(early stop / full epochs, 'Best test accuracy' in log); STOP = timed out /
manually stopped (resume checkpoint available).

Usage: python experiment_summary.py [--since-days N] [--match SUBSTR]
  --since-days N   only include batches whose logs changed within N days (default 3)
  --match SUBSTR   only include experiments whose name contains SUBSTR
"""
import argparse
import glob
import os
import re
import subprocess
import time
from datetime import datetime

BASES = ['/data/cxm/recursion', '/data/cxm/recursion_v2']


def running_names():
    out = subprocess.run(['pgrep', '-af', 'core.py config_tmp'],
                         capture_output=True, text=True).stdout
    return {m.group(1) for m in
            re.finditer(r'config_tmp_([A-Za-z0-9_.]+)\.json', out)}


def parse_log(path):
    txt = open(path, errors='replace').read()
    times = [datetime.fromisoformat(m.group(1)).timestamp() for m in
             re.finditer(r'Time: (\d{4}-\d{2}-\d{2}T[\d:\.]+)', txt)]
    seg_starts = [m.start() for m in re.finditer(r'Time: \d{4}-\d{2}-\d{2}T', txt)]
    seg_starts.append(len(txt))
    mtime = os.path.getmtime(path)
    total_eps, total_sec = 0, 0.0
    for i in range(len(times)):
        eps = [int(m.group(1)) for m in
               re.finditer(r'^Epoch\s+(\d+)', txt[seg_starts[i]:seg_starts[i + 1]], re.M)]
        if not eps:
            continue
        total_eps += max(eps) - min(eps) + 1
        t1 = times[i + 1] if i + 1 < len(times) else mtime
        total_sec += max(t1 - times[i], 1)
    bests = [float(m.group(1)) for m in re.finditer(r'Best=\s*([\d.]+)%', txt)]
    return total_eps, (max(bests) if bests else 0.0), total_sec / max(total_eps, 1), \
        'Best test accuracy' in txt, mtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since-days', type=float, default=3)
    ap.add_argument('--match', default='')
    args = ap.parse_args()

    running = running_names()
    now = time.time()
    rows = []
    for base in BASES:
        for log in glob.glob(f'{base}/*/logs/*.log'):
            if now - os.path.getmtime(log) > args.since_days * 86400:
                continue
            name = os.path.basename(log)[:-4]
            if args.match and args.match not in name:
                continue
            eps, best, spe, done, mtime = parse_log(log)
            if eps == 0:
                continue
            if name in running and now - mtime < 300:
                status = 'RUN'
            elif done:
                status = 'DONE'
            else:
                status = 'STOP'
            stem = log.split('/')[-3]
            rows.append((stem, name, status, eps, best, spe))

    rows.sort(key=lambda r: (r[0], r[1]))
    cur = None
    for stem, name, status, eps, best, spe in rows:
        if stem != cur:
            print(f'\n### {stem}')
            cur = stem
        short = name
        for pat in ('mixed_basic_', 'addition_', 'tribonacci_', 'tetranacci_', 'action_v2_',
                    '_d1024', '_P127', '_tr16ood32', '_e0.7'):
            short = short.replace(pat, '')
        print(f'{short:60s} {status:4s} ep={eps:5d} best={best:6.2f}% {spe:6.2f}s/ep')


if __name__ == '__main__':
    main()
