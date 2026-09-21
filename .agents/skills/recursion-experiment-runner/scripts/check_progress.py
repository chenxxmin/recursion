#!/usr/bin/env python3
"""Progress check for a batch of experiments.

Usage: python check_progress.py <batch-name> [--data-base DIR]

Prints per-experiment status: [完成]/[运行]/[存盘待续] + epoch + best acc.
batch-name is the config basename (e.g. action_v2_d1024_n2_miss01).
"""
import argparse
import os
import re
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('batch')
    ap.add_argument('--data-base', default='/mnt/workspace/hujiachen/recursion_results')
    args = ap.parse_args()
    D = os.path.join(args.data_base, args.batch, 'logs')
    if not os.path.isdir(D):
        print(f"批次 {args.batch} 无日志目录（未启动？）: {D}")
        return
    for f in sorted(os.listdir(D)):
        if not f.endswith('.log'):
            continue
        p = os.path.join(D, f)
        txt = open(p, errors='replace').read()
        eps = re.findall(r'^Epoch\s+(\d+)', txt, re.M)
        curb = re.findall(r'Best=([\d.]+)%', txt)
        done = 'Best test accuracy' in txt
        timeout = txt.rstrip().endswith(('---', 'ds')) or '[TIMEOUT]' in txt
        fresh = (datetime.now().timestamp() - os.path.getmtime(p)) < 900
        if done:
            st = '完成'
        elif fresh:
            st = '运行'
        elif '[TIMEOUT]' in txt or os.path.exists(os.path.join(
                '/mnt/workspace/hujiachen/models', args.batch, f[:-4] + '_resume.pth')):
            st = '存盘待续'
        else:
            st = '中断?'
        print(f"[{st}] {f[:-4]}: epoch {eps[-1] if eps else '-'}, best={curb[-1] if curb else '-'}%")


if __name__ == '__main__':
    main()
