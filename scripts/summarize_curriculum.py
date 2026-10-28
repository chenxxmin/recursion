#!/usr/bin/env python3
"""汇总各课程路线逐系数总 epoch（len5 全位置 / len8 全位置 的达成代价）。"""
import glob, json, os, re

ROOT = '/home/cxm/recursion'
COEFS = ['a25b56', 'a62b59', 'a56b93', 'a5b3', 'a67b65', 'a44b59']

CHAINS = {
    'from_scratch_len5': ['variant_o2_corruptfix_valueonly_d512l4_P127_len5_randpos'],
    'fixlast_init': ['variant_o2_corruptfix_valueonly_d512l4_P127_len5_initFixlast'],
    'last2_init': ['variant_o2_corruptfix_valueonly_d512l4_P127_len5_initLast2'],
    'pos04_init': ['variant_o2_corruptfix_valueonly_d512l4_P127_len5_initPos04'],
    'pos024_init': ['variant_o2_corruptfix_valueonly_d512l4_P127_len5_initPos024'],
    'pos_progressive_len5': [f'variant_o2_corruptfix_valueonly_d512l4_P127_len5_pos{i}'
                             for i in range(4)] + [
        'variant_o2_corruptfix_valueonly_d512l4_P127_len5_pos01234'],
    'len_progressive': [f'variant_o2_corruptfix_valueonly_d512l4_P127_len{L}_initLen{L-1}'
                        for L in (6, 7, 8)],
    'pos_progressive_len8': [f'variant_o2_corruptfix_valueonly_d512l4_P127_len8_pos{i}'
                             for i in range(7)] + [
        'variant_o2_corruptfix_valueonly_d512l4_P127_len8_pos01234567'],
}


def stage_stats(logdir):
    """返回 {coef: (epochs, best_acc, converged)}；未跑完/缺失返回 None。"""
    out = {}
    for f in glob.glob(f'{logdir}/logs/*.log'):
        ab = re.search(r'_(a\d+b\d+)_', f).group(1)
        txt = open(f).read()
        eps = re.findall(r'Epoch\s+(\d+) \|', txt)
        best = re.findall(r'Best=([\d.]+)%', txt)
        if not eps:
            return None
        ep = int(eps[-1]) + 1  # epoch 计数从 0 起，总训练 epoch 数 = 末次+1
        out[ab] = (ep, float(best[-1]) if best else 0.0, bool(re.search(r'99\.\d+% >= 99%', txt)))
    return out


def main():
    report = {}
    for chain, dirs in CHAINS.items():
        per = {}
        for coef in COEFS:
            total_ep, acc, conv = 0, None, True
            for d in dirs:
                st = stage_stats(f'/data/cxm/recursion/{d}')
                if st is None or coef not in st:
                    conv = False
                    continue
                ep, a, c = st[coef]
                total_ep += ep
                acc = a
                conv = conv and c
            per[coef] = {'total_epochs': total_ep, 'best': acc, 'converged': conv}
        report[chain] = per
    with open('reports/curriculum_epoch_summary.json', 'w') as f:
        json.dump(report, f, indent=1)

    for chain, per in report.items():
        print(f'\n===== {chain} =====')
        for coef, d in per.items():
            print(f'  {coef}: total_ep={d["total_epochs"]:5d} best={d["best"]} converged={d["converged"]}')
    print('\nsaved reports/curriculum_epoch_summary.json')


if __name__ == '__main__':
    main()
