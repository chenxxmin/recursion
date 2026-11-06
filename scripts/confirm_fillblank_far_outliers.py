#!/usr/bin/env python3
"""对 far_random 中 zero<100% 的单元用相同基底枚举全部非零 d。"""
import argparse
import gc
import json
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from probe_fillblank_len8to64_stride8_random import (  # noqa: E402
    BASE_SELECTION_SEED, DEFAULT_CKPT_TEMPLATE, DELTA_SEED, LEN8_CKPT, P,
    checkpoint_path, load_model, make_blank_inputs, predict, probe_relative,
    reconstruct_fixed_heldout, select_base_rows,
)
from probe_fillblank_len8to64_far_random import add_fit_metrics  # noqa: E402

SOURCE = os.path.join(ROOT, 'reports/data/fillblank_len8to64_far_random_a25b56.json')
OUT = os.path.join(ROOT, 'reports/data/fillblank_len8to64_far_outlier_confirm_a25b56.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default=SOURCE)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--batch-size', type=int, default=512)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    source = json.load(open(args.source))
    targets = {}
    for L, lr in source['by_length'].items():
        for k, br in lr['by_blank'].items():
            for rel, s in br['by_relative'].items():
                if s['zero_exact_count'] < s['n']:
                    targets.setdefault(int(L), []).append((int(k), int(rel)))
    assert sum(map(len, targets.values())) == 12
    print('target cells:', targets)

    output = {'source': os.path.abspath(args.source),
              'n_deltas_per_base': 126, 'targets': targets, 'by_length': {}}
    for L, cells in sorted(targets.items()):
        path = checkpoint_path(DEFAULT_CKPT_TEMPLATE, L, LEN8_CKPT)
        print(f'\n[L={L}] {path}', flush=True)
        model, cfg, ckpt = load_model(path, args.device)
        a, b = cfg['rules'][0]
        windows, heldout = reconstruct_fixed_heldout(cfg, L, a, b, BASE_SELECTION_SEED)
        by_blank = {}
        for k in sorted({k for k, _ in cells}):
            inputs = make_blank_inputs(windows, k, cfg['blank_token_id'], cfg['eos_token_id'])
            numeric0, full0 = predict(model, inputs, args.device, P, args.batch_size)
            correct = (full0 == windows[:, k]).nonzero(as_tuple=False).flatten().tolist()
            selected = select_base_rows(correct, L, k, n_bases=16, seed=BASE_SELECTION_SEED)
            base_inputs, base_outputs = inputs[selected], numeric0[selected]
            by_relative = {}
            for _, rel in [cell for cell in cells if cell[0] == k]:
                summary, raw = probe_relative(
                    model, base_inputs, selected, base_outputs, k + rel, L, k, rel,
                    a, b, args.device, args.batch_size, n_deltas=126, delta_seed=DELTA_SEED)
                add_fit_metrics(summary)
                summary['raw'] = raw
                by_relative[f'{rel:+d}'] = summary
                old = source['by_length'][str(L)]['by_blank'][str(k)]['by_relative'][f'{rel:+d}']
                print(f'  k={k} rel={rel:+d}: old zero={old["zero_rate"]:.4f}, '
                      f'new zero={summary["zero_rate"]:.4f}, '
                      f'best={summary["best_nonzero"]}:{summary["best_nonzero_rate"]:.4f}, '
                      f'change={summary["output_change_rate"]:.4f}', flush=True)
            by_blank[str(k)] = {'base_rows': selected, 'by_relative': by_relative}
        output['by_length'][str(L)] = {
            'ckpt': path, 'heldout': heldout, 'by_blank': by_blank,
        }
        del model, ckpt
        gc.collect()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print('\nsaved', args.out)


if __name__ == '__main__':
    main()
