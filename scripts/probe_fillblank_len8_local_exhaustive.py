#!/usr/bin/env python3
"""Re-run all nonzero local perturbations on the historical len8 a25b56 bases.

No training. Reuses each blank's 32 stored correct bases and all 128 baseline
windows. Only offsets -2,-1,+1,+2 are touched. Saves after every completed cell.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_fillblank_last_blank_linear import load_model, predict  # noqa: E402
from probe_fillblank_len8to64_stride8_random import (  # noqa: E402
    legal_relative_positions, make_blank_inputs, probe_relative,
)

STEM = 'fillblank_len8_local_exhaustive_a25b56'
DEFAULT_CKPT = ROOT.parent / 'circuit_analysis/checkpoints_len8_s1k1' / (
    'single_o2_fillblank_a25b56_d512l4r8h4_P127_len8_randpos_noaux_'
    'lr0.0003_wd2_initLen35_seed42_best.pth')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compare_historical(new, old):
    lookup = {(row, delta): i for i, (row, delta) in enumerate(zip(new['base_row'], new['d']))}
    mismatches = []
    for i, (row, delta) in enumerate(zip(old['base_row'], old['d'])):
        j = lookup[row, delta]
        if any(new[key][j] != old[key][i] for key in ['y0', 'y1', 'full1', 'D']):
            mismatches.append({'base_row': row, 'd': delta,
                               'old': {key: old[key][i] for key in ['y0', 'y1', 'full1', 'D']},
                               'new': {key: new[key][j] for key in ['y0', 'y1', 'full1', 'D']}})
    return {'n_compared': len(old['d']), 'n_mismatched': len(mismatches), 'mismatches': mismatches}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    ap.add_argument('--historical', type=Path, default=ROOT / 'reports/data/fillblank_len8to64_stride8_random_a25b56.json')
    ap.add_argument('--out', type=Path, default=ROOT / f'reports/data/{STEM}.json')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)
    n_cells = sum(len(legal_relative_positions(8, k)) for k in range(8))
    print(f'LEN=8, a=25, b=56, P=127, s=1, blank_count=1; checkpoint={args.ckpt}', flush=True)
    print(f'32 historical correct bases/blank, d=1..126, {n_cells} local cells, '
          f'{n_cells * 32 * 126} perturbations; baseline=8*128=1024', flush=True)
    print(f'device={args.device}, CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}, '
          f'fp32, AMP off, TF32 off, batch_size={args.batch_size}', flush=True)
    if args.dry_run:
        return
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError('batch-size and threads must be positive')
    if args.device.startswith('cuda') and os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2', '3'):
        raise ValueError('CUDA inference requires CUDA_VISIBLE_DEVICES=2 or 3 (physical GPU restriction)')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    old = json.loads(args.historical.read_text())['by_length']['8']
    model, cfg, checkpoint = load_model(args.ckpt, args.device)
    if cfg != old['model_config']:
        raise ValueError('checkpoint config differs from historical len8 checkpoint')
    for key, reference in [('final_epoch', 'checkpoint_final_epoch'), ('best_accuracy', 'checkpoint_best_accuracy')]:
        if checkpoint.get(key) != old[reference]:
            raise ValueError(f'checkpoint {key} differs from historical checkpoint')
    if cfg.get('blank_lens', [1]) != [1] or cfg.get('blank_counts', [1]) != [1]:
        raise ValueError('requires original single-value, single-blank layout')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('all model parameters must be fp32')
    report = {
        'task': 'fill_blank len8 a25b56 exhaustive local perturbations',
        'status': 'running', 'p': 127, 'lengths': [8], 'n_bases_per_blank': 32,
        'n_deltas_per_base': 126, 'offsets': [-2, -1, 1, 2],
        'device': args.device, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'precision': 'fp32; AMP off; TF32 off', 'batch_size': args.batch_size,
        'checkpoint_sha256': sha256(args.ckpt), 'historical_sha256': sha256(args.historical),
        'source_sha256': sha256(Path(__file__)), 'historical_source': str(args.historical.resolve()),
        'base_row_definition': 'row in historical fixed 128 held-out windows, NOT local 0..31',
        'by_length': {'8': {key: copy.deepcopy(old[key]) for key in [
            'model_config', 'checkpoint_final_epoch', 'checkpoint_best_accuracy', 'heldout']}},
    }
    current = report['by_length']['8']
    current['ckpt'] = str(args.ckpt.resolve())
    current['by_blank'] = {}
    windows = torch.tensor(old['heldout']['clean_windows'], dtype=torch.long)
    assert windows.shape == (128, 8)
    assert torch.equal(windows[:, 2:], (25 * windows[:, 1:-1] + 56 * windows[:, :-2]) % 127)
    assert (windows[:, 0] + 127 * windows[:, 1]).tolist() == old['heldout']['state_indices']
    started = time.monotonic()
    for k in range(8):
        old_blank = old['by_blank'][str(k)]
        inputs = make_blank_inputs(windows, k, 127, 128)
        numeric, full = predict(model, inputs, args.device, 127, args.batch_size)
        correct = full == windows[:, k]
        eligible = correct.nonzero(as_tuple=False).flatten().tolist()
        if eligible != old_blank['base_selection']['eligible_correct_rows']:
            raise RuntimeError(f'x{k}: held-out correct-window set differs from historical data')
        selected = old_blank['base_selection']['base_rows']
        if numeric[selected].tolist() != old_blank['base_selection']['base_outputs']:
            raise RuntimeError(f'x{k}: clean base outputs differ from historical data')
        current_blank = {key: copy.deepcopy(old_blank[key]) for key in ['blank_position', 'category', 'base_selection']}
        current_blank['accuracy'] = {'n': 128, 'full_vocab_correct_count': int(correct.sum()),
                                    'full_vocab_accuracy': int(correct.sum()) / 128}
        current_blank['by_relative'] = {}
        current['by_blank'][str(k)] = current_blank
        print(f'x{k}: baseline {int(correct.sum())}/128, all 32 selected base outputs match', flush=True)
        for relative, j in legal_relative_positions(8, k):
            summary, raw = probe_relative(model, inputs[selected], selected, numeric[selected], j,
                                          8, k, relative, 25, 56, args.device, args.batch_size,
                                          n_deltas=126)
            for base in selected:
                deltas = [d for row, d in zip(raw['base_row'], raw['d']) if row == base]
                assert sorted(deltas) == list(range(1, 127))
            validation = compare_historical(raw, old_blank['by_relative'][f'{relative:+d}']['raw'])
            summary.update(raw=raw, historical_validation=validation)
            current_blank['by_relative'][f'{relative:+d}'] = summary
            save(args.out, report)
            print(f'  r={relative:+d}: n={len(raw["d"])}, old replay mismatches='
                  f'{validation["n_mismatched"]}/{validation["n_compared"]}, '
                  f'elapsed={time.monotonic() - started:.1f}s', flush=True)
    cells = [cell for blank in current['by_blank'].values() for cell in blank['by_relative'].values()]
    report['validation'] = {
        'n_perturbations': sum(cell['n'] for cell in cells),
        'n_historical_perturbations_compared': sum(cell['historical_validation']['n_compared'] for cell in cells),
        'n_historical_mismatches': sum(cell['historical_validation']['n_mismatched'] for cell in cells),
        'clean_baselines_match_historical': True,
        'exhaustive_delta_enumeration_passed': True,
    }
    assert report['validation']['n_perturbations'] == 104832
    # Independent CPU spot-check, including deltas outside the old 8-delta sample.
    spot_inputs, expected_numeric, expected_full = [], [], []
    for k in range(8):
        blank_entry = current['by_blank'][str(k)]
        base = blank_entry['base_selection']['base_rows'][0]
        for rel_key, cell in blank_entry['by_relative'].items():
            raw = cell['raw']
            for delta in [1, 2, 63, 126]:
                index = next(i for i, (row, d) in enumerate(zip(raw['base_row'], raw['d']))
                             if row == base and d == delta)
                sample = make_blank_inputs(windows[base:base + 1], k, 127, 128)[0]
                j = k + int(rel_key)
                sample[j] = (sample[j] + delta) % 127
                spot_inputs.append(sample)
                expected_numeric.append(raw['y1'][index])
                expected_full.append(raw['full1'][index])
    model.to('cpu')
    cpu_numeric, cpu_full = predict(model, torch.stack(spot_inputs), 'cpu', 127, 128)
    if cpu_numeric.tolist() != expected_numeric or cpu_full.tolist() != expected_full:
        raise RuntimeError('independent CPU spot-check predictions differ')
    report['validation']['cpu_spotcheck_n'] = len(spot_inputs)
    report['validation']['cpu_spotcheck_predictions_match'] = True
    report['elapsed_seconds'] = time.monotonic() - started
    report['status'] = 'complete'
    save(args.out, report)
    print(f'DONE {args.out}; validation={report["validation"]}', flush=True)


if __name__ == '__main__':
    main()
