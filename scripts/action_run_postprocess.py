#!/usr/bin/env python3
"""Validate resumable action state; export curves, total acc and clean acc."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def load_full(path, cfg=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    required = {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                'next_epoch', 'rng_python', 'rng_torch', 'config'}
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError('Not a complete resume checkpoint: ' + str(path))
    if cfg:
        sc = state['config']
        for key, saved in [('P', 'p'), ('D_MODEL', 'd_model'), ('N_LAYER', 'n_layer'),
                           ('N_HEAD', 'n_head'), ('MLP_RATIO', 'mlp_ratio'),
                           ('TRAIN_LEN', 'train_len'), ('OOD_LEN', 'ood_len'),
                           ('MISS_LEN', 'miss_len'), ('MISSING_PROB', 'missing_prob')]:
            if cfg[key] != sc[saved]:
                raise ValueError('Resume config mismatch: ' + key)
        if [list(p) for p in sc['ab_pairs']] != cfg['AB_PAIRS']:
            raise ValueError('Resume rules mismatch')
        if sc.get('reshuffle_each_epoch', False) != cfg['RESHUFFLE_EACH_EPOCH']:
            raise ValueError('Resume sampler mismatch')
        scheduler = state['scheduler_state_dict']
        if scheduler['T_max'] != cfg['EPOCHS'] or any(
                not math.isclose(lr, cfg['LR'], rel_tol=1e-10) for lr in scheduler['base_lrs']):
            raise ValueError('Resume scheduler/base LR mismatch')
        if any(group['weight_decay'] != cfg['WEIGHT_DECAY']
               for group in state['optimizer_state_dict']['param_groups']):
            raise ValueError('Resume weight decay mismatch')
        if state['next_epoch'] >= cfg['EPOCHS']:
            raise ValueError('Checkpoint already reached the scheduler horizon')
    return state


def evaluate_run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from experiment import _prepare_action
    from datasets import ActionDataset, action_collate_fn
    from training import evaluate

    root = args.run_dir.resolve()
    cfg = json.loads((root / 'configs/experiment.json').read_text())['main']
    full_path = root / 'checkpoints/model_resume.pth'
    if not full_path.is_file():
        full_path = root / 'checkpoints/model_latest.pth'
    # At a naturally completed horizon the checkpoint is valid even though it
    # cannot be resumed further with the same horizon.
    full = load_full(full_path)
    log = (root / 'logs/train.log').read_text(errors='replace')
    pattern = r'^Epoch\s+(\d+)\s+\| Train: Loss=([\d.eE+-]+) Acc=([\d.]+)% \| Test: Acc=([\d.]+)%'
    rows = [[float(x) for x in row] for row in re.findall(pattern, log, re.M)]
    metrics = dict(total_acc=float(full['best_acc']), total_acc_source='best training-time fresh-test accuracy',
                   next_epoch=int(full['next_epoch']), resume_checkpoint=str(full_path),
                   clean_num_samples=args.num_samples, clean_eval_len=cfg['TRAIN_LEN'],
                   clean_sample_seed=args.sample_seed)
    summary = root / 'metrics/summary.json'
    summary.write_text(json.dumps(metrics, indent=2) + '\n')
    if rows:
        (root / 'metrics/learning_curve.csv').write_text(
            'epoch,train_loss,train_accuracy_percent,test_accuracy_percent\n' +
            ''.join(','.join(map(str, row)) + '\n' for row in rows))
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot([r[0] for r in rows], [r[1] for r in rows])
        axes[0].set(xlabel='Epoch', ylabel='Train loss')
        for col, label in ((2, 'Train'), (3, 'Fresh test')):
            axes[1].plot([r[0] for r in rows], [r[col] for r in rows], label=label)
        axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)')
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(root / 'plots/learning_curve.png', dpi=160)
        plt.close(figure)
    best = root / 'checkpoints/model_best.pth'
    weights = torch.load(best, map_location='cpu', weights_only=False) if best.is_file() else full['model_state_dict']
    if isinstance(weights, dict) and 'model_state_dict' in weights:
        weights = weights['model_state_dict']
    metrics['clean_checkpoint'] = str(best if best.is_file() else full_path)
    del full
    device = 'cpu' if args.cpu else 'cuda'
    if device == 'cuda' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('Clean evaluation requires the reserved GPU; no CPU fallback')
    torch.manual_seed(args.sample_seed)
    if device == 'cuda' and cfg['ALLOW_TF32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    small = dict(cfg, NUM_TRAIN_SAMPLES=1, NUM_TEST_SAMPLES=1,
                 MISSING_PROB=0.0, FRESH_TEST_PER_EVAL=False)
    model = _prepare_action({'main': small, '_BATCH_RUN_MERGED': True})['model']
    model.load_state_dict(weights)
    del weights
    model = model.to(device).eval()
    ds = ActionDataset(p=cfg['P'], ab_pairs=cfg['AB_PAIRS'], num_samples=args.num_samples,
                       length=cfg['TRAIN_LEN'], seed=args.sample_seed)
    loader = DataLoader(ds, batch_size=cfg['BATCH_SIZE'], shuffle=False, collate_fn=action_collate_fn)
    loss, acc, positions, _ = evaluate(model, loader, device, num_mask=0)
    metrics.update(clean_loss=float(loss), clean_acc=float(acc),
                   clean_per_position={f'x{k // 2 + 2}': float(v) for k, v in positions.items()})
    summary.write_text(json.dumps(metrics, indent=2) + '\n')
    print('RESULT ' + json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    ap = sub.add_parser('validate')
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--config-json', required=True)
    ap = sub.add_parser('evaluate')
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--num-samples', type=int, default=1000)
    ap.add_argument('--sample-seed', type=int, default=123)
    ap.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.command == 'validate':
        saved = load_full(args.checkpoint, json.loads(args.config_json))
        print('VALID RESUME', args.checkpoint, 'next_epoch=', saved['next_epoch'])
    else:
        evaluate_run(args)


if __name__ == '__main__':
    main()
