"""Evaluate clean-input (no corruption) next-token accuracy of an action+missing checkpoint.

Training-time test accuracy is measured on corrupted inputs (on-the-fly
missing runs via make_action_missing_collate). This script rebuilds the same
model and evaluates it through the same evaluate() path but with
missing_prob=0, i.e. completely clean inputs. Samples are fresh ActionDataset
draws with a fixed seed, at the checkpoint's test window length (OOD_LEN,
matching the training-time test split).

Usage: python3 scripts/eval_clean_action.py <checkpoint.pth> --gpu <0-3> [--num-samples 1000]
The checkpoint may be a full dict (_latest/_resume, carries 'config') or pure
weights (_best); in the latter case the config is read from a sibling
_latest/_resume checkpoint in the same directory.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--num-samples', type=int, default=1000)
    ap.add_argument('--sample-seed', type=int, default=123)
    ap.add_argument('--eval-len', type=int, default=None,
                    help='eval window length; default = checkpoint test split length (ood_len)')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--missing-prob', type=float, default=0.0,
                    help='>0 reproduces the training-time corrupted-input eval (sanity check)')
    return ap.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
from torch.utils.data import DataLoader

import experiment
from datasets import ActionDataset, make_action_missing_collate
from training import evaluate


def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict'], ckpt['config']
    # Pure-weights checkpoint (_best.pth): find config in a sibling full checkpoint.
    state_dict = ckpt
    d = os.path.dirname(path)
    for cand in sorted(glob.glob(os.path.join(d, '*_latest.pth')) +
                       glob.glob(os.path.join(d, '*_resume.pth'))):
        full = torch.load(cand, map_location='cpu', weights_only=False)
        if isinstance(full, dict) and 'config' in full:
            print(f"config taken from sibling {os.path.basename(cand)}")
            return state_dict, full['config']
    raise RuntimeError(f"no config found in {path} or its siblings")


def build_model(sc):
    """Rebuild the exact model via the repo's own prepare path (MISSING_PROB=0)."""
    base = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       'src', 'config.json')))
    mm = dict(base['main'])
    mm.update(base.get('action', {}))
    mm.update({
        'P': sc['p'],
        'D_MODEL': sc['d_model'],
        'N_HEAD': sc['n_head'],
        'N_LAYER': sc['n_layer'],
        'USE_LEARNABLE_PE': sc.get('use_learnable_pe', False),
        'MLP_RATIO': sc.get('mlp_ratio', 4),
        'AB_PAIRS': [list(pair) for pair in sc['ab_pairs']],
        'TRAIN_LEN': sc['train_len'],
        'OOD_LEN': sc['ood_len'],
        'MISSING_PROB': 0.0,
        'NUM_TRAIN_SAMPLES': 8,  # unused here; keep dataset construction cheap
        'NUM_TEST_SAMPLES': 8,
    })
    cfg = dict(base)
    cfg['main'] = mm
    cfg['_BATCH_RUN_MERGED'] = True
    ctx = experiment._prepare_action(cfg)
    return ctx


def main():
    state_dict, sc = load_checkpoint(args.checkpoint)
    print(f"checkpoint config: {sc}")
    ctx = build_model(sc)
    model = ctx['model'].to('cuda').eval()
    model.load_state_dict(state_dict)

    # Default window follows the training-time test split (ActionDataset length
    # = OOD_LEN); --eval-len overrides it (e.g. 64 = the training window).
    eval_len = args.eval_len if args.eval_len is not None else sc['ood_len']
    eval_ds = ActionDataset(p=sc['p'], ab_pairs=sc['ab_pairs'],
                            num_samples=args.num_samples, length=eval_len,
                            seed=args.sample_seed)
    # Same collate family as training, but missing_prob=0 -> no corruption.
    collate = make_action_missing_collate(p=sc['p'], missing_prob=args.missing_prob,
                                          miss_len=sc.get('miss_len', 1))
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate)

    loss, acc, per_pos_acc, _ = evaluate(model, loader, 'cuda',
                                         num_mask=ctx['num_mask'])

    # Per-position summary: target index t predicts value x_k with k = t/2 + 2.
    pos_items = sorted(per_pos_acc.items())
    summary = {f'x{t // 2 + 2}': round(a, 6) for t, a in pos_items}
    worst = min(pos_items, key=lambda kv: kv[1])
    print(f"\nclean-input eval: n={args.num_samples}, len={eval_len}, seed={args.sample_seed}")
    print(f"positions evaluated per sample: {len(pos_items)} (x3..x{eval_len})")
    print(f"worst position: x{worst[0] // 2 + 2} acc={worst[1]:.4f}")
    print('per-position acc:', json.dumps(summary))
    print('RESULT ' + json.dumps({
        'checkpoint': args.checkpoint,
        'num_samples': args.num_samples,
        'eval_len': eval_len,
        'sample_seed': args.sample_seed,
        'clean_acc': acc,
        'loss': loss,
        'worst_pos': {'pos': f"x{worst[0] // 2 + 2}", 'acc': worst[1]},
        'per_pos_acc': summary,
    }))


if __name__ == '__main__':
    main()
