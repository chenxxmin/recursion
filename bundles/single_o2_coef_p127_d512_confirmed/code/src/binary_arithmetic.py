"""Two-token modular-linear-function controls for single-rule recurrence experiments.

Run from the repository root:
    python src/binary_arithmetic.py experiments/binary_arithmetic_h1_p127.json

This intentionally reuses the recurrence repository's model, dataset and training
engine, rather than changing the behavior of existing 'single' experiments.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import BucketBatchSampler, RecurrenceDataset, collate_fn
from models import FibonacciTransformer
from training import run_training_engine


class SharedEmbeddingBinaryMLP(nn.Module):
    """Learn f(q0,q1) with one embedding table and tied output weights.

    Returns LM-shaped logits so the normal training engine supervises *only*
    target[1] (q2) using num_mask=1, just like an order-2 single task.
    """
    def __init__(self, p, d_model=128, mlp_ratio=8, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(p, d_model)
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.lm_head = nn.Linear(d_model, p, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, loss_mask=None, ab_labels=None):
        if idx.ndim != 2 or idx.size(1) != 3:
            raise ValueError('expected [q0, q1, q2] tokens; q2 is label only')
        e0 = self.embedding(idx[:, 0])
        e1 = self.embedding(idx[:, 1])
        pred = self.lm_head(self.net(torch.cat((e0, e1), dim=-1)))
        # target[0]=q1 is ignored; target[1]=q2 is the sole scored prediction.
        logits = torch.stack((torch.zeros_like(pred), pred), dim=1)
        loss = None
        if targets is not None:
            if targets.ndim != 2 or targets.size(1) < 2:
                raise ValueError('expected next-token targets [q1,q2]')
            if loss_mask is not None and not torch.all(loss_mask[:, 0] == 0):
                raise ValueError('binary MLP only supports last-position loss')
            loss = F.cross_entropy(pred, targets[:, 1])
        return logits, loss


def build_binary_samples(p, a, b, split, train_ratio, split_seed):
    """Return disjoint length-3 windows [q0,q1,q2], with q2=(a*q1+b*q0)%p.

    full_split: exactly the existing RecurrenceDataset.run_full_split() behavior
                (orbit traversal and possible boundary-orbit split).
    fixed_states: same (q0,q1) train/test partition for *all* coefficient sets.
    """
    if not (p >= 2 and 0 < train_ratio < 1 and 0 <= a < p and 0 <= b < p):
        raise ValueError('invalid modulus, train ratio, or coefficient')
    n = int(p * p * train_ratio)
    if not 0 < n < p*p:
        raise ValueError('train ratio produces an empty split')
    if split == 'fixed_states':
        states = list(range(p * p))
        random.Random(split_seed).shuffle(states)
        def make(indices):
            pairs = torch.tensor([[z // p, z % p] for z in indices], dtype=torch.long)
            next_value = (a * pairs[:, 1] + b * pairs[:, 0]).remainder(p)
            return list(torch.cat((pairs, next_value[:, None]), dim=1).unbind(0))
        train, test = make(states[:n]), make(states[n:])
    elif split == 'full_split':
        # RecurrenceDataset uses Python's module-level random; restore its
        # state so the caller's initialization RNG is unaffected.
        previous_rng = random.getstate()
        try:
            random.seed(split_seed)
            ds = RecurrenceDataset(
                p=p, init_len=2, length=3, num_samples=n, verbose=False,
                recurrence_fn=lambda seq, mod: (a * seq[-1] + b * seq[-2]) % mod,
                recurrence_name=f'{a},{b}')
            ds.run_full_split()
            train, test = ds.train_samples, ds.test_samples
        finally:
            random.setstate(previous_rng)
    else:
        raise ValueError(f'unknown split: {split}')
    if len(train) != n or len(test) != p*p-n:
        raise AssertionError('incorrect number of train/test states')
    return train, test


def create_model(kind, p, width, layers, heads, mlp_ratio, dropout):
    if kind == 'transformer':
        return FibonacciTransformer(
            p=p, vocab_size=p, block_size=4, d_model=width,
            n_layer=layers, n_head=heads, mlp_ratio=mlp_ratio, dropout=dropout)
    if kind == 'mlp':
        return SharedEmbeddingBinaryMLP(p=p, d_model=width,
                                        mlp_ratio=mlp_ratio, dropout=dropout)
    raise ValueError(f'unknown model kind: {kind}')


def run_one(cfg, kind, rule, seed, out_dir, device):
    p = int(cfg['p'])
    a, b = (int(x) for x in rule)
    split_seed = int(cfg['split_seed'])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train, test = build_binary_samples(
        p, a, b, cfg['split'], cfg['train_ratio'], split_seed)
    batch_size = int(cfg['batch_size'])
    train_loader = DataLoader(
        train, batch_sampler=BucketBatchSampler(
            train, batch_size, shuffle=True,
            reshuffle_each_epoch=cfg.get('reshuffle_each_epoch', False)),
        collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn)
    model = create_model(kind, p, int(cfg['d_model']), int(cfg['n_layer']),
                         int(cfg['n_head']), int(cfg['mlp_ratio']),
                         float(cfg.get('dropout', 0.0))).to(device)
    epochs = int(cfg['epochs'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']),
                                   weight_decay=float(cfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    label = f'{kind}_a{a}b{b}_d{cfg["d_model"]}_seed{seed}_{cfg["split"]}'
    output_path = out_dir / label
    output_path.mkdir(parents=True, exist_ok=True)
    save_config = {
        'task': 'binary_mod_arithmetic', 'model_kind': kind, 'p': p,
        'rules': [[a, b]], 'seed': seed, 'split_seed': split_seed,
        'split': cfg['split'], 'train_ratio': cfg['train_ratio'],
        'train_count': len(train), 'test_count': len(test),
        'd_model': cfg['d_model'], 'n_layer': cfg['n_layer'],
        'n_head': cfg['n_head'], 'mlp_ratio': cfg['mlp_ratio'],
        'epochs': epochs, 'learning_rate': cfg['lr'],
        'weight_decay': cfg['weight_decay'], 'batch_size': batch_size,
        'num_mask': 1, 'input_order': '[q0,q1]',
        'target': '(a*q1+b*q0)%p', 'shared_embedding': True,
        'tied_output_embedding': True,
    }
    (output_path / 'config.json').write_text(
        json.dumps(save_config, indent=2) + '\n', encoding='utf-8')
    print(f'\n=== {label}: {len(train)} train, {len(test)} test, '
          f'{sum(x.numel() for x in model.parameters()):,} parameters ===', flush=True)
    best, epoch, interrupted = run_training_engine(
        model, train_loader, test_loader, optimizer, scheduler, str(device),
        epochs=epochs, eval_interval=int(cfg['eval_interval']),
        early_stop_accuracy=float(cfg.get('early_stop_accuracy', 0.99)),
        early_stop_no_improve=int(cfg.get('early_stop_no_improve', epochs)),
        save_path=str(output_path / 'model.pth'), save_config=save_config,
        num_mask=1, grad_clip_norm=float(cfg.get('grad_clip_norm', 1.0)))
    return dict(model=kind, a=a, b=b, seed=seed, split=cfg['split'],
                best_test_acc=best, last_epoch=epoch, interrupted=interrupted,
                output_dir=str(output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='JSON experiment configuration')
    parser.add_argument('--model', choices=('transformer', 'mlp', 'both'),
                        default='both')
    parser.add_argument('--rule', help='Run one rule, e.g. 62,59; default all rules')
    parser.add_argument('--seed', type=int, help='Run one seed; default all config seeds')
    parser.add_argument('--epochs', type=int, help='Override epoch budget')
    parser.add_argument('--split', choices=('full_split', 'fixed_states'),
                        help='Override train/test split mode')
    parser.add_argument('--out-dir', help='Override output directory')
    parser.add_argument('--device', choices=('cpu', 'cuda'),
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.split:
        cfg['split'] = args.split
    rules = [tuple(map(int, args.rule.split(',')))] if args.rule else cfg['rules']
    seeds = [args.seed] if args.seed is not None else cfg['seeds']
    kinds = ('transformer', 'mlp') if args.model == 'both' else (args.model,)
    out_dir = Path(args.out_dir or cfg.get('output_dir', 'binary_arithmetic_results'))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    summary_path = out_dir / 'summary.csv'
    fields = ['model', 'a', 'b', 'seed', 'split', 'best_test_acc',
              'last_epoch', 'interrupted', 'output_dir']
    write_header = not summary_path.exists()
    with summary_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for kind in kinds:
            for rule in rules:
                for seed in seeds:
                    result = run_one(cfg, kind, rule, seed, out_dir, device)
                    writer.writerow(result)
                    f.flush()
                    print('RESULT:', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
