import os
import re
import sys
import json
import glob
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt


def parse_log(log_path):
    """Parse a training log file and return a dict of metrics."""
    data = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'best_acc': [],
        'best_epoch': [],
        'per_pos': [],   # list of (start_x, list of accs)
        'per_rule': [],  # list of list of accs
    }
    epoch_re = re.compile(
        r'Epoch\s+(\d+)\s+\|\s+Train:\s+Loss=([\d.]+)\s+Acc=([\d.]+%?)\s+\|\s+'
        r'Test:\s+Acc=([\d.]+%?)\s+\|\s+Best=([\d.]+%?)\(@(\d+)\)'
    )
    pos_re = re.compile(r'from x(\d+):\s+(.+)')
    rule_re = re.compile(r'per-rule acc:\s+(.+)')

    with open(log_path, 'r', encoding='utf-8') as f:
        text = f.read()

    current = {}
    for line in text.splitlines():
        m = epoch_re.search(line)
        if m:
            if current:
                data['per_pos'].append(current.get('per_pos'))
                data['per_rule'].append(current.get('per_rule'))
                current = {}
            epoch = int(m.group(1))
            data['epochs'].append(epoch)
            data['train_loss'].append(float(m.group(2)))
            data['train_acc'].append(parse_percent(m.group(3)))
            data['test_acc'].append(parse_percent(m.group(4)))
            data['best_acc'].append(parse_percent(m.group(5)))
            data['best_epoch'].append(int(m.group(6)))
            continue

        m = pos_re.search(line)
        if m:
            start_x = int(m.group(1))
            accs = [float(x) for x in m.group(2).split()]
            current['per_pos'] = (start_x, accs)
            continue

        m = rule_re.search(line)
        if m:
            accs = [float(x) for x in m.group(1).split()]
            current['per_rule'] = accs
            continue

    # flush last
    if current:
        data['per_pos'].append(current.get('per_pos'))
        data['per_rule'].append(current.get('per_rule'))

    return data


def parse_percent(s):
    """Parse '95.2%' to 0.952 or '0.95' to 0.95."""
    s = s.strip()
    if s.endswith('%'):
        return float(s[:-1]) / 100.0
    return float(s)


def plot_learning_curve(data, title, save_path=None):
    """Plot train/test accuracy and train loss over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = data['epochs']

    ax = axes[0]
    ax.plot(epochs, data['train_acc'], label='Train Acc', linewidth=1.5)
    ax.plot(epochs, data['test_acc'], label='Test Acc', linewidth=1.5)
    ax.plot(epochs, data['best_acc'], label='Best Test Acc', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{title} - Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    ax.plot(epochs, data['train_loss'], label='Train Loss', color='tab:red', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{title} - Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved learning curve to {save_path}")
    else:
        fig.savefig('learning_curve.png', dpi=150)
        print("Saved learning curve to learning_curve.png")
    plt.close(fig)


def plot_per_position(data, title, save_path=None):
    """Plot per-position accuracy heatmap over epochs."""
    per_pos = data['per_pos']
    if not per_pos or all(p is None for p in per_pos):
        print("No per-position accuracy data to plot.")
        return

    # Filter epochs with per_pos data
    epochs = []
    matrix = []
    for ep, pp in zip(data['epochs'], per_pos):
        if pp is not None:
            epochs.append(ep)
            matrix.append(pp[1])
    if not matrix:
        print("No per-position accuracy data to plot.")
        return

    matrix = list(reversed(matrix))  # oldest at bottom
    epochs = list(reversed(epochs))

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xlabel('Position (relative to from-x)')
    ax.set_ylabel('Epoch')
    ax.set_title(f'{title} - Per-position Accuracy')
    ax.set_yticks(range(0, len(epochs), max(1, len(epochs)//10)))
    ax.set_yticklabels([epochs[i] for i in range(0, len(epochs), max(1, len(epochs)//10))])
    fig.colorbar(im, ax=ax, label='Accuracy')
    fig.tight_layout()
    out = save_path or 'per_position.png'
    fig.savefig(out, dpi=150)
    print(f"Saved per-position heatmap to {out}")
    plt.close(fig)


def plot_per_rule(data, title, save_path=None):
    """Plot per-rule accuracy over epochs."""
    per_rule = data['per_rule']
    if not per_rule or all(r is None for r in per_rule):
        print("No per-rule accuracy data to plot.")
        return

    epochs = []
    rule_data = []
    n_rules = None
    for ep, pr in zip(data['epochs'], per_rule):
        if pr is not None:
            epochs.append(ep)
            rule_data.append(pr)
            if n_rules is None:
                n_rules = len(pr)

    if not rule_data or n_rules is None:
        print("No per-rule accuracy data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in range(n_rules):
        accs = [rd[r] for rd in rule_data]
        ax.plot(epochs, accs, label=f'Rule {r}', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{title} - Per-rule Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    out = save_path or 'per_rule.png'
    fig.savefig(out, dpi=150)
    print(f"Saved per-rule curve to {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize learning curves from training logs.')
    parser.add_argument('names', nargs='*', help='Experiment name(s).')
    parser.add_argument('--all', action='store_true', help='Visualize all experiments in experiments.json.')
    parser.add_argument('--log-dir', default='logs', help='Directory containing .log files.')
    parser.add_argument('--out-dir', default='plots', help='Directory to save plots.')
    parser.add_argument('--no-per-pos', action='store_true', help='Skip per-position plot.')
    parser.add_argument('--no-per-rule', action='store_true', help='Skip per-rule plot.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    names = args.names
    if args.all:
        exp_path = 'experiments.json'
        if os.path.exists(exp_path):
            with open(exp_path, 'r', encoding='utf-8') as f:
                names = [e['name'] for e in json.load(f)['experiments']]
        else:
            # fallback: all logs
            names = [Path(p).stem for p in glob.glob(os.path.join(args.log_dir, '*.log'))]

    if not names:
        print("No experiment names provided. Usage: python visualize.py <exp_name> [--all]")
        sys.exit(1)

    for name in names:
        log_path = os.path.join(args.log_dir, f'{name}.log')
        if not os.path.exists(log_path):
            print(f"Log not found: {log_path}, skipping.")
            continue
        data = parse_log(log_path)
        if not data['epochs']:
            print(f"No epoch data found in {log_path}, skipping.")
            continue

        base = os.path.join(args.out_dir, name)
        plot_learning_curve(data, name, save_path=f'{base}_curve.png')
        if not args.no_per_pos:
            plot_per_position(data, name, save_path=f'{base}_per_pos.png')
        if not args.no_per_rule:
            plot_per_rule(data, name, save_path=f'{base}_per_rule.png')


if __name__ == '__main__':
    main()

