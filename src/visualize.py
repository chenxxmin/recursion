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

# Plot style constants
ACC_YLIM = (-0.05, 1.05)   # y-range for accuracy axes (small pad around [0, 1])
FIG_DPI = 150              # saved figure resolution
LOSS_YLIM_QUANTILE = 0.95  # loss axis ceiling ignores spikes above this quantile
LOSS_YLIM_MARGIN = 1.2     # headroom multiplier above the quantile
LOSS_YLIM_MIN = 0.5        # lower bound for the loss axis ceiling
MAX_YTICKS_SINGLE = 10     # epoch tick count on single-experiment heatmaps
MAX_YTICKS_GRID = 5        # epoch tick count on small grid panels


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

    def flush_current(cur):
        """Append the pending per-epoch per_pos/per_rule of cur (if any) to data."""
        if cur:
            data['per_pos'].append(cur.get('per_pos'))
            data['per_rule'].append(cur.get('per_rule'))

    for line in text.splitlines():
        m = epoch_re.search(line)
        if m:
            flush_current(current)
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
    flush_current(current)

    return data


def parse_percent(s):
    """Parse '95.2%' to 0.952 or '0.95' to 0.95."""
    s = s.strip()
    if s.endswith('%'):
        return float(s[:-1]) / 100.0
    return float(s)


def extract_setting_and_seed(name):
    """Split experiment name into setting and seed.

    Example: 'mixed_basic_d512l2r8_e0.5_0.5_seed0'
             -> ('mixed_basic_d512l2r8_e0.5_0.5', '0')
    Returns (name, None) if no seed suffix.
    """
    m = re.match(r'(.+)_seed(\d+)$', name)
    if m:
        return m.group(1), m.group(2)
    return name, None


def _data_items(data):
    """Normalize input to a list of (label, data_dict).

    A plain dict (single experiment) becomes a one-item list with an EMPTY-LABEL
    marker (''); _is_single relies on this convention to detect single mode.
    """
    if isinstance(data, dict):
        return [('', data)]
    return data


def _is_single(items):
    """True when items hold a single, unlabeled experiment (see _data_items)."""
    return len(items) == 1 and items[0][0] == ''


def seed_sort_key(item):
    """Sort key for (seed_label, ...) items.

    Numeric seed labels sort numerically; the empty label (standalone
    experiment without a seed suffix) gets -1 so it sorts first.
    """
    seed = item[0]
    return int(seed) if seed.isdigit() else -1


def _make_subplot_grid(n, ncols=4):
    """Create an ncols-wide subplot grid for n panels; unused panels are hidden."""
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 2.5*nrows),
                             squeeze=False, constrained_layout=True)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis('off')
    return fig, axes


def plot_learning_curve(data, title, save_path=None):
    """Plot train/test accuracy and train loss over epochs.

    Parameters
    ----------
    data : dict or list of (label, dict)
        If a dict, plot a single experiment (legacy behaviour).
        If a list, plot multiple runs; label is used to annotate curves
        (e.g. the seed number).
    """
    items = _data_items(data)
    single_mode = _is_single(items)

    if single_mode:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        d = items[0][1]
        epochs = d['epochs']
        ax.plot(epochs, d['train_acc'], label='Train Acc', linewidth=1.5)
        ax.plot(epochs, d['test_acc'], label='Test Acc', linewidth=1.5)
        ax.plot(epochs, d['best_acc'], label='Best Test Acc',
                linewidth=1.5, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{title} - Accuracy')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ACC_YLIM)

        ax = axes[1]
        ax.plot(epochs, d['train_loss'], label='Train Loss',
                linewidth=1.5, color='tab:red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{title} - Loss')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
    else:
        n = len(items)
        fig, axes = _make_subplot_grid(n)

        for idx, (label, d) in enumerate(items):
            ax = axes[idx]
            ax_loss = ax.twinx()
            epochs = d['epochs']
            if not epochs:
                continue

            # Left y-axis: accuracy
            l1, = ax.plot(epochs, d['train_acc'], label='train acc',
                          color='tab:blue', linewidth=1.2)
            l2, = ax.plot(epochs, d['test_acc'], label='test acc',
                          color='tab:orange', linewidth=1.2)
            l3, = ax.plot(epochs, d['best_acc'], label='best',
                          color='tab:green', linewidth=1.0, linestyle='--')
            ax.set_ylabel('Accuracy', color='tab:blue')
            ax.tick_params(axis='y', labelcolor='tab:blue')
            ax.set_ylim(ACC_YLIM)

            # Right y-axis: loss
            l4, = ax_loss.plot(epochs, d['train_loss'], label='loss',
                               color='tab:red', linewidth=1.0, alpha=0.7)
            ax_loss.set_ylabel('Loss', color='tab:red')
            ax_loss.tick_params(axis='y', labelcolor='tab:red')
            # Keep the loss axis focused on the main trend (ignore first-epoch spikes).
            sorted_loss = sorted(d['train_loss'])
            p95_idx = int(LOSS_YLIM_QUANTILE * len(sorted_loss))
            loss_ylim = max(sorted_loss[p95_idx] * LOSS_YLIM_MARGIN, LOSS_YLIM_MIN)
            ax_loss.set_ylim(0, loss_ylim)

            ax.set_title(f'seed {label}', fontsize=9)
            ax.set_xlabel('Epoch', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax_loss.tick_params(axis='y', which='major', labelsize=6)
            ax.grid(True, alpha=0.3)

            # Combined legend
            lines = [l1, l2, l3, l4]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='best', fontsize=6)

        fig.suptitle(f'{title} - Learning Curve', fontsize=12)

    out = save_path or ('learning_curve.png' if single_mode else f'{title}_curve.png')
    fig.savefig(out, dpi=FIG_DPI)
    print(f"Saved learning curve to {out}")
    plt.close(fig)


def plot_per_position(data, title, save_path=None):
    """Plot per-position accuracy heatmap over epochs.

    Parameters
    ----------
    data : dict or list of (label, dict)
        Single experiment or a list of runs (e.g. different seeds).
    """
    def _regularize(per_pos, epochs):
        """Return (epochs, matrix) aligned to a common position axis.

        Per-position rows may have different lengths or start positions.
        Missing values are filled with NaN so imshow receives a regular array.
        """
        entries = []
        min_start = None
        max_end = None
        for ep, pp in zip(epochs, per_pos):
            if pp is None:
                continue
            start_x, accs = pp
            entries.append((ep, start_x, list(accs)))
            if min_start is None or start_x < min_start:
                min_start = start_x
            end = start_x + len(accs)
            if max_end is None or end > max_end:
                max_end = end
        if not entries or min_start is None or max_end is None:
            return [], []

        n_cols = max_end - min_start
        matrix = []
        out_epochs = []
        for ep, start_x, accs in entries:
            row = [float('nan')] * n_cols
            offset = start_x - min_start
            row[offset:offset + len(accs)] = accs
            matrix.append(row)
            out_epochs.append(ep)
        return out_epochs, matrix

    items = _data_items(data)
    single_mode = _is_single(items)

    if single_mode:
        d = items[0][1]
        per_pos = d['per_pos']
        if not per_pos or all(p is None for p in per_pos):
            print("No per-position accuracy data to plot.")
            return

        epochs, matrix = _regularize(per_pos, d['epochs'])
        if not matrix:
            print("No per-position accuracy data to plot.")
            return

        matrix = list(reversed(matrix))
        epochs = list(reversed(epochs))

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xlabel('Position (relative to from-x)')
        ax.set_ylabel('Epoch')
        ax.set_title(f'{title} - Per-position Accuracy')
        ax.set_yticks(range(0, len(epochs), max(1, len(epochs)//MAX_YTICKS_SINGLE)))
        ax.set_yticklabels([epochs[i] for i in range(0, len(epochs), max(1, len(epochs)//MAX_YTICKS_SINGLE))])
        fig.colorbar(im, ax=ax, label='Accuracy')
        fig.tight_layout()
    else:
        valid_items = []
        for label, d in items:
            epochs, matrix = _regularize(d['per_pos'], d['epochs'])
            if matrix:
                matrix = list(reversed(matrix))
                epochs = list(reversed(epochs))
                valid_items.append((label, epochs, matrix))
        if not valid_items:
            print("No per-position accuracy data to plot.")
            return

        n = len(valid_items)
        fig, axes = _make_subplot_grid(n)
        for idx, (label, epochs, matrix) in enumerate(valid_items):
            ax = axes[idx]
            im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_title(f'seed {label}', fontsize=9)
            ax.set_xlabel('Position', fontsize=8)
            ax.set_ylabel('Epoch', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            step = max(1, len(epochs)//MAX_YTICKS_GRID)
            ax.set_yticks(range(0, len(epochs), step))
            ax.set_yticklabels([epochs[i] for i in range(0, len(epochs), step)], fontsize=6)
        fig.suptitle(f'{title} - Per-position Accuracy', fontsize=12)
        # im is the AxesImage of the last panel; valid_items is non-empty
        # (checked above), so the loop always assigns it. All panels share
        # the same vmin/vmax, so one colorbar serves the whole figure.
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Accuracy')

    out = save_path or ('per_position.png' if single_mode else f'{title}_per_pos.png')
    fig.savefig(out, dpi=FIG_DPI)
    print(f"Saved per-position heatmap to {out}")
    plt.close(fig)


def plot_per_rule(data, title, save_path=None):
    """Plot per-rule accuracy over epochs.

    Parameters
    ----------
    data : dict or list of (label, dict)
        Single experiment or a list of runs.
    """
    items = _data_items(data)
    single_mode = _is_single(items)

    if single_mode:
        d = items[0][1]
        per_rule = d['per_rule']
        if not per_rule or all(r is None for r in per_rule):
            print("No per-rule accuracy data to plot.")
            return

        epochs = []
        rule_data = []
        n_rules = None
        for ep, pr in zip(d['epochs'], per_rule):
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
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ACC_YLIM)
        fig.tight_layout()
    else:
        parsed = []
        n_rules = None
        for label, d in items:
            epochs = []
            rule_data = []
            for ep, pr in zip(d['epochs'], d['per_rule']):
                if pr is not None:
                    epochs.append(ep)
                    rule_data.append(pr)
                    if n_rules is None:
                        n_rules = len(pr)
            if rule_data and n_rules is not None:
                parsed.append((label, epochs, rule_data))

        if not parsed or n_rules is None:
            print("No per-rule accuracy data to plot.")
            return

        n = len(parsed)
        fig, axes = _make_subplot_grid(n)
        colors = plt.cm.tab10.colors

        for idx, (label, epochs, rule_data) in enumerate(parsed):
            ax = axes[idx]
            for r in range(n_rules):
                accs = [rd[r] for rd in rule_data]
                line, = ax.plot(epochs, accs, label=f'rule {r}',
                                linewidth=1.5, color=colors[r % len(colors)])
                # Annotate the highest accuracy on this curve.
                max_acc = max(accs)
                max_idx = accs.index(max_acc)
                max_epoch = epochs[max_idx]
                ax.annotate(f'{max_acc:.2f}',
                            xy=(max_epoch, max_acc),
                            fontsize=6, color=line.get_color(),
                            textcoords='offset points', xytext=(0, 5),
                            ha='center', va='bottom')
            ax.set_title(f'seed {label}', fontsize=9)
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel('Accuracy', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(ACC_YLIM)

        fig.suptitle(f'{title} - Per-rule Accuracy', fontsize=12)

    out = save_path or ('per_rule.png' if single_mode else f'{title}_per_rule.png')
    fig.savefig(out, dpi=FIG_DPI)
    print(f"Saved per-rule curve to {out}")
    plt.close(fig)


def collect_log_groups(names, log_dir):
    """Collect log files and group them by experiment setting.

    Returns a dict mapping setting name to a list of (seed_label, log_path).
    Names that do not contain a seed suffix are treated as standalone
    experiments and grouped under themselves with seed label ''.
    """
    groups = {}
    for name in names:
        setting, seed = extract_setting_and_seed(name)
        log_path = os.path.join(log_dir, f'{name}.log')

        if os.path.exists(log_path):
            groups.setdefault(setting, []).append((seed or '', log_path))
            continue

        if seed is not None:
            # The name is a full experiment with a seed suffix but its log
            # is missing; this is common when using --all with partial runs,
            # so skip silently.
            continue

        # Treat name as a setting prefix and collect matching seeds.
        pattern = os.path.join(log_dir, f'{name}_seed*.log')
        found = glob.glob(pattern)
        if found:
            for lp in found:
                stem = os.path.splitext(os.path.basename(lp))[0]
                setting, seed = extract_setting_and_seed(stem)
                groups.setdefault(setting, []).append((seed, lp))
            continue

        print(f"Log not found: {log_path}, skipping.")
    return groups


def plot_setting_group(setting, seed_data_items, out_dir,
                       no_per_pos=False, no_per_rule=False):
    """Plot all seeds of one experiment setting on shared figures.

    Parameters
    ----------
    setting : str
        Experiment setting name (without seed suffix).
    seed_data_items : list of (seed_label, data_dict)
        Parsed data for each seed, sorted as desired.
    out_dir : str
        Directory where the PNGs are saved.
    no_per_pos, no_per_rule : bool
        Skip corresponding plot types.
    """
    data_items = [(seed, d) for seed, d in seed_data_items if d['epochs']]
    if not data_items:
        print(f"No epoch data for setting {setting}, skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, setting)
    plot_learning_curve(data_items, setting, save_path=f'{base}_curve.png')
    if not no_per_pos:
        plot_per_position(data_items, setting, save_path=f'{base}_per_pos.png')
    if not no_per_rule:
        plot_per_rule(data_items, setting, save_path=f'{base}_per_rule.png')


def _resolve_inputs(args):
    """Resolve (names, log_dir, out_dir) from CLI args.

    names comes from --all (experiments.json), positional args, or as a
    fallback every .log in log_dir. log_dir/out_dir default to
    <base_dir>/experiments/{logs,plots} for --all, else ./logs and ./plots.
    """
    log_dir = args.log_dir
    out_dir = args.out_dir

    names = args.names
    if args.all:
        exp_path = 'experiments/experiments.json'
        if os.path.exists(exp_path):
            # The experiments file lives in experiments/, so its output dirs
            # are fixed as <base>/experiments/{logs,plots}.
            default_log = os.path.join(args.base_dir, 'experiments', 'logs')
            default_out = os.path.join(args.base_dir, 'experiments', 'plots')
            log_dir = log_dir or default_log
            out_dir = out_dir or default_out
            with open(exp_path, 'r', encoding='utf-8') as f:
                names = [e['name'] for e in json.load(f)['experiments']]
        else:
            # fallback: all logs in the explicit/default log dir
            log_dir = log_dir or 'logs'
            out_dir = out_dir or 'plots'
            names = [Path(p).stem for p in glob.glob(os.path.join(log_dir, '*.log'))]
    else:
        log_dir = log_dir or 'logs'
        out_dir = out_dir or 'plots'
        if not names:
            # Default: visualize all logs found in log_dir.
            names = [Path(p).stem for p in glob.glob(os.path.join(log_dir, '*.log'))]

    return names, log_dir, out_dir


def main():
    parser = argparse.ArgumentParser(description='Visualize learning curves from training logs.')
    parser.add_argument('names', nargs='*', help='Experiment name(s) or setting prefix(es).')
    parser.add_argument('--all', action='store_true', help='Visualize all experiments in experiments.json.')
    parser.add_argument('--base-dir', default='/data/cxm/recursion', help='Base output directory (default: /data/cxm/recursion).')
    parser.add_argument('--log-dir', default=None, help='Directory containing .log files.')
    parser.add_argument('--out-dir', default=None, help='Directory to save plots.')
    parser.add_argument('--no-per-pos', action='store_true', help='Skip per-position plot.')
    parser.add_argument('--no-per-rule', action='store_true', help='Skip per-rule plot.')
    parser.add_argument('--no-group', action='store_true', help='Plot each log separately (do not group seeds).')
    args = parser.parse_args()

    names, log_dir, out_dir = _resolve_inputs(args)

    os.makedirs(out_dir, exist_ok=True)

    if not names:
        print(f"No experiment names or log files found in {log_dir}.")
        sys.exit(1)

    groups = collect_log_groups(names, log_dir)

    if args.no_group:
        # Legacy behaviour: one figure per log file.
        for setting, items in groups.items():
            for seed, log_path in sorted(items, key=seed_sort_key):
                name = f'{setting}_seed{seed}' if seed else setting
                data = parse_log(log_path)
                if not data['epochs']:
                    print(f"No epoch data found in {log_path}, skipping.")
                    continue
                base = os.path.join(out_dir, name)
                plot_learning_curve(data, name, save_path=f'{base}_curve.png')
                if not args.no_per_pos:
                    plot_per_position(data, name, save_path=f'{base}_per_pos.png')
                if not args.no_per_rule:
                    plot_per_rule(data, name, save_path=f'{base}_per_rule.png')
    else:
        # New behaviour: group by setting, plot all seeds on the same figure.
        for setting in sorted(groups.keys()):
            items = groups[setting]
            data_items = []
            for seed, log_path in sorted(items, key=seed_sort_key):
                data = parse_log(log_path)
                if not data['epochs']:
                    print(f"No epoch data found in {log_path}, skipping.")
                    continue
                data_items.append((seed, data))
            plot_setting_group(setting, data_items, out_dir,
                               no_per_pos=args.no_per_pos,
                               no_per_rule=args.no_per_rule)


if __name__ == '__main__':
    main()
