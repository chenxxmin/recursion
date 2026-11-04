#!/usr/bin/env python3
"""len5 fill_blank Logit Lens：逐层观察各输入槽对目标 xK 的预测。"""
import argparse
import json
import os
import statistics

import torch

from probe_fillblank_last_blank_linear import load_model

SEQ_LEN = 5
EOS_POSITION = 5
N_BASES = 32
N_LAYERS = 4
DEFAULT_CKPT = (
    '/data/cxm/models/variant_o2_fillblank_noaux_randpos_d512l4_P127_len3to5_wd2_ckpt/'
    'single_o2_fillblank_a25b56_d512l4r8h4_P127_len3to5_randpos_noaux_'
    'lr0.0003_wd2_ckpt_seed42_best.pth'
)
DEFAULT_BASE_DATA = 'reports/data/fillblank_len5_all_blank_linear_a25b56.json'
DEFAULT_OUT = 'reports/data/fillblank_len5_logit_lens_a25b56.json'
DEFAULT_PLOT = 'plots/fillblank_len5_logit_lens_a25b56.png'
CRITERIA = {
    'primary': {'top1_rate_min': 0.90, 'mean_prob_min': 0.50},
    'strict': {'top1_rate_min': 0.95, 'mean_prob_min': 0.90},
}


def legal_positions(k, seq_len=SEQ_LEN):
    """返回该空格位置合法的输入槽，普通位置升序，EOS 最后。"""
    if not 0 <= k < seq_len:
        raise ValueError(f'k must be in [0, {seq_len}), got {k}')
    positions = list(range(max(0, k - 2), min(seq_len - 1, k + 2) + 1))
    return positions + [seq_len]


def position_label(position):
    return 'EOS' if position == EOS_POSITION else f'x{position}'


def build_plan(n_bases=N_BASES, n_layers=N_LAYERS):
    plan = []
    for k in range(SEQ_LEN):
        positions = legal_positions(k)
        n_cells = n_layers * len(positions)
        plan.append({
            'k': k,
            'target': f'x{k}',
            'positions': positions,
            'position_labels': [position_label(j) for j in positions],
            'n_bases': n_bases,
            'n_layers': n_layers,
            'n_cells': n_cells,
            'n_raw_records': n_bases * n_cells,
        })
    return plan


def plan_totals(plan):
    return {
        'n_blanks': len(plan),
        'n_layers': plan[0]['n_layers'],
        'n_bases': plan[0]['n_bases'],
        'n_cells': sum(item['n_cells'] for item in plan),
        'n_raw_records': sum(item['n_raw_records'] for item in plan),
    }


def make_blank_inputs(windows, k, blank_token=127, eos_token=128):
    """干净 len5 窗口 -> 第 k 位 BLANK、末尾追加 EOS；target 仍为原 xK。"""
    inputs = windows.clone()
    inputs[:, k] = blank_token
    eos = torch.full((len(inputs), 1), eos_token, dtype=inputs.dtype)
    return torch.cat([inputs, eos], dim=1)


def target_metrics(logits, targets):
    """full-vocab target prob/logit、严格大于口径 rank、target-max(other) margin。"""
    probs = torch.softmax(logits, dim=-1)
    target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    ranks = 1 + (logits > target_logits.unsqueeze(1)).sum(dim=1)
    other_logits = logits.clone()
    other_logits.scatter_(1, targets.unsqueeze(1), -torch.inf)
    margins = target_logits - other_logits.max(dim=1).values
    return {
        'prob': target_probs,
        'logit': target_logits,
        'rank': ranks,
        'margin': margins,
    }


def aggregate_records(records):
    if not records:
        raise ValueError('cannot aggregate an empty cell')
    probs = [row['prob'] for row in records]
    ranks = [row['rank'] for row in records]
    margins = [row['margin'] for row in records]
    return {
        'k': records[0]['k'],
        'layer_index': records[0]['layer_index'],
        'position': records[0]['position'],
        'n': len(records),
        'mean_prob': sum(probs) / len(probs),
        'median_prob': statistics.median(probs),
        'top1_rate': sum(rank == 1 for rank in ranks) / len(ranks),
        'mean_margin': sum(margins) / len(margins),
        'mean_rank': sum(ranks) / len(ranks),
    }


def aggregate_raw(raw):
    grouped = {}
    for row in raw:
        key = (row['k'], row['layer_index'], row['position'])
        grouped.setdefault(key, []).append(row)
    return [aggregate_records(records) for records in grouped.values()]


def is_clear(cell, criterion):
    thresholds = CRITERIA[criterion] if isinstance(criterion, str) else criterion
    return (cell['top1_rate'] >= thresholds['top1_rate_min']
            and cell['mean_prob'] >= thresholds['mean_prob_min'])


def clear_summary(aggregates, k, criterion):
    """层优先、位置次序次之；另报所有 clear cells 中 mean_prob 最高者。"""
    position_order = {position: i for i, position in enumerate(legal_positions(k))}
    cells = [cell for cell in aggregates if cell['k'] == k]
    cells.sort(key=lambda cell: (cell['layer_index'], position_order[cell['position']]))
    clear_cells = [dict(cell) for cell in cells if is_clear(cell, criterion)]
    earliest = dict(clear_cells[0]) if clear_cells else None
    highest = (dict(max(clear_cells, key=lambda cell: cell['mean_prob']))
               if clear_cells else None)
    return {
        'earliest': earliest,
        'highest_mean_prob': highest,
        'clear_cells': clear_cells,
    }


def build_earliest(aggregates):
    return {
        criterion: {f'x{k}': clear_summary(aggregates, k, criterion)
                    for k in range(SEQ_LEN)}
        for criterion in CRITERIA
    }


@torch.inference_mode()
def probe_batch(model, inputs, targets, k, positions):
    """手动 wte/drop/block forward，并在每个 block 后应用 ln_f+lm_head。"""
    hidden = model.transformer.drop(model.transformer.wte(inputs))
    raw = []
    final_eos_logits = None
    for layer_index, block in enumerate(model.transformer.h):
        hidden = block(hidden)
        logits = model.lm_head(model.transformer.ln_f(hidden))
        for position in positions:
            metrics = target_metrics(logits[:, position, :], targets)
            metrics = {name: value.cpu() for name, value in metrics.items()}
            for sample_index in range(len(inputs)):
                raw.append({
                    'k': k,
                    'layer_index': layer_index,
                    'position': position,
                    'prob': float(metrics['prob'][sample_index]),
                    'logit': float(metrics['logit'][sample_index]),
                    'rank': int(metrics['rank'][sample_index]),
                    'margin': float(metrics['margin'][sample_index]),
                })
        if layer_index == len(model.transformer.h) - 1:
            final_eos_logits = logits[:, EOS_POSITION, :]
    reference_logits, _ = model(inputs)
    return raw, final_eos_logits, reference_logits[:, EOS_POSITION, :]


def _validate_loaded_model(model, cfg):
    expected = {
        'p': 127, 'd_model': 512, 'n_layer': N_LAYERS, 'vocab_size': 129,
        'blank_token_id': 127, 'eos_token_id': 128,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'unexpected checkpoint config {key}={cfg.get(key)!r}, expected {value}')
    if len(model.transformer.h) != N_LAYERS:
        raise ValueError('checkpoint must have exactly four transformer blocks')


@torch.inference_mode()
def analyze(model, base_data, device, batch_size):
    raw = []
    validation = {
        'rtol': 1e-5,
        'atol': 1e-6,
        'allclose': True,
        'max_abs_diff': 0.0,
        'n_batches': 0,
        'n_eos_logits_compared': 0,
        'model_eos_predictions_match_base_outputs': True,
        'manual_eos_predictions_match_base_outputs': True,
        'base_output_mismatches': [],
    }
    for item in build_plan():
        k = item['k']
        entry = base_data['by_blank'][f'x{k}']
        windows = torch.tensor(entry['base_clean_windows'], dtype=torch.long)
        if windows.shape != (N_BASES, SEQ_LEN):
            raise ValueError(f'x{k} base_clean_windows must have shape (32, 5)')
        base_outputs = torch.tensor(entry['base_outputs'], dtype=torch.long)
        if base_outputs.shape != (N_BASES,):
            raise ValueError(f'x{k} base_outputs must contain exactly 32 predictions')
        inputs = make_blank_inputs(
            windows, k,
            blank_token=model.transformer.wte.num_embeddings - 2,
            eos_token=model.transformer.wte.num_embeddings - 1,
        )
        targets = windows[:, k]
        for start in range(0, N_BASES, batch_size):
            end = min(start + batch_size, N_BASES)
            batch_inputs = inputs[start:end].to(device)
            batch_targets = targets[start:end].to(device)
            batch_raw, manual_eos, reference_eos = probe_batch(
                model, batch_inputs, batch_targets, k, item['positions'])
            raw.extend(batch_raw)

            allclose = torch.allclose(manual_eos, reference_eos, rtol=1e-5, atol=1e-6)
            max_abs_diff = (manual_eos - reference_eos).abs().max().item()
            validation['n_batches'] += 1
            validation['n_eos_logits_compared'] += reference_eos.numel()
            validation['max_abs_diff'] = max(validation['max_abs_diff'], max_abs_diff)
            validation['allclose'] = validation['allclose'] and allclose
            if not allclose:
                raise RuntimeError(
                    f'final EOS logits differ from model forward at k={k}, '
                    f'batch={start}:{end}, max_abs_diff={max_abs_diff}')

            expected = base_outputs[start:end]
            model_prediction = reference_eos.argmax(dim=-1).cpu()
            manual_prediction = manual_eos.argmax(dim=-1).cpu()
            model_matches = model_prediction == expected
            manual_matches = manual_prediction == expected
            validation['model_eos_predictions_match_base_outputs'] &= bool(model_matches.all())
            validation['manual_eos_predictions_match_base_outputs'] &= bool(manual_matches.all())
            for row in range(end - start):
                if not (bool(model_matches[row]) and bool(manual_matches[row])):
                    validation['base_output_mismatches'].append({
                        'k': k,
                        'base_row': start + row,
                        'expected': int(expected[row]),
                        'model_prediction': int(model_prediction[row]),
                        'manual_prediction': int(manual_prediction[row]),
                    })
        print(f'  x{k}: {len(item["positions"])} positions x 4 layers x 32 bases complete')
    if validation['base_output_mismatches']:
        raise RuntimeError(
            f'EOS predictions disagree with stored base_outputs for '
            f'{len(validation["base_output_mismatches"])} samples')
    aggregates = aggregate_raw(raw)
    expected_totals = plan_totals(build_plan())
    if len(raw) != expected_totals['n_raw_records']:
        raise AssertionError(f'expected 3072 raw records, got {len(raw)}')
    if len(aggregates) != expected_totals['n_cells']:
        raise AssertionError(f'expected 96 aggregate cells, got {len(aggregates)}')
    return raw, aggregates, build_earliest(aggregates), validation


def make_plot(aggregates, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lookup = {(cell['k'], cell['layer_index'], cell['position']): cell
              for cell in aggregates}
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    active_axes = []
    image = None
    for k in range(SEQ_LEN):
        ax = axes.flat[k]
        active_axes.append(ax)
        positions = legal_positions(k)
        matrix = [[lookup[(k, layer, position)]['mean_prob'] for position in positions]
                  for layer in range(N_LAYERS)]
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap='viridis', aspect='auto')
        ax.set_title(f'Blank at x{k}; target x{k}')
        ax.set_xlabel('Input position')
        ax.set_ylabel('Layer')
        ax.set_xticks(range(len(positions)),
                      [position_label(position) for position in positions])
        ax.set_yticks(range(N_LAYERS), [f'L{layer + 1}' for layer in range(N_LAYERS)])
        for layer in range(N_LAYERS):
            for column, value in enumerate(matrix[layer]):
                ax.text(column, layer, f'{value:.3f}', ha='center', va='center',
                        color='white' if value < 0.55 else 'black', fontsize=9)
    axes.flat[-1].axis('off')
    fig.colorbar(image, ax=active_axes, shrink=0.9, pad=0.02,
                 label='Mean target probability')
    fig.suptitle('Len5 fill_blank Logit Lens (32 held-out bases per blank)', fontsize=15)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--base-data', default=DEFAULT_BASE_DATA)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--plot', default=DEFAULT_PLOT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印固定分析计划，不读取基底或加载模型')
    return parser.parse_args(argv)


def print_dry_run(args):
    plan = build_plan()
    totals = plan_totals(plan)
    print('DRY RUN: len5 fill_blank Logit Lens (no data or model loaded)')
    print(f'  ckpt: {args.ckpt}')
    print(f'  base data: {args.base_data}')
    print(f'  output: {args.out}')
    print(f'  plot: {args.plot}')
    print(f'  device={args.device}, threads={args.threads}, batch_size={args.batch_size}')
    print('  Each base: replace xK with BLANK(127), append EOS(128), target=xK.')
    for item in plan:
        labels = ', '.join(item['position_labels'])
        print(f'  k={item["k"]}: positions [{labels}]; '
              f'{item["n_cells"]} cells; {item["n_raw_records"]} raw records')
    print(f'  TOTAL: {totals["n_cells"]} aggregate cells; '
          f'{totals["n_raw_records"]} raw records; N={totals["n_bases"]} per cell')
    n_batches = sum((N_BASES + args.batch_size - 1) // args.batch_size for _ in plan)
    print(f'  Manual final-forward EOS validation batches: {n_batches}')
    print('  Criteria: primary top1>=0.90 & mean_prob>=0.50; '
          'strict top1>=0.95 & mean_prob>=0.90')


def main(argv=None):
    args = parse_args(argv)
    if args.threads <= 0 or args.batch_size <= 0:
        raise ValueError('--threads and --batch-size must be positive')
    if args.dry_run:
        print_dry_run(args)
        return

    torch.set_num_threads(args.threads)
    print('[1/4] Loading held-out base windows (no resampling)')
    with open(args.base_data) as handle:
        base_data = json.load(handle)
    print('[2/4] Loading checkpoint with probe_fillblank_last_blank_linear.load_model')
    model, cfg, ckpt = load_model(args.ckpt, args.device)
    _validate_loaded_model(model, cfg)
    print(f'  d_model={cfg["d_model"]}, layers={cfg["n_layer"]}, '
          f'P={cfg["p"]}, vocab={cfg["vocab_size"]}, device={args.device}')

    print('[3/4] Running manual block-by-block Logit Lens')
    raw, aggregates, earliest, validation = analyze(
        model, base_data, args.device, args.batch_size)
    report = {
        'config': {
            'ckpt': os.path.abspath(args.ckpt),
            'base_data': os.path.abspath(args.base_data),
            'device': args.device,
            'threads': args.threads,
            'batch_size': args.batch_size,
            'seq_len': SEQ_LEN,
            'eos_position': EOS_POSITION,
            'n_bases': N_BASES,
            'n_layers': N_LAYERS,
            'model_config': cfg,
            'checkpoint_final_epoch': ckpt.get('final_epoch'),
            'checkpoint_best_accuracy': ckpt.get('best_accuracy'),
            'plan': build_plan(),
        },
        'criteria': CRITERIA,
        'raw': raw,
        'aggregates': aggregates,
        'earliest': earliest,
        'final_forward_validation': validation,
    }
    parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(parent, exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(f'  JSON: {args.out} ({len(raw)} raw records, {len(aggregates)} cells)')

    print('[4/4] Drawing mean-probability heatmaps')
    make_plot(aggregates, args.plot)
    print(f'  Plot: {args.plot}')
    print(f'Done. Final EOS allclose={validation["allclose"]}; '
          f'max_abs_diff={validation["max_abs_diff"]:.3g}; base outputs match.')


if __name__ == '__main__':
    main()
