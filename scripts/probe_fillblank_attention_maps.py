#!/usr/bin/env python3
"""Len5 fill_blank: per-blank, per-layer, per-head full attention maps."""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from analyze_attention import _forward_per_layer  # noqa: E402
from probe_fillblank_attention import AttentionCapture  # noqa: E402
from probe_fillblank_last_blank_linear import load_model  # noqa: E402
from probe_fillblank_logit_lens import (  # noqa: E402
    DEFAULT_BASE_DATA,
    DEFAULT_CKPT,
    _validate_loaded_model,
    make_blank_inputs,
)

SEQ_LEN = 5
INPUT_LEN = 6
EOS_POSITION = 5
N_BASES = 32
N_LAYERS = 4
N_HEADS = 4
REPLAY_RTOL = 1e-5
REPLAY_ATOL = 1e-6
ATTENTION_ATOL = 1e-6
DEFAULT_OUT = 'reports/data/fillblank_len5_attention_a25b56.json'
DEFAULT_PLOT_PREFIX = 'reports/circuit_analysis/plots/fillblank_len5_attention_a25b56'


def token_labels(k):
    """Input-slot labels with the actual blank location and final EOS."""
    if not 0 <= k < SEQ_LEN:
        raise ValueError(f'k must be in [0, {SEQ_LEN}), got {k}')
    return [f'BLANK(x{k})' if position == k else f'x{position}'
            for position in range(SEQ_LEN)] + ['EOS']


def mean_over_batch(attention):
    """Average only B from (B,H,T,T); never average or otherwise mix heads."""
    if attention.ndim != 4 or attention.shape[0] == 0:
        raise ValueError(f'expected nonempty (B,H,T,T), got {tuple(attention.shape)}')
    return attention.mean(dim=0)


def validate_attention(attention, atol=ATTENTION_ATOL):
    """Fail unless every causal row sums to one, future mass is zero, and q0=self."""
    if attention.ndim < 2 or attention.shape[-1] != attention.shape[-2]:
        raise ValueError(f'expected (...,T,T) attention, got {tuple(attention.shape)}')
    if attention.shape[-1] == 0 or not bool(torch.isfinite(attention).all()):
        raise RuntimeError('attention is empty or contains non-finite values')
    size = attention.shape[-1]
    upper_rows, upper_cols = torch.triu_indices(size, size, offset=1)
    upper = attention[..., upper_rows, upper_cols]
    metrics = {
        'shape': list(attention.shape),
        'atol': atol,
        'row_sum_max_abs_error': (attention.sum(dim=-1) - 1).abs().max().item(),
        'upper_triangle_max_abs': upper.abs().max().item() if upper.numel() else 0.0,
        'query0_self_max_abs_error': (attention[..., 0, 0] - 1).abs().max().item(),
    }
    metrics['row_sum_passed'] = metrics['row_sum_max_abs_error'] <= atol
    metrics['upper_triangle_passed'] = metrics['upper_triangle_max_abs'] <= atol
    metrics['query0_self_passed'] = metrics['query0_self_max_abs_error'] <= atol
    metrics['all_passed'] = all(metrics[name] for name in (
        'row_sum_passed', 'upper_triangle_passed', 'query0_self_passed'))
    if not metrics['all_passed']:
        raise RuntimeError(f'causal attention validation failed: {metrics}')
    return metrics


def validate_four_heads(model, cfg):
    """The shared Logit Lens config validator intentionally does not check heads."""
    if cfg.get('n_head') != N_HEADS:
        raise ValueError(f'unexpected checkpoint config n_head={cfg.get("n_head")!r}')
    for layer, block in enumerate(model.transformer.h):
        if block.attn.n_head != N_HEADS:
            raise ValueError(f'layer {layer} has n_head={block.attn.n_head}, expected 4')


@torch.inference_mode()
def capture_batch(model, inputs):
    """Run the original model forward and independently replay every attention layer."""
    cap = AttentionCapture(model)
    try:
        cap.records = []
        logits, _ = model(inputs)
        hooked = [record[0] for record in cap.records]
        expected_layers = len(model.transformer.h)
        if len(hooked) != expected_layers:
            raise RuntimeError(f'captured {len(hooked)} layers, expected {expected_layers}')
        replayed = _forward_per_layer(model, inputs)
        if len(replayed) != expected_layers:
            raise RuntimeError(f'replayed {len(replayed)} layers, expected {expected_layers}')

        layer_checks = []
        max_abs_diff = 0.0
        n_elements = 0
        for layer, (hook_att, replay) in enumerate(zip(hooked, replayed)):
            replay_att = replay['attn_weights'].detach().cpu()
            if hook_att.shape != replay_att.shape:
                raise RuntimeError(
                    f'layer {layer} attention shape {tuple(hook_att.shape)} != '
                    f'{tuple(replay_att.shape)}')
            if hook_att.shape[-2:] != (inputs.shape[1], inputs.shape[1]):
                raise RuntimeError(f'layer {layer} does not contain a full TxT attention map')
            allclose = torch.allclose(
                hook_att, replay_att, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
            layer_max_diff = (hook_att - replay_att).abs().max().item()
            causal = validate_attention(hook_att)
            layer_checks.append({
                'layer_index': layer,
                'allclose': bool(allclose),
                'max_abs_diff': layer_max_diff,
                'n_elements_compared': hook_att.numel(),
                'causal': causal,
            })
            max_abs_diff = max(max_abs_diff, layer_max_diff)
            n_elements += hook_att.numel()
            if not allclose:
                raise RuntimeError(
                    f'hook/replay attention mismatch at layer {layer}: '
                    f'max_abs_diff={layer_max_diff}')
        validation = {
            'rtol': REPLAY_RTOL,
            'atol': REPLAY_ATOL,
            'allclose': True,
            'max_abs_diff': max_abs_diff,
            'n_elements_compared': n_elements,
            'layers': layer_checks,
        }
        return logits, hooked, validation
    finally:
        cap.close()


@torch.inference_mode()
def analyze(model, base_data, device, batch_size):
    """Compute one (layer,head,query,key) batch mean for each blank position."""
    by_blank = {}
    global_max_diff = 0.0
    global_elements = 0
    global_eos = 0
    for k in range(SEQ_LEN):
        entry = base_data['by_blank'][f'x{k}']
        windows = torch.tensor(entry['base_clean_windows'], dtype=torch.long)
        base_outputs = torch.tensor(entry['base_outputs'], dtype=torch.long)
        if windows.shape != (N_BASES, SEQ_LEN):
            raise ValueError(f'x{k} base_clean_windows must have shape (32,5)')
        if base_outputs.shape != (N_BASES,):
            raise ValueError(f'x{k} base_outputs must contain exactly 32 predictions')
        state_indices = entry.get('base_state_indices')
        if state_indices is None or len(state_indices) != N_BASES:
            raise ValueError(f'x{k} base_state_indices must contain exactly 32 entries')

        inputs = make_blank_inputs(
            windows, k,
            blank_token=model.transformer.wte.num_embeddings - 2,
            eos_token=model.transformer.wte.num_embeddings - 1,
        )
        if inputs.shape != (N_BASES, INPUT_LEN):
            raise RuntimeError(f'unexpected input shape {tuple(inputs.shape)}')
        layer_batches = [[] for _ in range(N_LAYERS)]
        batch_checks = []
        for start in range(0, N_BASES, batch_size):
            end = min(start + batch_size, N_BASES)
            batch_inputs = inputs[start:end].to(device)
            logits, hooked, check = capture_batch(model, batch_inputs)
            if len(hooked) != N_LAYERS or any(
                    att.shape != (end - start, N_HEADS, INPUT_LEN, INPUT_LEN)
                    for att in hooked):
                raise RuntimeError(f'x{k} batch {start}:{end} has invalid attention shapes')
            predictions = logits[:, EOS_POSITION, :].argmax(dim=-1).detach().cpu()
            expected = base_outputs[start:end]
            matches = predictions == expected
            if not bool(matches.all()):
                mismatches = [{
                    'base_row': start + row,
                    'expected': int(expected[row]),
                    'model_prediction': int(predictions[row]),
                } for row in range(end - start) if not bool(matches[row])]
                raise RuntimeError(
                    f'EOS predictions differ from stored base_outputs at x{k}: {mismatches}')
            global_eos += end - start
            for layer, att in enumerate(hooked):
                layer_batches[layer].append(att)
            batch_checks.append({'batch_start': start, 'batch_end': end, **check})
            global_max_diff = max(global_max_diff, check['max_abs_diff'])
            global_elements += check['n_elements_compared']

        mean_attention = torch.stack([
            mean_over_batch(torch.cat(batches, dim=0)) for batches in layer_batches])
        expected_shape = (N_LAYERS, N_HEADS, INPUT_LEN, INPUT_LEN)
        if mean_attention.shape != expected_shape:
            raise RuntimeError(
                f'x{k} mean shape {tuple(mean_attention.shape)} != {expected_shape}')
        mean_causal = validate_attention(mean_attention)
        by_blank[f'x{k}'] = {
            'k': k,
            'N': N_BASES,
            'token_labels': token_labels(k),
            'base_state_indices': state_indices,
            'base_clean_windows': windows.tolist(),
            'base_outputs': base_outputs.tolist(),
            'attention_layout': '[layer][head][query][key]',
            'attention_mean': mean_attention.tolist(),
            'validation': {
                'hook_vs_independent_forward': {
                    'rtol': REPLAY_RTOL,
                    'atol': REPLAY_ATOL,
                    'allclose': all(check['allclose'] for check in batch_checks),
                    'max_abs_diff': max(check['max_abs_diff'] for check in batch_checks),
                    'n_elements_compared': sum(
                        check['n_elements_compared'] for check in batch_checks),
                    'batches': batch_checks,
                },
                'eos_predictions': {
                    'match_base_outputs': True,
                    'n_compared': N_BASES,
                    'mismatches': [],
                },
                'mean_attention_causal': mean_causal,
            },
        }
        print(f'  x{k}: N=32, attention=(4,4,6,6), '
              f'replay max_abs_diff={by_blank[f"x{k}"]["validation"]["hook_vs_independent_forward"]["max_abs_diff"]:.3g}, '
              'EOS predictions and causal checks passed')

    validation = {
        'all_checks_passed': True,
        'hook_vs_independent_forward': {
            'rtol': REPLAY_RTOL,
            'atol': REPLAY_ATOL,
            'allclose': True,
            'max_abs_diff': global_max_diff,
            'n_elements_compared': global_elements,
        },
        'eos_predictions_match_base_outputs': True,
        'n_eos_predictions_compared': global_eos,
        'causal_checks_passed': True,
        'causal_atol': ATTENTION_ATOL,
        'n_blank_groups': len(by_blank),
    }
    return by_blank, validation


def make_attention_figure(attention, k, n_bases=N_BASES):
    """Build a 4x4 panel figure; each panel retains a complete 6x6 matrix."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    attention = torch.as_tensor(attention, dtype=torch.float32)
    expected_shape = (N_LAYERS, N_HEADS, INPUT_LEN, INPUT_LEN)
    if attention.shape != expected_shape:
        raise ValueError(f'expected {expected_shape}, got {tuple(attention.shape)}')
    labels = token_labels(k)
    fig, axes = plt.subplots(
        N_LAYERS, N_HEADS, figsize=(20, 18), squeeze=False, constrained_layout=True)
    image = None
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            ax = axes[layer, head]
            image = ax.imshow(
                attention[layer, head].numpy(), cmap='viridis',
                vmin=0.0, vmax=1.0, origin='upper', aspect='equal')
            ax.set_title(f'Layer {layer + 1}, Head {head + 1}')
            ax.set_xlabel('Key token')
            ax.set_ylabel('Query token')
            ax.set_xticks(range(INPUT_LEN), labels, rotation=45, ha='right')
            ax.set_yticks(range(INPUT_LEN), labels)
    fig.colorbar(image, ax=list(axes.flat), shrink=0.8, pad=0.02,
                 label='Mean attention over batch')
    fig.suptitle(
        f'Len5 fill_blank Attention: BLANK at x{k} (N={n_bases} held-out bases)',
        fontsize=17)
    return fig


def make_plots(by_blank, path_prefix):
    """Write exactly five PNGs, one for each actual BLANK position."""
    import matplotlib.pyplot as plt

    paths = []
    for k in range(SEQ_LEN):
        entry = by_blank[f'x{k}']
        fig = make_attention_figure(entry['attention_mean'], k, entry['N'])
        path = f'{path_prefix}_blank_x{k}.png'
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            fig.savefig(path, dpi=160)
        finally:
            plt.close(fig)
        paths.append(path)
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--base-data', default=DEFAULT_BASE_DATA)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--plot-prefix', default=DEFAULT_PLOT_PREFIX)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dry-run', action='store_true',
                        help='print the fixed plan without loading data or a checkpoint')
    return parser.parse_args(argv)


def print_dry_run(args):
    print('DRY RUN: len5 fill_blank full attention maps (no data or model loaded)')
    print(f'  ckpt: {args.ckpt}')
    print(f'  base data: {args.base_data}')
    print(f'  JSON: {args.out}')
    print(f'  device={args.device}, threads={args.threads}, batch_size={args.batch_size}')
    print('  Per blank: N=32; input length 6; BLANK at xK plus EOS.')
    print('  Mean over batch only; output shape [layer][head][query][key]=4x4x6x6.')
    print(f'  Hook/replay allclose: rtol={REPLAY_RTOL}, atol={REPLAY_ATOL}.')
    print(f'  Row-sum/future/query0 checks: atol={ATTENTION_ATOL}; failures stop the run.')
    for k in range(SEQ_LEN):
        print(f'  x{k}: labels={token_labels(k)}; '
              f'plot={args.plot_prefix}_blank_x{k}.png')


def main(argv=None):
    args = parse_args(argv)
    if args.threads <= 0 or args.batch_size <= 0:
        raise ValueError('--threads and --batch-size must be positive')
    if args.dry_run:
        print_dry_run(args)
        return

    torch.set_num_threads(args.threads)
    print('[1/4] Loading held-out base data and shared checkpoint')
    with open(args.base_data) as handle:
        base_data = json.load(handle)
    model, cfg, ckpt = load_model(args.ckpt, args.device)
    _validate_loaded_model(model, cfg)
    validate_four_heads(model, cfg)
    print(f'  ckpt: {os.path.abspath(args.ckpt)}')
    print(f'  base data: {os.path.abspath(args.base_data)}')
    print(f'  model: d_model={cfg["d_model"]}, layers={cfg["n_layer"]}, '
          f'heads={cfg["n_head"]}, P={cfg["p"]}, vocab={cfg["vocab_size"]}')
    print(f'  device={args.device}, threads={torch.get_num_threads()}, '
          f'batch_size={args.batch_size}')

    print('[2/4] Capturing original-forward attention and independent replays')
    by_blank, validation = analyze(model, base_data, args.device, args.batch_size)

    print('[3/4] Drawing five 4-layer x 4-head full-attention figures')
    plot_paths = make_plots(by_blank, args.plot_prefix)
    for path in plot_paths:
        print(f'  PNG: {path}')

    print('[4/4] Writing matrices and all validation records')
    report = {
        'config': {
            'ckpt': os.path.abspath(args.ckpt),
            'base_data': os.path.abspath(args.base_data),
            'model_config': cfg,
            'checkpoint_final_epoch': ckpt.get('final_epoch'),
            'checkpoint_best_accuracy': ckpt.get('best_accuracy'),
            'device': args.device,
            'threads': args.threads,
            'batch_size': args.batch_size,
            'seq_len': SEQ_LEN,
            'input_len': INPUT_LEN,
            'eos_position': EOS_POSITION,
            'n_bases_per_blank': N_BASES,
            'n_layers': N_LAYERS,
            'n_heads': N_HEADS,
            'aggregation': 'arithmetic mean over batch dimension only; heads are never averaged',
            'attention_layout': '[layer][head][query][key]',
            'replay_rtol': REPLAY_RTOL,
            'replay_atol': REPLAY_ATOL,
            'causal_atol': ATTENTION_ATOL,
            'plot_vmin': 0.0,
            'plot_vmax': 1.0,
            'plot_cmap': 'viridis',
        },
        'by_blank': by_blank,
        'validation': validation,
        'plots': [os.path.abspath(path) for path in plot_paths],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f'  JSON: {args.out}')
    print(f'Done. all_checks_passed={validation["all_checks_passed"]}; '
          f'replay max_abs_diff={validation["hook_vs_independent_forward"]["max_abs_diff"]:.3g}; '
          f'EOS matches={validation["n_eos_predictions_compared"]}/160.')


if __name__ == '__main__':
    main()
