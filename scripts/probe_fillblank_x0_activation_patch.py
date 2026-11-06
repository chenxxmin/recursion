#!/usr/bin/env python3
"""Any-position len5 fill_blank clean->corrupted activation patching (a25b56).

For blank K, the clean input is the len5 window with xK replaced by BLANK and
EOS appended.  Each local F/M/R operand is shifted by +d mod P; clean residual
activations are patched at every token position 0..4 (including BLANK, excluding
EOS) and EOS is compared with the original windows[:,K].
"""
import argparse
import gzip
import json
import os
import statistics
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from activation_patch import capture_hidden, patched_logits  # noqa: E402
from probe_fillblank_logit_lens import (  # noqa: E402
    _validate_loaded_model, load_model, make_blank_inputs, target_metrics,
)

P = 127
N_BASES = 32
SEQ_LEN = 5
EOS_POSITION = 5
SITES = [('emb',), ('resid', 0), ('resid', 1), ('resid', 2), ('resid', 3)]
SITE_LABELS = [
    'before L1',
    'after L1 / before L2',
    'after L2 / before L3',
    'after L3 / before L4',
    'after L4 / before ln_f',
]
CONTROL_NAMES = ('emb_ceiling', 'emb_identity', 'causal_noop', 'final_resid_noop')
DEFAULT_CKPT = (
    '/data/cxm/models/variant_o2_fillblank_noaux_randpos_d512l4_P127_len3to5_wd2_ckpt/'
    'single_o2_fillblank_a25b56_d512l4r8h4_P127_len3to5_randpos_noaux_'
    'lr0.0003_wd2_ckpt_seed42_best.pth'
)
DEFAULT_BASE_DATA = 'reports/data/fillblank_len5_all_blank_linear_a25b56.json'
DEFAULT_OUT = 'reports/data/fillblank_x0_activation_patch_a25b56.json.gz'
DEFAULT_PLOT_PREFIX = (
    'reports/circuit_analysis/plots/fillblank_x0_activation_patch_a25b56')
# Clean B=32 and patch chunks can use different CPU GEMM reduction orders.
CONTROL_RTOL = 1e-5
CONTROL_ATOL = 1e-4


def operand_positions(k):
    """Union of operands in legal local routes for each len5 blank."""
    positions = {
        0: [1, 2],
        1: [0, 2, 3],
        2: [0, 1, 3, 4],
        3: [1, 2, 4],
        4: [2, 3],
    }
    if k not in positions:
        raise ValueError(f'blank must be in 0..{SEQ_LEN - 1}, got {k}')
    return positions[k]


def patch_positions(k):
    """Restore grid positions: all five tokens, including BLANK but not EOS."""
    operand_positions(k)  # shared 0..4 validation
    return list(range(SEQ_LEN))


def clean_input_label(k):
    labels = [f'x{j}' if j != k else 'BLANK' for j in range(SEQ_LEN)]
    return '[' + ','.join(labels + ['EOS']) + ']'


def site_label(site):
    site = tuple(site)
    if site not in SITES:
        raise ValueError(f'unknown residual site {site}')
    return SITE_LABELS[SITES.index(site)]


def position_label(position):
    return 'EOS' if position == EOS_POSITION else f'x{position}'


def patch_position_label(position, blank):
    return f'BLANK(x{blank})' if position == blank else position_label(position)


def _rule_coefficients(cfg):
    p = int(cfg['p'])
    rules = cfg['rules']
    if cfg.get('order') != 2 or len(rules) != 1 or len(rules[0]) != 2:
        raise ValueError('activation patching requires one order-2 rule')
    a, b = (int(value) for value in rules[0])
    a_inv = pow(a, -1, p)
    b_inv = pow(b, -1, p)
    return {
        'a': a, 'b': b, 'a_inv': a_inv, 'b_inv': b_inv,
        'neg_a_inv_b': (-a_inv * b) % p,
        'neg_b_inv_a': (-b_inv * a) % p,
    }


def route_slopes(cfg, blank=0):
    """Legal local routes -> per-source slopes, including explicit zero responses."""
    positions = operand_positions(blank)
    c = _rule_coefficients(cfg)
    candidates = {
        'F': ((blank - 2, c['b']), (blank - 1, c['a'])),
        'M': ((blank - 1, c['neg_a_inv_b']), (blank + 1, c['a_inv'])),
        'R': ((blank + 1, c['neg_b_inv_a']), (blank + 2, c['b_inv'])),
    }
    legal = {}
    for name, operands in candidates.items():
        if not all(0 <= position < SEQ_LEN for position, _ in operands):
            continue
        direct = dict(operands)
        legal[name] = {
            'operands': [{'position': position, 'coefficient': coefficient}
                         for position, coefficient in operands],
            'source_slopes': {position: direct.get(position, 0) for position in positions},
        }
    return legal


def route_definitions(cfg, blank=0):
    """JSON-ready complete local F/M/R definitions for report metadata."""
    legal = route_slopes(cfg, blank)
    return {
        'blank': blank,
        'modulus': int(cfg['p']),
        'coefficients': _rule_coefficients(cfg),
        'route_formulas': {
            'F': '(k-2,b),(k-1,a)',
            'M': "(k-1,-a'b),(k+1,a')",
            'R': "(k+1,-b'a),(k+2,b')",
        },
        'legal_routes': {
            name: {
                'operands': route['operands'],
                'source_slopes': {f'x{position}': slope for position, slope
                                  in route['source_slopes'].items()},
            }
            for name, route in legal.items()
        },
        'zero_response': 'A source absent from a legal route has slope 0 for that route.',
    }


def match_routes(D, d, full_output, source_position, routes, p=P):
    """Numeric-output modular route matches; multiple routes are allowed."""
    if int(full_output) >= p:
        return []
    return [
        name for name, route in routes.items()
        if int(D) == (route['source_slopes'][source_position] * int(d)) % p
    ]


def _as_long_tensor(values, device=None):
    return torch.as_tensor(values, dtype=torch.long, device=device)


def make_corrupted_inputs(clean_inputs, source_position, d_values, p=P):
    """Repeat one clean row over d values, changing exactly one input token."""
    if not 0 <= source_position < SEQ_LEN:
        raise ValueError(f'corruption source must be in 0..4, got {source_position}')
    deltas = _as_long_tensor(d_values, clean_inputs.device)
    if deltas.ndim != 1 or len(deltas) == 0:
        raise ValueError('d_values must be a nonempty one-dimensional sequence')
    if bool(((deltas <= 0) | (deltas >= p)).any()):
        raise ValueError(f'all d values must be in 1..{p - 1}')
    if len(clean_inputs) == 1:
        corrupted = clean_inputs.repeat(len(deltas), 1)
    elif len(clean_inputs) == len(deltas):
        corrupted = clean_inputs.clone()
    else:
        raise ValueError('clean_inputs must have one row or one row per d value')
    corrupted[:, source_position] = (
        corrupted[:, source_position] + deltas) % p
    return corrupted


def slice_clean_cache(clean_cache, base_row, batch_size):
    """Select one base and repeat it, preventing cross-base cache bleed."""
    if not 0 <= base_row < len(next(iter(clean_cache.values()))):
        raise IndexError(f'invalid base_row {base_row}')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    return {
        site: clean_cache[site][base_row:base_row + 1].repeat(batch_size, 1, 1)
        for site in SITES
    }


def prediction_metrics(eos_logits, targets, p=P, corrupted_full=None):
    """Full-vocab/numeric predictions and target metrics at the EOS slot."""
    full = eos_logits.argmax(dim=-1)
    numeric = eos_logits[:, :p].argmax(dim=-1)
    restore_full = full == targets
    restore_numeric = numeric == targets
    nonnumeric = full >= p
    if corrupted_full is None:
        persistence = torch.zeros_like(restore_full)
    else:
        persistence = (full == corrupted_full) & ~restore_full
    other = ~(restore_full | persistence | nonnumeric)
    result = {
        'full': full,
        'numeric': numeric,
        'restore_full': restore_full,
        'restore_numeric': restore_numeric,
        'corrupt_persistence': persistence,
        'nonnumeric': nonnumeric,
        'other': other,
    }
    result.update(target_metrics(eos_logits, targets))
    return result


def _bool_at(metrics, name, row):
    return bool(metrics[name][row])


def _raw_metric_fields(metrics, row):
    return {
        'target_prob': float(metrics['prob'][row]),
        'target_logit': float(metrics['logit'][row]),
        'target_rank': int(metrics['rank'][row]),
        'target_margin': float(metrics['margin'][row]),
    }


def _control_flags(site, patch_position, source_position):
    is_emb = tuple(site) == ('emb',)
    return {
        'emb_ceiling': is_emb and patch_position == source_position,
        'emb_identity': is_emb and patch_position != source_position,
        'causal_noop': source_position > patch_position,
        # After block 4, position-wise ln_f+lm_head cannot transmit a patch to EOS.
        'final_resid_noop': tuple(site) == ('resid', 3),
    }


def _check_control(name, patched, reference, targets=None):
    close = torch.allclose(patched, reference,
                           rtol=CONTROL_RTOL, atol=CONTROL_ATOL)
    max_abs_diff = float((patched - reference).abs().max())
    if not close:
        raise RuntimeError(f'{name} control failed: max_abs_diff={max_abs_diff}')
    if targets is not None:
        restored = patched[:, EOS_POSITION].argmax(dim=-1) == targets
        if not bool(restored.all()):
            raise RuntimeError('emb ceiling did not restore every original target')
    return max_abs_diff


def patch_chunk(model, corrupted_inputs, clean_cache, clean_logits,
                corrupted_logits, targets, base_row, d_values, source_position,
                route_exact, d_response, p=P, patch_positions=None,
                baseline_changed=None, blank=None):
    """Patch one base/d chunk at all sites and dynamic patch positions."""
    if patch_positions is None:
        patch_positions = list(range(SEQ_LEN))
    batch_size = len(corrupted_inputs)
    if blank is None:
        blank_slots = (corrupted_inputs[0, :SEQ_LEN] == p).nonzero().flatten()
        if len(blank_slots) != 1:
            raise ValueError('could not infer the unique BLANK position')
        blank = int(blank_slots[0])
    if baseline_changed is None:
        baseline_changed = (corrupted_logits[:, EOS_POSITION].argmax(dim=-1)
                            != targets).tolist()
    if not (len(targets) == len(d_values) == len(route_exact)
            == len(d_response) == len(baseline_changed) == batch_size):
        raise ValueError('clean/corrupt/patch rows are not aligned')
    source_cache = slice_clean_cache(clean_cache, base_row, batch_size)
    clean_reference = clean_logits[base_row:base_row + 1].repeat(batch_size, 1, 1)
    corrupted_full = corrupted_logits[:, EOS_POSITION].argmax(dim=-1)
    baseline_metrics = prediction_metrics(
        corrupted_logits[:, EOS_POSITION], targets, p, corrupted_full=None)
    raw = []
    control_max = {name: 0.0 for name in CONTROL_NAMES}
    for site in SITES:
        for patch_position in patch_positions:
            logits = patched_logits(
                model, corrupted_inputs, site, patch_position, source_cache)
            flags = _control_flags(site, patch_position, source_position)
            for name, active in flags.items():
                if not active:
                    continue
                reference = clean_reference if name == 'emb_ceiling' else corrupted_logits
                # Causal and final-residual controls are specifically EOS controls.
                eos_only = name in ('causal_noop', 'final_resid_noop')
                patched_control = logits[:, EOS_POSITION] if eos_only else logits
                reference_control = reference[:, EOS_POSITION] if eos_only else reference
                max_diff = _check_control(
                    name, patched_control, reference_control,
                    targets=targets if name == 'emb_ceiling' else None)
                control_max[name] = max(control_max[name], max_diff)
            metrics = prediction_metrics(
                logits[:, EOS_POSITION], targets, p, corrupted_full=corrupted_full)
            for row, d in enumerate(d_values):
                item = {
                    'base_row': int(base_row),
                    'd': int(d),
                    'source': f'x{source_position}',
                    'source_position': int(source_position),
                    'site': list(site),
                    'site_label': site_label(site),
                    'position': int(patch_position),
                    'position_label': patch_position_label(patch_position, blank),
                    'target': int(targets[row]),
                    'D': int(d_response[row]),
                    'route_exact': bool(route_exact[row]),
                    'baseline_changed': bool(baseline_changed[row]),
                    'corrupted_full_vocab_output': int(corrupted_full[row]),
                    'patched_full_vocab_output': int(metrics['full'][row]),
                    'patched_numeric_output': int(metrics['numeric'][row]),
                    'delta_logit': float(metrics['logit'][row]
                                         - baseline_metrics['logit'][row]),
                    'delta_margin': float(metrics['margin'][row]
                                          - baseline_metrics['margin'][row]),
                }
                item.update(_raw_metric_fields(metrics, row))
                for name in ('restore_full', 'restore_numeric', 'corrupt_persistence',
                             'nonnumeric', 'other'):
                    item[name] = _bool_at(metrics, name, row)
                item.update(flags)
                raw.append(item)
    return raw, control_max


def _rate(count, n):
    return count / n if n else None


def aggregate_records(records):
    """Exact raw counts plus rates/distributional metrics for one cell."""
    if not records:
        raise ValueError('cannot aggregate an empty cell')
    n = len(records)
    result = {
        'source': records[0]['source'],
        'source_position': records[0]['source_position'],
        'site': records[0]['site'],
        'site_label': records[0]['site_label'],
        'position': records[0]['position'],
        'position_label': records[0]['position_label'],
        'N': n,
        'n': n,
    }
    if 'base_row' in records[0] and len({r['base_row'] for r in records}) == 1:
        result['base_row'] = records[0]['base_row']
    for name in ('restore_full', 'corrupt_persistence', 'other', 'nonnumeric',
                 'restore_numeric'):
        count = sum(bool(row[name]) for row in records)
        result[f'{name}_count'] = count
        result[name] = _rate(count, n)
    result['persistence_count'] = result['corrupt_persistence_count']
    result['persistence'] = result['corrupt_persistence']

    changed = [row for row in records if row['baseline_changed']]
    unchanged = [row for row in records if not row['baseline_changed']]
    changed_count = sum(bool(row['restore_full']) for row in changed)
    preserved_count = sum(bool(row['restore_full']) for row in unchanged)
    result.update({
        'changed_N': len(changed),
        'changed_restore_full_count': changed_count,
        'changed_restore_full': _rate(changed_count, len(changed)),
        'unchanged_N': len(unchanged),
        'unchanged_preserved_count': preserved_count,
        'unchanged_preserved': _rate(preserved_count, len(unchanged)),
    })
    for name in ('prob', 'logit', 'margin'):
        values = [float(row[f'target_{name}']) for row in records]
        result[f'mean_{name}'] = sum(values) / n
        result[f'median_{name}'] = statistics.median(values)
    result['mean_delta_logit'] = sum(row['delta_logit'] for row in records) / n
    result['mean_delta_margin'] = sum(row['delta_margin'] for row in records) / n
    result['mean_rank'] = sum(row['target_rank'] for row in records) / n
    route = [row for row in records if row['route_exact']]
    route_n = len(route)
    route_count = sum(bool(row['restore_full']) for row in route)
    result['route_exact_N'] = route_n
    result['route_exact_restore_full_count'] = route_count
    result['route_exact_restore_full'] = _rate(route_count, route_n)
    result['controls'] = [name for name in CONTROL_NAMES
                          if any(row[name] for row in records)]
    return result


def aggregate_raw(raw, per_base=False):
    grouped = {}
    for row in raw:
        key = (row['source'], tuple(row['site']), row['position'])
        if per_base:
            key += (row['base_row'],)
        grouped.setdefault(key, []).append(row)
    return [aggregate_records(grouped[key]) for key in sorted(grouped)]


def _raw_record_index(raw):
    required = (
        'base_row', 'd', 'baseline_output', 'perturbed_numeric_output',
        'perturbed_full_vocab_output', 'D',
    )
    lengths = {len(raw[name]) for name in required}
    if len(lengths) != 1:
        raise ValueError('base raw_records columns have inconsistent lengths')
    index = {}
    for row in range(lengths.pop()):
        key = (int(raw['base_row'][row]), int(raw['d'][row]))
        if key in index:
            raise ValueError(f'duplicate base raw record {key}')
        index[key] = {name: int(raw[name][row]) for name in required}
    return index


def _assert_stored_outputs(computed, expected, description):
    computed = computed.detach().cpu()
    expected = _as_long_tensor(expected)
    if not torch.equal(computed, expected):
        mismatch = (computed != expected).nonzero().flatten()[0]
        raise RuntimeError(
            f'{description} mismatch at row {int(mismatch)}: '
            f'computed={int(computed[mismatch])}, stored={int(expected[mismatch])}')


def _baseline_rows(model, inputs, targets, base_row, d_values, source_position,
                   expected_rows, routes, p):
    logits, _ = model(inputs)
    eos_logits = logits[:, EOS_POSITION]
    metrics = prediction_metrics(eos_logits, targets, p)
    expected_full = [row['perturbed_full_vocab_output'] for row in expected_rows]
    expected_numeric = [row['perturbed_numeric_output'] for row in expected_rows]
    expected_baseline = [row['baseline_output'] for row in expected_rows]
    expected_d = [row['D'] for row in expected_rows]
    _assert_stored_outputs(metrics['full'], expected_full,
                           f'x{source_position} base={base_row} full-vocab output')
    _assert_stored_outputs(metrics['numeric'], expected_numeric,
                           f'x{source_position} base={base_row} numeric output')
    baseline_repeated = _as_long_tensor(expected_baseline, targets.device)
    computed_d = (metrics['numeric'] - baseline_repeated) % p
    _assert_stored_outputs(computed_d, expected_d,
                           f'x{source_position} base={base_row} D')
    matched = [
        match_routes(D, d, full, source_position, routes, p)
        for D, d, full in zip(expected_d, d_values, expected_full)
    ]
    route_exact = [bool(matches) for matches in matched]
    records = []
    for row, (d, expected) in enumerate(zip(d_values, expected_rows)):
        if expected['baseline_output'] != int(targets[row]):
            raise RuntimeError(
                f'stored baseline_output != original target at base={base_row}, d={d}')
        item = {
            'base_row': int(base_row),
            'd': int(d),
            'source': f'x{source_position}',
            'source_position': int(source_position),
            'target': int(targets[row]),
            'baseline_output': int(expected['baseline_output']),
            'perturbed_numeric_output': int(metrics['numeric'][row]),
            'perturbed_full_vocab_output': int(metrics['full'][row]),
            'D': int(computed_d[row]),
            'matched_routes': matched[row],
            'route_exact': bool(route_exact[row]),
            'baseline_changed': bool(metrics['full'][row] != targets[row]),
        }
        item.update(_raw_metric_fields(metrics, row))
        records.append(item)
    return logits, records, route_exact, computed_d


@torch.inference_mode()
def analyze(model, cfg, base_data, base_rows, d_values, batch_size, device, blank=0):
    """Validate every selected clean/corrupt baseline before any patching."""
    source_positions = operand_positions(blank)
    restore_positions = patch_positions(blank)
    entry = base_data['by_blank'][f'x{blank}']
    windows = _as_long_tensor(entry['base_clean_windows'])
    base_outputs = _as_long_tensor(entry['base_outputs'])
    if windows.shape != (N_BASES, SEQ_LEN) or base_outputs.shape != (N_BASES,):
        raise ValueError(f'by_blank.x{blank} must contain exactly 32 len5 bases and outputs')
    targets_all = windows[:, blank]  # Never read the target from a corrupted input.
    if not torch.equal(base_outputs, targets_all):
        raise ValueError(f'stored base_outputs do not equal windows[:,{blank}]')
    clean_inputs = make_blank_inputs(
        windows, blank, blank_token=cfg['blank_token_id'], eos_token=cfg['eos_token_id'])
    clean_inputs_device = clean_inputs.to(device)
    # This is the only clean forward/cache call for all 32 bases.
    clean_cache, clean_logits = capture_hidden(model, clean_inputs_device)
    clean_targets = targets_all.to(device)
    clean_metrics = prediction_metrics(
        clean_logits[:, EOS_POSITION], clean_targets, cfg['p'])
    _assert_stored_outputs(clean_metrics['full'], base_outputs,
                           'clean full-vocab EOS prediction')
    if not bool((clean_metrics['full'] == clean_targets).all()):
        raise RuntimeError(f'clean EOS prediction is not 32/32 original x{blank}')
    clean_raw = []
    for row in range(N_BASES):
        item = {
            'base_row': row,
            'blank': int(blank),
            'target': int(clean_targets[row]),
            'window': windows[row].tolist(),
            'input': clean_inputs[row].tolist(),
            'clean_numeric_output': int(clean_metrics['numeric'][row]),
            'clean_full_vocab_output': int(clean_metrics['full'][row]),
            'base_output': int(base_outputs[row]),
        }
        item.update(_raw_metric_fields(clean_metrics, row))
        clean_raw.append(item)

    routes = route_slopes(cfg, blank)
    raw_indices = {
        position: _raw_record_index(entry['raw_records'][f'x{position}'])
        for position in source_positions
    }
    corrupted_raw = []
    pending = []
    # Phase 1: all selected (source,base,d) outputs must match before phase 2.
    for base_row in base_rows:
        for source_position in source_positions:
            for start in range(0, len(d_values), batch_size):
                chunk_d = d_values[start:start + batch_size]
                expected_rows = []
                for d in chunk_d:
                    key = (base_row, d)
                    if key not in raw_indices[source_position]:
                        raise ValueError(f'missing stored raw record x{source_position} {key}')
                    expected_rows.append(raw_indices[source_position][key])
                inputs = make_corrupted_inputs(
                    clean_inputs[base_row:base_row + 1], source_position,
                    chunk_d, cfg['p']).to(device)
                targets = targets_all[base_row].repeat(len(chunk_d)).to(device)
                logits, records, route_exact, d_response = _baseline_rows(
                    model, inputs, targets, base_row, chunk_d, source_position,
                    expected_rows, routes, cfg['p'])
                corrupted_raw.extend(records)
                pending.append({
                    'inputs': inputs,
                    'logits': logits,
                    'targets': targets,
                    'base_row': base_row,
                    'd_values': chunk_d,
                    'source_position': source_position,
                    'route_exact': route_exact,
                    'baseline_changed': [row['baseline_changed'] for row in records],
                    'D': d_response,
                })
        print(f'  validated blank=x{blank} base_row={base_row}: '
              f'{len(source_positions)} sources x {len(d_values)} d values', flush=True)
    expected_corrupted = len(base_rows) * len(source_positions) * len(d_values)
    if len(corrupted_raw) != expected_corrupted:
        raise AssertionError('corrupted baseline count mismatch')

    # Phase 2: clean-cache activation patching and mandatory runtime controls.
    patched_raw = []
    control_max = {name: 0.0 for name in CONTROL_NAMES}
    for chunk_index, chunk in enumerate(pending, 1):
        rows, chunk_max = patch_chunk(
            model, chunk['inputs'], clean_cache, clean_logits, chunk['logits'],
            chunk['targets'], chunk['base_row'], chunk['d_values'],
            chunk['source_position'], chunk['route_exact'], chunk['D'], cfg['p'],
            patch_positions=restore_positions, baseline_changed=chunk['baseline_changed'],
            blank=blank)
        patched_raw.extend(rows)
        for name in control_max:
            control_max[name] = max(control_max[name], chunk_max[name])
        if chunk_index % 10 == 0 or chunk_index == len(pending):
            print(f'  patched {chunk_index}/{len(pending)} chunks', flush=True)
    expected_patched = expected_corrupted * len(SITES) * len(restore_positions)
    if len(patched_raw) != expected_patched:
        raise AssertionError(
            f'expected {expected_patched} patched rows, got {len(patched_raw)}')
    aggregates = aggregate_raw(patched_raw)
    per_base = aggregate_raw(patched_raw, per_base=True)
    validation = {
        'blank': int(blank),
        'clean': {
            'N': N_BASES,
            'full_vocab_match_count': int((clean_metrics['full'] == base_outputs.to(device)).sum()),
            'target_restore_count': int(clean_metrics['restore_full'].sum()),
            'cache_forward_calls': 1,
        },
        'corrupted': {
            'N': len(corrupted_raw),
            'expected_N': expected_corrupted,
            'full_sweep_N': N_BASES * len(source_positions) * (int(cfg['p']) - 1),
            'full_vocab_match_count': len(corrupted_raw),
            'numeric_match_count': len(corrupted_raw),
            'baseline_output_match_count': len(corrupted_raw),
            'D_match_count': len(corrupted_raw),
            'baseline_changed_count': sum(row['baseline_changed'] for row in corrupted_raw),
            'baseline_unchanged_count': sum(not row['baseline_changed'] for row in corrupted_raw),
            'all_validated_before_patching': True,
        },
    }
    controls = {
        'definitions': {
            'emb_ceiling': 'emb@corrupted source must reproduce all clean logits and Restore_full=1; a sanity ceiling, not a circuit finding',
            'emb_identity': 'emb@any other token position (including BLANK) must be allclose to corrupted logits',
            'causal_noop': 'when source_position > patch_position, every site must preserve EOS logits (an earlier position cannot see a future source)',
            'final_resid_noop': 'after L4 only position-wise ln_f+lm_head remains; no token-position patch can change EOS logits',
        },
        'allclose_rtol': CONTROL_RTOL,
        'allclose_atol': CONTROL_ATOL,
        'max_abs_diff': control_max,
    }
    for name in control_max:
        selected = [row for row in patched_raw if row[name]]
        controls[name] = {
            'N': len(selected),
            'restore_full_count': sum(row['restore_full'] for row in selected),
            'passed': True,
        }
    return (clean_raw, corrupted_raw, patched_raw, aggregates, per_base,
            validation, controls, routes)


def _matrix(aggregates, source_position, metric, blank=0):
    lookup = {
        (row['source_position'], tuple(row['site']), row['position']): row
        for row in aggregates
    }
    return [
        [lookup[(source_position, site, position)][metric]
         for position in patch_positions(blank)]
        for site in SITES
    ]


def make_plots(aggregates, path_prefix, n_bases, n_d, blank=0):
    """Two true-data 5 residual rows x 5 token-position columns per source."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    source_positions = operand_positions(blank)
    restore_positions = patch_positions(blank)
    paths = []
    for source_position in source_positions:
        source = f'x{source_position}'
        for metric, suffix in (('changed_restore_full', 'restore_full'),
                               ('mean_delta_margin', 'delta_margin')):
            matrix = _matrix(aggregates, source_position, metric, blank)
            flat_values = [value for row in matrix for value in row]
            fig, ax = plt.subplots(figsize=(4 + 1.4 * len(restore_positions), 10),
                                   constrained_layout=True)
            if metric == 'changed_restore_full':
                numeric_matrix = [[float('nan') if value is None else value
                                   for value in row] for row in matrix]
                image = ax.imshow(numeric_matrix, vmin=0.0, vmax=1.0,
                                  cmap='viridis', aspect='auto')
                colorbar_label = (
                    f'Changed Restore_full (EOS full-vocab argmax == original x{blank})')
            else:
                image = ax.imshow(matrix, cmap='coolwarm', aspect='auto',
                                  norm=TwoSlopeNorm(
                                      vmin=-max(1.0, max(abs(value) for value in flat_values)),
                                      vcenter=0.0,
                                      vmax=max(1.0, max(abs(value) for value in flat_values))))
                colorbar_label = 'Mean delta target margin (patched - corrupted)'
            ax.set_xticks(
                range(len(restore_positions)),
                [f'patch {patch_position_label(p, blank)}' for p in restore_positions])
            ax.set_yticks(range(5), SITE_LABELS)
            ax.set_xlabel('Patch position')
            ax.set_ylabel('Residual boundary')
            for row in range(5):
                for column in range(len(restore_positions)):
                    value = matrix[row][column]
                    if value is None:
                        text, color = 'n/a', 'black'
                    elif metric == 'changed_restore_full':
                        text = f'{value:.3f}'
                        color = 'white' if value < 0.55 else 'black'
                    else:
                        text, color = f'{value:.3g}', 'black'
                    ax.text(column, row, text, ha='center', va='center', color=color)
            fig.colorbar(image, ax=ax, shrink=0.8, label=colorbar_label)
            ax.set_title(
                f'blank x{blank} fill_blank activation patching | corrupt {source}\n'
                f'N={n_bases * n_d} total per cell; restore uses changed subset', fontsize=13)
            path = f'{path_prefix}_{source}_{suffix}.png'
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def _parse_int_items(items):
    if items is None:
        return None
    values = []
    for item in items:
        for part in str(item).split(','):
            if part.strip():
                values.append(int(part))
    return values


def _validate_selection(base_rows, d_values):
    if not base_rows or len(set(base_rows)) != len(base_rows):
        raise ValueError('--base-rows must be nonempty and unique')
    if not d_values or len(set(d_values)) != len(d_values):
        raise ValueError('--d-values must be nonempty and unique')
    if any(row < 0 or row >= N_BASES for row in base_rows):
        raise ValueError(f'--base-rows must be in 0..{N_BASES - 1}')
    if any(d < 1 or d >= P for d in d_values):
        raise ValueError(f'--d-values must be in 1..{P - 1}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blank', type=int, default=0, choices=range(SEQ_LEN))
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--base-data', default=DEFAULT_BASE_DATA)
    parser.add_argument('--base-rows', nargs='+', default=None,
                        help='base indices; comma-separated and/or space-separated (default: all 32)')
    parser.add_argument('--d-values', nargs='+', default=None,
                        help='nonzero modular deltas; comma-separated and/or space-separated (default: 1..126)')
    parser.add_argument('--out', default=None)
    parser.add_argument('--plot-prefix', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=126,
                        help='number of d values per forward chunk within each base/source')
    parser.add_argument('--dry-run', action='store_true',
                        help='print sizes and controls without loading data or checkpoint')
    args = parser.parse_args(argv)
    args.base_rows = _parse_int_items(args.base_rows)
    args.d_values = _parse_int_items(args.d_values)
    if args.base_rows is None:
        args.base_rows = list(range(N_BASES))
    if args.d_values is None:
        args.d_values = list(range(1, P))
    _validate_selection(args.base_rows, args.d_values)
    if args.threads <= 0 or args.batch_size <= 0:
        raise ValueError('--threads and --batch-size must be positive')
    if args.out is None:
        args.out = f'reports/data/fillblank_x{args.blank}_activation_patch_a25b56.json.gz'
    if args.plot_prefix is None:
        args.plot_prefix = ('reports/circuit_analysis/plots/'
                            f'fillblank_x{args.blank}_activation_patch_a25b56')
    return args


def plan_counts(base_rows, d_values, blank=0):
    n_sources = len(operand_positions(blank))
    n_patch_positions = len(patch_positions(blank))
    corrupted = len(base_rows) * n_sources * len(d_values)
    return {
        'clean': N_BASES,
        'corrupted': corrupted,
        'patched': corrupted * len(SITES) * n_patch_positions,
        'aggregate_cells': n_sources * len(SITES) * n_patch_positions,
        'per_base_cells': (len(base_rows) * n_sources
                           * len(SITES) * n_patch_positions),
    }


def print_dry_run(args):
    counts = plan_counts(args.base_rows, args.d_values, args.blank)
    full_counts = plan_counts(range(N_BASES), range(1, P), args.blank)
    full = (args.base_rows == list(range(N_BASES))
            and args.d_values == list(range(1, P)))
    source_positions = operand_positions(args.blank)
    restore_positions = patch_positions(args.blank)
    print(f'DRY RUN: blank x{args.blank} fill_blank clean->corrupted activation patching')
    print(f'  ckpt: {args.ckpt}')
    print(f'  base data: {args.base_data} (by_blank.x{args.blank})')
    print(f'  output: {args.out}')
    print(f'  plot prefix: {args.plot_prefix}')
    print(f'  device={args.device}; threads={args.threads}; d-chunk batch-size={args.batch_size}')
    print(f'  base rows ({len(args.base_rows)}): {args.base_rows}')
    print(f'  d values ({len(args.d_values)}): {args.d_values}')
    print(f'  full default sweep: {full}')
    print(f'  clean: {clean_input_label(args.blank)}, target=windows[:,{args.blank}]; '
          'one cache for all 32 bases')
    print(f'  corruption sources (operands only): {source_positions}')
    print(f'  restore patch positions (includes BLANK, excludes EOS): {restore_positions}')
    for site in SITES:
        print(f'    {site}: {site_label(site)}')
    print(f'  counts: clean={counts["clean"]}; corrupted={counts["corrupted"]}; '
          f'patched={counts["patched"]}; aggregate cells={counts["aggregate_cells"]}; '
          f'per-base cells={counts["per_base_cells"]}')
    print(f'  full sweep reference for blank x{args.blank}: clean=32; '
          f'corrupted={full_counts["corrupted"]}; patched={full_counts["patched"]}; '
          f'aggregate cells={full_counts["aggregate_cells"]}; '
          f'per-base cells={full_counts["per_base_cells"]}')
    print('  hard controls: emb@source clean ceiling + Restore_full=1; '
          'emb@other identity (including BLANK); source>patch causal EOS no-op; '
          'final-residual EOS no-op')
    print(f'  {2 * len(source_positions)} plots: each source x 5 residual rows x 5 patch columns; '
          '{changed Restore_full, mean delta margin}')


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print_dry_run(args)
        return
    torch.set_num_threads(args.threads)
    if not str(args.device).startswith('cpu'):
        print(f'WARNING: requested non-CPU device {args.device}; use --device cpu for CPU validation')
    print(f'[1/5] Loading fixed by_blank.x{args.blank} base data', flush=True)
    with open(args.base_data, encoding='utf-8') as handle:
        base_data = json.load(handle)
    print('[2/5] Loading the a25b56 best checkpoint', flush=True)
    model, cfg, ckpt = load_model(args.ckpt, args.device)
    _validate_loaded_model(model, cfg)
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError('activation patching requires the fp32 checkpoint')
    print(f'  device={args.device}; dtype={next(model.parameters()).dtype}; '
          f'rules={cfg["rules"]}; P={cfg["p"]}', flush=True)
    print('[3/5] Validating clean and all selected corrupted baselines, then patching',
          flush=True)
    (clean_raw, corrupted_raw, patched_raw, aggregates, per_base,
     validation, controls, routes) = analyze(
        model, cfg, base_data, args.base_rows, args.d_values,
        args.batch_size, args.device, args.blank)
    counts = plan_counts(args.base_rows, args.d_values, args.blank)
    full_counts = plan_counts(range(N_BASES), range(1, P), args.blank)
    route_metadata = route_definitions(cfg, args.blank)
    report = {
        'config': {
            'ckpt': os.path.abspath(args.ckpt),
            'base_data': os.path.abspath(args.base_data),
            'blank': f'x{args.blank}',
            'blank_index': args.blank,
            'clean_input': clean_input_label(args.blank),
            'target': f'original windows[:,{args.blank}], never corrupted input',
            'base_rows': args.base_rows,
            'd_values': args.d_values,
            'source_positions': operand_positions(args.blank),
            'patch_positions': patch_positions(args.blank),
            'sites': [list(site) for site in SITES],
            'site_labels': SITE_LABELS,
            'eos_position': EOS_POSITION,
            'route_definitions': route_metadata,
            'route_slopes': route_metadata['legal_routes'],
            'route_exact_definition': ('matched_routes is nonempty; a route matches when '
                                       'D == slope*d (mod P) and perturbed_full_vocab_output < P'),
            'device': args.device,
            'threads': args.threads,
            'batch_size_per_base_d_chunk': args.batch_size,
            'model_config': cfg,
            'checkpoint_final_epoch': ckpt.get('final_epoch'),
            'checkpoint_best_accuracy': ckpt.get('best_accuracy'),
            'counts': counts,
            'full_default_counts': full_counts,
        },
        'metric_definitions': {
            'Restore_full': f'patched EOS full-vocab argmax == original x{args.blank}',
            'restore_numeric': 'patched EOS argmax over classes 0..P-1 == original target',
            'corrupt_persistence': 'not Restore_full and patched full output == corrupted full output (aggregate alias: persistence)',
            'nonnumeric': 'patched full-vocab output >= P (may overlap corrupt_persistence)',
            'other': 'none of Restore_full, corrupt_persistence, or nonnumeric',
            'baseline_changed': 'corrupted full-vocab output != original clean target',
            'changed_restore_full': 'Restore_full count / N among baseline_changed samples',
            'unchanged_preserved': 'Restore_full count / N among unchanged (originally correct) samples',
            'delta': 'patched target metric minus corrupted-baseline target metric',
            'aggregation': 'rates are exact raw-count/N; means/medians are computed from raw rows',
        },
        'validation': validation,
        'controls': controls,
        'clean': clean_raw,
        'corrupted_baseline': corrupted_raw,
        'raw': patched_raw,
        'aggregates': aggregates,
        'per_base_aggregates': per_base,
    }
    print('[4/5] Writing JSON with exact raw counts', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.out.endswith('.gz'):
        with gzip.open(args.out, 'wt', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
    else:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
    print(f'  JSON: {args.out} ({len(patched_raw)} patched raw rows)', flush=True)
    print(f'[5/5] Drawing {2 * len(operand_positions(args.blank))} residual heatmaps '
          '(5x5 full-position grid)',
          flush=True)
    paths = make_plots(aggregates, args.plot_prefix,
                       len(args.base_rows), len(args.d_values), args.blank)
    for path in paths:
        print(f'  Plot: {path}', flush=True)
    print('Done. All baseline validations and mandatory controls passed.', flush=True)


if __name__ == '__main__':
    main()
