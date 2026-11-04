#!/usr/bin/env python3
"""len5 fill_blank Logit Lens：逐层观察最终答案或 F/M/R 中间量。"""
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
P = 127
A = 25
B = 56
A_INV = pow(A, -1, P)
B_INV = pow(B, -1, P)
NEG_A_INV_B = (-A_INV * B) % P
NEG_B_INV_A = (-B_INV * A) % P
COEFFICIENTS = {
    'a': A, 'b': B, "a'": A_INV, "b'": B_INV,
    "-a'b": NEG_A_INV_B, "-b'a": NEG_B_INV_A,
}
DEFAULT_CKPT = (
    '/data/cxm/models/variant_o2_fillblank_noaux_randpos_d512l4_P127_len3to5_wd2_ckpt/'
    'single_o2_fillblank_a25b56_d512l4r8h4_P127_len3to5_randpos_noaux_'
    'lr0.0003_wd2_ckpt_seed42_best.pth'
)
DEFAULT_BASE_DATA = 'reports/data/fillblank_len5_all_blank_linear_a25b56.json'
DEFAULT_OUT = 'reports/data/fillblank_len5_logit_lens_a25b56.json'
DEFAULT_PLOT = 'plots/fillblank_len5_logit_lens_a25b56.png'
DEFAULT_INTERMEDIATES_OUT = (
    'reports/data/fillblank_len5_logit_lens_intermediates_a25b56.json')
DEFAULT_INTERMEDIATES_PLOT = 'plots/fillblank_len5_logit_lens_intermediates_a25b56'
CRITERIA = {
    'primary': {'top1_rate_min': 0.90, 'mean_prob_min': 0.50},
    'strict': {'top1_rate_min': 0.95, 'mean_prob_min': 0.90},
}
PER_BASE_CRITERIA = {
    'primary': {'rank': 1, 'prob_min': 0.50},
    'strict': {'rank': 1, 'prob_min': 0.90},
}


def legal_positions(k, seq_len=SEQ_LEN):
    """返回该空格位置合法的输入槽，普通位置升序，EOS 最后。"""
    if not 0 <= k < seq_len:
        raise ValueError(f'k must be in [0, {seq_len}), got {k}')
    positions = list(range(max(0, k - 2), min(seq_len - 1, k + 2) + 1))
    return positions + [seq_len]


def position_label(position):
    return 'EOS' if position == EOS_POSITION else f'x{position}'


def route_operands(k, route):
    """F/M/R 的两个扰动操作数 (position, slope)；可能含越界位置。"""
    if route == 'F':
        return [(k - 2, B), (k - 1, A)]
    if route == 'M':
        return [(k - 1, NEG_A_INV_B), (k + 1, A_INV)]
    if route == 'R':
        return [(k + 1, NEG_B_INV_A), (k + 2, B_INV)]
    raise ValueError(f'unknown route {route!r}')


def legal_route(k, route):
    return all(0 <= position < SEQ_LEN for position, _ in route_operands(k, route))


def intermediate_targets(k):
    """枚举完整 F/M/R 局部公式合法的中间目标；目标真值不依赖空格位置。"""
    if not 0 <= k < SEQ_LEN:
        raise ValueError(f'k must be in [0, {SEQ_LEN}), got {k}')
    names = {
        'F': [('F_b', 'b'), ('F_a', 'a')],
        'M': [('M_neg_a_inv_b', "-a'b"), ('M_a_inv', "a'")],
        'R': [('R_neg_b_inv_a', "-b'a"), ('R_b_inv', "b'")],
    }
    targets = []
    for route in ('F', 'M', 'R'):
        if not legal_route(k, route):
            continue
        operands = route_operands(k, route)
        for (position, coefficient), (prefix, coefficient_name) in zip(operands, names[route]):
            targets.append({
                'target_id': f'{prefix}_x{position}',
                'route': route,
                'label': f'{coefficient_name}*x{position}',
                'terms': [[coefficient, position]],
            })
        if route == 'M':
            targets.append({
                'target_id': 'uM', 'route': 'M',
                'label': f'uM=x{k + 1}-b*x{k - 1}',
                'terms': [[1, k + 1], [-B, k - 1]],
            })
        elif route == 'R':
            targets.append({
                'target_id': 'uR', 'route': 'R',
                'label': f'uR=x{k + 2}-a*x{k + 1}',
                'terms': [[1, k + 2], [-A, k + 1]],
            })
    return targets


def targets_for_blank(k, target_mode='final'):
    if target_mode == 'final':
        return [{'target_id': f'x{k}', 'route': 'final', 'label': f'x{k}',
                 'terms': [[1, k]]}]
    if target_mode == 'intermediates':
        return intermediate_targets(k)
    raise ValueError(f'unknown target mode {target_mode!r}')


def compute_target_values(windows, target):
    """从干净窗口计算缩放/模减目标，负系数和减法均在 mod P 下处理。"""
    values = torch.zeros(len(windows), dtype=windows.dtype, device=windows.device)
    for coefficient, position in target['terms']:
        values = values + coefficient * windows[:, position]
    return values % P


def build_plan(n_bases=N_BASES, n_layers=N_LAYERS, target_mode='final'):
    plan = []
    for k in range(SEQ_LEN):
        positions = legal_positions(k)
        targets = targets_for_blank(k, target_mode)
        n_cells = n_layers * len(positions) * len(targets)
        plan.append({
            'k': k,
            'target': f'x{k}' if target_mode == 'final' else 'intermediates',
            'targets': targets,
            'n_targets': len(targets),
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
    """干净 len5 窗口 -> 第 k 位 BLANK、末尾追加 EOS；真值仍从原窗口计算。"""
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
    k = records[0]['k']
    return {
        'k': k,
        'target_id': records[0].get('target_id', f'x{k}'),
        'route': records[0].get('route', 'final'),
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
        k = row['k']
        target_id = row.get('target_id', f'x{k}')
        key = (k, target_id, row['layer_index'], row['position'])
        grouped.setdefault(key, []).append(row)
    return [aggregate_records(records) for records in grouped.values()]


def is_clear(cell, criterion):
    thresholds = CRITERIA[criterion] if isinstance(criterion, str) else criterion
    return (cell['top1_rate'] >= thresholds['top1_rate_min']
            and cell['mean_prob'] >= thresholds['mean_prob_min'])


def clear_summary(aggregates, k, criterion, target_id=None):
    """层优先、位置次序次之；另报所有 clear cells 中 mean_prob 最高者。"""
    position_order = {position: i for i, position in enumerate(legal_positions(k))}
    cells = [cell for cell in aggregates if cell['k'] == k
             and (target_id is None or cell.get('target_id', f'x{k}') == target_id)]
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


def build_earliest(aggregates, target_mode='final'):
    if target_mode == 'final':
        return {
            criterion: {f'x{k}': clear_summary(aggregates, k, criterion)
                        for k in range(SEQ_LEN)}
            for criterion in CRITERIA
        }
    if target_mode != 'intermediates':
        raise ValueError(f'unknown target mode {target_mode!r}')
    result = {criterion: {} for criterion in CRITERIA}
    for k in range(SEQ_LEN):
        target_ids = []
        for cell in aggregates:
            if cell['k'] == k and cell['target_id'] not in target_ids:
                target_ids.append(cell['target_id'])
        for criterion in CRITERIA:
            result[criterion][f'x{k}'] = {
                target_id: clear_summary(aggregates, k, criterion, target_id)
                for target_id in target_ids
            }
    return result


def is_readable(record, criterion):
    thresholds = PER_BASE_CRITERIA[criterion]
    return record['rank'] == thresholds['rank'] and record['prob'] >= thresholds['prob_min']


def _metric_cell(record):
    return {name: record[name]
            for name in ('layer_index', 'position', 'prob', 'logit', 'rank', 'margin')}


def _copy_flags(record, k, window):
    """区分位置直复制与更保守的任一可见输入复制；无 base_data 时为 None。"""
    if window is None:
        return {'direct_copy_at_position': None, 'target_in_visible_inputs': None}
    target_value = record.get('target_value')
    position = record['position']
    direct = (position < SEQ_LEN and position != k
              and target_value == window[position])
    visible = any(target_value == window[j] for j in range(SEQ_LEN) if j != k)
    return {
        'direct_copy_at_position': bool(direct),
        'target_in_visible_inputs': bool(visible),
    }


def _per_base_views(records, k, window=None):
    position_order = {position: i for i, position in enumerate(legal_positions(k))}
    ordered = sorted(records, key=lambda row: (
        row['layer_index'], position_order[row['position']]))
    decorated = [(row, {**_metric_cell(row), **_copy_flags(row, k, window)})
                 for row in ordered]
    views = {}
    for criterion in PER_BASE_CRITERIA:
        readable = [pair for pair in decorated if is_readable(pair[0], criterion)]
        best = None
        if readable:
            best = min(readable, key=lambda pair: (
                -pair[0]['prob'], pair[0]['rank'], -pair[0]['margin'],
                pair[0]['layer_index'], position_order[pair[0]['position']]))
        views[criterion] = {
            'readable_anywhere': bool(readable),
            'readable_cells': [cell for _, cell in readable],
            'earliest': dict(readable[0][1]) if readable else None,
            'best_cell': dict(best[1]) if best is not None else None,
        }
    return views


def build_per_base_readability(raw, base_data=None):
    """逐 (k,target,base) 审计阈值可读性，并提供两级 copy-control 汇总。"""
    grouped = {}
    for row in raw:
        k = row['k']
        key = (k, row.get('target_id', f'x{k}'), row['base_row'])
        grouped.setdefault(key, []).append(row)
    bases = []
    for (k, target_id, base_row), records in sorted(grouped.items()):
        window = None
        if base_data is not None:
            window = base_data['by_blank'][f'x{k}']['base_clean_windows'][base_row]
        entry = {
            'k': k,
            'target_id': target_id,
            'route': records[0].get('route', 'final'),
            'base_row': base_row,
            'target_value': records[0].get('target_value'),
            'target_in_visible_inputs': _copy_flags(records[0], k, window)[
                'target_in_visible_inputs'],
        }
        entry.update(_per_base_views(records, k, window))
        bases.append(entry)

    summary = []
    target_groups = {}
    for entry in bases:
        target_groups.setdefault((entry['k'], entry['target_id']), []).append(entry)
    for (k, target_id), entries in sorted(target_groups.items()):
        item = {
            'k': k,
            'target_id': target_id,
            'route': entries[0]['route'],
            'n_bases': len(entries),
        }
        for criterion in PER_BASE_CRITERIA:
            readable = [entry for entry in entries if entry[criterion]['readable_anywhere']]
            best_cells = []
            for entry in readable:
                best_cells.append({'base_row': entry['base_row'],
                                   **entry[criterion]['best_cell']})
            criterion_summary = {
                'n_readable_bases': len(readable),
                'readable_base_rows': [entry['base_row'] for entry in readable],
                'n_readable_cells': sum(len(entry[criterion]['readable_cells'])
                                        for entry in readable),
                'best_cells': best_cells,
            }
            for flag in ('direct_copy_at_position', 'target_in_visible_inputs'):
                suffix = f'excluding_{flag}'
                if base_data is None:
                    controlled = None
                    n_controlled_cells = None
                else:
                    controlled = [
                        entry for entry in readable
                        if any(cell[flag] is False
                               for cell in entry[criterion]['readable_cells'])
                    ]
                    n_controlled_cells = sum(
                        cell[flag] is False for entry in readable
                        for cell in entry[criterion]['readable_cells'])
                criterion_summary[f'n_readable_bases_{suffix}'] = (
                    None if controlled is None else len(controlled))
                criterion_summary[f'readable_base_rows_{suffix}'] = (
                    None if controlled is None
                    else [entry['base_row'] for entry in controlled])
                criterion_summary[f'n_readable_cells_{suffix}'] = n_controlled_cells
            item[criterion] = criterion_summary
        summary.append(item)
    return {'criteria': PER_BASE_CRITERIA, 'summary': summary, 'bases': bases}


def _route_matches_by_base(raw, slope):
    matches = {}
    for i, base_row in enumerate(raw['base_row']):
        base_row = int(base_row)
        d = int(raw['d'][i])
        if not 1 <= d < P:
            raise ValueError(f'route-support d must be in 1..{P - 1}, got {d}')
        full_output = int(raw['perturbed_full_vocab_output'][i])
        exact = (0 <= full_output < P
                 and (int(raw['D'][i]) - slope * d) % P == 0)
        matches.setdefault(base_row, {})
        if d in matches[base_row]:
            raise ValueError(f'duplicate perturbation d={d} for base_row={base_row}')
        matches[base_row][d] = exact
    expected_d = set(range(1, P))
    for base_row, base_matches in matches.items():
        if set(base_matches) != expected_d:
            raise ValueError(f'base_row={base_row} must contain all 126 perturbations')
    return matches


def route_support_for_blank(entry, k, route):
    """逐 base 比较两个操作数在相同 d 上的精确扰动响应支持。"""
    if not legal_route(k, route):
        raise ValueError(f'route {route} is not legal at k={k}')
    operands = route_operands(k, route)
    windows = entry['base_clean_windows']
    expected_bases = set(range(len(windows)))
    matches = []
    for position, slope in operands:
        raw = entry['raw_records'][f'x{position}']
        position_matches = _route_matches_by_base(raw, slope)
        if set(position_matches) != expected_bases:
            raise ValueError(f'x{position} raw base rows do not match base_clean_windows')
        matches.append(position_matches)

    bases = []
    for base_row in sorted(expected_bases):
        counts = {
            f'x{position}': sum(position_matches[base_row].values())
            for (position, _), position_matches in zip(operands, matches)
        }
        rates = {position: count / (P - 1) for position, count in counts.items()}
        shared_d = sorted(
            d for d in range(1, P)
            if all(position_matches[base_row][d] for position_matches in matches))
        bases.append({
            'base_row': base_row,
            'window': windows[base_row],
            'operand_exact_match_counts': counts,
            'operand_exact_match_rates': rates,
            'min_operand_exact_match_rate': min(rates.values()),
            'shared_exact_match_count': len(shared_d),
            'shared_exact_match_rate': len(shared_d) / (P - 1),
            'shared_d': shared_d,
        })
    ranked = sorted(bases, key=lambda row: (
        -row['shared_exact_match_count'], -row['min_operand_exact_match_rate'],
        row['base_row']))
    return {
        'k': k,
        'route': route,
        'operands': [{'position': position, 'slope': slope} for position, slope in operands],
        'n_d_per_base': P - 1,
        'bases': bases,
        'top3': [dict(row) for row in ranked[:3]],
    }


def build_route_support(base_data):
    by_blank = {}
    for k in range(SEQ_LEN):
        entry = base_data['by_blank'][f'x{k}']
        by_blank[f'x{k}'] = {
            route: route_support_for_blank(entry, k, route)
            for route in ('F', 'M', 'R') if legal_route(k, route)
        }
    return {
        'definition': ('Exact modular perturbation-response support over d=1..126; '
                       'shared d requires both operand positions to match at the same d.'),
        'interpretation': 'Observed perturbation labels, not clean-circuit ground truth.',
        'modulus': P,
        'by_blank': by_blank,
    }


@torch.inference_mode()
def probe_batch(model, inputs, targets, k, positions, target_specs=None, base_offset=0):
    """一次逐 block forward，在同一 hidden/logits 上循环所有 targets。"""
    if isinstance(targets, dict):
        if target_specs is None:
            raise ValueError('target_specs are required with multiple targets')
    else:
        target_specs = targets_for_blank(k, 'final')
        targets = {target_specs[0]['target_id']: targets}
    hidden = model.transformer.drop(model.transformer.wte(inputs))
    raw = []
    final_eos_logits = None
    for layer_index, block in enumerate(model.transformer.h):
        hidden = block(hidden)
        logits = model.lm_head(model.transformer.ln_f(hidden))
        for target in target_specs:
            target_id = target['target_id']
            target_values = targets[target_id].detach().cpu()
            for position in positions:
                metrics = target_metrics(logits[:, position, :], targets[target_id])
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
                        'base_row': base_offset + sample_index,
                        'target_id': target_id,
                        'target_value': int(target_values[sample_index]),
                        'route': target['route'],
                    })
        if layer_index == len(model.transformer.h) - 1:
            final_eos_logits = logits[:, EOS_POSITION, :]
    reference_logits, _ = model(inputs)
    return raw, final_eos_logits, reference_logits[:, EOS_POSITION, :]


def _validate_loaded_model(model, cfg):
    expected = {
        'p': P, 'd_model': 512, 'n_layer': N_LAYERS, 'vocab_size': 129,
        'blank_token_id': 127, 'eos_token_id': 128,
        'variant': 'fill_blank', 'order': 2, 'rules': [[25, 56]],
        'train_len': 5, 'body_supervision': False,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'unexpected checkpoint config {key}={cfg.get(key)!r}, expected {value}')
    if len(model.transformer.h) != N_LAYERS:
        raise ValueError('checkpoint must have exactly four transformer blocks')


@torch.inference_mode()
def analyze(model, base_data, device, batch_size, target_mode='final'):
    raw = []
    plan = build_plan(target_mode=target_mode)
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
    for item in plan:
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
        target_values = {
            target['target_id']: compute_target_values(windows, target)
            for target in item['targets']
        }
        for start in range(0, N_BASES, batch_size):
            end = min(start + batch_size, N_BASES)
            batch_inputs = inputs[start:end].to(device)
            batch_targets = {
                target_id: values[start:end].to(device)
                for target_id, values in target_values.items()
            }
            batch_raw, manual_eos, reference_eos = probe_batch(
                model, batch_inputs, batch_targets, k, item['positions'],
                target_specs=item['targets'], base_offset=start)
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
        print(f'  x{k}: {len(item["targets"])} targets x {len(item["positions"])} '
              f'positions x 4 layers x 32 bases complete')
    if validation['base_output_mismatches']:
        raise RuntimeError(
            f'EOS predictions disagree with stored base_outputs for '
            f'{len(validation["base_output_mismatches"])} samples')
    aggregates = aggregate_raw(raw)
    expected_totals = plan_totals(plan)
    if len(raw) != expected_totals['n_raw_records']:
        raise AssertionError(
            f'expected {expected_totals["n_raw_records"]} raw records, got {len(raw)}')
    if len(aggregates) != expected_totals['n_cells']:
        raise AssertionError(
            f'expected {expected_totals["n_cells"]} aggregate cells, got {len(aggregates)}')
    earliest = build_earliest(aggregates, target_mode)
    return raw, aggregates, earliest, validation


def _draw_heatmap(ax, matrix, positions, title):
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap='viridis', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Input position')
    ax.set_ylabel('Layer')
    ax.set_xticks(range(len(positions)),
                  [position_label(position) for position in positions])
    ax.set_yticks(range(N_LAYERS), [f'L{layer + 1}' for layer in range(N_LAYERS)])
    for layer in range(N_LAYERS):
        for column, value in enumerate(matrix[layer]):
            ax.text(column, layer, f'{value:.3f}', ha='center', va='center',
                    color='white' if value < 0.55 else 'black', fontsize=9)
    return image


def make_plot(aggregates, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lookup = {(cell['k'], cell.get('target_id', f'x{cell["k"]}'),
               cell['layer_index'], cell['position']): cell
              for cell in aggregates}
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    active_axes = []
    image = None
    for k in range(SEQ_LEN):
        ax = axes.flat[k]
        active_axes.append(ax)
        positions = legal_positions(k)
        matrix = [[lookup[(k, f'x{k}', layer, position)]['mean_prob']
                   for position in positions] for layer in range(N_LAYERS)]
        image = _draw_heatmap(ax, matrix, positions, f'Blank at x{k}; target x{k}')
    axes.flat[-1].axis('off')
    fig.colorbar(image, ax=active_axes, shrink=0.9, pad=0.02,
                 label='Mean target probability')
    fig.suptitle('Len5 fill_blank Logit Lens (32 held-out bases per blank)', fontsize=15)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_intermediate_plots(aggregates, path_prefix):
    """每个 blank 一张图；每个 target 一个独立层×位置热图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lookup = {(cell['k'], cell['target_id'], cell['layer_index'], cell['position']): cell
              for cell in aggregates}
    paths = []
    for item in build_plan(target_mode='intermediates'):
        k = item['k']
        targets = item['targets']
        positions = item['positions']
        fig, axes = plt.subplots(len(targets), 1, figsize=(12, 2.5 * len(targets)),
                                 constrained_layout=True, squeeze=False)
        image = None
        for ax, target in zip(axes.flat, targets):
            target_id = target['target_id']
            matrix = [[lookup[(k, target_id, layer, position)]['mean_prob']
                       for position in positions] for layer in range(N_LAYERS)]
            image = _draw_heatmap(
                ax, matrix, positions,
                f'Blank at x{k}; target {target["label"]}; route {target["route"]}')
        fig.colorbar(image, ax=list(axes.flat), shrink=0.9, pad=0.02,
                     label='Mean target probability')
        fig.suptitle(f'Len5 intermediate Logit Lens: blank x{k} (32 bases)', fontsize=15)
        path = f'{path_prefix}_blank_x{k}.png'
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def make_route_representative_plots(raw, route_support, path_prefix):
    """每个合法 (k,route) 一张 top3-base × route-target 单样本概率图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lookup = {(row['k'], row['target_id'], row['base_row'],
               row['layer_index'], row['position']): row['prob'] for row in raw}
    plan = {item['k']: item for item in build_plan(target_mode='intermediates')}
    paths = []
    for k in range(SEQ_LEN):
        item = plan[k]
        positions = item['positions']
        for route, support in route_support['by_blank'][f'x{k}'].items():
            targets = [target for target in item['targets'] if target['route'] == route]
            rows = [(base, target) for base in support['top3'] for target in targets]
            fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.5 * len(rows)),
                                     constrained_layout=True, squeeze=False)
            image = None
            for ax, (base, target) in zip(axes.flat, rows):
                base_row = base['base_row']
                target_id = target['target_id']
                matrix = [[lookup[(k, target_id, base_row, layer, position)]
                           for position in positions] for layer in range(N_LAYERS)]
                title = (f'base_row={base_row}; target {target["label"]}; '
                         f'shared_exact_match_count={base["shared_exact_match_count"]}')
                image = _draw_heatmap(ax, matrix, positions, title)
            fig.colorbar(image, ax=list(axes.flat), shrink=0.9, pad=0.02,
                         label='Target probability (single base)')
            fig.suptitle(
                f'Len5 intermediate Logit Lens: blank x{k}, route {route}; '
                'top3 perturbation-support bases', fontsize=15)
            path = f'{path_prefix}_route_x{k}_{route}.png'
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-mode', choices=('final', 'intermediates'), default='final')
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--base-data', default=DEFAULT_BASE_DATA)
    parser.add_argument('--out', default=None)
    parser.add_argument('--plot', default=None,
                        help='final: output PNG; intermediates: prefix for overall/route PNGs')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印固定分析计划，不读取基底或加载模型')
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = (DEFAULT_OUT if args.target_mode == 'final'
                    else DEFAULT_INTERMEDIATES_OUT)
    if args.plot is None:
        args.plot = (DEFAULT_PLOT if args.target_mode == 'final'
                     else DEFAULT_INTERMEDIATES_PLOT)
    return args


def print_dry_run(args):
    plan = build_plan(target_mode=args.target_mode)
    totals = plan_totals(plan)
    print('DRY RUN: len5 fill_blank Logit Lens (no data or model loaded)')
    print(f'  target mode: {args.target_mode}')
    print(f'  ckpt: {args.ckpt}')
    print(f'  base data: {args.base_data}')
    print(f'  output: {args.out}')
    print(f'  plot: {args.plot}')
    print(f'  device={args.device}, threads={args.threads}, batch_size={args.batch_size}')
    if args.target_mode == 'final':
        print('  Each base: replace xK with BLANK(127), append EOS(128), target=xK.')
    else:
        print('  Each base: replace xK with BLANK(127), append EOS(128); '
              'all target values come from the clean base window.')
        print('  Targets share each block-by-block forward (no per-target forward).')
    for item in plan:
        labels = ', '.join(item['position_labels'])
        print(f'  k={item["k"]}: positions [{labels}]; '
              f'{item["n_targets"]} targets; {item["n_cells"]} cells; '
              f'{item["n_raw_records"]} raw records')
        for target in item['targets']:
            print(f'    {target["target_id"]} | route={target["route"]} | {target["label"]}')
    print(f'  TOTAL: {sum(item["n_targets"] for item in plan)} targets; '
          f'{totals["n_cells"]} aggregate cells; {totals["n_raw_records"]} raw records; '
          f'N={totals["n_bases"]} per cell')
    batch_counts = [(N_BASES + args.batch_size - 1) // args.batch_size for _ in plan]
    print(f'  Forward/EOS-validation batches by blank: {batch_counts}; '
          f'total={sum(batch_counts)}')
    print('  Aggregate criteria: primary top1>=0.90 & mean_prob>=0.50; '
          'strict top1>=0.95 & mean_prob>=0.90')
    print('  Per-base criteria: primary rank==1 & prob>=0.50; '
          'strict rank==1 & prob>=0.90')


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

    print(f'[3/4] Running manual block-by-block Logit Lens ({args.target_mode})')
    raw, aggregates, earliest, validation = analyze(
        model, base_data, args.device, args.batch_size, args.target_mode)
    per_base = build_per_base_readability(raw, base_data)
    report = {
        'config': {
            'ckpt': os.path.abspath(args.ckpt),
            'base_data': os.path.abspath(args.base_data),
            'device': args.device,
            'threads': args.threads,
            'batch_size': args.batch_size,
            'target_mode': args.target_mode,
            'seq_len': SEQ_LEN,
            'eos_position': EOS_POSITION,
            'n_bases': N_BASES,
            'n_layers': N_LAYERS,
            'coefficients': COEFFICIENTS,
            'model_config': cfg,
            'checkpoint_final_epoch': ckpt.get('final_epoch'),
            'checkpoint_best_accuracy': ckpt.get('best_accuracy'),
            'plan': build_plan(target_mode=args.target_mode),
        },
        'criteria': CRITERIA,
        'raw': raw,
        'aggregates': aggregates,
        'earliest': earliest,
        'per_base_readability': per_base,
        'final_forward_validation': validation,
    }
    if args.target_mode == 'intermediates':
        report['perturbation_route_support'] = build_route_support(base_data)
    parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(parent, exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(f'  JSON: {args.out} ({len(raw)} raw records, {len(aggregates)} cells)')

    print('[4/4] Drawing mean-probability heatmaps')
    if args.target_mode == 'final':
        make_plot(aggregates, args.plot)
        print(f'  Plot: {args.plot}')
    else:
        paths = make_intermediate_plots(aggregates, args.plot)
        paths += make_route_representative_plots(
            raw, report['perturbation_route_support'], args.plot)
        for path in paths:
            print(f'  Plot: {path}')
    print(f'Done. Final EOS allclose={validation["allclose"]}; '
          f'max_abs_diff={validation["max_abs_diff"]:.3g}; base outputs match.')


if __name__ == '__main__':
    main()
