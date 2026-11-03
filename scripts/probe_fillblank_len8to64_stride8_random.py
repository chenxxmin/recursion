#!/usr/bin/env python3
"""fill_blank a25b56：len8..64（步长 8）的随机局部扰动探针。

每个长度固定 128 个 seed42 held-out 窗口；每个挖空位置从完整词表答对的
窗口中选 32 个基底，对合法的 k-2/k-1/k+1/k+2 位置分别随机施加 8 个非零
模 127 增量。输出保留全部 raw 记录，并与已有 len5 全增量探针按
category × relative 聚合比较。checkpoint 只读，默认完全在 CPU 上运行。
"""
import argparse
import gc
import json
import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from datasets import RecurrenceDataset  # noqa: E402
from rules import LinearRecurrenceRule  # noqa: E402
from probe_fillblank_last_blank_linear import (  # noqa: E402
    CANDIDATE_NAMES, candidate_values, circular_mae, load_model, predict, rollout,
)

LENGTHS = list(range(8, 65, 8))
RELATIVES = (-2, -1, 1, 2)
CATEGORIES = (
    'left_boundary', 'left_inner', 'interior', 'right_inner', 'right_boundary',
)
P = 127
N_BASELINE = 128
BASE_SELECTION_SEED = 20261014
DELTA_SEED = 20261103
DEFAULT_CKPT_TEMPLATE = (
    '/data/cxm/models/variant_o2_fillblank_noaux_randpos_d512l4_P127_'
    'len{L}_wd2_initLen{L-1}/single_o2_fillblank_a25b56_d512l4r8h4_P127_'
    'len{L}_randpos_noaux_lr0.0003_wd2_initLen{L-1}_seed42_best.pth'
)
LEN8_CKPT = (
    '/data/cxm/models/variant_o2_fillblank_noaux_randpos_d512l4_P127_'
    'len8_wd2_initLen35/single_o2_fillblank_a25b56_d512l4r8h4_P127_'
    'len8_randpos_noaux_lr0.0003_wd2_initLen35_seed42_best.pth')
DEFAULT_OUT = os.path.join(
    ROOT, 'reports', 'data', 'fillblank_len8to64_stride8_random_a25b56.json')
DEFAULT_LEN5 = os.path.join(
    ROOT, 'reports', 'data', 'fillblank_len5_all_blank_linear_a25b56.json')


def blank_category(k, length):
    """与 len5 对齐的五类位置划分。"""
    if length < 5 or not 0 <= k < length:
        raise ValueError(f'invalid blank position k={k}, length={length}')
    if k == 0:
        return 'left_boundary'
    if k == 1:
        return 'left_inner'
    if 2 <= k <= length - 3:
        return 'interior'
    if k == length - 2:
        return 'right_inner'
    return 'right_boundary'


def legal_relative_positions(length, k):
    """返回 (relative, j)，顺序固定为 -2,-1,+1,+2。"""
    if not 0 <= k < length:
        raise ValueError(f'invalid blank position k={k}, length={length}')
    return [(r, k + r) for r in RELATIVES
            if 0 <= k + r < length and k + r != k]


def relative_key(relative):
    return f'{relative:+d}'


def seed_label(seed, *parts):
    """字符串种子由所有实验坐标显式确定，避免 Python 随机 hash。"""
    return '/'.join(str(x) for x in (seed, *parts))


def sample_deltas(length, k, relative, base_row, n_deltas=8, p=P,
                  seed=DELTA_SEED):
    """每个 base_row 使用独立的局部 Random，抽不同且确定性的非零 d。"""
    if not 1 <= n_deltas < p:
        raise ValueError(f'n_deltas must be in [1, {p - 1}], got {n_deltas}')
    label = seed_label(seed, 'delta', f'L{length}', f'k{k}',
                       f'r{relative:+d}', f'base{base_row}')
    return random.Random(label).sample(range(1, p), n_deltas)


def select_base_rows(correct_rows, length, k, n_bases=32,
                     seed=BASE_SELECTION_SEED):
    if len(correct_rows) < n_bases:
        raise RuntimeError(
            f'L={length}, k={k}: only {len(correct_rows)}/{N_BASELINE} baseline '
            f'windows are full-vocab correct; need {n_bases} bases')
    label = seed_label(seed, 'bases', f'L{length}', f'k{k}')
    return random.Random(label).sample(list(correct_rows), n_bases)


def checkpoint_path(template, length, len8_ckpt=LEN8_CKPT):
    """len8 使用旧 len3to5->initLen35 特例，其余按 initLen(L-1)。"""
    if length == 8:
        return len8_ckpt
    return template.replace('{L-1}', str(length - 1)).replace('{L}', str(length))


def build_plan(n_bases=32, n_deltas=8):
    plan = []
    for length in LENGTHS:
        n_pairs = sum(len(legal_relative_positions(length, k))
                      for k in range(length))
        plan.append({
            'length': length,
            'n_blanks': length,
            'n_blank_relative': n_pairs,
            'n_baseline_predictions': length * N_BASELINE,
            'n_perturbations': n_pairs * n_bases * n_deltas,
        })
    return plan


def plan_totals(plan):
    return {
        'n_lengths': len(plan),
        'n_blanks': sum(x['n_blanks'] for x in plan),
        'n_blank_relative': sum(x['n_blank_relative'] for x in plan),
        'n_baseline_predictions': sum(x['n_baseline_predictions'] for x in plan),
        'n_perturbations': sum(x['n_perturbations'] for x in plan),
    }


def make_blank_inputs(windows, blank_pos, blank, eos):
    view = windows.clone()
    view[:, blank_pos] = blank
    return torch.cat([
        view, torch.full((len(view), 1), eos, dtype=torch.long),
    ], dim=1)


def _summary_count(summary, count_key, rate_key):
    if count_key in summary:
        return int(summary[count_key])
    return int(round(float(summary[rate_key]) * int(summary['n'])))


def aggregate_cells(summaries):
    """按 exact_count/N 聚合；output/nonnumeric 也使用计数加权。"""
    summaries = list(summaries)
    if not summaries:
        raise ValueError('cannot aggregate an empty cell list')
    n = sum(int(s['n']) for s in summaries)
    fits = {}
    for name in CANDIDATE_NAMES:
        exact_count = sum(int(s['fits'][name]['exact_count']) for s in summaries)
        fits[name] = {
            'k': int(summaries[0]['fits'][name]['k']),
            'exact_count': exact_count,
            'exact_match': exact_count / n,
        }
    output_change_count = sum(
        _summary_count(s, 'output_change_count', 'output_change_rate')
        for s in summaries)
    nonnumeric_count = sum(
        _summary_count(s, 'nonnumeric_count', 'nonnumeric_output_rate')
        for s in summaries)
    return {
        'n': n,
        'n_cells': len(summaries),
        'fits': fits,
        'output_change_count': output_change_count,
        'output_change_rate': output_change_count / n,
        'nonnumeric_count': nonnumeric_count,
        'nonnumeric_output_rate': nonnumeric_count / n,
    }


def aggregate_grouped(cells):
    """cells 为 (category, relative, summary) 列表。"""
    grouped = {}
    for category, relative, summary in cells:
        grouped.setdefault((category, relative), []).append(summary)
    result = {'by_category': {}}
    for category in CATEGORIES:
        by_relative = {}
        for relative in RELATIVES:
            values = grouped.get((category, relative))
            if values:
                by_relative[relative_key(relative)] = aggregate_cells(values)
        if by_relative:
            result['by_category'][category] = {'by_relative': by_relative}
    return result


def aggregate_by_length(by_length):
    output = {'by_length': {}, 'all': None}
    all_cells = []
    for length, length_report in by_length.items():
        cells = []
        for blank_report in length_report['by_blank'].values():
            category = blank_report['category']
            for rel_key, summary in blank_report['by_relative'].items():
                relative = int(rel_key)
                cells.append((category, relative, summary))
                all_cells.append((category, relative, summary))
        output['by_length'][str(length)] = aggregate_grouped(cells)
    output['all'] = aggregate_grouped(all_cells)
    return output


def build_len5_reference(data, source=None):
    """只读已有 len5 by_blank/summaries，并映射到五类及 ±1/±2。"""
    cells = []
    category_map = {}
    for k in range(5):
        blank_key = f'x{k}'
        blank = data['by_blank'][blank_key]
        category = blank_category(k, 5)
        category_map[blank_key] = category
        for position_key, summary in blank['summaries'].items():
            position = int(summary.get('position', position_key[1:]))
            if position != int(position_key[1:]):
                raise ValueError(f'inconsistent len5 position: {blank_key}/{position_key}')
            relative = position - k
            if relative in RELATIVES:
                cells.append((category, relative, summary))
    aggregated = aggregate_grouped(cells)
    return {
        'source': source,
        'category_map': category_map,
        'candidate_values': data.get('candidate_values'),
        'by_length': {'5': aggregated},
        'all': aggregated,
    }


def best_candidate(summary):
    """与 argmax 一样按固定候选顺序打破平局。"""
    name = max(CANDIDATE_NAMES,
               key=lambda n: int(summary['fits'][n]['exact_count']))
    return name, float(summary['fits'][name]['exact_match'])


def compare_cell(len5_summary, new_summary):
    len5_best, len5_rate = best_candidate(len5_summary)
    new_best, new_rate = best_candidate(new_summary)
    mae_pp = sum(
        abs(float(len5_summary['fits'][name]['exact_match'])
            - float(new_summary['fits'][name]['exact_match']))
        for name in CANDIDATE_NAMES
    ) * 100.0 / len(CANDIDATE_NAMES)
    return {
        'len5_best': len5_best,
        'len5_rate': len5_rate,
        'new_best': new_best,
        'new_rate': new_rate,
        'dominant_match': len5_best == new_best,
        'fit_vector_mae_pp': mae_pp,
    }


def _comparison_summary(rows):
    if not rows:
        return {'dominant_matches': 0, 'total': 0, 'mean_mae_pp': None}
    return {
        'dominant_matches': sum(int(row['dominant_match']) for row in rows),
        'total': len(rows),
        'mean_mae_pp': sum(row['fit_vector_mae_pp'] for row in rows) / len(rows),
    }


def compare_aggregates(new_aggregate, len5_aggregate):
    by_category = {}
    all_rows = []
    for category in CATEGORIES:
        new_rel = new_aggregate['by_category'].get(category, {}).get('by_relative', {})
        len5_rel = len5_aggregate['by_category'].get(category, {}).get('by_relative', {})
        rows = {}
        for relative in RELATIVES:
            key = relative_key(relative)
            if key in new_rel and key in len5_rel:
                rows[key] = compare_cell(len5_rel[key], new_rel[key])
        if rows:
            row_values = list(rows.values())
            all_rows.extend(row_values)
            by_category[category] = {
                'by_relative': rows,
                'summary': _comparison_summary(row_values),
            }
    return {'by_category': by_category, 'summary': _comparison_summary(all_rows)}


def build_comparison(new_aggregates, len5_reference):
    reference = len5_reference['by_length']['5']
    output = {'by_length': {}, 'all': None}
    for length, aggregate in new_aggregates['by_length'].items():
        output['by_length'][str(length)] = compare_aggregates(aggregate, reference)
    output['all'] = compare_aggregates(new_aggregates['all'], reference)
    return output


def probe_relative(model, base_inputs, base_rows, base_outputs, position,
                   length, k, relative, a, b, device, batch_size,
                   n_deltas=8, delta_seed=DELTA_SEED):
    """基底顺序在外、随机 d 在内，raw 的 base_row 指向固定 128 窗口。"""
    input_rows = []
    raw_base_rows = []
    raw_deltas = []
    for base_input, base_row in zip(base_inputs, base_rows):
        deltas = sample_deltas(length, k, relative, int(base_row),
                               n_deltas=n_deltas, p=P, seed=delta_seed)
        delta_tensor = torch.tensor(deltas, dtype=torch.long)
        repeated = base_input.unsqueeze(0).repeat(n_deltas, 1)
        repeated[:, position] = (repeated[:, position] + delta_tensor) % P
        input_rows.append(repeated)
        raw_base_rows.extend([int(base_row)] * n_deltas)
        raw_deltas.extend(deltas)
    inputs = torch.cat(input_rows)
    y1, full1 = predict(model, inputs, device, P, batch_size)
    y0 = base_outputs.repeat_interleave(n_deltas)
    d = torch.tensor(raw_deltas, dtype=torch.long)
    D = (y1 - y0) % P
    valid_numeric = full1 < P
    fits = {}
    for name, slope in candidate_values(a, b, P).items():
        residual = (D - slope * d) % P
        exact = (residual == 0) & valid_numeric
        fits[name] = {
            'k': slope,
            'exact_count': int(exact.sum()),
            'exact_match': float(exact.float().mean()),
            'circular_mae': circular_mae(residual, P),
        }
    n = len(inputs)
    output_change_count = int((D != 0).sum())
    nonnumeric_count = int((~valid_numeric).sum())
    summary = {
        'position': position,
        'relative': relative,
        'n': n,
        'output_change_count': output_change_count,
        'output_change_rate': output_change_count / n,
        'nonnumeric_count': nonnumeric_count,
        'nonnumeric_output_rate': nonnumeric_count / n,
        'fits': fits,
    }
    raw = {
        'base_row': raw_base_rows,
        'd': raw_deltas,
        'y0': y0.tolist(),
        'y1': y1.tolist(),
        'full1': full1.tolist(),
        'D': D.tolist(),
    }
    return summary, raw


def reconstruct_fixed_heldout(cfg, length, a, b, base_seed):
    """重建训练 seed42 的划分，再确定性地固定 128 个 held-out 窗口。"""
    random.seed(42)
    rule = LinearRecurrenceRule([a, b], P)
    dataset = RecurrenceDataset(
        p=P, recurrence_fn=rule.next_fn(), recurrence_name=rule.name,
        init_len=2, num_samples=cfg['num_train_samples'], length=length,
        verbose=False)
    dataset.run_full_split()
    test_state_indices = [
        int(seq[0]) + P * int(seq[1]) for seq in dataset.test_samples
    ]
    if len(test_state_indices) < N_BASELINE:
        raise RuntimeError(
            f'L={length}: only {len(test_state_indices)} held-out windows; '
            f'need {N_BASELINE}')
    label = seed_label(base_seed, 'fixed128', f'L{length}')
    heldout_rows = random.Random(label).sample(
        range(len(test_state_indices)), N_BASELINE)
    state_indices = [test_state_indices[row] for row in heldout_rows]
    windows = torch.tensor([
        rollout(index % P, index // P, a, b, P, length=length)
        for index in state_indices
    ], dtype=torch.long)
    # rollout 与 Dataset 必须使用完全相同的递推和窗口定义。
    for heldout_row, window in zip(heldout_rows, windows):
        if not torch.equal(window, dataset.test_samples[heldout_row]):
            raise RuntimeError(f'L={length}: rollout/held-out window mismatch')
    return windows, {
        'split_seed': 42,
        'heldout_total': len(test_state_indices),
        'fixed_n': N_BASELINE,
        'fixed_selection_seed': label,
        'heldout_rows': heldout_rows,
        'state_indices': state_indices,
        'clean_windows': windows.tolist(),
    }


def run_length(length, args):
    path = checkpoint_path(args.ckpt_template, length, args.len8_ckpt)
    print(f'\n{"=" * 78}\n[L={length}] 加载 {path}', flush=True)
    model, cfg, ckpt = load_model(path, args.device)
    if not (cfg['variant'] == 'fill_blank' and cfg['order'] == 2):
        raise ValueError(f'L={length}: expected order-2 fill_blank checkpoint')
    if cfg['train_len'] != length or cfg['p'] != P:
        raise ValueError(f'L={length}: checkpoint train_len/P does not match')
    if cfg.get('body_supervision') or cfg.get('const', 0):
        raise ValueError(f'L={length}: expected body_supervision=false, const=0')
    a, b = (int(x) for x in cfg['rules'][0])
    if (a, b) != (25, 56):
        raise ValueError(f'L={length}: expected a25b56, got a{a}b{b}')
    blank, eos = int(cfg['blank_token_id']), int(cfg['eos_token_id'])
    if (blank, eos, int(cfg['vocab_size'])) != (127, 128, 129):
        raise ValueError(f'L={length}: unexpected fill_blank vocabulary')

    windows, heldout = reconstruct_fixed_heldout(
        cfg, length, a, b, args.base_seed)
    print(f'[L={length}] held-out={heldout["heldout_total"]}, '
          f'固定窗口={N_BASELINE}, candidates={candidate_values(a, b, P)}',
          flush=True)
    length_report = {
        'ckpt': os.path.abspath(path),
        'model_config': cfg,
        'checkpoint_final_epoch': ckpt.get('final_epoch'),
        'checkpoint_best_accuracy': ckpt.get('best_accuracy'),
        'heldout': heldout,
        'by_blank': {},
    }
    for k in range(length):
        inputs = make_blank_inputs(windows, k, blank, eos)
        numeric0, full0 = predict(model, inputs, args.device, P, args.batch_size)
        targets = windows[:, k]
        full_correct = full0 == targets
        numeric_correct = numeric0 == targets
        correct_rows = full_correct.nonzero(as_tuple=False).flatten().tolist()
        selected = select_base_rows(
            correct_rows, length, k, n_bases=args.n_bases, seed=args.base_seed)
        base_inputs = inputs[selected]
        base_outputs = numeric0[selected]
        base_rows = [int(row) for row in selected]
        blank_report = {
            'blank_position': k,
            'category': blank_category(k, length),
            'accuracy': {
                'n': N_BASELINE,
                'full_vocab_correct_count': int(full_correct.sum()),
                'full_vocab_accuracy': float(full_correct.float().mean()),
                'numeric_only_correct_count': int(numeric_correct.sum()),
                'numeric_only_accuracy': float(numeric_correct.float().mean()),
            },
            'base_selection': {
                'seed': seed_label(args.base_seed, 'bases', f'L{length}', f'k{k}'),
                'eligible_correct_rows': correct_rows,
                'base_rows': base_rows,
                'base_state_indices': [heldout['state_indices'][row] for row in base_rows],
                'base_clean_windows': windows[selected].tolist(),
                'base_outputs': base_outputs.tolist(),
            },
            'by_relative': {},
        }
        for relative, position in legal_relative_positions(length, k):
            summary, raw = probe_relative(
                model, base_inputs, base_rows, base_outputs, position,
                length, k, relative, a, b, args.device, args.batch_size,
                n_deltas=args.n_deltas, delta_seed=args.delta_seed)
            summary['raw'] = raw
            blank_report['by_relative'][relative_key(relative)] = summary
        length_report['by_blank'][str(k)] = blank_report
        print(f'  L={length} k={k:2d} {blank_report["category"]:15s} '
              f'baseline={int(full_correct.sum())}/{N_BASELINE}, '
              f'eligible={len(correct_rows)}, bases={len(selected)}, '
              f'relatives={len(blank_report["by_relative"])}', flush=True)
    print(f'[L={length}] 完成；释放模型。', flush=True)
    del model, ckpt
    gc.collect()
    if args.device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return length_report


def print_comparison(comparison):
    print('\n' + '=' * 78 + '\n最终比较（len5 vs len8..64；MAE 单位为百分点）')
    entries = list(comparison['by_length'].items()) + [('all', comparison['all'])]
    for length, result in entries:
        summary = result['summary']
        mean_mae = summary['mean_mae_pp']
        mean_text = 'n/a' if mean_mae is None else f'{mean_mae:.4f}'
        print(f'L={length}: dominant={summary["dominant_matches"]}/{summary["total"]}, '
              f'mean_mae_pp={mean_text}')
        for category in CATEGORIES:
            if category not in result['by_category']:
                continue
            cat_summary = result['by_category'][category]['summary']
            cat_mae = cat_summary['mean_mae_pp']
            print(f'  {category:15s} dominant={cat_summary["dominant_matches"]}/'
                  f'{cat_summary["total"]}, mean_mae_pp={cat_mae:.4f}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt-template', default=DEFAULT_CKPT_TEMPLATE)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--len8-ckpt', default=LEN8_CKPT,
                        help='旧课程 len3to5 -> len8_initLen35 的 best checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--n-bases', type=int, default=32)
    parser.add_argument('--n-deltas', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--base-seed', type=int, default=BASE_SELECTION_SEED)
    parser.add_argument('--delta-seed', type=int, default=DELTA_SEED)
    parser.add_argument('--len5-reference', default=DEFAULT_LEN5,
                        help='已有 len5 原始 JSON；只读取，不重新运行 len5')
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印静态计划，不读模型/len5，也不创建输出')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.threads <= 0 or args.batch_size <= 0 or args.n_bases <= 0:
        raise ValueError('threads, batch-size and n-bases must be positive')
    if not 1 <= args.n_deltas < P:
        raise ValueError(f'n-deltas must be in [1, {P - 1}]')
    plan = build_plan(args.n_bases, args.n_deltas)
    totals = plan_totals(plan)
    print('静态实验计划:')
    for item in plan:
        length = item['length']
        print(f'  L={length:2d}: blanks={item["n_blanks"]}, '
              f'blank-relative={item["n_blank_relative"]}, '
              f'baseline={item["n_baseline_predictions"]}, '
              f'perturbations={item["n_perturbations"]}, '
              f'ckpt={checkpoint_path(args.ckpt_template, length, args.len8_ckpt)}')
    print(f'  合计: {totals}')
    if args.dry_run:
        print('[dry-run] 未加载模型，未读取 len5，未创建输出。')
        return

    torch.set_num_threads(args.threads)
    with open(args.len5_reference) as f:
        len5_data = json.load(f)
    len5_reference = build_len5_reference(
        len5_data, source=os.path.abspath(args.len5_reference))
    expected_candidates = candidate_values(25, 56, P)
    if len5_reference['candidate_values'] != expected_candidates:
        raise ValueError('len5 reference candidate values are not a25b56/P127')

    by_length = {}
    for length in LENGTHS:
        by_length[str(length)] = run_length(length, args)
    aggregates = aggregate_by_length(by_length)
    comparison = build_comparison(aggregates, len5_reference)
    report = {
        'task': 'fill_blank a25b56 random +/-1/+/-2 probe, len8..64 stride8',
        'lengths': LENGTHS,
        'p': P,
        'device': args.device,
        'threads': args.threads,
        'batch_size': args.batch_size,
        'n_baseline_per_blank': N_BASELINE,
        'n_bases_per_blank': args.n_bases,
        'n_deltas_per_base': args.n_deltas,
        'base_selection_seed': args.base_seed,
        'delta_seed': args.delta_seed,
        'candidate_values': expected_candidates,
        'fit_definition': (
            'exact fraction of (D-k_candidate*d)%127==0; nonnumeric full-vocab '
            'outputs count as mismatch'),
        'D_definition': '(y1-y0)%127; y0/y1 are numeric-class argmax at EOS',
        'raw_base_row_definition': 'row in the fixed 128 held-out windows for this L',
        'plan': {'by_length': plan, 'totals': totals},
        'by_length': by_length,
        'aggregates': aggregates,
        'len5_reference': len5_reference,
        'comparison': comparison,
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print_comparison(comparison)
    print(f'\n[完成] raw、聚合、len5 参考与比较已写入: {out}')


if __name__ == '__main__':
    main()
