#!/usr/bin/env python3
"""fill_blank a25b56 len8..64 的远距离随机扰动探针。

对每个 blank k 枚举所有 |relative| >= 3 的位置。默认每个 cell 使用
16 个基线答对的 held-out 窗口、每个窗口抽 4 个不同非零模 127 增量。
仅异常 cell 保存 raw：D=0 命中率 <0.95，或最佳非零候选命中率 >=0.10。
checkpoint 只读；dry-run 不加载模型、不创建输出。
"""
import argparse
import gc
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from probe_fillblank_len8to64_stride8_random import (  # noqa: E402
    BASE_SELECTION_SEED, CANDIDATE_NAMES, CATEGORIES, DEFAULT_CKPT_TEMPLATE,
    DELTA_SEED, LEN8_CKPT, N_BASELINE, P, aggregate_cells as _aggregate_fits,
    blank_category, candidate_values, checkpoint_path, load_model,
    make_blank_inputs, predict, probe_relative, reconstruct_fixed_heldout,
    relative_key, seed_label, select_base_rows,
)

LENGTHS = list(range(8, 65, 8))
NONZERO_CANDIDATE_NAMES = tuple(n for n in CANDIDATE_NAMES if n != '0')
ZERO_MATCH_THRESHOLD = 0.95
BEST_NONZERO_THRESHOLD = 0.10
TOP_ANOMALIES = 20
DEFAULT_OUT = os.path.join(
    ROOT, 'reports', 'data', 'fillblank_len8to64_far_random_a25b56.json')


def far_relative_positions(length, k):
    """返回 (relative=j-k, j)，先按 |relative|、再按 relative 排序。"""
    if not 0 <= k < length:
        raise ValueError(f'invalid blank position k={k}, length={length}')
    pairs = [(j - k, j) for j in range(length) if abs(j - k) >= 3]
    return sorted(pairs, key=lambda pair: (abs(pair[0]), pair[0]))


def build_plan(n_bases=16, n_deltas=4):
    plan = []
    for length in LENGTHS:
        n_pairs = sum(len(far_relative_positions(length, k))
                      for k in range(length))
        expected = (length - 2) * (length - 3)
        if n_pairs != expected:
            raise AssertionError(f'L={length}: far-cell count {n_pairs} != {expected}')
        plan.append({
            'length': length,
            'n_blanks': length,
            'n_far_cells': n_pairs,
            'n_baseline_predictions': length * N_BASELINE,
            'n_per_cell': n_bases * n_deltas,
            'n_perturbations': n_pairs * n_bases * n_deltas,
        })
    return plan


def plan_totals(plan):
    return {
        'n_lengths': len(plan),
        'n_blanks': sum(x['n_blanks'] for x in plan),
        'n_far_cells': sum(x['n_far_cells'] for x in plan),
        'n_baseline_predictions': sum(x['n_baseline_predictions'] for x in plan),
        'n_perturbations': sum(x['n_perturbations'] for x in plan),
    }


def best_nonzero(summary):
    """排除候选 '0'；并列时保持七候选中的固定先后顺序。"""
    name = max(NONZERO_CANDIDATE_NAMES,
               key=lambda n: int(summary['fits'][n]['exact_count']))
    return name, float(summary['fits'][name]['exact_match'])


def add_fit_metrics(summary):
    """添加 zero/best_nonzero 便捷字段，数值仍由精确计数计算。"""
    n = int(summary['n'])
    zero_count = int(summary['fits']['0']['exact_count'])
    best_name, _ = best_nonzero(summary)
    best_count = int(summary['fits'][best_name]['exact_count'])
    summary.update({
        'zero_exact_count': zero_count,
        'zero_rate': zero_count / n,
        'best_nonzero': best_name,
        'best_nonzero_exact_count': best_count,
        'best_nonzero_rate': best_count / n,
    })
    return summary


def is_anomaly(summary):
    """阈值固定，不随 n-bases/n-deltas 改变。"""
    zero_rate = int(summary['fits']['0']['exact_count']) / int(summary['n'])
    best_name = max(
        NONZERO_CANDIDATE_NAMES,
        key=lambda name: int(summary['fits'][name]['exact_count']))
    best_rate = (int(summary['fits'][best_name]['exact_count'])
                 / int(summary['n']))
    return zero_rate < ZERO_MATCH_THRESHOLD or best_rate >= BEST_NONZERO_THRESHOLD


def attach_raw_if_anomaly(summary, raw):
    """异常 cell 嵌入 raw；正常 cell 明确丢弃 raw。"""
    if is_anomaly(summary):
        summary['raw'] = raw
        return True
    summary.pop('raw', None)
    return False


def anomaly_record(length, k, category, relative, summary):
    if not is_anomaly(summary):
        return None
    best_name, best_rate = best_nonzero(summary)
    return {
        'L': int(length),
        'k': int(k),
        'category': category,
        'relative': int(relative),
        'zero_rate': float(summary['fits']['0']['exact_match']),
        'best_nonzero': best_name,
        'best_nonzero_rate': best_rate,
        'output_change_rate': float(summary['output_change_rate']),
        'raw_included': 'raw' in summary,
    }


def aggregate_cells(summaries):
    """复用原计数加权聚合，再补 zero/best_nonzero；绝不平均百分比。"""
    return add_fit_metrics(_aggregate_fits(summaries))


def iter_cells(by_length):
    for length_key in sorted(by_length, key=int):
        length_report = by_length[length_key]
        for blank_key in sorted(length_report['by_blank'], key=int):
            blank = length_report['by_blank'][blank_key]
            for rel_key, summary in blank['by_relative'].items():
                yield (int(length_key), int(blank_key), blank['category'],
                       int(rel_key), summary)


def build_aggregates(by_length):
    grouped = {
        'by_length': {},
        'by_abs_distance': {},
        'by_category': {},
        'by_category_abs_distance': {},
    }
    for length, _k, category, relative, summary in iter_cells(by_length):
        distance = abs(relative)
        grouped['by_length'].setdefault(length, []).append(summary)
        grouped['by_abs_distance'].setdefault(distance, []).append(summary)
        grouped['by_category'].setdefault(category, []).append(summary)
        cat_dist = grouped['by_category_abs_distance'].setdefault(category, {})
        cat_dist.setdefault(distance, []).append(summary)

    result = {
        'by_length': {
            str(key): aggregate_cells(grouped['by_length'][key])
            for key in sorted(grouped['by_length'])
        },
        'by_abs_distance': {
            str(key): aggregate_cells(grouped['by_abs_distance'][key])
            for key in sorted(grouped['by_abs_distance'])
        },
        'by_category': {},
        'by_category_abs_distance': {},
    }
    for category in CATEGORIES:
        if category in grouped['by_category']:
            result['by_category'][category] = aggregate_cells(
                grouped['by_category'][category])
        if category in grouped['by_category_abs_distance']:
            distances = grouped['by_category_abs_distance'][category]
            result['by_category_abs_distance'][category] = {
                str(distance): aggregate_cells(distances[distance])
                for distance in sorted(distances)
            }
    return result


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
    anomalies = []
    n_cells = 0
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
            },
            'by_relative': {},
        }
        for relative, position in far_relative_positions(length, k):
            summary, raw = probe_relative(
                model, base_inputs, base_rows, base_outputs, position,
                length, k, relative, a, b, args.device, args.batch_size,
                n_deltas=args.n_deltas, delta_seed=args.delta_seed)
            add_fit_metrics(summary)
            attach_raw_if_anomaly(summary, raw)
            blank_report['by_relative'][relative_key(relative)] = summary
            n_cells += 1
            anomaly = anomaly_record(
                length, k, blank_report['category'], relative, summary)
            if anomaly is not None:
                anomalies.append(anomaly)
        length_report['by_blank'][str(k)] = blank_report
        print(f'  L={length} k={k:2d} baseline={int(full_correct.sum())}/'
              f'{N_BASELINE}, bases={len(selected)}, '
              f'far_cells={len(blank_report["by_relative"])}, '
              f'anomalies={len(anomalies)}', flush=True)

    print(f'[L={length}] 完成：cells={n_cells}, '
          f'perturbations={n_cells * args.n_bases * args.n_deltas}, '
          f'anomalies={len(anomalies)}；释放模型。', flush=True)
    del model, ckpt
    gc.collect()
    if args.device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return length_report, anomalies


def print_final(totals, anomalies):
    ordered = sorted(
        anomalies,
        key=lambda row: (-row['best_nonzero_rate'], row['zero_rate'],
                         row['L'], row['k'], row['relative']))
    print('\n' + '=' * 78)
    print(f'最终总数：cells={totals["n_far_cells"]}, '
          f'perturbations={totals["n_perturbations"]}, '
          f'anomalies={len(anomalies)}')
    print(f'Top anomalies（最多 {TOP_ANOMALIES}；先 best_nonzero 高，再 zero 低）:')
    for rank, row in enumerate(ordered[:TOP_ANOMALIES], 1):
        print(f'  #{rank:2d} L={row["L"]} k={row["k"]} '
              f'{row["category"]} relative={row["relative"]:+d} '
              f'zero={row["zero_rate"]:.4f} '
              f'best_nonzero={row["best_nonzero"]}:'
              f'{row["best_nonzero_rate"]:.4f} '
              f'output_change={row["output_change_rate"]:.4f} '
              f'raw_included={row["raw_included"]}')
    if not ordered:
        print('  无')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt-template', default=DEFAULT_CKPT_TEMPLATE)
    parser.add_argument('--len8-ckpt', default=LEN8_CKPT,
                        help='旧课程 len3to5 -> len8_initLen35 的 best checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--n-bases', type=int, default=16)
    parser.add_argument('--n-deltas', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--base-seed', type=int, default=BASE_SELECTION_SEED)
    parser.add_argument('--delta-seed', type=int, default=DELTA_SEED)
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印静态计划，不读模型，也不创建输出')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.threads <= 0 or args.batch_size <= 0:
        raise ValueError('threads and batch-size must be positive')
    if not 1 <= args.n_bases <= N_BASELINE:
        raise ValueError(f'n-bases must be in [1, {N_BASELINE}]')
    if not 1 <= args.n_deltas < P:
        raise ValueError(f'n-deltas must be in [1, {P - 1}]')
    plan = build_plan(args.n_bases, args.n_deltas)
    totals = plan_totals(plan)
    print('静态实验计划:')
    for item in plan:
        length = item['length']
        print(f'  L={length:2d}: blanks={item["n_blanks"]}, '
              f'far_cells={item["n_far_cells"]}, '
              f'baseline={item["n_baseline_predictions"]}, '
              f'N/cell={item["n_per_cell"]}, '
              f'perturbations={item["n_perturbations"]}, '
              f'ckpt={checkpoint_path(args.ckpt_template, length, args.len8_ckpt)}')
    print(f'  合计: {totals}')
    print(f'  异常/raw 阈值: zero<{ZERO_MATCH_THRESHOLD} 或 '
          f'best_nonzero>={BEST_NONZERO_THRESHOLD}（候选 0 不参与 best_nonzero）')
    if args.dry_run:
        print('[dry-run] 未加载模型，未创建输出。')
        return

    torch.set_num_threads(args.threads)
    by_length = {}
    anomalies = []
    for length in LENGTHS:
        length_report, length_anomalies = run_length(length, args)
        by_length[str(length)] = length_report
        anomalies.extend(length_anomalies)
    aggregates = build_aggregates(by_length)
    report = {
        'task': 'fill_blank a25b56 far random probe, len8..64 stride8, |relative|>=3',
        'plan': {'by_length': plan, 'totals': totals},
        'config': {
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
            'ckpt_template': args.ckpt_template,
            'len8_ckpt': args.len8_ckpt,
            'candidate_values': candidate_values(25, 56, P),
            'zero_match_threshold': ZERO_MATCH_THRESHOLD,
            'best_nonzero_threshold': BEST_NONZERO_THRESHOLD,
            'fit_definition': (
                'exact fraction of (D-k_candidate*d)%127==0; nonnumeric '
                'full-vocab outputs count as mismatch'),
            'D_definition': '(y1-y0)%127; y0/y1 are numeric-class argmax at EOS',
            'raw_policy': 'raw embedded only in anomaly summaries',
            'raw_base_row_definition': 'row in the fixed 128 held-out windows for this L',
        },
        'by_length': by_length,
        'aggregates': aggregates,
        'anomalies': anomalies,
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print_final(totals, anomalies)
    print(f'\n[完成] summaries、计数加权聚合与 anomalies 已写入: {out}')


if __name__ == '__main__':
    main()
