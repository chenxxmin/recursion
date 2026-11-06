#!/usr/bin/env python3
"""从 len5 fill_blank 原始 JSON 提取“其他”样本分布，不加载模型。"""
import argparse
import collections
import json
import os


def parse_slopes(text):
    try:
        values = [int(x.strip()) for x in text.split(',') if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('slopes 必须是逗号分隔整数') from exc
    if not values:
        raise argparse.ArgumentTypeError('至少需要一个排除斜率')
    return values


def counter_report(values, top):
    counter = collections.Counter(values)
    n = len(values)
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return {
        'unique_count': len(counter),
        'top_coverage': {
            str(k): (sum(v for _, v in ranked[:k]) / n if n else 0.0)
            for k in (1, 3, 5, 10, 20)
        },
        'counts': [{'value': value, 'count': count, 'rate': count / n}
                   for value, count in ranked],
    }


def print_metric(name, report, top):
    print(f'\n{name}: unique={report["unique_count"]}')
    print('  coverage: ' + ', '.join(
        f'top{k}={100 * rate:.2f}%' for k, rate in report['top_coverage'].items()))
    for row in report['counts'][:top]:
        print(f'  {row["value"]}: {row["count"]} ({100 * row["rate"]:.2f}%)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='len5 all_blank_linear JSON')
    parser.add_argument('--blank', required=True, help='例如 x1')
    parser.add_argument('--position', required=True, help='例如 x2')
    parser.add_argument('--slopes', required=True, type=parse_slopes,
                        help='“其他”需要排除的斜率，逗号分隔，如 61,88')
    parser.add_argument('--p', type=int, default=127)
    parser.add_argument('--top', type=int, default=12)
    parser.add_argument('--out', help='可选：保存完整分布 JSON')
    args = parser.parse_args()
    if args.top <= 0:
        parser.error('--top 必须为正数')

    with open(args.data, encoding='utf-8') as handle:
        data = json.load(handle)
    blank = data['by_blank'][args.blank]
    raw = blank['raw_records'][args.position]
    required = ['base_row', 'd', 'D', 'perturbed_numeric_output']
    for key in required:
        if key not in raw:
            raise KeyError(f'raw 缺少字段 {key}')
    total = len(raw['d'])
    if any(len(raw[key]) != total for key in required):
        raise ValueError('raw 各字段长度不一致')

    selected = []
    for i, (d, D) in enumerate(zip(raw['d'], raw['D'])):
        if d == 0:
            raise ValueError('等效斜率要求 d 非零')
        if all((D - slope * d) % args.p != 0 for slope in args.slopes):
            selected.append((i, d, D))
    n = len(selected)
    print(f'筛选: blank={args.blank}, position={args.position}, '
          f'exclude_slopes={args.slopes}, P={args.p}')
    print(f'其他样本: {n}/{total} = {100 * n / total:.4f}%')
    if n == 0:
        print('没有未解释样本。')

    metrics = {
        'output': counter_report(
            [raw['perturbed_numeric_output'][i] for i, _, _ in selected], args.top),
        'D': counter_report([D for _, _, D in selected], args.top),
        'effective_slope': counter_report(
            [(D * pow(d, -1, args.p)) % args.p for _, d, D in selected], args.top),
        'delta': counter_report([d for _, d, _ in selected], args.top),
    }
    for name, report in metrics.items():
        print_metric(name, report, args.top)

    base_counter = collections.Counter(raw['base_row'][i] for i, _, _ in selected)
    base_ranked = sorted(base_counter.items(), key=lambda item: (-item[1], item[0]))
    base_windows = blank.get('base_clean_windows', [])
    base_counts = []
    for row, count in base_ranked:
        item = {'base_row': row, 'count': count, 'rate': count / n}
        if row < len(base_windows):
            item['window'] = base_windows[row]
        base_counts.append(item)
    print(f'\n基底窗口: unique={len(base_counts)}')
    for item in base_counts[:args.top]:
        print(f'  row={item["base_row"]}: {item["count"]} '
              f'({100 * item["rate"]:.2f}%) window={item.get("window")}')

    result = {
        'source': os.path.abspath(args.data),
        'blank': args.blank,
        'position': args.position,
        'exclude_slopes': args.slopes,
        'p': args.p,
        'total': total,
        'unmatched_count': n,
        'unmatched_rate': n / total,
        'zero_D_count': sum(D == 0 for _, _, D in selected),
        'metrics': metrics,
        'base_distribution': base_counts,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
