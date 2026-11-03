#!/usr/bin/env python3
"""len5 fill_blank 五个空格位置的模块化因果扰动分析。

复用 last_blank 探针的模型加载和拟合函数。每个空格位置均从 held-out 且
原预测正确的窗口中抽取 32 个基底，对其余四个位置分别枚举全部非零 d。
"""
import argparse
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
    CANDIDATE_NAMES, candidate_values, fit_position, load_model, predict, rollout,
)


def make_blank_inputs(windows, blank_pos, blank, eos):
    view = windows.clone()
    view[:, blank_pos] = blank
    return torch.cat([view, torch.full((len(view), 1), eos, dtype=torch.long)], dim=1)


def manual_d1_examples(model, seq, a, b, blank_pos, blank, eos, p, device, batch_size):
    windows = torch.tensor([seq], dtype=torch.long)
    base = make_blank_inputs(windows, blank_pos, blank, eos)
    y0, full0 = predict(model, base, device, p, batch_size)
    records = []
    for j in range(5):
        if j == blank_pos:
            continue
        perturbed = base.clone()
        old = int(perturbed[0, j])
        perturbed[0, j] = (old + 1) % p
        y1, full1 = predict(model, perturbed, device, p, batch_size)
        D = (int(y1[0]) - int(y0[0])) % p
        matched = [name for name, k in candidate_values(a, b, p).items()
                   if D == k and int(full1[0]) < p]
        records.append({
            'position': j, 'd': 1, 'input': perturbed[0].tolist(),
            'baseline_output': int(y0[0]), 'numeric_output': int(y1[0]),
            'full_vocab_output': int(full1[0]), 'D': D,
            'matched_candidates': matched,
        })
    return {
        'clean_sequence': seq, 'base_input': base[0].tolist(),
        'baseline_numeric_output': int(y0[0]),
        'baseline_full_vocab_output': int(full0[0]), 'records': records,
    }


def print_fit_table(blank_pos, summaries):
    print(f'\n  空格 x{blank_pos}：D=k*d (mod P) 精确命中率')
    print(' | '.join(['扰动位置'] + CANDIDATE_NAMES))
    print(' | '.join(['---'] + ['---:'] * len(CANDIDATE_NAMES)))
    for j in range(5):
        if j == blank_pos:
            continue
        summary = summaries[f'x{j}']
        row = [f'x{j}'] + [f'{summary["fits"][name]["exact_match"] * 100:.2f}%'
                           for name in CANDIDATE_NAMES]
        print(' | '.join(row))
        best = max(CANDIDATE_NAMES,
                   key=lambda name: summary['fits'][name]['exact_match'])
        fit = summary['fits'][best]
        print(f'    best x{j}: {best}={fit["k"]}, '
              f'{fit["exact_count"]}/{summary["n"]}, '
              f'输出改变率={summary["output_change_rate"] * 100:.2f}%, '
              f'非数值率={summary["nonnumeric_output_rate"] * 100:.2f}%')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--n-bases', type=int, default=32)
    ap.add_argument('--seed', type=int, default=20261014)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--out', default='reports/data/fillblank_len5_all_blank_linear_a25b56.json')
    args = ap.parse_args()

    print('[步骤 1] 加载模型并构造 len5 全状态空间')
    print(f'  ckpt: {args.ckpt}')
    model, cfg, ckpt = load_model(args.ckpt, args.device)
    p, a, b = cfg['p'], cfg['rules'][0][0], cfg['rules'][0][1]
    blank, eos = cfg['blank_token_id'], cfg['eos_token_id']
    assert cfg['variant'] == 'fill_blank' and cfg['order'] == 2
    assert cfg['train_len'] == 5 and not cfg['body_supervision']
    print(f'  rules=[[{a},{b}]], P={p}, BLANK={blank}, EOS={eos}, '
          f'final_epoch={ckpt.get("final_epoch")}, device={args.device}')
    windows = torch.tensor([rollout(i % p, i // p, a, b, p)
                            for i in range(p * p)], dtype=torch.long)
    targets_by_blank = [windows[:, k] for k in range(5)]

    print('\n[步骤 2] 重建 seed42 held-out 划分并打印候选斜率')
    random.seed(42)
    rule = LinearRecurrenceRule([a, b], p)
    ds = RecurrenceDataset(p=p, recurrence_fn=rule.next_fn(),
                           recurrence_name=rule.name, init_len=2,
                           num_samples=cfg['num_train_samples'], length=5,
                           verbose=False)
    ds.run_full_split()
    test_indices = torch.tensor([int(seq[0]) + p * int(seq[1])
                                 for seq in ds.test_samples], dtype=torch.long)
    assert len(test_indices) + len(ds.train_samples) == p * p
    candidates = candidate_values(a, b, p)
    print('  ' + '; '.join(f'{name}={candidates[name]}' for name in CANDIDATE_NAMES))
    print(f'  每个空格位置抽取 {args.n_bases} 个 held-out 正确基底；'
          f'每扰动位置 N={args.n_bases * (p - 1)}。')
    print('  D=(y_perturbed-y_base) mod P；拟合为精确同余命中率；'
          '非数值输出计入分母且不匹配。')

    report = {
        'ckpt': os.path.abspath(args.ckpt), 'model_config': cfg,
        'checkpoint_final_epoch': ckpt.get('final_epoch'),
        'checkpoint_best_accuracy': ckpt.get('best_accuracy'),
        'state_index_convention': 'index=x0+P*x1',
        'candidate_values': candidates, 'n_bases_per_blank': args.n_bases,
        'base_selection_seed': args.seed, 'delta_values': list(range(1, p)),
        'fit_definition': 'exact fraction of (D-k*d)%P==0; nonnumeric outputs count as mismatch',
        'D_definition': '(perturbed_numeric_output-baseline_output)%P',
        'by_blank': {},
    }
    manual_seq = rollout(1, 2, a, b, p)

    for blank_pos in range(5):
        print(f'\n{"=" * 72}\n[空格位置 x{blank_pos}]')
        inputs = make_blank_inputs(windows, blank_pos, blank, eos)
        numeric0, full0 = predict(model, inputs, args.device, p, args.batch_size)
        targets = targets_by_blank[blank_pos]
        correct = full0 == targets
        test_correct = correct[test_indices]
        numeric_test_correct = numeric0[test_indices] == targets[test_indices]
        print(f'  held-out 完整词表: {int(test_correct.sum())}/{len(test_indices)} = '
              f'{test_correct.float().mean().item() * 100:.4f}%')
        print(f'  held-out 数值类:   {int(numeric_test_correct.sum())}/{len(test_indices)} = '
              f'{numeric_test_correct.float().mean().item() * 100:.4f}%')
        print(f'  全状态完整词表:    {int(correct.sum())}/{p * p} = '
              f'{correct.float().mean().item() * 100:.4f}%')

        correct_test_indices = test_indices[test_correct]
        generator = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(len(correct_test_indices), generator=generator)
        selected = correct_test_indices[permutation[:args.n_bases]]
        assert len(selected) == args.n_bases
        base_inputs = inputs[selected]
        base_outputs = numeric0[selected]
        print(f'  基底状态编号: {selected.tolist()}')

        summaries, raw_records = {}, {}
        for j in range(5):
            if j == blank_pos:
                continue
            summary, raw = fit_position(model, base_inputs, base_outputs, j, a, b, p,
                                        args.device, args.batch_size)
            summaries[f'x{j}'] = summary
            raw_records[f'x{j}'] = raw
        print_fit_table(blank_pos, summaries)

        example = manual_d1_examples(model, manual_seq, a, b, blank_pos, blank, eos,
                                     p, args.device, args.batch_size)
        print(f'  手动样例干净序列={manual_seq}, 输入={example["base_input"]}, '
              f'输出={example["baseline_numeric_output"]}, 真值={manual_seq[blank_pos]}')
        for record in example['records']:
            print(f'    d=1, x{record["position"]}: y={record["baseline_output"]}->'
                  f'{record["numeric_output"]}, D={record["D"]}, '
                  f'match={record["matched_candidates"]}')

        report['by_blank'][f'x{blank_pos}'] = {
            'accuracy': {
                'heldout_n': len(test_indices),
                'heldout_full_vocab': test_correct.float().mean().item(),
                'heldout_numeric_only': numeric_test_correct.float().mean().item(),
                'all_state_full_vocab': correct.float().mean().item(),
            },
            'base_state_indices': selected.tolist(),
            'base_clean_windows': windows[selected].tolist(),
            'base_outputs': base_outputs.tolist(),
            'summaries': summaries, 'manual_example': example,
            'raw_records': raw_records,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n[完成] 五个空格位置的汇总与全部原始记录已写入: {args.out}')


if __name__ == '__main__':
    main()
