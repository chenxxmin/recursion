#!/usr/bin/env python3
"""len5 corrupt_fix/value_only：五错误位置 × 五扰动位置的模 P 拟合。

基线错误值由固定 seed 独立抽样。扰动错误位时，若错误值恰好恢复为真值，
该样本单独标记为 clean_hit，不与仍然含错误的样本混合解释。
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


def make_corrupt_inputs(windows, corrupt_pos, wrong_offsets, eos, p):
    view = windows.clone()
    target = view[:, corrupt_pos]
    view[:, corrupt_pos] = (target + wrong_offsets) % p
    return torch.cat([view, torch.full((len(view), 1), eos, dtype=torch.long)], dim=1)


def add_clean_hit_metrics(summary, raw, base_wrong_offsets, corrupt_pos, perturb_pos,
                          candidate_map, p):
    """为错误位自身的扰动拆出 clean_hit / nonclean 两种拟合口径。"""
    if perturb_pos != corrupt_pos:
        summary['clean_hit_count'] = 0
        return
    offsets = torch.tensor(base_wrong_offsets, dtype=torch.long)
    rows = torch.tensor(raw['base_row'], dtype=torch.long)
    d = torch.tensor(raw['d'], dtype=torch.long)
    clean_delta = (p - offsets) % p
    clean_hit = d == clean_delta[rows]
    raw['clean_hit'] = clean_hit.tolist()
    raw['clean_delta_per_trial'] = clean_delta[rows].tolist()
    D = torch.tensor(raw['D'], dtype=torch.long)
    full = torch.tensor(raw['perturbed_full_vocab_output'], dtype=torch.long)
    summary['clean_hit_count'] = int(clean_hit.sum())
    summary['nonclean_count'] = int((~clean_hit).sum())
    summary['clean_hit_output_change_rate'] = (
        (D[clean_hit] != 0).float().mean().item() if clean_hit.any() else None)
    summary['nonclean_output_change_rate'] = (
        (D[~clean_hit] != 0).float().mean().item() if (~clean_hit).any() else None)
    for name, k in candidate_map.items():
        exact = ((D - k * d) % p == 0) & (full < p)
        for label, mask in [('clean', clean_hit), ('nonclean', ~clean_hit)]:
            count = int(mask.sum())
            summary['fits'][name][f'{label}_exact_count'] = int((exact & mask).sum())
            summary['fits'][name][f'{label}_exact_match'] = (
                (exact & mask).sum().item() / count if count else None)
            summary['fits'][name][f'{label}_n'] = count


def print_fit_table(corrupt_pos, summaries):
    print(f'\n  错误位置 x{corrupt_pos}：D=k*d (mod P) 精确命中率')
    print(' | '.join(['扰动位置'] + CANDIDATE_NAMES))
    print(' | '.join(['---'] + ['---:'] * len(CANDIDATE_NAMES)))
    for j in range(5):
        summary = summaries[f'x{j}']
        row = [f'x{j}' + ('（错误位）' if j == corrupt_pos else '')]
        row += [f'{summary["fits"][name]["exact_match"] * 100:.2f}%'
                for name in CANDIDATE_NAMES]
        print(' | '.join(row))
        best = max(CANDIDATE_NAMES,
                   key=lambda name: summary['fits'][name]['exact_match'])
        fit = summary['fits'][best]
        extra = ''
        if j == corrupt_pos:
            extra = (f', clean_hit={summary["clean_hit_count"]}, '
                     f'nonclean best={summary["fits"][best]["nonclean_exact_match"] * 100:.2f}%')
        print(f'    best x{j}: {best}={fit["k"]}, '
              f'{fit["exact_count"]}/{summary["n"]}, '
              f'输出改变率={summary["output_change_rate"] * 100:.2f}%' + extra)


def manual_examples(model, seq, a, b, corrupt_pos, eos, p, device, batch_size):
    windows = torch.tensor([seq], dtype=torch.long)
    wrong_offset = 37
    offsets = torch.tensor([wrong_offset], dtype=torch.long)
    base = make_corrupt_inputs(windows, corrupt_pos, offsets, eos, p)
    y0, full0 = predict(model, base, device, p, batch_size)
    records = []
    for j in range(5):
        deltas = [1]
        if j == corrupt_pos:
            deltas.append((p - wrong_offset) % p)  # 错值恢复为正确值
        for d in deltas:
            perturbed = base.clone()
            old = int(perturbed[0, j])
            perturbed[0, j] = (old + d) % p
            y1, full1 = predict(model, perturbed, device, p, batch_size)
            D = (int(y1[0]) - int(y0[0])) % p
            matched = [] if d != 1 else [
                name for name, k in candidate_values(a, b, p).items()
                if D == k and int(full1[0]) < p]
            records.append({
                'position': j, 'd': d, 'old_value': old,
                'new_value': int(perturbed[0, j]), 'input': perturbed[0].tolist(),
                'baseline_output': int(y0[0]), 'numeric_output': int(y1[0]),
                'full_vocab_output': int(full1[0]), 'D': D,
                'clean_hit': j == corrupt_pos and d == (p - wrong_offset) % p,
                'matched_candidates_for_d1': matched,
            })
    return {
        'clean_sequence': seq, 'wrong_offset': wrong_offset,
        'base_input': base[0].tolist(), 'true_value': seq[corrupt_pos],
        'baseline_numeric_output': int(y0[0]),
        'baseline_full_vocab_output': int(full0[0]), 'records': records,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--n-bases', type=int, default=32)
    ap.add_argument('--seed', type=int, default=20261014, help='基底抽样 seed')
    ap.add_argument('--corruption-seed', type=int, default=20261103,
                    help='每状态/位置固定一个错误值的 seed')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--out', default='reports/data/corruptfix_len5_all_positions_linear_a25b56.json')
    args = ap.parse_args()
    if args.device == 'cpu':
        torch.set_num_threads(args.threads)

    print('[步骤 1] 加载 len5 value_only 模型并构造全状态窗口')
    print(f'  ckpt: {args.ckpt}')
    model, cfg, ckpt = load_model(args.ckpt, args.device)
    p, a, b = cfg['p'], cfg['rules'][0][0], cfg['rules'][0][1]
    eos = cfg['eos_token_id']
    assert cfg['variant'] == 'corrupt_fix' and cfg['value_only']
    assert cfg['train_len'] == 5 and cfg['order'] == 2 and not cfg['body_supervision']
    print(f'  rules=[[{a},{b}]], P={p}, EOS={eos}, final_epoch={ckpt.get("final_epoch")}, '
          f'training best={ckpt.get("best_accuracy")}, device={args.device}')
    windows = torch.tensor([rollout(i % p, i // p, a, b, p)
                            for i in range(p * p)], dtype=torch.long)

    print('\n[步骤 2] 重建 held-out，并为每状态/错误位置固定一个非零错误偏移')
    random.seed(42)
    rule = LinearRecurrenceRule([a, b], p)
    ds = RecurrenceDataset(p=p, recurrence_fn=rule.next_fn(),
                           recurrence_name=rule.name, init_len=2,
                           num_samples=cfg['num_train_samples'], length=5,
                           verbose=False)
    ds.run_full_split()
    test_indices = torch.tensor([int(seq[0]) + p * int(seq[1])
                                 for seq in ds.test_samples], dtype=torch.long)
    generator = torch.Generator().manual_seed(args.corruption_seed)
    all_wrong_offsets = 1 + torch.randint(0, p - 1, (5, p * p), generator=generator)
    candidates = candidate_values(a, b, p)
    print('  ' + '; '.join(f'{name}={candidates[name]}' for name in CANDIDATE_NAMES))
    print(f'  corruption_seed={args.corruption_seed}; 每个基底有32×126=4032次扰动/位置，'
          f'共100800次；错误位 clean_hit=32/4032。')

    report = {
        'ckpt': os.path.abspath(args.ckpt), 'model_config': cfg,
        'checkpoint_final_epoch': ckpt.get('final_epoch'),
        'checkpoint_best_accuracy': ckpt.get('best_accuracy'),
        'candidate_values': candidates, 'n_bases_per_corrupt_position': args.n_bases,
        'base_selection_seed': args.seed, 'corruption_seed': args.corruption_seed,
        'delta_values': list(range(1, p)),
        'fit_definition': 'exact fraction of (D-k*d)%P==0; nonnumeric outputs count as mismatch',
        'D_definition': '(perturbed_numeric_output-baseline_output)%P',
        'clean_hit_definition': 'only when perturbing corrupt position: wrong_value+d==true_value mod P',
        'by_corrupt_position': {},
    }
    manual_seq = rollout(1, 2, a, b, p)

    for corrupt_pos in range(5):
        print(f'\n{"=" * 72}\n[错误位置 x{corrupt_pos}]')
        offsets = all_wrong_offsets[corrupt_pos]
        inputs = make_corrupt_inputs(windows, corrupt_pos, offsets, eos, p)
        targets = windows[:, corrupt_pos]
        numeric0, full0 = predict(model, inputs, args.device, p, args.batch_size)
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
        selection_generator = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(len(correct_test_indices), generator=selection_generator)
        selected = correct_test_indices[permutation[:args.n_bases]]
        base_inputs = inputs[selected]
        base_outputs = numeric0[selected]
        base_offsets = offsets[selected]
        print(f'  基底状态编号: {selected.tolist()}')

        summaries, raw_records = {}, {}
        for j in range(5):
            summary, raw = fit_position(model, base_inputs, base_outputs, j, a, b, p,
                                        args.device, args.batch_size)
            add_clean_hit_metrics(summary, raw, base_offsets.tolist(), corrupt_pos, j,
                                  candidates, p)
            summaries[f'x{j}'] = summary
            raw_records[f'x{j}'] = raw
        print_fit_table(corrupt_pos, summaries)

        example = manual_examples(model, manual_seq, a, b, corrupt_pos, eos, p,
                                  args.device, args.batch_size)
        print(f'  手动样例: clean={manual_seq}, wrong_offset=37, '
              f'输入={example["base_input"]}, 输出={example["baseline_numeric_output"]}, '
              f'真值={example["true_value"]}')
        for record in example['records']:
            tag = ' CLEAN_HIT' if record['clean_hit'] else ''
            print(f'    x{record["position"]} d={record["d"]}: '
                  f'{record["old_value"]}->{record["new_value"]}, '
                  f'y={record["baseline_output"]}->{record["numeric_output"]}, '
                  f'D={record["D"]}{tag}')

        report['by_corrupt_position'][f'x{corrupt_pos}'] = {
            'accuracy': {
                'heldout_n': len(test_indices),
                'heldout_full_vocab': test_correct.float().mean().item(),
                'heldout_numeric_only': numeric_test_correct.float().mean().item(),
                'all_state_full_vocab': correct.float().mean().item(),
            },
            'base_state_indices': selected.tolist(),
            'base_clean_windows': windows[selected].tolist(),
            'base_wrong_offsets': base_offsets.tolist(),
            'base_corrupt_inputs': base_inputs.tolist(),
            'base_outputs': base_outputs.tolist(),
            'summaries': summaries, 'manual_example': example,
            'raw_records': raw_records,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n[完成] 汇总与全部100800条原始扰动记录已写入: {args.out}')


if __name__ == '__main__':
    main()
