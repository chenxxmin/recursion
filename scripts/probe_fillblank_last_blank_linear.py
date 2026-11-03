#!/usr/bin/env python3
"""len5 fill_blank 末位空格的模块化因果扰动分析。

对四个非空格位置分别施加 x[j] <- x[j] + d (mod P)，测量 EOS 数值答案的
D = y_perturbed - y_base (mod P)，并检验 D = k*d (mod P)。拟合程度定义为
精确同余命中率；非数值（BLANK/EOS）输出记为所有候选均不匹配。
"""
import argparse
import json
import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from datasets import RecurrenceDataset  # noqa: E402
from models import FibonacciTransformer  # noqa: E402
from rules import LinearRecurrenceRule  # noqa: E402

CANDIDATE_NAMES = ['b', 'a', "-a'b", "a'", "-b'a", "b'", '0']


def load_model(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = FibonacciTransformer(
        p=cfg['p'], d_model=cfg['d_model'], n_head=cfg['n_head'],
        n_layer=cfg['n_layer'], block_size=cfg['block_size'],
        mlp_ratio=cfg['mlp_ratio'], vocab_size=cfg['vocab_size'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, cfg, ckpt


def rollout(x0, x1, a, b, p, length=5):
    seq = [x0, x1]
    while len(seq) < length:
        seq.append((a * seq[-1] + b * seq[-2]) % p)
    return seq


def make_last_blank_inputs(windows, blank, eos):
    """[x0,x1,x2,x3,x4] -> [x0,x1,x2,x3,BLANK,EOS]。"""
    inputs = windows.clone()
    inputs[:, -1] = blank
    return torch.cat([inputs, torch.full((len(inputs), 1), eos, dtype=torch.long)], dim=1)


@torch.inference_mode()
def predict(model, inputs, device, p, batch_size):
    """返回数值类 argmax 和完整词表 argmax；答案始终读 EOS 输入槽。"""
    numeric, full = [], []
    for start in range(0, len(inputs), batch_size):
        x = inputs[start:start + batch_size].to(device)
        logits, _ = model(x)
        last = logits[:, -1]
        numeric.append(last[:, :p].argmax(dim=-1).cpu())
        full.append(last.argmax(dim=-1).cpu())
    return torch.cat(numeric), torch.cat(full)


def candidate_values(a, b, p):
    ai, bi = pow(a, -1, p), pow(b, -1, p)
    values = [b, a, (-ai * b) % p, ai, (-bi * a) % p, bi, 0]
    return dict(zip(CANDIDATE_NAMES, values))


def circular_mae(residual, p):
    r = residual % p
    return torch.minimum(r, p - r).float().mean().item()


def fit_position(model, base_inputs, base_outputs, position, a, b, p, device,
                 batch_size):
    """基底顺序在外、d=1..P-1 在内，返回汇总及可审计的列式原始记录。"""
    n_bases = len(base_inputs)
    deltas = torch.arange(1, p, dtype=torch.long)
    rows = []
    for base in base_inputs:
        repeated = base.unsqueeze(0).repeat(p - 1, 1)
        repeated[:, position] = (repeated[:, position] + deltas) % p
        rows.append(repeated)
    inputs = torch.cat(rows)
    y1, full1 = predict(model, inputs, device, p, batch_size)
    y0 = base_outputs.repeat_interleave(p - 1)
    d = deltas.repeat(n_bases)
    D = (y1 - y0) % p
    valid_numeric = full1 < p

    fits = {}
    for name, k in candidate_values(a, b, p).items():
        residual = (D - k * d) % p
        exact = ((residual == 0) & valid_numeric).float().mean().item()
        fits[name] = {
            'k': k,
            'exact_match': exact,
            'exact_count': int(((residual == 0) & valid_numeric).sum()),
            'circular_mae': circular_mae(residual, p),
        }

    base_indices = torch.arange(n_bases).repeat_interleave(p - 1)
    raw = {
        'base_row': base_indices.tolist(),
        'd': d.tolist(),
        'baseline_output': y0.tolist(),
        'perturbed_numeric_output': y1.tolist(),
        'perturbed_full_vocab_output': full1.tolist(),
        'D': D.tolist(),
    }
    summary = {
        'position': position,
        'n': len(inputs),
        'output_change_rate': (D != 0).float().mean().item(),
        'nonnumeric_output_rate': (~valid_numeric).float().mean().item(),
        'fits': fits,
    }
    return summary, raw


def print_example(model, a, b, p, blank, eos, device, batch_size):
    seq = rollout(1, 2, a, b, p)
    base = make_last_blank_inputs(torch.tensor([seq]), blank, eos)
    y0, full0 = predict(model, base, device, p, batch_size)
    print('\n[步骤 5] 手动样例：初值 x0=1, x1=2')
    print(f'  干净序列: {seq}; 真值 x4={seq[-1]}')
    print(f'  模型输入: {base[0].tolist()}; 数值输出={int(y0[0])}, '
          f'完整词表输出={int(full0[0])}')
    print('  以下 d=-7 以等价剩余类 120 施加；ratio=D*d^(-1) (mod P)。')
    example_records = []
    for j in range(4):
        for signed_d in (1, 7, -7):
            d = signed_d % p
            perturbed = base.clone()
            old = int(perturbed[0, j])
            perturbed[0, j] = (old + d) % p
            y1, full1 = predict(model, perturbed, device, p, batch_size)
            D = (int(y1[0]) - int(y0[0])) % p
            ratio = D * pow(d, -1, p) % p
            matched = [name for name, k in candidate_values(a, b, p).items()
                       if k == ratio and int(full1[0]) < p]
            print(f'  x{j}: d={signed_d:+d}, {old}->{int(perturbed[0, j])}, '
                  f'y {int(y0[0])}->{int(y1[0])}, D={D}, ratio={ratio}, '
                  f'match={matched}')
            example_records.append({
                'position': j, 'signed_d': signed_d, 'd': d,
                'input': perturbed[0].tolist(), 'baseline_output': int(y0[0]),
                'numeric_output': int(y1[0]), 'full_vocab_output': int(full1[0]),
                'D': D, 'ratio': ratio, 'matched_candidates': matched,
            })
    return {'clean_sequence': seq, 'base_input': base[0].tolist(),
            'baseline_numeric_output': int(y0[0]),
            'baseline_full_vocab_output': int(full0[0]),
            'records': example_records}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--n-bases', type=int, default=32,
                    help='从 held-out 且原预测正确的 len5 窗口中抽取的基底数量')
    ap.add_argument('--seed', type=int, default=20261014, help='基底窗口抽样种子')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--out', default='reports/data/fillblank_len5_last_blank_linear_a25b56.json')
    args = ap.parse_args()

    print('[步骤 1] 加载 checkpoint 并核对任务配置')
    print(f'  ckpt: {args.ckpt}')
    model, cfg, ckpt = load_model(args.ckpt, args.device)
    p, a, b = cfg['p'], cfg['rules'][0][0], cfg['rules'][0][1]
    blank, eos = cfg['blank_token_id'], cfg['eos_token_id']
    print(f'  checkpoint final_epoch={ckpt.get("final_epoch")}, '
          f'best_accuracy={ckpt.get("best_accuracy")}')
    print(f'  variant={cfg["variant"]}, seq_len={cfg.get("seq_len_min")}..'
          f'{cfg.get("seq_len_max")}, body_supervision={cfg["body_supervision"]}')
    print(f'  rules=[[{a},{b}]], P={p}, BLANK={blank}, EOS={eos}, '
          f'device={args.device}, dtype={next(model.parameters()).dtype}')
    assert cfg['variant'] == 'fill_blank' and cfg['order'] == 2
    assert cfg['train_len'] == 5 and not cfg['body_supervision']
    assert p == 127 and cfg['vocab_size'] == p + 2

    print('\n[步骤 2] 枚举 len5 全状态空间，并重建 seed42 的 held-out 划分')
    print('  状态编号 index=x0+P*x1；每个初值 (x0,x1) 生成一个干净 len5 窗口。')
    windows = torch.tensor([rollout(i % p, i // p, a, b, p)
                            for i in range(p * p)], dtype=torch.long)
    inputs = make_last_blank_inputs(windows, blank, eos)
    targets = windows[:, -1]
    numeric0, full0 = predict(model, inputs, args.device, p, args.batch_size)

    # 与训练相同：在构造 RecurrenceDataset 前设置 random.seed(RANDOM_SEED)。
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
    test_correct = full0[test_indices] == targets[test_indices]
    test_numeric_correct = numeric0[test_indices] == targets[test_indices]
    all_correct = full0 == targets
    print(f'  held-out: {int(test_correct.sum())}/{len(test_indices)} = '
          f'{test_correct.float().mean().item() * 100:.4f}% (完整词表 argmax)')
    print(f'  held-out: {int(test_numeric_correct.sum())}/{len(test_indices)} = '
          f'{test_numeric_correct.float().mean().item() * 100:.4f}% (仅数值类 argmax)')
    print(f'  全状态: {int(all_correct.sum())}/{p * p} = '
          f'{all_correct.float().mean().item() * 100:.4f}% (完整词表 argmax)')
    assert test_correct.float().mean().item() > 0.99

    print('\n[步骤 3] 候选斜率（全部数值均在 mod P 剩余类中）')
    candidates = candidate_values(a, b, p)
    print(f'  逆元校验: a\'={candidates["a\'"]}, a*a\'={(a * candidates["a\'"]) % p}; '
          f'b\'={candidates["b\'"]}, b*b\'={(b * candidates["b\'"]) % p}')
    print('  ' + '; '.join(f'{name}={candidates[name]}' for name in CANDIDATE_NAMES))
    print('  末位空格 x4 的局部公式为 x4=a*x3+b*x2，因此局部规则预期: '
          'x0->0, x1->0, x2->b, x3->a。')

    print('\n[步骤 4] 扰动设计')
    correct_test_indices = test_indices[test_correct]
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(correct_test_indices), generator=generator)
    selected = correct_test_indices[permutation[:args.n_bases]]
    assert len(selected) == args.n_bases
    base_inputs = inputs[selected]
    base_outputs = numeric0[selected]
    print(f'  从 {len(correct_test_indices)} 个 held-out 且原预测正确的窗口中，'
          f'用 seed={args.seed} 抽取 {len(selected)} 个。')
    print(f'  每位置使用全部 d=1..{p - 1}，每位置 N={len(selected) * (p - 1)}；'
          f'四位置总扰动数={4 * len(selected) * (p - 1)}。')
    print('  D=(y_perturbed-y_base) mod P；精确拟合为 D==(k*d) mod P。'
          ' 非数值输出计入分母且不算匹配。')
    print(f'  基底状态编号: {selected.tolist()}')

    summaries, raw_records = {}, {}
    for j in range(4):
        summary, raw = fit_position(model, base_inputs, base_outputs, j, a, b, p,
                                    args.device, args.batch_size)
        summaries[f'x{j}'] = summary
        raw_records[f'x{j}'] = raw
        best = max(CANDIDATE_NAMES, key=lambda name: summary['fits'][name]['exact_match'])
        print(f'  x{j}: 输出改变率={summary["output_change_rate"] * 100:.2f}%, '
              f'非数值输出率={summary["nonnumeric_output_rate"] * 100:.2f}%, '
              f'best={best} ({summary["fits"][best]["exact_match"] * 100:.2f}%), '
              f'circular_MAE={summary["fits"][best]["circular_mae"]:.4f}')

    example = print_example(model, a, b, p, blank, eos, args.device, args.batch_size)

    print('\n[步骤 6] 拟合程度表：单元格为 D=k*d (mod P) 精确命中率')
    header = ['扰动位置'] + CANDIDATE_NAMES
    print(' | '.join(header))
    print(' | '.join(['---'] + ['---:'] * len(CANDIDATE_NAMES)))
    for j in range(4):
        row = [f'x{j}'] + [f'{summaries[f"x{j}"]["fits"][name]["exact_match"] * 100:.2f}%'
                           for name in CANDIDATE_NAMES]
        print(' | '.join(row))

    report = {
        'ckpt': os.path.abspath(args.ckpt),
        'model_config': cfg,
        'checkpoint_final_epoch': ckpt.get('final_epoch'),
        'checkpoint_best_accuracy': ckpt.get('best_accuracy'),
        'task': 'len5 fill_blank, blank at x4, read EOS slot',
        'state_index_convention': 'index=x0+P*x1',
        'accuracy': {
            'heldout_n': len(test_indices),
            'heldout_full_vocab': test_correct.float().mean().item(),
            'heldout_numeric_only': test_numeric_correct.float().mean().item(),
            'all_state_full_vocab': all_correct.float().mean().item(),
        },
        'candidate_values': candidates,
        'n_bases': len(selected),
        'base_state_indices': selected.tolist(),
        'base_clean_windows': windows[selected].tolist(),
        'base_outputs': base_outputs.tolist(),
        'delta_values': list(range(1, p)),
        'fit_definition': 'exact fraction of (D-k*d)%P==0; nonnumeric full-vocab outputs count as mismatch',
        'D_definition': '(perturbed_numeric_output-baseline_output)%P',
        'summaries': summaries,
        'manual_example': example,
        'raw_records': raw_records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n[完成] 汇总、手动样例及全部原始扰动记录已写入: {args.out}')


if __name__ == '__main__':
    main()
