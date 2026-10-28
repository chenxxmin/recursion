#!/usr/bin/env python3
"""pos024 vs last2：检测电路的位置通用性对比探针。

测量（对 len5 value_only 全位置暖启动模型 vs last2 暖启动模型）：
1) EOS 对污损位的 attention 权重（每层、逐 k）：模型是否"看见"错误位置；
2) 与干净窗对照的 attention 差分：检测是否由污损触发；
3) 逐位置纠错准确率（全位置，验证行为层差异）。
输出 reports/pos024_vs_last2_detection.json + 摘要打印。
"""
import json, os, re, sys
from collections import defaultdict

import torch

ROOT = '/home/cxm/recursion'
sys.path.insert(0, os.path.join(ROOT, 'src'))
from probe_fillblank_attention import AttentionCapture
from probe_fillblank_operands import load_model, test_windows, predict, P, BLANK, EOS

MODELS = {
    'initPos024': '/data/cxm/models/variant_o2_corruptfix_valueonly_d512l4_P127_len5_initPos024',
    'initLast2': '/data/cxm/models/variant_o2_corruptfix_valueonly_d512l4_P127_len5_initLast2',
}
N_WINDOWS = 64


@torch.no_grad()
def probe_one(path, a, b, device):
    model, _ = load_model(path, device)
    cap = AttentionCapture(model)
    wins = test_windows(a, b)[:N_WINDOWS]
    out = {'acc': {}, 'attn_corrupt': {}, 'attn_clean': {}}
    for k in range(5):
        # 污损窗：位置 k 换成随机错误值
        seqs, vs = [], []
        for w in wins:
            seq = w[:5].clone()
            orig = int(seq[k])
            vs.append(orig)
            seq[k] = (orig + 37) % P
            seqs.append(torch.cat([seq, torch.tensor([EOS])])
            )
        x = torch.stack(seqs).to(device)
        cap.records = []
        logits, _ = model(x)
        preds = logits[:, -1, :P].argmax(dim=-1).tolist()
        out['acc'][k] = sum(p == v for p, v in zip(preds, vs)) / len(vs)
        # EOS 对位置 k 的 attention（每层平均）
        out['attn_corrupt'][k] = [att[:, :, 5, k].mean().item()
                                  for att, _ in cap.records]
        # 干净窗对照（不污损任何位置）
        cap.records = []
        model(x)
        out['attn_clean'][k] = [att[:, :, 5, k].mean().item()
                                for att, _ in cap.records]
    cap.close()
    return out


def main():
    device = 'cuda:0'
    result = {}
    for group, mdir in MODELS.items():
        for fn in sorted(os.listdir(mdir)):
            if not fn.endswith('_best.pth'):
                continue
            a, b = map(int, re.search(r'a(\d+)b(\d+)', fn).groups())
            result[f'{group}/{a}b{b}'] = probe_one(
                os.path.join(mdir, fn), a, b, device)
    with open('reports/pos024_vs_last2_detection.json', 'w') as f:
        json.dump(result, f, indent=1)

    # 摘要：每组每位置 平均 EOS→污损位 attention（l0）与 acc
    for group in MODELS:
        print(f'\n===== {group} =====')
        print('coef      acc(k0..k4)                EOS→污损位 att l0 (k0..k4)')
        for name, d in sorted(result.items()):
            if not name.startswith(group):
                continue
            ab = name.split('/')[1]
            accs = ' '.join(f'{d["acc"][k]*100:5.1f}' for k in range(5))
            atts = ' '.join(f'{d["attn_corrupt"][k][0]:.2f}' for k in range(5))
            print(f'{ab:9s} {accs}   |  {atts}')
    print('\nsaved reports/pos024_vs_last2_detection.json')


if __name__ == '__main__':
    main()
