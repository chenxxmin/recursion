"""Probe: accuracy on the 1st vs 2nd position of length-2 missing runs.

Loads a trained action+missing model, rebuilds its exact test set, and splits
prediction accuracy on corrupted value positions by their index within a run
(1st hole vs 2nd hole). Only runs of exactly length 2 are counted (the config
guarantees every sample has one; len-1 runs are reported separately).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python scripts/probe_two_miss_positions.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch

import experiment

CHECKPOINT = ('/mnt/workspace/hujiachen/models/action_p127_tr64_ood128_misslen/'
              'action_d1024l2r4h4_P127_tr64ood128_N2_randmiss0.1len2_seed17996.pth')
EXP_NAME = 'action_d1024l2r4h4_P127_tr64ood128_N2_randmiss0.1len2_seed17996'


def build():
    base = json.load(open('src/config.json'))
    exp = next(e for e in json.load(open('experiments/action_p127_tr64_ood128_misslen.json'))['experiments']
               if e['name'] == EXP_NAME)
    mm = dict(base['main'])
    mm.update(base.get('action', {}))
    mm.update(exp['config'])
    cfg = dict(base)
    cfg['main'] = mm
    cfg['_BATCH_RUN_MERGED'] = True
    ctx = experiment._prepare_action(cfg)
    model = ctx['model'].to('cuda').eval()
    ckpt = torch.load(CHECKPOINT, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"loaded {EXP_NAME}: best_accuracy={ckpt.get('best_accuracy'):.4f}")
    return ctx, model, mm['P']


def main():
    ctx, model, P = build()
    test_ds = ctx['test_dataset']
    num_mask = ctx['num_mask']

    # value x_k sits at seq index 2k-3; its prediction is logits at target
    # index 2k-4 (targets = seq[1:])
    stats = {'first': [0, 0], 'second': [0, 0], 'single': [0, 0]}  # [correct, total]
    cond = {'2nd|1st_ok': [0, 0], '2nd|1st_bad': [0, 0]}
    n_runs2 = n_runs1 = 0

    B = 512
    with torch.no_grad():
        for i in range(0, len(test_ds), B):
            batch = [test_ds[j] for j in range(i, min(i + B, len(test_ds)))]
            views = torch.stack([b[0] for b in batch]).cuda()
            cleans = torch.stack([b[1] for b in batch])
            logits = model(views)[0]
            preds = logits.argmax(dim=-1).cpu()  # (b, seq_len-1): preds[t] ~ seq[t+1]

            for b in range(len(batch)):
                view, clean = batch[b][0], batch[b][1]
                L = test_ds.length
                # missing flags per value index k=3..L
                miss = [bool(view[2 * k - 3] == P) for k in range(3, L + 1)]
                # find runs
                k = 0
                while k < len(miss):
                    if not miss[k]:
                        k += 1
                        continue
                    j = k
                    while j < len(miss) and miss[j]:
                        j += 1
                    run_len = j - k
                    val_indices = list(range(k + 3, j + 3))  # value indices k' = 3..L
                    if run_len == 2:
                        n_runs2 += 1
                        for slot, vk in zip(('first', 'second'), val_indices):
                            t = 2 * vk - 4
                            ok = bool(preds[b, t] == clean[2 * vk - 3])
                            stats[slot][0] += ok
                            stats[slot][1] += 1
                        ok1 = bool(preds[b, 2 * val_indices[0] - 4] == clean[2 * val_indices[0] - 3])
                        ok2 = bool(preds[b, 2 * val_indices[1] - 4] == clean[2 * val_indices[1] - 3])
                        key = '2nd|1st_ok' if ok1 else '2nd|1st_bad'
                        cond[key][0] += ok2
                        cond[key][1] += 1
                    elif run_len == 1:
                        n_runs1 += 1
                        vk = val_indices[0]
                        t = 2 * vk - 4
                        stats['single'][0] += bool(preds[b, t] == clean[2 * vk - 3])
                        stats['single'][1] += 1
                    k = j

    def pct(c, n):
        return f"{c / n * 100:.2f}%" if n else "N/A"

    print(f"\ntest samples: {len(test_ds)} | len-2 runs: {n_runs2} | len-1 runs: {n_runs1}")
    print(f"单空 (len-1 run)      : {pct(*stats['single'])}  ({stats['single'][0]}/{stats['single'][1]})")
    print(f"连续两空-第1空        : {pct(*stats['first'])}  ({stats['first'][0]}/{stats['first'][1]})")
    print(f"连续两空-第2空        : {pct(*stats['second'])}  ({stats['second'][0]}/{stats['second'][1]})")
    print(f"条件: P(第2空对 | 第1空对) = {pct(*cond['2nd|1st_ok'])}  ({cond['2nd|1st_ok'][0]}/{cond['2nd|1st_ok'][1]})")
    print(f"条件: P(第2空对 | 第1空错) = {pct(*cond['2nd|1st_bad'])}  ({cond['2nd|1st_bad'][0]}/{cond['2nd|1st_bad'][1]})")


if __name__ == '__main__':
    main()
