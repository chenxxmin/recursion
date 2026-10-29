#!/usr/bin/env python3
"""Extract len16--64 curriculum results from logs into one standalone JSON file."""
import json
import os
import re

COEFS = ['a25b56', 'a62b59', 'a56b93', 'a5b3', 'a67b65', 'a44b59']
FILLBLANK_COEFS = COEFS[:3]
LENGTHS = range(16, 65)
OUT = 'reports/data/length_curriculum_epochs.json'
EPOCH_RE = re.compile(r'Epoch\s+(\d+) \| Train:.*Test: Acc=.*?Best=([\d.]+)%')


def cf_log(a, b, length, seed=42, special=False):
    if special:
        return (f'/data/cxm/recursion/variant_o2_corruptfix_a{a}b{b}_len{length}_initLen{length-1}_seed{seed}/logs/'
                f'single_o2_corruptfix_a{a}b{b}_d512l4r8h4_P127_len{length}_initLen{length-1}_seed{seed}.log')
    return (f'/data/cxm/recursion/variant_o2_corruptfix_valueonly_d512l4_P127_len{length}_initLen{length-1}/logs/'
            f'single_o2_corruptfix_a{a}b{b}_d512l4r8h4_P127_len{length}_initLen{length-1}_seed{seed}.log')


def fb_log(a, b, length):
    return (f'/data/cxm/recursion/variant_o2_fillblank_noaux_randpos_d512l4_P127_len{length}_wd2_initLen{length-1}/logs/'
            f'single_o2_fillblank_a{a}b{b}_d512l4r8h4_P127_len{length}_randpos_noaux_lr0.0003_wd2_initLen{length-1}_seed42.log')


def candidates(task, coef, length):
    a, b = map(int, re.fullmatch(r'a(\d+)b(\d+)', coef).groups())
    if task == 'fill_blank':
        return [(42, fb_log(a, b, length))]
    logs = [(42, cf_log(a, b, length, 42))]
    if coef == 'a25b56' and length >= 48:
        logs.extend([(43, cf_log(a, b, length, 43, special=True)),
                     (42, cf_log(a, b, length, 42, special=True))])
    return logs


def parse_log(path, seed):
    result = {'seed': seed, 'source_log': path}
    if not os.path.isfile(path):
        result['status'] = 'missing'
        return result
    text = open(path, encoding='utf-8', errors='replace').read()
    matches = EPOCH_RE.findall(text)
    if not matches:
        result['status'] = 'launch_failed' if 'Traceback' in text else 'no_eval'
        return result
    last_epoch, best = matches[-1]
    result['best_test_acc'] = float(best)
    if float(best) >= 99.0:
        result['status'] = 'success'
        # Epoch 0 is evaluated after the first training epoch.
        result['epochs'] = int(last_epoch) + 1
    else:
        result['status'] = 'failed'
        result['epochs'] = None
    return result


def select_record(task, coef, length):
    attempts = [parse_log(path, seed) for seed, path in candidates(task, coef, length)]
    existing = [a for a in attempts if a['status'] != 'missing']
    successful = [a for a in existing if a['status'] == 'success']
    if successful:
        # The a25b56 retry chain uses seed43 from len48 onward.
        if task == 'corrupt_fix' and coef == 'a25b56' and length >= 48:
            chosen = next((a for a in successful if a['seed'] == 43), successful[0])
        else:
            chosen = successful[0]
    elif existing:
        if task == 'corrupt_fix' and coef == 'a25b56' and length >= 48:
            chosen = next((a for a in existing if a['seed'] == 43), existing[0])
        else:
            chosen = existing[0]
    else:
        chosen = attempts[0]
    record = {'task': task, 'coef': coef, 'length': length, **chosen}
    if len(existing) > 1:
        record['other_attempts'] = [a for a in existing if a is not chosen]
    return record


def main():
    records = []
    for task, coefs in [('corrupt_fix', COEFS), ('fill_blank', FILLBLANK_COEFS)]:
        for coef in coefs:
            for length in LENGTHS:
                records.append(select_record(task, coef, length))
    payload = {
        'description': 'Epochs used at each len16--64 curriculum stage; one record per task/coefficient/length.',
        'epoch_convention': 'last logged zero-based Epoch N plus 1; Epoch 0 therefore counts as 1 epoch',
        'success_criterion': 'last logged Best test accuracy >= 99%',
        'statuses': {
            'success': 'converged; epochs is an integer',
            'failed': 'ran and evaluated but did not reach 99%; epochs is null',
            'launch_failed': 'log exists but failed before evaluation; epochs is null',
            'no_eval': 'log exists without an evaluation record; epochs is null',
            'missing': 'log does not exist; epochs is null',
        },
        'records': records,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'wrote {len(records)} records to {OUT}')
    for task in ('corrupt_fix', 'fill_blank'):
        for coef in COEFS:
            subset = [r for r in records if r['task'] == task and r['coef'] == coef]
            if not subset:
                continue
            counts = {status: sum(r['status'] == status for r in subset)
                      for status in ('success', 'failed', 'launch_failed', 'no_eval', 'missing')}
            print(task, coef, counts)


if __name__ == '__main__':
    main()
