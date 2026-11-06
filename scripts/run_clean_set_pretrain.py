#!/usr/bin/env python3
"""clean-set-pretrain-v1：等待空卡、运行两阶段并校验第一阶段 gate。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/cxm/miniconda3/bin/python'
STAGE1 = 'single_o2_cleanvalueset_d512l4_P127_len5'
STAGE2 = 'variant_o2_corruptfix_valueonly_d512l4_P127_len5_initCleanValueSet'
MODEL_BASE = Path('/data/cxm/models')
LOG_BASE = Path('/data/cxm/recursion')


def gpu_snapshot():
    def run(query):
        return subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                                       text=True)
    used = {}
    for line in run('--query-gpu=index,uuid,memory.used,utilization.gpu').splitlines():
        index, uuid, memory, util = [x.strip() for x in line.split(',')]
        used[index] = {'uuid': uuid, 'memory': int(memory), 'util': int(util)}
    busy_uuids = set()
    for line in run('--query-compute-apps=gpu_uuid,pid').splitlines():
        if line.strip():
            busy_uuids.add(line.split(',')[0].strip())
    free = []
    for index, info in used.items():
        if info['memory'] <= 100 and info['util'] == 0 and info['uuid'] not in busy_uuids:
            free.append(int(index))
    return sorted(free), used, busy_uuids


def wait_for_gpus(required, poll_seconds):
    while True:
        free, _, _ = gpu_snapshot()
        if len(free) >= required:
            return free
        print(f'[WAIT] free={free}; need {required} GPUs; sleep {poll_seconds}s', flush=True)
        time.sleep(poll_seconds)


def load_batch(path):
    with open(path) as handle:
        return json.load(handle)


def run_batch(batch):
    path = ROOT / 'experiments' / f'{batch}.json'
    log_path = ROOT / 'logs' / f'run_{batch}.log'
    env = dict(os.environ)
    env.update({'BATCH_RUN_REQUIRE_GPU': '1', 'BATCH_RUN_IDLE_MEM_MB': '100',
                'PYTHONUNBUFFERED': '1'})
    print(f'[START] {batch}; log={log_path}', flush=True)
    with open(log_path, 'w') as handle:
        process = subprocess.Popen(
            [PYTHON, '-u', str(ROOT / 'src' / 'batch_run.py'), str(path),
             '--base-dir', str(LOG_BASE), '--model-base-dir', str(MODEL_BASE)],
            cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True)
    rc = process.wait()
    print(f'[BATCH_RC] {batch}: {rc}', flush=True)
    return rc


def gate(batch, expected_variant):
    experiments = load_batch(ROOT / 'experiments' / f'{batch}.json')['experiments']
    failures = []
    for experiment in experiments:
        name = experiment['name']
        final = MODEL_BASE / batch / f'{name}.pth'
        best = MODEL_BASE / batch / f'{name}_best.pth'
        try:
            if not final.is_file():
                raise ValueError('final checkpoint missing')
            checkpoint = __import__('torch').load(best, map_location='cpu', weights_only=True)
            accuracy = float(checkpoint['best_accuracy'])
            config = checkpoint['config']
            if accuracy < 0.99:
                raise ValueError(f'best_accuracy={accuracy:.6f}<0.99')
            if config['variant'] != expected_variant:
                raise ValueError(f'variant={config["variant"]}')
            if config['train_len'] != 5 or config['rules'] != experiment['config']['RULES']:
                raise ValueError('train_len/rules mismatch')
            print(f'[GATE] {name}: best={accuracy:.4%}', flush=True)
        except Exception as exc:
            failures.append(f'{name}: {type(exc).__name__}: {exc}')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stage1', default=STAGE1)
    parser.add_argument('--stage2', default=STAGE2)
    parser.add_argument('--required-free', type=int, default=6)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--only-stage', choices=['1', '2', 'all'], default='all')
    args = parser.parse_args()
    if not 1 <= args.required_free <= 8 or args.poll_seconds <= 0:
        parser.error('invalid resource settings')

    free, _, _ = gpu_snapshot()
    print(f'[GPU] currently free={free}; required={args.required_free}', flush=True)
    if args.dry_run:
        for batch in (args.stage1, args.stage2):
            raw = load_batch(ROOT / 'experiments' / f'{batch}.json')
            experiments = raw['experiments']
            print(f'  {batch}: {len(experiments)} experiments, '
                  f'concurrency={raw["concurrency"]}')
        print('[DRY-RUN] no processes launched')
        return

    if args.only_stage in ('1', 'all'):
        free = wait_for_gpus(args.required_free, args.poll_seconds)
        print(f'[GPU] stage1 starting with free={free}', flush=True)
        rc = run_batch(args.stage1)
        failures = gate(args.stage1, 'clean_value_set')
        if rc != 0 or failures:
            for failure in failures:
                print(f'[GATE_FAILED] {failure}', flush=True)
            raise SystemExit(f'stage1 failed rc={rc}, gate_failures={len(failures)}')
        print('[GATE_PASS] stage1 all six passed', flush=True)

    if args.only_stage in ('2', 'all'):
        free = wait_for_gpus(args.required_free, args.poll_seconds)
        print(f'[GPU] stage2 starting with free={free}', flush=True)
        rc = run_batch(args.stage2)
        failures = gate(args.stage2, 'corrupt_fix')
        if rc != 0 or failures:
            for failure in failures:
                print(f'[GATE_FAILED] {failure}', flush=True)
            raise SystemExit(f'stage2 failed rc={rc}, gate_failures={len(failures)}')
        print('[GATE_PASS] stage2 all six passed', flush=True)


if __name__ == '__main__':
    main()
