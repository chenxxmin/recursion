#!/usr/bin/env python3
"""Find idle GPUs, pin an exact GPU set, and run one batch_run config.

Default mode only prints the current plan.  Require --execute for training:

python gpu_batch_launch.py experiments/batch.json --required 4 \
  --gpus 2,3,4,5 --base-dir /data/cxm/recursion \
  --model-base-dir /data/cxm/models --execute
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime


REPO_ROOT = Path(__file__).resolve().parents[4]


def parse_gpu_list(text):
    """Parse an ordered, duplicate-free physical GPU candidate pool."""
    try:
        cards = [int(item.strip()) for item in text.split(',') if item.strip()]
    except ValueError as exc:
        raise ValueError(f'invalid GPU list: {text!r}') from exc
    if not cards or any(card < 0 for card in cards) or len(set(cards)) != len(cards):
        raise ValueError(f'invalid GPU list: {text!r}')
    return cards


def _nvidia_smi(query):
    return subprocess.check_output(
        ['nvidia-smi', query, '--format=csv,noheader,nounits'],
        text=True, stderr=subprocess.STDOUT)


def gpu_snapshot():
    """Return (card states, busy UUIDs); never fall back to CPU detection."""
    cards = {}
    for line in _nvidia_smi('--query-gpu=index,uuid,memory.used,utilization.gpu').splitlines():
        index, uuid, memory, util = [item.strip() for item in line.split(',')]
        cards[int(index)] = {
            'uuid': uuid,
            'memory_mb': int(memory),
            'utilization': int(util),
        }
    busy_uuids = set()
    for line in _nvidia_smi('--query-compute-apps=gpu_uuid,pid').splitlines():
        if line.strip():
            busy_uuids.add(line.split(',')[0].strip())
    return cards, busy_uuids


def free_gpus(candidates, cards, busy_uuids, memory_limit_mb=100):
    """Preserve candidate order; require idle memory, 0% util and no compute app."""
    return [
        index for index in candidates
        if index in cards
        and cards[index]['memory_mb'] <= memory_limit_mb
        and cards[index]['utilization'] == 0
        and cards[index]['uuid'] not in busy_uuids
    ]


def wait_for_gpus(candidates, required, poll_seconds):
    """Wait until the candidate pool has enough idle cards, then pin exactly N."""
    while True:
        cards, busy_uuids = gpu_snapshot()
        missing = [index for index in candidates if index not in cards]
        if missing:
            raise RuntimeError(f'candidate GPUs not reported by nvidia-smi: {missing}')
        free = free_gpus(candidates, cards, busy_uuids)
        if len(free) >= required:
            selected = free[:required]
            print(f'[GPU] free={free}; selected={selected}', flush=True)
            return selected
        print(f'[WAIT] free={free}; need {required} GPUs from {candidates}; '
              f'sleep {poll_seconds}s', flush=True)
        time.sleep(poll_seconds)


def batch_shape(config_path):
    with open(config_path, encoding='utf-8') as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        return len(raw.get('experiments', [])), int(raw.get('concurrency', 1))
    return len(raw), 1


def default_batch_log(config_path, required):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return REPO_ROOT / 'logs' / f'run_{config_path.stem}_{required}gpu_{stamp}.log'


def run_batch(config_path, selected, base_dir, model_base_dir, log_path):
    """Launch one detached batch process with an exact physical GPU whitelist."""
    env = dict(os.environ)
    env.update({
        'BATCH_RUN_GPUS': ','.join(str(index) for index in selected),
        'BATCH_RUN_REQUIRE_GPU': '1',
        'BATCH_RUN_IDLE_MEM_MB': '100',
        'PYTHONUNBUFFERED': '1',
    })
    command = [
        sys.executable, '-u', str(REPO_ROOT / 'src' / 'batch_run.py'),
        str(config_path), '--base-dir', str(base_dir),
        '--model-base-dir', str(model_base_dir),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'[START] config={config_path}; log={log_path}', flush=True)
    with open(log_path, 'w', encoding='utf-8') as handle:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True)
        print(f'[BATCH_PID] {process.pid}; BATCH_RUN_GPUS={env["BATCH_RUN_GPUS"]}',
              flush=True)
        rc = process.wait()
    print(f'[BATCH_RC] {rc}', flush=True)
    return rc


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='experiments/*.json batch config')
    parser.add_argument('--required', type=int, required=True,
                        help='exact number of GPUs to pin for this batch')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7',
                        help='ordered physical GPU candidate pool')
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--base-dir', type=Path, required=True,
                        help='explicit result root; avoids accidental /mnt/workspace writes')
    parser.add_argument('--model-base-dir', type=Path, required=True,
                        help='explicit checkpoint root')
    parser.add_argument('--batch-log', type=Path,
                        help='default: logs/run_<config>_Ngpu_<timestamp>.log')
    parser.add_argument('--execute', action='store_true',
                        help='actually wait and train; default only prints the plan')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.required < 1 or args.poll_seconds <= 0:
        raise SystemExit('invalid --required/--poll-seconds')
    candidates = parse_gpu_list(args.gpus)
    if args.required > len(candidates):
        raise SystemExit('--required exceeds the candidate GPU pool size')
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f'config not found: {config_path}')
    if not args.base_dir.is_absolute() or not args.model_base_dir.is_absolute():
        raise SystemExit('--base-dir and --model-base-dir must be absolute paths')
    experiment_count, configured_concurrency = batch_shape(config_path)
    if not experiment_count:
        raise SystemExit('batch config contains no experiments')

    cards, busy_uuids = gpu_snapshot()
    free = free_gpus(candidates, cards, busy_uuids)
    print(f'[PLAN] experiments={experiment_count}; '
          f'json_concurrency={configured_concurrency}; required_gpus={args.required}',
          flush=True)
    print(f'[PLAN] candidates={candidates}; currently_free={free}', flush=True)
    print('[PLAN] BATCH_RUN_GPUS will pin exactly the selected cards; '
          'JSON concurrency larger than N is safe.', flush=True)
    if not args.execute:
        print('[DRY-RUN] no processes launched', flush=True)
        return 0

    selected = wait_for_gpus(candidates, args.required, args.poll_seconds)
    log_path = args.batch_log or default_batch_log(config_path, args.required)
    if not log_path.is_absolute():
        log_path = REPO_ROOT / log_path
    return run_batch(config_path, selected, args.base_dir,
                     args.model_base_dir, log_path)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(2)
