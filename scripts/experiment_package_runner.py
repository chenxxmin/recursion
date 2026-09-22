#!/usr/bin/env python3
"""Run an approved experiment bundle on explicitly selected GPUs."""
import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def process_info(pid):
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (root / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), start_ticks=fields[19], argv=argv)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def children(pid):
    table = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            info = process_info(int(entry.name))
            if info:
                table[info['pid']] = info
    selected = {pid}
    while True:
        additions = {p for p, info in table.items() if info['ppid'] in selected} - selected
        if not additions:
            break
        selected |= additions
    return [info for p, info in table.items() if p in selected and p != pid]


def stop_training(process, signalled, training_done=False):
    for info in children(process.pid):
        core = any(Path(arg).name == 'core.py' for arg in info['argv'])
        if (core or training_done) and info['pid'] not in signalled:
            try:
                os.kill(info['pid'], signal.SIGTERM)
                signalled.add(info['pid'])
            except ProcessLookupError:
                pass


def check_bundle(bundle, timers):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for relative, checksum in manifest['files'].items():
        path = (bundle / relative).resolve()
        require(path.is_relative_to(bundle) and path.is_file(), 'Missing/unsafe package file: ' + relative)
        require(sha_file(path) == checksum, 'Package file changed: ' + relative)
    data = json.loads((bundle / 'experiments.json').read_text())
    actual = {e['name']: e['config']['MAX_TRAIN_HOURS'] for e in data['experiments']}
    require(actual == manifest['timers'] == timers, 'run.sh timer settings disagree with approved JSON')
    print('Verified approved review:', manifest['confirmed_review_id'], flush=True)
    for exp in data['experiments']:
        cfg = exp['config']
        print(exp['name'], 'task=' + exp['task'], 'hours=' + str(cfg['MAX_TRAIN_HOURS']),
              'cosine_T_max=' + str(cfg['EPOCHS']),
              'start=' + ('resume' if cfg.get('RESUME_FROM') else 'weights_only' if cfg.get('INIT_FROM') else 'fresh'),
              flush=True)
    return manifest, data


def parse_gpus(value):
    parts = value.split(',')
    require(all(re.fullmatch(r'\d+', p) for p in parts), 'Use physical GPU indices, e.g. 0,1')
    gpus = [int(p) for p in parts]
    require(gpus and len(set(gpus)) == len(gpus), 'Duplicate/empty GPU list')
    return gpus


def reserve_gpus(gpus, manifest):
    if socket.gethostname() == manifest['packaging_host']:
        require(set(gpus) <= set(manifest['local_gpu_allowlist']), 'This source machine allows only GPUs 0-3')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                   '--format=csv,noheader,nounits'], text=True)
    table = {}
    for line in raw.splitlines():
        index, uuid, memory = [part.strip() for part in line.split(',')]
        table[int(index)] = (uuid, float(memory))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        tokens = {p.strip() for p in visible.split(',')}
        require(all(str(g) in tokens or table.get(g, ('',))[0] in tokens for g in gpus),
                'Selected GPUs are outside the inherited CUDA_VISIBLE_DEVICES allocation')
    locks = []
    try:
        for gpu in gpus:
            require(gpu in table, 'GPU does not exist: ' + str(gpu))
            uuid, used = table[gpu]
            require(used < 100, f'GPU {gpu} is occupied ({used} MiB)')
            lock = open('/tmp/recursion_experiment_' + str(os.getuid()) + '_' + uuid + '.lock', 'a')
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            subprocess.run([sys.executable, '-c',
                            'import torch, matplotlib; assert torch.cuda.is_available() and torch.cuda.device_count() == 1'],
                           env=env, check=True)
        return locks
    except Exception:
        for lock in locks:
            lock.close()
        raise


def gpu_snapshot():
    """Read physical GPU availability without allocating CUDA resources."""
    raw = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True, timeout=20)
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'],
        text=True, timeout=20)
    occupied = {line.strip() for line in apps.splitlines() if line.strip()}
    table = {}
    for line in raw.splitlines():
        index, uuid, memory, utilization = [part.strip() for part in line.split(',')]
        table[int(index)] = dict(uuid=uuid, memory_mib=float(memory),
                                utilization=float(utilization), has_compute_process=uuid in occupied)
    return table


def validate_gpu_pool(gpus, manifest, table):
    if socket.gethostname() == manifest['packaging_host']:
        require(set(gpus) <= set(manifest['local_gpu_allowlist']),
                'This source machine allows only GPUs 0-3')
    require(set(gpus) <= set(table), 'Requested GPU is missing from nvidia-smi')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        tokens = {part.strip() for part in visible.split(',')}
        require(all(str(gpu) in tokens or table[gpu]['uuid'] in tokens for gpu in gpus),
                'Selected GPUs are outside the inherited CUDA_VISIBLE_DEVICES allocation')


def launch_one(bundle, output, exp, gpu):
    name = exp['name']
    work = output / 'work' / name
    work.mkdir(parents=True)
    (work / 'src').symlink_to(bundle / 'code/src', target_is_directory=True)
    copied = json.loads(json.dumps(exp))
    cfg = copied['config']
    for key in ('INIT_FROM', 'RESUME_FROM'):
        if cfg.get(key):
            checkpoint = (bundle / cfg[key]).resolve()
            require(checkpoint.is_relative_to(bundle / 'checkpoints') and checkpoint.is_file(),
                    'Checkpoint must be included in this bundle')
            cfg[key] = str(checkpoint)
    config_path = work / (name + '.json')
    atomic_json(config_path, dict(concurrency=1, experiments=[copied]))
    env = dict(os.environ, PYTHONUNBUFFERED='1',
               CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               BATCH_RUN_GPUS='' if gpu is None else str(gpu),
               BATCH_RUN_REQUIRE_GPU='0' if gpu is None else '1',
               BATCH_RUN_IDLE_MEM_MB='100')
    console = (output / 'logs' / (name + '.launcher.log')).open('x')
    command = [sys.executable, '-u', str(bundle / 'code/src/batch_run.py'),
               '--base-dir', str(output / 'results'), '--model-base-dir', str(output / 'models'),
               str(config_path)]
    try:
        process = subprocess.Popen(command, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                   stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
    except Exception:
        console.close()
        raise
    return process, console


def log_path(output, name):
    return output / 'results' / name / 'logs' / (name + '.log')


def log_text(output, name):
    path = log_path(output, name)
    return path.read_text(errors='replace') if path.exists() else ''


def run(args, bundle, manifest, data):
    output = args.output.resolve()
    selected = set(args.only.split(',')) if args.only else {e['name'] for e in data['experiments']}
    names = {e['name'] for e in data['experiments']}
    require(selected and selected <= names, 'Unknown/empty experiment selection')
    experiments = [e for e in data['experiments'] if e['name'] in selected]
    require(bool(args.gpus) != args.cpu, 'Choose --gpus or explicit --cpu')
    gpus = [None] if args.cpu else parse_gpus(args.gpus)
    require(args.poll_seconds >= 1, '--poll-seconds must be >=1')
    require(not (args.wait_for_idle and args.cpu), '--wait-for-idle requires GPUs')
    max_active = min(data['concurrency'], len(experiments), len(gpus))
    if not args.wait_for_idle:
        gpus = gpus[:max_active]
    if args.background:
        require(not args.prepared_output, 'Invalid background arguments')
        output.mkdir(parents=True, exist_ok=False)
        with (output / 'controller.log').open('x') as stream:
            argv = ['--bundle', str(bundle), '--timers', args.timers, 'run',
                    '--output', str(output), '--prepared-output']
            argv += ['--cpu'] if args.cpu else ['--gpus', args.gpus]
            if args.only:
                argv += ['--only', args.only]
            if args.wait_for_idle:
                argv += ['--wait-for-idle', '--poll-seconds', str(args.poll_seconds)]
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                        *argv], cwd=bundle,
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        info = process_info(process.pid)
        atomic_json(output / 'controller_launch.json', dict(pid=process.pid, start_ticks=info['start_ticks'] if info else None,
                    runner=str(Path(__file__).resolve()), launched_utc=stamp()))
        print('Background controller PID:', process.pid, 'log:', output / 'controller.log')
        return 0
    if args.prepared_output:
        require(output.is_dir() and not (output / 'state.json').exists(), 'Output already used')
    else:
        output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    info = process_info(os.getpid())
    state = dict(review_id=manifest['confirmed_review_id'], status='starting', controller_pid=os.getpid(),
                 controller_start_ticks=info['start_ticks'], runner=str(Path(__file__).resolve()),
                 started_utc=stamp(), bundle=str(bundle), runs=[
                     dict(name=e['name'], task=e['task'], hours=e['config']['MAX_TRAIN_HOURS'], status='pending')
                     for e in experiments])
    if args.wait_for_idle:
        state.update(wait_for_idle=True, gpu_pool=gpus, poll_seconds=args.poll_seconds,
                     poll_count=0, last_gpu_check_utc=None, next_gpu_check_utc=None)
    atomic_json(output / 'state.json', state)
    locks = []
    gpu_locks = {}
    active = {}
    pending = deque(experiments)
    records = {r['name']: r for r in state['runs']}
    try:
        if args.cpu:
            subprocess.run([sys.executable, '-c', 'import torch, matplotlib'], check=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        elif not args.wait_for_idle:
            locks = reserve_gpus(gpus, manifest)
        state['status'] = 'running'
        signalled = set()
        next_gpu_poll = 0.0
        while pending or active:
            if stopping:
                for exp in pending:
                    records[exp['name']]['status'] = 'cancelled'
                pending.clear()
                state['status'] = 'stopping'
                for process, stream, exp in active.values():
                    stop_training(process, signalled, '--- Return code:' in log_text(output, exp['name']))
            available = gpus if not args.wait_for_idle else []
            if (args.wait_for_idle and not stopping and pending and
                    len(active) < max_active and time.monotonic() >= next_gpu_poll):
                state['last_gpu_check_utc'] = stamp()
                state['poll_count'] += 1
                state['gpu_reservation_deferred'] = {}
                try:
                    table = gpu_snapshot()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    state['gpu_probe_error'] = str(error)
                    print('GPU_CHECK_FAILED', str(error), flush=True)
                else:
                    validate_gpu_pool(gpus, manifest, table)
                    state.pop('gpu_probe_error', None)
                    state['gpu_snapshot'] = {str(gpu): table[gpu] for gpu in gpus}
                    available = [gpu for gpu in gpus if gpu not in active
                                 and table[gpu]['memory_mib'] < 100
                                 and table[gpu]['utilization'] == 0
                                 and not table[gpu]['has_compute_process']]
                    print('GPU_CHECK', state['poll_count'], state['last_gpu_check_utc'],
                          'idle=', available, 'pending=', len(pending), flush=True)
                next_gpu_poll = time.monotonic() + args.poll_seconds
                state['next_gpu_check_utc'] = (
                    datetime.now(timezone.utc) + timedelta(seconds=args.poll_seconds)).isoformat()
            for gpu in available:
                if not stopping and pending and gpu not in active and len(active) < max_active:
                    if args.wait_for_idle:
                        try:
                            handles = reserve_gpus([gpu], manifest)
                        except (ValueError, BlockingIOError) as error:
                            state['gpu_reservation_deferred'][str(gpu)] = str(error)
                            print('GPU_BUSY_ON_RECHECK', gpu, str(error), flush=True)
                            continue
                        gpu_locks[gpu] = handles
                        # A stop can arrive during the CUDA preflight.
                        if stopping:
                            break
                    exp = pending.popleft()
                    process, stream = launch_one(bundle, output, exp, gpu)
                    active[gpu] = (process, stream, exp)
                    records[exp['name']].update(status='running', gpu=gpu, batch_pid=process.pid,
                                                started_utc=stamp(), log=str(log_path(output, exp['name'])))
                    print('STARTED', exp['name'], 'GPU', gpu, 'hours', exp['config']['MAX_TRAIN_HOURS'], flush=True)
            for gpu, (process, stream, exp) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                text = log_text(output, exp['name'])
                codes = re.findall(r'--- Return code: (-?\d+) ---', text)
                trainer_code = int(codes[-1]) if codes else None
                status = 'failed'
                checkpoint = output / 'models' / exp['name'] / (exp['name'] + '_resume.pth')
                if trainer_code == 42 and checkpoint.is_file():
                    status = 'stopped_by_user' if stopping or '[STOP]' in text else 'timed_out'
                elif trainer_code == 0 and code == 0:
                    status = 'completed'
                records[exp['name']].update(status=status, finished_utc=stamp(), launcher_returncode=code,
                                            trainer_returncode=trainer_code,
                                            resume_checkpoint=str(checkpoint) if checkpoint.exists() else None)
                print('FINISHED', exp['name'], status, 'trainer_returncode', trainer_code, flush=True)
                del active[gpu]
                for handle in gpu_locks.pop(gpu, []):
                    handle.close()
            if not stopping:
                state['status'] = 'waiting_for_gpu' if pending and not active and args.wait_for_idle else 'running'
            if not pending:
                state['next_gpu_check_utc'] = None
            atomic_json(output / 'state.json', state)
            if pending or active:
                time.sleep(1)
        state.update(status='stopped' if stopping else 'finished', finished_utc=stamp())
        atomic_json(output / 'state.json', state)
        return 1 if any(r['status'] == 'failed' for r in state['runs']) else 0
    except Exception as error:
        # Protect training progress if the controller itself encounters an error.
        signalled = set()
        while any(process.poll() is None for process, _, _ in active.values()):
            for process, _, exp in active.values():
                if process.poll() is None:
                    stop_training(process, signalled, '--- Return code:' in log_text(output, exp['name']))
            time.sleep(0.5)
        for _, stream, _ in active.values():
            stream.close()
        state.update(status='failed', error=str(error), finished_utc=stamp())
        atomic_json(output / 'state.json', state)
        raise
    finally:
        for handle in locks:
            handle.close()
        for handles in gpu_locks.values():
            for handle in handles:
                handle.close()
        lock.close()


def show_status(output):
    state = json.loads((output / 'state.json').read_text())
    print('Controller:', state['status'])
    if state.get('wait_for_idle'):
        print('GPU pool:', state['gpu_pool'], 'poll seconds:', state['poll_seconds'],
              'checks:', state['poll_count'])
        print('Last GPU check:', state['last_gpu_check_utc'], 'next:', state['next_gpu_check_utc'])
        if state.get('gpu_probe_error'):
            print('GPU probe error:', state['gpu_probe_error'])
    for record in state['runs']:
        text = log_text(output, record['name'])
        rows = re.findall(r'^Epoch\s+(\d+)\s+\| Train: Loss=([\d.eE+-]+) Acc=([\d.]+)% \| Test: Acc=([\d.]+)%', text, re.M)
        print(record['name'], record['status'], 'GPU', record.get('gpu'),
              'last epoch/loss/train%/test%=', rows[-1] if rows else None)


def stop(output):
    state_path = output / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['status'] in ('finished', 'failed', 'stopped'):
            print('Already stopped:', state['status'])
            return
        pid = state['controller_pid']
        start = state['controller_start_ticks']
    else:
        state = json.loads((output / 'controller_launch.json').read_text())
        pid, start = state['pid'], state['start_ticks']
    info = process_info(pid)
    require(info and info['state'] != 'Z' and info['start_ticks'] == start
            and state['runner'] in info['argv'], 'Controller is absent or PID no longer matches')
    os.kill(pid, signal.SIGTERM)
    print('Stop requested: pending tasks cancelled; trainers save at the next eval point.')
    print('Use status to confirm all jobs have stopped.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--timers', required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    start = sub.add_parser('run')
    start.add_argument('--gpus')
    start.add_argument('--cpu', action='store_true', help='Explicit CPU smoke-test mode')
    start.add_argument('--output', required=True, type=Path)
    start.add_argument('--only', help='Comma-separated experiment names for this machine')
    start.add_argument('--background', action='store_true')
    start.add_argument('--wait-for-idle', action='store_true', help='Queue until an allowed GPU is idle')
    start.add_argument('--poll-seconds', type=int, default=300, help='GPU availability polling interval')
    start.add_argument('--prepared-output', action='store_true', help=argparse.SUPPRESS)
    for command in ('status', 'stop'):
        child = sub.add_parser(command)
        child.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        bundle = args.bundle.resolve()
        if args.command in ('status', 'stop'):
            return show_status(args.output.resolve()) if args.command == 'status' else stop(args.output.resolve())
        manifest, data = check_bundle(bundle, json.loads(args.timers))
        return 0 if args.command == 'check' else run(args, bundle, manifest, data)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, 'Error: ' + str(error) + '\n')


if __name__ == '__main__':
    sys.exit(main())
