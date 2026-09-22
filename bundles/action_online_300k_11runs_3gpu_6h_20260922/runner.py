#!/usr/bin/env python3
"""All-or-nothing GPU admission and a portable FIFO experiment queue."""
import argparse
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

try:
    import package_utils as common
except ModuleNotFoundError:
    import experiment_package_runner as common

require = common.require
atomic_json = common.atomic_json
stamp = common.stamp


def inventory():
    raw = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True, timeout=20)
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'],
        text=True, timeout=20)
    occupied = set(apps.split())
    result = {}
    for line in raw.splitlines():
        index, identity, name, memory, utilization = [p.strip() for p in line.split(',')]
        result[int(index)] = dict(uuid=identity, name=name, memory_mib=float(memory),
                                 utilization=float(utilization), has_compute_process=identity in occupied)
    return result


def gpu_pool(requested, manifest, table):
    if requested != 'auto':
        pool = common.parse_gpus(requested)
    else:
        pool = sorted(table)
        if socket.gethostname() == manifest['packaging_host']:
            pool = [g for g in pool if g in manifest['local_gpu_allowlist']]
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is not None:
            tokens = set(p.strip() for p in visible.split(','))
            pool = [g for g in pool if str(g) in tokens or table[g]['uuid'] in tokens]
    common.validate_gpu_pool(pool, manifest, table)
    return pool


def is_idle(row):
    return row['memory_mib'] < 100 and row['utilization'] == 0 and not row['has_compute_process']


def admit(requested, required, manifest):
    """Reserve every required card before any trainer or background controller."""
    table = inventory()
    pool = gpu_pool(requested, manifest, table)
    idle = [g for g in pool if is_idle(table[g])]
    print(f'GPU CHECK: required={required}, idle={len(idle)}, candidates={pool}', flush=True)
    for gpu in pool:
        row = table[gpu]
        print(f"  GPU {gpu}: {row['name']} | {row['memory_mib']} MiB | "
              f"util={row['utilization']}% | {'IDLE' if gpu in idle else 'BUSY'}", flush=True)
    require(len(idle) >= required,
            f'Need {required} idle GPUs, found {len(idle)}; NO experiments started. No automatic waiting.')
    selected, handles = [], []
    try:
        # Lock contention is also an unavailable resource. Try other idle cards.
        for gpu in idle:
            try:
                lease = common.reserve_gpus([gpu], manifest)
            except (BlockingIOError, ValueError) as error:
                print(f'GPU {gpu} unavailable on reservation: {error}', flush=True)
                continue
            selected.append(gpu)
            handles.extend(lease)
            if len(selected) == required:
                break
        require(len(selected) == required,
                f'Could reserve only {len(selected)}/{required} GPUs; NO experiments started.')
        # Detect foreign jobs arriving while CUDA dependencies were checked.
        final = inventory()
        require(all(g in final and final[g]['uuid'] == table[g]['uuid'] and is_idle(final[g])
                    for g in selected), 'GPU allocation changed during preflight; NO experiments started.')
        return selected, handles, table
    except BaseException:
        for handle in handles:
            handle.close()
        raise


def choose_experiments(data, only=None, previous=None, manifest=None):
    names = {e['name'] for e in data['experiments']}
    chosen = set(only.split(',')) if only else names
    require(chosen and chosen <= names, 'Unknown/empty --only selection')
    old = None
    if previous:
        old = json.loads((Path(previous) / 'state/status.json').read_text())
        require(old['review_id'] == manifest['confirmed_review_id'],
                'Resume batch must use the same approved parameter version')
        require(old['status'] in ('finished', 'failed', 'stopped'), 'Previous batch is still active')
        old = {r['name']: r for r in old['runs']}
    selected = []
    for source in data['experiments']:
        if source['name'] not in chosen:
            continue
        exp = json.loads(json.dumps(source))
        if old is not None:
            record = old.get(exp['name'])
            if not record or (not only and record['status'] == 'completed'):
                continue
            directory = Path(record['run_dir']) / 'checkpoints'
            candidates = [directory / 'model_resume.pth', directory / 'model_latest.pth']
            checkpoint = next((p for p in candidates if p.is_file()), None)
            if checkpoint is None:
                # An explicitly requested missing checkpoint is an error; do not start fresh silently.
                if only:
                    raise ValueError('No full resume checkpoint for ' + exp['name'])
                continue
            exp['config']['RESUME_FROM'] = str(checkpoint.resolve())
            exp['config']['INIT_FROM'] = None
            best = directory / 'model_best.pth'
            if best.is_file():
                exp['previous_best_checkpoint'] = str(best.resolve())
        selected.append(exp)
    require(selected, 'No eligible experiments; use --only to explicitly resume a completed run')
    return selected


def print_mapping(experiments, gpus, table):
    print('\nINITIAL GPU -> TASK (physical indices)', flush=True)
    for gpu, exp in zip(gpus, experiments):
        cfg = exp['config']
        print(f"  GPU {gpu} ({table[gpu]['name']}) -> {exp['name']} | "
              f"MISS_LEN={cfg['MISS_LEN']} L={cfg['N_LAYER']} | timer={cfg['MAX_TRAIN_HOURS']}h", flush=True)
    for number, exp in enumerate(experiments[len(gpus):], 1):
        print(f"  QUEUED {number}: {exp['name']} -> next available reserved GPU", flush=True)
    print('Each timer starts at its own training loop; queue time is excluded.\n', flush=True)


def environment(gpu):
    return dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                EXPERIMENT_REQUIRE_GPU='0' if gpu is None else '1')


def launch_training(bundle, record, gpu):
    directory = Path(record['run_dir'])
    directory.mkdir(parents=True, exist_ok=False)
    for kind in ('configs', 'logs', 'checkpoints', 'plots', 'metrics'):
        (directory / kind).mkdir()
    cfg = dict(record['experiment']['config'], SAVE_PATH=str(directory / 'checkpoints/model.pth'))
    if record['experiment'].get('previous_best_checkpoint'):
        shutil.copyfile(record['experiment']['previous_best_checkpoint'],
                        directory / 'checkpoints/model_best.pth')
    actual = directory / 'configs/experiment.json'
    atomic_json(actual, dict(main=cfg, _BATCH_RUN_MERGED=True))
    atomic_json(directory / 'configs/run.json', dict(
        name=record['name'], gpu=gpu, hours=cfg['MAX_TRAIN_HOURS'],
        review_id=record['review_id'], started_utc=stamp(), bundle=str(bundle)))
    shutil.copyfile(bundle / 'PARAMETERS.md', directory / 'configs/PARAMETERS.md')
    out = (directory / 'logs/train.log').open('x')
    err = (directory / 'logs/train.err').open('x')
    out.write(f"Experiment: {record['name']}\nPhysical GPU: {gpu}\n"
              f"MAX_TRAIN_HOURS: {cfg['MAX_TRAIN_HOURS']}\n"
              f"Resolved config: {json.dumps(cfg, sort_keys=True)}\n\n")
    out.flush()
    bootstrap = (
        'import os,sys,torch\n'
        'if os.environ["EXPERIMENT_REQUIRE_GPU"] == "1":\n'
        ' assert torch.cuda.is_available() and torch.cuda.device_count() == 1, '
        '"CUDA unavailable; refusing CPU/cuda:0 fallback"\n'
        'sys.path.insert(0,sys.argv[1])\n'
        'from experiment import run_experiment\n'
        'run_experiment(sys.argv[2])\n')
    try:
        process = subprocess.Popen(
            [sys.executable, '-u', '-c', bootstrap, str(bundle / 'code/src'), str(actual)],
            cwd=directory, env=environment(gpu), stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, start_new_session=True)
    except BaseException:
        out.close()
        err.close()
        raise
    return process, [out, err]


def launch_postprocess(bundle, record, gpu, delivery):
    directory = Path(record['run_dir'])
    out = (directory / 'logs/evaluation.log').open('x')
    err = (directory / 'logs/evaluation.err').open('x')
    command = [sys.executable, '-u', str(bundle / 'code/scripts/action_run_postprocess.py'),
               'evaluate', '--run-dir', str(directory),
               '--num-samples', str(delivery['clean_eval_samples']),
               '--sample-seed', str(delivery['clean_eval_seed'])]
    if gpu is None:  # Internal CPU integration tests; not exposed in production CLI.
        command.append('--cpu')
    try:
        process = subprocess.Popen(command, env=environment(gpu), cwd=directory,
                                   stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
    except BaseException:
        out.close()
        err.close()
        raise
    return process, [out, err]


def checkpoint_for(record):
    directory = Path(record['run_dir']) / 'checkpoints'
    return next((str(p) for p in (directory / 'model_resume.pth', directory / 'model_latest.pth')
                 if p.is_file()), None)


def training_status(code, record, stopping):
    if code == 0:
        return 'completed'
    if code == 42:
        log = (Path(record['run_dir']) / 'logs/train.log').read_text(errors='replace')
        return 'stopped_by_user' if stopping or '[STOP]' in log else 'timed_out'
    return 'failed'


def dispatch(bundle, manifest, session, handles):
    """All leases are already held. A slot includes training and its clean evaluation."""
    payload = json.loads((session / 'configs/dispatch.json').read_text())
    state = payload['state']
    require(state['review_id'] == manifest['confirmed_review_id'], 'Dispatch review mismatch')
    require(len(handles) == manifest['delivery']['required_gpus'], 'Missing GPU leases')
    info = common.process_info(os.getpid())
    state.update(status='running', controller_pid=os.getpid(),
                 controller_start_ticks=info['start_ticks'], controller_path=str(Path(__file__).resolve()))
    path = session / 'state/status.json'
    atomic_json(path, state)
    pending = deque(state['runs'])
    active = {}
    stopping = False
    failure = None

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while pending or active:
            if stopping:
                for record in pending:
                    record['status'] = 'cancelled'
                pending.clear()
                state['status'] = 'stopping'
                for item in active.values():
                    if not item.get('stop_sent'):
                        try:
                            item['process'].send_signal(signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        item['stop_sent'] = True
            if not stopping:
                for gpu in state['gpus']:
                    if pending and gpu not in active:
                        record = pending.popleft()
                        try:
                            process, streams = launch_training(bundle, record, gpu)
                        except BaseException:
                            record['status'] = 'failed'
                            raise
                        record.update(status='running', phase='training', gpu=gpu,
                                      started_utc=stamp(), trainer_pid=process.pid)
                        active[gpu] = dict(record=record, process=process, streams=streams, phase='training')
                        print('STARTED', record['name'], 'GPU', gpu, 'timer=',
                              record['experiment']['config']['MAX_TRAIN_HOURS'], 'h', flush=True)
            for gpu, item in list(active.items()):
                code = item['process'].poll()
                if code is None:
                    continue
                for stream in item['streams']:
                    stream.close()
                record = item['record']
                if item['phase'] == 'training':
                    result = training_status(code, record, stopping)
                    record.update(training_status=result, trainer_returncode=code,
                                  training_finished_utc=stamp(), resume_checkpoint=checkpoint_for(record))
                    if code in (0, 42) and not record['resume_checkpoint']:
                        result = 'failed'
                        record['error'] = 'Trainer exited without a full resume checkpoint'
                    if result in ('completed', 'timed_out') and not stopping:
                        process, streams = launch_postprocess(bundle, record, gpu, manifest['delivery'])
                        record.update(status='evaluating', phase='clean_evaluation')
                        active[gpu] = dict(record=record, process=process, streams=streams, phase='evaluation')
                        atomic_json(path, state)
                        continue
                    record['status'] = result
                else:
                    record.update(evaluation_returncode=code,
                                  status=record['training_status'] if code == 0 else
                                  ('stopped_by_user' if stopping else 'evaluation_failed'))
                record.update(phase='finished', finished_utc=stamp())
                print('FINISHED', record['name'], record['status'], 'GPU', gpu,
                      'resume=', record['resume_checkpoint'], flush=True)
                del active[gpu]
            atomic_json(path, state)
            if pending or active:
                time.sleep(0.5)
        state.update(status='stopped' if stopping else 'finished', finished_utc=stamp())
        atomic_json(path, state)
        return int(any(r['status'] in ('failed', 'evaluation_failed') for r in state['runs']))
    except BaseException as error:
        failure = str(error)
        for record in pending:
            record['status'] = 'cancelled'
        # SIGTERM is handled by the training engine at its next checkpoint boundary.
        for item in active.values():
            if item['process'].poll() is None:
                try:
                    item['process'].send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
        while any(item['process'].poll() is None for item in active.values()):
            time.sleep(0.5)
        for item in active.values():
            for stream in item['streams']:
                stream.close()
            item['record'].update(status='stopped_by_user', resume_checkpoint=checkpoint_for(item['record']))
        state.update(status='failed', error=failure, finished_utc=stamp())
        atomic_json(path, state)
        raise
    finally:
        for handle in handles:
            handle.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def start(args, bundle, manifest, data):
    delivery = manifest['delivery']
    experiments = choose_experiments(data, args.only, args.resume_batch, manifest)
    gpus, handles, table = admit(args.gpus, delivery['required_gpus'], manifest)
    try:
        # Validate full-state resume files before starting any of the jobs.
        for exp in experiments:
            checkpoint = exp['config'].get('RESUME_FROM')
            if checkpoint:
                subprocess.run(
                    [sys.executable, str(bundle / 'code/scripts/action_run_postprocess.py'),
                     'validate', '--checkpoint', checkpoint, '--config-json', json.dumps(exp['config'])],
                    check=True, env=environment(None))
        print_mapping(experiments, gpus, table)
        root = Path(args.output_root or delivery['output_root']).expanduser().resolve()
        tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
        session = root / (manifest['batch_name'] + '__' + tag)
        for kind in ('configs', 'logs', 'state'):
            (session / kind).mkdir(parents=True, exist_ok=False)
        records = [dict(name=e['name'], experiment=e, status='pending', review_id=manifest['confirmed_review_id'],
                        run_dir=str(root / (e['name'] + '__' + tag))) for e in experiments]
        state = dict(review_id=manifest['confirmed_review_id'], status='prepared', started_utc=stamp(),
                     bundle=str(bundle), gpus=gpus, required_gpus=delivery['required_gpus'],
                     resume_batch=str(args.resume_batch) if args.resume_batch else None, runs=records)
        atomic_json(session / 'configs/dispatch.json', dict(state=state))
        atomic_json(session / 'state/status.json', state)
        shutil.copyfile(bundle / 'manifest.json', session / 'configs/manifest.json')
        shutil.copyfile(bundle / 'PARAMETERS.md', session / 'configs/PARAMETERS.md')
        print('Batch directory:', session, flush=True)
        if not args.background:
            return dispatch(bundle, manifest, session, handles)
        fds = [handle.fileno() for handle in handles]
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--bundle', str(bundle), '--timers', args.timers, '_dispatch',
                   '--batch', str(session), '--lease-fds', ','.join(map(str, fds))]
        with (session / 'logs/controller.log').open('x') as stream:
            # Inherit the SAME locked file descriptions, so there is no unlock/relock gap.
            process = subprocess.Popen(command, cwd=bundle, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=fds)
        info = common.process_info(process.pid)
        atomic_json(session / 'state/controller_launch.json', dict(
            pid=process.pid, start_ticks=info['start_ticks'] if info else None))
        print('Background controller PID:', process.pid, flush=True)
        print('Controller log:', session / 'logs/controller.log', flush=True)
        print('Status: bash run.sh status --batch', session, flush=True)
        return 0
    finally:
        for handle in handles:
            handle.close()


def status(session):
    state = json.loads((session / 'state/status.json').read_text())
    print('Batch:', session, '\nController:', state['status'], 'GPUs:', state['gpus'])
    for record in state['runs']:
        log = Path(record['run_dir']) / 'logs/train.log'
        rows = re.findall(r'^Epoch\s+(\d+)\s+\| Train: Loss=([\d.eE+-]+) Acc=([\d.]+)% \| Test: Acc=([\d.]+)%',
                          log.read_text(errors='replace') if log.exists() else '', re.M)
        print(record['name'], record['status'], 'GPU', record.get('gpu'),
              'epoch/loss/train%/test%=', rows[-1] if rows else None)


def stop(session):
    state = json.loads((session / 'state/status.json').read_text())
    if state['status'] in ('finished', 'failed', 'stopped'):
        print('Already stopped:', state['status'])
        return
    if 'controller_pid' in state:
        pid, ticks = state['controller_pid'], state['controller_start_ticks']
    else:
        launch = json.loads((session / 'state/controller_launch.json').read_text())
        pid, ticks = launch['pid'], launch['start_ticks']
    info = common.process_info(pid)
    require(info and info['start_ticks'] == ticks, 'Controller no longer exists (PID identity mismatch)')
    os.kill(pid, signal.SIGTERM)
    print('Stop requested: cancel queued tasks; active training saves at its next evaluation boundary.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--timers', required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    for command in ('run', 'resume'):
        ap = sub.add_parser(command)
        ap.add_argument('--gpus', default='auto', help='auto or physical candidate indices, e.g. 0,2,5,7')
        ap.add_argument('--output-root', default=None)
        ap.add_argument('--background', action='store_true')
        ap.add_argument('--only', default=None)
        ap.add_argument('--resume-batch', type=Path, required=command == 'resume')
    for command in ('status', 'stop'):
        sub.add_parser(command).add_argument('--batch', type=Path, required=True)
    internal = sub.add_parser('_dispatch', help=argparse.SUPPRESS)
    internal.add_argument('--batch', type=Path, required=True)
    internal.add_argument('--lease-fds', required=True)
    args = parser.parse_args()
    try:
        bundle = args.bundle.resolve()
        manifest, data = common.check_bundle(bundle, json.loads(args.timers))
        require(manifest.get('delivery'), 'This package has no remote delivery settings')
        if args.command in ('run', 'resume'):
            return start(args, bundle, manifest, data)
        if args.command == '_dispatch':
            handles = [os.fdopen(int(fd), 'a') for fd in args.lease_fds.split(',')]
            for handle in handles:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return dispatch(bundle, manifest, args.batch.resolve(), handles)
        if args.command == 'status':
            status(args.batch.resolve())
        if args.command == 'stop':
            stop(args.batch.resolve())
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('ERROR:', error, file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
