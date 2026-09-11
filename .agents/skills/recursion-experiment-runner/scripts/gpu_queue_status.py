#!/usr/bin/env python3
"""Per-GPU queue status: one line per GPU card.

Running column: experiment, cumulative epochs (across resume rounds), current
and best test acc, elapsed wall time, timer (MAX_TRAIN_HOURS or ad-hoc timer
set via timer.sh, with remaining time).
Queued column: experiments still waiting in the card's orchestrator config
(and whether they carry a timer).

Usage: python gpu_queue_status.py
"""
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime

BASES = ['/data/cxm/recursion']
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
TIMER_DIR = os.path.join(REPO, 'logs', 'timers')


def sh(cmd):
    return subprocess.check_output(cmd, text=True, shell=False)


def gpu_uuid_map():
    out = sh(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'])
    return {uuid.strip(): int(idx) for idx, uuid in
            (l.split(',') for l in out.strip().splitlines())}


def cmdline(pid):
    try:
        return open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode()
    except OSError:
        return ''


def environ(pid):
    try:
        raw = open(f'/proc/{pid}/environ', 'rb').read().decode()
        return dict(kv.split('=', 1) for kv in raw.split('\0') if '=' in kv)
    except OSError:
        return {}


def find_log(name):
    """Locate an experiment's log across data bases; newest wins (names can
    repeat across batches, e.g. same experiment in 30k and 300k regimes)."""
    cands = []
    for base in BASES:
        cands += glob.glob(f'{base}/*/logs/{name}.log')
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def parse_log(path):
    """-> dict(total_eps, best, last_acc, elapsed_sec, max_train_hours, done)"""
    txt = open(path, errors='replace').read()
    times = [datetime.fromisoformat(m.group(1)).timestamp() for m in
             re.finditer(r'Time: (\d{4}-\d{2}-\d{2}T[\d:\.]+)', txt)]
    seg_starts = [m.start() for m in re.finditer(r'Time: \d{4}-\d{2}-\d{2}T', txt)]
    seg_starts.append(len(txt))
    mtime = os.path.getmtime(path)
    total_eps, elapsed = 0, 0.0
    for i in range(len(times)):
        eps = [int(m.group(1)) for m in
               re.finditer(r'^Epoch\s+(\d+)', txt[seg_starts[i]:seg_starts[i + 1]], re.M)]
        if not eps:
            continue
        total_eps += max(eps) - min(eps) + 1
        t1 = times[i + 1] if i + 1 < len(times) else mtime
        elapsed += max(t1 - times[i], 0)
    epochs = re.findall(r'^Epoch\s+\d+.*Test: Acc=\s*([\d.]+)%', txt, re.M)
    bests = [float(m.group(1)) for m in re.finditer(r'Best=\s*([\d.]+)%', txt)]
    mh = re.findall(r'^MAX_TRAIN_HOURS\s+(\S+)', txt, re.M)
    mth = None
    if mh and mh[-1] not in ('None', 'null'):
        mth = float(mh[-1])
    return {
        'eps': total_eps,
        'best': max(bests) if bests else None,
        'last_acc': float(epochs[-1]) if epochs else None,
        'elapsed': elapsed,
        'mth': mth,
        'done': 'Best test accuracy' in txt,
        'mtime': mtime,
    }


def fmt_dur(sec):
    h = sec / 3600
    return f'{h:.1f}h' if h >= 1 else f'{sec / 60:.0f}min'


def short_name(name):
    s = name
    for pat in ('mixed_basic_', 'addition_', 'tribonacci_', 'tetranacci_', 'action_v2_',
                '_d1024', '_P127', '_tr16ood32', '_e0.7', 'randmiss0.1', 'rules12', 'rules123'):
        s = s.replace(pat, '')
    s = s.replace('_randmiss', 'miss').replace('_', ' ')
    return s.strip()


def adhoc_timers():
    """timer.sh state files -> {name_substring: remaining_sec}"""
    out = {}
    for f in glob.glob(os.path.join(TIMER_DIR, '*.timer')):
        try:
            d = json.load(open(f))
            out[d['pattern']] = max(0, d['deadline'] - time.time())
        except Exception:
            pass
    return out


def main():
    uuid2idx = gpu_uuid_map()
    # trainers on GPUs
    gpu_trainers = {}  # gpu_idx -> (pid, name)
    out = sh(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'])
    for line in out.strip().splitlines():
        uuid, pid = (x.strip() for x in line.split(','))
        cmd = cmdline(pid)
        m = re.search(r'core\.py config_tmp_([A-Za-z0-9_.]+)\.json', cmd)
        if m and uuid in uuid2idx:
            gpu_trainers[uuid2idx[uuid]] = (pid, m.group(1))

    # orchestrators -> queued experiments
    gpu_queues = {}  # gpu_idx -> list of (name, mth)
    # chain_rounds.sh progress: config path -> (current_round, total_rounds),
    # parsed from the chain logs' '=== round r/N: <config> ...' lines
    chain_progress = {}
    for logf in glob.glob(os.path.join(REPO, 'logs', '*.log')):
        try:
            head = open(logf, errors='replace').read()
        except OSError:
            continue
        for m in re.finditer(r'round (\d+)/(\d+): (\S+\.json)', head):
            chain_progress[os.path.basename(m.group(3))] = (int(m.group(1)), int(m.group(2)))

    procs = subprocess.run(['pgrep', '-af', 'batch_run.py'], capture_output=True, text=True).stdout
    procs += subprocess.run(['pgrep', '-af', 'chain_rounds.sh'], capture_output=True, text=True).stdout
    seen_orchestrators = set()  # a chain script and its batch_run child share a config
    for line in procs.splitlines():
        pid_s, _, cmd = line.partition(' ')
        m = re.search(r'(experiments/[A-Za-z0-9_./]+\.json)', cmd)
        if not m or 'bash -c' in cmd:
            continue
        cfg_path = os.path.join(REPO, m.group(1))
        if not os.path.exists(cfg_path) or cfg_path in seen_orchestrators:
            continue
        seen_orchestrators.add(cfg_path)
        is_chain = 'chain_rounds.sh' in cmd or cfg_path in \
            {os.path.join(REPO, 'experiments', os.path.basename(k)) for k in chain_progress} | \
            {os.path.join(REPO, 'experiments', 'v2', os.path.basename(k)) for k in chain_progress}
        env = environ(pid_s.strip())
        gpus = env.get('BATCH_RUN_GPUS')
        if gpus is None:  # chain_rounds.sh: gpus is argv
            m2 = re.search(r'chain_rounds\.sh \S+ ([0-9,]+) (\d+)', cmd)
            gpus = m2.group(1) if m2 else None
        gpu_list = [int(g) for g in gpus.split(',')] if gpus else []
        base = '/data/cxm/recursion_v2' if 'recursion_v2' in cmd else '/data/cxm/recursion'
        stem = os.path.splitext(os.path.basename(cfg_path))[0]
        try:
            exps = json.load(open(cfg_path))['experiments']
        except Exception:
            continue

        def _completed(name):
            log = f'{base}/{stem}/logs/{name}.log'
            return os.path.exists(log) and 'Best test accuracy' in open(log, errors='replace').read()

        # Running trainers attributed to THIS orchestrator's GPUs (names can
        # repeat across batches, so match by GPU, not just by name).
        own_running = {gpu_trainers[g][1] for g in gpu_list if g in gpu_trainers}
        all_running = {t[1] for t in gpu_trainers.values()}
        queued = []
        if is_chain:
            # chain re-runs all incomplete experiments each round; if the
            # chain is in its last round, nothing is queued behind it
            r, n = chain_progress.get(os.path.basename(cfg_path), (0, 0))
            if r < n:
                queued = [(e['name'], e['config'].get('MAX_TRAIN_HOURS'))
                          for e in exps
                          if not _completed(e['name']) and e['name'] not in all_running]
        else:
            # plain batch_run: queue = entries after the EARLIEST experiment
            # currently running on this orchestrator's GPUs (batch never
            # revisits earlier entries, whether done or timed out)
            run_indices = [i for i, e in enumerate(exps) if e['name'] in own_running]
            if run_indices:
                anchor = min(run_indices)
                queued = [(e['name'], e['config'].get('MAX_TRAIN_HOURS'))
                          for e in exps[anchor + 1:]
                          if not _completed(e['name']) and e['name'] not in all_running]
        for g in gpu_list:
            for item in queued:
                if item not in gpu_queues.setdefault(g, []):
                    gpu_queues[g].append(item)

    timers = adhoc_timers()
    now = time.time()
    busy_gpus = sorted(set(gpu_trainers) | set(gpu_queues))
    for g in busy_gpus:
        run_part = 'idle'
        if g in gpu_trainers:
            pid, name = gpu_trainers[g]
            log = find_log(name)
            if log:
                st = parse_log(log)
                if not st['done'] and now - st['mtime'] < 600:
                    timer = '无定时'
                    if st['mth']:
                        timer = f'定时{st["mth"]:g}h'
                    for pat, rem in timers.items():
                        if pat in name:
                            timer += f'+临时timer(剩{fmt_dur(rem)})'
                    run_part = (f'{short_name(name)} | ep={st["eps"]} '
                                f'acc={st["last_acc"]}%(best {st["best"]}%) '
                                f'已跑{fmt_dur(st["elapsed"])} | {timer}')
                else:
                    run_part = f'{short_name(name)} | 进程在但日志停更/已结束'
        q = gpu_queues.get(g, [])
        q_part = '排队: ' + (', '.join(f'{short_name(n)}({"定时" + str(int(m)) + "h" if m else "无定时"})'
                                     for n, m in q) if q else '无')
        print(f'GPU{g}: {run_part}  ||  {q_part}')


if __name__ == '__main__':
    main()
