"""Curriculum chain scheduler: warm-start N=2 -> N=3 -> N=4 -> N=5.

For each scale chain, serially:
  stage N: train mixed_ab on rules[:N] with INIT_FROM = previous stage's
  best checkpoint; success = best test acc >= --threshold; on success the
  stage's checkpoint becomes the next stage's donor; on failure the chain
  stops.

Chains run in parallel threads, each pinned to its own GPU (scheduler
spawns core.py directly, bypassing batch_run's GPU auto-detection).

Donors:
  l1d256r8h2: existing rules12345 N=2 checkpoints (already trained)
  others:     mixed_basic_n2_ladder_rules12 batch (must finish first;
              scheduler polls until donor exists)

Logs: /mnt/workspace/hujiachen/recursion_results/curriculum_chains/logs/
State: /mnt/workspace/hujiachen/recursion_results/curriculum_chains/chain_state.json

Usage:
  setsid nohup python src/curriculum_chain_Ntask.py > .chain_tmp/curriculum_scheduler.log 2>&1 &
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = '/mnt/workspace/hujiachen/recursion_results/curriculum_chains'
LOG_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_DIR = '/mnt/workspace/hujiachen/models/curriculum_chains'
STATE_PATH = os.path.join(BASE_DIR, 'chain_state.json')

RULES = [[1, 1], [2, 3], [3, 5], [4, 7], [5, 11]]
P = 127
# scale -> (D_MODEL, N_LAYER, N_HEAD, donor glob dir, donor name pattern)
SCALES = {
    'l1d256r8h2': dict(d=256, l=1, h=2,
                       donor_dir='/mnt/workspace/hujiachen/models/mixed_basic_d256l1r8h2_p127_rules12345',
                       donor_pat='mixed_basic_d256l1r8h2_P127_N2_e0.7_seed{s}.pth',
                       donor_logs='/mnt/workspace/hujiachen/recursion_results/mixed_basic_d256l1r8h2_p127_rules12345/logs',
                       donor_log_pat='mixed_basic_d256l1r8h2_P127_N2_e0.7_seed{s}.log'),
    'l1d512r8h4': dict(d=512, l=1, h=4,
                       donor_dir='/mnt/workspace/hujiachen/models/mixed_basic_n2_ladder_rules12',
                       donor_pat='mixed_basic_d512l1r8h4_P127_rules12_N2_e0.7_seed{s}.pth',
                       donor_logs='/mnt/workspace/hujiachen/recursion_results/mixed_basic_n2_ladder_rules12/logs',
                       donor_log_pat='mixed_basic_d512l1r8h4_P127_rules12_N2_e0.7_seed{s}.log'),
    'l2d256r8h2': dict(d=256, l=2, h=2,
                       donor_dir='/mnt/workspace/hujiachen/models/mixed_basic_n2_ladder_rules12',
                       donor_pat='mixed_basic_d256l2r8h2_P127_rules12_N2_e0.7_seed{s}.pth',
                       donor_logs='/mnt/workspace/hujiachen/recursion_results/mixed_basic_n2_ladder_rules12/logs',
                       donor_log_pat='mixed_basic_d256l2r8h2_P127_rules12_N2_e0.7_seed{s}.log'),
    'l2d512r8h4': dict(d=512, l=2, h=4,
                       donor_dir='/mnt/workspace/hujiachen/models/mixed_basic_n2_ladder_rules12',
                       donor_pat='mixed_basic_d512l2r8h4_P127_rules12_N2_e0.7_seed{s}.pth',
                       donor_logs='/mnt/workspace/hujiachen/recursion_results/mixed_basic_n2_ladder_rules12/logs',
                       donor_log_pat='mixed_basic_d512l2r8h4_P127_rules12_N2_e0.7_seed{s}.log'),
    'l4d512r8h4': dict(d=512, l=4, h=4,
                       donor_dir='/mnt/workspace/hujiachen/models/mixed_basic_scale_matrix_h4_rules12',
                       donor_pat='mixed_basic_d512l4r8h4_P127_rules12_N2_e0.7_seed{s}.pth',
                       donor_logs='/mnt/workspace/hujiachen/recursion_results/mixed_basic_scale_matrix_h4_rules12/logs',
                       donor_log_pat='mixed_basic_d512l4r8h4_P127_rules12_N2_e0.7_seed{s}.log'),
}
SEEDS = [17996, 18318, 34789, 44536]
GPUS = {'l1d256r8h2': '1', 'l1d512r8h4': '2', 'l2d256r8h2': '3', 'l2d512r8h4': '5', 'l4d512r8h4': '6'}

THRESHOLD = float(os.environ.get('CURRICULUM_THRESHOLD', '0.95'))
DONOR_MIN = float(os.environ.get('CURRICULUM_DONOR_MIN', '0.99'))
_only = os.environ.get('CURRICULUM_SCALES')
if _only:
    SCALES = {k: v for k, v in SCALES.items() if k in _only.split(',')}
POLL_S = 120

_state_lock = threading.Lock()
_state = {}


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, '_scheduler.log'), 'a') as f:
        f.write(line + '\n')


def save_state():
    with _state_lock:
        with open(STATE_PATH, 'w') as f:
            json.dump(_state, f, indent=1)


def best_acc(log_path):
    """Parse final best test accuracy from a training log."""
    if not os.path.exists(log_path):
        return None
    txt = open(log_path, errors='ignore').read()
    m = re.findall(r'Best test accuracy: ([0-9.]+)%', txt)
    if m:
        return float(m[-1]) / 100
    m = re.findall(r'Best=([0-9.]+)%\(@\d+\)', txt)
    return float(m[-1]) / 100 if m else None


def pick_donor(scale):
    """Best-seed N=2 donor checkpoint; None until a good one exists."""
    cfg = SCALES[scale]
    best = (None, 0.0)
    for s in SEEDS:
        pth = os.path.join(cfg['donor_dir'], cfg['donor_pat'].format(s=s))
        acc = best_acc(os.path.join(cfg['donor_logs'],
                                    cfg['donor_log_pat'].format(s=s)))
        if os.path.exists(pth) and acc is not None and acc > best[1]:
            best = (pth, acc, s)
    if best[0] and best[1] >= DONOR_MIN:
        return best
    return None


def run_stage(scale, n, donor, donor_seed, gpu):
    """Train stage N warm-started from donor; return (ckpt_path, acc)."""
    cfg = SCALES[scale]
    name = f'curr_{scale}_N{n}_seed{donor_seed}'
    save_path = os.path.join(MODEL_DIR, f'{name}.pth')
    base = json.load(open(os.path.join(REPO, 'src', 'config.json')))
    override = {
        'TASK': 'mixed_ab',
        'RANDOM_SEED': donor_seed, 'P': P, 'D_MODEL': cfg['d'],
        'N_LAYER': cfg['l'], 'N_HEAD': cfg['h'], 'MLP_RATIO': 8,
        'AB_PAIRS': RULES[:n],
        'MIXED_AB_MAX_UNIQUE_RATIOS': [0.7] * n,
        'USE_AB_TAG': False, 'USE_CONDITIONAL_WTE': False, 'NUM_MASK': 2,
        'INIT_FROM': donor, 'SAVE_PATH': save_path,
    }
    from protocol import BATCH_RUN_MERGED_FLAG
    merged = dict(base)
    merged_main = dict(base.get('main', {}))
    merged_main.update(base.get('mixed_ab', {}))
    merged_main.update(override)
    merged['main'] = merged_main
    merged[BATCH_RUN_MERGED_FLAG] = True
    tmp_cfg = os.path.join(REPO, f'config_tmp_{name}.json')
    with open(tmp_cfg, 'w') as f:
        json.dump(merged, f)

    log_path = os.path.join(LOG_DIR, f'{name}.log')
    err_path = os.path.join(LOG_DIR, f'{name}.err')
    log(f'{name}: start (INIT_FROM={os.path.basename(donor)}, gpu={gpu})')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'PYTHONUNBUFFERED': '1'}
    with open(log_path, 'w') as lf, open(err_path, 'w') as ef:
        subprocess.run([sys.executable, os.path.join(REPO, 'src', 'core.py'),
                        tmp_cfg], cwd=REPO, env=env, stdout=lf, stderr=ef)
    acc = best_acc(log_path)
    log(f'{name}: finished, best acc={acc}')
    return save_path, acc


def run_chain(scale):
    gpu = GPUS[scale]
    donor_info = None
    while donor_info is None:
        donor_info = pick_donor(scale)
        if donor_info is None:
            log(f'{scale}: waiting for N=2 donor...')
            time.sleep(POLL_S)
    donor, acc0, seed = donor_info
    log(f'{scale}: donor={os.path.basename(donor)} acc={acc0:.4f} seed={seed}')
    with _state_lock:
        _state[scale] = {'donor': donor, 'donor_acc': acc0, 'seed': seed,
                         'stages': {}}
    save_state()

    for n in (3, 4, 5):
        ckpt, acc = run_stage(scale, n, donor, seed, gpu)
        ok = acc is not None and acc >= THRESHOLD
        with _state_lock:
            _state[scale]['stages'][f'N{n}'] = {'ckpt': ckpt, 'acc': acc,
                                                'success': ok}
        save_state()
        if not ok:
            log(f'{scale}: N{n} FAILED (acc={acc} < {THRESHOLD}); chain stops')
            return
        donor = ckpt
        log(f'{scale}: N{n} OK (acc={acc:.4f}), continue')
    log(f'{scale}: N5 succeeded; chain complete')


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    log(f'scheduler start; threshold={THRESHOLD}')
    threads = [threading.Thread(target=run_chain, args=(s,), daemon=False)
               for s in SCALES]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log('all chains done')


if __name__ == '__main__':
    main()
