"""Reverse-order curriculum chain: rules45 -> +rule3 -> +rule2 -> +rule1.

Mirror of curriculum_chain_Ntask.py, but the rule order is reversed and
stage 1 (rules45, N=2) is trained from scratch (no external donor exists).
Each later stage warm-starts from the previous stage's best checkpoint;
success = best test acc >= THRESHOLD (stage1 needs >= DONOR_MIN since it
plays the donor role). On failure the chain stops.

One chain per seed, each pinned to its own GPU (spawns core.py directly,
bypassing batch_run's GPU auto-detection).

Logs:   /data/cxm/recursion/curriculum_chains_rev/logs/
Models: /data/cxm/models/curriculum_chains_rev/
State:  /data/cxm/recursion/curriculum_chains_rev/chain_state.json

Usage:
  setsid nohup python src/curriculum_chain_reverse.py > .chain_tmp/curriculum_rev_scheduler.log 2>&1 &
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = '/data/cxm/recursion/curriculum_chains_rev'
LOG_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_DIR = '/data/cxm/models/curriculum_chains_rev'
STATE_PATH = os.path.join(BASE_DIR, 'chain_state.json')

# rule1..rule5 = [1,1],[2,3],[3,5],[4,7],[5,11]; reversed chain order.
STAGES = [
    ('N2', [[4, 7], [5, 11]]),
    ('N3', [[4, 7], [5, 11], [3, 5]]),
    ('N4', [[4, 7], [5, 11], [3, 5], [2, 3]]),
    ('N5', [[4, 7], [5, 11], [3, 5], [2, 3], [1, 1]]),
]
P = 127
D_MODEL, N_LAYER, N_HEAD, MLP_RATIO = 512, 2, 4, 8
SEED_GPUS = {17996: '1', 18318: '0', 34789: '4', 44536: '6'}

THRESHOLD = float(os.environ.get('CURRICULUM_THRESHOLD', '0.95'))
DONOR_MIN = float(os.environ.get('CURRICULUM_DONOR_MIN', '0.99'))

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


def run_stage(seed, tag, pairs, donor, gpu):
    """Train one stage (INIT_FROM=donor, None = from scratch); return (ckpt, acc)."""
    name = f'currrev_d512l2_{tag}_seed{seed}'
    save_path = os.path.join(MODEL_DIR, f'{name}.pth')
    base = json.load(open(os.path.join(REPO, 'src', 'config.json')))
    override = {
        'TASK': 'mixed_ab',
        'RANDOM_SEED': seed, 'P': P, 'D_MODEL': D_MODEL,
        'N_LAYER': N_LAYER, 'N_HEAD': N_HEAD, 'MLP_RATIO': MLP_RATIO,
        'AB_PAIRS': pairs,
        'MIXED_AB_MAX_UNIQUE_RATIOS': [0.7] * len(pairs),
        'USE_AB_TAG': False, 'USE_CONDITIONAL_WTE': False, 'NUM_MASK': 2,
        'SAVE_PATH': save_path, 'MAX_TRAIN_HOURS': 6,
    }
    if donor:
        override['INIT_FROM'] = donor
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
    log(f'{name}: start (INIT_FROM={os.path.basename(donor) if donor else "scratch"}, gpu={gpu})')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'PYTHONUNBUFFERED': '1'}
    with open(log_path, 'w') as lf, open(err_path, 'w') as ef:
        subprocess.run([sys.executable, os.path.join(REPO, 'src', 'core.py'),
                        tmp_cfg], cwd=REPO, env=env, stdout=lf, stderr=ef)
    acc = best_acc(log_path)
    log(f'{name}: finished, best acc={acc}')
    return save_path, acc


def donor_ckpt(save_path):
    """Existing donor for the next stage: final SAVE_PATH if the stage
    completed, else the rolling _best.pth (covers timer-killed stages whose
    SAVE_PATH is never written)."""
    if os.path.exists(save_path):
        return save_path
    best = save_path.replace('.pth', '_best.pth')
    return best if os.path.exists(best) else None


def run_chain(seed):
    gpu = SEED_GPUS[seed]
    with _state_lock:
        _state[str(seed)] = {'stages': {}}
    save_state()
    donor = None
    for i, (tag, pairs) in enumerate(STAGES):
        need = DONOR_MIN if i == 0 else THRESHOLD
        ckpt, acc = run_stage(seed, tag, pairs, donor, gpu)
        donor_next = donor_ckpt(ckpt)
        ok = acc is not None and acc >= need and donor_next is not None
        with _state_lock:
            _state[str(seed)]['stages'][tag] = {'ckpt': donor_next, 'acc': acc,
                                                'success': ok}
        save_state()
        if not ok:
            log(f'seed{seed}: {tag} FAILED (acc={acc} < {need} or no ckpt); chain stops')
            return
        donor = donor_next
        log(f'seed{seed}: {tag} OK (acc={acc:.4f}, donor={os.path.basename(donor)}), continue')
    log(f'seed{seed}: N5 succeeded; chain complete')


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    log(f'scheduler start; threshold={THRESHOLD} donor_min={DONOR_MIN}')
    threads = [threading.Thread(target=run_chain, args=(s,), daemon=False)
               for s in SEED_GPUS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log('all chains done')


if __name__ == '__main__':
    main()
