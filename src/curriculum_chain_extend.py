"""Extend curriculum chains l2d512r8h4 / l4d512r8h4 from N5 up to N10.

Rules 6-10 continue the (i, 2i-1) pattern: (6,11),(7,13),(8,15),(9,17),(10,19).
Donors are the completed N5 checkpoints of the original forward chains.
Stage N warm-starts from stage N-1's checkpoint (INIT_FROM); success =
best test acc >= 0.95, otherwise the chain stops. Same training regime as
the original chains (see REPRODUCE_l2d512r8h4_chain.md): mixed_ab, P=127,
r8, e0.7, NUM_MASK=2, online missing 0.1/len1, TRAIN_LEN=16.

Logs:   /data/cxm/recursion/mixed_ab/curriculum/curriculum_chains_extend/logs/
Models: /data/cxm/models/curriculum_chains_extend/
State:  /data/cxm/recursion/mixed_ab/curriculum/curriculum_chains_extend/chain_state.json

Usage:
  setsid nohup python src/curriculum_chain_extend.py > .chain_tmp/chain_extend_scheduler.log 2>&1 &
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = '/data/cxm/recursion/mixed_ab/curriculum/curriculum_chains_extend'
LOG_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_DIR = '/data/cxm/models/curriculum_chains_extend'
STATE_PATH = os.path.join(BASE_DIR, 'chain_state.json')

# rule1..rule10; rules 6-10 continue (i, 2i-1).
RULES = [[1, 1], [2, 3], [3, 5], [4, 7], [5, 11],
         [6, 11], [7, 13], [8, 15], [9, 17], [10, 19]]
P = 127
DONOR_DIR = '/data/cxm/models/curriculum_chains'
# scale -> (D_MODEL, N_LAYER, donor ckpt, donor seed, gpu)
SCALES = {
    'l2d512r8h4': dict(d=512, l=2, donor=f'{DONOR_DIR}/curr_l2d512r8h4_N5_seed17996.pth',
                       seed=17996, gpu='0'),
    'l4d512r8h4': dict(d=512, l=4, donor=f'{DONOR_DIR}/curr_l4d512r8h4_N5_seed44536.pth',
                       seed=44536, gpu='1'),
}
N_HEAD, MLP_RATIO = 4, 8
STAGES = [(f'N{n}', RULES[:n]) for n in range(6, 11)]  # N6..N10

THRESHOLD = float(os.environ.get('CURRICULUM_THRESHOLD', '0.95'))

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
    if not os.path.exists(log_path):
        return None
    txt = open(log_path, errors='ignore').read()
    m = re.findall(r'Best test accuracy: ([0-9.]+)%', txt)
    if m:
        return float(m[-1]) / 100
    m = re.findall(r'Best=([0-9.]+)%\(@\d+\)', txt)
    return float(m[-1]) / 100 if m else None


def donor_ckpt(save_path):
    if os.path.exists(save_path):
        return save_path
    best = save_path.replace('.pth', '_best.pth')
    return best if os.path.exists(best) else None


def run_stage(scale, tag, pairs, donor, seed, gpu):
    cfg = SCALES[scale]
    name = f'curr_ext_{scale}_{tag}_seed{seed}'
    save_path = os.path.join(MODEL_DIR, f'{name}.pth')
    base = json.load(open(os.path.join(REPO, 'src', 'config.json')))
    override = {
        'TASK': 'mixed_ab',
        'RANDOM_SEED': seed, 'P': P, 'D_MODEL': cfg['d'],
        'N_LAYER': cfg['l'], 'N_HEAD': N_HEAD, 'MLP_RATIO': MLP_RATIO,
        'AB_PAIRS': pairs,
        'MIXED_AB_MAX_UNIQUE_RATIOS': [0.7] * len(pairs),
        'USE_AB_TAG': False, 'USE_CONDITIONAL_WTE': False, 'NUM_MASK': 2,
        'INIT_FROM': donor, 'SAVE_PATH': save_path, 'MAX_TRAIN_HOURS': 6,
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
    cfg = SCALES[scale]
    gpu, seed = cfg['gpu'], cfg['seed']
    donor = cfg['donor']
    if not os.path.exists(donor):
        log(f'{scale}: donor missing: {donor}; chain aborted')
        return
    with _state_lock:
        _state[scale] = {'donor': donor, 'seed': seed, 'stages': {}}
    save_state()
    for tag, pairs in STAGES:
        ckpt, acc = run_stage(scale, tag, pairs, donor, seed, gpu)
        donor_next = donor_ckpt(ckpt)
        ok = acc is not None and acc >= THRESHOLD and donor_next is not None
        with _state_lock:
            _state[scale]['stages'][tag] = {'ckpt': donor_next, 'acc': acc,
                                            'success': ok}
        save_state()
        if not ok:
            log(f'{scale}: {tag} FAILED (acc={acc} < {THRESHOLD} or no ckpt); chain stops')
            return
        donor = donor_next
        log(f'{scale}: {tag} OK (acc={acc:.4f}), continue')
    log(f'{scale}: N10 succeeded; chain complete')


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
