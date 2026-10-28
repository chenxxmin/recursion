#!/usr/bin/env python3
"""noaux 填空 LR/WD 自动调参调度器（24h 预算，多阶段）。

阶段：
  A = len5_l4     （固定长5 + l4）      —— 起点
  B = len3to5_l4  （不定长3~5 + l4）    —— A/C 成功后进入；成功即全部收工
  C = len5_l6     （固定长5 + l6）      —— A 网格扫完仍无突破时进入
只调 LR/WD；评分轮 1500ep，确认轮 3000ep，确认 6/6 为成功。
机会主义 GPU 调度：每 60s 扫 nvidia-smi，显存 <1GB 且本调度器未占用的卡
立刻塞入下一个待跑实验。决策由 kimi --print --afk 后台会话完成，CLI 失败
时确定性坐标下降兜底。重启不丢在跑实验（按日志+进程名领养）。
"""
import glob, json, os, re, subprocess, sys, time, traceback
from datetime import datetime, timedelta

ROOT = '/home/cxm/recursion'
TUNER = os.path.join(ROOT, 'scripts/noaux_len5_tuner')
STATE_DIR = os.path.join(TUNER, 'state')
STATE_PATH = os.path.join(STATE_DIR, 'state.json')
LOG_PATH = os.path.join(TUNER, 'tuner.log')
DECIDE_PROMPT = os.path.join(TUNER, 'decide_prompt.md')
KIMI = '/home/cxm/miniconda3/bin/kimi'
PYBIN = '/home/cxm/miniconda3/bin/python'
MEM_LIMIT_MB = 1024
POLL_S = 60
CLI_TIMEOUT_S = 900
BUDGET_HOURS = 24
SCORE_EPOCH = 1500
CONFIRM_EPOCH = 3000
LR_GRID = [1e-4, 3e-4, 6e-4, 1e-3, 2e-3, 3e-3]
WD_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
SIZES = {'d512l4': (512, 4), 'd1024l4': (1024, 4),
         'd512l6': (512, 6), 'd1024l6': (1024, 6)}
# EPOCHS 恒为 CONFIRM_EPOCH（T_max 固定），评分轮用 EPOCH_CAP 截断，
# 保证评分轮与确认轮的 LR 逐 epoch 一致。
COEFS = [(25, 56), (62, 59), (56, 93), (5, 3), (67, 65), (44, 59)]
BASE_LR, BASE_WD = 3e-4, 1.0

PHASES = {
    'len5_l4': {
        'n_layer': 4, 'seq_min': None, 'len_tag': 'len5',
        'baseline_dir': 'variant_o2_fillblank_noaux_randpos_d512l4_P127_len5'},
    'len3to5_l4': {
        'n_layer': 4, 'seq_min': (3, 5), 'len_tag': 'len3to5',
        'baseline_dir': 'variant_o2_fillblank_noaux_randpos_d512l4_P127_len3to5'},
    'len5_l6': {
        'n_layer': 6, 'seq_min': None, 'len_tag': 'len5',
        'baseline_dir': 'variant_o2_fillblank_noaux_randpos_d512l6_P127_len5'},
}

sys.path.insert(0, os.path.join(ROOT, 'src'))
import batch_run  # 复用配置合并（不会触发 main）

DRY_RUN = os.environ.get('TUNER_DRY_RUN') == '1'


def log(msg):
    line = f"[{datetime.now().strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def load_state():
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(st):
    tmp = STATE_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE_PATH)


def tag_of(lr, wd, size):
    return f'{size}_lr{lr:g}_wd{wd:g}'


def make_experiments(lr, wd, size, cap, phase):
    ph = PHASES[phase]
    d, l = SIZES[size]
    exps = []
    for a, b in COEFS:
        name = (f'single_o2_fillblank_a{a}b{b}_d{d}l{l}r8h4_P127_'
                f'{ph["len_tag"]}_randpos_noaux_lr{lr:g}_wd{wd:g}_'
                f'cap{cap or CONFIRM_EPOCH}_seed42')
        cfg = {
            'TASK': 'single', 'TASK_VARIANT': 'fill_blank', 'ORDER': 2,
            'P': 127, 'D_MODEL': d, 'N_LAYER': l, 'N_HEAD': 4,
            'MLP_RATIO': 8, 'TRAIN_LEN': 5, 'OOD_LEN': 5,
            'BODY_SUPERVISION': False,
            'EPOCHS': CONFIRM_EPOCH, 'EVAL_INTERVAL': 5,
            'MAX_TRAIN_HOURS': 1.0,
            'SAVE_CHECKPOINTS': False, 'RANDOM_SEED': 42,
            'LR': lr, 'WEIGHT_DECAY': wd, 'RULES': [[a, b]],
        }
        if cap is not None:
            cfg['EPOCH_CAP'] = cap
        if ph['seq_min'] is not None:
            cfg['SEQ_LEN_MIN'], cfg['SEQ_LEN_MAX'] = ph['seq_min']
        exps.append({'name': name, 'task': 'single', 'config': cfg})
    return exps


def gpu_free_list():
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used',
         '--format=csv,noheader,nounits'], text=True)
    free = []
    for line in out.strip().splitlines():
        idx, mem = [x.strip() for x in line.split(',')]
        if int(mem) < MEM_LIMIT_MB:
            free.append(int(idx))
    return free


def exp_alive(name):
    """该实验的 core.py 是否还活着（按临时配置名匹配）。"""
    r = subprocess.run(['pgrep', '-f', f'core.py .*config_tmp_{name}.json'],
                       capture_output=True, text=True)
    return r.returncode == 0


def launch(exp, batch_tag, gpu):
    base_config = batch_run.load_json(batch_run.BASE_CONFIG_PATH)
    work = f'/data/cxm/recursion/{batch_tag}'
    dirs = {'log': f'{work}/logs', 'plot': f'{work}/plots',
            'model': f'/data/cxm/models/{batch_tag}'}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    name, task, merged_main, merged = batch_run.build_merged_config(
        exp, base_config, dirs['model'])
    tmp_cfg = f'{STATE_DIR}/config_tmp_{name}.json'
    batch_run.save_json(tmp_cfg, merged)
    log_path = os.path.join(dirs['log'], f'{name}.log')
    err_path = os.path.join(dirs['log'], f'{name}.err')
    env = {**os.environ, 'PYTHONUNBUFFERED': '1',
           'CUDA_VISIBLE_DEVICES': str(gpu)}
    log_f = open(log_path, 'w', encoding='utf-8')
    log_f.write(f'=== Experiment: {name} ===\nGPU: cuda:{gpu}\n')
    log_f.write(batch_run.format_config_table(merged_main))
    log_f.write('\n--- output ---\n')
    log_f.flush()
    err_f = open(err_path, 'wb')
    proc = subprocess.Popen(
        [PYBIN, os.path.join(ROOT, 'src/core.py'), tmp_cfg],
        stdout=log_f, stderr=err_f, env=env)
    log(f'LAUNCH {name} on cuda:{gpu} (pid {proc.pid})')
    return {'proc': proc, 'name': name, 'gpu': gpu,
            'log': log_path, 'err': err_path,
            'log_f': log_f, 'err_f': err_f, 'tmp_cfg': tmp_cfg}


EPOCH_RE = re.compile(r'Epoch\s+(\d+) \| Train:.*Test: Acc=([\d.]+)%')


def score_one_log(path, score_epoch):
    txt = open(path).read()
    evals = [(int(e), float(a)) for e, a in EPOCH_RE.findall(txt)]
    best = max([a for _, a in evals], default=0.0)
    conv_full = next((e for e, a in evals if a >= 99.0), None)
    conv_ep = conv_full if conv_full is not None and conv_full <= score_epoch else None
    at = [a for e, a in evals if e <= score_epoch]
    acc_at = at[-1] if at else 0.0
    nan = 'nan' in txt.lower()
    return {'best': best, 'conv_epoch': conv_ep, 'conv_epoch_full': conv_full,
            'acc_at': acc_at, 'nan': nan,
            'last_epoch': evals[-1][0] if evals else 0}


def score_candidate(batch_tag, lr, wd, epochs):
    d = f'/data/cxm/recursion/{batch_tag}/logs'
    per = {}
    for a, b in COEFS:
        files = glob.glob(f'{d}/*a{a}b{b}_*.log')
        if not files:
            per[f'a{a}b{b}'] = {'best': 0.0, 'conv_epoch': None,
                                'acc_at': 0.0, 'nan': True}
            continue
        per[f'a{a}b{b}'] = score_one_log(files[0], epochs)
    accs = [v['acc_at'] for v in per.values()]
    nconv = sum(1 for v in per.values() if v['conv_epoch'] is not None)
    nnan = sum(1 for v in per.values() if v['nan'])
    return {'lr': lr, 'wd': wd, 'epochs': epochs, 'per_coef': per,
            'mean_acc': sum(accs) / len(accs), 'n_converged': nconv,
            'n_nan': nnan, 'worst_two': sorted(accs)[:2]}


def seed_baseline(st, phase):
    """以 len3~5 已跑过的 3000ep 原生 T_max=3000 日志为参照系播种。
    该配置（lr=0.001,wd=1,d512l4）是当前已知最优；其 1500ep 处分数与
    新方案（EPOCHS=3000 + EPOCH_CAP=1500）完全同口径。"""
    scores = score_candidate('tune_len3to5_l4_lr0.001_wd1_ep3000',
                             1e-3, 1.0, SCORE_EPOCH)
    st['baseline'] = scores
    st['best'] = {'lr': 1e-3, 'wd': 1.0, 'size': 'd512l4',
                  'mean_acc': scores['mean_acc'],
                  'n_converged': scores['n_converged']}
    st['history'] = [{'action': 'baseline', 'phase': phase,
                      **{k: v for k, v in scores.items() if k != 'per_coef'}}]
    st['tried'] = [tag_of(1e-3, 1.0, 'd512l4')]
    st['last_move'] = None
    log(f"baseline[{phase}]: mean@1500={scores['mean_acc']:.1f} "
        f"nconv={scores['n_converged']}/6 (from 3000ep log)")
    return st


def better(new, best):
    if new['n_converged'] != best['n_converged']:
        return new['n_converged'] > best['n_converged']
    return new['mean_acc'] > best['mean_acc'] + 1e-9


def fallback_next(st):
    tried = set(st['tried'])
    best = st['best']
    lr, wd, size = best['lr'], best['wd'], best.get('size', 'd512l4')
    cands = []
    lm = st.get('last_move')
    if lm:
        axis, direction = lm['axis'], lm['direction']
        grid = WD_GRID if axis == 'wd' else LR_GRID
        cur = wd if axis == 'wd' else lr
        if cur in grid:
            i = grid.index(cur) + direction
            if 0 <= i < len(grid):
                nwd = grid[i] if axis == 'wd' else wd
                nlr = grid[i] if axis == 'lr' else lr
                cands.append((nlr, nwd, size, axis, direction))
    for w in [1.5, 2.0, 0.75, 3.0, 0.5, 4.0]:
        cands.append((lr, w, size, 'wd', 1 if w > wd else -1))
    for r in [2e-3, 6e-4, 3e-4, 1e-3, 3e-3, 1e-4]:
        cands.append((r, wd, size, 'lr', 1 if r > lr else -1))
    for s in SIZES:  # 尺寸探索放在 LR/WD 之后
        cands.append((lr, wd, s, 'size', 0))
    for nlr, nwd, nsize, axis, direction in cands:
        if tag_of(nlr, nwd, nsize) not in tried:
            return {'action': 'score', 'lr': nlr, 'wd': nwd, 'size': nsize,
                    'move': {'axis': axis, 'direction': direction},
                    'reason': 'fallback coordinate descent'}
    return {'action': 'stop_giveup', 'reason': 'fallback grid exhausted'}


def cli_next(st, last_scores):
    next_path = os.path.join(STATE_DIR, 'next.json')
    if os.path.exists(next_path):
        os.remove(next_path)
    prompt = open(DECIDE_PROMPT).read()
    summary = {
        'phase': st['phase'], 'best': st['best'], 'tried': sorted(st['tried']),
        'last_scores': last_scores, 'history': st['history'][-8:],
        'phases_done': st.get('phases_done', {}), 'deadline': st['deadline'],
    }
    full = prompt + '\n\n当前状态（JSON）：\n' + json.dumps(summary, indent=1)
    try:
        r = subprocess.run(
            [KIMI, '--print', '--afk', '-w', ROOT, '-p', full],
            capture_output=True, text=True, timeout=CLI_TIMEOUT_S)
        with open(os.path.join(TUNER, 'last_decision_cli.log'), 'w') as f:
            f.write(r.stdout + '\n--- STDERR ---\n' + r.stderr)
        if not os.path.exists(next_path):
            log(f'CLI decided nothing (rc={r.returncode}), see last_decision_cli.log')
            return None
        nxt = json.load(open(next_path))
        if nxt.get('action') not in ('score', 'confirm', 'stop_success', 'stop_giveup'):
            raise ValueError(f"bad action {nxt.get('action')}")
        if nxt['action'] in ('score', 'confirm'):
            if nxt['lr'] not in LR_GRID or nxt['wd'] not in WD_GRID:
                raise ValueError(f"off-grid lr/wd: {nxt}")
            if nxt.get('size', 'd512l4') not in SIZES:
                raise ValueError(f"bad size: {nxt}")
            if nxt['action'] == 'score' and tag_of(nxt['lr'], nxt['wd'], nxt.get('size', 'd512l4')) in st['tried']:
                raise ValueError(f"already tried: {nxt}")
        return nxt
    except Exception as e:
        log(f'CLI decision failed: {e}')
        return None


def write_scores(tag, scores):
    with open(os.path.join(STATE_DIR, 'scores', f'{tag}.json'), 'w') as f:
        json.dump(scores, f, indent=1)


def archive_phase(st):
    st.setdefault('phases_done', {})[st['phase']] = {
        'best': st['best'],
        'baseline': {k: v for k, v in st.get('baseline', {}).items()
                     if k != 'per_coef'},
        'history': st['history']}


def summarize_and_exit(st, reason):
    archive_phase(st)
    save_state(st)
    report = {'reason': reason, 'phases_done': st['phases_done']}
    with open(os.path.join(TUNER, 'final_report.json'), 'w') as f:
        json.dump(report, f, indent=1)
    log(f'EXIT: {reason}')
    sys.exit(0)


def transition(st, event):
    """event: 'success'（本阶段确认6/6）或 'giveup'（网格扫完）。"""
    phase = st['phase']
    archive_phase(st)
    log(f'PHASE {phase} {event}; best={st["best"]}')
    if event == 'success':
        if phase == 'len3to5_l4':
            summarize_and_exit(st, 'SUCCESS_ALL: len3to5 也 6/6')
        nxt_phase = 'len3to5_l4'
    else:
        if phase != 'len5_l4':
            summarize_and_exit(st, f'GIVEUP at {phase}')
        nxt_phase = 'len5_l6'
    st['phase'] = nxt_phase
    st['current'] = None
    st = seed_baseline(st, nxt_phase)
    save_state(st)
    log(f'PHASE -> {nxt_phase}')
    return st


def new_candidate(lr, wd, size, score_epoch, action, phase, tag=None):
    cap = score_epoch if score_epoch == SCORE_EPOCH else None
    if tag is None:
        tag = f'tune_{phase}_{tag_of(lr, wd, size)}_cap{cap or CONFIRM_EPOCH}'
    return {'lr': lr, 'wd': wd, 'size': size, 'epochs': score_epoch,
            'cap': cap, 'action': action, 'tag': tag, 'phase': phase,
            'pending': make_experiments(lr, wd, size, cap, phase),
            'running': []}


def adopt_current(st):
    """重启后按日志+进程存活状态领养 current 候选，不重复启动。"""
    cur = new_candidate(st['current']['lr'], st['current']['wd'],
                        st['current']['size'], st['current']['epochs'],
                        st['current']['action'],
                        st['current']['phase'], tag=st['current']['tag'])
    d = f"/data/cxm/recursion/{cur['tag']}/logs"
    for exp in list(cur['pending']):
        name = exp['name']
        log_path = os.path.join(d, f'{name}.log')
        if not os.path.exists(log_path):
            continue  # 仍 pending
        cur['pending'].remove(exp)
        if exp_alive(name):
            cur['running'].append({'proc': None, 'name': name,
                                   'gpu': None, 'log': log_path,
                                   'adopted': True})
            log(f'ADOPT running {name}')
        else:
            log(f'ADOPT finished {name}')
    return cur


def main():
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(STATE_PATH):
        st = load_state()
    else:
        st = {'started': datetime.now().isoformat(),
              'deadline': (datetime.now() + timedelta(hours=BUDGET_HOURS)).isoformat(),
              'phase': 'len3to5_l4', 'phases_done': {}, 'current': None}
        st = seed_baseline(st, 'len3to5_l4')
        save_state(st)
    deadline = datetime.fromisoformat(st['deadline'])
    log(f"tuner start; phase={st['phase']}; deadline={st['deadline']}")
    if DRY_RUN:
        print(json.dumps(st['baseline'], indent=1)[:1500])
        return

    current = adopt_current(st) if st.get('current') else None
    while True:
        try:
            now = datetime.now()
            # ---------- 收尾当前候选 ----------
            if current and not current['pending'] and not current['running']:
                sc = score_candidate(current['tag'], current['lr'],
                                     current['wd'], current['epochs'])
                write_scores(current['tag'], sc)
                st['history'].append({'action': current['action'],
                                      'phase': current['phase'],
                                      **{k: v for k, v in sc.items()
                                         if k != 'per_coef'}})
                log(f"SCORED {current['tag']}: mean@{current['epochs']}="
                    f"{sc['mean_acc']:.1f} nconv={sc['n_converged']}/6 "
                    f"worst2={sc['worst_two']}")
                st['current'] = None
                if current['action'] == 'confirm':
                    if sc['n_converged'] == 6:
                        st['best'] = {'lr': current['lr'], 'wd': current['wd'],
                                      'size': current['size'],
                                      'mean_acc': sc['mean_acc'],
                                      'n_converged': 6}
                        save_state(st)
                        st = transition(st, 'success')
                        current = None
                        continue
                    nxt = cli_next(st, sc) or fallback_next(st)
                else:
                    st['tried'].append(tag_of(current['lr'], current['wd'],
                                              current['size']))
                    if sc['n_converged'] == 6:
                        log('6/6 at scoring round -> auto confirm 3000ep')
                        nxt = {'action': 'confirm', 'lr': current['lr'],
                               'wd': current['wd'], 'size': current['size'],
                               'reason': 'auto'}
                    else:
                        if better(sc, st['best']):
                            st['best'] = {'lr': current['lr'], 'wd': current['wd'],
                                          'size': current['size'],
                                          'mean_acc': sc['mean_acc'],
                                          'n_converged': sc['n_converged']}
                        nxt = cli_next(st, sc) or fallback_next(st)
                st['last_move'] = nxt.get('move')
                save_state(st)
                if nxt['action'] == 'stop_success':
                    st = transition(st, 'success')
                    current = None
                    continue
                if nxt['action'] == 'stop_giveup':
                    st = transition(st, 'giveup')
                    current = None
                    continue
                ep = CONFIRM_EPOCH if nxt['action'] == 'confirm' else SCORE_EPOCH
                size = nxt.get('size', st['best'].get('size', 'd512l4'))
                current = new_candidate(nxt['lr'], nxt['wd'], size, ep,
                                        nxt['action'], st['phase'])
                st['current'] = {k: current[k] for k in
                                 ('lr', 'wd', 'size', 'epochs', 'cap',
                                  'action', 'tag', 'phase')}
                save_state(st)
                log(f"NEXT {nxt['action']}: lr={nxt['lr']} wd={nxt['wd']} "
                    f"size={size} cap={current['cap'] or CONFIRM_EPOCH} "
                    f"reason={nxt.get('reason','')}")

            # ---------- 起新候选 ----------
            if current is None:
                if now >= deadline:
                    summarize_and_exit(st, '24h deadline reached')
                nxt = cli_next(st, None) or fallback_next(st)
                if nxt['action'] == 'stop_success':
                    st = transition(st, 'success')
                    continue
                if nxt['action'] == 'stop_giveup':
                    st = transition(st, 'giveup')
                    continue
                ep = CONFIRM_EPOCH if nxt['action'] == 'confirm' else SCORE_EPOCH
                size = nxt.get('size', st['best'].get('size', 'd512l4'))
                current = new_candidate(nxt['lr'], nxt['wd'], size, ep,
                                        nxt['action'], st['phase'])
                st['current'] = {k: current[k] for k in
                                 ('lr', 'wd', 'size', 'epochs', 'cap',
                                  'action', 'tag', 'phase')}
                save_state(st)
                log(f"START {nxt['action']}: lr={nxt['lr']} wd={nxt['wd']} "
                    f"size={size} cap={current['cap'] or CONFIRM_EPOCH}")

            # ---------- 回收已完成的实验 ----------
            for r in list(current['running']):
                if r.get('adopted'):
                    finished = not exp_alive(r['name'])
                    rc = 'adopted'
                else:
                    rc = r['proc'].poll()
                    finished = rc is not None
                if not finished:
                    continue
                if not r.get('adopted'):
                    r['log_f'].close()
                    r['err_f'].close()
                    with open(r['log'], 'a') as f:
                        f.write(f'\n--- Return code: {rc} ---\n')
                    if os.path.exists(r['tmp_cfg']):
                        os.remove(r['tmp_cfg'])
                log(f"FINISH {r['name']} rc={rc} cuda:{r['gpu']}")
                current['running'].remove(r)

            # ---------- 机会主义发卡 ----------
            if current['pending']:
                used = {r['gpu'] for r in current['running'] if r['gpu'] is not None}
                for gpu in gpu_free_list():
                    if gpu in used or not current['pending']:
                        continue
                    exp = current['pending'].pop(0)
                    r = launch(exp, current['tag'], gpu)
                    current['running'].append(r)
                    used.add(gpu)

            time.sleep(POLL_S)
        except SystemExit:
            raise
        except Exception:
            log('LOOP ERROR:\n' + traceback.format_exc())
            time.sleep(POLL_S)


if __name__ == '__main__':
    main()
