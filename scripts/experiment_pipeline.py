#!/usr/bin/env python3
"""Review effective experiment parameters, then build an approved portable bundle."""
import argparse
import ast
import base64
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import socket
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[1]
SINGLE = {'addition': 2, 'multiplication': 2, 'nonlinear': 2,
          'nonlinear_mul': 2, 'tribonacci': 3, 'tetranacci': 4}
TASKS = set(SINGLE) | {'mixed_ab', 'mixed_abc', 'action', 'action_trib'}
FALLBACKS = dict(NUM_TRAIN_SAMPLES=None, NUM_TEST_SAMPLES=256,
                 MISSING_PROB=0.0, MISS_LEN=1, MISS_SECOND=False,
                 PREDICT_MISSING=False, GRAD_ACCUM_STEPS=1,
                 SKIP_FINAL_GENERATION_TEST=False)
SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,159}\Z')
SNAPSHOT = re.compile(r'<!-- PIPELINE_SNAPSHOT ([A-Za-z0-9+/=]+) -->')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(repo, delivery=None):
    paths = sorted((repo / 'src').glob('*.py')) + [
        repo / 'src/config.json', repo / 'requirements.txt',
        repo / 'scripts/experiment_package_runner.py']
    if delivery:
        paths += [repo / 'scripts/remote_experiment_runner.py',
                  repo / 'scripts/action_run_postprocess.py']
    return {str(path.relative_to(repo)): sha_file(path) for path in paths}


def known_keys(repo, base):
    keys = set(FALLBACKS) | {'TASK', 'SAVE_PATH'}
    for section in base.values():
        if isinstance(section, dict):
            keys.update(section)
    # Recognize keys actually consumed by training/model code; reject typos.
    for path in (repo / 'src').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if re.fullmatch(r'[A-Z][A-Z_0-9]*', value):
                    keys.add(value)
    return keys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_int(value, name):
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
            name + ' must be a positive integer')


def normalize(task, cfg, origins):
    def derived(key, value):
        if cfg.get(key) != value:
            origins[key] = '代码语义推导；原值=' + canonical(cfg.get(key))
        cfg[key] = value

    for key in ('P', 'D_MODEL', 'N_HEAD', 'N_LAYER', 'MLP_RATIO', 'BATCH_SIZE',
                'EPOCHS', 'TRAIN_LEN', 'OOD_LEN', 'EVAL_INTERVAL', 'GRAD_ACCUM_STEPS'):
        positive_int(cfg[key], key)
    require(cfg['P'] >= 2, 'P must be >=2')
    require(cfg['D_MODEL'] % cfg['N_HEAD'] == 0, 'D_MODEL must be divisible by N_HEAD')
    require(isinstance(cfg['RANDOM_SEED'], int), 'RANDOM_SEED must be an integer')
    for key in ('LR', 'WEIGHT_DECAY', 'GRAD_CLIP_NORM', 'MAX_TRAIN_HOURS'):
        value = cfg[key]
        if value is None and key in ('GRAD_CLIP_NORM', 'MAX_TRAIN_HOURS'):
            continue
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0, key + ' must be finite and nonnegative')
    require(cfg['LR'] > 0, 'LR must be positive')
    require(cfg['GRAD_CLIP_NORM'] is None or cfg['GRAD_CLIP_NORM'] > 0,
            'Use GRAD_CLIP_NORM=null for no clipping')
    require(not (cfg.get('INIT_FROM') and cfg.get('RESUME_FROM')),
            'INIT_FROM and RESUME_FROM cannot both be set')
    notes = []
    positive_int(cfg['NUM_TEST_SAMPLES'], 'NUM_TEST_SAMPLES')
    positive_int(cfg['MISS_LEN'], 'MISS_LEN')
    require(isinstance(cfg['MISSING_PROB'], (int, float)) and 0 <= cfg['MISSING_PROB'] <= 1,
            'MISSING_PROB must be in [0, 1]')
    if task in SINGLE:
        order = SINGLE[task]
        mode = cfg['DATA_MODE']
        if mode is None:
            mode = 'sampled_fresh_test' if cfg['FRESH_TEST_PER_EVAL'] or cfg['STATE_SPACE_CAP'] is not None else 'full_split'
        require(mode in ('sampled_fresh_test', 'full_split'), 'Invalid DATA_MODE')
        derived('DATA_MODE', mode)
        derived('FRESH_TEST_PER_EVAL', mode == 'sampled_fresh_test')
        if cfg['NUM_TRAIN_SAMPLES'] is None:
            require(mode == 'full_split', 'sampled_fresh_test requires NUM_TRAIN_SAMPLES')
            derived('NUM_TRAIN_SAMPLES', max(1, int(cfg['P'] ** order * cfg['MAX_UNIQUE_RATIO'])))
        positive_int(cfg['NUM_TRAIN_SAMPLES'], 'NUM_TRAIN_SAMPLES')
        require(cfg['NUM_TRAIN_SAMPLES'] <= cfg['P'] ** order, 'Training count exceeds state space')
        if cfg['NUM_MASK'] is None:
            derived('NUM_MASK', order - 1)
        count = cfg['NUM_TRAIN_SAMPLES']
        rules = [cfg[k] for k in 'ABCD'[:order]] if task in ('addition', 'tribonacci', 'tetranacci') else task
        if isinstance(rules, list):
            require(all(isinstance(c, int) and not isinstance(c, bool) for c in rules),
                    'Recurrence coefficients must be integers')
        convention = '第一个系数乘最新值' if isinstance(rules, list) else '由 task 固定的非线性公式；A/B 不参与'
        test_count = cfg['NUM_TEST_SAMPLES'] if cfg['FRESH_TEST_PER_EVAL'] else cfg['P'] ** order - count
        exposure = count / cfg['P'] ** order
        data_mode = mode
        train_tokens = cfg['TRAIN_LEN']
    elif task.startswith('mixed_'):
        order = 3 if task == 'mixed_abc' else 2
        key = 'ABC_PAIRS' if order == 3 else 'AB_PAIRS'
        rules = cfg.get(key)
        require(isinstance(rules, list) and rules, key + ' must contain rules')
        require(all(isinstance(p, list) and len(p) == order and
                    all(isinstance(c, int) and 0 <= c < cfg['P'] for c in p) for p in rules),
                'Invalid recurrence coefficients')
        require(len({tuple(p) for p in rules}) == len(rules), 'Duplicate mixed rules')
        cfg.setdefault('USE_AB_TAG', True)
        origins.setdefault('USE_AB_TAG', '代码默认')
        if cfg['NUM_MASK'] is None:
            derived('NUM_MASK', order if cfg['USE_AB_TAG'] else order - 1)
        counts = cfg['NUM_TRAIN_SAMPLES']
        if counts is None:
            ratios = cfg.get('MIXED_AB_MAX_UNIQUE_RATIOS', cfg['MAX_UNIQUE_RATIO'])
            ratios = [ratios] * len(rules) if isinstance(ratios, (int, float)) else ratios
            require(len(ratios) == len(rules), 'Exposure ratios must match rule count')
            counts = [max(1, int(cfg['P'] ** order * r)) for r in ratios]
        elif isinstance(counts, int):
            counts = [counts] * len(rules)
        require(isinstance(counts, list) and len(counts) == len(rules), 'Training counts must match rule count')
        for count_value in counts:
            positive_int(count_value, 'NUM_TRAIN_SAMPLES per rule')
            require(count_value <= cfg['P'] ** order, 'Training count per rule exceeds state space')
        derived('NUM_TRAIN_SAMPLES', counts)
        count = {'per_rule': counts, 'total': sum(counts)}
        per_rule_test = ([min(cfg['NUM_TEST_SAMPLES'], cfg['P'] ** order)] * len(rules)
                         if cfg['FRESH_TEST_PER_EVAL'] else [cfg['P'] ** order - n for n in counts])
        test_count = {'per_rule': per_rule_test, 'total': sum(per_rule_test)}
        exposure = [n / cfg['P'] ** order for n in counts]
        data_mode = 'full_split + fresh_test' if cfg['FRESH_TEST_PER_EVAL'] else 'full_split'
        train_tokens = cfg['TRAIN_LEN'] + int(cfg['USE_AB_TAG'])
        notes.append('mixed 的 NUM_TRAIN_SAMPLES 标量表示每规则数量；表中展开逐规则和总量。DATA_MODE/STATE_SPACE_CAP 不控制 mixed 数据生成。')
        convention = '第一个系数乘最新值'
    else:
        order = 3 if task == 'action_trib' else 2
        key = 'ABC_PAIRS' if order == 3 else 'AB_PAIRS'
        rules = cfg.get(key)
        require(isinstance(rules, list) and rules and all(
            isinstance(p, list) and len(p) == order and all(isinstance(c, int) and 0 <= c < cfg['P'] for c in p)
            for p in rules), 'Invalid action rules')
        positive_int(cfg['NUM_TRAIN_SAMPLES'], 'NUM_TRAIN_SAMPLES')
        count = cfg['NUM_TRAIN_SAMPLES']
        notes.append('action 使用数据集的逐位置 loss_mask；NUM_MASK 配置不参与 loss。')
        require(not cfg['MISS_SECOND'], 'MISS_SECOND is unsupported for action tasks')
        convention = '第一个系数乘最老值；flag 在每一步选择规则'
        test_count = cfg['NUM_TEST_SAMPLES']
        exposure = 'action 动态生成；规则逐步随机选择，无固定每规则样本配额'
        data_mode = 'generated + fresh_test' if cfg['FRESH_TEST_PER_EVAL'] else 'generated + fixed_test'
        train_tokens = 2 * cfg['TRAIN_LEN'] - order
    require(cfg['TRAIN_LEN'] > order and cfg['OOD_LEN'] > order, 'Sequence lengths must exceed recurrence order')
    if not task.startswith('action'):
        require(isinstance(cfg['NUM_MASK'], int) and 0 <= cfg['NUM_MASK'] < cfg['TRAIN_LEN'] - 1,
                'NUM_MASK leaves no evaluated positions or is invalid')
    if cfg['MAX_TRAIN_HOURS'] is not None:
        notes.append('每条实验独立计时，从训练循环开始；每 ' + str(cfg['EVAL_INTERVAL']) +
                     ' epoch 的评估点检查，到时保存完整存档退出，可能超过名义时限；不改变 cosine horizon。')
    if cfg['USE_AMP']:
        notes.append('精度为 AMP/' + cfg['AMP_DTYPE'] + '；需在确认表中明确接受该精度。')
    eval_len = cfg['OOD_LEN'] if cfg['FRESH_TEST_PER_EVAL'] or task.startswith('action') else cfg['TRAIN_LEN']
    # The current mixed fresh-test closure returns untagged windows, even
    # when training prepends a rule tag. Report the actual code behavior.
    eval_tag = task.startswith('mixed_') and cfg.get('USE_AB_TAG', False) and not cfg['FRESH_TEST_PER_EVAL']
    eval_tokens = 2 * eval_len - order if task.startswith('action') else eval_len + int(eval_tag)
    if task.startswith('mixed_') and cfg.get('USE_AB_TAG', False) and cfg['FRESH_TEST_PER_EVAL']:
        notes.append('当前代码的 mixed fresh test 未前置 rule tag，训练与测试 token 布局不一致；正式采用该组合前应先修复代码并重新审阅。')
    max_tokens = (2 * max(cfg['TRAIN_LEN'], cfg['OOD_LEN']) - order if task.startswith('action') else
                  max(cfg['TRAIN_LEN'], cfg['OOD_LEN']) + int(task.startswith('mixed_') and cfg.get('USE_AB_TAG', False)))
    notes.append('静态测试在达到 EARLY_STOP_ACCURACY 时停止；fresh test 在最近 10 次评估均值超过阈值时停止。EARLY_STOP_EXTRA_EPOCHS 已废弃，不生效。')
    notes.append('正常完成后沿用代码中的 final generation（除非 SKIP_FINAL_GENERATION_TEST=true）和 attention 分析；存档退出码 42 跳过这些后处理。定时约束训练循环。')
    return dict(actual_train_samples=count, actual_test_samples=test_count,
                recurrence_order=order, effective_data_mode=data_mode, train_exposure=exposure,
                train_sequence_tokens=train_tokens, eval_base_length=eval_len,
                eval_sequence_tokens=eval_tokens, block_size=2 ** (max_tokens - 1).bit_length(),
                rules=rules, coefficient_convention=convention,
                actual_num_mask='数据集 loss_mask' if task.startswith('action') else cfg['NUM_MASK'],
                initialization='resume' if cfg.get('RESUME_FROM') else 'weights_only' if cfg.get('INIT_FROM') else 'fresh',
                optimizer='AdamW', scheduler='CosineAnnealingLR', scheduler_T_max=cfg['EPOCHS'],
                output_path='<output>/models/<experiment>/<experiment>.pth', notes=notes)


def inspect_resume(path, cfg):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    required = {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                'next_epoch', 'rng_python', 'rng_torch', 'best_acc', 'best_epoch',
                'no_improve', 'high_acc_epoch'}
    require(required <= state.keys(), 'RESUME_FROM must be a full training checkpoint')
    scheduler = state['scheduler_state_dict']
    groups = state['optimizer_state_dict']['param_groups']
    require(scheduler.get('T_max') == cfg['EPOCHS'], 'Checkpoint T_max overrides EPOCHS; resolve before approval')
    require(all(math.isclose(lr, cfg['LR']) for lr in scheduler['base_lrs']),
            'Checkpoint overrides LR; use fork_training_lr.py first, then review again')
    require(all(math.isclose(group['weight_decay'], cfg['WEIGHT_DECAY']) for group in groups),
            'Checkpoint overrides WEIGHT_DECAY; resolve before approval')
    require(state['next_epoch'] < cfg['EPOCHS'], 'Checkpoint has no remaining epochs')
    for key in ('P', 'D_MODEL', 'N_HEAD', 'N_LAYER', 'MLP_RATIO'):
        saved = state.get('config', {}).get(key.lower())
        require(saved is None or saved == cfg[key], 'Checkpoint model mismatch: ' + key)
    return dict(next_epoch=state['next_epoch'], current_lr=[g['lr'] for g in groups],
                base_lrs=scheduler['base_lrs'], T_max=scheduler['T_max'])


def prepare(request, repo=REPO):
    repo = Path(repo).resolve()
    require(set(request) <= {'batch_name', 'concurrency', 'common', 'experiments', 'unresolved', 'delivery'},
            'Unknown request field')
    name = request['batch_name']
    require(bool(SAFE_NAME.fullmatch(name)), 'Unsafe batch_name')
    experiments = request['experiments']
    require(isinstance(experiments, list) and experiments, 'No experiments')
    concurrency = request.get('concurrency', 1)
    positive_int(concurrency, 'concurrency')
    delivery = copy.deepcopy(request.get('delivery'))
    if delivery is not None:
        require(set(delivery) == {'required_gpus', 'output_root', 'clean_eval_samples', 'clean_eval_seed'},
                'Remote delivery requires exact GPU/output/evaluation settings')
        positive_int(delivery['required_gpus'], 'required_gpus')
        positive_int(delivery['clean_eval_samples'], 'clean_eval_samples')
        require(delivery['required_gpus'] == concurrency, 'Required GPUs must equal concurrency')
        require(Path(delivery['output_root']).is_absolute(), 'output_root must be absolute')
        require(isinstance(delivery['clean_eval_seed'], int), 'clean_eval_seed must be an integer')
        require(all(e['task'] == 'action' for e in experiments), 'Remote clean evaluation supports action')
    base = json.loads((repo / 'src/config.json').read_text())
    accepted = known_keys(repo, base)
    assets = {}
    resolved = []
    used_names = set()
    for item in experiments:
        require(set(item) <= {'name', 'task', 'reference', 'config'}, 'Unknown experiment field')
        name = item['name']
        task = item['task']
        require(bool(SAFE_NAME.fullmatch(name)) and name not in used_names, 'Unsafe/duplicate experiment name')
        require(task in TASKS, 'Unsupported task: ' + task)
        used_names.add(name)
        reference = {}
        reference_source = ''
        if item.get('reference'):
            ref = item['reference']
            reference_path = (repo / ref['path']).resolve()
            data = json.loads(reference_path.read_text())
            entries = data['experiments'] if isinstance(data, dict) else data
            selected = [e for e in entries if e['name'] == ref.get('name')] if ref.get('name') else entries
            require(len(selected) == 1, 'Reference must identify exactly one experiment')
            require(selected[0]['task'] == task, 'Reference task mismatch')
            reference = selected[0]['config']
            reference_source = '参考 ' + str(ref['path']) + ' / ' + selected[0]['name']
        cfg = {}
        origins = {}
        for values, source in [(FALLBACKS, '代码默认'), (base['main'], 'main 默认'),
                               (base.get(task, {}), task + ' 默认'), (reference, reference_source),
                               (request.get('common', {}), '用户公共参数'), (item.get('config', {}), '用户逐实验参数')]:
            require(isinstance(values, dict), 'Config must be an object')
            require(not (set(values) - accepted), 'Unknown config keys: ' + ', '.join(sorted(set(values) - accepted)))
            if source.startswith('用户'):
                require('SAVE_PATH' not in values, 'SAVE_PATH is generated under target output directory')
                require('TASK' not in values or values['TASK'] == task, 'TASK conflict')
            cfg.update(copy.deepcopy(values))
            origins.update({k: source for k in values})
        cfg.pop('SAVE_PATH', None)
        origins.pop('SAVE_PATH', None)
        cfg['TASK'] = task
        origins['TASK'] = '实验任务'
        effective = normalize(task, cfg, origins)
        if delivery:
            effective['output_path'] = delivery['output_root'] + '/<experiment>__<timestamp>/checkpoints/model.pth'
            effective['notes'] = [n for n in effective['notes'] if 'attention' not in n]
            effective['notes'].append('训练结束后保存完整 resume，执行同长度 clean 评估并导出曲线；评估使用原 GPU，不计入训练时限。')
        for key in ('INIT_FROM', 'RESUME_FROM'):
            if not cfg.get(key):
                continue
            path = (repo / cfg[key]).resolve()
            require(path.is_file(), key + ' checkpoint is missing')
            if key == 'RESUME_FROM':
                effective['resume_state'] = inspect_resume(path, cfg)
            checksum = sha_file(path)
            destination = 'checkpoints/' + checksum[:16] + '.pth'
            assets[destination] = dict(source=str(path), sha256=checksum, bytes=path.stat().st_size)
            effective[key + '_source'] = str(path)
            cfg[key] = destination
        resolved.append(dict(name=name, task=task, config=cfg, origins=origins, effective=effective))
    plan = dict(version=1, batch_name=request['batch_name'], concurrency=concurrency,
                experiments=resolved, assets=assets, source_files=source_files(repo, delivery),
                packaging_host=socket.gethostname(), local_gpu_allowlist=[0, 1, 2, 3],
                unresolved=request.get('unresolved', []))
    if delivery:
        plan['delivery'] = delivery
    return plan


def review_id(plan):
    return hashlib.sha256(canonical(plan).encode()).hexdigest()[:20]


def cell(value):
    return canonical(value).replace('|', r'\|').replace('\n', '<br>')


def render_review(plan, snapshot=True):
    rid = review_id(plan)
    lines = ['# 实验参数确认表', '', '审阅版本：' + rid,
             '批次：' + plan['batch_name'] + '；最大并发：' + str(plan['concurrency']),
             '此阶段只审阅参数，尚未生成可执行实验 JSON 或启动实验。', '',
             '| 实验 | 任务 | 实际训练样本 | Seed | LR / WD / clip | 初始化 | 每条定时(h) |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for e in plan['experiments']:
        c = e['config']
        lines.append('| ' + ' | '.join([e['name'], e['task'], cell(e['effective']['actual_train_samples']),
                     str(c['RANDOM_SEED']), f"{c['LR']} / {c['WEIGHT_DECAY']} / {c['GRAD_CLIP_NORM']}",
                     e['effective']['initialization'], '不限' if c['MAX_TRAIN_HOURS'] is None else str(c['MAX_TRAIN_HOURS'])]) + ' |')
    if plan.get('delivery'):
        lines += ['', '跨机运行规则：启动前必须取得全部预设空闲 GPU，否则整批退出；启动后 FIFO 接续。',
                  '每个实验在 output_root 下单独建目录，包含 configs/logs/checkpoints/plots/metrics。',
                  '每组独立计时；后台运行继承 GPU 运行锁；resume 保持优化器与 scheduler 状态。',
                  '', '| 运行参数 | 确认值 |', '| --- | --- |']
        for key, value in sorted(plan['delivery'].items()):
            lines.append(f'| {key} | {cell(value)} |')
    for e in plan['experiments']:
        lines += ['', '## ' + e['name'], '', '| 参数 | 实际值 | 来源 |', '| --- | --- | --- |']
        for key in sorted(e['config']):
            lines.append(f"| {key} | {cell(e['config'][key])} | {e['origins'].get(key, '代码默认')} |")
        lines += ['', '| 推导项 | 实际值 |', '| --- | --- |']
        for key, value in sorted(e['effective'].items()):
            if key != 'notes':
                lines.append(f'| {key} | {cell(value)} |')
        lines += ['', *['- ' + note for note in e['effective']['notes']]]
    lines += ['', '打包内容：已确认 JSON、定时写入 run.sh、训练代码快照、依赖清单、检查和运行工具。']
    if plan['assets']:
        lines.append('附带存档总大小：' + str(sum(a['bytes'] for a in plan['assets'].values())) + ' bytes。')
    if plan['unresolved']:
        lines += ['', '待澄清（未解决前禁止打包）：', *['- ' + str(x) for x in plan['unresolved']]]
    lines += ['', '请确认此版本；修改参数或训练代码后需重新出表。']
    if snapshot:
        lines += ['', '<!-- PIPELINE_SNAPSHOT ' + base64.b64encode(canonical(plan).encode()).decode() + ' -->']
    return '\n'.join(lines) + '\n'


def load_review(path):
    text = Path(path).read_text()
    match = SNAPSHOT.search(text)
    require(match is not None, 'Review snapshot is missing')
    plan = json.loads(base64.b64decode(match.group(1)))
    require(text == render_review(plan), 'Review table was edited; regenerate it before confirmation')
    return plan


def run_script(timers):
    script = '''#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "@DOLLAR@{BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="@DOLLAR@{EXPERIMENT_PYTHON:-python3}"
# 每个实验单独计时，单位小时；null=不限。达到时限后在 eval 点保存退出。
# 定时由训练循环 MAX_TRAIN_HOURS 执行，不改变 EPOCHS，不使用强杀。
TIMERS_JSON=@TIMERS@
exec "$PYTHON_BIN" "$BUNDLE_DIR/runner.py" --bundle "$BUNDLE_DIR" --timers "$TIMERS_JSON" "$@"
'''
    return script.replace('@DOLLAR@', '$').replace('@TIMERS@', shlex.quote(canonical(timers)))



def remote_readme(plan):
    d = plan['delivery']
    return f"""# Action online 实验运行包

{len(plan['experiments'])} 组；{d['required_gpus']} 卡并行；每组独立定时 6h。
完整参数见 PARAMETERS.md；JSON、脚本和 manifest 的定时必须一致。

环境：Linux、Python 3.10+、匹配机器 CUDA 的 PyTorch、numpy、matplotlib。
可用 EXPERIMENT_PYTHON 指定解释器。训练代码已随包冻结，不依赖机器上的旧版本。

```bash
bash run.sh check
# 自动从可见且获准的 GPU 中选择 3 张空闲卡，后台运行
bash run.sh run --background
# 也可指定物理 GPU 候选池；仍必须至少有 3 张空闲卡
bash run.sh run --gpus 0,2,4,6 --background
```

启动检查：显存 <100 MiB、利用率=0、没有 compute 进程且能取得运行锁。
先比较空闲数与预设卡数，数量不足则整批退出，不启动任何训练，也不自动等待。
检查和取得全部 GPU 锁在后台分离前完成；终端打印实际卡号、型号和首批任务表。
后续任务按 JSON 顺序接续空出的卡，训练失败不自动重试。
继承 CUDA_VISIBLE_DEVICES 限制；在打包本机上还严格限制为 GPU 0–3。

默认根目录：{d['output_root']}（可用 --output-root 指定其他机器上的路径）。
每个实验独立建立 <实验名>__<UTC时间>_<唯一编号>/，内部按 configs、logs、
checkpoints、plots、metrics 分类。批次控制目录也在同一根目录内，
包含 configs、logs、state；启动时打印其绝对路径。再次运行创建新目录，不覆盖旧结果。

```bash
bash run.sh status --batch /data/cxm/final/<启动时打印的批次目录>
bash run.sh stop --batch /data/cxm/final/<启动时打印的批次目录>
# 只续跑有完整存档、尚未正常完成的任务；每条再计时 6h
bash run.sh resume --resume-batch /data/cxm/final/<旧批次目录> --background
# 显式选择正常完成的任务也可续跑，仍要求预设的 3 张空闲卡
bash run.sh resume --resume-batch /data/cxm/final/<旧批次目录> --only <完整实验名> --background
```

6h 从每条实验进入训练循环开始，在每个 epoch 的 eval 点检查；排队和初始化不计时，
可能超出一个 epoch 及存档时间。EPOCHS=6000 始终是 cosine horizon。
既有早停仍生效：最近10次新鲜测试准确率均值 >99%，或3000 epoch 无提升。
正常完成、早停、定时和手动存档退出都保留完整 model_resume.pth；
每个常规 eval 另写 model_latest.pth，最佳权重为 model_best.pth。
resume 校验配置、完整状态、base LR、WD 和 scheduler horizon，再从下一 epoch 继续。
续跑创建新的 run 目录，原文件保留；不会自动重置 scheduler 或无限续跑。

训练结束后在同一卡上用 {d['clean_eval_samples']} 条干净样本、seed={d['clean_eval_seed']}、
TRAIN_LEN 长度评估 best 权重（缺失时用完整最新存档）；汇总日志中的 total accuracy。
写 metrics/summary.json、metrics/learning_curve.csv、plots/learning_curve.png。
clean 评估和曲线生成不计入 6h 训练预算。手动停止可跳过评估以优先保存退出。

本包不自动安装依赖，不会回退 CPU 或降低预设并行卡数。
"""


def build(review_path, confirmation, output, repo=REPO):
    repo = Path(repo).resolve()
    plan = load_review(review_path)
    require(confirmation == review_id(plan), 'Confirmation does not match reviewed version')
    require(not plan['unresolved'], 'Unresolved parameters remain')
    require(source_files(repo, plan.get('delivery')) == plan['source_files'], 'Code/defaults changed since review; review again')
    output = Path(output).resolve()
    archive = Path(str(output) + '.tar.gz')
    require(not output.exists() and not archive.exists(), 'Refusing to overwrite a bundle/archive')
    for asset in plan['assets'].values():
        require(sha_file(asset['source']) == asset['sha256'], 'Checkpoint changed since review')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='experiment_bundle_', dir=output.parent) as staging:
        bundle = Path(staging) / output.name
        bundle.mkdir()
        for relative in plan['source_files']:
            if relative.startswith('src/'):
                destination = bundle / 'code' / relative
            elif relative == 'requirements.txt':
                destination = bundle / relative
            elif relative == 'scripts/experiment_package_runner.py':
                destination = bundle / ('package_utils.py' if plan.get('delivery') else 'runner.py')
            elif relative == 'scripts/remote_experiment_runner.py':
                destination = bundle / 'runner.py'
            else:
                destination = bundle / 'code' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / relative, destination)
            require(sha_file(destination) == plan['source_files'][relative], 'Code changed during packaging; review again')
        for relative, asset in plan['assets'].items():
            destination = bundle / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(asset['source'], destination)
            require(sha_file(destination) == asset['sha256'], 'Checkpoint copy mismatch')
        experiments = [dict(name=e['name'], task=e['task'], config=e['config']) for e in plan['experiments']]
        (bundle / 'experiments.json').write_text(json.dumps(dict(concurrency=plan['concurrency'], experiments=experiments), indent=2) + '\n')
        (bundle / 'PARAMETERS.md').write_text(render_review(plan))
        timers = {e['name']: e['config']['MAX_TRAIN_HOURS'] for e in experiments}
        (bundle / 'run.sh').write_text(run_script(timers))
        (bundle / 'run.sh').chmod(0o755)
        (bundle / 'README.md').write_text(
            '# 已确认实验运行包\n\n'
            'Linux；Python 3.10+；预先安装匹配目标机器 CUDA 的 PyTorch 及 requirements.txt 中的依赖。\n'
            '本包包含当前实际代码快照，不依赖目标机器旧仓库。\n\n'
            '检查：bash run.sh check\n\n'
            '启动：bash run.sh run --gpus 0,1 --output /your/output --background\n\n'
            '分发到多机时，用 --only experiment_name[,experiment_name] 选择各机负责的实验。\n\n'
            '空闲排队：bash run.sh run --gpus 2,3,0,1 --output /your/output --wait-for-idle --poll-seconds 300 --background\n\n'
            '排队时GPU列表是候选池；一张空闲即可先启动一条，等待不计入训练预算。\n\n'
            '进度：bash run.sh status --output /your/output\n\n'
            '存档停止：bash run.sh stop --output /your/output\n\n'
            'EXPERIMENT_PYTHON 环境变量可指定 Python。输出目录必须全新，避免覆盖旧结果。\n'
            '每条定时见 run.sh 中 TIMERS_JSON 和 PARAMETERS.md；从进入训练循环开始独立计时，'
            '到下一个 eval 点完整存盘，退出码 42 为正常定时/手动存档退出。'
            '无定时为 null。12h 是训练上限，不含排队和初始化；早停条件仍生效。\n'
            '参数修改后应重新审阅打包；运行脚本只允许选择 GPU、输出位置和实验子集。\n'
            '在线 train 指标不是固定权重全训练集评估；本包不自动添加额外实验或全量复评。\n')
        if plan.get('delivery'):
            (bundle / 'README.md').write_text(remote_readme(plan))
        checksums = {str(p.relative_to(bundle)): sha_file(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
        manifest = dict(version=1, confirmed_review_id=review_id(plan), built_utc=datetime.now(timezone.utc).isoformat(),
                        packaging_host=plan['packaging_host'], local_gpu_allowlist=plan['local_gpu_allowlist'],
                        batch_name=plan['batch_name'], timers=timers, files=checksums)
        if plan.get('delivery'):
            manifest['delivery'] = plan['delivery']
        (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        bundle.rename(output)
    # The complete directory remains available if writing the archive fails.
    with tarfile.open(archive, 'x:gz') as tar:
        tar.add(output, arcname=output.name)
    return output, archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    review = sub.add_parser('review', help='Print the full parameter table; do not create runnable JSON')
    review.add_argument('--request', default='-', help='Draft request from stdin or a JSON request file')
    review.add_argument('--out', required=True, type=Path, help='Markdown review artifact')
    package = sub.add_parser('build', help='Only after the user explicitly confirms the review')
    package.add_argument('--review', required=True, type=Path)
    package.add_argument('--confirm', required=True, help='Exact review ID confirmed by the user')
    package.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'review':
            request = json.load(sys.stdin) if args.request == '-' else json.loads(Path(args.request).read_text())
            plan = prepare(request)
            require(not args.out.exists(), 'Review path already exists; use a new version')
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(render_review(plan))
            print(render_review(plan, snapshot=False), end='')
            print('Review file:', args.out)
        else:
            folder, archive = build(args.review, args.confirm, args.out)
            print('Bundle:', folder)
            print('Archive:', archive)
            print('No experiments have been launched.')
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, 'Error: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
