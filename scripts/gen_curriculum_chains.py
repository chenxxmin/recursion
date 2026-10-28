#!/usr/bin/env python3
"""生成三条课程链的实验 JSON（len5 位置递进 / len8 长度递进 / len8 位置递进）。
每级 INIT_FROM 指向上级 ckpt；配置沿用 value_only 获胜参数。
"""
import json, os

COEFS = [(25,56),(62,59),(56,93),(5,3),(67,65),(44,59)]
BASE_LR, BASE_WD = 0.0003, 2.0
EPOCHS = 3000


def base_cfg(a, b, length):
    return {
        "TASK": "single", "TASK_VARIANT": "corrupt_fix", "ORDER": 2, "P": 127,
        "D_MODEL": 512, "N_LAYER": 4, "N_HEAD": 4, "MLP_RATIO": 8,
        "TRAIN_LEN": length, "OOD_LEN": length, "BODY_SUPERVISION": False,
        "CORRUPT_FIX_VALUE_ONLY": True,
        "EPOCHS": EPOCHS, "EVAL_INTERVAL": 5, "MAX_TRAIN_HOURS": 1.0,
        "SAVE_CHECKPOINTS": True, "RANDOM_SEED": 42,
        "LR": BASE_LR, "WEIGHT_DECAY": BASE_WD, "RULES": [[a, b]],
    }


def batch_name(prefix):
    return f'variant_o2_corruptfix_valueonly_d512l4_P127_{prefix}'


def ckpt_dir(prefix):
    return f'/data/cxm/models/{batch_name(prefix)}'


def ckpt_path(prefix, a, b, suffix):
    return f'{ckpt_dir(prefix)}/single_o2_corruptfix_a{a}b{b}_d512l4r8h4_P127_{suffix}_best.pth'


def make_stage(prefix, suffix, length, pos_set, init_prefix, init_suffix, concurrency=6):
    exps = []
    for a, b in COEFS:
        cfg = base_cfg(a, b, length)
        if pos_set is not None:
            cfg['CORRUPT_FIX_POS_SET'] = pos_set
        if init_prefix is not None:
            cfg['INIT_FROM'] = ckpt_path(init_prefix, a, b, init_suffix)
        exps.append({'name': f'single_o2_corruptfix_a{a}b{b}_d512l4r8h4_P127_{suffix}_seed42',
                     'task': 'single', 'config': cfg})
    path = f'experiments/{batch_name(prefix)}.json'
    json.dump({'concurrency': concurrency, 'experiments': exps}, open(path, 'w'), indent=1)
    return path


stages = []
# --- 链1：len5 位置递进 ---
for i in range(5):
    pos = list(range(i + 1)) if i < 4 else None
    prev = f'len5_pos{i-1}' if i > 0 else None
    suffix = f'len5_pos{i}' if i < 4 else 'len5_pos01234'
    init_suffix = f'{prev}_seed42' if prev else None
    stages.append(make_stage(f'len5_pos{i}', suffix, 5, pos, prev, init_suffix))
# --- 链2：len8 长度递进（从 len5 全位置 ckpt 起） ---
prev = None
for L in (6, 7, 8):
    suffix = f'len{L}_initLen{L-1}'
    init_suffix = 'len5_valueonly_initPos024_seed42' if L == 6 else f'len{L-1}_initLen{L-2}_seed42'
    init_prefix = 'len5_valueonly_initPos024' if L == 6 else f'len{L-1}_initLen{L-2}'
    stages.append(make_stage(f'len{L}_initLen{L-1}', suffix, L, None,
                             init_prefix, init_suffix))
# --- 链3：len8 位置递进 ---
for i in range(8):
    pos = list(range(i + 1)) if i < 7 else None
    prev = f'len8_pos{i-1}' if i > 0 else None
    suffix = f'len8_pos{i}' if i < 7 else 'len8_pos01234567'
    init_suffix = f'{prev}_seed42' if prev else None
    stages.append(make_stage(f'len8_pos{i}', suffix, 8, pos, prev, init_suffix))

with open('scripts/curriculum_stages.json', 'w') as f:
    json.dump(stages, f, indent=1)
print(f'{len(stages)} stages written to scripts/curriculum_stages.json')
for s in stages:
    print(' ', s)
