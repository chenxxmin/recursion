"""Golden-master equivalence: legacy task spelling vs new scenario spelling.

Same seed, tiny CPU runs (2 epochs). single/action: the two spellings of the
same experiment must produce bitwise-identical model weights (normalization
is pure translation and consumes no RNG). mixed: the model differs by design
(rule_head removed), so only the datasets are compared sample-for-sample;
this doubles as the coefficient-reversal check for action_o3 vs action_trib.
"""
import contextlib
import io
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

from experiment import run_experiment, _prepare_mixed_recurrence
from scenarios import normalize_task_config


def _base_main():
    return {'P': 7, 'D_MODEL': 16, 'N_HEAD': 1, 'N_LAYER': 1, 'BATCH_SIZE': 8,
            'EPOCHS': 2, 'LR': 0.001, 'RANDOM_SEED': 42, 'TRAIN_LEN': 8,
            'OOD_LEN': 8, 'DROPOUT': 0.0, 'WEIGHT_DECAY': 0.1,
            'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0,
            'EARLY_STOP_NO_IMPROVE': 3000, 'MLP_RATIO': 2, 'NUM_MASK': None}


def _run(main, tmpdir, tag):
    main = dict(main)
    main['SAVE_PATH'] = os.path.join(tmpdir, f'{tag}.pth')
    path = os.path.join(tmpdir, f'{tag}.json')
    with open(path, 'w') as f:
        json.dump({'main': main, '_BATCH_RUN_MERGED': True}, f)
    with contextlib.redirect_stdout(io.StringIO()):
        run_experiment(path)
    return torch.load(main['SAVE_PATH'], map_location='cpu', weights_only=False)


def _assert_same_weights(a, b):
    sa, sb = a['model_state_dict'], b['model_state_dict']
    assert set(sa) == set(sb), f"state keys differ: {sorted(set(sa) ^ set(sb))}"
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"weights differ at {k}"


def test_single_o3_equals_tribonacci():
    with tempfile.TemporaryDirectory() as tmpdir:
        old = _run({**_base_main(), 'TASK': 'tribonacci', 'A': 1, 'B': 2, 'C': 3},
                   tmpdir, 'old')
        new = _run({**_base_main(), 'TASK': 'single', 'ORDER': 3,
                    'RULES': [[1, 2, 3]]}, tmpdir, 'new')
        _assert_same_weights(old, new)
        assert new['config']['scenario'] == 'single'
        assert new['config']['order'] == 3
        assert new['config']['rules'] == [[1, 2, 3]]
        # 旧字段照写，供旧分析工具
        assert (new['config']['a'], new['config']['b'], new['config']['c']) == (1, 2, 3)
        assert new['config']['recurrence'] == 'tribonacci'


def test_action_o3_equals_action_trib():
    # RULES 是 canonical（第一个系数乘最新值）；ABC_PAIRS 是 action 旧约定
    # （第一个系数乘最老值）。权重逐位相等即证明两处反转互为镜像、净效果为零。
    with tempfile.TemporaryDirectory() as tmpdir:
        old = _run({**_base_main(), 'TASK': 'action_trib',
                    'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]],
                    'NUM_TRAIN_SAMPLES': 40, 'NUM_TEST_SAMPLES': 16}, tmpdir, 'old')
        new = _run({**_base_main(), 'TASK': 'action', 'ORDER': 3,
                    'RULES': [[1, 1, 1], [3, 2, 1]],
                    'NUM_TRAIN_SAMPLES': 40, 'NUM_TEST_SAMPLES': 16}, tmpdir, 'new')
        _assert_same_weights(old, new)
        assert new['config']['scenario'] == 'action'
        assert new['config']['rules'] == [[1, 1, 1], [3, 2, 1]]      # canonical
        assert new['config']['ab_pairs'] == [[1, 1, 1], [1, 2, 3]]   # action 约定照写


def test_mixed_o2_dataset_equals_mixed_ab():
    # 模型结构不同（rule_head 已删），只比数据集逐点相同与样本数（规格 §5）。
    old_cfg = {'P': 7, 'D_MODEL': 16, 'N_HEAD': 1, 'N_LAYER': 1, 'BATCH_SIZE': 8,
               'TRAIN_LEN': 8, 'OOD_LEN': 8, 'DROPOUT': 0.0,
               'AB_PAIRS': [[1, 1], [2, 3]], 'MIXED_AB_MAX_UNIQUE_RATIOS': [0.4, 0.4]}
    new_cfg = {'P': 7, 'D_MODEL': 16, 'N_HEAD': 1, 'N_LAYER': 1, 'BATCH_SIZE': 8,
               'TRAIN_LEN': 8, 'OOD_LEN': 8, 'DROPOUT': 0.0,
               'ORDER': 2, 'RULES': [[1, 1], [2, 3]], 'EXPOSURE_RATIOS': [0.4, 0.4]}
    random.seed(0)
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        old_ctx = _prepare_mixed_recurrence({'main': old_cfg}, 'cpu',
                                            normalize_task_config('mixed_ab', old_cfg))
    random.seed(0)
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        new_ctx = _prepare_mixed_recurrence({'main': new_cfg}, 'cpu',
                                            normalize_task_config('mixed', new_cfg))
    assert len(old_ctx['train_dataset']) == len(new_ctx['train_dataset'])
    assert len(old_ctx['test_dataset']) == len(new_ctx['test_dataset'])
    for (so, lo), (sn, ln) in zip(old_ctx['train_dataset'], new_ctx['train_dataset']):
        assert lo == ln and so.tolist() == sn.tolist()
    for (so, lo), (sn, ln) in zip(old_ctx['test_dataset'], new_ctx['test_dataset']):
        assert lo == ln and so.tolist() == sn.tolist()


if __name__ == '__main__':
    test_single_o3_equals_tribonacci()
    test_action_o3_equals_action_trib()
    test_mixed_o2_dataset_equals_mixed_ab()
    print("ALL TESTS PASSED: test_scenario_equivalence.py")
