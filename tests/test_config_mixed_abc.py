"""Config tests: new scenario sections, deleted keys, batch_run merge order."""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

CONFIG_PATH = os.path.join(REPO_ROOT, 'src', 'config.json')

DELETED_MAIN_KEYS = ('A', 'B', 'AB_PAIRS', 'ABC_PAIRS', 'MAX_UNIQUE_RATIO',
                     'USE_LEARNABLE_PE', 'ENTROPY_PENALTY_WEIGHT', 'COND_FIX',
                     'COND_FIX_START', 'COND_FIX_START_A1', 'COND_FIX_START_A2',
                     'USE_AMP', 'AMP_DTYPE', 'ALLOW_TF32', 'PREDICT_MISSING',
                     'MISS_SECOND', 'USE_AB_TAG', 'USE_CONDITIONAL_WTE',
                     'COND_WTE_SHARED_RATIO', 'SKIP_FINAL_GENERATION_TEST',
                     'ORDER', 'RULES', 'EXPOSURE_RATIOS')


def test_config_json_main_rewrite():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    main = cfg['main']
    assert main['OOD_LEN'] == 16           # 默认从 32 改 16
    assert main['EARLY_STOP_EXTRA_EPOCHS'] == 0
    for key in DELETED_MAIN_KEYS:
        assert key not in main, f"{key} must not be in main"
    # main 不注入 ORDER/RULES/EXPOSURE_RATIOS：否则旧 action/mixed JSON 的
    # AB_PAIRS 会与 main 注入的 RULES 触发假冲突
    assert 'single' in cfg and 'mixed' in cfg and 'action' in cfg
    assert cfg['action']['NUM_TRAIN_SAMPLES'] == 10000
    assert cfg['action']['OOD_LEN'] == 32  # 旧 action JSON 的默认窗口保持
    assert cfg['tribonacci'] == {'P': 23, 'A': 1, 'B': 2, 'C': 3}
    assert cfg['mixed_ab'] == {'MIXED_AB_MAX_UNIQUE_RATIOS': [0.7, 0.7]}


def test_batch_run_merge_legacy_task():
    from batch_run import build_merged_config, BATCH_RUN_MERGED_FLAG
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        base = json.load(f)
    exp = {'name': 'smoke', 'task': 'mixed_abc',
           'config': {'P': 23, 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]]}}
    name, task, merged_main, merged = build_merged_config(exp, base, 'models/smoke')
    assert task == 'mixed_abc' and merged_main['TASK'] == 'mixed_abc'
    assert merged_main['MIXED_AB_MAX_UNIQUE_RATIOS'] == [0.7, 0.7]  # 旧 task 段生效
    assert merged_main['P'] == 23
    assert merged[BATCH_RUN_MERGED_FLAG] is True


def test_batch_run_merge_scenario_then_legacy_section():
    """查找顺序 = 场景段 -> 旧 task 段：task='action_trib' 应先吃到 action
    场景段（NUM_TRAIN_SAMPLES 等），再吃 action_trib 段。"""
    from batch_run import build_merged_config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        base = json.load(f)
    exp = {'name': 'smoke3', 'task': 'action_trib', 'config': {}}
    _, _, merged_main, _ = build_merged_config(exp, base, 'models/smoke3')
    assert merged_main['NUM_TRAIN_SAMPLES'] == 10000
    assert merged_main['TRAIN_LEN'] == 16


def test_batch_run_merge_new_action_task():
    from batch_run import build_merged_config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        base = json.load(f)
    exp = {'name': 'smoke2', 'task': 'action',
           'config': {'ORDER': 3, 'RULES': [[1, 1, 1], [3, 2, 1]]}}
    _, task, merged_main, _ = build_merged_config(exp, base, 'models/smoke2')
    assert task == 'action'
    assert merged_main['RULES'] == [[1, 1, 1], [3, 2, 1]]
    assert merged_main['NUM_TEST_SAMPLES'] == 2000  # action 场景段默认


if __name__ == '__main__':
    test_config_json_main_rewrite()
    test_batch_run_merge_legacy_task()
    test_batch_run_merge_scenario_then_legacy_section()
    test_batch_run_merge_new_action_task()
    print("ALL TESTS PASSED: test_config_mixed_abc.py")
