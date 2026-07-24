"""Config tests: mixed_abc section defaults and batch_run merge behavior."""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

CONFIG_PATH = os.path.join(REPO_ROOT, 'src', 'config.json')


def test_config_json_has_mixed_abc_section():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    section = cfg['mixed_abc']
    assert section['USE_AB_TAG'] is False
    assert section['USE_CONDITIONAL_WTE'] is False
    assert section['COND_WTE_SHARED_RATIO'] == 0.0
    assert section['MIXED_AB_MAX_UNIQUE_RATIOS'] == [0.7, 0.7]


def test_batch_run_merge_mixed_abc():
    from batch_run import build_merged_config, BATCH_RUN_MERGED_FLAG
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        base = json.load(f)
    exp = {'name': 'smoke', 'task': 'mixed_abc',
           'config': {'P': 23, 'ABC_PAIRS': [[1, 1, 1], [1, 2, 3]]}}
    name, task, merged_main, merged = build_merged_config(exp, base, 'models/smoke')
    assert task == 'mixed_abc'
    assert merged_main['TASK'] == 'mixed_abc'
    # task-section defaults land in merged main
    assert merged_main['USE_AB_TAG'] is False
    assert merged_main['MIXED_AB_MAX_UNIQUE_RATIOS'] == [0.7, 0.7]
    # experiment override wins
    assert merged_main['P'] == 23
    assert merged_main['ABC_PAIRS'] == [[1, 1, 1], [1, 2, 3]]
    assert merged[BATCH_RUN_MERGED_FLAG] is True


if __name__ == '__main__':
    test_config_json_has_mixed_abc_section()
    test_batch_run_merge_mixed_abc()
    print("ALL TESTS PASSED: test_config_mixed_abc.py")
