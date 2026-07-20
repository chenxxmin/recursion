"""Prepare a rerun JSON from an experiments config and its log directory.

Checks:
1. Whether each experiment's .log file exists.
2. Whether the log indicates a successful run (return code 0).
3. Removes the matching .err file for successful runs.

All missing or failed experiments are written to a new JSON that can be
fed directly to batch_run.py.
"""
import argparse
import json
import os
import re

DEFAULT_BASE_DIR = '/data/cxm/recursion'


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log_status(log_path):
    """Return 'missing', 'success', or 'failed' for a log file."""
    if not os.path.exists(log_path):
        return 'missing'

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Look for the last return-code line.
    matches = re.findall(r'--- Return code:\s*(\d+)\s*---', text)
    if not matches:
        return 'failed'

    return 'success' if int(matches[-1]) == 0 else 'failed'


def default_log_dir(experiments_path):
    """Derive the default log dir matching batch_run.py conventions."""
    exp_name = os.path.splitext(os.path.basename(experiments_path))[0]
    return os.path.join(DEFAULT_BASE_DIR, exp_name, 'logs')


def main():
    parser = argparse.ArgumentParser(
        description='Prepare a rerun JSON for missing/failed experiments.'
    )
    parser.add_argument(
        'experiments_path',
        nargs='?',
        default='experiments/experiments.json',
        help='Input experiments JSON (default: experiments/experiments.json)'
    )
    parser.add_argument(
        '--log-dir',
        default=None,
        help=f'Directory containing .log files (default: {DEFAULT_BASE_DIR}/<exp_name>/logs)'
    )
    parser.add_argument(
        '--out',
        default=None,
        help='Output JSON for rerun (default: <exp_name>_rerun.json)'
    )
    args = parser.parse_args()

    log_dir = args.log_dir or default_log_dir(args.experiments_path)
    out_path = args.out or os.path.splitext(args.experiments_path)[0] + '_rerun.json'

    raw = load_json(args.experiments_path)
    if isinstance(raw, dict):
        experiments = raw.get('experiments', [])
        metadata = {k: v for k, v in raw.items() if k != 'experiments'}
    else:
        experiments = raw
        metadata = {}

    missing = []
    failed = []
    success = 0

    for exp in experiments:
        name = exp['name']
        log_path = os.path.join(log_dir, f'{name}.log')
        err_path = os.path.join(log_dir, f'{name}.err')

        status = log_status(log_path)
        if status == 'missing':
            missing.append(exp)
        elif status == 'failed':
            failed.append(exp)
        else:
            success += 1
            if os.path.exists(err_path):
                os.remove(err_path)
                print(f'Removed err file: {err_path}')

    rerun_experiments = missing + failed

    if metadata:
        out_data = dict(metadata)
        out_data['experiments'] = rerun_experiments
    else:
        out_data = rerun_experiments

    save_json(out_path, out_data)

    print('-' * 50)
    print(f'Total experiments : {len(experiments)}')
    print(f'Success           : {success}')
    print(f'Missing logs      : {len(missing)}')
    print(f'Failed logs       : {len(failed)}')
    print(f'Rerun experiments : {len(rerun_experiments)}')
    print(f'Output written to : {out_path}')


if __name__ == '__main__':
    main()
