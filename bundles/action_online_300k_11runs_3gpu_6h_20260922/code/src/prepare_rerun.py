"""Prepare a rerun JSON from an experiments config and its log directory.

Checks:
1. Whether each experiment's .log file exists.
2. Whether the log indicates a successful run (return code 0).
3. Removes the matching .err file for successful runs.

All missing or failed experiments are written to a new JSON that can be
fed directly to batch_run.py.
"""
import argparse
import os
import re

# Shared with batch_run.py (same repo, no import side effects beyond matplotlib
# pulled in by visualize). batch_run is the canonical home of these helpers.
from batch_run import load_json, save_json, DEFAULT_BASE_DIR

# Log verdict returned by log_status().
STATUS_MISSING = 'missing'    # no .log file
STATUS_SUCCESS = 'success'    # last return code is 0
STATUS_FAILED = 'failed'      # log exists but no success marker


def log_status(log_path):
    """Return STATUS_MISSING, STATUS_SUCCESS, or STATUS_FAILED for a log file."""
    if not os.path.exists(log_path):
        return STATUS_MISSING

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Look for the last return-code line.
    matches = re.findall(r'--- Return code:\s*(\d+)\s*---', text)
    if not matches:
        return STATUS_FAILED

    return STATUS_SUCCESS if int(matches[-1]) == 0 else STATUS_FAILED


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
        if status == STATUS_MISSING:
            missing.append(exp)
        elif status == STATUS_FAILED:
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
