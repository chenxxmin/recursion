import argparse

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

BASE_CONFIG_PATH = 'config.json'
DEFAULT_EXPERIMENTS_PATH = 'experiments.json'
DEFAULT_BASE_DIR = '/data/cxm/recursion'

# These are updated in main() based on the experiments config filename.
LOG_DIR = 'logs'
MODEL_DIR = 'models'
PLOT_DIR = 'plots'

import visualize


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_config_table(config):
    """Format merged config into a readable three-section table."""
    net_keys = [
        'D_MODEL', 'N_HEAD', 'N_LAYER', 'MLP_RATIO', 'DROPOUT',
        'USE_LEARNABLE_PE'
    ]
    data_keys = [
        'P', 'TASK', 'TRAIN_LEN', 'OOD_LEN', 'MAX_UNIQUE_RATIO',
        'MIXED_AB_MAX_UNIQUE_RATIOS', 'AB_PAIRS', 'A', 'B', 'C',
        'NUM_MASK'
    ]
    train_keys = [
        'BATCH_SIZE', 'LR', 'WEIGHT_DECAY', 'EPOCHS', 'RANDOM_SEED',
        'FIRST_TASK_WEIGHT', 'ENTROPY_PENALTY_WEIGHT', 'EVAL_INTERVAL',
        'EARLY_STOP_ACCURACY', 'EARLY_STOP_NO_IMPROVE', 'USE_GREEDY_GENERATE'
    ]

    lines = []
    lines.append("-" * 50)
    lines.append("Network Config")
    lines.append("-" * 50)
    for key in net_keys:
        if key in config:
            value = config[key]
            if isinstance(value, float):
                lines.append(f"{key:<25} {value}")
            elif isinstance(value, bool):
                lines.append(f"{key:<25} {value}")
            else:
                lines.append(f"{key:<25} {value}")

    lines.append("-" * 50)
    lines.append("Dataset Config")
    lines.append("-" * 50)
    for key in data_keys:
        if key in config:
            value = config[key]
            if isinstance(value, (list, tuple)):
                lines.append(f"{key:<25} {value}")
            elif isinstance(value, float):
                lines.append(f"{key:<25} {value}")
            else:
                lines.append(f"{key:<25} {value}")

    lines.append("-" * 50)
    lines.append("Training Config")
    lines.append("-" * 50)
    lines.append(f"{'OPTIMIZER':<25} AdamW")
    lines.append(f"{'SCHEDULER':<25} CosineAnnealingLR(T_max=EPOCHS)")
    for key in train_keys:
        if key in config:
            value = config[key]
            if isinstance(value, float):
                lines.append(f"{key:<25} {value}")
            elif isinstance(value, bool):
                lines.append(f"{key:<25} {value}")
            else:
                lines.append(f"{key:<25} {value}")
    lines.append("-" * 50)
    lines.append("")

    return "\n".join(lines)


def run_single(exp, base_config, concurrency=1, gpu_id=None):
    name = exp['name']
    override = exp.get('config', {})

    # 1. Read task type
    task = exp.get('task', 'addition')

    # Auto-inject TASK into config if not explicitly set
    if 'TASK' not in override:
        override = {**override, 'TASK': task}

    # 2. Merge config: main -> task defaults -> override
    merged = dict(base_config)
    merged_main = dict(base_config.get('main', {}))
    task_defaults = base_config.get(task, {})
    merged_main.update(task_defaults)

    # Auto-fill SAVE_PATH if not explicitly set
    if 'SAVE_PATH' not in override:
        override = {**override, 'SAVE_PATH': os.path.join(MODEL_DIR, f"{name}.pth")}

    merged_main.update(override)
    merged['main'] = merged_main
    merged['_BATCH_RUN_MERGED'] = True

    # 3. Use independent temp config per experiment (avoid concurrency conflicts)
    tmp_config_path = f"config_tmp_{name}.json"
    save_json(tmp_config_path, merged)

    log_path = os.path.join(LOG_DIR, f"{name}.log")
    prefix = f"[{name}] " if concurrency > 1 else ""
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] Start experiment: {name}")

    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Separate stdout and stderr:
    # - stdout: training log (clean model output)
    # - stderr: errors/warnings (PyTorch/CUDA low-level output, may contain null bytes)
    script = 'main.py'
    process = subprocess.Popen(
        [sys.executable, script, tmp_config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )

    returncode = None
    err_log_path = os.path.join(LOG_DIR, f"{name}.err")

    def _decode(raw):
        return raw.decode('utf-8', errors='replace').replace('\x00', '')

    def read_stdout():
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Experiment: {name} ===\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Task type: {task}\n")
            f.write("\n=== Merged Config ===\n")
            f.write(format_config_table(merged_main))
            f.write("\n--- output ---\n")
            f.flush()
            for raw in process.stdout:
                line = _decode(raw)
                f.write(line)
                f.flush()

    def read_stderr():
        with open(err_log_path, 'w', encoding='utf-8') as f:
            for raw in process.stderr:
                line = _decode(raw)
                if line.strip():
                    f.write(line)
                    f.flush()

    stdout_thread = threading.Thread(target=read_stdout)
    stderr_thread = threading.Thread(target=read_stderr)
    stdout_thread.start()
    stderr_thread.start()

    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n--- Return code: {returncode} ---\n")

    # After experiment succeeds, run attention analysis and append to the same log
    if returncode == 0:
        pth_path = merged_main.get('SAVE_PATH', 'fibonacci_transformer.pth')
        if os.path.exists(pth_path):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Start attention analysis: {name}")
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*70}\n")
                f.write("Attention Analysis\n")
                f.write(f"{'='*70}\n")

            analyze_process = subprocess.Popen(
                [sys.executable, 'analyze_attention.py', pth_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env
            )

            with open(log_path, 'a', encoding='utf-8') as f:
                for raw in analyze_process.stdout:
                    line = _decode(raw)
                    f.write(line)
                    f.flush()

            analyze_process.wait()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Attention analysis completed: {name}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: model file {pth_path} not found, skip attention analysis")

    # Remove empty err log
    if os.path.exists(err_log_path) and os.path.getsize(err_log_path) == 0:
        os.remove(err_log_path)

    # Clean up temp config files
    if os.path.exists(tmp_config_path):
        os.remove(tmp_config_path)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    if returncode == 0:
        print(f"[{end_time.strftime('%H:%M:%S')}] Experiment completed: {name} ({duration:.0f}s), log: {log_path}")
    else:
        print(f"[{end_time.strftime('%H:%M:%S')}] Experiment failed: {name} ({duration:.0f}s), return code: {returncode}, log: {log_path}")

    return name, returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run a batch of experiments defined in a JSON file.')
    parser.add_argument(
        'experiments_path',
        nargs='?',
        default=DEFAULT_EXPERIMENTS_PATH,
        help=f'Path to experiments config JSON (default: {DEFAULT_EXPERIMENTS_PATH})'
    )
    parser.add_argument(
        '--base-dir',
        default=DEFAULT_BASE_DIR,
        help=f'Base output directory (default: {DEFAULT_BASE_DIR})'
    )
    args = parser.parse_args()
    experiments_path = args.experiments_path

    # Place outputs under /data/<experiments filename>/logs|plots|models
    exp_name = os.path.splitext(os.path.basename(experiments_path))[0]
    work_dir = os.path.join(args.base_dir, exp_name)

    global LOG_DIR, MODEL_DIR, PLOT_DIR
    LOG_DIR = os.path.join(work_dir, 'logs')
    MODEL_DIR = os.path.join(work_dir, 'models')
    PLOT_DIR = os.path.join(work_dir, 'plots')

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(experiments_path):
        print(f"Error: experiments config file {experiments_path} not found")
        print(f"Please create {experiments_path} and define the experiment list.")
        sys.exit(1)

    base_config = load_json(BASE_CONFIG_PATH)
    raw_experiments = load_json(experiments_path)

    # Support two formats:
    # Old format: [{"name": "...", "config": {...}}, ...]
    # New format: {"concurrency": 3, "experiments": [{...}, ...]}
    if isinstance(raw_experiments, dict):
        concurrency = raw_experiments.get('concurrency', 1)
        experiments = raw_experiments.get('experiments', [])
    else:
        concurrency = 1
        experiments = raw_experiments

    if not experiments:
        print("Error: experiment list is empty")
        sys.exit(1)

    print(f"Using experiments config: {experiments_path}")
    print(f"Total experiments: {len(experiments)}, concurrency: {concurrency}")
    print("-" * 50)

    # Detect available GPUs and assign each experiment to one
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except Exception:
        num_gpus = 0
    if num_gpus > 1 and concurrency > 1:
        print(f"Detected {num_gpus} GPUs, distributing experiments round-robin")
    gpu_assignments = [i % num_gpus if num_gpus > 0 else None for i in range(len(experiments))]

    results = []

    if concurrency == 1:
        # Serial execution
        for i, exp in enumerate(experiments):
            gpu_id = gpu_assignments[i]
            name, ok = run_single(exp, base_config, concurrency=1, gpu_id=gpu_id)
            results.append((name, ok))
    else:
        # Concurrent execution
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for i, exp in enumerate(experiments):
                gpu_id = gpu_assignments[i]
                future = executor.submit(run_single, exp, base_config, concurrency, gpu_id)
                futures[future] = exp
            for future in as_completed(futures):
                name, ok = future.result()
                results.append((name, ok))

    print("\n" + "=" * 50)
    print("Experiment Summary")
    print("=" * 50)
    for name, ok in results:
        status = "[OK] Success" if ok else "[NG] Failed"
        print(f"{status}: {name}")

    # Run detailed summary
    summarize_experiments()

    # Generate grouped plots (one figure per setting, all seeds overlaid)
    generate_grouped_plots()


def generate_grouped_plots():
    """Group logs by experiment setting and plot all seeds together."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, '*.log')))

    groups = {}
    for log_path in log_files:
        name = os.path.basename(log_path)[:-4]  # remove .log
        setting, seed = visualize.extract_setting_and_seed(name)
        groups.setdefault(setting, []).append((seed or '', log_path))

    for setting in sorted(groups.keys()):
        items = groups[setting]
        data_items = []
        for seed, log_path in sorted(items, key=visualize._seed_sort_key):
            data = visualize.parse_log(log_path)
            if data['epochs']:
                data_items.append((seed, data))
        if not data_items:
            continue
        try:
            visualize.plot_setting_group(setting, data_items, PLOT_DIR)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Grouped plots saved: {setting}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: failed to generate grouped plots for {setting}: {e}")


def summarize_experiments():
    """Parse all log files and print a detailed summary table."""

    def parse_log(log_path):
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()

        result = {}

        # 1. Final epoch
        m = re.search(r'Training epochs:\s*(\d+)', text)
        result['final_epoch'] = int(m.group(1)) if m else None

        # 2. Best test accuracy
        m = re.search(r'Best test accuracy:\s*([\d.]+)', text)
        result['best_test_acc'] = float(m.group(1)) if m else None

        # 3. First task accuracy - last occurrence of targets[2] (predict x4)
        matches = re.findall(r'targets\[2\] \(predict x4\):\s*([\d.]+)%', text)
        result['first_task_acc'] = float(matches[-1]) if matches else None

        # 4. Final generation test - Exposed/Unexposed In-dist/OOD
        exposed_match = re.search(
            r'Exposed\s+samples:\s*\d+\s*\|\s*In-dist:\s*\d+/\d+\s*\(\s*([\d.]+)%\)\s*\|\s*OOD:\s*\d+/\d+\s*\(\s*([\d.]+)%\)',
            text
        )
        if exposed_match:
            result['exposed_in_dist'] = float(exposed_match.group(1))
            result['exposed_ood'] = float(exposed_match.group(2))

        unexposed_match = re.search(
            r'Unexposed\s+samples:\s*\d+\s*\|\s*In-dist:\s*\d+/\d+\s*\(\s*([\d.]+)%\)\s*\|\s*OOD:\s*\d+/\d+\s*\(\s*([\d.]+)%\)',
            text
        )
        if unexposed_match:
            result['unexposed_in_dist'] = float(unexposed_match.group(1))
            result['unexposed_ood'] = float(unexposed_match.group(2))

        return result

    log_files = sorted(glob.glob(os.path.join(LOG_DIR, '*.log')))

    if not log_files:
        print('No log files found for summary.')
        return

    print()
    print('=' * 140)
    print('Detailed Experiment Summary')
    print('=' * 140)
    print(f'{"Experiment":<45} {"Epoch":>6} {"BestTest":>8} {"1stTask":>8} {"Exp-ID":>8} {"Exp-OOD":>8} {"Unexp-ID":>8} {"Unexp-OOD":>8}')
    print('-' * 140)

    for log_path in log_files:
        name = os.path.basename(log_path)[:-4]  # Remove .log
        r = parse_log(log_path)

        epoch_str = f"{r['final_epoch']}" if r.get('final_epoch') is not None else 'N/A'
        best_str = f"{r['best_test_acc']:.1f}" if r.get('best_test_acc') is not None else 'N/A'
        first_str = f"{r['first_task_acc']:.1f}" if r.get('first_task_acc') is not None else 'N/A'
        eid_str = f"{r['exposed_in_dist']:.1f}" if r.get('exposed_in_dist') is not None else 'N/A'
        eood_str = f"{r['exposed_ood']:.1f}" if r.get('exposed_ood') is not None else 'N/A'
        uid_str = f"{r['unexposed_in_dist']:.1f}" if r.get('unexposed_in_dist') is not None else 'N/A'
        uood_str = f"{r['unexposed_ood']:.1f}" if r.get('unexposed_ood') is not None else 'N/A'

        print(f'{name:<45} {epoch_str:>6} {best_str:>8} {first_str:>8} {eid_str:>8} {eood_str:>8} {uid_str:>8} {uood_str:>8}')

    print('=' * 140)


if __name__ == '__main__':
    main()
