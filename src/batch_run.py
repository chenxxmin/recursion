import argparse
import glob
import json
import os
import re
import subprocess
import sys
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Ensure imports of src/ modules work when running from the repo root as:
#   python src/batch_run.py experiments/some_task.json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from protocol import BATCH_RUN_MERGED_FLAG

BASE_CONFIG_PATH = 'src/config.json'
DEFAULT_EXPERIMENTS_PATH = 'experiments/experiments.json'
DEFAULT_BASE_DIR = '/data/cxm/recursion'
DEFAULT_MODEL_BASE_DIR = '/data/cxm/models'

# Fallback when the merged config has no SAVE_PATH (should not happen since
# build_merged_config auto-fills it; kept for defensive .get()).
DEFAULT_SAVE_PATH = 'fibonacci_transformer.pth'

SEP_WIDTH = 50      # width of console section separators
SUMMARY_WIDTH = 140  # width of the detailed summary table
NAME_COL = 45       # summary table: experiment name column width
EPOCH_COL = 6       # summary table: epoch column width
NUM_COL = 8         # summary table: numeric column width

import visualize


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_config_table(config):
    """Format the full merged config as a sorted key table.

    Every key passed to core.py is printed so the log is complete evidence
    of what actually ran (the old section whitelists omitted live keys such
    as MISSING_PROB / USE_AB_TAG / COND_FIX_*).
    """
    lines = ["-" * SEP_WIDTH, "Merged Config (all keys)", "-" * SEP_WIDTH]
    for key in sorted(config):
        lines.append(f"{key:<25} {config[key]}")
    lines.append("-" * SEP_WIDTH)
    lines.append("")
    return "\n".join(lines)


def get_idle_gpus(memory_threshold_mb=100):
    """Return GPU indices that appear to be idle (low memory usage).

    Uses nvidia-smi to query per-GPU memory usage. GPUs with used memory
    below ``memory_threshold_mb`` are considered idle and safe to use.

    Returns ``None`` if nvidia-smi is unavailable, so callers can fall back
    to torch.cuda.device_count().
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        idle_gpus = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(',')
            idx = int(parts[0].strip())
            used_mb = float(parts[1].strip())
            if used_mb < memory_threshold_mb:
                idle_gpus.append(idx)
        return idle_gpus
    except Exception:
        return None


def _decode(raw):
    return raw.decode('utf-8', errors='replace').replace('\x00', '')


def _spawn_script(script_name, args, env, stderr=subprocess.PIPE):
    """Launch a src/ script as a subprocess: python src/<script_name> <args...>.

    Script-file entry points are used instead of `python -c` strings so that
    renaming/moving a function never silently breaks the command line; the
    script's own directory lands on sys.path automatically.
    """
    return subprocess.Popen([sys.executable, os.path.join(SCRIPT_DIR, script_name), *args],
                            stdout=subprocess.PIPE, stderr=stderr, env=env)


def build_merged_config(exp, base_config, model_dir):
    """Merge base config with per-experiment overrides.

    Merge order: main -> task defaults -> experiment override.
    Returns (name, task, merged_main, merged): merged_main is the effective
    'main' section; merged is the full config to write to the temp file.
    """
    name = exp['name']
    override = exp.get('config', {})

    # 1. Read task type. The top-level `task` field is the single routing
    # source; a divergent TASK inside the experiment's config override is
    # ignored (with a warning) so the two can never silently disagree.
    task = exp.get('task', 'addition')
    if task == 'dynamic_mixed':
        print(f"[{name}] Warning: task 'dynamic_mixed' is deprecated, use 'action'")
        task = 'action'
    if override.get('TASK') == 'dynamic_mixed':
        override = {**override, 'TASK': 'action'}
    if 'TASK' in override and override['TASK'] != task:
        print(f"[{name}] Warning: config.TASK={override['TASK']!r} ignored; "
              f"routing uses top-level task={task!r}")
    override = {**override, 'TASK': task}

    # 2. Merge config: main -> task defaults -> override
    merged = dict(base_config)
    merged_main = dict(base_config.get('main', {}))
    task_defaults = base_config.get(task, {})
    merged_main.update(task_defaults)

    # Auto-fill SAVE_PATH if not explicitly set
    if 'SAVE_PATH' not in override:
        override = {**override, 'SAVE_PATH': os.path.join(model_dir, f"{name}.pth")}

    merged_main.update(override)
    merged['main'] = merged_main
    merged[BATCH_RUN_MERGED_FLAG] = True
    return name, task, merged_main, merged


def run_attention_analysis(name, pth_path, log_path, env):
    """Run attention analysis on a trained checkpoint and append it to the log."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Start attention analysis: {name}")
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*70}\n")
        f.write("Attention Analysis\n")
        f.write(f"{'='*70}\n")

    analyze_process = _spawn_script('analyze_attention.py', [pth_path, log_path],
                                    env, stderr=subprocess.STDOUT)

    with open(log_path, 'a', encoding='utf-8') as f:
        for raw in analyze_process.stdout:
            line = _decode(raw)
            f.write(line)
            f.flush()

    analyze_ret = analyze_process.wait()
    if analyze_ret != 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: attention analysis for "
              f"{name} exited with code {analyze_ret}; see {log_path} for details")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Attention analysis completed: {name}")


def run_single(exp, base_config, dirs, concurrency=1, gpu_id=None):
    name, task, merged_main, merged = build_merged_config(exp, base_config, dirs['model'])

    # Use independent temp config per experiment (avoid concurrency conflicts)
    tmp_config_path = f"config_tmp_{name}.json"
    save_json(tmp_config_path, merged)

    try:
        log_path = os.path.join(dirs['log'], f"{name}.log")
        start_time = datetime.now()
        gpu_label = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
        print(f"[{start_time.strftime('%H:%M:%S')}] Start experiment: {name} on {gpu_label}")

        env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
        if gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # Separate stdout and stderr:
        # - stdout: training log (clean model output)
        # - stderr: errors/warnings (PyTorch/CUDA low-level output, may contain null bytes)
        process = _spawn_script('core.py', [tmp_config_path], env, stderr=subprocess.PIPE)

        err_log_path = os.path.join(dirs['log'], f"{name}.err")

        def read_stdout():
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"=== Experiment: {name} ===\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(f"Task type: {task}\n")
                f.write(f"GPU: {gpu_label}\n")
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
            pth_path = merged_main.get('SAVE_PATH', DEFAULT_SAVE_PATH)
            if os.path.exists(pth_path):
                run_attention_analysis(name, pth_path, log_path, env)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: model file {pth_path} not found, skip attention analysis")

        # Remove empty err log
        if os.path.exists(err_log_path) and os.path.getsize(err_log_path) == 0:
            os.remove(err_log_path)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        if returncode == 0:
            print(f"[{end_time.strftime('%H:%M:%S')}] Experiment completed: {name} ({duration:.0f}s), log: {log_path}")
        elif returncode == 42:
            print(f"[{end_time.strftime('%H:%M:%S')}] Experiment timed out (resume checkpoint saved): {name} ({duration:.0f}s), "
                  f"resume later with RESUME_FROM, log: {log_path}")
        else:
            print(f"[{end_time.strftime('%H:%M:%S')}] Experiment failed: {name} ({duration:.0f}s), return code: {returncode}, log: {log_path}")

        return name, returncode == 0, returncode
    finally:
        # Clean up the temp config even on KeyboardInterrupt / spawn failure.
        if os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)


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
        help=f'Base directory for logs and plots (default: {DEFAULT_BASE_DIR})'
    )
    parser.add_argument(
        '--model-base-dir',
        default=DEFAULT_MODEL_BASE_DIR,
        help=f'Base directory for models (default: {DEFAULT_MODEL_BASE_DIR})'
    )
    args = parser.parse_args()
    experiments_path = args.experiments_path

    # Per-experiment output directories:
    # logs:   <base-dir>/<experiment-name>/logs/
    # plots:  <base-dir>/<experiment-name>/plots/
    # models: <model-base-dir>/<experiment-name>/
    exp_name = os.path.splitext(os.path.basename(experiments_path))[0]
    work_dir = os.path.join(args.base_dir, exp_name)

    # Per-experiment-batch output directories, passed explicitly to consumers:
    # log:   <base-dir>/<experiment-name>/logs/
    # plot:  <base-dir>/<experiment-name>/plots/
    # model: <model-base-dir>/<experiment-name>/
    dirs = {
        'log': os.path.join(work_dir, 'logs'),
        'plot': os.path.join(work_dir, 'plots'),
        'model': os.path.join(args.model_base_dir, exp_name),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

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
    print("-" * SEP_WIDTH)

    # Detect GPUs. Prefer idle GPUs reported by nvidia-smi; fall back to the
    # total GPU count from torch if nvidia-smi is unavailable. Then cap
    # concurrency so each GPU runs at most one experiment at a time.
    try:
        import torch
        num_gpus_total = torch.cuda.device_count()
    except Exception:
        num_gpus_total = 0

    idle_gpus = get_idle_gpus() if num_gpus_total > 0 else None

    if idle_gpus is not None:
        gpu_ids = idle_gpus
        detection_msg = f"{len(gpu_ids)} idle of {num_gpus_total} GPUs: {gpu_ids}"
    elif num_gpus_total > 0:
        gpu_ids = list(range(num_gpus_total))
        detection_msg = f"{num_gpus_total} GPUs (could not check idle status)"
    else:
        gpu_ids = []
        detection_msg = "No GPUs detected"

    print("=" * SEP_WIDTH)

    # Optional GPU whitelist: BATCH_RUN_GPUS="0,1,2,3" restricts which
    # (physical) GPU ids this batch may use, so two batch_run instances can
    # partition the machine without landing on the same card.
    env_gpus = os.environ.get('BATCH_RUN_GPUS')
    if env_gpus:
        allowed = {int(x) for x in env_gpus.split(',') if x.strip()}
        gpu_ids = [g for g in gpu_ids if g in allowed]
        print(f"BATCH_RUN_GPUS whitelist: {sorted(allowed)} -> usable: {gpu_ids}")

    if gpu_ids:
        effective_workers = min(concurrency, len(gpu_ids))
        gpu_queue = queue.Queue()
        for gpu_id in gpu_ids:
            gpu_queue.put(gpu_id)
        print(f"GPU detection: {detection_msg}")
        print(f"Effective concurrency: {effective_workers} (one experiment per GPU)")
    else:
        effective_workers = concurrency
        gpu_queue = None
        print(f"GPU detection: {detection_msg}")
        print("Running on CPU")
    print("=" * SEP_WIDTH)

    results = []

    def run_with_gpu(exp):
        """Pick a free GPU, run one experiment, then return the GPU."""
        if gpu_queue is not None:
            gpu_id = gpu_queue.get()
            try:
                return run_single(exp, base_config, dirs, effective_workers, gpu_id)
            finally:
                gpu_queue.put(gpu_id)
        else:
            return run_single(exp, base_config, dirs, effective_workers, None)

    if effective_workers == 1:
        # Avoid thread overhead for purely serial execution.
        for exp in experiments:
            name, ok, _ = run_with_gpu(exp)
            results.append((name, ok))
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(run_with_gpu, exp) for exp in experiments]
            for future in as_completed(futures):
                name, ok, _ = future.result()
                results.append((name, ok))

    print("\n" + "=" * SEP_WIDTH)
    print("Experiment Summary")
    print("=" * SEP_WIDTH)
    for name, ok in results:
        status = "[OK] Success" if ok else "[NG] Failed"
        print(f"{status}: {name}")

    # Run detailed summary
    summarize_experiments(dirs['log'])

    # Generate grouped plots (one figure per setting, all seeds overlaid)
    generate_grouped_plots(dirs['log'], dirs['plot'])


def generate_grouped_plots(log_dir, plot_dir):
    """Group logs by experiment setting and plot all seeds together."""
    os.makedirs(plot_dir, exist_ok=True)
    log_files = sorted(glob.glob(os.path.join(log_dir, '*.log')))

    groups = {}
    for log_path in log_files:
        name = os.path.splitext(os.path.basename(log_path))[0]  # remove .log
        setting, seed = visualize.extract_setting_and_seed(name)
        groups.setdefault(setting, []).append((seed or '', log_path))

    for setting in sorted(groups.keys()):
        items = groups[setting]
        data_items = []
        for seed, log_path in sorted(items, key=visualize.seed_sort_key):
            data = visualize.parse_log(log_path)
            if data['epochs']:
                data_items.append((seed, data))
        if not data_items:
            continue
        try:
            visualize.plot_setting_group(setting, data_items, plot_dir)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Grouped plots saved: {setting}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: failed to generate grouped plots for {setting}: {e}")


def summarize_experiments(log_dir):
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

    log_files = sorted(glob.glob(os.path.join(log_dir, '*.log')))

    if not log_files:
        print('No log files found for summary.')
        return

    print()
    print('=' * SUMMARY_WIDTH)
    print('Detailed Experiment Summary')
    print('=' * SUMMARY_WIDTH)
    print(f'{"Experiment":<{NAME_COL}} {"Epoch":>{EPOCH_COL}} {"BestTest":>{NUM_COL}} {"1stTask":>{NUM_COL}} '
          f'{"Exp-ID":>{NUM_COL}} {"Exp-OOD":>{NUM_COL}} {"Unexp-ID":>{NUM_COL}} {"Unexp-OOD":>{NUM_COL}}')
    print('-' * SUMMARY_WIDTH)

    def fmt(value, spec='.1f'):
        return format(value, spec) if value is not None else 'N/A'

    for log_path in log_files:
        name = os.path.splitext(os.path.basename(log_path))[0]  # Remove .log
        r = parse_log(log_path)

        cols = [fmt(r.get('final_epoch'), ''),
                fmt(r.get('best_test_acc')), fmt(r.get('first_task_acc')),
                fmt(r.get('exposed_in_dist')), fmt(r.get('exposed_ood')),
                fmt(r.get('unexposed_in_dist')), fmt(r.get('unexposed_ood'))]

        print(f'{name:<{NAME_COL}} {cols[0]:>{EPOCH_COL}} '
              + ' '.join(f'{c:>{NUM_COL}}' for c in cols[1:]))

    print('=' * SUMMARY_WIDTH)


if __name__ == '__main__':
    main()
