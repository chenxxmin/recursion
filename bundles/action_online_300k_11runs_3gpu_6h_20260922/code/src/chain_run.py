"""Run a chain of experiment stages in strict order (curriculum-style runs).

A chain JSON looks like:

{
  "stages": [
    {"name": "phase1_A",
     "gate": {"min_best_accuracy": 0.99},
     "experiments": [{"name": "curr_A_seed0", "task": "addition", "config": {...}}]},
    {"name": "phase2_AB",
     "experiments": [{"name": "curr_AB_seed0", "task": "mixed_ab",
                      "config": {"INIT_FROM": "@prev", ...}}]}
  ]
}

Semantics:
- Stages run one at a time, in order, each via a separate batch_run.py
  invocation (batch dir = <chain-stem>_<stage-name>).
- "INIT_FROM": "@<stage>" in an experiment's config resolves to the
  checkpoint of the experiment with the same trailing "_seed<N>" name
  suffix from the named earlier stage; "@prev" is shorthand for the
  immediately previous stage (error if there is no match).
- After each stage, every experiment's checkpoint must exist, and (when the
  stage declares a gate) every experiment's best test accuracy must reach
  gate.min_best_accuracy; otherwise the chain aborts before the next stage.
- A top-level "concurrency": N is passed through to each stage's batch_run
  invocation (e.g. 2 to run two seeds in parallel on two GPUs); absent
  means batch_run's default of 1 (stage experiments run serially).

Usage (repo root):
    python src/chain_run.py experiments/curriculum_ab_p53.json \
        [--base-dir /mnt/workspace/hujiachen/recursion_results] [--model-base-dir /mnt/workspace/hujiachen/models]
    python src/chain_run.py <chain.json> --from-stage T3
        # resume at a named stage: earlier stages are not re-run, but their
        # checkpoints are verified (they anchor the @-references)
"""
import argparse
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_BASE_DIR = '/mnt/workspace/hujiachen/recursion_results'
DEFAULT_MODEL_BASE_DIR = '/mnt/workspace/hujiachen/models'

SEED_SUFFIX_RE = re.compile(r'_seed(\d+)$')
BEST_ACC_RE = re.compile(r'Best test accuracy: ([\d.]+)%')


def _seed_suffix(name):
    m = SEED_SUFFIX_RE.search(name)
    return m.group(0) if m else None


def _resolve_init_from(stage_exps, stages_so_far, chain_stem, model_base_dir):
    """Replace INIT_FROM='@...' with the same-seed checkpoint path from an
    earlier completed stage, in place. '@prev' = the immediately previous
    stage; '@<name>' = any earlier stage with that name."""
    refs = {e['name']: e['config']['INIT_FROM'][1:] for e in stage_exps
            if isinstance(e.get('config', {}).get('INIT_FROM'), str)
            and e['config']['INIT_FROM'].startswith('@')}
    for exp in stage_exps:
        if exp['name'] not in refs:
            continue
        ref = refs[exp['name']]
        if ref == 'prev':
            stage = stages_so_far[-1] if stages_so_far else None
        else:
            stage = next((s for s in stages_so_far if s['name'] == ref), None)
        if stage is None:
            raise ValueError(
                f"{exp['name']}: cannot resolve INIT_FROM '@{ref}' — "
                f"no such earlier stage")
        suffix = _seed_suffix(exp['name'])
        candidates = [e for e in stage['experiments']
                      if suffix is not None and e['name'].endswith(suffix)]
        if len(candidates) != 1:
            raise ValueError(
                f"{exp['name']}: cannot resolve INIT_FROM '@{ref}' — need exactly one "
                f"experiment ending with '{suffix}' in stage {stage['name']}")
        src = candidates[0]
        src_path = os.path.join(model_base_dir, f"{chain_stem}_{stage['name']}",
                                f"{src['name']}.pth")
        exp['config']['INIT_FROM'] = src_path
        print(f"[chain] {exp['name']}: INIT_FROM <- {stage['name']}/{src['name']} ({src_path})")


def _check_stage_outputs(stage, batch_name, base_dir, model_base_dir):
    """Verify every experiment's checkpoint exists after its stage ran."""
    for exp in stage['experiments']:
        pth = os.path.join(model_base_dir, batch_name, f"{exp['name']}.pth")
        if not os.path.exists(pth):
            raise RuntimeError(f"stage {stage['name']}: checkpoint missing after run: {pth}")


def _check_gate(stage, batch_name, base_dir):
    """Enforce the stage's min_best_accuracy gate from the training logs."""
    gate = stage.get('gate')
    if not gate:
        return
    threshold = gate['min_best_accuracy']
    for exp in stage['experiments']:
        log_path = os.path.join(base_dir, batch_name, 'logs', f"{exp['name']}.log")
        best = None
        with open(log_path, encoding='utf-8') as f:
            for m in BEST_ACC_RE.finditer(f.read()):
                best = float(m.group(1)) / 100.0  # last occurrence wins
        if best is None:
            raise RuntimeError(f"gate: no 'Best test accuracy' line in {log_path}")
        print(f"[chain] gate: {exp['name']} best test accuracy = {best:.4f} "
              f"(required >= {threshold})")
        if best < threshold:
            raise RuntimeError(
                f"chain aborted: {exp['name']} did not reach gate "
                f"min_best_accuracy={threshold} (got {best:.4f})")


def main():
    ap = argparse.ArgumentParser(description='Run chained experiment stages in order.')
    ap.add_argument('chain_json', help='chain JSON path (see module docstring)')
    ap.add_argument('--base-dir', default=DEFAULT_BASE_DIR,
                    help=f'log/plot base dir (default: {DEFAULT_BASE_DIR})')
    ap.add_argument('--model-base-dir', default=DEFAULT_MODEL_BASE_DIR,
                    help=f'model base dir (default: {DEFAULT_MODEL_BASE_DIR})')
    ap.add_argument('--from-stage', default=None, metavar='STAGE',
                    help='resume the chain at the named stage: earlier stages are not '
                         're-run, but their checkpoints must already exist (they are '
                         'verified, since @-references resolve to them)')
    args = ap.parse_args()

    with open(args.chain_json, encoding='utf-8') as f:
        chain = json.load(f)
    stages = chain['stages']
    if not stages:
        raise ValueError('chain JSON has no stages')
    chain_stem = os.path.splitext(os.path.basename(args.chain_json))[0]

    start_idx = 0
    if args.from_stage is not None:
        names = [s['name'] for s in stages]
        if args.from_stage not in names:
            raise ValueError(f"--from-stage {args.from_stage!r} not in stage names {names}")
        start_idx = names.index(args.from_stage)
        # Skipped stages are resume anchors for @-references; their checkpoints
        # must already exist.
        for s in stages[:start_idx]:
            _check_stage_outputs(s, f"{chain_stem}_{s['name']}",
                                 args.base_dir, args.model_base_dir)
        print(f"[chain] resuming at stage {args.from_stage!r} "
              f"({start_idx} earlier stage(s) verified, not re-run)")

    tmp_dir = os.path.join(REPO_ROOT, '.chain_tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    for idx in range(start_idx, len(stages)):
        stage = stages[idx]
        batch_name = f"{chain_stem}_{stage['name']}"
        print(f"\n[chain] ===== stage {idx + 1}/{len(stages)}: {stage['name']} "
              f"({len(stage['experiments'])} experiments, batch dir {batch_name}) =====")
        _resolve_init_from(stage['experiments'], stages[:idx], chain_stem,
                           args.model_base_dir)

        stage_path = os.path.join(tmp_dir, f'{batch_name}.json')
        stage_batch = {'experiments': stage['experiments']}
        if 'concurrency' in chain:
            stage_batch['concurrency'] = chain['concurrency']
        with open(stage_path, 'w', encoding='utf-8') as f:
            json.dump(stage_batch, f, indent=2)

        ret = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, 'batch_run.py'), stage_path,
             '--base-dir', args.base_dir, '--model-base-dir', args.model_base_dir]).returncode
        if ret != 0:
            raise RuntimeError(f"stage {stage['name']}: batch_run exited with {ret}")

        _check_stage_outputs(stage, batch_name, args.base_dir, args.model_base_dir)
        _check_gate(stage, batch_name, args.base_dir)

    print(f"\n[chain] all {len(stages)} stages completed.")


if __name__ == '__main__':
    main()
