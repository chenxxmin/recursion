"""CPU integration checks for reviewed, portable, timed experiment bundles."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import experiment_pipeline as pipeline

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
           EXPERIMENT_PYTHON=sys.executable)


def expect_failure(fn, message):
    try:
        fn()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError('Expected failure: ' + message)


def request():
    return dict(batch_name='cpu_pipeline_test', concurrency=2, common=dict(
        P=7, D_MODEL=16, N_LAYER=1, N_HEAD=1, MLP_RATIO=2,
        BATCH_SIZE=16, TRAIN_LEN=6, OOD_LEN=8, EPOCHS=4,
        NUM_TRAIN_SAMPLES=32, NUM_TEST_SAMPLES=16,
        DATA_MODE='sampled_fresh_test', NUM_MASK=1,
        LR=0.001, WEIGHT_DECAY=0, GRAD_CLIP_NORM=5,
        RESHUFFLE_EACH_EPOCH=True, MAX_TRAIN_HOURS=0,
        EVAL_INTERVAL=1, EARLY_STOP_ACCURACY=2.0,
        SKIP_FINAL_GENERATION_TEST=True, OPT_DIAG_INTERVAL=1),
        experiments=[dict(name='tiny_seed0', task='addition', config=dict(RANDOM_SEED=0)),
                     dict(name='tiny_seed1', task='addition', config=dict(RANDOM_SEED=1))])


def shell(bundle, *args, timeout=90):
    return subprocess.run(['bash', str(bundle / 'run.sh'), *map(str, args)], env=ENV,
                          text=True, capture_output=True, timeout=timeout)


def make_review(folder, spec):
    plan = pipeline.prepare(spec)
    review = folder / (spec['batch_name'] + '.md')
    review.write_text(pipeline.render_review(plan))
    return plan, review


def test_review_and_freeze(root):
    spec = request()
    plan, review = make_review(root, spec)
    assert not list(root.glob('*.json')), 'Review must not emit runnable experiment JSON'
    assert pipeline.load_review(review) == plan
    text = review.read_text()
    assert 'MAX_TRAIN_HOURS' in text and 'GRAD_CLIP_NORM' in text and 'RANDOM_SEED' in text
    assert '用户公共参数' in text and '代码语义推导' in text
    assert all(e['config']['RESHUFFLE_EACH_EPOCH'] for e in plan['experiments'])
    expect_failure(lambda: pipeline.build(review, 'not-confirmed', root / 'wrong'), 'Confirmation')
    changed = text.replace('| LR | 0.001 |', '| LR | 0.5 |', 1)
    assert changed != text
    edited = root / 'edited.md'
    edited.write_text(changed)
    expect_failure(lambda: pipeline.load_review(edited), 'edited')
    # A reviewed source change must invalidate packaging, even if defaults match.
    stale = copy.deepcopy(plan)
    stale['source_files']['src/training.py'] = '0' * 64
    stale_review = root / 'stale.md'
    stale_review.write_text(pipeline.render_review(stale))
    expect_failure(lambda: pipeline.build(stale_review, pipeline.review_id(stale), root / 'stale'), 'Code/defaults changed')
    unresolved = copy.deepcopy(plan)
    unresolved['unresolved'] = ['seed 尚未确定']
    unresolved_review = root / 'unresolved.md'
    unresolved_review.write_text(pipeline.render_review(unresolved))
    expect_failure(lambda: pipeline.build(unresolved_review, pipeline.review_id(unresolved), root / 'unresolved'), 'Unresolved')
    bad = request()
    bad['common']['GRAD_CLIP_NROM'] = 5
    expect_failure(lambda: pipeline.prepare(bad), 'Unknown config keys')
    bundle, archive = pipeline.build(review, pipeline.review_id(plan), root / 'bundle with spaces')
    assert archive.is_file()
    assert 'TIMERS_JSON=' in (bundle / 'run.sh').read_text()
    assert shell(bundle, 'check').returncode == 0
    data_path = bundle / 'experiments.json'
    original = data_path.read_text()
    data = json.loads(original)
    assert data['experiments'][0]['config']['NUM_MASK'] == 1
    assert data['experiments'][0]['config']['EPOCHS'] == 4
    data['experiments'][0]['config']['MAX_TRAIN_HOURS'] = 100
    data_path.write_text(json.dumps(data))
    check = shell(bundle, 'check')
    assert check.returncode != 0 and 'Package file changed' in check.stderr
    data_path.write_text(original)
    return bundle


def test_timed_cpu_and_resume(root, bundle):
    output = root / 'timed output'
    ran = shell(bundle, 'run', '--cpu', '--output', output)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    state = json.loads((output / 'state.json').read_text())
    assert all(r['status'] == 'timed_out' and r['trainer_returncode'] == 42 for r in state['runs']), state
    import torch
    checkpoint = Path(state['runs'][0]['resume_checkpoint'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['next_epoch'] == 1
    assert saved['scheduler_state_dict']['T_max'] == 4
    assert saved['config']['reshuffle_each_epoch']
    assert {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'rng_python', 'rng_torch'} <= saved.keys()
    second = shell(bundle, 'run', '--cpu', '--output', output)
    assert second.returncode != 0, 'Never overwrite an earlier output'
    fresh = request()
    fresh['batch_name'] = 'resume_cpu'
    fresh['experiments'] = [dict(name='resumed_seed0', task='addition',
                                 config=dict(RANDOM_SEED=0, RESUME_FROM=str(checkpoint)))]
    plan, review = make_review(root, fresh)
    assert plan['experiments'][0]['effective']['resume_state']['next_epoch'] == 1
    res_bundle, _ = pipeline.build(review, pipeline.review_id(plan), root / 'resume bundle')
    checkpoint.rename(checkpoint.with_suffix('.original_hidden'))
    # Included checkpoint is sufficient after moving the package to another location.
    moved = root / 'relocated bundle'
    res_bundle.rename(moved)
    resumed = shell(moved, 'run', '--cpu', '--output', root / 'resumed output')
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    state2 = json.loads((root / 'resumed output/state.json').read_text())
    saved2 = torch.load(state2['runs'][0]['resume_checkpoint'], map_location='cpu', weights_only=False)
    assert saved2['next_epoch'] == 2
    assert saved2['scheduler_state_dict']['T_max'] == 4
    log = Path(state2['runs'][0]['log']).read_text()
    assert '[RESUME] continue from epoch 1' in log
    bad = request()
    bad['experiments'][0]['config'].update(RESUME_FROM=state2['runs'][0]['resume_checkpoint'], LR=0.01)
    expect_failure(lambda: pipeline.prepare(bad), 'Checkpoint overrides LR')


def test_untimed_completion_and_subset(root):
    spec = request()
    spec['batch_name'] = 'untimed_cpu'
    spec['common'].update(EPOCHS=1, MAX_TRAIN_HOURS=None)
    plan, review = make_review(root, spec)
    bundle, _ = pipeline.build(review, pipeline.review_id(plan), root / 'untimed bundle')
    output = root / 'untimed output'
    result = shell(bundle, 'run', '--cpu', '--only', 'tiny_seed1', '--output', output)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads((output / 'state.json').read_text())
    assert len(state['runs']) == 1 and state['runs'][0]['name'] == 'tiny_seed1'
    assert state['runs'][0]['status'] == 'completed', state
    assert state['runs'][0]['trainer_returncode'] == 0
    assert not (output / 'work/tiny_seed0').exists()
    status = shell(bundle, 'status', '--output', output)
    assert status.returncode == 0 and 'completed' in status.stdout

def test_manual_stop_cancels_queue(root):
    spec = request()
    spec['batch_name'] = 'manual_stop_cpu'
    spec['concurrency'] = 1
    spec['common'].update(EPOCHS=100000, MAX_TRAIN_HOURS=None)
    spec['experiments'] = [dict(name='stop_first', task='addition', config=dict(RANDOM_SEED=0)),
                           dict(name='never_start_second', task='addition', config=dict(RANDOM_SEED=1))]
    plan, review = make_review(root, spec)
    bundle, _ = pipeline.build(review, pipeline.review_id(plan), root / 'manual stop bundle')
    output = root / 'manual stop output'
    started = shell(bundle, 'run', '--cpu', '--output=' + str(output), '--background')
    assert started.returncode == 0, started.stdout + started.stderr
    deadline = time.monotonic() + 60
    log = output / 'results/stop_first/logs/stop_first.log'
    while time.monotonic() < deadline:
        if log.exists() and '[OptDiag]' in log.read_text():
            break
        time.sleep(0.2)
    else:
        raise AssertionError('Background trainer never started')
    stopped = shell(bundle, 'stop', '--output', output)
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    while time.monotonic() < deadline:
        state = json.loads((output / 'state.json').read_text())
        if state['status'] == 'stopped':
            break
        time.sleep(0.2)
    else:
        raise AssertionError('Controller did not finish graceful stop')
    assert state['runs'][0]['status'] == 'stopped_by_user', state
    assert Path(state['runs'][0]['resume_checkpoint']).is_file()
    assert state['runs'][1]['status'] == 'cancelled'
    assert not (output / 'work/never_start_second').exists()


def test_gpu_fallback_is_rejected(root):
    path = root / 'require_gpu.json'
    path.write_text(json.dumps(dict(concurrency=1, experiments=[
        dict(name='should_not_start', task='addition', config={})])))
    result = subprocess.run([sys.executable, 'src/batch_run.py', '--base-dir', str(root / 'gpu_results'),
                             '--model-base-dir', str(root / 'gpu_models'), str(path)],
                            cwd=REPO, env=dict(ENV, BATCH_RUN_REQUIRE_GPU='1', BATCH_RUN_GPUS='0'),
                            text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert 'refusing CPU/cuda:0 fallback' in result.stderr
    assert not (REPO / 'config_tmp_should_not_start.json').exists()


def test_effective_data_counts():
    sys.path.insert(0, str(REPO / 'src'))
    from experiment import _prepare_mixed_recurrence, _prepare_single_recurrence
    spec = request()
    spec['experiments'] = [dict(name='mixed_counts', task='mixed_ab', config=dict(
        AB_PAIRS=[[1, 1], [2, 3], [3, 5]], NUM_TRAIN_SAMPLES=None,
        MIXED_AB_MAX_UNIQUE_RATIOS=[0.3, 0.4, 0.5], NUM_MASK=None, DATA_MODE=None))]
    plan = pipeline.prepare(spec)
    e = plan['experiments'][0]
    with contextlib.redirect_stdout(io.StringIO()):
        actual = _prepare_mixed_recurrence({'main': e['config']}, 'cpu', order=2)
    assert e['effective']['actual_train_samples']['total'] == len(actual['train_dataset']) == 57
    assert e['effective']['actual_test_samples']['total'] == len(actual['test_dataset']) == 90
    assert e['config']['NUM_TRAIN_SAMPLES'] == [14, 19, 24]
    assert e['effective']['eval_base_length'] == 6
    spec['experiments'][0]['config'].update(FRESH_TEST_PER_EVAL=True, NUM_TEST_SAMPLES=7)
    e = pipeline.prepare(spec)['experiments'][0]
    with contextlib.redirect_stdout(io.StringIO()):
        actual = _prepare_mixed_recurrence({'main': e['config']}, 'cpu', order=2)
        loader = actual['test_loader_fn']()
    assert e['effective']['actual_test_samples']['total'] == len(loader.dataset) == 21
    assert e['effective']['eval_base_length'] == 8
    spec['experiments'][0]['config']['USE_AB_TAG'] = True
    e = pipeline.prepare(spec)['experiments'][0]
    with contextlib.redirect_stdout(io.StringIO()):
        actual = _prepare_mixed_recurrence({'main': e['config']}, 'cpu', order=2)
        loader = actual['test_loader_fn']()
    assert e['effective']['eval_sequence_tokens'] == len(loader.dataset[0][0]) == 8
    assert any('rule tag' in note for note in e['effective']['notes'])
    spec['experiments'][0]['config']['NUM_TRAIN_SAMPLES'] = 50
    expect_failure(lambda: pipeline.prepare(spec), 'per rule exceeds state space')

    spec = request()
    spec['common']['DATA_MODE'] = 'full_split'
    e = pipeline.prepare(spec)['experiments'][0]
    with contextlib.redirect_stdout(io.StringIO()):
        actual = _prepare_single_recurrence({'main': e['config']}, 'addition')
    assert e['effective']['actual_test_samples'] == len(actual['test_dataset']) == 17
    assert e['effective']['eval_base_length'] == 6
    assert e['config']['FRESH_TEST_PER_EVAL'] is False


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='experiment_pipeline_') as temporary:
        root = Path(temporary)
        test_effective_data_counts()
        print('PASS effective mixed/single counts match actual datasets', flush=True)
        bundle = test_review_and_freeze(root)
        print('PASS review, confirmation, integrity, packaging', flush=True)
        test_timed_cpu_and_resume(root, bundle)
        print('PASS portable CPU timeout/full resume', flush=True)
        test_untimed_completion_and_subset(root)
        print('PASS untimed completion, subset selection, status', flush=True)
        test_manual_stop_cancels_queue(root)
        print('PASS background start, graceful stop, pending cancellation', flush=True)
        test_gpu_fallback_is_rejected(root)
        print('PASS explicit GPU failure; no CPU/cuda:0 fallback', flush=True)
    print('ALL TESTS PASSED: test_experiment_pipeline.py', flush=True)
