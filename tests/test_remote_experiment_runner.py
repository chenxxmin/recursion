"""CPU-only checks for remote admission, FIFO dispatch, output layout and full resume."""
import argparse
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from unittest import mock

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import experiment_pipeline as pipeline
import remote_experiment_runner as runner


class Lease:
    def __init__(self):
        self.closed = False
    def close(self):
        self.closed = True


def table(idle):
    return {g: dict(uuid=f'TEST-{g}', name='Mock GPU', memory_mib=0 if g in idle else 9000,
                    utilization=0 if g in idle else 100, has_compute_process=g not in idle)
            for g in range(8)}


def failure(fn, contains):
    try:
        fn()
    except ValueError as error:
        assert contains in str(error), str(error)
    else:
        raise AssertionError('Expected failure: ' + contains)


def test_admission():
    manifest = dict(packaging_host='other-machine', local_gpu_allowlist=[0, 1, 2, 3])
    with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'):
        with mock.patch.object(runner, 'inventory', return_value=table({1, 4})), \
             mock.patch.object(runner.common, 'reserve_gpus') as reserve:
            failure(lambda: runner.admit('auto', 3, manifest), 'NO experiments started')
            reserve.assert_not_called()
        leases = []
        def reserve(gpus, metadata):
            if gpus == [1]:
                raise BlockingIOError('locked by another queue')
            lease = Lease()
            leases.append(lease)
            return [lease]
        with mock.patch.object(runner, 'inventory', return_value=table({0, 1, 2})), \
             mock.patch.object(runner.common, 'reserve_gpus', side_effect=reserve):
            failure(lambda: runner.admit('auto', 3, manifest), 'Could reserve only 2/3')
            assert all(l.closed for l in leases)
        leases.clear()
        with mock.patch.object(runner, 'inventory', return_value=table({0, 1, 2, 3, 4})), \
             mock.patch.object(runner.common, 'reserve_gpus', side_effect=reserve):
            selected, held, _ = runner.admit('auto', 3, manifest)
            assert selected == [0, 2, 3] and len(held) == 3
            assert not any(l.closed for l in held)
            for l in held:
                l.close()
        leases.clear()
        with mock.patch.object(runner, 'inventory', side_effect=[table({0, 2, 3}), table({0, 2})]), \
             mock.patch.object(runner.common, 'reserve_gpus', side_effect=reserve):
            failure(lambda: runner.admit('auto', 3, manifest), 'allocation changed')
            assert all(l.closed for l in leases)
    with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='2,4'):
        assert runner.gpu_pool('auto', manifest, table(set(range(8)))) == [2, 4]
    local = dict(manifest, packaging_host=runner.socket.gethostname())
    with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'):
        assert runner.gpu_pool('auto', local, table(set(range(8)))) == [0, 1, 2, 3]


def test_fifo(root):
    session = root / 'queue'
    for d in ('configs', 'state'):
        (session / d).mkdir(parents=True)
    matrix = [(1, 2), (1, 4), (1, 6), (2, 2), (2, 6), (3, 2),
              (3, 6), (5, 2), (8, 2), (8, 4), (8, 6)]
    records = [dict(name=f'job{i}', run_dir=str(root / f'job{i}'), review_id='test', status='pending',
                    experiment=dict(name=f'job{i}', task='action', config=dict(
                        MISS_LEN=m, N_LAYER=l, MAX_TRAIN_HOURS=6)))
               for i, (m, l) in enumerate(matrix)]
    runner.atomic_json(session / 'configs/dispatch.json', dict(state=dict(
        review_id='test', gpus=[1, 3, 7], runs=records)))
    active, starts, post = set(), [], []
    class Process:
        pid = 99999999
        def __init__(self, gpu, phase):
            self.gpu, self.phase, self.polls = gpu, phase, 0
        def poll(self):
            self.polls += 1
            if self.phase == 'training':
                return 42 if self.polls >= 2 else None
            active.remove(self.gpu)
            return 0
    def launch(bundle, record, gpu):
        assert gpu not in active and len(active) < 3
        assert record['experiment']['config']['MAX_TRAIN_HOURS'] == 6
        active.add(gpu)
        starts.append((record['name'], gpu))
        path = Path(record['run_dir'])
        (path / 'logs').mkdir(parents=True)
        (path / 'logs/train.log').write_text('[TIMEOUT]\n')
        (path / 'checkpoints').mkdir()
        (path / 'checkpoints/model_resume.pth').write_text('fixture')
        return Process(gpu, 'training'), [io.StringIO()]
    def evaluate(bundle, record, gpu, delivery):
        assert gpu in active
        post.append(record['name'])
        return Process(gpu, 'evaluation'), [io.StringIO()]
    handles = [Lease() for _ in range(3)]
    manifest = dict(confirmed_review_id='test', delivery=dict(required_gpus=3))
    with mock.patch.object(runner, 'launch_training', side_effect=launch), \
         mock.patch.object(runner, 'launch_postprocess', side_effect=evaluate), \
         mock.patch.object(runner.time, 'sleep'):
        assert runner.dispatch(root, manifest, session, handles) == 0
    state = json.loads((session / 'state/status.json').read_text())
    assert [name for name, _ in starts] == [r['name'] for r in records]
    assert starts[:3] == [('job0', 1), ('job1', 3), ('job2', 7)]
    assert len(post) == 11 and all(r['status'] == 'timed_out' for r in state['runs'])
    assert all(h.closed for h in handles) and not active



def test_background_lease_handoff(root):
    bundle = root / 'background bundle'
    bundle.mkdir()
    (bundle / 'manifest.json').write_text('{}')
    (bundle / 'PARAMETERS.md').write_text('fixture')
    manifest = dict(batch_name='bg_fixture', confirmed_review_id='test',
                    delivery=dict(required_gpus=3, output_root=str(root / 'background output')))
    data = dict(experiments=[dict(name='only_fixture', task='action', config=dict(
        MISS_LEN=1, N_LAYER=2, MAX_TRAIN_HOURS=6))])
    handles = []
    lock_paths = [root / f'lease_{i}' for i in range(3)]
    for path in lock_paths:
        handle = path.open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(handle)
    actual_popen = subprocess.Popen
    children = []
    def spawn(command, **kwargs):
        assert kwargs['start_new_session'] and len(kwargs['pass_fds']) == 3
        assert '--lease-fds' in command and '_dispatch' in command
        # A real detached child holds the same file descriptions after parent closes.
        child = actual_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        children.append(child)
        return child
    args = argparse.Namespace(only=None, resume_batch=None, gpus='auto',
                              output_root=None, background=True, timers='{}')
    try:
        with mock.patch.object(runner, 'admit', return_value=([0, 1, 2], handles, table({0, 1, 2}))), \
             mock.patch.object(runner.subprocess, 'Popen', side_effect=spawn):
            assert runner.start(args, bundle, manifest, data) == 0
        assert all(h.closed for h in handles)
        for path in lock_paths:
            with path.open('a') as fresh:
                try:
                    fcntl.flock(fresh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise AssertionError('GPU lease lost across background handoff')
    finally:
        for child in children:
            child.terminate()
            child.wait(timeout=10)
        for h in handles:
            h.close()
    for path in lock_paths:
        with path.open('a') as fresh:
            fcntl.flock(fresh, fcntl.LOCK_EX | fcntl.LOCK_NB)

def test_cpu_resume(root):
    import torch
    spec = dict(batch_name='cpu_remote', concurrency=3,
                delivery=dict(required_gpus=3, output_root=str(root / 'output'),
                              clean_eval_samples=12, clean_eval_seed=123),
                common=dict(P=7, AB_PAIRS=[[1, 1], [2, 3]],
                            D_MODEL=16, N_LAYER=1, N_HEAD=1, MLP_RATIO=2,
                            BATCH_SIZE=8, GRAD_ACCUM_STEPS=1, NUM_TRAIN_SAMPLES=16,
                            NUM_TEST_SAMPLES=8, TRAIN_LEN=6, OOD_LEN=6,
                            EPOCHS=4, EVAL_INTERVAL=1, EARLY_STOP_ACCURACY=2.0,
                            MISSING_PROB=0.1, MISS_LEN=2, FRESH_TEST_PER_EVAL=True,
                            LR=0.001, WEIGHT_DECAY=0, GRAD_CLIP_NORM=1,
                            MAX_TRAIN_HOURS=0, RANDOM_SEED=17996, USE_AMP=False),
                experiments=[dict(name='tiny_action', task='action', config={})])
    plan = pipeline.prepare(spec)
    review = root / 'review.md'
    review.write_text(pipeline.render_review(plan))
    assert pipeline.load_review(review) == plan
    bundle, archive = pipeline.build(review, pipeline.review_id(plan), root / 'bundle with spaces')
    assert archive.is_file() and (bundle / 'package_utils.py').is_file()
    check = subprocess.run(['bash', str(bundle / 'run.sh'), 'check'],
                           env=dict(os.environ, EXPERIMENT_PYTHON=sys.executable),
                           capture_output=True, text=True, timeout=30)
    assert check.returncode == 0, check.stderr
    exp = json.loads((bundle / 'experiments.json').read_text())['experiments'][0]
    def train(directory, experiment, expected_code):
        record = dict(name=experiment['name'], experiment=experiment,
                      run_dir=str(directory), review_id=pipeline.review_id(plan))
        process, streams = runner.launch_training(bundle, record, None)
        code = process.wait(timeout=60)
        for stream in streams:
            stream.close()
        assert code == expected_code, (directory / 'logs/train.err').read_text()
        saved = torch.load(directory / 'checkpoints/model_resume.pth', map_location='cpu', weights_only=False)
        assert {'optimizer_state_dict', 'scheduler_state_dict', 'rng_python', 'rng_torch'} <= saved.keys()
        return record, saved
    first, ckpt = train(root / 'first run', exp, 42)
    assert ckpt['next_epoch'] == 1 and ckpt['scheduler_state_dict']['T_max'] == 4
    # Exercise the structured best format written by the real training engine.
    torch.save({'model_state_dict': ckpt['model_state_dict'], 'config': ckpt['config']},
               root / 'first run/checkpoints/model_best.pth')
    process, streams = runner.launch_postprocess(bundle, first, None, plan['delivery'])
    assert process.wait(timeout=60) == 0, (root / 'first run/logs/evaluation.err').read_text()
    for stream in streams:
        stream.close()
    metrics = json.loads((root / 'first run/metrics/summary.json').read_text())
    assert metrics['clean_num_samples'] == 12 and 0 <= metrics['clean_acc'] <= 1
    assert (root / 'first run/plots/learning_curve.png').is_file()
    previous = root / 'previous batch'
    (previous / 'state').mkdir(parents=True)
    first.update(status='timed_out')
    runner.atomic_json(previous / 'state/status.json', dict(
        status='finished', review_id=pipeline.review_id(plan), runs=[first]))
    manifest = json.loads((bundle / 'manifest.json').read_text())
    data = json.loads((bundle / 'experiments.json').read_text())
    resumed = runner.choose_experiments(data, previous=previous, manifest=manifest)[0]
    # Move package: continued training must use bundled code, not build-machine paths.
    moved = root / 'moved package'
    bundle.rename(moved)
    bundle = moved
    _, saved = train(root / 'second run', resumed, 42)
    assert saved['next_epoch'] == 2 and saved['scheduler_state_dict']['T_max'] == 4
    sys.path.insert(0, str(bundle / 'code/scripts'))
    import action_run_postprocess as postprocess
    assert postprocess.load_full(root / 'second run/checkpoints/model_resume.pth', exp['config'])['next_epoch'] == 2
    bad = dict(exp['config'], LR=0.2)
    failure(lambda: postprocess.load_full(root / 'second run/checkpoints/model_resume.pth', bad), 'base LR')
    # A static early stop on epoch 0 must also save a full checkpoint.
    normal = json.loads(json.dumps(exp))
    normal['config'].update(MAX_TRAIN_HOURS=6, FRESH_TEST_PER_EVAL=False, EARLY_STOP_ACCURACY=0)
    _, saved = train(root / 'normal early stop', normal, 0)
    assert saved['next_epoch'] == 1 and saved['optimizer_state_dict']['state']
    # Full 11-run review input uses only the approved matrix and 6h.
    assert len(list((root / 'first run').iterdir())) == 5


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='remote_action_test_') as tmp:
        root = Path(tmp)
        with contextlib.redirect_stdout(io.StringIO()):
            test_admission()
        print('PASS admission: insufficient cards, lock contention, allocation race, visibility/local limits', flush=True)
        test_fifo(root)
        print('PASS 11 jobs FIFO on three slots; no duplicate/overlapping jobs; per-job 6h', flush=True)
        test_background_lease_handoff(root)
        print('PASS real inherited GPU lease locks survive background detach', flush=True)
        test_cpu_resume(root)
        print('PASS real CPU action: timer save, clean metrics/plot, portable resume, early-stop full state', flush=True)
    print('ALL TESTS PASSED: test_remote_experiment_runner.py', flush=True)
