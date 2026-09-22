"""CPU-only tests for polling, GPU locks and single-card queue dispatch."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile
import time
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import experiment_package_runner as runner


def arguments(output, background=False):
    return argparse.Namespace(output=output, only=None, cpu=False, gpus='2,3,0,1',
        wait_for_idle=True, poll_seconds=1, background=background,
        prepared_output=False, timers='{"first":12,"second":12}')


def fixture():
    manifest = dict(confirmed_review_id='cpu_queue_test', packaging_host=socket.gethostname(),
                    local_gpu_allowlist=[0, 1, 2, 3])
    data = dict(concurrency=2, experiments=[
        dict(name=name, task='tetranacci', config=dict(MAX_TRAIN_HOURS=12))
        for name in ('first', 'second')])
    return manifest, data


def snapshot(free=()):
    # GPU 4 is always idle, but is outside the authorized pool.
    return {gpu: dict(uuid='TEST-' + str(gpu),
            memory_mib=1 if gpu in free or gpu == 4 else 22000,
            utilization=0 if gpu in free or gpu == 4 else 100,
            has_compute_process=gpu not in free and gpu != 4)
            for gpu in range(5)}


def run_isolated(args, root, query, reserve, launch):
    manifest, data = fixture()
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    captured = io.StringIO()
    try:
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3'), \
             mock.patch.object(runner, 'gpu_snapshot', side_effect=query), \
             mock.patch.object(runner, 'reserve_gpus', side_effect=reserve), \
             mock.patch.object(runner, 'launch_one', side_effect=launch), \
             contextlib.redirect_stdout(captured):
            result = runner.run(args, root, manifest, data)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return result, json.loads((args.output / 'state.json').read_text()), captured.getvalue()


def test_wait_then_dispatch_independently(root):
    checks, launched, leases, reservations = [], [], [], []

    def query():
        checks.append(time.monotonic())
        if len(checks) == 1:
            return snapshot()
        return snapshot((2, 3) if len(checks) >= 4 else (2,))

    class Lease:
        def __init__(self, gpu):
            self.gpu, self.closed = gpu, False
        def close(self):
            self.closed = True

    def reserve(gpus, manifest):
        gpu = gpus[0]
        reservations.append(gpu)
        if len(reservations) == 1:
            raise BlockingIOError('another package still holds this GPU lock')
        handle = Lease(gpu)
        leases.append(handle)
        return [handle]

    class FakeTrainer:
        pid = 99999999
        def poll(self):
            return 0 if len(launched) == 2 else None

    def launch(bundle, output, exp, gpu):
        launched.append((exp['name'], gpu, len(checks), exp['config']['MAX_TRAIN_HOURS']))
        path = runner.log_path(output, exp['name'])
        path.parent.mkdir(parents=True)
        path.write_text('--- Return code: 0 ---\n')
        return FakeTrainer(), io.StringIO()

    args = arguments(root / 'dispatch')
    code, state, log = run_isolated(args, root, query, reserve, launch)
    assert code == 0 and state['status'] == 'finished', state
    assert launched == [('first', 2, 3, 12), ('second', 3, 4, 12)], launched
    assert reservations == [2, 2, 3]
    assert all(lease.closed for lease in leases)
    assert all(record['status'] == 'completed' for record in state['runs'])
    assert state['poll_count'] == 4 and state['next_gpu_check_utc'] is None
    assert all(b - a >= 0.95 for a, b in zip(checks, checks[1:]))
    assert 'GPU_BUSY_ON_RECHECK' in log


def test_probe_failure_and_cancel_before_launch(root):
    calls = []

    def query():
        calls.append(time.monotonic())
        if len(calls) == 1:
            raise FileNotFoundError('simulated nvidia-smi unavailable')
        # Stop arriving while the poll is in progress must prevent dispatch.
        os.kill(os.getpid(), signal.SIGTERM)
        return snapshot((2, 3))

    def forbidden(*args, **kwargs):
        raise AssertionError('Must not reserve or start a GPU on failure/cancellation')

    args = arguments(root / 'cancel')
    code, state, log = run_isolated(args, root, query, forbidden, forbidden)
    assert code == 0 and state['status'] == 'stopped'
    assert all(record['status'] == 'cancelled' for record in state['runs'])
    assert state['poll_count'] == 2
    assert 'GPU_CHECK_FAILED' in log


def test_background_forwards_monitor_flags(root):
    args = arguments(root / 'background', background=True)
    args.poll_seconds = 300
    manifest, data = fixture()
    commands = []

    def spawn(command, **kwargs):
        commands.append((command, kwargs))
        return argparse.Namespace(pid=os.getpid())

    with mock.patch.object(runner.subprocess, 'Popen', side_effect=spawn):
        assert runner.run(args, root, manifest, data) == 0
    command, kwargs = commands[0]
    assert '--wait-for-idle' in command
    assert command[command.index('--poll-seconds') + 1] == '300'
    assert command[command.index('--gpus') + 1] == '2,3,0,1'
    assert kwargs['start_new_session'] is True
    assert (args.output / 'controller_launch.json').is_file()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='experiment_queue_test_') as folder:
        root = Path(folder)
        test_wait_then_dispatch_independently(root)
        print('PASS busy GPUs, reservation race, independent dispatch, no duplicate launch', flush=True)
        test_probe_failure_and_cancel_before_launch(root)
        print('PASS failed GPU probe and cancellation fail closed', flush=True)
        test_background_forwards_monitor_flags(root)
        print('PASS detached controller preserves 300-second polling and GPU pool', flush=True)
    print('ALL TESTS PASSED: test_experiment_queue.py', flush=True)
