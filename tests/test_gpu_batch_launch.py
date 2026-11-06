"""CPU-only tests for the experiment-runner GPU batch launcher."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                     '.agents' / 'skills' / 'recursion-experiment-runner' / 'scripts'))
import gpu_batch_launch as launcher


def _snapshot():
    cards = {
        0: {'uuid': 'GPU-0', 'memory_mb': 0, 'utilization': 0},
        1: {'uuid': 'GPU-1', 'memory_mb': 500, 'utilization': 0},
        2: {'uuid': 'GPU-2', 'memory_mb': 0, 'utilization': 80},
        3: {'uuid': 'GPU-3', 'memory_mb': 0, 'utilization': 0},
        4: {'uuid': 'GPU-4', 'memory_mb': 0, 'utilization': 0},
    }
    return cards, {'GPU-3'}


def test_parse_gpu_list():
    assert launcher.parse_gpu_list('4, 2,0') == [4, 2, 0]
    for bad in ('', '0,0', '-1', 'a,2'):
        try:
            launcher.parse_gpu_list(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted invalid GPU list: {bad!r}')


def test_idle_filters_and_order():
    cards, busy = _snapshot()
    assert launcher.free_gpus([4, 3, 2, 1, 0], cards, busy) == [4, 0]
    assert launcher.free_gpus([3, 2, 1], cards, busy) == []
    with mock.patch.object(launcher, 'gpu_snapshot', return_value=(cards, busy)):
        selected = launcher.wait_for_gpus([4, 3, 2, 1, 0], 2, 1)
        assert selected == [4, 0]


def test_batch_shape_and_dry_run():
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / 'batch.json'
        config.write_text(json.dumps({
            'concurrency': 6,
            'experiments': [{'name': 'a'}, {'name': 'b'}],
        }))
        assert launcher.batch_shape(config) == (2, 6)
        argv = [str(config), '--required', '2', '--gpus', '4,0',
                '--base-dir', '/data/cxm/recursion',
                '--model-base-dir', '/data/cxm/models']
        output = io.StringIO()
        with mock.patch.object(launcher, 'gpu_snapshot', return_value=_snapshot()):
            with mock.patch.object(launcher, 'run_batch') as run_batch:
                with contextlib.redirect_stdout(output):
                    assert launcher.main(argv) == 0
        run_batch.assert_not_called()
        text = output.getvalue()
        assert 'json_concurrency=6' in text and 'required_gpus=2' in text
        assert '[DRY-RUN]' in text


def test_execute_pins_exact_cards_and_output_dirs():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / 'batch.json'
        config.write_text('{"concurrency": 6, "experiments": [{"name": "a"}]}')
        log = root / 'batch.log'
        captured = {}

        class FakeProcess:
            pid = 12345

            def wait(self):
                return 0

        def fake_popen(command, **kwargs):
            captured['command'] = command
            captured.update(kwargs)
            return FakeProcess()

        with mock.patch.object(launcher.subprocess, 'Popen', fake_popen):
            rc = launcher.run_batch(config, [4, 0], Path('/data/results'),
                                    Path('/data/models'), log)
        assert rc == 0
        command = captured['command']
        assert command[-4:] == ['--base-dir', '/data/results',
                                '--model-base-dir', '/data/models']
        assert captured['env']['BATCH_RUN_GPUS'] == '4,0'
        assert captured['env']['BATCH_RUN_REQUIRE_GPU'] == '1'
        assert captured['start_new_session'] is True
        assert captured['cwd'] == launcher.REPO_ROOT
        assert log.is_file()


def test_repo_root():
    assert (launcher.REPO_ROOT / 'src' / 'batch_run.py').is_file()


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items())
             if name.startswith('test_') and callable(value)]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'ALL TESTS PASSED: {len(tests)}')
