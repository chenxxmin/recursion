"""A failed rolling-checkpoint write must not destroy the last usable state."""
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from training import _atomic_torch_save


def test_failed_write_preserves_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'latest.pth'
        torch.save({'next_epoch': 10, 'weight': torch.tensor([1.0])}, path)
        original = path.read_bytes()

        def fail_save(state, temporary_path):
            Path(temporary_path).write_bytes(b'partial checkpoint')
            raise RuntimeError('simulated filesystem write failure')

        with patch('training.torch.save', side_effect=fail_save) as save, \
                patch('training.time.sleep'):
            try:
                _atomic_torch_save({'next_epoch': 11}, path)
            except RuntimeError as exc:
                assert 'simulated filesystem' in str(exc)
            else:
                raise AssertionError('persistent write failure must propagate')
            assert save.call_count == 3

        assert path.read_bytes() == original
        assert torch.load(path, weights_only=True)['next_epoch'] == 10
        assert list(Path(directory).iterdir()) == [path]


def test_transient_failure_recovers():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'latest.pth'
        torch.save({'next_epoch': 10}, path)
        real_save = torch.save
        calls = 0

        def flaky_save(state, temporary_path):
            nonlocal calls
            calls += 1
            assert torch.load(path, weights_only=True)['next_epoch'] == 10
            if calls == 1:
                Path(temporary_path).write_bytes(b'partial checkpoint')
                raise OSError('simulated transient I/O failure')
            real_save(state, temporary_path)

        with patch('training.torch.save', side_effect=flaky_save), \
                patch('training.time.sleep'):
            _atomic_torch_save({'next_epoch': 11, 'weight': torch.tensor([2.0])}, path)

        result = torch.load(path, weights_only=True)
        assert calls == 2 and result['next_epoch'] == 11
        assert torch.equal(result['weight'], torch.tensor([2.0]))
        assert list(Path(directory).iterdir()) == [path]


if __name__ == '__main__':
    test_failed_write_preserves_checkpoint()
    test_transient_failure_recovers()
    print('ALL TESTS PASSED: test_checkpoint_atomic.py')
