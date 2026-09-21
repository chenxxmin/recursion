"""End-to-end sanity checks for the tetranacci training pipeline.

Verifies the exact objects seen by train_epoch after dataset generation,
collation, target shifting, and default loss-mask construction.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from experiment import _prepare_single_recurrence
from training import _unpack_batch, _default_loss_mask


def _tiny_tetranacci_config(coeffs):
    a, b, c, d = coeffs
    return {
        'main': {
            'TASK': 'tetranacci',
            'P': 7,
            'A': a,
            'B': b,
            'C': c,
            'D': d,
            'D_MODEL': 32,
            'N_HEAD': 1,
            'N_LAYER': 1,
            'BATCH_SIZE': 8,
            'TRAIN_LEN': 9,
            'OOD_LEN': 10,
            'DROPOUT': 0.0,
            'ENTROPY_PENALTY_WEIGHT': 0.0,
            'MAX_UNIQUE_RATIO': 0.7,
            'USE_LEARNABLE_PE': False,
            'MLP_RATIO': 4,
            'NUM_MASK': 3,
            'NUM_TRAIN_SAMPLES': 32,
            'NUM_TEST_SAMPLES': 8,
            'DATA_MODE': 'sampled_fresh_test',
            'RANDOM_SEED': 0,
            'MISSING_PROB': 0.0,
        }
    }


def _check_tetranacci_pipeline(coeffs):
    ctx = _prepare_single_recurrence(_tiny_tetranacci_config(coeffs), 'tetranacci')
    batch = next(iter(ctx['train_loader']))

    x, loss_mask, kwargs, _, targets_override = _unpack_batch(
        batch, 'cpu', ctx['extra_kwargs_fn'])

    assert kwargs == {}
    assert targets_override is None
    assert loss_mask is None  # plain batch; train_epoch creates the default mask

    targets = x[:, 1:]
    loss_mask = _default_loss_mask(
        x.size(0), targets.size(1), ctx['num_mask'], device='cpu')

    # target index 0,1,2 correspond to x1,x2,x3 and must be ignored.
    assert torch.all(loss_mask[:, :3] == 0)
    # target index 3 corresponds to x4, the first value determined by order-4 recurrence.
    assert torch.all(loss_mask[:, 3:] == 1)

    p = ctx['p']
    a, b, c, d = coeffs
    for row in x.tolist():
        for t in range(4, len(row)):
            expected = (
                a * row[t - 1]
                + b * row[t - 2]
                + c * row[t - 3]
                + d * row[t - 4]
            ) % p
            assert row[t] == expected

    # Verify every position that contributes to the actual training loss is a
    # recurrence-generated target aligned with the model's next-token logits.
    for bidx, row in enumerate(x.tolist()):
        for target_idx in range(targets.size(1)):
            if loss_mask[bidx, target_idx] <= 0:
                continue
            t = target_idx + 1
            assert t >= 4
            expected = (
                a * row[t - 1]
                + b * row[t - 2]
                + c * row[t - 3]
                + d * row[t - 4]
            ) % p
            assert targets[bidx, target_idx].item() == expected


def test_tetranacci_a1b2c4d8_pipeline():
    _check_tetranacci_pipeline((1, 2, 4, 8))


def test_tetranacci_a1b3c5d7_pipeline():
    _check_tetranacci_pipeline((1, 3, 5, 7))


if __name__ == '__main__':
    test_tetranacci_a1b2c4d8_pipeline()
    test_tetranacci_a1b3c5d7_pipeline()
    print('ALL TESTS PASSED: test_tetranacci_pipeline.py')
