"""Pure-CPU tests for probe_fillblank_logit_lens.py (no checkpoint/model loaded)."""
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import probe_fillblank_logit_lens as probe


def make_cell(layer, position, mean_prob, top1_rate, k=2):
    return {
        'k': k,
        'layer_index': layer,
        'position': position,
        'n': 32,
        'mean_prob': mean_prob,
        'median_prob': mean_prob,
        'top1_rate': top1_rate,
        'mean_margin': 0.0,
        'mean_rank': 1.0,
    }


def test_legal_positions_and_count_plan():
    expected = {
        0: [0, 1, 2, 5],
        1: [0, 1, 2, 3, 5],
        2: [0, 1, 2, 3, 4, 5],
        3: [1, 2, 3, 4, 5],
        4: [2, 3, 4, 5],
    }
    for k, positions in expected.items():
        assert probe.legal_positions(k) == positions
        assert positions[-1] == 5
        assert positions[:-1] == list(range(max(0, k - 2), min(4, k + 2) + 1))
    for bad_k in (-1, 5):
        try:
            probe.legal_positions(bad_k)
            assert False, 'expected invalid blank position to fail'
        except ValueError:
            pass

    plan = probe.build_plan()
    assert [item['n_cells'] for item in plan] == [16, 20, 24, 20, 16]
    assert [item['n_raw_records'] for item in plan] == [512, 640, 768, 640, 512]
    assert probe.plan_totals(plan) == {
        'n_blanks': 5,
        'n_layers': 4,
        'n_bases': 32,
        'n_cells': 96,
        'n_raw_records': 3072,
    }
    small = probe.plan_totals(probe.build_plan(n_bases=2, n_layers=3))
    assert small['n_cells'] == 72 and small['n_raw_records'] == 144


def test_full_vocab_rank_margin_and_probability():
    logits = torch.tensor([
        [2.0, 3.0, 3.0, 1.0],
        [5.0, 4.0, 4.0, 5.0],
        [0.0, -1.0, 2.0, 1.0],
        [3.0, 3.0, 0.0, -2.0],
    ])
    targets = torch.tensor([0, 1, 2, 0])
    metrics = probe.target_metrics(logits, targets)
    torch.testing.assert_close(metrics['logit'], torch.tensor([2.0, 4.0, 2.0, 3.0]))
    assert metrics['rank'].tolist() == [3, 3, 1, 1]
    # The last row ties another class: strict-greater rank is still 1 and margin is 0.
    torch.testing.assert_close(metrics['margin'], torch.tensor([-1.0, -1.0, 1.0, 0.0]))
    expected_prob = torch.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(metrics['prob'], expected_prob)
    assert math.isclose(float(metrics['prob'][0]), math.exp(2) / torch.exp(logits[0]).sum())


def test_aggregation_metrics():
    raw = []
    for prob, rank, margin in zip([0.1, 0.5, 0.9, 0.7], [2, 1, 1, 1], [-1.0, 0.0, 2.0, 1.0]):
        raw.append({
            'k': 1, 'layer_index': 0, 'position': 3,
            'prob': prob, 'logit': 0.0, 'rank': rank, 'margin': margin,
        })
    cell = probe.aggregate_records(raw)
    assert cell['n'] == 4
    assert math.isclose(cell['mean_prob'], 0.55)
    assert math.isclose(cell['median_prob'], 0.60)
    assert cell['top1_rate'] == 0.75
    assert math.isclose(cell['mean_margin'], 0.50)
    assert cell['mean_rank'] == 1.25


def test_clear_thresholds_are_inclusive_and_conjunctive():
    assert probe.is_clear(make_cell(0, 0, 0.50, 0.90), 'primary')
    assert not probe.is_clear(make_cell(0, 0, 0.4999, 1.0), 'primary')
    assert not probe.is_clear(make_cell(0, 0, 1.0, 0.8999), 'primary')
    assert probe.is_clear(make_cell(0, 0, 0.90, 0.95), 'strict')
    assert not probe.is_clear(make_cell(0, 0, 0.8999, 1.0), 'strict')
    assert not probe.is_clear(make_cell(0, 0, 1.0, 0.9499), 'strict')


def test_earliest_layer_position_order_and_highest_probability():
    cells = [
        make_cell(2, 0, 0.99, 1.00),
        make_cell(0, 5, 0.70, 0.90),
        make_cell(0, 2, 0.80, 1.00),
        make_cell(3, 4, 0.95, 0.95),
        make_cell(0, 1, 0.40, 1.00),
        make_cell(0, 0, 0.60, 0.90),
    ]
    primary = probe.clear_summary(cells, 2, 'primary')
    earliest = primary['earliest']
    assert (earliest['layer_index'], earliest['position']) == (0, 0)
    # EOS follows ordinary input positions within L1, even with a larger probability.
    assert [(c['layer_index'], c['position']) for c in primary['clear_cells']] == [
        (0, 0), (0, 2), (0, 5), (2, 0), (3, 4),
    ]
    assert (primary['highest_mean_prob']['layer_index'],
            primary['highest_mean_prob']['position']) == (2, 0)

    strict = probe.clear_summary(cells, 2, 'strict')
    assert (strict['earliest']['layer_index'], strict['earliest']['position']) == (2, 0)
    assert len(strict['clear_cells']) == 2
    none = probe.clear_summary([make_cell(0, 0, 0.1, 0.0)], 2, 'primary')
    assert none == {'earliest': None, 'highest_mean_prob': None, 'clear_cells': []}


def test_input_construction_and_cli_defaults():
    windows = torch.arange(15).reshape(3, 5)
    original = windows.clone()
    inputs = probe.make_blank_inputs(windows, 2)
    assert inputs.shape == (3, 6)
    assert inputs[:, 2].tolist() == [127, 127, 127]
    assert inputs[:, 5].tolist() == [128, 128, 128]
    assert inputs[:, 4].tolist() == windows[:, 4].tolist()
    assert torch.equal(windows, original)

    args = probe.parse_args(['--dry-run'])
    assert args.device == 'cpu' and args.threads == 16 and args.batch_size == 32
    assert args.base_data.endswith('fillblank_len5_all_blank_linear_a25b56.json')
    assert args.out.endswith('fillblank_len5_logit_lens_a25b56.json')
    assert args.plot.endswith('fillblank_len5_logit_lens_a25b56.png')
    assert args.ckpt.endswith('seed42_best.pth') and args.dry_run


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items())
             if name.startswith('test_') and callable(value)]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'\nALL TESTS PASSED: {len(tests)}/{len(tests)} (pure CPU; no model loaded)')
