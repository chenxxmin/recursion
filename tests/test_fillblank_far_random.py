"""Pure-CPU tests for probe_fillblank_len8to64_far_random.py (no checkpoints)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import probe_fillblank_len8to64_far_random as probe


def make_summary(n, counts=None, output_changes=0, nonnumeric=0):
    counts = counts or {}
    candidates = probe.candidate_values(25, 56, probe.P)
    fits = {}
    for name in probe.CANDIDATE_NAMES:
        count = counts.get(name, 0)
        fits[name] = {
            'k': candidates[name],
            'exact_count': count,
            'exact_match': count / n,
        }
    return {
        'n': n,
        'output_change_count': output_changes,
        'output_change_rate': output_changes / n,
        'nonnumeric_count': nonnumeric,
        'nonnumeric_output_rate': nonnumeric / n,
        'fits': fits,
    }


def test_far_positions_order_and_static_totals():
    assert probe.far_relative_positions(8, 0) == [
        (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
    ]
    assert probe.far_relative_positions(8, 3) == [(-3, 0), (3, 6), (4, 7)]
    assert probe.far_relative_positions(8, 4) == [(-3, 1), (3, 7), (-4, 0)]
    assert probe.far_relative_positions(8, 7) == [
        (-3, 4), (-4, 3), (-5, 2), (-6, 1), (-7, 0),
    ]
    for length in probe.LENGTHS:
        n_cells = 0
        for k in range(length):
            pairs = probe.far_relative_positions(length, k)
            n_cells += len(pairs)
            assert pairs == sorted(pairs, key=lambda x: (abs(x[0]), x[0]))
            assert len(pairs) == len(set(pairs))
            for relative, position in pairs:
                assert relative == position - k
                assert 0 <= position < length and abs(relative) >= 3
        assert n_cells == (length - 2) * (length - 3)

    plan = probe.build_plan()
    totals = probe.plan_totals(plan)
    assert [item['n_far_cells'] for item in plan] == [
        30, 182, 462, 870, 1406, 2070, 2862, 3782,
    ]
    assert totals == {
        'n_lengths': 8,
        'n_blanks': 288,
        'n_far_cells': 11664,
        'n_baseline_predictions': 36864,
        'n_perturbations': 746496,
    }
    assert all(item['n_per_cell'] == 64 for item in plan)
    assert sum(item['n_perturbations'] for item in plan) == 11664 * 16 * 4


def test_anomaly_detection_separates_low_zero_from_high_nonzero():
    # At the exact thresholds there is no anomaly: zero == .95, nonzero < .10.
    normal = make_summary(100, {'0': 95, 'b': 9, 'a': 8})
    assert not probe.is_anomaly(normal)
    assert probe.anomaly_record(8, 3, 'interior', -3, normal) is None

    # Low D=0 fit triggers independently of all nonzero candidates being low.
    low_zero = make_summary(100, {'0': 94, 'b': 1, 'a': 2})
    assert probe.is_anomaly(low_zero)
    probe.add_fit_metrics(low_zero)
    assert low_zero['zero_rate'] == 0.94
    assert low_zero['best_nonzero_rate'] == 0.02

    # A nonzero candidate at exactly .10 triggers even when D=0 is perfect.
    high_nonzero = make_summary(100, {'0': 100, 'b': 6, 'a': 10})
    assert probe.is_anomaly(high_nonzero)
    anomaly = probe.anomaly_record(16, 4, 'interior', 3, high_nonzero)
    assert anomaly['zero_rate'] == 1.0
    assert anomaly['best_nonzero'] == 'a'
    assert anomaly['best_nonzero_rate'] == 0.10

    # The high D=0 candidate itself never becomes best_nonzero.
    zero_only = make_summary(100, {'0': 100})
    assert not probe.is_anomaly(zero_only)
    assert probe.best_nonzero(zero_only) == ('b', 0.0)


def test_raw_is_embedded_only_for_anomalies_and_index_is_complete():
    normal = make_summary(64, {'0': 64, 'b': 6})
    normal['raw'] = {'stale': True}
    assert not probe.attach_raw_if_anomaly(normal, {'D': [0]})
    assert 'raw' not in normal

    raw = {'base_row': [2] * 4, 'd': [1, 2, 3, 4], 'D': [0, 1, 2, 3]}
    low_zero = make_summary(64, {'0': 60, 'b': 1}, output_changes=8, nonnumeric=2)
    assert probe.attach_raw_if_anomaly(low_zero, raw)
    assert low_zero['raw'] is raw
    record = probe.anomaly_record(8, 3, 'interior', -3, low_zero)
    assert set(record) == {
        'L', 'k', 'category', 'relative', 'zero_rate', 'best_nonzero',
        'best_nonzero_rate', 'output_change_rate', 'raw_included',
    }
    assert record == {
        'L': 8,
        'k': 3,
        'category': 'interior',
        'relative': -3,
        'zero_rate': 60 / 64,
        'best_nonzero': 'b',
        'best_nonzero_rate': 1 / 64,
        'output_change_rate': 8 / 64,
        'raw_included': True,
    }

    high_nonzero = make_summary(64, {'0': 64, 'a': 7})
    assert probe.attach_raw_if_anomaly(high_nonzero, raw)
    assert high_nonzero['raw'] is raw


def test_aggregation_is_count_weighted_and_has_zero_best_nonzero():
    small = make_summary(100, {'0': 90, 'b': 10, 'a': 20}, output_changes=10)
    large = make_summary(
        300, {'0': 240, 'b': 150, 'a': 30}, output_changes=150, nonnumeric=30)
    result = probe.aggregate_cells([small, large])
    assert result['n'] == 400 and result['n_cells'] == 2
    assert result['fits']['b']['exact_count'] == 160
    assert result['fits']['b']['exact_match'] == 0.4  # not mean(.10, .50)
    assert result['fits']['a']['exact_count'] == 50
    assert result['fits']['a']['exact_match'] == 0.125
    assert result['zero_exact_count'] == 330
    assert result['zero_rate'] == 330 / 400
    assert result['best_nonzero'] == 'b'
    assert result['best_nonzero_exact_count'] == 160
    assert result['best_nonzero_rate'] == 0.4
    assert result['output_change_count'] == 160
    assert result['output_change_rate'] == 0.4
    assert result['nonnumeric_count'] == 30
    assert result['nonnumeric_output_rate'] == 0.075

    by_length = {
        '8': {
            'by_blank': {
                '0': {'category': 'left_boundary', 'by_relative': {'+3': small}},
                '3': {'category': 'interior', 'by_relative': {'-3': large}},
            },
        },
    }
    grouped = probe.build_aggregates(by_length)
    assert set(grouped) == {
        'by_length', 'by_abs_distance', 'by_category', 'by_category_abs_distance',
    }
    assert grouped['by_length']['8']['n'] == 400
    assert grouped['by_abs_distance']['3']['zero_exact_count'] == 330
    assert grouped['by_abs_distance']['3']['best_nonzero'] == 'b'
    assert grouped['by_category']['left_boundary']['n'] == 100
    assert grouped['by_category']['interior']['n'] == 300
    assert grouped['by_category_abs_distance']['left_boundary']['3']['n'] == 100
    assert grouped['by_category_abs_distance']['interior']['3']['n'] == 300


def test_cli_defaults_reuse_and_changed_sampling_totals():
    args = probe.parse_args(['--dry-run'])
    assert args.device == 'cpu'
    assert args.threads == 16 and args.batch_size == 512
    assert args.n_bases == 16 and args.n_deltas == 4
    assert args.out.endswith('fillblank_len8to64_far_random_a25b56.json')
    assert probe.NONZERO_CANDIDATE_NAMES == tuple(
        name for name in probe.CANDIDATE_NAMES if name != '0')
    assert len(probe.candidate_values(25, 56, probe.P)) == 7
    path = probe.checkpoint_path(args.ckpt_template, 16)
    assert 'len16_wd2_initLen15' in path and '{L' not in path

    changed = probe.plan_totals(probe.build_plan(n_bases=2, n_deltas=3))
    assert changed['n_far_cells'] == 11664
    assert changed['n_perturbations'] == 11664 * 2 * 3
    # Thresholds are module constants and do not depend on sampling arguments.
    assert probe.ZERO_MATCH_THRESHOLD == 0.95
    assert probe.BEST_NONZERO_THRESHOLD == 0.10


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items())
             if name.startswith('test_') and callable(value)]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'\nALL TESTS PASSED: {len(tests)}/{len(tests)} (pure CPU; no model loaded)')
