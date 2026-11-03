"""Pure-CPU tests for probe_fillblank_len8to64_stride8_random.py (no checkpoints)."""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import probe_fillblank_len8to64_stride8_random as probe


def make_summary(n, counts=None, position=None, output_changes=0, nonnumeric=0):
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
        'position': position,
        'n': n,
        'output_change_count': output_changes,
        'output_change_rate': output_changes / n,
        'nonnumeric_count': nonnumeric,
        'nonnumeric_output_rate': nonnumeric / n,
        'fits': fits,
    }


def test_categories_and_legal_relatives():
    expected_len5 = [
        'left_boundary', 'left_inner', 'interior', 'right_inner', 'right_boundary',
    ]
    assert [probe.blank_category(k, 5) for k in range(5)] == expected_len5
    assert [probe.blank_category(k, 8) for k in range(8)] == [
        'left_boundary', 'left_inner', 'interior', 'interior',
        'interior', 'interior', 'right_inner', 'right_boundary',
    ]
    assert probe.legal_relative_positions(8, 0) == [(1, 1), (2, 2)]
    assert probe.legal_relative_positions(8, 1) == [(-1, 0), (1, 2), (2, 3)]
    assert probe.legal_relative_positions(8, 3) == [
        (-2, 1), (-1, 2), (1, 4), (2, 5),
    ]
    assert probe.legal_relative_positions(8, 6) == [(-2, 4), (-1, 5), (1, 7)]
    assert probe.legal_relative_positions(8, 7) == [(-2, 5), (-1, 6)]
    for length in probe.LENGTHS:
        assert sum(len(probe.legal_relative_positions(length, k))
                   for k in range(length)) == 4 * length - 6


def test_random_deltas_are_deterministic_unique_and_local():
    args = (16, 3, -2, 17)
    first = probe.sample_deltas(*args)
    second = probe.sample_deltas(*args)
    assert first == second
    assert len(first) == 8 and len(set(first)) == 8
    assert all(1 <= d <= 126 for d in first)
    assert first != probe.sample_deltas(16, 3, -2, 18)

    # A local Random instance must not consume or reseed global randomness.
    random.seed(12345)
    state = random.getstate()
    probe.sample_deltas(*args)
    assert random.getstate() == state
    assert set(probe.sample_deltas(*args, n_deltas=126)) == set(range(1, 127))

    selected1 = probe.select_base_rows(list(range(50)), 16, 3)
    selected2 = probe.select_base_rows(list(range(50)), 16, 3)
    assert selected1 == selected2 and len(set(selected1)) == 32
    try:
        probe.select_base_rows(list(range(31)), 16, 3)
        assert False, 'expected insufficient-base error'
    except RuntimeError:
        pass


def test_aggregation_uses_counts_not_average_of_percentages():
    small = make_summary(100, {'b': 10, 'a': 20}, output_changes=10, nonnumeric=0)
    large = make_summary(300, {'b': 150, 'a': 30}, output_changes=150, nonnumeric=30)
    result = probe.aggregate_cells([small, large])
    assert result['n'] == 400 and result['n_cells'] == 2
    assert result['fits']['b']['exact_count'] == 160
    assert result['fits']['b']['exact_match'] == 0.4  # not (0.1 + 0.5) / 2
    assert result['fits']['a']['exact_count'] == 50
    assert result['fits']['a']['exact_match'] == 0.125
    assert result['output_change_count'] == 160
    assert result['output_change_rate'] == 0.4
    assert result['nonnumeric_count'] == 30
    assert result['nonnumeric_output_rate'] == 0.075

    # Existing len5 summaries have rates but not explicit output/nonnumeric counts.
    old_small = dict(small)
    old_large = dict(large)
    for old in (old_small, old_large):
        old.pop('output_change_count')
        old.pop('nonnumeric_count')
    old_result = probe.aggregate_cells([old_small, old_large])
    assert old_result['output_change_count'] == 160
    assert old_result['nonnumeric_count'] == 30


def test_len5_reference_mapping_and_filtering():
    data = {'candidate_values': probe.candidate_values(25, 56, probe.P), 'by_blank': {}}
    originals = {}
    for k in range(5):
        summaries = {}
        for j in range(5):
            if j == k:
                continue
            n = 1000 + 10 * k + j
            summary = make_summary(n, {'b': 10 * k + j, 'a': j + 1}, position=j,
                                   output_changes=j, nonnumeric=k)
            summaries[f'x{j}'] = summary
            originals[k, j] = summary
        data['by_blank'][f'x{k}'] = {'summaries': summaries}

    reference = probe.build_len5_reference(data, source='synthetic-len5.json')
    assert reference['source'] == 'synthetic-len5.json'
    assert reference['category_map'] == {
        'x0': 'left_boundary', 'x1': 'left_inner', 'x2': 'interior',
        'x3': 'right_inner', 'x4': 'right_boundary',
    }
    aggregate = reference['by_length']['5']
    assert reference['all'] == aggregate
    expected_relatives = {
        'left_boundary': {'+1', '+2'},
        'left_inner': {'-1', '+1', '+2'},
        'interior': {'-2', '-1', '+1', '+2'},
        'right_inner': {'-2', '-1', '+1'},
        'right_boundary': {'-2', '-1'},
    }
    assert set(aggregate['by_category']) == set(expected_relatives)
    total_cells = 0
    for category, relatives in expected_relatives.items():
        actual = aggregate['by_category'][category]['by_relative']
        assert set(actual) == relatives
        total_cells += len(actual)
    assert total_cells == 14  # x3/x4 at blank x0 and other |r|>2 entries are excluded

    cell = aggregate['by_category']['left_boundary']['by_relative']['+1']
    assert cell['n'] == originals[0, 1]['n']
    assert cell['fits']['b']['exact_count'] == originals[0, 1]['fits']['b']['exact_count']
    cell = aggregate['by_category']['right_boundary']['by_relative']['-2']
    assert cell['n'] == originals[4, 2]['n']
    assert cell['fits']['a']['exact_count'] == originals[4, 2]['fits']['a']['exact_count']


def test_dominant_comparison_and_fit_vector_mae():
    len5 = make_summary(100, {'b': 25, 'a': 50})
    new = make_summary(100, {'b': 75, 'a': 25})
    row = probe.compare_cell(len5, new)
    assert row['len5_best'] == 'a' and row['len5_rate'] == 0.5
    assert row['new_best'] == 'b' and row['new_rate'] == 0.75
    assert row['dominant_match'] is False
    expected_mae = (0.5 + 0.25) * 100 / 7
    assert abs(row['fit_vector_mae_pp'] - expected_mae) < 1e-12
    identical = probe.compare_cell(len5, len5)
    assert identical['dominant_match'] is True
    assert identical['fit_vector_mae_pp'] == 0.0

    def wrapped(summary):
        return {'by_category': {'left_boundary': {'by_relative': {'+1': summary}}}}

    compared = probe.compare_aggregates(wrapped(new), wrapped(len5))
    assert compared['summary']['dominant_matches'] == 0
    assert compared['summary']['total'] == 1
    assert compared['summary']['mean_mae_pp'] == expected_mae
    category = compared['by_category']['left_boundary']
    assert category['summary']['total'] == 1
    assert category['by_relative']['+1']['new_best'] == 'b'


def test_static_plan_totals_and_cli_defaults():
    plan = probe.build_plan()
    totals = probe.plan_totals(plan)
    assert totals == {
        'n_lengths': 8,
        'n_blanks': 288,
        'n_blank_relative': 1104,
        'n_baseline_predictions': 36864,
        'n_perturbations': 282624,
    }
    assert len(plan) == 8
    for item in plan:
        assert item['n_perturbations'] == item['n_blank_relative'] * 32 * 8
    args = probe.parse_args(['--dry-run'])
    assert args.device == 'cpu'
    assert args.threads == 16 and args.batch_size == 512
    assert args.n_bases == 32 and args.n_deltas == 8
    path = probe.checkpoint_path(args.ckpt_template, 16)
    assert 'len16_wd2_initLen15' in path
    assert '{L' not in path


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items())
             if name.startswith('test_') and callable(value)]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'\nALL TESTS PASSED: {len(tests)}/{len(tests)} (pure CPU; no model loaded)')
