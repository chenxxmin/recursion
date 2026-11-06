"""CPU tests of formula union, ambiguity and exhaustive raw-record auditing."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_fillblank_local_routes as analysis
from probe_fillblank_len8_local_exhaustive import compare_historical


def raw_with_slope(slope, baseline=37, p=127):
    deltas = list(range(1, p))
    changes = [slope * d % p for d in deltas]
    outputs = [(baseline + change) % p for change in changes]
    return {'base_row': [11] * (p - 1), 'd': deltas, 'y0': [baseline] * (p - 1),
            'y1': outputs, 'full1': outputs.copy(), 'D': changes}


def test_pure_response_and_ambiguous_zero_counted_once():
    a = analysis.summarize_cell(raw_with_slope(25), 3, 2, 8, [11])
    assert a['signature_counts'] == {'F': 126} and a['coverage'] == 1
    assert a['n_base_cells_one_slope_all_deltas'] == 1
    b = analysis.summarize_cell(raw_with_slope(0), 3, 5, 8, [11])
    assert b['signature_counts'] == {'F|M': 126}
    assert b['covered_count'] == b['ambiguous_count'] == 126
    assert b['route_match_counts']['F'] == b['route_match_counts']['M'] == 126
    combined = analysis.combine([a, b])
    assert combined['covered_count'] == combined['n'] == 252
    assert combined['changed_coverage'] == 1


def test_other_nonnumeric_and_raw_consistency():
    raw = raw_with_slope(13)
    raw['full1'][0] = 127
    a = analysis.summarize_cell(raw, 3, 2, 8, [11])
    assert a['signature_counts'] == {'other': 1, 'M': 125}
    assert a['nonnumeric_count'] == a['other_count'] == 1
    assert a['other_D_histogram_numeric_only']['n'] == 0
    b = analysis.summarize_cell(raw_with_slope(7), 3, 2, 8, [11])
    assert b['other_count'] == 126 and b['coverage'] == 0
    assert b['other_effective_slope_histogram_numeric_only']['counts'] == {7: 126}
    raw['D'][0] += 1
    try:
        analysis.summarize_cell(raw, 3, 2, 8, [11])
    except ValueError:
        pass
    else:
        raise AssertionError('inconsistent D must fail')


def test_same_base_same_position_multiple_responses():
    raw = raw_with_slope(25)
    for i, d in enumerate(raw['d']):
        if d % 2 == 0:
            raw['D'][i] = 13 * d % 127
            raw['y1'][i] = raw['full1'][i] = (raw['y0'][i] + raw['D'][i]) % 127
    result = analysis.summarize_cell(raw, 3, 2, 8, [11])
    assert result['signature_counts'] == {'F': 63, 'M': 63}
    assert result['coverage'] == 1  # Union coverage is NOT fixed-formula consistency.
    assert result['n_base_cells_multiple_compatible_slopes'] == 1
    assert result['n_base_cells_one_slope_all_deltas'] == 0


def test_boundaries_and_historical_lookup():
    assert set(analysis.routes(0, 8)) == {'R'}
    assert set(analysis.routes(1, 8)) == {'M', 'R'}
    assert set(analysis.routes(6, 8)) == {'F', 'M'}
    assert set(analysis.routes(7, 8)) == {'F'}
    # Boundary zero response does not get credited to an unavailable formula.
    a = analysis.summarize_cell(raw_with_slope(0), 0, 1, 8, [11])
    assert a['other_count'] == 126
    raw = raw_with_slope(25)
    old = {key: [value[i] for i in [125, 1, 10]] for key, value in raw.items()}
    assert compare_historical(raw, old)['n_mismatched'] == 0
    old['full1'][1] = 128
    assert compare_historical(raw, old)['n_mismatched'] == 1


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
