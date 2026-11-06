"""Pure-CPU tests for dynamic len5 fill_blank activation patching (no checkpoint)."""
import contextlib
import io
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import models
from activation_patch import capture_hidden, patched_logits
import probe_fillblank_x0_activation_patch as probe


def make_model(seed=20261104):
    torch.manual_seed(seed)
    return models.FibonacciTransformer(
        p=127, d_model=16, n_head=2, n_layer=4, block_size=8,
        dropout=0.0, mlp_ratio=2, vocab_size=129).eval()


def make_windows():
    return torch.tensor([
        [13, 1, 118, 85, 97],
        [30, 114, 85, 0, 61],
        [66, 22, 55, 67, 56],
    ], dtype=torch.long)


def clean_setup(model, blank=0):
    windows = make_windows()
    inputs = probe.make_blank_inputs(windows, blank, blank_token=127, eos_token=128)
    cache, logits = capture_hidden(model, inputs)
    return windows, inputs, cache, logits


def test_operand_positions_and_dynamic_plan_counts():
    expected = {
        0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3, 4],
        3: [1, 2, 4], 4: [2, 3],
    }
    assert {k: probe.operand_positions(k) for k in range(5)} == expected
    full_expected = {
        0: (8064, 201600, 50, 1600),
        1: (12096, 302400, 75, 2400),
        2: (16128, 403200, 100, 3200),
        3: (12096, 302400, 75, 2400),
        4: (8064, 201600, 50, 1600),
    }
    for k in range(5):
        assert probe.patch_positions(k) == [0, 1, 2, 3, 4]
        assert k not in probe.operand_positions(k)
        assert k in probe.patch_positions(k)
        counts = probe.plan_counts(list(range(32)), list(range(1, 127)), k)
        assert counts['clean'] == 32
        actual = (counts['corrupted'], counts['patched'],
                  counts['aggregate_cells'], counts['per_base_cells'])
        assert actual == full_expected[k]
        smoke = probe.plan_counts([0, 1], [1, 126], k)
        n_sources = len(expected[k])
        assert smoke['corrupted'] == 4 * n_sources
        assert smoke['patched'] == 100 * n_sources
        assert smoke['aggregate_cells'] == 25 * n_sources
        assert smoke['per_base_cells'] == 50 * n_sources


def test_single_token_corruption_alignment_and_target_preservation():
    windows = make_windows()
    original = windows.clone()
    for k in range(5):
        clean = probe.make_blank_inputs(windows, k, blank_token=127, eos_token=128)
        source = probe.operand_positions(k)[0]
        corrupted = probe.make_corrupted_inputs(clean[0:1], source, [1, 126, 7], 127)
        assert corrupted.shape == (3, 6)
        for row, d in enumerate([1, 126, 7]):
            expected = clean[0].clone()
            expected[source] = (expected[source] + d) % 127
            assert torch.equal(corrupted[row], expected)
            assert int((corrupted[row] != clean[0]).sum()) == 1
            assert int(corrupted[row, k]) == 127 and int(corrupted[row, 5]) == 128
        assert int(windows[0, k]) == int(original[0, k])
    assert torch.equal(windows, original)
    assert probe.clean_input_label(2) == '[x0,x1,BLANK,x3,x4,EOS]'
    assert probe.patch_position_label(2, 2) == 'BLANK(x2)'
    assert probe.patch_position_label(1, 2) == 'x1'


def test_site_mapping():
    assert [probe.site_label(site) for site in probe.SITES] == [
        'before L1', 'after L1 / before L2', 'after L2 / before L3',
        'after L3 / before L4', 'after L4 / before ln_f',
    ]


def test_all_local_routes_zero_response_and_full_vocab_gate():
    cfg = {'p': 127, 'order': 2, 'rules': [[25, 56]]}
    coefficients = probe._rule_coefficients(cfg)
    assert coefficients == {
        'a': 25, 'b': 56, 'a_inv': 61, 'b_inv': 93,
        'neg_a_inv_b': 13, 'neg_b_inv_a': 88,
    }
    expected = {
        0: {'R': {1: 88, 2: 93}},
        1: {'M': {0: 13, 2: 61, 3: 0}, 'R': {0: 0, 2: 88, 3: 93}},
        2: {
            'F': {0: 56, 1: 25, 3: 0, 4: 0},
            'M': {0: 0, 1: 13, 3: 61, 4: 0},
            'R': {0: 0, 1: 0, 3: 88, 4: 93},
        },
        3: {'F': {1: 56, 2: 25, 4: 0}, 'M': {1: 0, 2: 13, 4: 61}},
        4: {'F': {2: 56, 3: 25}},
    }
    for k in range(5):
        routes = probe.route_slopes(cfg, k)
        assert set(routes) == set(expected[k])
        for name, slopes in expected[k].items():
            assert routes[name]['source_slopes'] == slopes

    x0 = probe.route_slopes(cfg, 0)
    assert probe.match_routes(88, 1, 50, 1, x0) == ['R']
    assert probe.match_routes(88, 1, 127, 1, x0) == []
    k2 = probe.route_slopes(cfg, 2)
    # x0 is absent from M/R: D=0 matches both zero responses simultaneously.
    assert probe.match_routes(0, 1, 50, 0, k2) == ['M', 'R']
    assert probe.match_routes(0, 1, 128, 0, k2) == []
    assert probe.match_routes(56, 1, 50, 0, k2) == ['F']
    metadata = probe.route_definitions(cfg, 2)
    assert metadata['legal_routes']['R']['source_slopes']['x0'] == 0
    assert metadata['route_formulas']['M'] == "(k-1,-a'b),(k+1,a')"


def test_self_patch_identity_all_sites_and_positions():
    torch.set_num_threads(1)
    model = make_model()
    _, inputs, cache, logits = clean_setup(model, blank=2)
    for site in probe.SITES:
        for position in range(6):
            patched = patched_logits(model, inputs, site, position, cache)
            assert torch.equal(patched, logits), (site, position)


def test_generalized_emb_causal_and_final_controls():
    torch.set_num_threads(1)
    model = make_model()
    blank = 2
    full_positions = probe.patch_positions(blank)
    _, clean_inputs, clean_cache, clean_logits = clean_setup(model, blank)
    targets = clean_logits[:, 5].argmax(dim=-1)
    base_row = 1
    source_position = 4
    d_values = [1, 126]
    corrupted = probe.make_corrupted_inputs(
        clean_inputs[base_row:base_row + 1], source_position, d_values, 127)
    corrupted_logits, _ = model(corrupted)
    chunk_targets = targets[base_row].repeat(2)
    response = ((corrupted_logits[:, 5, :127].argmax(dim=-1)
                 - targets[base_row]) % 127).tolist()
    raw, max_diff = probe.patch_chunk(
        model, corrupted, clean_cache, clean_logits, corrupted_logits,
        chunk_targets, base_row, d_values, source_position, [False, True], response,
        patch_positions=full_positions, blank=blank)
    assert len(raw) == 2 * 5 * 5
    keys = {(row['base_row'], row['d'], row['source_position'],
             tuple(row['site']), row['position']) for row in raw}
    assert len(keys) == len(raw)
    assert {row['d'] for row in raw} == {1, 126}
    assert {row['position'] for row in raw} == set(range(5))
    assert sum(row['emb_ceiling'] for row in raw) == 2
    assert sum(row['emb_identity'] for row in raw) == 8
    assert sum(row['causal_noop'] for row in raw) == 2 * 5 * 4
    assert sum(row['final_resid_noop'] for row in raw) == 2 * 5
    assert all(row['restore_full'] for row in raw if row['emb_ceiling'])
    assert all(value <= probe.CONTROL_ATOL for value in max_diff.values())
    for row in raw:
        assert row['baseline_changed'] == (
            row['corrupted_full_vocab_output'] != row['target'])
        if row['causal_noop'] or row['final_resid_noop']:
            assert row['patched_full_vocab_output'] == row['corrupted_full_vocab_output']
        if row['position'] == blank:
            assert row['position_label'] == 'BLANK(x2)'
            assert row['causal_noop']  # source x4 is to the right of BLANK x2
            if row['site'] == ['emb']:
                assert row['emb_identity']

    # The old operand-only result is an exact subset of the new full-position grid.
    legacy_raw, _ = probe.patch_chunk(
        model, corrupted, clean_cache, clean_logits, corrupted_logits,
        chunk_targets, base_row, d_values, source_position, [False, True], response,
        patch_positions=probe.operand_positions(blank), blank=blank)
    assert [row for row in raw if row['position'] != blank] == legacy_raw

    source_cache = probe.slice_clean_cache(clean_cache, base_row, 2)
    for site in probe.SITES:
        for patch_position in (0, 1, 2, 3):
            noop = patched_logits(model, corrupted, site, patch_position, source_cache)
            assert torch.allclose(noop[:, 5], corrupted_logits[:, 5],
                                  rtol=probe.CONTROL_RTOL, atol=probe.CONTROL_ATOL)
    assert probe._control_flags(('resid', 0), 0, 1)['causal_noop']
    assert probe._control_flags(('resid', 0), 3, 4)['causal_noop']
    assert not probe._control_flags(('resid', 0), 4, 3)['causal_noop']


def test_last_resid_eos_noop_and_ceiling():
    torch.set_num_threads(1)
    model = make_model()
    _, clean_inputs, clean_cache, clean_logits = clean_setup(model, blank=4)
    corrupted = probe.make_corrupted_inputs(clean_inputs[1:2], 3, [7], 127)
    corrupted_logits, _ = model(corrupted)
    source_cache = probe.slice_clean_cache(clean_cache, 1, 1)
    for position in probe.patch_positions(4):
        noop = patched_logits(model, corrupted, ('resid', 3), position, source_cache)
        assert torch.equal(noop[:, 5], corrupted_logits[:, 5])
    patched = patched_logits(model, corrupted, ('resid', 3), 5, source_cache)
    assert torch.allclose(patched[:, 5], clean_logits[1:2, 5], rtol=0, atol=1e-6)
    assert int(patched[0, 5].argmax()) == int(clean_logits[1, 5].argmax())


def test_cross_base_and_cross_d_cache_isolation():
    torch.set_num_threads(1)
    model = make_model()
    _, clean_inputs, clean_cache, _ = clean_setup(model, blank=1)
    for base_row, d_values in ((0, [1, 7]), (1, [126, 31]), (2, [63, 64])):
        corrupted = probe.make_corrupted_inputs(
            clean_inputs[base_row:base_row + 1], 0, d_values, 127)
        source_cache = probe.slice_clean_cache(clean_cache, base_row, len(d_values))
        for site in probe.SITES:
            assert torch.equal(source_cache[site][0], clean_cache[site][base_row])
            assert torch.equal(source_cache[site][1], clean_cache[site][base_row])
        batched = patched_logits(model, corrupted, ('resid', 1), 2, source_cache)
        independent = []
        for row in range(len(d_values)):
            single_cache = probe.slice_clean_cache(clean_cache, base_row, 1)
            single = patched_logits(
                model, corrupted[row:row + 1], ('resid', 1), 2, single_cache)
            independent.append(single)
            assert torch.allclose(batched[row:row + 1], single, rtol=0, atol=1e-6)
        assert torch.allclose(batched, torch.cat(independent), rtol=0, atol=1e-6)
        wrong_cache = probe.slice_clean_cache(
            clean_cache, (base_row + 1) % 3, len(d_values))
        wrong = patched_logits(model, corrupted, ('resid', 1), 2, wrong_cache)
        assert not torch.allclose(batched, wrong, rtol=0, atol=1e-6)


def _synthetic_raw(base_row=0, d=1, restore=False, numeric=False, persistence=False,
                   nonnumeric=False, other=False, route=False, changed=True,
                   prob=0.5, logit=2.0, rank=1, margin=1.0, delta_logit=0.5,
                   delta_margin=0.25):
    return {
        'base_row': base_row, 'd': d, 'source': 'x1', 'source_position': 1,
        'site': ['resid', 1], 'site_label': probe.site_label(('resid', 1)),
        'position': 2, 'position_label': 'x2', 'route_exact': route,
        'baseline_changed': changed,
        'restore_full': restore, 'restore_numeric': numeric,
        'corrupt_persistence': persistence, 'nonnumeric': nonnumeric,
        'other': other, 'target_prob': prob, 'target_logit': logit,
        'target_rank': rank, 'target_margin': margin,
        'delta_logit': delta_logit, 'delta_margin': delta_margin,
        'emb_ceiling': False, 'emb_identity': False, 'causal_noop': False,
        'final_resid_noop': False,
    }


def test_aggregation_changed_unchanged_counts_and_legacy_metrics():
    raw = [
        _synthetic_raw(restore=True, numeric=True, route=True, changed=True,
                       prob=0.2, logit=1.0, rank=2, margin=-1.0,
                       delta_logit=0.5, delta_margin=-0.5),
        _synthetic_raw(numeric=True, nonnumeric=True, route=True, changed=True,
                       prob=0.4, logit=3.0, rank=1, margin=2.0,
                       delta_logit=1.0, delta_margin=0.5),
        _synthetic_raw(restore=True, numeric=True, changed=False,
                       prob=0.6, logit=5.0, rank=4, margin=-3.0,
                       delta_logit=-1.0, delta_margin=1.0),
        _synthetic_raw(other=True, changed=False,
                       prob=0.8, logit=7.0, rank=3, margin=4.0,
                       delta_logit=2.0, delta_margin=-1.0),
    ]
    cell = probe.aggregate_records(raw)
    assert cell['N'] == cell['n'] == 4
    assert cell['restore_full_count'] == 2 and cell['restore_full'] == 0.5
    assert cell['restore_numeric_count'] == 3 and cell['restore_numeric'] == 0.75
    assert cell['changed_N'] == 2 and cell['changed_restore_full_count'] == 1
    assert cell['changed_restore_full'] == 0.5
    assert cell['unchanged_N'] == 2 and cell['unchanged_preserved_count'] == 1
    assert cell['unchanged_preserved'] == 0.5
    assert cell['nonnumeric_count'] == 1 and cell['other_count'] == 1
    assert math.isclose(cell['mean_prob'], 0.5)
    assert math.isclose(cell['median_prob'], 0.5)
    assert cell['mean_logit'] == 4.0 and cell['median_logit'] == 4.0
    assert cell['mean_margin'] == 0.5 and cell['median_margin'] == 0.5
    assert math.isclose(cell['mean_delta_logit'], 0.625)
    assert math.isclose(cell['mean_delta_margin'], 0.0)
    assert cell['mean_rank'] == 2.5
    assert cell['route_exact_N'] == 2
    assert cell['route_exact_restore_full_count'] == 1

    all_changed = probe.aggregate_records(raw[:2])
    assert all_changed['changed_N'] == 2 and all_changed['unchanged_N'] == 0
    assert all_changed['unchanged_preserved'] is None
    all_unchanged = probe.aggregate_records(raw[2:])
    assert all_unchanged['changed_N'] == 0
    assert all_unchanged['changed_restore_full'] is None
    assert all_unchanged['unchanged_preserved'] == 0.5

    by_base = raw + [_synthetic_raw(base_row=1, restore=True, numeric=True, route=True)]
    overall = probe.aggregate_raw(by_base)
    per_base = probe.aggregate_raw(by_base, per_base=True)
    assert len(overall) == 1 and overall[0]['N'] == 5
    assert len(per_base) == 2
    assert {(row['base_row'], row['N']) for row in per_base} == {(0, 4), (1, 1)}


def test_per_base_unique_raw_keys_and_full_126d_coverage():
    raw = [
        _synthetic_raw(base_row=base, d=d, restore=(d % 2 == 0),
                       numeric=True, changed=(d % 3 != 0))
        for base in (0, 1) for d in range(1, 127)
    ]
    keys = {(row['base_row'], row['d'], row['source_position'],
             tuple(row['site']), row['position']) for row in raw}
    assert len(keys) == len(raw) == 2 * 126
    per_base = probe.aggregate_raw(raw, per_base=True)
    assert len(per_base) == 2
    for cell in per_base:
        rows = [row for row in raw if row['base_row'] == cell['base_row']]
        assert {row['d'] for row in rows} == set(range(1, 127))
        assert cell['N'] == 126
        assert cell['changed_N'] == sum(row['baseline_changed'] for row in rows)
        assert cell['restore_full_count'] == 63
        assert cell['changed_restore_full_count'] == sum(
            row['restore_full'] for row in rows if row['baseline_changed'])
    overall = probe.aggregate_raw(raw)
    assert len(overall) == 1 and overall[0]['N'] == 252


def test_dynamic_cli_paths_counts_and_dry_run():
    args = probe.parse_args([
        '--dry-run', '--blank', '2', '--base-rows', '0,1',
        '--d-values', '1,126', '--device', 'cpu', '--threads', '16',
        '--batch-size', '2',
    ])
    assert args.blank == 2 and args.base_rows == [0, 1] and args.d_values == [1, 126]
    assert args.out.endswith('fillblank_x2_activation_patch_a25b56.json.gz')
    assert args.plot_prefix.endswith('fillblank_x2_activation_patch_a25b56')
    counts = probe.plan_counts(args.base_rows, args.d_values, args.blank)
    assert counts['corrupted'] == 16 and counts['patched'] == 400
    assert counts['aggregate_cells'] == 100 and counts['per_base_cells'] == 200
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        probe.print_dry_run(args)
    text = output.getvalue()
    assert 'blank x2' in text and 'by_blank.x2' in text
    assert 'corrupted=16' in text and 'patched=400' in text
    assert 'corruption sources (operands only): [0, 1, 3, 4]' in text
    assert 'restore patch positions (includes BLANK, excludes EOS): [0, 1, 2, 3, 4]' in text
    assert '[x0,x1,BLANK,x3,x4,EOS]' in text
    assert 'source>patch causal EOS no-op' in text
    assert '8 plots' in text

    for k in range(5):
        dynamic = probe.parse_args(['--dry-run', '--blank', str(k)])
        assert dynamic.out.endswith(f'fillblank_x{k}_activation_patch_a25b56.json.gz')
        assert dynamic.plot_prefix.endswith(f'fillblank_x{k}_activation_patch_a25b56')
    defaults = probe.parse_args(['--dry-run'])
    assert defaults.blank == 0
    assert defaults.out == probe.DEFAULT_OUT
    assert defaults.plot_prefix == probe.DEFAULT_PLOT_PREFIX
    explicit = probe.parse_args([
        '--dry-run', '--blank', '3', '--out', 'custom.json.gz', '--plot-prefix', 'custom'])
    assert (explicit.out, explicit.plot_prefix) == ('custom.json.gz', 'custom')


def test_raw_index_alignment_and_mismatch_detection():
    raw = {
        'base_row': [1, 0, 0], 'd': [126, 2, 1],
        'baseline_output': [30, 13, 13],
        'perturbed_numeric_output': [7, 8, 9],
        'perturbed_full_vocab_output': [7, 8, 128],
        'D': [104, 122, 123],
    }
    index = probe._raw_record_index(raw)
    assert index[(0, 1)]['perturbed_full_vocab_output'] == 128
    assert index[(1, 126)]['baseline_output'] == 30
    try:
        probe._assert_stored_outputs(torch.tensor([7, 8]), [7, 9], 'synthetic')
        assert False, 'expected output mismatch to stop validation'
    except RuntimeError as error:
        assert 'computed=8' in str(error) and 'stored=9' in str(error)
    duplicate = {name: values[:2] + values[:1] for name, values in raw.items()}
    try:
        probe._raw_record_index(duplicate)
        assert False, 'expected duplicate raw key to fail'
    except ValueError:
        pass


def test_prediction_categories_and_delta_metrics():
    logits = torch.tensor([
        [0.0] * 13 + [5.0] + [0.0] * 113 + [1.0, 0.0],
        [0.0] * 13 + [5.0] + [0.0] * 113 + [6.0, 0.0],
        [0.0] * 30 + [5.0] + [0.0] * 96 + [1.0, 0.0],
        [0.0] * 66 + [5.0] + [0.0] * 60 + [1.0, 0.0],
    ])
    targets = torch.tensor([13, 13, 13, 13])
    corrupted_full = torch.tensor([30, 30, 30, 30])
    metrics = probe.prediction_metrics(logits, targets, 127, corrupted_full)
    assert metrics['restore_full'].tolist() == [True, False, False, False]
    assert metrics['restore_numeric'].tolist() == [True, True, False, False]
    assert metrics['nonnumeric'].tolist() == [False, True, False, False]
    assert metrics['corrupt_persistence'].tolist() == [False, False, True, False]
    assert metrics['other'].tolist() == [False, False, False, True]


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items())
             if name.startswith('test_') and callable(value)]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'\nALL TESTS PASSED: {len(tests)}/{len(tests)} (pure CPU; no checkpoint loaded)')
