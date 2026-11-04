"""Pure-CPU tests for probe_fillblank_logit_lens.py (no checkpoint loaded)."""
import contextlib
import io
import math
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import probe_fillblank_logit_lens as probe


def make_cell(layer, position, mean_prob, top1_rate, k=2, target_id=None):
    return {
        'k': k,
        'target_id': target_id or f'x{k}',
        'route': 'final' if target_id is None else 'F',
        'layer_index': layer,
        'position': position,
        'n': 32,
        'mean_prob': mean_prob,
        'median_prob': mean_prob,
        'top1_rate': top1_rate,
        'mean_margin': 0.0,
        'mean_rank': 1.0,
    }


def make_raw(k=2, target_id='target_a', base_row=0, layer=0, position=0,
             prob=0.5, rank=1, target_value=7, route='F'):
    return {
        'k': k, 'target_id': target_id, 'base_row': base_row,
        'target_value': target_value, 'route': route,
        'layer_index': layer, 'position': position,
        'prob': prob, 'logit': 0.0, 'rank': rank, 'margin': 0.0,
    }


def test_legal_positions_and_final_count_plan():
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
    assert [item['n_targets'] for item in plan] == [1, 1, 1, 1, 1]
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


def test_modular_coefficients_and_intermediate_target_enumeration():
    assert (probe.A, probe.B) == (25, 56)
    assert (probe.A_INV, probe.B_INV) == (61, 93)
    assert probe.A * probe.A_INV % probe.P == 1
    assert probe.B * probe.B_INV % probe.P == 1
    assert probe.NEG_A_INV_B == 13 and probe.NEG_B_INV_A == 88
    assert probe.COEFFICIENTS == {
        'a': 25, 'b': 56, "a'": 61, "b'": 93, "-a'b": 13, "-b'a": 88,
    }

    expected_routes = [
        {'R'}, {'M', 'R'}, {'F', 'M', 'R'}, {'F', 'M'}, {'F'},
    ]
    all_targets = []
    for k, routes in enumerate(expected_routes):
        targets = probe.intermediate_targets(k)
        all_targets.extend((k, target['target_id']) for target in targets)
        assert {target['route'] for target in targets} == routes
        for target in targets:
            assert all(0 <= position < 5 for _, position in target['terms'])
        for route in ('F', 'M', 'R'):
            assert any(target['route'] == route for target in targets) == (route in routes)
    assert [len(probe.intermediate_targets(k)) for k in range(5)] == [3, 6, 8, 5, 2]
    assert len(all_targets) == 24 and len(set(all_targets)) == 24

    plan = probe.build_plan(target_mode='intermediates')
    assert [item['n_cells'] for item in plan] == [48, 120, 192, 100, 32]
    assert [item['n_raw_records'] for item in plan] == [1536, 3840, 6144, 3200, 1024]
    totals = probe.plan_totals(plan)
    assert totals['n_cells'] == 492 and totals['n_raw_records'] == 15744


def test_clean_target_truth_scaling_subtraction_and_blank_target_separation():
    windows = torch.tensor([
        [120, 5, 7, 9, 11],
        [3, 126, 80, 2, 100],
    ], dtype=torch.long)
    by_id = {target['target_id']: target for target in probe.intermediate_targets(2)}
    b_x0 = probe.compute_target_values(windows, by_id['F_b_x0'])
    assert b_x0.tolist() == [(56 * 120) % 127, (56 * 3) % 127]
    neg_a_inv_b_x1 = probe.compute_target_values(windows, by_id['M_neg_a_inv_b_x1'])
    assert neg_a_inv_b_x1.tolist() == [(13 * 5) % 127, (13 * 126) % 127]
    u_m = probe.compute_target_values(windows, by_id['uM'])
    assert u_m.tolist() == [(9 - 56 * 5) % 127, (2 - 56 * 126) % 127]
    u_r = probe.compute_target_values(windows, by_id['uR'])
    assert u_r.tolist() == [(11 - 25 * 9) % 127, (100 - 25 * 2) % 127]

    # k is the blank, not an intermediate target index: uM/uR never read x2 here.
    assert all(position != 2 for target in (by_id['uM'], by_id['uR'])
               for _, position in target['terms'])
    blanked = probe.make_blank_inputs(windows, 2)
    assert blanked[:, 2].tolist() == [127, 127]
    assert windows[:, 2].tolist() == [7, 80]
    assert probe.compute_target_values(windows, by_id['uM']).tolist() == u_m.tolist()

    final = probe.targets_for_blank(2, 'final')[0]
    assert final['target_id'] == 'x2' and final['route'] == 'final'
    assert probe.compute_target_values(windows, final).tolist() == [7, 80]


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
    torch.testing.assert_close(metrics['margin'], torch.tensor([-1.0, -1.0, 1.0, 0.0]))
    expected_prob = torch.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(metrics['prob'], expected_prob)
    assert math.isclose(float(metrics['prob'][0]), math.exp(2) / torch.exp(logits[0]).sum())


def test_aggregation_metrics_and_target_base_grouping():
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

    separated = [
        make_raw(target_id='A', base_row=0, prob=0.9),
        make_raw(target_id='A', base_row=1, prob=0.7),
        make_raw(target_id='B', base_row=0, prob=0.1, rank=2),
        make_raw(target_id='B', base_row=1, prob=0.3, rank=1),
    ]
    aggregates = probe.aggregate_raw(separated)
    assert len(aggregates) == 2
    by_target = {cell['target_id']: cell for cell in aggregates}
    assert by_target['A']['n'] == 2 and math.isclose(by_target['A']['mean_prob'], 0.8)
    assert by_target['B']['n'] == 2 and math.isclose(by_target['B']['mean_prob'], 0.2)
    per_base = probe.build_per_base_readability(separated)
    assert len(per_base['bases']) == 4
    assert {(row['target_id'], row['base_row']) for row in per_base['bases']} == {
        ('A', 0), ('A', 1), ('B', 0), ('B', 1),
    }
    summary = {row['target_id']: row for row in per_base['summary']}
    assert summary['A']['primary']['n_readable_bases'] == 2
    assert summary['B']['primary']['n_readable_bases'] == 0  # probabilities below 0.50


def test_clear_thresholds_are_inclusive_and_conjunctive():
    assert probe.is_clear(make_cell(0, 0, 0.50, 0.90), 'primary')
    assert not probe.is_clear(make_cell(0, 0, 0.4999, 1.0), 'primary')
    assert not probe.is_clear(make_cell(0, 0, 1.0, 0.8999), 'primary')
    assert probe.is_clear(make_cell(0, 0, 0.90, 0.95), 'strict')
    assert not probe.is_clear(make_cell(0, 0, 0.8999, 1.0), 'strict')
    assert not probe.is_clear(make_cell(0, 0, 1.0, 0.9499), 'strict')


def test_earliest_layer_position_order_highest_and_target_grouping():
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

    target_cells = [
        make_cell(1, 0, 0.99, 1.0, target_id='A'),
        make_cell(0, 0, 0.10, 0.0, target_id='A'),
        make_cell(0, 0, 0.99, 1.0, target_id='B'),
    ]
    only_a = probe.clear_summary(target_cells, 2, 'primary', target_id='A')
    assert len(only_a['clear_cells']) == 1
    assert only_a['earliest']['layer_index'] == 1
    earliest = probe.build_earliest(target_cells, 'intermediates')
    assert set(earliest['primary']['x2']) == {'A', 'B'}
    assert earliest['primary']['x2']['B']['earliest']['layer_index'] == 0
    final_earliest = probe.build_earliest(cells, 'final')
    assert set(final_earliest['primary']) == {f'x{k}' for k in range(5)}


def test_single_base_thresholds_earliest_best_and_null_audit():
    assert probe.is_readable(make_raw(prob=0.50, rank=1), 'primary')
    assert not probe.is_readable(make_raw(prob=0.4999, rank=1), 'primary')
    assert not probe.is_readable(make_raw(prob=0.99, rank=2), 'primary')
    assert probe.is_readable(make_raw(prob=0.90, rank=1), 'strict')
    assert not probe.is_readable(make_raw(prob=0.8999, rank=1), 'strict')

    raw = [
        make_raw(base_row=0, layer=0, position=0, prob=0.99, rank=2),
        make_raw(base_row=0, layer=0, position=1, prob=0.50, rank=1),
        make_raw(base_row=0, layer=1, position=0, prob=0.90, rank=1),
        make_raw(base_row=1, layer=0, position=0, prob=0.20, rank=2),
        make_raw(base_row=1, layer=1, position=0, prob=0.40, rank=1),
    ]
    report = probe.build_per_base_readability(raw)
    assert len(report['bases']) == 2  # includes the base with no readable cell
    base0, base1 = report['bases']
    assert (base0['primary']['earliest']['layer_index'],
            base0['primary']['earliest']['position']) == (0, 1)
    assert (base0['primary']['best_cell']['layer_index'],
            base0['primary']['best_cell']['position']) == (1, 0)
    assert (base0['strict']['earliest']['layer_index'],
            base0['strict']['earliest']['position']) == (1, 0)
    assert base1['primary']['earliest'] is None
    assert base1['primary']['best_cell'] is None
    assert not base1['primary']['readable_anywhere']
    summary = report['summary'][0]
    assert summary['n_bases'] == 2
    assert summary['primary']['n_readable_bases'] == 1
    assert summary['primary']['readable_base_rows'] == [0]
    assert len(summary['primary']['best_cells']) == 1
    assert summary['strict']['n_readable_bases'] == 1
    assert base0['target_in_visible_inputs'] is None
    assert base0['primary']['earliest']['direct_copy_at_position'] is None
    assert base0['primary']['earliest']['target_in_visible_inputs'] is None


def test_copy_control_flags_counts_rows_and_blank_exclusion():
    raw = [
        make_raw(k=2, base_row=0, position=1, target_value=8, prob=0.60),
        make_raw(k=2, base_row=1, position=2, target_value=9, prob=0.60),
        make_raw(k=2, base_row=2, position=2, target_value=7, prob=0.60),
    ]
    base_data = {'by_blank': {'x2': {'base_clean_windows': [
        [7, 8, 9, 10, 11],  # direct copy at x1; target also visible
        [1, 2, 9, 4, 5],    # blank slot equals target but is not visible/copyable
        [7, 2, 9, 4, 5],    # non-direct cell, but target is visible at x0
    ]}}}
    report = probe.build_per_base_readability(raw, base_data)
    base0, base1, base2 = report['bases']
    assert [row['target_in_visible_inputs'] for row in report['bases']] == [True, False, True]
    assert base0['primary']['earliest']['direct_copy_at_position'] is True
    assert base0['primary']['best_cell']['target_in_visible_inputs'] is True
    assert base1['primary']['earliest']['direct_copy_at_position'] is False
    assert base1['primary']['earliest']['target_in_visible_inputs'] is False
    assert base2['primary']['earliest']['direct_copy_at_position'] is False
    assert base2['primary']['earliest']['target_in_visible_inputs'] is True

    summary = report['summary'][0]['primary']
    assert summary['n_readable_bases'] == 3 and summary['n_readable_cells'] == 3
    assert summary['n_readable_bases_excluding_direct_copy_at_position'] == 2
    assert summary['readable_base_rows_excluding_direct_copy_at_position'] == [1, 2]
    assert summary['n_readable_cells_excluding_direct_copy_at_position'] == 2
    assert summary['n_readable_bases_excluding_target_in_visible_inputs'] == 1
    assert summary['readable_base_rows_excluding_target_in_visible_inputs'] == [1]
    assert summary['n_readable_cells_excluding_target_in_visible_inputs'] == 1
    assert len(summary['best_cells']) == 3
    assert report['summary'][0]['strict']['n_readable_bases'] == 0
    assert report['summary'][0]['strict'][
        'n_readable_bases_excluding_direct_copy_at_position'] == 0


def _synthetic_route_raw(slope, mismatches):
    raw = {
        'base_row': [], 'd': [], 'D': [], 'perturbed_full_vocab_output': [],
    }
    for base_row in range(2):
        for d in range(1, 127):
            wrong = d in mismatches.get((base_row, slope), set())
            raw['base_row'].append(base_row)
            raw['d'].append(d)
            raw['D'].append((slope * d + int(wrong)) % 127)
            raw['perturbed_full_vocab_output'].append(0)
    return raw


def test_route_support_exact_rates_shared_d_and_ranking():
    slopes = [56, 25]
    mismatches = {
        (0, 56): {1},
        (0, 25): {1, 2},
    }
    entry = {
        'base_clean_windows': [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
        'raw_records': {
            'x0': _synthetic_route_raw(56, mismatches),
            'x1': _synthetic_route_raw(25, mismatches),
        },
    }
    support = probe.route_support_for_blank(entry, 2, 'F')
    assert support['operands'] == [
        {'position': 0, 'slope': 56}, {'position': 1, 'slope': 25},
    ]
    base0, base1 = support['bases']
    assert base0['operand_exact_match_counts'] == {'x0': 125, 'x1': 124}
    assert math.isclose(base0['operand_exact_match_rates']['x0'], 125 / 126)
    assert math.isclose(base0['min_operand_exact_match_rate'], 124 / 126)
    assert base0['shared_exact_match_count'] == 124
    assert 1 not in base0['shared_d'] and 2 not in base0['shared_d']
    assert 3 in base0['shared_d']
    assert base1['shared_exact_match_count'] == 126
    assert len(support['top3']) == 2
    assert support['top3'][0]['base_row'] == 1
    assert support['top3'][0]['window'] == [5, 4, 3, 2, 1]
    assert all(len(row['shared_d']) == row['shared_exact_match_count']
               for row in support['bases'])
    assert slopes == [operand['slope'] for operand in support['operands']]


def test_loaded_model_validation_without_checkpoint():
    model = SimpleNamespace(transformer=SimpleNamespace(h=[None] * 4))
    cfg = {
        'p': 127, 'd_model': 512, 'n_layer': 4, 'vocab_size': 129,
        'blank_token_id': 127, 'eos_token_id': 128,
        'variant': 'fill_blank', 'order': 2, 'rules': [[25, 56]],
        'train_len': 5, 'body_supervision': False,
    }
    probe._validate_loaded_model(model, cfg)
    for key, bad_value in [
        ('variant', None), ('order', 3), ('rules', [[56, 25]]),
        ('train_len', 16), ('body_supervision', True), ('n_layer', 2),
    ]:
        bad = dict(cfg)
        bad[key] = bad_value
        try:
            probe._validate_loaded_model(model, bad)
            assert False, f'expected {key} validation to fail'
        except ValueError:
            pass
    try:
        probe._validate_loaded_model(SimpleNamespace(transformer=SimpleNamespace(h=[None] * 3)), cfg)
        assert False, 'expected transformer block validation to fail'
    except ValueError:
        pass


class _TinyBlock(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_model)
        self.forward_calls = 0

    def forward(self, hidden):
        self.forward_calls += 1
        return hidden + torch.tanh(self.linear(hidden))


class _TinyTransformer(torch.nn.Module):
    def __init__(self, vocab_size=129, d_model=8):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size, d_model)
        self.drop = torch.nn.Dropout(0.0)
        self.h = torch.nn.ModuleList([_TinyBlock(d_model) for _ in range(4)])
        self.ln_f = torch.nn.LayerNorm(d_model)


class _TinyFourLayerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _TinyTransformer()
        self.lm_head = torch.nn.Linear(8, 129, bias=False)
        self.forward_calls = 0

    @property
    def block_calls(self):
        return sum(block.forward_calls for block in self.transformer.h)

    def reset_calls(self):
        self.forward_calls = 0
        for block in self.transformer.h:
            block.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        hidden = self.transformer.drop(self.transformer.wte(inputs))
        for block in self.transformer.h:
            hidden = block(hidden)
        logits = self.lm_head(self.transformer.ln_f(hidden))
        return logits, None


def test_probe_batch_multi_target_matches_independent_calls_and_base_offset():
    torch.manual_seed(20261104)
    model = _TinyFourLayerModel().eval()
    windows = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.long)
    inputs = probe.make_blank_inputs(windows, 2)
    specs = probe.intermediate_targets(2)
    target_values = {
        target['target_id']: probe.compute_target_values(windows, target)
        for target in specs
    }
    positions = probe.legal_positions(2)

    model.reset_calls()
    multi_raw, manual_eos, reference_eos = probe.probe_batch(
        model, inputs, target_values, 2, positions,
        target_specs=specs, base_offset=11)
    assert model.forward_calls == 1
    assert model.block_calls == 8  # four manual blocks plus four in model.forward
    torch.testing.assert_close(manual_eos, reference_eos, rtol=0, atol=0)

    model.reset_calls()
    independent_raw = []
    for target in specs:
        target_id = target['target_id']
        raw, _, _ = probe.probe_batch(
            model, inputs, {target_id: target_values[target_id]}, 2, positions,
            target_specs=[target], base_offset=11)
        independent_raw.extend(raw)
    assert model.forward_calls == len(specs)
    assert model.block_calls == 8 * len(specs)

    # Multi-target raw is layer-major while independent calls are target-major.
    order = lambda row: (row['target_id'], row['base_row'],
                         row['layer_index'], row['position'])
    # Exact equality covers prob/logit/rank/margin and every metadata field.
    assert sorted(multi_raw, key=order) == sorted(independent_raw, key=order)
    assert len(multi_raw) == 8 * 4 * 6 * 2
    assert {row['base_row'] for row in multi_raw} == {11, 12}
    assert (multi_raw[0]['base_row'], multi_raw[1]['base_row']) == (11, 12)
    assert multi_raw[0]['target_value'] == int(target_values['F_b_x0'][0])
    assert multi_raw[1]['target_value'] == int(target_values['F_b_x0'][1])
    assert {row['target_id'] for row in multi_raw} == set(target_values)


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
    assert args.target_mode == 'final'
    assert args.device == 'cpu' and args.threads == 16 and args.batch_size == 32
    assert args.base_data.endswith('fillblank_len5_all_blank_linear_a25b56.json')
    assert args.out.endswith('fillblank_len5_logit_lens_a25b56.json')
    assert args.plot.endswith('fillblank_len5_logit_lens_a25b56.png')
    assert args.ckpt.endswith('seed42_best.pth') and args.dry_run

    old_command = probe.parse_args([
        '--ckpt', 'model.pth', '--base-data', 'base.json', '--out', 'old.json',
        '--plot', 'old.png', '--device', 'cpu', '--threads', '2', '--batch-size', '8',
    ])
    assert old_command.target_mode == 'final'
    assert (old_command.out, old_command.plot) == ('old.json', 'old.png')

    intermediate = probe.parse_args(['--target-mode', 'intermediates', '--dry-run'])
    assert intermediate.out.endswith('fillblank_len5_logit_lens_intermediates_a25b56.json')
    assert intermediate.plot.endswith('fillblank_len5_logit_lens_intermediates_a25b56')
    explicit = probe.parse_args(['--target-mode', 'intermediates', '--plot', 'prefix'])
    assert explicit.plot == 'prefix'


def test_dry_run_prints_mode_targets_counts_and_five_shared_batches():
    args = probe.parse_args(['--dry-run', '--target-mode', 'intermediates'])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        probe.print_dry_run(args)
    text = output.getvalue()
    assert 'target mode: intermediates' in text
    assert 'uM' in text and 'uR' in text
    assert "route='F'" not in text  # human-readable route labels are used
    assert '24 targets; 492 aggregate cells; 15744 raw records' in text
    assert '[1, 1, 1, 1, 1]; total=5' in text
    assert 'no per-target forward' in text
    assert 'rank==1 & prob>=0.50' in text


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items())
             if name.startswith('test_') and callable(value)]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'\nALL TESTS PASSED: {len(tests)}/{len(tests)} (pure CPU; no checkpoint loaded)')
