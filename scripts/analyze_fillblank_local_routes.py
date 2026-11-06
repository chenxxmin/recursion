#!/usr/bin/env python3
"""Audit and compare len5/len8 local formula compatibility from saved predictions.

No model forward. Formula union counts each perturbation once. Shared zero slopes
are explicitly ambiguous. Base/position response mixtures are not circuit labels.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_fillblank_len8_local_exhaustive import save, sha256  # noqa: E402

RELATIVES = [-2, -1, 1, 2]
SIGNATURES = ['F', 'M', 'R', 'F|M', 'M|R', 'F|R', 'other']
COLORS = ['#277da8', '#f8961e', '#43aa8b', '#a992d4', '#c7ba96', '#7a9fa1', '#b53e53']


def routes(k, length, a=25, b=56, p=127):
    out = {}
    if k >= 2:
        out['F'] = {k - 2: b, k - 1: a}
    if 1 <= k <= length - 2:
        out['M'] = {k - 1: (-pow(a, -1, p) * b) % p, k + 1: pow(a, -1, p)}
    if k <= length - 3:
        out['R'] = {k + 1: (-pow(b, -1, p) * a) % p, k + 2: pow(b, -1, p)}
    return out


def histogram(values):
    counts = Counter(map(int, values))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = sum(counts.values())
    return {'n': total, 'n_unique': len(counts), 'counts': dict(ordered),
            'top1_fraction': sum(n for _, n in ordered[:1]) / total if total else None,
            'top3_fraction': sum(n for _, n in ordered[:3]) / total if total else None,
            'top20_fraction': sum(n for _, n in ordered[:20]) / total if total else None}


def summarize_cell(raw, k, j, length, base_rows, p=127):
    if not 0 < abs(j - k) <= 2:
        raise ValueError('only offsets -2,-1,+1,+2 are permitted')
    arrays = {name: np.asarray(raw[name], dtype=int) for name in ['base_row', 'd', 'y0', 'y1', 'full1', 'D']}
    n = len(arrays['d'])
    if not n or any(value.shape != (n,) for value in arrays.values()):
        raise ValueError('inconsistent raw arrays')
    if not np.array_equal(arrays['D'], (arrays['y1'] - arrays['y0']) % p):
        raise ValueError('stored D differs from (y1-y0) mod P')
    if not np.all((arrays['d'] >= 1) & (arrays['d'] < p)):
        raise ValueError('d must be nonzero mod P')
    if set(arrays['base_row']) != set(base_rows):
        raise ValueError('raw base indices differ from selected historical bases')
    numeric = (arrays['full1'] >= 0) & (arrays['full1'] < p)
    if not np.all(arrays['y1'][numeric] == arrays['full1'][numeric]):
        raise ValueError('numeric argmax differs from full argmax within numeric vocabulary')
    if not np.all((arrays['y0'] >= 0) & (arrays['y0'] < p) & (arrays['y1'] >= 0) & (arrays['y1'] < p)):
        raise ValueError('numeric predictions outside vocabulary')
    slope_map = {name: terms.get(j, 0) for name, terms in routes(k, length).items()}
    matches = {name: numeric & (arrays['D'] == coefficient * arrays['d'] % p)
               for name, coefficient in slope_map.items()}
    signatures = ['|'.join(name for name in matches if matches[name][i]) or 'other' for i in range(n)]
    counts = Counter(signatures)
    covered = np.array([s != 'other' for s in signatures])
    changed = numeric & (arrays['D'] != 0)
    effective = np.array([(int(D) * pow(int(d), -1, p)) % p for D, d in zip(arrays['D'], arrays['d'])])
    other = ~covered
    per_base = []
    for row in base_rows:
        keep = arrays['base_row'] == row
        deltas = arrays['d'][keep]
        if len(set(deltas)) != len(deltas):
            raise ValueError('duplicate delta within a base/position cell')
        base_counts = Counter(s for s, yes in zip(signatures, keep) if yes)
        route_counts = {name: int((hit & keep).sum()) for name, hit in matches.items()}
        seen_slopes = sorted(set(effective[keep & covered].tolist()))
        per_base.append({
            'base_row': int(row), 'n': int(keep.sum()), 'signature_counts': dict(base_counts),
            'route_match_counts': route_counts, 'n_distinct_compatible_slopes': len(seen_slopes),
            'compatible_slopes_seen': seen_slopes,
            'all_deltas_fit_one_slope': max(route_counts.values()) == int(keep.sum()),
            'all_deltas_fit_formula_union': base_counts['other'] == 0,
            'other_count': base_counts['other'],
            'other_output_histogram': histogram(arrays['full1'][keep & other]),
            'other_D_histogram_numeric_only': histogram(arrays['D'][keep & other & numeric]),
            'other_effective_slope_histogram_numeric_only': histogram(effective[keep & other & numeric]),
        })
    return {
        'blank': k, 'position': j, 'relative': j - k, 'n': n,
        'legal_route_slopes': slope_map, 'signature_counts': dict(counts),
        'covered_count': int(covered.sum()), 'coverage': float(covered.mean()),
        'other_count': int(other.sum()), 'other_fraction': float(other.mean()),
        'unchanged_count': int((numeric & (arrays['D'] == 0)).sum()),
        'ambiguous_count': sum(value for name, value in counts.items() if '|' in name),
        'changed_numeric_count': int(changed.sum()),
        'changed_covered_count': int((changed & covered).sum()),
        'changed_coverage': float(covered[changed].mean()) if changed.any() else None,
        'nonnumeric_count': int((~numeric).sum()),
        'route_match_counts': {name: int(hit.sum()) for name, hit in matches.items()},
        'n_base_cells': len(base_rows),
        'n_base_cells_multiple_compatible_slopes': sum(b['n_distinct_compatible_slopes'] >= 2 for b in per_base),
        'n_base_cells_one_slope_all_deltas': sum(b['all_deltas_fit_one_slope'] for b in per_base),
        'n_base_cells_with_other': sum(b['other_count'] > 0 for b in per_base),
        'other_output_histogram': histogram(arrays['full1'][other]),
        'other_D_histogram_numeric_only': histogram(arrays['D'][other & numeric]),
        'other_effective_slope_histogram_numeric_only': histogram(effective[other & numeric]),
        'per_base': per_base,
    }


def combine(cells):
    cells = list(cells)
    count_keys = ['n', 'covered_count', 'other_count', 'unchanged_count', 'ambiguous_count',
                  'changed_numeric_count', 'changed_covered_count', 'nonnumeric_count', 'n_base_cells',
                  'n_base_cells_multiple_compatible_slopes', 'n_base_cells_one_slope_all_deltas',
                  'n_base_cells_with_other']
    out = {name: sum(cell[name] for cell in cells) for name in count_keys}
    if not out['n']:
        raise ValueError('empty aggregation')
    counts = Counter()
    for cell in cells:
        counts.update(cell['signature_counts'])
    out.update(signature_counts=dict(counts), coverage=out['covered_count'] / out['n'],
               other_fraction=out['other_count'] / out['n'],
               changed_coverage=out['changed_covered_count'] / out['changed_numeric_count']
               if out['changed_numeric_count'] else None)
    assert out['n'] == out['covered_count'] + out['other_count'] == sum(counts.values())
    return out


def normalize(data, length):
    if length == 8:
        if data.get('status', 'complete') != 'complete':
            raise ValueError('len8 raw run is not complete')
        return data['by_length']['8']
    out = {'ckpt': data['ckpt'], 'model_config': data['model_config'], 'by_blank': {}}
    for k in range(5):
        old = data['by_blank'][f'x{k}']
        current = {'accuracy': old['accuracy'], 'base_selection': {
            'base_rows': list(range(len(old['base_outputs']))),
            **{key: old[key] for key in ['base_clean_windows', 'base_outputs', 'base_state_indices']}},
            'by_relative': {}}
        for relative in RELATIVES:
            j = k + relative
            if not 0 <= j < 5:
                continue
            raw = old['raw_records'][f'x{j}']
            current['by_relative'][f'{relative:+d}'] = {'raw': {
                'base_row': raw['base_row'], 'd': raw['d'], 'y0': raw['baseline_output'],
                'y1': raw['perturbed_numeric_output'], 'full1': raw['perturbed_full_vocab_output'], 'D': raw['D']}}
        out['by_blank'][str(k)] = current
    return out


def analyze(normalized, length):
    output = {'length': length, 'checkpoint': normalized['ckpt'], 'by_blank': {}}
    for k in range(length):
        entry = normalized['by_blank'][str(k)]
        base = entry['base_selection']
        lookup = dict(zip(base['base_rows'], base['base_outputs']))
        if not all(window[k] == y for window, y in zip(base['base_clean_windows'], base['base_outputs'])):
            raise ValueError('baseline bases must be correct')
        cells = {}
        for rel_key, entry_cell in entry['by_relative'].items():
            if int(rel_key) not in RELATIVES:
                raise ValueError('out-of-scope perturbation offset')
            raw = entry_cell['raw']
            if not all(lookup[row] == y for row, y in zip(raw['base_row'], raw['y0'])):
                raise ValueError('raw baseline predictions do not match base selection')
            cells[rel_key] = summarize_cell(raw, k, k + int(rel_key), length, base['base_rows'])
        per_base_routes = []
        for index, row in enumerate(base['base_rows']):
            route_scores = {}
            for name, operands in routes(k, length).items():
                groups = {'all': [0, 0], 'operands': [0, 0], 'nonoperands': [0, 0]}
                for cell in cells.values():
                    rec = next(r for r in cell['per_base'] if r['base_row'] == row)
                    hit = rec['route_match_counts'][name]
                    groups['all'][0] += hit
                    groups['all'][1] += rec['n']
                    key = 'operands' if cell['position'] in operands else 'nonoperands'
                    groups[key][0] += hit
                    groups[key][1] += rec['n']
                route_scores[name] = {group: {'hits': hit, 'n': n, 'fit': hit / n if n else None}
                                      for group, (hit, n) in groups.items()}
            best = max(route_scores, key=lambda name: route_scores[name]['all']['fit'])
            per_base_routes.append({'base_row': row, 'selection_row': index,
                                    'state_index': base['base_state_indices'][index],
                                    'clean_window': base['base_clean_windows'][index],
                                    'route_scores': route_scores, 'best_fit_route': best,
                                    'best_fixed_formula_fit': route_scores[best]['all']['fit']})
        output['by_blank'][str(k)] = {
            'baseline_accuracy': entry['accuracy'], 'by_relative': cells,
            'aggregate': combine(cells.values()), 'per_base_routes': per_base_routes,
            'best_fixed_formula_fit_median': float(np.median([r['best_fixed_formula_fit'] for r in per_base_routes])),
            'best_fit_route_counts': dict(Counter(r['best_fit_route'] for r in per_base_routes)),
            'perfect_fixed_formula_bases': sum(r['best_fixed_formula_fit'] == 1 for r in per_base_routes),
        }
    output['interior'] = combine(cell for k, blank in output['by_blank'].items()
                                 if 2 <= int(k) <= length - 3 for cell in blank['by_relative'].values())
    output['interior_by_relative'] = {str(r): combine(blank['by_relative'][f'{r:+d}']
                                                     for k, blank in output['by_blank'].items()
                                                     if 2 <= int(k) <= length - 3) for r in RELATIVES}
    output['all'] = combine(cell for blank in output['by_blank'].values() for cell in blank['by_relative'].values())
    return output


def save_figure(fig, path):
    fig.savefig(path, dpi=130)
    plt.close(fig)
    from PIL import Image
    with Image.open(path) as im:
        im.verify()


def plot_comparison(results, plot_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for column, length in enumerate([5, 8]):
        for row, field in enumerate(['coverage', 'changed_coverage']):
            values = np.full((length, 4), np.nan)
            for k, entry in results[str(length)]['by_blank'].items():
                for relative, cell in entry['by_relative'].items():
                    value = cell[field]
                    if value is not None:
                        values[int(k), RELATIVES.index(int(relative))] = value
            ax = axes[row, column]
            im = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap='viridis', aspect='auto')
            for k in range(length):
                for r in range(4):
                    if np.isfinite(values[k, r]):
                        ax.text(r, k, f'{100 * values[k, r]:.1f}%', ha='center', va='center',
                                color='white' if values[k, r] < .55 else 'black', fontsize=10)
            ax.set_xticks(range(4), [f'{r:+d}' for r in RELATIVES])
            ax.set_yticks(range(length), [f'x{k}' for k in range(length)])
            ax.set_xlabel('Perturbed position minus blank position')
            ax.set_ylabel('Blank position')
            ax.set_title(f'len{length}: ' + ('all perturbations' if row == 0 else 'changed numeric outputs only'))
            fig.colorbar(im, ax=ax, label='F/M/R union coverage')
    fig.suptitle('Exhaustive d=1..126, 32 correct bases / blank; local offsets only\n'
                 'Coverage counts each response once; zero response may fit multiple formulas')
    fig.tight_layout(rect=[0, 0, 1, .93])
    save_figure(fig, plot_dir / 'coverage_len5_vs_len8.png')

    fig, axes = plt.subplots(1, 4, figsize=(17, 5), sharey=True)
    groups = [(5, 2)] + [(8, k) for k in range(2, 6)]
    for ax, relative in zip(axes, RELATIVES):
        left = np.zeros(len(groups))
        for signature, color in zip(SIGNATURES, COLORS):
            values = np.array([results[str(L)]['by_blank'][str(k)]['by_relative'][f'{relative:+d}']['signature_counts'].get(signature, 0)
                               / results[str(L)]['by_blank'][str(k)]['by_relative'][f'{relative:+d}']['n']
                               for L, k in groups])
            ax.barh(range(len(groups)), values, left=left, color=color, label=signature)
            left += values
        ax.set_yticks(range(len(groups)), [f'len{L} blank x{k}' for L, k in groups])
        ax.set_xlim(0, 1)
        ax.set_xlabel('Fraction of perturbations')
        ax.set_title(f'Relative position {relative:+d}')
    axes[0].invert_yaxis()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=7)
    fig.suptitle('Interior blanks: compatible response signatures\n'
                 'F|M and M|R are ambiguous zero responses; labels do not prove internal routes')
    fig.tight_layout(rect=[0, .08, 1, .88])
    save_figure(fig, plot_dir / 'interior_response_signatures.png')


def plot_delta_maps(normalized, length, k, plot_dir):
    entry = normalized['by_blank'][str(k)]
    base_rows = entry['base_selection']['base_rows']
    rel_keys = list(entry['by_relative'])
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    cmap = ListedColormap(COLORS)
    cmap.set_bad('#eeeeee')
    norm = BoundaryNorm(np.arange(-.5, len(SIGNATURES) + .5), cmap.N)
    for ax, rel_key in zip(axes.flat, rel_keys):
        raw = entry['by_relative'][rel_key]['raw']
        j = k + int(rel_key)
        matrix = np.full((len(base_rows), 126), np.nan)
        for row, delta, change, full in zip(raw['base_row'], raw['d'], raw['D'], raw['full1']):
            compatible = [name for name, terms in routes(k, length).items()
                          if full < 127 and change == terms.get(j, 0) * delta % 127]
            signature = '|'.join(compatible) or 'other'
            matrix[base_rows.index(row), delta - 1] = SIGNATURES.index(signature)
        ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        ax.set_xticks([0, 24, 49, 74, 99, 125], [1, 25, 50, 75, 100, 126])
        ax.set_yticks(range(0, len(base_rows), 4), [f'b{i:02d}' for i in range(0, len(base_rows), 4)])
        ax.set_xlabel('Perturbation d (mod127)')
        ax.set_ylabel('Base selection row (0..31)')
        ax.set_title(f'Perturb x{j}, relative {rel_key}')
    for ax in list(axes.flat)[len(rel_keys):]:
        ax.set_visible(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, label=s) for s, c in zip(SIGNATURES, COLORS)],
               loc='lower center', ncol=7)
    fig.suptitle(f'len{length}, blank x{k}: SAME base / SAME position across all 126 deltas\n'
                 'A row with several colors has several compatible response forms; red = other')
    fig.tight_layout(rect=[0, .06, 1, .93])
    save_figure(fig, plot_dir / f'len{length}_blank_x{k}_delta_signatures.png')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--len5', type=Path, default=ROOT / 'reports/data/fillblank_len5_all_blank_linear_a25b56.json')
    ap.add_argument('--len8', type=Path, default=ROOT / 'reports/data/fillblank_len8_local_exhaustive_a25b56.json')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'reports/data/fillblank_len8_local_exhaustive_a25b56_analysis')
    ap.add_argument('--plot-dir', type=Path, default=ROOT / 'reports/circuit_analysis/plots/fillblank_len8_local_exhaustive_a25b56')
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    sources = {5: args.len5, 8: args.len8}
    normalized = {L: normalize(json.loads(path.read_text()), L) for L, path in sources.items()}
    for length, data in normalized.items():
        for blank in data['by_blank'].values():
            selected = blank['base_selection']['base_rows']
            for cell in blank['by_relative'].values():
                raw = cell['raw']
                for row in selected:
                    if sorted(d for b, d in zip(raw['base_row'], raw['d']) if b == row) != list(range(1, 127)):
                        raise ValueError(f'len{length}: expected all126 deltas per base; sampled old runs are not interchangeable')
    results = {str(length): analyze(data, length) for length, data in normalized.items()}
    summary = {
        'sources': {str(L): {'path': str(path.resolve()), 'sha256': sha256(path)} for L, path in sources.items()},
        'source_sha256': sha256(Path(__file__)), 'by_length': results,
        'definition': 'Union of locally legal F/M/R exact modular responses; count each record once.',
        'cautions': ['Conditioned on originally correct bases; no inference of clean circuit identity.',
                     'Length, checkpoint, and base selection differ between len5 and len8.',
                     'Only distance<=2 tested; this does not rule out other operand pairs or distant computations.',
                     'Unchanged responses may be compatible with multiple formulas; see explicit joined labels.'],
    }
    save(args.out_dir / 'summary.json', summary)
    with (args.out_dir / 'cells.csv').open('w', newline='') as stream:
        fields = ['length', 'blank', 'relative', 'n', 'coverage', 'other_fraction', 'changed_coverage',
                  'ambiguous_count', 'n_base_cells_multiple_compatible_slopes', 'n_base_cells_one_slope_all_deltas',
                  'n_base_cells_with_other']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for L, data in results.items():
            for blank in data['by_blank'].values():
                for cell in blank['by_relative'].values():
                    writer.writerow({'length': L, **{key: cell[key] for key in fields[1:]}})
    plot_comparison(results, args.plot_dir)
    plot_delta_maps(normalized[5], 5, 2, args.plot_dir)
    for k in range(8):
        plot_delta_maps(normalized[8], 8, k, args.plot_dir)
    for L, data in results.items():
        for k, blank in data['by_blank'].items():
            print(f'len{L} x{k}: union={blank["aggregate"]["coverage"]:.4%}, '
                  f'best-fixed-formula/base median={blank["best_fixed_formula_fit_median"]:.4%}, '
                  f'best labels={blank["best_fit_route_counts"]}')
    print(f'DONE {args.out_dir / "summary.json"}; plots={args.plot_dir}')


if __name__ == '__main__':
    main()
