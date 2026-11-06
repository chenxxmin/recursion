#!/usr/bin/env python3
"""Per-base len5 attention, Euclidean clustering, and independent route comparison.

The historical a25b56 bases are fixed. Clustering never sees perturbation labels.
All distances use unnormalized probabilities, with no layer/head/base averaging.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_fillblank_attention_maps import (  # noqa: E402
    capture_batch, token_labels, validate_attention, validate_four_heads,
)
from probe_fillblank_last_blank_linear import load_model  # noqa: E402
from probe_fillblank_logit_lens import (  # noqa: E402
    DEFAULT_BASE_DATA, _validate_loaded_model, legal_route, make_blank_inputs,
    route_operands,
)

STEM = 'fillblank_len5_attention_individual_a25b56'
DEFAULT_CKPT = ROOT.parent / 'circuit_analysis/checkpoints' / (
    'single_o2_fillblank_a25b56_d512l4r8h4_P127_len3to5_randpos_noaux_'
    'lr0.0003_wd2_ckpt_seed42_best.pth')


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def cluster_vectors(vectors, distance_threshold=None):
    """Complete linkage: a cut at t guarantees every cluster diameter <= t.

    Automatic cut: maximize silhouette among attainable cuts with 2..8 groups.
    This is exploratory, not a test that discrete circuit types exist.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) < 3 or not np.isfinite(vectors).all():
        raise ValueError('need >=3 finite vectors of equal dimension')
    if distance_threshold is not None and distance_threshold < 0:
        raise ValueError('distance threshold must be nonnegative')
    condensed = pdist(vectors, metric='euclidean')
    distances = squareform(condensed)
    tree = linkage(condensed, method='complete', optimal_ordering=True)
    heights = np.unique(tree[:, 2])
    candidates = []
    for lower, upper in zip(heights[:-1], heights[1:]):
        threshold = float((lower + upper) / 2)
        labels = fcluster(tree, threshold, criterion='distance')
        sizes = np.bincount(labels)[1:]
        if 2 <= len(sizes) <= min(8, len(vectors) - 1):
            candidates.append({
                'n_clusters': len(sizes), 'threshold': threshold,
                'silhouette': float(silhouette_score(distances, labels, metric='precomputed')),
                'sizes': sizes.tolist(),
            })
    if distance_threshold is None:
        choice = max(candidates, key=lambda c: (c['silhouette'], -c['n_clusters'])) if candidates else None
        threshold = choice['threshold'] if choice else float(heights[-1])
    else:
        threshold = float(distance_threshold)
    labels = fcluster(tree, threshold, criterion='distance')
    order = leaves_list(tree)
    # Stable, human-readable IDs follow the leaf order in the saved dendrogram.
    ids = list(dict.fromkeys(labels[order].tolist()))
    labels = np.array([ids.index(int(label)) + 1 for label in labels])
    score = (float(silhouette_score(distances, labels, metric='precomputed'))
             if 1 < len(ids) < len(vectors) else None)
    groups = []
    for label in range(1, len(ids) + 1):
        rows = np.flatnonzero(labels == label)
        sub = distances[np.ix_(rows, rows)]
        medoid = int(rows[np.argmin(sub.sum(axis=1))])
        groups.append({'cluster': label, 'base_rows': rows.tolist(), 'size': len(rows),
                       'medoid_base_row': medoid, 'diameter': float(sub.max())})
        assert sub.max() <= threshold + 1e-10
    return {
        'metric': 'Euclidean', 'linkage_method': 'complete',
        'selection': 'user_threshold' if distance_threshold is not None else 'max_silhouette_2_to_8',
        'threshold': threshold, 'n_clusters': len(ids), 'silhouette': score,
        'labels': labels.tolist(), 'leaf_order': order.tolist(),
        'linkage': tree.tolist(), 'candidates': sorted(candidates, key=lambda c: c['n_clusters']),
        'groups': groups,
    }, distances


def route_evidence(entry, k, p=127):
    """Fit each full local formula to all 4 * 126 historical interventions.

    Operand slopes are the local formula coefficients, other positions slope 0.
    Non-numeric full-vocabulary argmax counts as a miss. No model rerun needed.
    These off-manifold fits are evidence, not ground-truth clean circuit labels.
    """
    n = len(entry['base_outputs'])
    routes = [r for r in ('F', 'M', 'R') if legal_route(k, r)]
    fits = np.zeros((n, len(routes)))
    by_position = {}
    for ri, route in enumerate(routes):
        slopes = dict(route_operands(k, route))
        counts = np.zeros(n, dtype=int)
        totals = np.zeros(n, dtype=int)
        by_position[route] = {}
        for j in range(5):
            if j == k:
                continue
            raw = entry['raw_records'][f'x{j}']
            rows, delta, change, full = (np.asarray(raw[key]) for key in (
                'base_row', 'd', 'D', 'perturbed_full_vocab_output'))
            if not (rows.shape == delta.shape == change.shape == full.shape):
                raise ValueError('inconsistent raw record lengths')
            for row in range(n):
                if sorted(delta[rows == row].tolist()) != list(range(1, p)):
                    raise ValueError('expected each nonzero delta exactly once per base')
            hit = (change == (slopes.get(j, 0) * delta) % p) & (full < p) & (full >= 0)
            local_n = np.bincount(rows, minlength=n)
            local_hit = np.bincount(rows, weights=hit, minlength=n).astype(int)
            counts += local_hit
            totals += local_n
            by_position[route][f'x{j}'] = (local_hit / local_n).tolist()
        fits[:, ri] = counts / totals
    result = []
    for row in range(n):
        order = np.argsort(-fits[row], kind='stable')
        best = int(order[0])
        margin = float(fits[row, best] - fits[row, order[1]]) if len(routes) > 1 else None
        strong = fits[row, best] >= 0.8 and (margin is None or margin >= 0.1)
        result.append({
            'route_fits': dict(zip(routes, fits[row].tolist())),
            'route_position_fits': {r: {j: values[row] for j, values in by_position[r].items()}
                                    for r in routes},
            'best_fit_route': routes[best], 'best_fit_margin': margin,
            'strong_route': routes[best] if strong else 'unresolved',
        })
    return result


def compare_routes(distances, records, permutations=1999):
    """Pairwise summaries; permute BASE labels (pairs are not independent)."""
    labels = np.array([r['best_fit_route'] for r in records])
    i, j = np.triu_indices(len(labels), 1)
    d = distances[i, j]
    same = labels[i] == labels[j]
    if same.all() or not same.any():
        return {'informative': False, 'reason': 'only one best-fit route or no same-route pair'}
    gap = float(d[~same].mean() - d[same].mean())
    rng = np.random.default_rng(20261106)
    exceed = 0
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        equal = shuffled[i] == shuffled[j]
        exceed += d[~equal].mean() - d[equal].mean() >= gap - 1e-12
    return {
        'informative': True, 'label_type': 'best_fit_route (tentative, including weak fits)',
        'n_same_pairs': int(same.sum()), 'n_different_pairs': int((~same).sum()),
        'same_route_mean_distance': float(d[same].mean()),
        'different_route_mean_distance': float(d[~same].mean()),
        'different_minus_same': gap,
        'base_label_permutation_p_one_sided': (exceed + 1) / (permutations + 1),
        'n_permutations': permutations, 'seed': 20261106,
        'caution': 'exploratory; labels are imperfect; no multiple-comparison correction',
    }


def attention_figure(attention, k, title):
    fig, axes = plt.subplots(4, 4, figsize=(12, 11))
    for layer in range(4):
        for head in range(4):
            ax = axes[layer, head]
            im = ax.imshow(attention[layer, head], vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
            ax.set_title(f'L{layer + 1} H{head + 1}', fontsize=9)
            ax.set_xticks(range(6), token_labels(k), rotation=50, ha='right', fontsize=6)
            ax.set_yticks(range(6), token_labels(k), fontsize=6)
            ax.set_xlabel('Key token', fontsize=7)
            ax.set_ylabel('Query token', fontsize=7)
    fig.subplots_adjust(left=.07, right=.91, bottom=.07, top=.91, wspace=.65, hspace=.85)
    fig.colorbar(im, cax=fig.add_axes([.94, .15, .015, .65]), label='Attention probability')
    fig.suptitle(title, fontsize=11)
    return fig


def overview_figure(k, cluster, distances, records, feature_label='Full 576-D attention'):
    order = np.array(cluster['leaf_order'])
    fig = plt.figure(figsize=(17, 6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, .8])
    ax = fig.add_subplot(grid[0])
    dendrogram(np.array(cluster['linkage']), ax=ax,
               labels=[f"b{i:02d} {r['best_fit_route']}" for i, r in enumerate(records)],
               leaf_rotation=90, leaf_font_size=8, color_threshold=cluster['threshold'])
    ax.axhline(cluster['threshold'], color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Euclidean distance / complete-linkage height')
    ax.set_title('Distance tree (F/M/R labels added afterwards)')
    ax = fig.add_subplot(grid[1])
    im = ax.imshow(distances[np.ix_(order, order)], vmin=0, cmap='magma')
    ax.set_xticks(range(len(order)), [f'{i:02d}' for i in order], rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)), [f'{i:02d}' for i in order], fontsize=6)
    ax.set_xlabel('Base row, tree order')
    ax.set_ylabel('Base row, tree order')
    ax.set_title(f'{feature_label} distance')
    fig.colorbar(im, ax=ax, shrink=.6)
    ax = fig.add_subplot(grid[2])
    routes = list(records[0]['route_fits'])
    support = np.array([[r['route_fits'][name] for name in routes] for r in records])
    im = ax.imshow(support[order], vmin=0, vmax=1, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(routes)), routes)
    ax.set_yticks(range(len(order)), [f'b{i:02d} / C{cluster["labels"][i]}' for i in order], fontsize=6)
    ax.set_title('Historical perturbation fit\n(not clustering input)')
    fig.colorbar(im, ax=ax, shrink=.6)
    score = cluster['silhouette']
    fig.suptitle(f'Blank x{k} | 32 individual bases | {cluster["n_clusters"]} clusters | '
                 f'silhouette={score:.3f}' if score is not None else f'Blank x{k} | one cluster')
    fig.tight_layout(rect=[0, 0, 1, .94])
    return fig


def eos_groups_figure(k, attention, cluster, records):
    """All 32 individual readout patterns, ordered by the 24-D L4 EOS tree."""
    order = cluster['leaf_order']
    values = attention[order, -1, :, -1, :].reshape(len(order), 24)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(values, vmin=0, vmax=1, cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_xticks(range(24), [f'H{head + 1}:{token}' for head in range(4) for token in token_labels(k)],
                  rotation=60, ha='right', fontsize=9)
    ax.set_yticks(range(len(order)), [f'b{i:02d}  C{cluster["labels"][i]}  '
                                     f'fit:{records[i]["best_fit_route"]}' for i in order], fontsize=9)
    for x in [5.5, 11.5, 17.5]:
        ax.axvline(x, color='white', linewidth=2)
    for row in range(1, len(order)):
        if cluster['labels'][order[row]] != cluster['labels'][order[row - 1]]:
            ax.axhline(row - .5, color='white', linewidth=2)
    ax.set_xlabel('Key token in each head (L4, EOS query only)')
    ax.set_ylabel('Individual base / attention cluster / tentative perturbation label')
    ax.set_title(f'Blank x{k}: L4 EOS attention for all 32 bases\n'
                 '24-D Euclidean clustering; no averaging; labels added after clustering')
    fig.colorbar(im, ax=ax, shrink=.7, label='Attention probability')
    fig.tight_layout()
    return fig


def render_diagnostics(summary, data_dir, plot_dir):
    """Re-render supplementary feature views from saved arrays, without a forward."""
    names = {'all_eos_queries': 'All EOS 96-D', 'last_layer_eos': 'L4 EOS 24-D'}
    for k in summary['positions']:
        entry = summary['by_blank'][f'x{k}']
        with np.load(data_dir / f'x{k}_arrays.npz') as arrays:
            attention = arrays['attention']
        for name, label in names.items():
            distances = np.load(data_dir / f'x{k}_{name}_distances.npy')
            cluster = entry['sensitivity'][name]['clustering']
            save_figure(overview_figure(k, cluster, distances, entry['bases'], label),
                        plot_dir / f'x{k}_{name}_overview.png')
        save_figure(eos_groups_figure(k, attention, entry['sensitivity']['last_layer_eos']['clustering'],
                                     entry['bases']), plot_dir / f'x{k}_last_layer_eos_bases.png')
    if 'x2' in summary['by_blank']:
        entry = summary['by_blank']['x2']
        indices = [i for i, r in enumerate(entry['bases']) if r['best_fit_route'] in ('F', 'M')]
        records = [entry['bases'][i] for i in indices]
        comparisons = {}
        with np.load(data_dir / 'x2_arrays.npz') as arrays:
            full_distance = arrays['distances']
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        labels = np.array([r['best_fit_route'] for r in records])
        i, j = np.triu_indices(len(indices), 1)
        for ax, name, label in zip(axes, ['full', *names], ['Full 576-D', *names.values()]):
            distance = full_distance if name == 'full' else np.load(data_dir / f'x2_{name}_distances.npy')
            distance = distance[np.ix_(indices, indices)]
            comparisons[name] = compare_routes(distance, records)
            groups, group_labels = [], []
            for left, right in [('F', 'F'), ('M', 'M'), ('F', 'M')]:
                keep = (((labels[i] == left) & (labels[j] == right)) |
                        ((labels[i] == right) & (labels[j] == left)))
                values = distance[i[keep], j[keep]]
                groups.append(values)
                group_labels.append(f'{left}-{right}\nn={len(values)} pairs')
            ax.boxplot(groups, labels=group_labels, showmeans=True)
            ax.set_title(label)
            ax.set_ylabel('Euclidean distance')
        fig.suptitle('Blank x2: tentative F/M labels, single R base excluded\n'
                     'Pair distributions overlap; pairs are not independent observations')
        fig.tight_layout(rect=[0, 0, 1, .9])
        save_figure(fig, plot_dir / 'x2_FM_distance_comparison.png')
        dump_json(data_dir / 'x2_excluding_single_R.json', comparisons)
        summary['x2_FM_comparison'] = comparisons


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=115)
    plt.close(fig)
    # Re-open every PNG, verify it was fully written (GPFS can fail transiently).
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        if min(im.size) < 500:
            raise RuntimeError(f'invalid plot dimensions: {path}')


def write_gallery(summary, data_dir, plot_dir):
    chunks = ['<!doctype html><meta charset="utf-8"><title>Individual attention: a25b56</title>',
              '<style>body{font:16px sans-serif;margin:30px} .grid{display:grid;'
              'grid-template-columns:repeat(2,minmax(0,1fr));gap:20px} img{max-width:100%}'
              'article{border:1px solid #ccc;padding:10px}small{color:#555}</style>',
              '<h1>(25,56), len5, s1：逐 base 注意力</h1>',
              '<p>每图一个真实 base；层和头均保留，色阶固定 0–1。按完整 576 维注意力的欧氏距离聚类。'
              'F/M/R 仅为历史扰动最佳拟合标签，不等于已确认的计算电路。</p>',
              '<nav>' + ' | '.join(f'<a href="#x{k}">x{k}</a>' for k in summary['positions']) + '</nav>']
    for k in summary['positions']:
        entry = summary['by_blank'][f'x{k}']
        chunks += [f'<h2 id="x{k}">Blank x{k}</h2>', f'<img src="x{k}_overview.png">']
        chunks += [f'<details><summary>补充：只比较 EOS query（96 维 / 24 维）</summary>'
                   f'<img src="x{k}_all_eos_queries_overview.png">'
                   f'<img src="x{k}_last_layer_eos_overview.png">'
                   f'<img src="x{k}_last_layer_eos_bases.png"></details>']
        for group in entry['clustering']['groups']:
            chunks += [f'<h3>C{group["cluster"]}: {group["size"]} bases; '
                       f'diameter {group["diameter"]:.4f}; medoid b{group["medoid_base_row"]:02d}</h3>', '<div class="grid">']
            for row in group['base_rows']:
                record = entry['bases'][row]
                rel = os.path.relpath(data_dir / f'x{k}/base_{row:02d}.json', plot_dir)
                fits = ', '.join(f'{r}={v:.3f}' for r, v in record['route_fits'].items())
                chunks += [f'<article><a href="x{k}/base_{row:02d}.png"><img loading="lazy" '
                           f'src="x{k}/base_{row:02d}.png"></a><p>b{row:02d}; '
                           f'{html.escape(str(record["clean_window"]))}; {fits}</p>'
                           f'<a href="{rel}">Attention + vector JSON</a></article>']
            chunks.append('</div>')
    (plot_dir / 'index.html').write_text('\n'.join(chunks))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--base-data', type=Path, default=ROOT / DEFAULT_BASE_DATA)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'reports/data' / STEM)
    parser.add_argument('--plot-dir', type=Path, default=ROOT / 'reports/circuit_analysis/plots' / STEM)
    parser.add_argument('--positions', nargs='+', type=int, choices=range(5), default=list(range(5)))
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--distance-threshold', type=float)
    parser.add_argument('--plots-only', action='store_true',
                        help='Re-render supplementary views/gallery from saved summary and arrays')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1 or len(set(args.positions)) != len(args.positions):
        parser.error('positive batch size/threads and distinct positions required')
    if args.plots_only:
        if args.distance_threshold is not None:
            parser.error('--plots-only preserves saved clusters; omit --distance-threshold')
        summary = json.loads((args.data_dir / 'summary.json').read_text())
        render_diagnostics(summary, args.data_dir, args.plot_dir)
        write_gallery(summary, args.data_dir, args.plot_dir)
        dump_json(args.data_dir / 'summary.json', summary)
        print('Supplementary plots and gallery regenerated from saved individual arrays.')
        return
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model, cfg, checkpoint = load_model(args.ckpt, 'cpu')
    _validate_loaded_model(model, cfg)
    validate_four_heads(model, cfg)
    if cfg.get('blank_lens', [1]) != [1] or cfg.get('blank_counts', [1]) != [1]:
        raise ValueError('only original single-blank layout is supported')
    base_data = json.loads(args.base_data.read_text())
    if cfg != base_data['model_config']:
        raise ValueError('checkpoint config does not match historical perturbation data')
    if (checkpoint['final_epoch'] != base_data['checkpoint_final_epoch'] or
            checkpoint['best_accuracy'] != base_data['checkpoint_best_accuracy']):
        raise ValueError('checkpoint training metadata does not match base data')
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'checkpoint': str(args.ckpt.resolve()), 'checkpoint_sha256': sha256(args.ckpt),
        'base_data': str(args.base_data.resolve()), 'base_data_sha256': sha256(args.base_data),
        'model_config': cfg, 'checkpoint_final_epoch': checkpoint['final_epoch'],
        'checkpoint_best_accuracy': checkpoint['best_accuracy'],
        'positions': args.positions, 'device': 'cpu', 'precision': 'float32',
        'threads': args.threads, 'batch_size': args.batch_size,
        'aggregation': 'none', 'vector_layout': 'C-order flatten [layer,head,query,key]',
        'primary_dimension': 576, 'normalization': 'none',
        'primary_metric': 'Euclidean norm of difference',
        'selection_caution': 'Exploratory maximum silhouette over 2..8 clusters; not proof of discrete routes.',
        'route_caution': 'Off-manifold perturbation fits; best-fit label need not be the clean computation route.',
        'strong_route_criterion': 'best fit >=0.8 and gap >=0.1 (gap waived if only one legal route)',
        'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'by_blank': {},
    }
    for k in args.positions:
        entry = base_data['by_blank'][f'x{k}']
        windows = torch.tensor(entry['base_clean_windows'], dtype=torch.long)
        expected = torch.tensor(entry['base_outputs'], dtype=torch.long)
        if windows.shape != (32, 5) or expected.shape != (32,):
            raise ValueError('expected historical 32 x 5 bases')
        if not torch.equal(expected, windows[:, k]):
            raise ValueError('historical bases must have correct clean predictions')
        state_ids = windows[:, 0] + 127 * windows[:, 1]
        if state_ids.tolist() != entry['base_state_indices']:
            raise ValueError('state indices do not match clean windows')
        if not torch.equal(windows[:, 2:], (25 * windows[:, 1:-1] + 56 * windows[:, :-2]) % 127):
            raise ValueError('base windows violate the checkpoint recurrence')
        inputs = make_blank_inputs(windows, k)
        batches, checks, predictions = [], [], []
        for start in range(0, len(inputs), args.batch_size):
            stop = min(len(inputs), start + args.batch_size)
            logits, layers, check = capture_batch(model, inputs[start:stop])
            pred = logits[:, -1].argmax(-1).cpu()
            if not torch.equal(pred, expected[start:stop]):
                raise RuntimeError(f'EOS prediction mismatch x{k} rows {start}:{stop}')
            batches.append(torch.stack(layers, dim=1).numpy())
            checks.append({'start': start, 'stop': stop, **check})
            predictions.extend(pred.tolist())
        attention = np.concatenate(batches)
        assert attention.shape == (32, 4, 4, 6, 6)
        assert np.min(attention) >= 0 and np.max(attention) <= 1
        causal = validate_attention(torch.from_numpy(attention))
        vectors = attention.reshape(32, -1)
        cluster, distances = cluster_vectors(vectors, args.distance_threshold)
        records = route_evidence(entry, k)
        for row, record in enumerate(records):
            record.update(base_row=row, state_index=int(state_ids[row]),
                          clean_window=windows[row].tolist(), input_tokens=inputs[row].tolist(),
                          prediction=predictions[row], target=int(windows[row, k]),
                          cluster=cluster['labels'][row])
        sensitivity = {}
        for name, array in [('all_eos_queries', attention[:, :, :, -1, :]),
                            ('last_layer_eos', attention[:, -1, :, -1, :])]:
            subcluster, subdistance = cluster_vectors(array.reshape(32, -1))
            sensitivity[name] = {'dimension': int(array[0].size), 'clustering': subcluster,
                                 'route_comparison': compare_routes(subdistance, records)}
            np.save(args.data_dir / f'x{k}_{name}_distances.npy', subdistance)
        result = {
            'blank_position': k, 'n_bases': 32, 'token_labels': token_labels(k),
            'clustering': cluster, 'bases': records,
            'route_comparison': compare_routes(distances, records), 'sensitivity': sensitivity,
            'validation': {'hook_vs_replay': checks, 'causal': causal,
                           'range_min': float(attention.min()), 'range_max': float(attention.max()),
                           'eos_predictions_match': True, 'n_eos_compared': 32},
        }
        summary['by_blank'][f'x{k}'] = result
        np.savez_compressed(args.data_dir / f'x{k}_arrays.npz', attention=attention,
                            vectors=vectors, distances=distances, state_indices=state_ids.numpy(),
                            inputs=inputs.numpy(), predictions=np.array(predictions),
                            cluster_labels=np.array(cluster['labels']), linkage=np.array(cluster['linkage']))
        np.savetxt(args.data_dir / f'x{k}_distances.csv', distances, delimiter=',',
                   header=','.join(f'base_{row:02d}' for row in range(32)), comments='')
        membership_path = args.data_dir / f'x{k}_membership.csv'
        with membership_path.open('w', newline='') as stream:
            fields = ['base_row', 'state_index', 'cluster', 'best_fit_route', 'strong_route', 'F', 'M', 'R']
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            for record in records:
                writer.writerow({**{f: record[f] for f in fields[:5]}, **record['route_fits']})
        for row, record in enumerate(records):
            path = args.data_dir / f'x{k}/base_{row:02d}.json'
            dump_json(path, {**record, 'blank_position': k, 'token_labels': token_labels(k),
                             'attention_layout': '[layer][head][query][key]',
                             'attention': attention[row].tolist(), 'vector': vectors[row].tolist()})
            reloaded = json.loads(path.read_text())
            restored = np.asarray(reloaded['attention'], dtype=np.float32)
            if not np.array_equal(restored, attention[row]):
                raise RuntimeError('attention JSON roundtrip mismatch')
            if not np.array_equal(np.asarray(reloaded['vector']), restored.flatten()):
                raise RuntimeError('vector order mismatch')
            validate_attention(torch.from_numpy(restored))
        with np.load(args.data_dir / f'x{k}_arrays.npz') as saved:
            if not np.array_equal(saved['attention'], attention):
                raise RuntimeError('NPZ roundtrip mismatch')
            np.testing.assert_allclose(squareform(pdist(saved['vectors'])), saved['distances'], rtol=0, atol=1e-12)
        result['validation']['json_npz_roundtrip_passed'] = True
        dump_json(args.data_dir / 'summary.json', summary)
        print(f'x{k}: capture validated; cluster sizes={[g["size"] for g in cluster["groups"]]}, '
              f'silhouette={cluster["silhouette"]}; drawing 32 individual maps', flush=True)
        save_figure(overview_figure(k, cluster, distances, records), args.plot_dir / f'x{k}_overview.png')
        for row, record in enumerate(records):
            fits = ', '.join(f'{name}={value:.3f}' for name, value in record['route_fits'].items())
            title = (f'Blank x{k} | base {row:02d} | state {record["state_index"]} | C{record["cluster"]}\n'
                     f'Clean window: {record["clean_window"]} | prediction={record["prediction"]}\n'
                     f'Historical perturbation fit: {fits} | individual attention (no averaging)')
            save_figure(attention_figure(attention[row], k, title), args.plot_dir / f'x{k}/base_{row:02d}.png')
            if (row + 1) % 8 == 0:
                print(f'x{k}: maps {row + 1}/32 saved and re-opened', flush=True)
    render_diagnostics(summary, args.data_dir, args.plot_dir)
    write_gallery(summary, args.data_dir, args.plot_dir)
    summary['validation_complete'] = True
    summary['n_individual_maps'] = 32 * len(args.positions)
    dump_json(args.data_dir / 'summary.json', summary)
    print(f'DONE: {args.data_dir / "summary.json"}\nGallery: {args.plot_dir / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
