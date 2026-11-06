"""CPU mathematical checks for attention distance, clustering and route evidence."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import probe_fillblank_attention_clusters as probe


def test_distances_and_threshold_clusters():
    vectors = np.array([[0., 0.], [0., 1.], [3., 4.], [3., 5.]])
    result, distances = probe.cluster_vectors(vectors, distance_threshold=1.1)
    assert distances[0, 2] == 5
    assert distances[0, 1] == distances[2, 3] == 1
    assert sorted(g['base_rows'] for g in result['groups']) == [[0, 1], [2, 3]]
    assert all(g['diameter'] <= 1.1 for g in result['groups'])
    # Duplicating each feature multiplies distances by sqrt(2); no normalization.
    _, doubled = probe.cluster_vectors(np.tile(vectors, (1, 2)))
    np.testing.assert_allclose(doubled, distances * np.sqrt(2))
    auto, _ = probe.cluster_vectors(vectors)
    assert auto['n_clusters'] == 2 and auto['silhouette'] > .7
    # Zero/tied distances must not cause invalid silhouette scoring or NaNs.
    uniform, zero = probe.cluster_vectors(np.zeros((4, 3)))
    assert uniform['n_clusters'] == 1 and uniform['silhouette'] is None
    assert np.all(zero == 0)


def test_route_evidence_known_forward_and_middle():
    p = 127
    raw_records = {}
    for j in [0, 1, 3, 4]:
        rows, ds, changes, outputs = [], [], [], []
        for row, route in enumerate(['F', 'M']):
            slopes = dict(probe.route_operands(2, route))
            for delta in range(1, p):
                rows.append(row)
                ds.append(delta)
                changes.append(slopes.get(j, 0) * delta % p)
                outputs.append(changes[-1])
        raw_records[f'x{j}'] = dict(base_row=rows, d=ds, D=changes, perturbed_full_vocab_output=outputs)
    entry = dict(base_outputs=[0, 0], raw_records=raw_records)
    evidence = probe.route_evidence(entry, 2)
    assert [r['best_fit_route'] for r in evidence] == ['F', 'M']
    assert [r['strong_route'] for r in evidence] == ['F', 'M']
    assert evidence[0]['route_fits']['F'] == evidence[1]['route_fits']['M'] == 1
    # A special-token full argmax must count as a miss despite numeric match.
    entry['raw_records']['x0']['perturbed_full_vocab_output'][0] = 127
    changed = probe.route_evidence(entry, 2)
    assert changed[0]['route_fits']['F'] == 503 / 504


def test_pairwise_comparison_and_visual_layout():
    _, d = probe.cluster_vectors([[0., 0.], [0., .1], [10., 0.], [10., .1]])
    evidence = [{'best_fit_route': r} for r in ['F', 'F', 'M', 'M']]
    result = probe.compare_routes(d, evidence, permutations=99)
    assert result['n_same_pairs'] == 2 and result['n_different_pairs'] == 4
    assert result['same_route_mean_distance'] == .1
    assert result['different_minus_same'] > 9
    assert 0 < result['base_label_permutation_p_one_sided'] <= 1
    a = np.zeros((4, 4, 6, 6))
    for q in range(6):
        a[:, :, q, :q + 1] = 1 / (q + 1)
    fig = probe.attention_figure(a, 2, 'individual base')
    axes = [ax for ax in fig.axes if ax.images]
    assert len(axes) == 16
    for ax in axes:
        assert ax.images[0].get_clim() == (0, 1)
        assert ax.get_xlabel() == 'Key token' and ax.get_ylabel() == 'Query token'
        assert [t.get_text() for t in ax.get_xticklabels()] == probe.token_labels(2)
    probe.plt.close(fig)


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
