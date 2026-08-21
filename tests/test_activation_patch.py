"""Tests for src/activation_patch.py. Standalone runnable (python tests/test_activation_patch.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import models
from activation_patch import (generate_paired_samples, rule_targets,
                              capture_hidden, patched_logits, classify,
                              run_probe)


def make_model(seed=0):
    torch.manual_seed(seed)
    return models.MixedABTransformer(
        num_ab_pairs=2, use_ab_tag=False, order=2,
        p=11, d_model=32, n_head=2, n_layer=2, block_size=32,
        mlp_ratio=4).eval()


def test_paired_samples_share_init_and_follow_rules():
    p = 11
    a, b = generate_paired_samples(p, (1, 1), (1, 2), num_pairs=8, length=10, seed=1)
    assert a.shape == b.shape == (8, 10)
    assert torch.equal(a[:, :2], b[:, :2])  # shared initials
    for k in range(2, 10):
        assert torch.equal(a[:, k], (a[:, k - 2] + a[:, k - 1]) % p)
        assert torch.equal(b[:, k], (b[:, k - 2] + 2 * b[:, k - 1]) % p)


def test_rule_targets():
    p = 11
    seqs, _ = generate_paired_samples(p, (1, 1), (1, 2), 4, 8, seed=2)
    v = rule_targets(seqs, (2, 3), p)
    assert (v[:, :2] == -1).all()
    assert torch.equal(v[:, 2:], (2 * seqs[:, :-2] + 3 * seqs[:, 1:-1]) % p)


def test_capture_shapes():
    model = make_model()
    idx = torch.randint(0, 11, (4, 12))
    cache, logits = capture_hidden(model, idx)
    assert logits.shape == (4, 12, model.vocab_size)
    for li in range(2):
        assert cache[('attn', li)].shape == (4, 12, 32)
        assert cache[('mlp', li)].shape == (4, 12, 128)


def test_patch_with_own_cache_is_identity():
    model = make_model()
    idx = torch.randint(0, 11, (4, 12))
    cache, logits = capture_hidden(model, idx)
    for site in [('attn', 0, 0), ('attn', 1, 1), ('mlp', 0)]:
        patched = patched_logits(model, idx, site, 5, cache)
        assert torch.equal(patched, logits), f"self-patch changed logits at {site}"


def test_patch_actually_changes_logits():
    model = make_model()
    a = torch.randint(0, 11, (4, 12))
    b = torch.randint(0, 11, (4, 12))
    cache_a, _ = capture_hidden(model, a)
    _, logits_b = capture_hidden(model, b)
    patched = patched_logits(model, b, ('attn', 0, 0), 3, cache_a)
    assert not torch.equal(patched, logits_b)
    # positions before the patch point are causally unaffected
    assert torch.equal(patched[:, :3], logits_b[:, :3])


def test_classify():
    logits = torch.zeros(2, 6, 11)
    seqs = torch.randint(0, 11, (2, 6))
    v_a = rule_targets(seqs, (1, 1), 11)
    v_b = rule_targets(seqs, (1, 2), 11)
    # force argmax to v_a at every predicted position
    for i in range(2):
        for k in range(2, 6):
            logits[i, k - 1, v_a[i, k]] = 1.0
    res = classify(logits, seqs, v_a, v_b)
    assert res['match_a'][:, 2:].all()
    both = (v_a[:, 2:] == v_b[:, 2:])
    assert torch.equal(res['match_b'][:, 2:], both)


def test_run_probe_end_to_end():
    model = make_model()
    config = {'p': 11, 'ab_pairs': [[1, 1], [1, 2]], 'train_len': 10,
              'block_size': 32}
    report = run_probe(model, config, num_pairs=4, length=10, verbose=False)
    n_sites = 2 * (2 + 1)  # layers x (heads + mlp)
    assert len(report['results']) == 2 * n_sites * 10  # directions x sites x pos
    for r in report['results']:
        assert 0.0 <= r['match_a_rate'] <= 1.0
        assert 0.0 <= r['match_b_rate'] <= 1.0
    assert set(report['baseline']) == {'input_A', 'input_B'}


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
