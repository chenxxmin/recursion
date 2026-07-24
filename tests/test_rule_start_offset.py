"""Unit tests for MixedABTransformer rule_start_offset generalization."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

import core


def _make_model(order=None, use_ab_tag=False):
    kwargs = dict(p=7, d_model=16, n_head=1, n_layer=1, block_size=16, dropout=0.0,
                  entropy_penalty_weight=0.0, num_ab_pairs=2, use_ab_tag=use_ab_tag)
    if order is not None:
        kwargs['order'] = order
    return core.MixedABTransformer(**kwargs)


def _rule_logits_width(model, use_ab_tag):
    vocab = 7 + 1 + 2 if use_ab_tag else 7 + 1
    idx = torch.randint(0, vocab, (2, 8))
    labels = torch.tensor([0, 1])
    _, _, rule_logits = model(idx, targets=idx, ab_labels=labels)
    return rule_logits.shape[1]


def test_default_order_is_two():
    model = _make_model(use_ab_tag=False)
    assert model.order == 2
    assert _rule_logits_width(model, False) == 8 - 3  # legacy offset 3


def test_order2_with_tag_matches_legacy():
    model = _make_model(order=2, use_ab_tag=True)
    assert _rule_logits_width(model, True) == 8 - 4  # legacy offset 4


def test_order3_offsets():
    model = _make_model(order=3, use_ab_tag=False)
    assert _rule_logits_width(model, False) == 8 - 4  # order + 1
    model = _make_model(order=3, use_ab_tag=True)
    assert _rule_logits_width(model, True) == 8 - 5  # order + 1 + 1


if __name__ == '__main__':
    test_default_order_is_two()
    test_order2_with_tag_matches_legacy()
    test_order3_offsets()
    print("ALL TESTS PASSED: test_rule_start_offset.py")
