"""Regression tests for the two-token modular-arithmetic experiment."""
import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from binary_arithmetic import (SharedEmbeddingBinaryMLP, build_binary_samples,
                               create_model)


class TestBinaryArithmetic(unittest.TestCase):
    def test_fixed_state_split_shared_and_disjoint(self):
        p = 11
        tr1, te1 = build_binary_samples(p, 5, 3, 'fixed_states', .7, 42)
        tr2, te2 = build_binary_samples(p, 7, 2, 'fixed_states', .7, 42)
        self.assertEqual(len(tr1), int(p*p*.7))
        self.assertEqual(len(tr1) + len(te1), p*p)
        self.assertEqual({tuple(x[:2].tolist()) for x in tr1},
                         {tuple(x[:2].tolist()) for x in tr2})
        self.assertFalse({tuple(x[:2].tolist()) for x in tr1} &
                         {tuple(x[:2].tolist()) for x in te1})
        for q0, q1, q2 in tr1 + te1:
            self.assertEqual(q2.item(), (5*q1.item()+3*q0.item())%p)
        self.assertNotEqual([v[2].item() for v in tr1],
                            [v[2].item() for v in tr2])

    def test_full_split_orbit_assignment_and_rng(self):
        state = random.getstate()
        tr, te = build_binary_samples(11, 5, 3, 'full_split', .7, 42)
        self.assertEqual(state, random.getstate())
        self.assertEqual(len(tr) + len(te), 121)
        self.assertEqual(len(tr), int(121*.7))
        self.assertEqual(len({tuple(x[:2].tolist()) for x in tr + te}), 121)
        for q0, q1, q2 in tr + te:
            self.assertEqual(q2.item(), (5*q1.item()+3*q0.item())%11)

    def test_mlp_ties_weights_and_ignores_label_input(self):
        model = SharedEmbeddingBinaryMLP(p=11, d_model=16, mlp_ratio=2)
        self.assertIs(model.lm_head.weight, model.embedding.weight)
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        logits, loss = model(tokens, tokens[:, 1:])
        self.assertEqual(tuple(logits.shape), (2, 2, 11))
        self.assertTrue(torch.isfinite(loss))
        other = tokens.clone()
        other[:, 2] = (other[:, 2] + 4) % 11
        logits2, _ = model(other)
        torch.testing.assert_close(logits[:, 1], logits2[:, 1])
        loss.backward()
        self.assertIsNotNone(model.embedding.weight.grad)

    def test_transformer_cannot_attend_to_target(self):
        torch.manual_seed(17)
        model = create_model('transformer', p=11, width=16, layers=1,
                             heads=4, mlp_ratio=2, dropout=0.0).eval()
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        b = a.clone()
        b[:, 2] = (b[:, 2] + 5) % 11
        with torch.no_grad():
            logits_a, _ = model(a)
            logits_b, _ = model(b)
        torch.testing.assert_close(logits_a[:, 1, :], logits_b[:, 1, :])


if __name__ == '__main__':
    unittest.main()
