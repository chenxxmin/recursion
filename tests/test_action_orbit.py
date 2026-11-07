"""CPU checks of mixed-orbit action data, both two-rule orders."""
import json
import random
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from datasets import MixedRecurrenceDataset
from rules import rules_from_config
from experiment import _prepare_action
from scenarios import normalize_task_config

torch.set_num_threads(2)
for order, coeffs in [(2, [[1, 2], [3, 1]]), (3, [[1, 2, 3], [3, 1, 2]])]:
    p = 7
    for counts in [10, [9, 13], None]:
        for ood in [8, 10]:
            cfg = json.loads((ROOT / 'src/config.json').read_text())['main']
            cfg.update(TASK='action', ORDER=order, RULES=coeffs, P=p,
                       D_MODEL=16, N_LAYER=1, N_HEAD=4, TRAIN_LEN=8, OOD_LEN=ood,
                       BATCH_SIZE=8, NUM_TRAIN_SAMPLES=counts, NUM_TEST_SAMPLES=5,
                       ACTION_RULE_MODE='per_sequence_orbit', MISSING_PROB=0,
                       FRESH_TEST_PER_EVAL=True, RANDOM_SEED=42)
            random.seed(42)
            reference = MixedRecurrenceDataset(rules_from_config(coeffs, p),
                                                num_samples=counts if counts is not None else int(p**order * 0.7), length=8, verbose=False)
            random.seed(42)
            ctx = _prepare_action({'main': cfg}, normalize_task_config('action', cfg))
            sets = []
            for split, ref in [('train_dataset', reference.train_data),
                               ('test_dataset', reference.test_data)]:
                actual = ctx[split]
                assert len(actual) == len(ref)
                keys = set()
                for (seq, mask), (values, rid) in zip(actual, ref):
                    decoded = torch.cat((seq[:order], seq[order+1::2]))
                    length = 8 if split == 'train_dataset' else ood
                    assert len(decoded) == length
                    assert torch.equal(decoded[:min(8, length)], values[:min(8, length)])
                    assert set(seq[order::2].tolist()) == {p+1+rid}
                    assert mask.nonzero().flatten().tolist() == list(range(order, len(seq)-1, 2))
                    for k in range(order, length):
                        assert decoded[k].item() == sum(c*decoded[k-1-i].item()
                                                       for i,c in enumerate(coeffs[rid])) % p
                    keys.add((rid, *decoded[:order].tolist()))
                assert len(keys) == len(actual)
                sets.append(keys)
            assert sets[0].isdisjoint(sets[1])
            assert len(sets[0] | sets[1]) == 2*p**order
            state = random.getstate()
            fresh = ctx['test_loader_fn']()
            assert random.getstate() == state
            assert len(fresh.dataset) == 10
            assert ctx['save_config']['action_rule_mode'] == 'per_sequence_orbit'
            x, _, mask = next(iter(ctx['train_loader']))
            logits, loss = ctx['model'](x, x[:,1:], mask)
            assert torch.isfinite(loss)
            loss.backward()
    print(f'PASS order={order} ntask=2: exact mixed split/order, unique full coverage, '
          'flags, masks, recurrence, OOD, fresh test, CPU forward/backward')
