"""Reconstruct representative clean order-2 splits; CPU only, no model inference."""
import ast
import hashlib
import json
from pathlib import Path
import random
import subprocess

import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
REV = "b85b635^"
SOURCE_PATH = "src/datasets.py"
source = subprocess.check_output(
    ["git", "show", f"{REV}:{SOURCE_PATH}"], cwd=ROOT, text=True
)
node = next(n for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef) and n.name == "RecurrenceDataset")
namespace = {"random": random, "torch": torch, "Dataset": Dataset}
exec(compile(ast.Module(body=[node], type_ignores=[]), SOURCE_PATH, "exec"), namespace)
LegacyDataset = namespace["RecurrenceDataset"]


def segments(rows, length):
    return {tuple(row[j:j + length]) for row in rows
            for j in range(len(row) - length + 1)}


records = []
for seed in [0, 17996, 18318]:
    random.seed(seed)
    p, k, length = 127, 2, 16
    ds = LegacyDataset(p=p, init_len=k, length=length,
                       num_samples=int(0.7 * p ** k),
                       recurrence_fn=lambda values, modulus: sum(values) % modulus,
                       verbose=False)
    ds.run()
    train = [row.tolist() for row in ds.train_samples]
    test = [row.tolist() for row in ds.test_samples]
    starts_train = {tuple(row[:k]) for row in train}
    starts_test = {tuple(row[:k]) for row in test}
    assert not starts_train & starts_test
    assert len(starts_train | starts_test) == p ** k
    assert len(train) == 11290 and len(test) == 4839
    assert all(row[j] == sum(row[j-k:j]) % p
               for row in train + test for j in range(k, length))

    # Independently walk every cycle and count its train/test interfaces.
    visited, split_orbits, interfaces = set(), 0, 0
    for x in range(p):
        for y in range(p):
            if (x, y) in visited:
                continue
            orbit = []
            state = (x, y)
            while state not in visited:
                visited.add(state)
                orbit.append(state)
                state = (state[1], sum(state) % p)
            assert state == orbit[0]
            membership = [state in starts_train for state in orbit]
            if any(membership) and not all(membership):
                split_orbits += 1
                interfaces += sum(membership[j] != membership[(j+1) % len(orbit)]
                                  for j in range(len(orbit)))
    assert split_orbits <= 1

    states_train, states_test = segments(train, k), segments(test, k)
    transitions_train = segments(train, k + 1)
    transitions_test = segments(test, k + 1)
    by_position = [sum(tuple(row[j-k:j+1]) in transitions_train for row in test)
                   for j in range(k, length)]
    seen = sum(by_position)
    total = len(test) * (length-k)
    records.append(dict(
        p=p, order=k, rule_canonical=[1, 1], numeric_length=length, seed=seed,
        clean_single_rule=True, one_based_prediction_positions=list(range(3, 17)),
        train_windows=len(train), test_windows=len(test),
        initial_state_intersection=len(starts_train & starts_test),
        train_unique_local_states=len(states_train),
        test_unique_local_states=len(states_test),
        unique_local_state_intersection=len(states_train & states_test),
        train_unique_transitions=len(transitions_train),
        test_unique_transitions=len(transitions_test),
        unique_transition_intersection=len(transitions_train & transitions_test),
        split_orbits=split_orbits, cyclic_interfaces=interfaces,
        seen_transition_predictions_by_position=by_position,
        seen_transition_predictions=seen, total_predictions=total,
        seen_transition_prediction_fraction=seen / total,
    ))

result = dict(
    scope="Representative reconstructed clean order-2 baseline, not a full paper-wide audit.",
    method="Historical RecurrenceDataset.run; direct tuple-set intersections; no purging.",
    historical_revision=subprocess.check_output(
        ["git", "rev-parse", REV], cwd=ROOT, text=True).strip(),
    source_path=SOURCE_PATH,
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    validation="Disjoint starts exhaust p^2; all recurrence values checked; independent cycle walk.",
    runs=records,
)
output = Path(__file__).with_name("results.json")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
