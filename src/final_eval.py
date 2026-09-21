"""Final generation tests (stage 3) and their statistics/printing helpers.

Extracted from core.py (core.py split, step 4). The two stage-3 blocks of
run_experiment (mixed_ab / single_recurrence final generation tests) became
_run_mixed_ab_final_test / _run_single_recurrence_final_test: each whole
block moved verbatim into a function, with the formerly enclosing-scope
variables (P/TRAIN_LEN/OOD_LEN -> p/train_len/ood_len etc.) passed explicitly
as parameters, so RNG call order and stdout are unchanged. Depends on
training (_sample_seq, _default_loss_mask); never imports core. core.py
does not re-export these symbols (no external `core.<name>` references
exist).
"""
import itertools
import random

import torch

from training import _sample_seq, _default_loss_mask


# Batch size for teacher-forced evaluation over the full initial-state space
# in the final generation tests (large for throughput; no gradients taken).
EVAL_BATCH_SIZE = 1024


def _teacher_forced_correct(model, full_sequences, device, ab_labels=None):
    """Teacher-forced next-token correctness over full sequences, in batches.

    full_sequences: (N, L) long tensor; ab_labels: optional (N,) rule indices
    for mixed models. Returns a float tensor (N, L-1) of 0/1 correctness.
    """
    n = full_sequences.size(0)
    all_preds = []
    with torch.no_grad():
        for start in range(0, n, EVAL_BATCH_SIZE):
            batch_seq = full_sequences[start:start + EVAL_BATCH_SIZE]
            if ab_labels is None:
                logits = model(batch_seq)[0]
            else:
                logits = model(batch_seq, ab_labels=ab_labels[start:start + EVAL_BATCH_SIZE])[0]
            all_preds.append(logits.argmax(dim=-1)[:, :-1])
    preds = torch.cat(all_preds, dim=0)
    return (preds == full_sequences[:, 1:]).float()


def _safe_div(a, b):
    return a / b if b > 0 else 0


def _split_exposure_stats(correct, loss_mask, exposed, in_dist_mask, ood_mask):
    """Compute correct/total counts split by exposure and in-dist/OOD.

    `exposed` marks initial states seen during training. Returns a dict with
    keys 'exp_id', 'exp_ood', 'unexp_id', 'unexp_ood', each a (correct, total)
    tuple of ints.
    """
    masked_correct = correct * loss_mask
    exp_corr = masked_correct[exposed]
    unexp_corr = masked_correct[~exposed]
    exp_mask = loss_mask[exposed]
    unexp_mask = loss_mask[~exposed]
    return {
        'exp_id': (int((exp_corr * in_dist_mask).sum().item()),
                   int((exp_mask * in_dist_mask).sum().item())),
        'exp_ood': (int((exp_corr * ood_mask).sum().item()),
                    int((exp_mask * ood_mask).sum().item())),
        'unexp_id': (int((unexp_corr * in_dist_mask).sum().item()),
                     int((unexp_mask * in_dist_mask).sum().item())),
        'unexp_ood': (int((unexp_corr * ood_mask).sum().item()),
                      int((unexp_mask * ood_mask).sum().item())),
    }


def _print_exposure_stats(label, n_exposed, n_unexposed, stats):
    """Print the Exposed/Unexposed x In-dist/OOD summary lines."""
    (eic, eit), (eoc, eot) = stats['exp_id'], stats['exp_ood']
    (uic, uit), (uoc, uot) = stats['unexp_id'], stats['unexp_ood']
    print(f"\n--- Statistics for {label} ---")
    print(f"Exposed    samples: {n_exposed:5d} | In-dist: {eic:5d}/{eit:5d} ({_safe_div(eic, eit)*100:5.1f}%) | OOD: {eoc:5d}/{eot:5d} ({_safe_div(eoc, eot)*100:5.1f}%)")
    print(f"Unexposed  samples: {n_unexposed:5d} | In-dist: {uic:5d}/{uit:5d} ({_safe_div(uic, uit)*100:5.1f}%) | OOD: {uoc:5d}/{uot:5d} ({_safe_div(uoc, uot)*100:5.1f}%)")


def _print_per_position_exposure(correct, loss_mask, exposed, positions, train_len):
    """Print per-position accuracy split by exposure and in-dist/OOD.

    positions iterates target indices; absolute position i = pos + 1, and
    positions before train_len count as in-distribution.
    """
    for pos in positions:
        i = pos + 1
        loss_pos = loss_mask[:, pos]
        corr_pos = correct[:, pos]
        exp_pos = exposed & (loss_pos > 0)
        unexp_pos = (~exposed) & (loss_pos > 0)
        is_id = i < train_len
        parts = []
        for label, mask in [('Exposed-ID' if is_id else 'Exposed-OOD', exp_pos),
                            ('Unexposed-ID' if is_id else 'Unexposed-OOD', unexp_pos)]:
            if mask.any():
                cnt = int(corr_pos[mask].sum().item())
                tot = int(mask.sum().item())
                parts.append(f"{label}: {cnt}/{tot} ({_safe_div(cnt, tot)*100:5.1f}%)")
        if parts:
            print(f"  Position {i:3d}: " + " | ".join(parts))


def _run_mixed_ab_final_test(model, train_dataset, rules, order, p,
                             train_len, ood_len, num_mask, device):
    """Stage-3 final generation test for mixed_ab/mixed_abc (exposure split).

    Verbatim move of the mixed_ab stage-3 block from run_experiment; the
    enclosing-scope variables are passed in explicitly.
    """
    print(f"\n{'='*50}")
    print(f"Final generation test (batched teacher forcing, validate from position {num_mask+1})")
    print(f"{'='*50}")
    model.eval()
    test_cases = list(itertools.product(range(p), repeat=order))

    # Collect exposed initial states per rule from the training set
    tag_offset = 1 if model.use_ab_tag else 0
    train_seen_inits = {idx: set() for idx in range(len(rules))}
    for i in range(len(train_dataset)):
        item = train_dataset[i]
        seq, rule_idx = _sample_seq(item), item[-1]
        init_state = tuple(seq[tag_offset:tag_offset + order].tolist())
        train_seen_inits[rule_idx].add(init_state)

    for rule_idx, rule in enumerate(rules):
        print(f"\n--- Rule {rule_idx+1}: {rule.name} {rule.coeffs} ---")
        seen = train_seen_inits[rule_idx]
        next_fn = rule.next_fn()

        # Build full true sequences for all initial states under this rule
        full_sequences = []
        for init_state in test_cases:
            seq = list(init_state)
            while len(seq) < ood_len:
                seq.append(next_fn(seq[-order:], p))
            if model.use_ab_tag:
                seq = [p + rule_idx] + seq
            full_sequences.append(seq)
        full_sequences = torch.tensor(full_sequences, dtype=torch.long, device=device)

        ab_labels = torch.full((len(test_cases),), rule_idx, dtype=torch.long, device=device)
        exposed_list = [init_state in seen for init_state in test_cases]
        exposed = torch.tensor(exposed_list, dtype=torch.bool, device=device)

        seq_len = full_sequences.size(1)
        target_len = seq_len - 1

        correct = _teacher_forced_correct(model, full_sequences, device, ab_labels=ab_labels)

        # Loss mask: ignore first num_mask positions
        loss_mask = _default_loss_mask(len(test_cases), target_len, num_mask, device=device)

        # Position masks (absolute position i = t + 1)
        in_dist_mask = torch.arange(target_len, device=device) < (train_len - 1)
        ood_mask = torch.arange(target_len, device=device) >= (train_len - 1)

        stats = _split_exposure_stats(correct, loss_mask, exposed, in_dist_mask, ood_mask)
        _print_exposure_stats(f"rule={rule.name}",
                              int(exposed.sum().item()), int((~exposed).sum().item()), stats)

        # Per-position accuracy for this rule
        print(f"\n--- Per-position accuracy for rule={rule.name} ---")
        _print_per_position_exposure(correct, loss_mask, exposed,
                                     range(num_mask, target_len), train_len)


def _run_single_recurrence_final_test(model, train_dataset, recurrence_fn,
                                      init_len, recurrence_name, p,
                                      train_len, ood_len, num_mask, device,
                                      data_mode='full_split',
                                      num_test_samples=256):
    """Stage-3 final generation test for single-recurrence tasks.

    Verbatim move of the single_recurrence stage-3 block from run_experiment;
    the enclosing-scope variables are passed in explicitly.
    """
    print(f"\n{'='*50}")
    print(f"Final generation test (batched teacher forcing, validate from position {num_mask+1})")
    print(f"{'='*50}")
    model.eval()
    # Collect exposed initial states from training set
    train_seen_inits = set()
    for i in range(len(train_dataset)):
        seq = _sample_seq(train_dataset[i])
        init_state = tuple(seq[:init_len].tolist())
        train_seen_inits.add(init_state)

    state_space = p ** init_len
    if data_mode == 'sampled_fresh_test':
        # Match the online evaluation regime: independent draws with replacement.
        all_inits = []
        for _ in range(num_test_samples):
            idx = random.randrange(state_space)
            vals = []
            for _ in range(init_len):
                vals.insert(0, idx % p)
                idx //= p
            all_inits.append(tuple(vals))
        print(f"[Final test] sampled_fresh_test: {num_test_samples} initial states "
              f"drawn with replacement from {state_space}")
    else:
        all_inits = list(itertools.product(range(p), repeat=init_len))
    total_states = len(all_inits)

    # Build full true sequences of length OOD_LEN for every initial state
    full_sequences = []
    for init_vals in all_inits:
        seq = list(init_vals)
        while len(seq) < ood_len:
            seq.append(recurrence_fn(seq, p))
        full_sequences.append(seq)
    full_sequences = torch.tensor(full_sequences, dtype=torch.long, device=device)

    correct = _teacher_forced_correct(model, full_sequences, device)

    # Loss mask: ignore first num_mask positions
    loss_mask = _default_loss_mask(total_states, ood_len - 1, num_mask, device=device)

    # Position masks (t indexes targets, absolute position i = t + 1)
    in_dist_mask = torch.arange(ood_len - 1, device=device) < (train_len - 1)
    ood_mask = torch.arange(ood_len - 1, device=device) >= (train_len - 1)

    # Exposed / unexposed mask
    exposed_list = [init_vals in train_seen_inits for init_vals in all_inits]
    exposed = torch.tensor(exposed_list, dtype=torch.bool, device=device)

    stats = _split_exposure_stats(correct, loss_mask, exposed, in_dist_mask, ood_mask)
    _print_exposure_stats(recurrence_name,
                          int(exposed.sum().item()), int((~exposed).sum().item()), stats)

    print(f"\n--- Per-position accuracy ---")
    _print_per_position_exposure(correct, loss_mask, exposed,
                                 range(num_mask, ood_len - 1), train_len)
