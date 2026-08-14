import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import sys
import time
import json
import itertools
from torch.utils.data import DataLoader
from rules import rules_from_config, single_rule_from_task, save_config_extra
from protocol import BATCH_RUN_MERGED_FLAG
from models import (RotaryEmbedding, apply_rotary_emb, ATTN_MASK_NEG,
                    CausalSelfAttention, TransformerBlock, _lm_loss,
                    FibonacciTransformer, MixedABTransformer, RULE_LOSS_WEIGHT)
from datasets import (RecurrenceDataset, MixedRecurrenceDataset,
                      DynamicMixedDataset, BucketBatchSampler, BatchTag,
                      collate_fn, collate_fn_masked, collate_fn_predict,
                      mixed_ab_collate_fn, mixed_ab_collate_fn_masked,
                      mixed_ab_collate_fn_predict, dynamic_mixed_collate_fn,
                      generate_dynamic_sample, corrupt_window, missing_token_id)


# ==================== Training Functions ====================
def freeze_partial(model, cond_fix):
    """
    Freeze part of the model for conditional_wte analysis.
    cond_fix: 'WTE'    -> freeze embedding-related params (cond_wte, wte, wpe)
              'LINEAR' -> freeze transformer backbone + rule head
    Returns a list of (param, saved_value) for restoring after optimizer steps.
    """
    if cond_fix is None:
        return []
    
    frozen = []
    for name, param in model.named_parameters():
        freeze = False
        if cond_fix == "WTE":
            if any(k in name for k in ('cond_wte', 'transformer.wte', 'transformer.wpe')):
                freeze = True
        elif cond_fix == "LINEAR":
            if (name.startswith('transformer.h.') or 
                name.startswith('transformer.ln_f.') or
                name.startswith('rule_head.')):
                freeze = True
        if freeze:
            param.requires_grad = False
            frozen.append((param, param.detach().clone()))
    
    print(f"[CondFix] Frozen {cond_fix}: {len(frozen)} param groups")
    return frozen


def _unpack_batch(batch, device, extra_kwargs_fn):
    """Unpack a dataloader batch of the form (x, tag, *payload).

    Returns (x, loss_mask, kwargs, ab_labels, targets_override). The tag (see
    BatchTag) selects the payload layout:
      PLAIN          -> no payload
      MIXED_AB       -> (ab_indices,)
      DYNAMIC_MIXED  -> (loss_mask,)
      MIXED_AB_MASKED -> (ab_indices, loss_mask)
      PLAIN_TARGET   -> (clean_seqs,); targets_override = clean_seqs
      MIXED_AB_TARGET -> (ab_indices, clean_seqs)
    ab_indices are passed through extra_kwargs_fn and also returned as
    ab_labels for per-rule evaluation stats. targets_override (when not None)
    replaces the default targets x[:, 1:] — used by PREDICT_MISSING mode where
    the input is corrupted but the loss targets are the clean values.

    Malformed batches raise immediately instead of being guessed by shape.
    """
    if not isinstance(batch, (list, tuple)) or len(batch) < 2 or not isinstance(batch[1], int):
        raise ValueError(f"Batch must be (x, tag, *payload) with a BatchTag, got: {type(batch)}")
    x = batch[0].to(device)
    tag = batch[1]
    if tag == BatchTag.PLAIN:
        if len(batch) != 2:
            raise ValueError(f"PLAIN batch must have no payload, got {len(batch) - 2} extra item(s)")
        return x, None, {}, None, None
    if tag == BatchTag.MIXED_AB:
        if len(batch) != 3:
            raise ValueError(f"MIXED_AB batch must carry exactly (ab_indices,), got {len(batch) - 2} payload item(s)")
        ab_labels = batch[2].to(device)
        kwargs = extra_kwargs_fn(ab_labels) if extra_kwargs_fn is not None else {}
        return x, None, kwargs, ab_labels, None
    if tag == BatchTag.DYNAMIC_MIXED:
        if len(batch) != 3:
            raise ValueError(f"DYNAMIC_MIXED batch must carry exactly (loss_mask,), got {len(batch) - 2} payload item(s)")
        return x, batch[2].to(device), {}, None, None
    if tag == BatchTag.MIXED_AB_MASKED:
        if len(batch) != 4:
            raise ValueError(f"MIXED_AB_MASKED batch must carry exactly (ab_indices, loss_mask), got {len(batch) - 2} payload item(s)")
        ab_labels = batch[2].to(device)
        kwargs = extra_kwargs_fn(ab_labels) if extra_kwargs_fn is not None else {}
        return x, batch[3].to(device), kwargs, ab_labels, None
    if tag == BatchTag.PLAIN_TARGET:
        if len(batch) != 3:
            raise ValueError(f"PLAIN_TARGET batch must carry exactly (clean_seqs,), got {len(batch) - 2} payload item(s)")
        return x, None, {}, None, batch[2].to(device)
    if tag == BatchTag.MIXED_AB_TARGET:
        if len(batch) != 4:
            raise ValueError(f"MIXED_AB_TARGET batch must carry exactly (ab_indices, clean_seqs), got {len(batch) - 2} payload item(s)")
        ab_labels = batch[2].to(device)
        kwargs = extra_kwargs_fn(ab_labels) if extra_kwargs_fn is not None else {}
        return x, None, kwargs, ab_labels, batch[3].to(device)
    raise ValueError(f"Unknown batch tag: {tag}")


def _sample_seq(item):
    """Extract the sequence tensor from a dataset item.

    Items are plain tensors normally, but become (seq, loss_mask) /
    (seq, loss_mask, rule_idx) tuples when MISSING_PROB is enabled. Consumers
    that index the raw sample lists directly (post-training generation tests)
    must normalize through this helper.
    """
    return item[0] if isinstance(item, tuple) else item


def _print_nan_diagnostics(x, logits, loss_mask, targets):
    """Dump batch statistics to stderr when the training loss goes NaN."""
    print("\n[NaN Alert] loss is NaN", file=sys.stderr, flush=True)
    print(f"  batch shape: {x.shape}, device: {x.device}", file=sys.stderr, flush=True)
    print(f"  input min/max: {x.min().item()} / {x.max().item()}", file=sys.stderr, flush=True)
    print(f"  logits has NaN: {torch.isnan(logits).any().item()}", file=sys.stderr, flush=True)
    print(f"  logits has Inf: {torch.isinf(logits).any().item()}", file=sys.stderr, flush=True)
    print(f"  logits min/max: {logits.min().item()} / {logits.max().item()}", file=sys.stderr, flush=True)
    print(f"  loss_mask sum: {loss_mask.sum().item()}", file=sys.stderr, flush=True)
    print(f"  targets min/max: {targets.min().item()} / {targets.max().item()}", file=sys.stderr, flush=True)


def _accumulate_accuracy(logits, targets, loss_mask, pos_correct, pos_total):
    """Accumulate per-position and overall accuracy counts for one batch.

    Updates pos_correct/pos_total in place. Returns (match, batch_correct,
    batch_samples); `match` is the per-position correctness matrix, also used
    by evaluate() for per-group stats.
    """
    preds = logits.argmax(dim=-1)
    preds = preds[:, :targets.size(1)]  # Align length with targets
    valid_mask = loss_mask > 0
    match = ((preds == targets) & valid_mask).float()
    # One host sync for the whole batch instead of two .item() calls per
    # position; per-position syncs dominated epoch time on small models.
    match_per_pos = match.sum(dim=0).tolist()
    valid_per_pos = valid_mask.float().sum(dim=0).tolist()
    for pos, (correct, total) in enumerate(zip(match_per_pos, valid_per_pos)):
        if total > 0:
            pos_correct[pos] = pos_correct.get(pos, 0) + correct
            pos_total[pos] = pos_total.get(pos, 0) + total
    return match, sum(match_per_pos), sum(valid_per_pos)


def _default_loss_mask(batch_size, target_len, num_mask, first_task_weight=1.0, device='cpu'):
    """Default mask: zeros for the first num_mask targets, ones after.

    first_task_weight upweights the first evaluated target (position num_mask).
    train_epoch passes it through; evaluate and the final generation tests keep
    the default 1.0 so eval loss stays comparable across runs -- an intentional
    train/eval difference.
    """
    mask = torch.zeros(batch_size, target_len, dtype=torch.float, device=device)
    if target_len > num_mask:
        mask[:, num_mask:] = 1.0
        if first_task_weight != 1.0:
            mask[:, num_mask] = first_task_weight
    return mask


def train_epoch(model, dataloader, optimizer, device, num_mask=1, extra_kwargs_fn=None, first_task_weight=1.0, frozen_param_states=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    pos_correct = {}
    pos_total = {}
    
    for batch in dataloader:
        x, loss_mask, kwargs, _, targets_override = _unpack_batch(batch, device, extra_kwargs_fn)
        B = x.size(0)

        # Construct targets (no padding, no PAD token)
        # Input [a,b,c,d,e], targets [b,c,d,e], aligned to L-1
        # PREDICT_MISSING batches carry the clean sequence as override.
        targets = targets_override[:, 1:] if targets_override is not None else x[:, 1:]  # (B, L-1)
        if loss_mask is None:
            loss_mask = _default_loss_mask(B, targets.size(1), num_mask, first_task_weight, device)
        
        optimizer.zero_grad()
        logits, loss, *_ = model(x, targets, loss_mask, **kwargs)
        
        if loss is not None and torch.isnan(loss):
            # NaN diagnostic: print to stderr so it shows up in .err logs
            _print_nan_diagnostics(x, logits, loss_mask, targets)
        
        if loss is not None and not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Restore frozen parameters (AdamW weight decay would otherwise drift them)
            if frozen_param_states:
                with torch.no_grad():
                    for param, saved_value in frozen_param_states:
                        param.copy_(saved_value)
        
        with torch.no_grad():
            _, batch_correct, batch_samples = _accumulate_accuracy(
                logits, targets, loss_mask, pos_correct, pos_total)
            total_correct += batch_correct
            total_samples += batch_samples

        total_loss += loss.item() if loss is not None and not torch.isnan(loss) else 0
    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    per_pos_acc = {pos: pos_correct[pos] / pos_total[pos] for pos in sorted(pos_total.keys())}
    return total_loss / len(dataloader), overall_acc, per_pos_acc

def evaluate(model, dataloader, device, num_mask=1, extra_kwargs_fn=None):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    pos_correct = {}
    pos_total = {}
    group_correct = {}
    group_total = {}
    
    with torch.no_grad():
        for batch in dataloader:
            x, loss_mask, kwargs, ab_labels, targets_override = _unpack_batch(batch, device, extra_kwargs_fn)

            B = x.size(0)

            # PREDICT_MISSING batches carry the clean sequence as override.
            targets = targets_override[:, 1:] if targets_override is not None else x[:, 1:]  # (B, L-1)
            if loss_mask is None:
                # Intentionally no first_task_weight here so eval loss stays
                # comparable across runs; see _default_loss_mask.
                loss_mask = _default_loss_mask(B, targets.size(1), num_mask, device=device)
            
            logits, loss, *_ = model(x, targets, loss_mask, **kwargs)

            match, batch_correct, batch_samples = _accumulate_accuracy(
                logits, targets, loss_mask, pos_correct, pos_total)
            total_correct += batch_correct
            total_samples += batch_samples
            total_loss += loss.item()
            
            # Per-rule/group accuracy for mixed_ab
            if ab_labels is not None:
                # One host sync for the whole batch instead of three .item()
                # calls per sample.
                labels = ab_labels.tolist()
                match_per_sample = match.sum(dim=1).tolist()
                valid_per_sample = loss_mask.float().sum(dim=1).tolist()
                for g, correct, total in zip(labels, match_per_sample, valid_per_sample):
                    group_correct[g] = group_correct.get(g, 0) + correct
                    group_total[g] = group_total.get(g, 0) + total
    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    per_pos_acc = {pos: pos_correct[pos] / pos_total[pos] for pos in sorted(pos_total.keys())}
    group_acc = {g: group_correct[g] / group_total[g] for g in sorted(group_total.keys())}
    return total_loss / len(dataloader), overall_acc, per_pos_acc, group_acc


def run_training_engine(model, train_loader, test_loader, optimizer, scheduler, device,
                        epochs, eval_interval, early_stop_accuracy, early_stop_no_improve,
                        save_path, save_config, num_mask=1, extra_kwargs_fn=None, first_task_weight=1.0,
                        cond_fix=None, cond_fix_start=None,
                        cond_fix_start_a1=None, cond_fix_start_a2=None):
    print(f"\nStart training...")
    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    frozen_param_states = None
    cond_fix_triggered = False
    # After test acc first reaches early_stop_accuracy, keep training for this
    # many extra epochs before stopping (instead of stopping immediately).
    extra_epochs_after_high_acc = 200
    high_acc_epoch = None

    for epoch in range(epochs):
        train_loss, train_acc, train_pos_acc = train_epoch(
            model, train_loader, optimizer, device,
            num_mask=num_mask, extra_kwargs_fn=extra_kwargs_fn,
            first_task_weight=first_task_weight,
            frozen_param_states=frozen_param_states)
        scheduler.step()
        
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            _, test_acc, test_pos_acc, test_group_acc = evaluate(
                model, test_loader, device,
                num_mask=num_mask, extra_kwargs_fn=extra_kwargs_fn)

            # Conditional freeze: freeze cond_fix params when the start condition is met.
            # New behavior: max per-rule acc >= a1 AND min per-rule acc >= a2.
            # Old behavior (backward compat): any per-rule acc >= cond_fix_start.
            if cond_fix is not None and not cond_fix_triggered:
                use_new_cond = (cond_fix_start_a1 is not None) and (cond_fix_start_a2 is not None)
                use_old_cond = cond_fix_start is not None

                if use_new_cond and test_group_acc:
                    max_acc = max(test_group_acc.values())
                    min_acc = min(test_group_acc.values())
                    trigger = (max_acc >= cond_fix_start_a1) and (min_acc >= cond_fix_start_a2)
                    trigger_desc = f"max>=a1 ({cond_fix_start_a1:.2f}) & min>=a2 ({cond_fix_start_a2:.2f})"
                elif use_old_cond:
                    if test_group_acc:
                        trigger = any(acc >= cond_fix_start for acc in test_group_acc.values())
                    else:
                        trigger = test_acc >= cond_fix_start
                    trigger_desc = f"threshold {cond_fix_start:.2f}"
                else:
                    trigger = False
                    trigger_desc = ""

                if trigger:
                    frozen_param_states = freeze_partial(model, cond_fix)
                    cond_fix_triggered = True
                    no_improve = 0  # reset patience after freeze
                    print(f"[CondFix] Epoch {epoch}: triggered at {trigger_desc}")
                    accs = ' '.join(f"{test_group_acc[g]:.2f}" for g in sorted(test_group_acc.keys()))
                    print(f"  per-rule acc: {accs}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += eval_interval
            
            print(f"Epoch {epoch:3d} | Train: Loss={train_loss:.4f} Acc={train_acc:.1%} | "
                  f"Test: Acc={test_acc:.1%} | Best={best_acc:.1%}(@{best_epoch})")
            
            # Print per-position accuracy for test (compact single line)
            if test_pos_acc:
                positions = sorted(test_pos_acc.keys())
                # Skip the first num_mask positions (initial values, not evaluated)
                positions = [p for p in positions if p >= num_mask]
                if positions:
                    start_pos = positions[0]
                    # use_ab_tag=True: targets[pos] predicts x_pos; otherwise predicts x_{pos+1}
                    offset = 0 if getattr(model, 'use_ab_tag', False) else 1
                    start_x = start_pos + offset
                    accs = ' '.join(f"{test_pos_acc[p]:.2f}" for p in positions)
                    print(f"  from x{start_x}: {accs}")
            
            # Print per-rule/group accuracy for mixed_ab (compact single line)
            if test_group_acc:
                accs = ' '.join(f"{test_group_acc[g]:.2f}" for g in sorted(test_group_acc.keys()))
                print(f"  per-rule acc: {accs}")
            
            if test_acc >= early_stop_accuracy and high_acc_epoch is None:
                high_acc_epoch = epoch
                print(f"[Early stop] Epoch {epoch}: test set reached high accuracy, "
                      f"continuing {extra_epochs_after_high_acc} extra epochs")
            if high_acc_epoch is not None and epoch - high_acc_epoch >= extra_epochs_after_high_acc:
                print(f"[Early stop] Epoch {epoch}: finished {extra_epochs_after_high_acc} "
                      f"extra epochs after reaching high accuracy")
                break
            if no_improve >= early_stop_no_improve:
                print(f"[Early stop] No improvement for {early_stop_no_improve} consecutive epochs")
                break
    
    print(f"\n{'='*50}")
    print("Saving model...")
    print(f"Training epochs: {epoch}")
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': save_config,
        'best_accuracy': best_acc,
        'final_epoch': epoch
    }
    torch.save(save_dict, save_path)
    print(f"[OK] Model saved to: {os.path.abspath(save_path)}")
    print(f"  File size: {os.path.getsize(save_path)/1024:.1f} KB")
    print(f"  Best test accuracy: {best_acc:.2%}")
    
    return best_acc, epoch


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


# ==================== Unified Experiment Entry ====================
# BATCH_RUN_MERGED_FLAG lives in protocol.py (single source shared with
# batch_run.py); run_experiment refuses to run on an unmerged config.


def _round_up_pow2(n):
    """Smallest power of two >= n (used for block_size)."""
    return 2 ** (n - 1).bit_length()


def _make_loaders(train_dataset, test_dataset, batch_size, collate_fn):
    """Build train/test DataLoaders with length-grouped batch sampling."""
    train_sampler = BucketBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    test_sampler = BucketBatchSampler(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)
    return train_loader, test_loader


def _print_task_banner(block_size, train_len, number_theory_msg):
    print(f"[Model config] block_size: {block_size}, train length: {train_len}")
    print(f"[Number theory] {number_theory_msg}")
    print()


def _prepare_mixed_recurrence(config, device, order):
    """Prepare dataset, model, loaders and training params for mixed_ab (order=2) / mixed_abc (order=3)."""
    cfg = dict(config.get('main', {}))
    P = cfg['P']
    D_MODEL = cfg['D_MODEL']
    N_HEAD = cfg['N_HEAD']
    N_LAYER = cfg['N_LAYER']
    BATCH_SIZE = cfg['BATCH_SIZE']
    TRAIN_LEN = cfg['TRAIN_LEN']
    OOD_LEN = cfg['OOD_LEN']
    DROPOUT = cfg['DROPOUT']
    MAX_UNIQUE_RATIO = cfg.get('MAX_UNIQUE_RATIO', 0.7)  # default matches src/config.json
    rules = rules_from_config(cfg, order)
    state_space_size = P ** order

    # block_size must accommodate max sequence length plus an optional leading rule token
    max_seq_len = max(TRAIN_LEN, OOD_LEN)
    if cfg.get('USE_AB_TAG', True):
        max_seq_len += 1
    BLOCK_SIZE = _round_up_pow2(max_seq_len)

    # Per-rule exposure ratio for mixed tasks. MAX_UNIQUE_RATIO means exposed (train) proportion.
    if 'MIXED_AB_MAX_UNIQUE_RATIOS' in cfg:
        ratios = cfg['MIXED_AB_MAX_UNIQUE_RATIOS']
    else:
        ratios = [MAX_UNIQUE_RATIO] * len(rules)
    if isinstance(ratios, (int, float)):
        ratios = [ratios] * len(rules)
    assert len(ratios) == len(rules), \
        f"MIXED_AB_MAX_UNIQUE_RATIOS length ({len(ratios)}) must equal number of rules ({len(rules)})"
    NUM_TRAIN_SAMPLES = [max(1, int(state_space_size * r)) for r in ratios]

    # NUM_MASK unset (None) means: the first num_mask positions are initial
    # values and are not evaluated. Default 2, matching the tribonacci task.
    # Computed before dataset construction because MISSING_PROB corruption
    # bakes the mask prefix into per-sample loss masks.
    num_mask_cfg = cfg.get('NUM_MASK')
    num_mask = 2 if num_mask_cfg is None else num_mask_cfg
    # An all-zero loss mask yields a grad-less constant loss and crashes
    # backward() far from the cause; require at least one evaluated position.
    assert num_mask < TRAIN_LEN - 1, \
        f"NUM_MASK ({num_mask}) must be < TRAIN_LEN - 1 ({TRAIN_LEN - 1})"

    missing_prob = cfg.get('MISSING_PROB', 0.0)
    predict_missing = cfg.get('PREDICT_MISSING', False)
    ds = MixedRecurrenceDataset(rules=rules, num_samples=NUM_TRAIN_SAMPLES, length=TRAIN_LEN,
                                verbose=True, use_ab_tag=cfg.get('USE_AB_TAG', True),
                                missing_prob=missing_prob,
                                miss_len=cfg.get('MISS_LEN', 1),
                                miss_second=cfg.get('MISS_SECOND', False),
                                predict_missing=predict_missing,
                                num_mask=num_mask,
                                first_task_weight=cfg.get('FIRST_TASK_WEIGHT', 1.0))
    train_dataset = ds.train_data
    test_dataset = ds.test_data

    _print_task_banner(BLOCK_SIZE, TRAIN_LEN, f"Rules: {[r.name for r in rules]}")

    # Explicit collate selection matching the dataset's item layout.
    if missing_prob > 0 and predict_missing:
        mixed_collate = mixed_ab_collate_fn_predict   # (view, clean, label) -> MIXED_AB_TARGET
    elif missing_prob > 0:
        mixed_collate = mixed_ab_collate_fn_masked    # (seq, mask, label) -> MIXED_AB_MASKED
    else:
        mixed_collate = mixed_ab_collate_fn           # (seq, label) -> MIXED_AB
    train_loader, test_loader = _make_loaders(train_dataset, test_dataset, BATCH_SIZE, mixed_collate)

    model = MixedABTransformer(p=P, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER, block_size=BLOCK_SIZE,
                               dropout=DROPOUT,
                               entropy_penalty_weight=cfg['ENTROPY_PENALTY_WEIGHT'],
                               num_ab_pairs=len(rules),
                               order=order,
                               use_learnable_pe=cfg.get('USE_LEARNABLE_PE', False),
                               mlp_ratio=cfg.get('MLP_RATIO', 4),
                               use_ab_tag=cfg.get('USE_AB_TAG', True),
                               use_conditional_wte=cfg.get('USE_CONDITIONAL_WTE', False),
                               cond_wte_shared_ratio=cfg.get('COND_WTE_SHARED_RATIO', 0.0),
                               )
    extra_kwargs_fn = lambda ab_indices: {'ab_labels': ab_indices.to(device)}
    save_config = {
        'p': P,
        'ab_pairs': [list(r.coeffs) for r in rules],
        'order': order,
        'mixed_ab_max_unique_ratios': ratios,
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'use_learnable_pe': cfg.get('USE_LEARNABLE_PE', False),
        'mlp_ratio': cfg.get('MLP_RATIO', 4),
        'use_ab_tag': cfg.get('USE_AB_TAG', True),
        'use_conditional_wte': cfg.get('USE_CONDITIONAL_WTE', False),
        'cond_wte_shared_ratio': cfg.get('COND_WTE_SHARED_RATIO', 0.0),
        'vocab_size': P + 1 + len(rules) if cfg.get('USE_AB_TAG', True) else P + 1,
        'pad_token_id': P + len(rules) if cfg.get('USE_AB_TAG', True) else P,
        # for analyze_attention.py: in-dist length and corruption settings
        'train_len': TRAIN_LEN,
        'missing_prob': cfg.get('MISSING_PROB', 0.0),
        'miss_len': cfg.get('MISS_LEN', 1),
        'miss_second': cfg.get('MISS_SECOND', False),
        'predict_missing': cfg.get('PREDICT_MISSING', False),
    }
    return {
        'post_train_mode': 'mixed_ab',
        'model': model,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'num_mask': num_mask,
        'extra_kwargs_fn': extra_kwargs_fn,
        'save_config': save_config,
        'cfg': cfg,
        'p': P,
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
        'rules': rules,
        'order': order,
    }


def _prepare_dynamic_mixed(config):
    """Prepare dataset, model, loaders and training params for the dynamic_mixed task."""
    cfg_main = config.get('main', {})
    # batch_run already merges the dynamic_mixed section into main with the
    # correct precedence (main -> task defaults -> experiment override).
    # Do NOT re-apply the section here: merged configs still carry the base
    # section at top level, and re-applying it would clobber experiment
    # overrides (e.g. every N-variant's AB_PAIRS silently reverted to base).
    cfg = dict(cfg_main)

    if cfg.get('MISSING_PROB', 0.0) > 0:
        print("WARNING: MISSING_PROB > 0 is only supported for single-rule tasks; ignoring it for dynamic_mixed.")
    if cfg.get('PREDICT_MISSING', False):
        print("WARNING: PREDICT_MISSING is only supported for single-rule and mixed_ab/mixed_abc tasks; ignoring it for dynamic_mixed.")
    P = cfg['P']
    D_MODEL = cfg_main['D_MODEL']
    N_HEAD = cfg_main['N_HEAD']
    N_LAYER = cfg_main['N_LAYER']
    BATCH_SIZE = cfg_main['BATCH_SIZE']
    DROPOUT = cfg_main['DROPOUT']
    ENTROPY_PENALTY_WEIGHT = cfg_main.get('ENTROPY_PENALTY_WEIGHT', 0.0)
    USE_LEARNABLE_PE = cfg_main.get('USE_LEARNABLE_PE', False)
    AB_PAIRS = [tuple(pair) for pair in cfg.get('AB_PAIRS', [[1, 1], [1, 2]])]
    for pair in AB_PAIRS:
        if len(pair) != 2 or any(not 0 <= c < P for c in pair):
            raise ValueError(f"AB_PAIRS entry {pair} must be two coefficients in [0, {P})")
    NUM_TRAIN_SAMPLES = cfg.get('NUM_TRAIN_SAMPLES', 10000)
    NUM_TEST_SAMPLES = cfg.get('NUM_TEST_SAMPLES', 2000)  # default matches src/config.json
    TRAIN_LEN = cfg.get('TRAIN_LEN', 16)
    OOD_LEN = cfg.get('OOD_LEN', 32)

    # In dynamic_mixed, each generated token is preceded by a flag token,
    # so the actual sequence length is 2*length - 2.
    max_seq_len = 2 * max(TRAIN_LEN, OOD_LEN) - 2
    BLOCK_SIZE = _round_up_pow2(max_seq_len)

    train_dataset = DynamicMixedDataset(
        p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TRAIN_SAMPLES,
        length=TRAIN_LEN, seed=cfg.get('RANDOM_SEED', 42)
    )
    test_dataset = DynamicMixedDataset(
        p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TEST_SAMPLES,
        length=OOD_LEN, seed=cfg.get('RANDOM_SEED', 42) + 1
    )

    _print_task_banner(BLOCK_SIZE, TRAIN_LEN, f"Dynamic mixed rules: {AB_PAIRS}")

    train_loader, test_loader = _make_loaders(train_dataset, test_dataset, BATCH_SIZE, dynamic_mixed_collate_fn)

    model = FibonacciTransformer(
        p=P, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER,
        block_size=BLOCK_SIZE, dropout=DROPOUT,
        entropy_penalty_weight=ENTROPY_PENALTY_WEIGHT,
        use_learnable_pe=USE_LEARNABLE_PE,
        mlp_ratio=cfg.get('MLP_RATIO', 4)
    )
    # vocab_size needs to include flag tokens
    model.vocab_size = P + 1 + len(AB_PAIRS)
    model.pad_token_id = P
    # Expand lm_head and transformer.wte to accommodate flags while preserving weight tying
    if model.lm_head.weight.size(0) < model.vocab_size:
        with torch.no_grad():
            old_head = model.lm_head
            old_wte = model.transformer.wte
            new_wte = nn.Embedding(model.vocab_size, old_wte.embedding_dim)
            new_wte.weight.data[:old_wte.weight.size(0)] = old_wte.weight.data
            model.transformer.wte = new_wte
            # Tie lm_head to wte, matching FibonacciTransformer's original design
            model.lm_head = nn.Linear(old_head.in_features, model.vocab_size, bias=False)
            model.lm_head.weight = model.transformer.wte.weight

    save_config = {
        'p': P,
        'ab_pairs': AB_PAIRS,
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'use_learnable_pe': USE_LEARNABLE_PE,
        'mlp_ratio': cfg.get('MLP_RATIO', 4),
        'vocab_size': model.vocab_size,
        'pad_token_id': model.pad_token_id,
        'recurrence': 'dynamic_mixed',
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
    }
    return {
        'post_train_mode': 'dynamic_mixed',
        'model': model,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'num_mask': 0,  # unused; loss_mask comes from the dataset
        'extra_kwargs_fn': None,
        'save_config': save_config,
        'cfg': cfg,
        'p': P,
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
    }


def _prepare_single_recurrence(config, task):
    """Prepare dataset, model, loaders and training params for addition/multiplication/tribonacci/nonlinear."""
    cfg = dict(config.get('main', {}))
    P = cfg['P']
    D_MODEL = cfg['D_MODEL']
    N_HEAD = cfg['N_HEAD']
    N_LAYER = cfg['N_LAYER']
    BATCH_SIZE = cfg['BATCH_SIZE']
    TRAIN_LEN = cfg['TRAIN_LEN']
    OOD_LEN = cfg['OOD_LEN']
    DROPOUT = cfg['DROPOUT']
    ENTROPY_PENALTY_WEIGHT = cfg.get('ENTROPY_PENALTY_WEIGHT', 0.0)
    MAX_UNIQUE_RATIO = cfg.get('MAX_UNIQUE_RATIO', 0.7)  # default matches src/config.json
    USE_LEARNABLE_PE = cfg.get('USE_LEARNABLE_PE', False)

    BLOCK_SIZE = _round_up_pow2(max(TRAIN_LEN, OOD_LEN))

    default_num_mask = {'addition': 1, 'multiplication': 1, 'tribonacci': 2, 'nonlinear': 1}[task]
    init_len, recurrence_fn, recurrence_name = single_rule_from_task(task, cfg)
    save_extra_config = save_config_extra(task, cfg)

    state_space_size = P ** init_len
    NUM_TRAIN_SAMPLES = max(1, int(state_space_size * MAX_UNIQUE_RATIO))
    # NUM_MASK unset (None) falls back to the task's default mask count.
    num_mask_cfg = cfg.get('NUM_MASK')
    num_mask = default_num_mask if num_mask_cfg is None else num_mask_cfg
    # An all-zero loss mask yields a grad-less constant loss and crashes
    # backward() far from the cause; require at least one evaluated position.
    assert num_mask < TRAIN_LEN - 1, \
        f"NUM_MASK ({num_mask}) must be < TRAIN_LEN - 1 ({TRAIN_LEN - 1})"

    missing_prob = cfg.get('MISSING_PROB', 0.0)
    predict_missing = cfg.get('PREDICT_MISSING', False)
    ds = RecurrenceDataset(
        p=P, recurrence_fn=recurrence_fn, recurrence_name=recurrence_name,
        init_len=init_len, num_samples=NUM_TRAIN_SAMPLES,
        length=TRAIN_LEN,
        missing_prob=missing_prob,
        miss_len=cfg.get('MISS_LEN', 1),
        miss_second=cfg.get('MISS_SECOND', False),
        predict_missing=predict_missing,
        num_mask=num_mask,
        first_task_weight=cfg.get('FIRST_TASK_WEIGHT', 1.0)
    )
    ds.run()
    train_dataset = ds.train_samples
    test_dataset = ds.test_samples

    _print_task_banner(BLOCK_SIZE, TRAIN_LEN, recurrence_name)

    # Explicit collate selection: the dataset's item layout is determined by
    # the missing-value config, and the collate must match it.
    if missing_prob > 0 and predict_missing:
        collate = collate_fn_predict      # (view, clean) -> PLAIN_TARGET
    elif missing_prob > 0:
        collate = collate_fn_masked       # (seq, mask) -> DYNAMIC_MIXED
    else:
        collate = collate_fn              # plain tensors -> PLAIN
    train_loader, test_loader = _make_loaders(train_dataset, test_dataset, BATCH_SIZE, collate)

    model = FibonacciTransformer(
        p=P, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER,
        block_size=BLOCK_SIZE, dropout=DROPOUT,
        entropy_penalty_weight=ENTROPY_PENALTY_WEIGHT,
        use_learnable_pe=USE_LEARNABLE_PE,
        mlp_ratio=cfg.get('MLP_RATIO', 4)
    )
    save_config = {
        'p': P,
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'use_learnable_pe': USE_LEARNABLE_PE,
        'mlp_ratio': cfg.get('MLP_RATIO', 4),
        # for analyze_attention.py: in-dist length and corruption settings
        'train_len': TRAIN_LEN,
        'missing_prob': cfg.get('MISSING_PROB', 0.0),
        'miss_len': cfg.get('MISS_LEN', 1),
        'miss_second': cfg.get('MISS_SECOND', False),
        'predict_missing': cfg.get('PREDICT_MISSING', False),
    }
    save_config.update(save_extra_config)
    return {
        'post_train_mode': 'single_recurrence',
        'model': model,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'num_mask': num_mask,
        'extra_kwargs_fn': None,
        'save_config': save_config,
        'cfg': cfg,
        'p': P,
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
        'recurrence_fn': recurrence_fn,
        'init_len': init_len,
        'recurrence_name': recurrence_name,
    }


def run_experiment(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    cfg_main = config.get('main', {})
    TASK = cfg_main.get('TASK', 'addition')
    EPOCHS = cfg_main['EPOCHS']
    LR = cfg_main['LR']
    SAVE_PATH = cfg_main['SAVE_PATH']
    seed = cfg_main['RANDOM_SEED']
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    if not config.get(BATCH_RUN_MERGED_FLAG):
        print("[Error] Config not merged. Please run via batch_run.py or merge config manually.")
        sys.exit(2)  # non-zero so batch_run records failure instead of a silent "success"

    # ========================================================================
    # Stage 1: Task branch -- prepare dataset, model, loader, training params
    # ========================================================================
    if TASK in ('mixed_ab', 'mixed_abc'):
        order = 2 if TASK == 'mixed_ab' else 3
        ctx = _prepare_mixed_recurrence(config, device, order)
    elif TASK == 'dynamic_mixed':
        ctx = _prepare_dynamic_mixed(config)
    elif TASK in ('addition', 'multiplication', 'tribonacci', 'nonlinear'):
        ctx = _prepare_single_recurrence(config, TASK)
    else:
        print(f"Unknown task: {TASK}")
        sys.exit(2)  # non-zero so batch_run records failure instead of a silent "success"

    model = ctx['model']
    train_dataset = ctx['train_dataset']
    train_loader = ctx['train_loader']
    test_loader = ctx['test_loader']
    num_mask = ctx['num_mask']
    extra_kwargs_fn = ctx['extra_kwargs_fn']
    save_config = ctx['save_config']
    cfg = ctx['cfg']
    P = ctx['p']
    TRAIN_LEN = ctx['train_len']
    OOD_LEN = ctx['ood_len']
    post_train_mode = ctx['post_train_mode']

    # ========================================================================
    # Stage 2: Common training
    # ========================================================================
    # Print a random training sample for sanity check
    try:
        sample_idx = random.randint(0, len(train_dataset) - 1)
        sample = train_dataset[sample_idx]
        if isinstance(sample, (list, tuple)):
            sample_seq = sample[0]
        else:
            sample_seq = sample
        print(f"\nRandom sample (index {sample_idx}): {sample_seq.tolist()}")
    except Exception:
        pass

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, epoch = run_training_engine(
        model, train_loader, test_loader, optimizer, scheduler, device,
        epochs=EPOCHS, eval_interval=cfg['EVAL_INTERVAL'],
        early_stop_accuracy=cfg.get('EARLY_STOP_ACCURACY', 0.99),
        early_stop_no_improve=cfg.get('EARLY_STOP_NO_IMPROVE', 3000),  # default matches src/config.json
        save_path=SAVE_PATH, save_config=save_config,
        num_mask=num_mask, extra_kwargs_fn=extra_kwargs_fn,
        first_task_weight=cfg.get('FIRST_TASK_WEIGHT', 1.0),
        cond_fix=cfg.get('COND_FIX', None),
        cond_fix_start=cfg.get('COND_FIX_START', None),
        cond_fix_start_a1=cfg.get('COND_FIX_START_A1', None),
        cond_fix_start_a2=cfg.get('COND_FIX_START_A2', None)
    )

    # ========================================================================
    # Stage 3: Post-processing (mixed_ab final generation test with exposure split)
    # ========================================================================
    if cfg.get('SKIP_FINAL_GENERATION_TEST', False):
        print("\n[Config] SKIP_FINAL_GENERATION_TEST=true, skipping final generation test.")
    elif post_train_mode == 'mixed_ab':
        rules = ctx['rules']
        order = ctx['order']
        print(f"\n{'='*50}")
        print(f"Final generation test (batched teacher forcing, validate from position {num_mask+1})")
        print(f"{'='*50}")
        model.eval()
        test_cases = list(itertools.product(range(P), repeat=order))

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
                while len(seq) < OOD_LEN:
                    seq.append(next_fn(seq[-order:], P))
                if model.use_ab_tag:
                    seq = [P + rule_idx] + seq
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
            in_dist_mask = torch.arange(target_len, device=device) < (TRAIN_LEN - 1)
            ood_mask = torch.arange(target_len, device=device) >= (TRAIN_LEN - 1)

            stats = _split_exposure_stats(correct, loss_mask, exposed, in_dist_mask, ood_mask)
            _print_exposure_stats(f"rule={rule.name}",
                                  int(exposed.sum().item()), int((~exposed).sum().item()), stats)

            # Per-position accuracy for this rule
            print(f"\n--- Per-position accuracy for rule={rule.name} ---")
            _print_per_position_exposure(correct, loss_mask, exposed,
                                         range(num_mask, target_len), TRAIN_LEN)

    elif post_train_mode == 'single_recurrence':
        recurrence_fn = ctx['recurrence_fn']
        init_len = ctx['init_len']
        recurrence_name = ctx['recurrence_name']
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

        all_inits = list(itertools.product(range(P), repeat=init_len))
        total_states = len(all_inits)

        # Build full true sequences of length OOD_LEN for every initial state
        full_sequences = []
        for init_vals in all_inits:
            seq = list(init_vals)
            while len(seq) < OOD_LEN:
                seq.append(recurrence_fn(seq, P))
            full_sequences.append(seq)
        full_sequences = torch.tensor(full_sequences, dtype=torch.long, device=device)

        correct = _teacher_forced_correct(model, full_sequences, device)

        # Loss mask: ignore first num_mask positions
        loss_mask = _default_loss_mask(total_states, OOD_LEN - 1, num_mask, device=device)

        # Position masks (t indexes targets, absolute position i = t + 1)
        in_dist_mask = torch.arange(OOD_LEN - 1, device=device) < (TRAIN_LEN - 1)
        ood_mask = torch.arange(OOD_LEN - 1, device=device) >= (TRAIN_LEN - 1)

        # Exposed / unexposed mask
        exposed_list = [init_vals in train_seen_inits for init_vals in all_inits]
        exposed = torch.tensor(exposed_list, dtype=torch.bool, device=device)

        stats = _split_exposure_stats(correct, loss_mask, exposed, in_dist_mask, ood_mask)
        _print_exposure_stats(recurrence_name,
                              int(exposed.sum().item()), int((~exposed).sum().item()), stats)

        print(f"\n--- Per-position accuracy ---")
        _print_per_position_exposure(correct, loss_mask, exposed,
                                     range(num_mask, OOD_LEN - 1), TRAIN_LEN)


if __name__ == '__main__':
    # Entry point for the training subprocess spawned by batch_run.py:
    #   python src/core.py <merged_config.json>
    if len(sys.argv) != 2:
        print("Usage: python src/core.py <merged_config.json>", file=sys.stderr)
        sys.exit(2)
    run_experiment(sys.argv[1])
