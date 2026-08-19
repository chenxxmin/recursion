"""Training loop and batch-handling helpers for the recurrence experiments.

Extracted from core.py (core.py split, step 3). Depends on datasets (BatchTag)
only; never imports core. core.py re-exports the externally referenced
symbols defined here (_unpack_batch, _sample_seq), so existing
`from core import ...` users (tests) are unaffected.
"""
import os
import sys

import torch

from datasets import BatchTag


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
      LOSS_MASK      -> (loss_mask,)
      ACTION_MISS    -> (clean_seqs, loss_mask); targets_override = clean_seqs
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
    if tag == BatchTag.LOSS_MASK:
        if len(batch) != 3:
            raise ValueError(f"LOSS_MASK batch must carry exactly (loss_mask,), got {len(batch) - 2} payload item(s)")
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
    if tag == BatchTag.ACTION_MISS:
        if len(batch) != 4:
            raise ValueError(f"ACTION_MISS batch must carry exactly (clean_seqs, loss_mask), got {len(batch) - 2} payload item(s)")
        return x, batch[3].to(device), {}, None, batch[2].to(device)
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

    # cond_fix_start == 0: freeze BEFORE the first gradient step. (Any other
    # start value keeps the threshold semantics: freeze at the first eval
    # whose accuracy reaches it.)
    if cond_fix is not None and cond_fix_start == 0:
        frozen_param_states = freeze_partial(model, cond_fix)
        cond_fix_triggered = True
        print(f"[CondFix] frozen from start (cond_fix_start=0): {cond_fix}")

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
