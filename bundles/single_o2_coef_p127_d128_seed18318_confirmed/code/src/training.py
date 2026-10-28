"""Training loop and batch-handling helpers for the recurrence experiments.

Extracted from core.py (core.py split, step 3). Depends on datasets (BatchTag)
only; never imports core. core.py re-exports the externally referenced
symbols defined here (_unpack_batch), so existing
`from core import ...` users (tests) are unaffected.
"""
import os
import random
import signal
import sys
import tempfile
import time
from collections import deque

import torch

from datasets import BatchTag

# Set by the SIGTERM handler: checked at each eval point to save a resume
# checkpoint and exit (manual graceful stop).
_STOP_REQUESTED = False


def _sigterm_handler(signum, frame):
    # NOTE: do NOT print here — the signal may interrupt an in-flight print()
    # and re-entering the writer raises RuntimeError. Just set the flag.
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


# ==================== Training Functions ====================
def _unpack_batch(batch, device, extra_kwargs_fn):
    """Unpack a dataloader batch of the form (x, tag, *payload).

    Returns (x, loss_mask, kwargs, ab_labels, targets_override). The tag (see
    BatchTag) selects the payload layout:
      PLAIN          -> no payload
      MIXED_AB       -> (ab_indices,)
      LOSS_MASK      -> (loss_mask,)
      ACTION_MISS    -> (clean_seqs, loss_mask); targets_override = clean_seqs
      CORRUPT_FIX    -> (target_seqs, loss_mask); targets_override = target_seqs
      FILL_BLANK     -> (target_seqs, loss_mask); targets_override = target_seqs
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
    if tag == BatchTag.PLAIN_TARGET:
        if len(batch) != 3:
            raise ValueError(f"PLAIN_TARGET batch must carry exactly (clean_seqs,), got {len(batch) - 2} payload item(s)")
        return x, None, {}, None, batch[2].to(device)
    if tag == BatchTag.ACTION_MISS:
        if len(batch) != 4:
            raise ValueError(f"ACTION_MISS batch must carry exactly (clean_seqs, loss_mask), got {len(batch) - 2} payload item(s)")
        return x, batch[3].to(device), {}, None, batch[2].to(device)
    if tag == BatchTag.CORRUPT_FIX:
        if len(batch) != 4:
            raise ValueError(f"CORRUPT_FIX batch must carry exactly (target_seqs, loss_mask), got {len(batch) - 2} payload item(s)")
        return x, batch[3].to(device), {}, None, batch[2].to(device)
    if tag == BatchTag.FILL_BLANK:
        if len(batch) != 4:
            raise ValueError(f"FILL_BLANK batch must carry exactly (target_seqs, loss_mask), got {len(batch) - 2} payload item(s)")
        return x, batch[3].to(device), {}, None, batch[2].to(device)
    if tag == BatchTag.MIXED_AB_TARGET:
        if len(batch) != 4:
            raise ValueError(f"MIXED_AB_TARGET batch must carry exactly (ab_indices, clean_seqs), got {len(batch) - 2} payload item(s)")
        ab_labels = batch[2].to(device)
        kwargs = extra_kwargs_fn(ab_labels) if extra_kwargs_fn is not None else {}
        return x, None, kwargs, ab_labels, batch[3].to(device)
    raise ValueError(f"Unknown batch tag: {tag}")


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


def train_epoch(model, dataloader, optimizer, device, num_mask=1, extra_kwargs_fn=None,
                first_task_weight=1.0, grad_accum_steps=1, grad_clip_norm=1.0,
                opt_diag_interval=0, epoch=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    pos_correct = {}
    pos_total = {}
    n_batches = len(dataloader)
    # Optimizer diagnostics are accumulated on device and synchronized only at
    # epoch end (plus optional sparse step logs), so instrumentation does not
    # add a host sync on every optimizer step.
    diag_norm_sum = torch.zeros((), device=device)
    diag_norm_max = torch.zeros((), device=device)
    diag_clip_count = torch.zeros((), device=device)
    diag_step_count = 0

    for i, batch in enumerate(dataloader):
        x, loss_mask, kwargs, _, targets_override = _unpack_batch(batch, device, extra_kwargs_fn)
        B = x.size(0)

        # Construct targets (no padding, no PAD token)
        # Input [a,b,c,d,e], targets [b,c,d,e], aligned to L-1
        # PREDICT_MISSING batches carry the clean sequence as override.
        targets = targets_override[:, 1:] if targets_override is not None else x[:, 1:]  # (B, L-1)
        if loss_mask is None:
            loss_mask = _default_loss_mask(B, targets.size(1), num_mask, first_task_weight, device)
        
        if i % grad_accum_steps == 0:
            optimizer.zero_grad()
        logits, loss = model(x, targets, loss_mask, **kwargs)

        if loss is not None and torch.isnan(loss):
            # NaN diagnostic: print to stderr so it shows up in .err logs
            _print_nan_diagnostics(x, logits, loss_mask, targets)

        step_now = ((i + 1) % grad_accum_steps == 0) or (i == n_batches - 1)
        if loss is not None and not torch.isnan(loss):
            (loss / grad_accum_steps).backward()
            if step_now:
                # clip_grad_norm_ returns the PRE-CLIP global norm. Passing inf
                # measures it without changing gradients, which lets the same
                # diagnostics work for a no-clipping ablation.
                clip_limit = float('inf') if grad_clip_norm is None else float(grad_clip_norm)
                preclip_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_limit)

                diag_norm_sum += preclip_norm.detach()
                diag_norm_max = torch.maximum(diag_norm_max, preclip_norm.detach())
                if grad_clip_norm is not None:
                    diag_clip_count += (preclip_norm.detach() > clip_limit)
                diag_step_count += 1

                if opt_diag_interval and diag_step_count % opt_diag_interval == 0:
                    lr_now = optimizer.param_groups[0]['lr']
                    clipped = (grad_clip_norm is not None
                               and preclip_norm.item() > clip_limit)
                    epoch_text = '?' if epoch is None else str(epoch)
                    print(f"[OptDiag] epoch={epoch_text} opt_step={diag_step_count} "
                          f"preclip_norm={preclip_norm.item():.6g} "
                          f"clip={grad_clip_norm} clipped={int(clipped)} "
                          f"lr={lr_now:.8g}")

                optimizer.step()
        
        with torch.no_grad():
            _, batch_correct, batch_samples = _accumulate_accuracy(
                logits, targets, loss_mask, pos_correct, pos_total)
            total_correct += batch_correct
            total_samples += batch_samples

        total_loss += loss.item() if loss is not None and not torch.isnan(loss) else 0
    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    per_pos_acc = {pos: pos_correct[pos] / pos_total[pos] for pos in sorted(pos_total.keys())}
    if diag_step_count:
        grad_stats = {
            'preclip_mean': (diag_norm_sum / diag_step_count).item(),
            'preclip_max': diag_norm_max.item(),
            'clip_freq': ((diag_clip_count / diag_step_count).item()
                          if grad_clip_norm is not None else 0.0),
            'optimizer_steps': diag_step_count,
            'lr': optimizer.param_groups[0]['lr'],
        }
    else:
        grad_stats = {
            'preclip_mean': 0.0, 'preclip_max': 0.0, 'clip_freq': 0.0,
            'optimizer_steps': 0, 'lr': optimizer.param_groups[0]['lr'],
        }
    return total_loss / len(dataloader), overall_acc, per_pos_acc, grad_stats

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
            
            logits, loss = model(x, targets, loss_mask, **kwargs)

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


def resume_checkpoint_path(save_path):
    """Path of the wall-clock-timeout resume checkpoint for a given SAVE_PATH."""
    return os.path.splitext(save_path)[0] + '_resume.pth'


def best_checkpoint_path(save_path):
    """Path of the rolling best-weights checkpoint for a given SAVE_PATH."""
    return os.path.splitext(save_path)[0] + '_best.pth'


def latest_checkpoint_path(save_path):
    """Path of the rolling latest-full-state checkpoint for a given SAVE_PATH."""
    return os.path.splitext(save_path)[0] + '_latest.pth'


def _atomic_torch_save(state, path, attempts=3):
    """Keep the previous checkpoint intact if a filesystem write fails."""
    path = os.path.abspath(path)
    for attempt in range(attempts):
        temporary_path = None
        try:
            fd, temporary_path = tempfile.mkstemp(
                prefix=os.path.basename(path) + '.tmp.', dir=os.path.dirname(path))
            os.close(fd)
            torch.save(state, temporary_path)
            os.replace(temporary_path, path)
            return
        except (OSError, RuntimeError) as exc:
            if attempt + 1 == attempts:
                raise
            print(f"[Checkpoint] save failed ({attempt + 1}/{attempts}): {exc}; retrying",
                  file=sys.stderr, flush=True)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.remove(temporary_path)
        time.sleep(attempt + 1)


def save_resume_checkpoint(path, model, optimizer, scheduler, next_epoch,
                           best_acc, best_epoch, no_improve, high_acc_epoch,
                           save_config):
    """Save full training state so a timed-out run can be resumed later.

    Unlike the final model checkpoint (weights only), this captures
    optimizer/scheduler state and RNG states so training continues as
    uninterrupted as possible.
    """
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'next_epoch': next_epoch,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'no_improve': no_improve,
        'high_acc_epoch': high_acc_epoch,
        'rng_python': random.getstate(),
        'rng_torch': torch.get_rng_state(),
        'config': save_config,
    }
    if torch.cuda.is_available():
        state['rng_cuda'] = torch.cuda.get_rng_state_all()
    _atomic_torch_save(state, path)


def run_training_engine(model, train_loader, test_loader, optimizer, scheduler, device,
                        epochs, eval_interval, early_stop_accuracy, early_stop_no_improve,
                        save_path, save_config, num_mask=1, extra_kwargs_fn=None,
                        first_task_weight=1.0, max_train_hours=None, resume_state=None,
                        test_loader_fn=None, grad_accum_steps=1, grad_clip_norm=1.0,
                        opt_diag_interval=0, extra_epochs_after_high_acc=0, epoch_cap=None,
                        extra_eval_fn=None):
    print(f"\nStart training...")
    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    start_epoch = 0
    high_acc_epoch = None
    # Fresh-test regime (test_loader_fn set, e.g. 300k train + fresh 256-case
    # test per eval): early stop requires the MEAN test acc of the last 10
    # evals to exceed early_stop_accuracy, so single-eval spikes don't count.
    fresh_test = test_loader_fn is not None
    recent_test_accs = deque(maxlen=10)
    if resume_state is not None:
        best_acc = resume_state['best_acc']
        best_epoch = resume_state['best_epoch']
        no_improve = resume_state['no_improve']
        high_acc_epoch = resume_state['high_acc_epoch']
        start_epoch = resume_state['next_epoch']
        print(f"[RESUME] continue from epoch {start_epoch}, "
              f"best={best_acc:.2%}(@{best_epoch})")
    train_t0 = time.time()
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    # Graceful stop: SIGTERM (e.g. kill <pid>) triggers checkpoint-save+exit
    # at the next eval point instead of losing in-round progress.
    previous_handler = signal.signal(signal.SIGTERM, _sigterm_handler)
    # Early-stop rule (2026-09-07): static-split regimes stop DIRECTLY at
    # early_stop_accuracy; fresh-test regimes (test_loader_fn set) stop when
    # the last-10-eval mean exceeds it. high_acc_epoch no longer affects
    # stopping and is kept only for resume-checkpoint compatibility.

    # EPOCH_CAP: optional hard stop before `epochs` (LR schedule keeps its
    # full T_max). Normal completion/early-stop/timeout paths unchanged.
    end_epoch = min(epochs, epoch_cap) if epoch_cap else epochs
    epoch = start_epoch
    for epoch in range(start_epoch, end_epoch):
        train_loss, train_acc, train_pos_acc, grad_stats = train_epoch(
            model, train_loader, optimizer, device,
            num_mask=num_mask, extra_kwargs_fn=extra_kwargs_fn,
            first_task_weight=first_task_weight,
            grad_accum_steps=grad_accum_steps,
            grad_clip_norm=grad_clip_norm,
            opt_diag_interval=opt_diag_interval,
            epoch=epoch)
        scheduler.step()

        if epoch % eval_interval == 0 or epoch == end_epoch - 1:
            eval_loader = test_loader_fn() if test_loader_fn is not None else test_loader
            _, test_acc, test_pos_acc, test_group_acc = evaluate(
                model, eval_loader, device,
                num_mask=num_mask, extra_kwargs_fn=extra_kwargs_fn)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                no_improve = 0
                # Rolling best checkpoint (weights only, same layout as the
                # final save): overwritten whenever the best improves, so the
                # best weights are always on disk even if the process dies.
                _atomic_torch_save({'model_state_dict': model.state_dict(),
                            'config': save_config,
                            'best_accuracy': best_acc,
                            'final_epoch': epoch},
                           best_checkpoint_path(save_path))
            else:
                no_improve += eval_interval
            
            print(f"Epoch {epoch:3d} | Train: Loss={train_loss:.4f} Acc={train_acc:.1%} | "
                  f"Test: Acc={test_acc:.1%} | Best={best_acc:.1%}(@{best_epoch})")
            print(f"  Opt: LR={grad_stats['lr']:.8g} | "
                  f"PreClipNorm mean={grad_stats['preclip_mean']:.4g} "
                  f"max={grad_stats['preclip_max']:.4g} | "
                  f"Clip={grad_clip_norm} freq={grad_stats['clip_freq']:.1%} "
                  f"({grad_stats['optimizer_steps']} optimizer steps)")
            if train_pos_acc:
                # Target indices are zero-based in x[:, 1:]; index 3 is x4
                # for the ordinary tetranacci sequence [x0, x1, ...].
                accs = ' '.join(f"target[{p}]={train_pos_acc[p]:.4f}"
                                for p in sorted(train_pos_acc))
                print(f"  TrainPos: {accs}")
            
            # Print per-position accuracy for test (compact single line)
            if test_pos_acc:
                positions = sorted(test_pos_acc.keys())
                # Skip the first num_mask positions (initial values, not evaluated)
                positions = [p for p in positions if p >= num_mask]
                if positions:
                    start_pos = positions[0]
                    offset = 1  # targets[pos] predicts x_{pos+1}
                    start_x = start_pos + offset
                    accs = ' '.join(f"{test_pos_acc[p]:.2f}" for p in positions)
                    print(f"  from x{start_x}: {accs}")
            
            # Print per-rule/group accuracy for mixed_ab (compact single line)
            if test_group_acc:
                accs = ' '.join(f"{test_group_acc[g]:.2f}" for g in sorted(test_group_acc.keys()))
                print(f"  per-rule acc: {accs}")
            # Optional extra eval hook (e.g. POS_PROBE per-blank-position stats
            # for the corrupt_fix/fill_blank variants); returns log lines.
            if extra_eval_fn is not None:
                for line in extra_eval_fn(model, device):
                    print(line, flush=True)

            if fresh_test:
                recent_test_accs.append(test_acc)
                if len(recent_test_accs) == recent_test_accs.maxlen:
                    ma10 = sum(recent_test_accs) / len(recent_test_accs)
                    print(f"  last-10-eval mean test acc: {ma10:.1%}")
                    if ma10 > early_stop_accuracy:
                        print(f"[Early stop] Epoch {epoch}: last-10-eval mean test acc "
                              f"{ma10:.2%} > {early_stop_accuracy:.0%}")
                        break
            else:
                # Static-split regime (e.g. 70/30 state split): stop directly
                # once test acc reaches the threshold.
                if test_acc >= early_stop_accuracy:
                    print(f"[Early stop] Epoch {epoch}: test acc {test_acc:.2%} >= "
                          f"{early_stop_accuracy:.0%}, stopping directly")
                    break
            if no_improve >= early_stop_no_improve:
                print(f"[Early stop] No improvement for {early_stop_no_improve} consecutive epochs")
                break

            # Manual stop requested via SIGTERM: same checkpoint path as timeout.
            if _STOP_REQUESTED:
                resume_path = resume_checkpoint_path(save_path)
                save_resume_checkpoint(resume_path, model, optimizer, scheduler,
                                       epoch + 1, best_acc, best_epoch, no_improve,
                                       high_acc_epoch, save_config)
                print(f"[STOP] Checkpoint saved at epoch {epoch}: "
                      f"{os.path.abspath(resume_path)}")
                return best_acc, epoch, True

            # Wall-clock limit: save a full resume checkpoint and bail out.
            # Checked at eval points so best_acc/no_improve are up to date.
            if max_train_hours is not None and \
                    (time.time() - train_t0) >= max_train_hours * 3600:
                resume_path = resume_checkpoint_path(save_path)
                save_resume_checkpoint(resume_path, model, optimizer, scheduler,
                                       epoch + 1, best_acc, best_epoch, no_improve,
                                       high_acc_epoch, save_config)
                print(f"[TIMEOUT] Reached MAX_TRAIN_HOURS={max_train_hours} at epoch {epoch}; "
                      f"resume checkpoint saved to: {os.path.abspath(resume_path)}")
                return best_acc, epoch, True

            # Rolling latest checkpoint (full training state, RESUME_FROM-
            # compatible): overwritten at every eval point so a crash loses
            # at most one eval interval of progress.
            save_resume_checkpoint(latest_checkpoint_path(save_path), model,
                                   optimizer, scheduler, epoch + 1, best_acc,
                                   best_epoch, no_improve, high_acc_epoch,
                                   save_config)
    
    print(f"\n{'='*50}")
    # Normal completion and early stopping retain a full resumable state.
    save_resume_checkpoint(resume_checkpoint_path(save_path), model, optimizer,
                           scheduler, epoch + 1, best_acc, best_epoch, no_improve,
                           high_acc_epoch, save_config)
    print("Saving model...")
    print(f"Training epochs: {epoch}")
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': save_config,
        'best_accuracy': best_acc,
        'final_epoch': epoch
    }
    _atomic_torch_save(save_dict, save_path)
    print(f"[OK] Model saved to: {os.path.abspath(save_path)}")
    print(f"  File size: {os.path.getsize(save_path)/1024:.1f} KB")
    print(f"  Best test accuracy: {best_acc:.2%}")
    
    return best_acc, epoch, False
