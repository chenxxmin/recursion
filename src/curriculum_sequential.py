"""Task-level sequential learning: single-rule stage1 -> mixed-rule stage2.

Reuses the same model/dataset/collate/training-engine as the standard
mixed_ab pipeline (experiment._prepare_mixed_recurrence + training.
run_training_engine), so the only difference from a joint-training run is
the staged data exposure and warm-start checkpoint transfer.

Usage (one process per ordering, run two in parallel on different GPUs):
  python src/curriculum_sequential.py --seed 17996 --gpu 4 --stage1-rule "1,1"
  python src/curriculum_sequential.py --seed 17996 --gpu 6 --stage1-rule "2,3"

Optional flags:
  --no-mask            : disable online missing (use plain mixed_ab_collate_fn)
  --stage1-epochs N    : Stage1 max-epochs cap (default 1200)
  --stage2-epochs N    : Stage2 max-epochs cap (default 1200)
  --stage1-acc 0.95    : Stage1 accuracy early-stop threshold (default 0.95)
  --stage2-acc 0.98    : Stage2 per-rule accuracy early-stop threshold (default 0.98)
"""
import argparse
import os
import random
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from rules import LinearRecurrenceRule
from datasets import (MixedRecurrenceDataset, make_mixed_missing_collate,
                       mixed_ab_collate_fn)
from models import MixedABTransformer
from training import run_training_engine, best_checkpoint_path
from experiment import _load_partial_checkpoint, _make_loaders, _round_up_pow2


def parse_args():
    p = argparse.ArgumentParser(description='Sequential task-learning curriculum for mixed_ab.')
    p.add_argument('--order', type=int, default=2, help='recurrence order (mixed_ab=2)')
    p.add_argument('--seed', type=int, default=17996)
    p.add_argument('--gpu', type=int, required=True, help='CUDA visible GPU id')
    p.add_argument('--stage1-rule', type=str, required=True,
                   help='comma-separated coeffs for stage1 rule, e.g. "1,1" or "2,3"')
    p.add_argument('--stage1-acc', type=float, default=0.95, help='stage1 early-stop test acc')
    p.add_argument('--stage2-acc', type=float, default=0.98,
                   help='stage2 per-rule early-stop test acc threshold')
    p.add_argument('--stage1-epochs', type=int, default=1200,
                   help='stage1 max epochs cap (no wall-clock limit)')
    p.add_argument('--stage2-epochs', type=int, default=1200,
                   help='stage2 max epochs cap (no wall-clock limit)')
    p.add_argument('--no-mask', action='store_true',
                   help='disable online missing (MISSING_PROB=0, use plain collate)')
    p.add_argument('--output-root', type=str,
                   default='/data/lly/recursion/recursion_results/curriculum_sequential')
    return p.parse_args()


def build_cfg(no_mask=False):
    """Config aligned with mixed_basic_d1024l4r8h4_p127_rules12_miss01_len64_online.

    Set no_mask=True to disable online missing (for nomask comparison runs).
    """
    missing_prob = 0.0 if no_mask else 0.1
    return {
        'P': 127, 'D_MODEL': 1024, 'N_HEAD': 4, 'N_LAYER': 4, 'MLP_RATIO': 8,
        'ORDER': 2, 'TRAIN_LEN': 64, 'OOD_LEN': 128, 'DROPOUT': 0.0,
        'ENTROPY_PENALTY_WEIGHT': 0.0, 'USE_AB_TAG': False,
        'USE_LEARNABLE_PE': False, 'USE_CONDITIONAL_WTE': False,
        'COND_WTE_SHARED_RATIO': 0.0,
        'MISSING_PROB': missing_prob, 'MISS_LEN': 1, 'PREDICT_MISSING': False,
        'BATCH_SIZE': 512, 'LR': 3e-4, 'WEIGHT_DECAY': 1.0,
        'EPOCHS': 6000, 'EVAL_INTERVAL': 20, 'EARLY_STOP_NO_IMPROVE': 3000,
        'USE_AMP': False, 'AMP_DTYPE': 'bfloat16',
        'NO_MASK': no_mask,
    }


def build_save_config(P, rules, ratios, cfg, num_mask):
    """Mirrors experiment._prepare_mixed_recurrence save_config (experiment.py:199-221)."""
    train_len = cfg['TRAIN_LEN']
    ood_len = cfg['OOD_LEN']
    block_size = _round_up_pow2(max(train_len, ood_len))
    return {
        'p': P,
        'ab_pairs': [list(r.coeffs) for r in rules],
        'order': cfg['ORDER'],
        'mixed_ab_max_unique_ratios': ratios,
        'd_model': cfg['D_MODEL'], 'n_head': cfg['N_HEAD'], 'n_layer': cfg['N_LAYER'],
        'block_size': block_size,
        'use_learnable_pe': cfg['USE_LEARNABLE_PE'], 'mlp_ratio': cfg['MLP_RATIO'],
        'use_ab_tag': cfg['USE_AB_TAG'],
        'use_conditional_wte': cfg['USE_CONDITIONAL_WTE'],
        'cond_wte_shared_ratio': cfg['COND_WTE_SHARED_RATIO'],
        'vocab_size': P + 1 + len(rules) if cfg['USE_AB_TAG'] else P + 1,
        'pad_token_id': P + len(rules) if cfg['USE_AB_TAG'] else P,
        'train_len': train_len,
        'missing_prob': cfg['MISSING_PROB'], 'miss_len': cfg['MISS_LEN'],
        'miss_second': False, 'predict_missing': cfg['PREDICT_MISSING'],
        'num_mask': num_mask,
    }


def build_model(rules, cfg, device):
    """MixedABTransformer with num_ab_pairs fixed to final stage count (2)."""
    P = cfg['P']
    block_size = _round_up_pow2(max(cfg['TRAIN_LEN'], cfg['OOD_LEN']))
    model = MixedABTransformer(
        p=P, d_model=cfg['D_MODEL'], n_head=cfg['N_HEAD'],
        n_layer=cfg['N_LAYER'], block_size=block_size,
        dropout=cfg['DROPOUT'],
        entropy_penalty_weight=cfg['ENTROPY_PENALTY_WEIGHT'],
        num_ab_pairs=len(rules), order=cfg['ORDER'],
        use_learnable_pe=cfg['USE_LEARNABLE_PE'],
        mlp_ratio=cfg['MLP_RATIO'],
        use_ab_tag=cfg['USE_AB_TAG'],
        use_conditional_wte=cfg['USE_CONDITIONAL_WTE'],
        cond_wte_shared_ratio=cfg['COND_WTE_SHARED_RATIO'],
    )
    return model, block_size


def build_loaders(rules, ratios, cfg, num_mask):
    """Dataset + collate + BucketBatchSampler loaders.

    Uses make_mixed_missing_collate when MISSING_PROB>0 (online missing),
    otherwise falls back to plain mixed_ab_collate_fn (no corruption).
    """
    P = cfg['P']
    state_space = P ** cfg['ORDER']
    num_samples = [max(1, int(state_space * r)) for r in ratios]
    ds = MixedRecurrenceDataset(
        rules=rules, num_samples=num_samples, length=cfg['TRAIN_LEN'],
        verbose=True, use_ab_tag=cfg['USE_AB_TAG'])
    if cfg.get('NO_MASK', False) or cfg['MISSING_PROB'] == 0.0:
        collate = mixed_ab_collate_fn
    else:
        collate = make_mixed_missing_collate(
            p=P, order=cfg['ORDER'], n_rules=len(rules),
            use_ab_tag=cfg['USE_AB_TAG'],
            missing_prob=cfg['MISSING_PROB'], miss_len=cfg['MISS_LEN'],
            miss_second=False, num_mask=num_mask,
            first_task_weight=1.0, predict_missing=cfg['PREDICT_MISSING'])
    train_loader, test_loader = _make_loaders(
        ds.train_data, ds.test_data, cfg['BATCH_SIZE'], collate)
    return ds, train_loader, test_loader


def log_line(msg, log_path):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run_stage(model, rules, ratios, cfg, num_mask, save_path, log_path, device,
              early_stop_accuracy, max_epochs, stage_name,
              per_rule_threshold=None):
    """One training stage: build loaders, optimizer, call run_training_engine.

    max_epochs replaces wall-clock cap: stage stops after `max_epochs` epochs
    OR when early_stop_accuracy reached (stage1) / per_rule_threshold reached
    by all rules (stage2).
    """
    P = cfg['P']
    ds, train_loader, test_loader = build_loaders(rules, ratios, cfg, num_mask)
    save_config = build_save_config(P, rules, ratios, cfg, num_mask)
    extra_kwargs_fn = lambda ab_indices: {'ab_labels': ab_indices.to(device)}

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['LR'], weight_decay=cfg['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs)

    log_line(f"[{stage_name}] rules={[r.name for r in rules]} ratios={ratios} "
             f"early_stop={early_stop_accuracy} max_epochs={max_epochs} "
             f"per_rule_threshold={per_rule_threshold}", log_path)
    print(f"\n--- {stage_name} output ---", flush=True)

    best_acc, epoch, timed_out = run_training_engine(
        model, train_loader, test_loader, optimizer, scheduler, device,
        epochs=max_epochs, eval_interval=cfg['EVAL_INTERVAL'],
        early_stop_accuracy=early_stop_accuracy,
        early_stop_no_improve=cfg['EARLY_STOP_NO_IMPROVE'],
        save_path=save_path, save_config=save_config,
        num_mask=num_mask, extra_kwargs_fn=extra_kwargs_fn,
        first_task_weight=1.0,
        max_train_hours=None,
        use_amp=cfg['USE_AMP'], amp_dtype=cfg['AMP_DTYPE'],
        per_rule_threshold=per_rule_threshold)
    log_line(f"[{stage_name}] finished: best_acc={best_acc:.4f} epoch={epoch} "
             f"timed_out={timed_out}", log_path)
    return best_acc, epoch, timed_out


def main():
    args = parse_args()
    cfg = build_cfg(no_mask=args.no_mask)
    P = cfg['P']

    # Parse stage1 rule and build fixed stage2 rules (order matches target: [[1,1],[2,3]])
    s1_coeffs = tuple(int(x) for x in args.stage1_rule.split(','))
    rule_a = LinearRecurrenceRule(coeffs=(1, 1), p=P)   # rule_idx=0 in stage2
    rule_b = LinearRecurrenceRule(coeffs=(2, 3), p=P)   # rule_idx=1 in stage2
    stage1_rule = LinearRecurrenceRule(coeffs=s1_coeffs, p=P)
    stage2_rules = [rule_a, rule_b]
    assert s1_coeffs in ((1, 1), (2, 3)), "stage1-rule must be '1,1' or '2,3'"

    # exp_name
    s1_tag = 'a' + ''.join(str(c) for c in s1_coeffs)  # a11 or a23
    mask_tag = 'nomask' if args.no_mask else 'mask'
    exp_name = (f"curriculum_seq_d{cfg['D_MODEL']}l{cfg['N_LAYER']}"
                f"r{cfg['MLP_RATIO']}h{cfg['N_HEAD']}_P{P}"
                f"_rules12_stage1[{s1_tag}]_{mask_tag}_seed{args.seed}_gpu{args.gpu}")
    out_dir = os.path.join(args.output_root, exp_name)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    log_dir = os.path.join(out_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{exp_name}.log')

    # Seed + device + TF32
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if cfg.get('USE_AB_TAG', False) is False and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log_line(f"Using device: {device}  exp={exp_name}", log_path)
    log_line(f"Stage1 rule: {stage1_rule.name} {stage1_rule.coeffs} | "
             f"Stage2 rules: {[r.name for r in stage2_rules]}", log_path)
    log_line(f"Config: P={P} D={cfg['D_MODEL']} L={cfg['N_LAYER']} H={cfg['N_HEAD']} "
             f"R={cfg['MLP_RATIO']} TRAIN_LEN={cfg['TRAIN_LEN']} OOD_LEN={cfg['OOD_LEN']} "
             f"MISSING_PROB={cfg['MISSING_PROB']} MISS_LEN={cfg['MISS_LEN']} "
             f"BATCH={cfg['BATCH_SIZE']} LR={cfg['LR']} "
             f"stage1_epochs={args.stage1_epochs} stage2_epochs={args.stage2_epochs} "
             f"stage1_acc={args.stage1_acc} stage2_per_rule_acc={args.stage2_acc}",
             log_path)

    # Build model (num_ab_pairs=2 fixed for both stages) + save init checkpoint
    model, block_size = build_model(stage2_rules, cfg, device)
    log_line(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
             f"block_size={block_size}", log_path)

    init_pth = os.path.join(ckpt_dir, f'{exp_name}_init.pth')
    init_save_config = build_save_config(P, stage2_rules, [0.7, 0.7], cfg, 2)
    torch.save({'model_state_dict': model.state_dict(),
                'config': init_save_config, 'stage': 'init',
                'exp_name': exp_name}, init_pth)
    log_line(f"[Init] saved initial checkpoint: {init_pth}", log_path)

    model = model.to(device)

    # Stage 1: single-rule training, early-stop at 0.95 or 1200 epochs
    stage1_save = os.path.join(ckpt_dir, f'{exp_name}_stage1.pth')
    s1_best, s1_epoch, s1_to = run_stage(
        model, [stage1_rule], [0.7], cfg, 2, stage1_save, log_path, device,
        early_stop_accuracy=args.stage1_acc, max_epochs=args.stage1_epochs,
        stage_name='Stage1', per_rule_threshold=None)
    stage1_best_pth = best_checkpoint_path(stage1_save)
    log_line(f"[Stage1 done] best={s1_best:.4f}@{s1_epoch} timed_out={s1_to} "
             f"ckpt={stage1_best_pth}", log_path)

    # Stage 2: warm-start from stage1 best, mixed two-rule training,
    # early-stop when all per-rule acc >= 0.98 or 1200 epochs cap.
    _load_partial_checkpoint(model, stage1_best_pth, device)
    log_line(f"[Stage2] warm-start from {stage1_best_pth}", log_path)
    stage2_save = os.path.join(ckpt_dir, f'{exp_name}_stage2.pth')
    s2_best, s2_epoch, s2_to = run_stage(
        model, stage2_rules, [0.7, 0.7], cfg, 2, stage2_save, log_path, device,
        early_stop_accuracy=1.5, max_epochs=args.stage2_epochs,
        stage_name='Stage2', per_rule_threshold=args.stage2_acc)
    log_line(f"[Stage2 done] best={s2_best:.4f}@{s2_epoch} timed_out={s2_to}", log_path)
    log_line(f"All stages complete. Final ckpt: {stage2_save}", log_path)
    log_line(f"Checkpoints: init={init_pth}  stage1_best={stage1_best_pth}  "
             f"stage2_final={stage2_save}", log_path)


if __name__ == '__main__':
    main()
