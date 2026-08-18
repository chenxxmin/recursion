"""Unified experiment entry: config parsing, per-task preparation, training.

Extracted from core.py (core.py split, step 4). run_experiment is now a slim
orchestrator: parse config -> branch into _prepare_* -> run_training_engine
-> stage-3 final generation test (final_eval). Depends on
models/datasets/training/final_eval/rules/protocol; never imports core.
core.py re-exports run_experiment (used by its __main__ entry and tests)
and _prepare_mixed_recurrence (used by tests).
"""
import json
import os
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rules import rules_from_config, single_rule_from_task, save_config_extra
from protocol import BATCH_RUN_MERGED_FLAG
from models import FibonacciTransformer, MixedABTransformer
from datasets import (RecurrenceDataset, MixedRecurrenceDataset,
                      DynamicMixedDataset, BucketBatchSampler,
                      collate_fn, collate_fn_masked, collate_fn_predict,
                      mixed_ab_collate_fn, mixed_ab_collate_fn_masked,
                      mixed_ab_collate_fn_predict, dynamic_mixed_collate_fn)
from training import run_training_engine
from final_eval import (_run_mixed_ab_final_test,
                        _run_single_recurrence_final_test)


# ==================== Unified Experiment Entry ====================
# BATCH_RUN_MERGED_FLAG lives in protocol.py (single source shared with
# batch_run.py); run_experiment refuses to run on an unmerged config.


def _load_partial_checkpoint(model, path, device):
    """Resume/transfer weights from a checkpoint (INIT_FROM).

    Only tensors whose name AND shape match are copied; the rest keep their
    fresh init (e.g. rule_head when the source is a single-rule model, or wte
    when vocab sizes differ). Prints a load/skip report so the transfer is
    visible in the log.
    """
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint)
    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(own)
    print(f"[INIT_FROM] {path}")
    print(f"[INIT_FROM] loaded {len(loaded)} tensors; skipped {len(skipped)}: {skipped}")


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
    """Prepare dataset, model, loaders and training params for addition/multiplication/tribonacci/nonlinear/nonlinear_mul."""
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

    default_num_mask = {'addition': 1, 'multiplication': 1, 'tribonacci': 2, 'nonlinear': 1, 'nonlinear_mul': 1}[task]
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
    elif TASK in ('addition', 'multiplication', 'tribonacci', 'nonlinear', 'nonlinear_mul'):
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

    # Curriculum/transfer: optionally initialize weights from a previous run's
    # checkpoint (e.g. train on rule A first, then continue on mixed A+B).
    # Optimizer/scheduler below always start fresh.
    init_from = cfg_main.get('INIT_FROM')
    if init_from:
        _load_partial_checkpoint(model, init_from, device)

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
        _run_mixed_ab_final_test(model, train_dataset, ctx['rules'], ctx['order'],
                                 P, TRAIN_LEN, OOD_LEN, num_mask, device)
    elif post_train_mode == 'single_recurrence':
        _run_single_recurrence_final_test(model, train_dataset,
                                          ctx['recurrence_fn'], ctx['init_len'],
                                          ctx['recurrence_name'], P, TRAIN_LEN,
                                          OOD_LEN, num_mask, device)
