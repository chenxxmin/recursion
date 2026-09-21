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
                      ActionDataset, BucketBatchSampler,
                      collate_fn, make_missing_collate,
                      mixed_ab_collate_fn, make_mixed_missing_collate,
                      action_collate_fn, make_action_missing_collate)
from training import run_training_engine, resume_checkpoint_path
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


def _load_resume_checkpoint(path, model, optimizer, scheduler, device):
    """Load full training state saved by a wall-clock timeout (RESUME_FROM).

    Restores model/optimizer/scheduler state plus RNG states, and returns the
    raw checkpoint dict (run_training_engine reads next_epoch/best_acc/etc).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    random.setstate(ckpt['rng_python'])
    # map_location may have moved RNG ByteTensors off CPU; move them back.
    torch.set_rng_state(ckpt['rng_torch'].cpu())
    if torch.cuda.is_available() and 'rng_cuda' in ckpt:
        for i, s in enumerate(ckpt['rng_cuda']):
            if i < torch.cuda.device_count():
                torch.cuda.set_rng_state(s.cpu(), device=i)
    print(f"[RESUME_FROM] {path}")
    print(f"[RESUME_FROM] next_epoch={ckpt['next_epoch']}, "
          f"best={ckpt['best_acc']:.2%}(@{ckpt['best_epoch']})")
    return ckpt


def _round_up_pow2(n):
    """Smallest power of two >= n (used for block_size)."""
    return 2 ** (n - 1).bit_length()


def _make_loaders(train_dataset, test_dataset, batch_size, collate_fn):
    """Build train/test DataLoaders with length-grouped batch sampling."""
    train_sampler = BucketBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    test_sampler = BucketBatchSampler(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
    test_loader = _make_test_loader(test_dataset, batch_size, collate_fn, test_sampler)
    return train_loader, test_loader


def _make_test_loader(test_dataset, batch_size, collate_fn, sampler=None):
    """Build the eval DataLoader (separated so fresh-test-per-eval can rebuild it)."""
    if sampler is None:
        sampler = BucketBatchSampler(test_dataset, batch_size=batch_size, shuffle=False)
    return DataLoader(test_dataset, batch_sampler=sampler, collate_fn=collate_fn)


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

    # Explicit NUM_TRAIN_SAMPLES overrides the ratio-derived counts (same
    # semantics as the single-rule path): a scalar gives every rule the same
    # count, a list sets per-rule counts.
    nts = cfg.get('NUM_TRAIN_SAMPLES')
    if nts is not None:
        if isinstance(nts, (int, float)):
            nts = [nts] * len(rules)
        assert len(nts) == len(rules), \
            f"NUM_TRAIN_SAMPLES length ({len(nts)}) must equal number of rules ({len(rules)})"
        NUM_TRAIN_SAMPLES = [max(1, int(n)) for n in nts]
        # Keep `ratios` defined for save_config: actual exposure per rule.
        ratios = [n / state_space_size for n in NUM_TRAIN_SAMPLES]
    else:
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
    num_mask_cfg = cfg.get('NUM_MASK')
    num_mask = 2 if num_mask_cfg is None else num_mask_cfg
    # An all-zero loss mask yields a grad-less constant loss and crashes
    # backward() far from the cause; require at least one evaluated position.
    assert num_mask < TRAIN_LEN - 1, \
        f"NUM_MASK ({num_mask}) must be < TRAIN_LEN - 1 ({TRAIN_LEN - 1})"

    missing_prob = cfg.get('MISSING_PROB', 0.0)
    predict_missing = cfg.get('PREDICT_MISSING', False)
    ds = MixedRecurrenceDataset(rules=rules, num_samples=NUM_TRAIN_SAMPLES, length=TRAIN_LEN,
                                verbose=True, use_ab_tag=cfg.get('USE_AB_TAG', True))
    train_dataset = ds.train_data
    test_dataset = ds.test_data

    _print_task_banner(BLOCK_SIZE, TRAIN_LEN, f"Rules: {[r.name for r in rules]}")

    # Explicit collate selection: with MISSING_PROB, corruption is applied
    # on the fly in the collate (fresh randomness per batch; datasets store
    # clean windows only).
    if missing_prob > 0:
        mixed_collate = make_mixed_missing_collate(
            p=P, order=order, n_rules=len(rules), use_ab_tag=cfg.get('USE_AB_TAG', True),
            missing_prob=missing_prob, miss_len=cfg.get('MISS_LEN', 1),
            miss_second=cfg.get('MISS_SECOND', False), num_mask=num_mask,
            first_task_weight=cfg.get('FIRST_TASK_WEIGHT', 1.0),
            predict_missing=predict_missing)
    else:
        mixed_collate = mixed_ab_collate_fn           # (seq, label) -> MIXED_AB
    train_loader, test_loader = _make_loaders(train_dataset, test_dataset, BATCH_SIZE, mixed_collate)

    # FRESH_TEST_PER_EVAL (mixed): at every eval, sample NUM_TEST_SAMPLES fresh
    # initial states PER RULE from each rule's full state space and roll out
    # OOD_LEN windows (num_samples=0 sends them all to the test split). Same
    # regime as the single-rule fresh test; eval cost stays tiny even when the
    # static split has millions of held-out states.
    test_loader_fn = None
    if cfg.get('FRESH_TEST_PER_EVAL', False):
        NUM_TEST_SAMPLES = cfg.get('NUM_TEST_SAMPLES', 256)
        fresh_rng = random.Random(cfg.get('RANDOM_SEED', 42) + 10 ** 6 + 7)

        def test_loader_fn():  # noqa: F811 (intentional closure name)
            fresh_seed = fresh_rng.randrange(2 ** 31)
            rng_state = random.getstate()
            random.seed(fresh_seed)
            fresh_samples, fresh_labels = [], []
            for idx, rule in enumerate(rules):
                fresh_ds = RecurrenceDataset(
                    p=P, recurrence_fn=rule.next_fn(), recurrence_name=rule.name,
                    init_len=rule.order, num_samples=0, length=OOD_LEN,
                    verbose=False, state_cap=NUM_TEST_SAMPLES)
                fresh_ds.run()
                fresh_samples += fresh_ds.test_samples
                fresh_labels += [idx] * len(fresh_ds.test_samples)
            random.setstate(rng_state)
            print(f"[FreshTest] rebuilt mixed test set: n={NUM_TEST_SAMPLES}/rule x {len(rules)}, len={OOD_LEN}, seed={fresh_seed}")
            return _make_test_loader(list(zip(fresh_samples, fresh_labels)),
                                     BATCH_SIZE, mixed_collate)

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
        'data_mode': data_mode,
        'num_train_samples': NUM_TRAIN_SAMPLES,
    }
    return {
        'post_train_mode': 'mixed_ab',
        'model': model,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'test_loader_fn': test_loader_fn,
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


def _prepare_action(config, order=2):
    """Prepare dataset, model, loaders and training params for the action task family.

    order: recurrence order. order=2 reads AB_PAIRS (task 'action'),
    order=3 reads ABC_PAIRS (task 'action_trib'). The ACTION coefficient
    convention applies: the first coefficient multiplies the OLDEST value
    (opposite of LinearRecurrenceRule).
    """
    cfg_main = config.get('main', {})
    # batch_run already merges the action section into main with the
    # correct precedence (main -> task defaults -> experiment override).
    # Do NOT re-apply the section here: merged configs still carry the base
    # section at top level, and re-applying it would clobber experiment
    # overrides (e.g. every N-variant's AB_PAIRS silently reverted to base).
    cfg = dict(cfg_main)

    missing_prob = cfg.get('MISSING_PROB', 0.0)
    miss_len = cfg.get('MISS_LEN', 1)
    if missing_prob > 0 and cfg.get('MISS_SECOND', False):
        print("WARNING: MISS_SECOND is not supported for action; ignoring it.")
    P = cfg['P']
    D_MODEL = cfg_main['D_MODEL']
    N_HEAD = cfg_main['N_HEAD']
    N_LAYER = cfg_main['N_LAYER']
    BATCH_SIZE = cfg_main['BATCH_SIZE']
    DROPOUT = cfg_main['DROPOUT']
    ENTROPY_PENALTY_WEIGHT = cfg_main.get('ENTROPY_PENALTY_WEIGHT', 0.0)
    USE_LEARNABLE_PE = cfg_main.get('USE_LEARNABLE_PE', False)
    pairs_key = 'AB_PAIRS' if order == 2 else 'ABC_PAIRS'
    pairs_default = [[1, 1], [1, 2]] if order == 2 else [[1, 1, 1], [1, 2, 3]]
    AB_PAIRS = [tuple(pair) for pair in cfg.get(pairs_key, pairs_default)]
    for pair in AB_PAIRS:
        if len(pair) != order or any(not 0 <= c < P for c in pair):
            raise ValueError(f"{pairs_key} entry {pair} must be {order} coefficients in [0, {P})")
    NUM_TRAIN_SAMPLES = cfg.get('NUM_TRAIN_SAMPLES', 10000)
    NUM_TEST_SAMPLES = cfg.get('NUM_TEST_SAMPLES', 2000)  # default matches src/config.json
    TRAIN_LEN = cfg.get('TRAIN_LEN', 16)
    OOD_LEN = cfg.get('OOD_LEN', 32)

    # In action, each generated token is preceded by a flag token,
    # so the actual sequence length is 2*length - order.
    max_seq_len = 2 * max(TRAIN_LEN, OOD_LEN) - order
    BLOCK_SIZE = _round_up_pow2(max_seq_len)

    train_dataset = ActionDataset(
        p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TRAIN_SAMPLES,
        length=TRAIN_LEN, seed=cfg.get('RANDOM_SEED', 42), order=order
    )
    test_dataset = ActionDataset(
        p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TEST_SAMPLES,
        length=OOD_LEN, seed=cfg.get('RANDOM_SEED', 42) + 1, order=order
    )

    _print_task_banner(BLOCK_SIZE, TRAIN_LEN, f"Dynamic mixed rules: {AB_PAIRS}")

    # With MISSING_PROB, corruption is applied on the fly in the collate
    # (fresh randomness per batch; datasets store clean samples only).
    collate = (make_action_missing_collate(p=P, missing_prob=missing_prob,
                                           miss_len=miss_len, order=order)
               if missing_prob > 0 else action_collate_fn)
    train_loader, test_loader = _make_loaders(train_dataset, test_dataset, BATCH_SIZE, collate)

    # FRESH_TEST_PER_EVAL: rebuild the test set at every eval with a fresh seed
    # (same length as train — no OOD split in this regime). A dedicated RNG
    # keeps the fresh-seed stream reproducible per run but distinct per eval.
    test_loader_fn = None
    if cfg.get('FRESH_TEST_PER_EVAL', False):
        fresh_rng = random.Random(cfg.get('RANDOM_SEED', 42) + 10 ** 6 + 7)

        def test_loader_fn():  # noqa: F811 (intentional closure name)
            fresh_seed = fresh_rng.randrange(2 ** 31)
            ds = ActionDataset(
                p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TEST_SAMPLES,
                length=OOD_LEN, seed=fresh_seed, order=order)
            print(f"[FreshTest] rebuilt test set: n={NUM_TEST_SAMPLES}, len={OOD_LEN}, seed={fresh_seed}")
            return _make_test_loader(ds, BATCH_SIZE, collate)

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
        'order': order,
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'use_learnable_pe': USE_LEARNABLE_PE,
        'mlp_ratio': cfg.get('MLP_RATIO', 4),
        'vocab_size': model.vocab_size,
        'pad_token_id': model.pad_token_id,
        'recurrence': 'action',
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
        'missing_prob': missing_prob,
        'miss_len': miss_len,
    }
    return {
        'post_train_mode': 'action',
        'model': model,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'test_loader_fn': test_loader_fn,
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

    default_num_mask = {'addition': 1, 'multiplication': 1, 'tribonacci': 2, 'tetranacci': 3, 'nonlinear': 1, 'nonlinear_mul': 1}[task]
    init_len, recurrence_fn, recurrence_name = single_rule_from_task(task, cfg)
    save_extra_config = save_config_extra(task, cfg)

    # Explicit data-generation strategy.
    #   full_split: traverse the full state space, then split into train/test.
    #   sampled_fresh_test: draw a fixed unique training set without materializing
    #     the state space; every evaluation draws a fresh test set with replacement.
    data_mode = cfg.get('DATA_MODE')
    if data_mode is None:
        # Backward compatibility for existing experiment files.
        data_mode = 'sampled_fresh_test' if cfg.get('FRESH_TEST_PER_EVAL', False) else 'full_split'
    if data_mode not in ('full_split', 'sampled_fresh_test'):
        raise ValueError(
            f"DATA_MODE must be 'full_split' or 'sampled_fresh_test', got {data_mode!r}")
    print(f"[Data mode] {data_mode}")

    state_space_size = P ** init_len
    NUM_TRAIN_SAMPLES = cfg.get('NUM_TRAIN_SAMPLES')
    if data_mode == 'full_split':
        if NUM_TRAIN_SAMPLES is None:
            NUM_TRAIN_SAMPLES = max(1, int(state_space_size * MAX_UNIQUE_RATIO))
        if NUM_TRAIN_SAMPLES > state_space_size:
            raise ValueError(
                f"NUM_TRAIN_SAMPLES ({NUM_TRAIN_SAMPLES}) exceeds state space "
                f"({state_space_size}) in full_split mode")
    else:
        if NUM_TRAIN_SAMPLES is None:
            raise ValueError(
                "sampled_fresh_test mode requires explicit NUM_TRAIN_SAMPLES")
        if NUM_TRAIN_SAMPLES > state_space_size:
            raise ValueError(
                f"NUM_TRAIN_SAMPLES ({NUM_TRAIN_SAMPLES}) exceeds state space "
                f"({state_space_size}); unique training sampling is impossible")
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
    )
    if data_mode == 'full_split':
        ds.run_full_split()
        train_dataset = ds.train_samples
        test_dataset = ds.test_samples
    else:
        train_dataset = ds.sample_unique_train()
        test_dataset = []

    _print_task_banner(BLOCK_SIZE, TRAIN_LEN, recurrence_name)

    # Explicit collate selection: with MISSING_PROB, corruption is applied on
    # the fly in the collate (fresh randomness per batch; the dataset stores
    # clean windows only).
    if missing_prob > 0:
        collate = make_missing_collate(
            p=P, init_len=init_len, missing_prob=missing_prob,
            miss_len=cfg.get('MISS_LEN', 1),
            miss_second=cfg.get('MISS_SECOND', False),
            num_mask=num_mask,
            first_task_weight=cfg.get('FIRST_TASK_WEIGHT', 1.0),
            predict_missing=predict_missing)
    else:
        collate = collate_fn              # plain tensors -> PLAIN
    train_sampler = BucketBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate)

    test_loader = None
    test_loader_fn = None
    if data_mode == 'full_split':
        test_loader = _make_test_loader(test_dataset, BATCH_SIZE, collate)
    else:
        NUM_TEST_SAMPLES = cfg.get('NUM_TEST_SAMPLES', 256)
        fresh_rng = random.Random(cfg.get('RANDOM_SEED', 42) + 10 ** 6 + 7)

        def test_loader_fn():  # noqa: F811 (intentional closure name)
            fresh_seed = fresh_rng.randrange(2 ** 31)
            eval_rng = random.Random(fresh_seed)
            fresh_samples = ds.sample_random_windows(
                NUM_TEST_SAMPLES, length=OOD_LEN, rng=eval_rng)
            print(f"[FreshTest] sampled with replacement: "
                  f"n={NUM_TEST_SAMPLES}, len={OOD_LEN}, seed={fresh_seed}")
            return _make_test_loader(fresh_samples, BATCH_SIZE, collate)

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
        'test_loader_fn': test_loader_fn,
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
        'data_mode': data_mode,
        'num_test_samples': cfg.get('NUM_TEST_SAMPLES', 256),
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

    # TF32: enable tensor-core fp32 matmul acceleration (negligible numeric
    # difference for training; ~1.3x on L20). Config key ALLOW_TF32.
    if cfg_main.get('ALLOW_TF32', False) and device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[Config] TF32 enabled (matmul + cudnn)")

    if not config.get(BATCH_RUN_MERGED_FLAG):
        print("[Error] Config not merged. Please run via batch_run.py or merge config manually.")
        sys.exit(2)  # non-zero so batch_run records failure instead of a silent "success"

    # ========================================================================
    # Stage 1: Task branch -- prepare dataset, model, loader, training params
    # ========================================================================
    if TASK in ('mixed_ab', 'mixed_abc'):
        order = 2 if TASK == 'mixed_ab' else 3
        ctx = _prepare_mixed_recurrence(config, device, order)
    elif TASK in ('action', 'action_trib'):
        ctx = _prepare_action(config, order=3 if TASK == 'action_trib' else 2)
    elif TASK in ('addition', 'multiplication', 'tribonacci', 'tetranacci', 'nonlinear', 'nonlinear_mul'):
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

    # Wall-clock-timeout resume: load full training state (weights, optimizer,
    # scheduler, RNG). Applied after INIT_FROM; if both are set, RESUME_FROM wins.
    resume_state = None
    resume_from = cfg_main.get('RESUME_FROM')
    if resume_from:
        resume_state = _load_resume_checkpoint(resume_from, model, optimizer, scheduler, device)

    best_acc, epoch, timed_out = run_training_engine(
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
        cond_fix_start_a2=cfg.get('COND_FIX_START_A2', None),
        max_train_hours=cfg.get('MAX_TRAIN_HOURS'),
        resume_state=resume_state,
        use_amp=cfg.get('USE_AMP', False),
        amp_dtype=cfg.get('AMP_DTYPE', 'bfloat16'),
        test_loader_fn=ctx.get('test_loader_fn'),
        grad_accum_steps=cfg.get('GRAD_ACCUM_STEPS', 1),
        extra_epochs_after_high_acc=cfg.get('EARLY_STOP_EXTRA_EPOCHS', 200)
    )

    if timed_out:
        # Wall-clock limit hit: resume checkpoint already saved by the engine.
        # Exit 42 so batch_run reports it as timeout (not success, not crash).
        print(f"[TIMEOUT] Resume later with RESUME_FROM="
              f"{resume_checkpoint_path(SAVE_PATH)}")
        sys.exit(42)

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
                                          OOD_LEN, num_mask, device,
                                          data_mode=ctx.get('data_mode', 'full_split'),
                                          num_test_samples=ctx.get('num_test_samples', 256))
