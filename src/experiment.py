"""Unified experiment entry: config parsing, scenario dispatch, training.

Extracted from core.py (core.py split, step 4). run_experiment is a slim
orchestrator: parse config -> normalize_task_config -> scenario dispatch
(_prepare_single_recurrence / _prepare_mixed_recurrence / _prepare_action
/ _prepare_single_variant（corrupt_fix/fill_blank）)
-> run_training_engine. The stage-3 final generation test (final_eval.py)
was removed 2026-10-08; result reporting is the training log's best acc
plus scripts/eval_clean_action.py. Depends on
models/datasets/training/scenarios/rules/protocol; never imports core.
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

from rules import (
    rules_from_config, single_rule_from_task, save_config_extra, field_from_cfg,
    coeffs_poly_list, format_recurrence_name,
)
from finite_field import poly_str
from protocol import BATCH_RUN_MERGED_FLAG
from scenarios import normalize_task_config
from models import FibonacciTransformer
from datasets import (RecurrenceDataset, MixedRecurrenceDataset,
                      ActionDataset, BucketBatchSampler,
                      collate_fn, make_missing_collate,
                      mixed_ab_collate_fn, make_mixed_missing_collate,
                      action_collate_fn, make_action_missing_collate,
                      make_corrupt_fix_collate, make_fill_blank_collate)
from training import run_training_engine, resume_checkpoint_path, evaluate


# ==================== Unified Experiment Entry ====================
# BATCH_RUN_MERGED_FLAG lives in protocol.py (single source shared with
# batch_run.py); run_experiment refuses to run on an unmerged config.


def _load_partial_checkpoint(model, path, device):
    """Resume/transfer weights from a checkpoint (INIT_FROM).

    Only tensors whose name AND shape match are copied; the rest keep their
    fresh init (e.g. wte/lm_head when vocab sizes differ). Prints a load/skip
    report so the transfer is visible in the log.
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


def _make_loaders(train_dataset, test_dataset, batch_size, collate_fn, reshuffle_each_epoch=False):
    """Build train/test DataLoaders with length-grouped batch sampling."""
    train_sampler = BucketBatchSampler(train_dataset, batch_size=batch_size, shuffle=True,
                                       reshuffle_each_epoch=reshuffle_each_epoch)
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


def _make_pos_probe(windows, collate, length, batch_size):
    """POS_PROBE: 固定探针窗口集上按挖空/污损位置 k 分桶的评估钩子。

    返回 fn(model, device) -> list[str]（每 eval 点由引擎打印）。
    对每个 k 用 (seq, k) 元组固定挖空位置，跑 evaluate 取 per_pos_acc：
    fill_blank 只有值槽（target[L]）；corrupt_fix 有位置槽 target[L] 和
    值槽 target[L+1]，分两行打印。corrupt_fix 的错误值仍逐次随机
    （位置 acc 对错值不敏感）；fill_blank 探针完全确定。
    """
    def probe(model, device):
        model.eval()
        first_slot, second_slot = {}, {}
        with torch.no_grad():
            for k in range(length):
                batch = [(seq, k) for seq in windows]
                loader = DataLoader(batch, batch_size=batch_size, collate_fn=collate)
                _, _, per_pos, _ = evaluate(model, loader, device, num_mask=0)
                if length in per_pos:
                    first_slot[k] = per_pos[length]
                if length + 1 in per_pos:
                    second_slot[k] = per_pos[length + 1]

        def fmt(d):
            return ' '.join(f"k{k:02d}={d[k] * 100:.1f}" for k in sorted(d))

        if second_slot:
            return ['  PosProbe pos| ' + fmt(first_slot),
                    '  PosProbe val| ' + fmt(second_slot)]
        return ['  PosProbe val| ' + fmt(first_slot)]

    return probe


def _prepare_mixed_recurrence(config, device, norm):
    """Prepare dataset, model, loaders and training params for the mixed
    scenario (any order >= 2; canonical rules from the normalized config).

    Model is always FibonacciTransformer, vocab always P+1 (tag mode and
    cond_wte were removed 2026-10-08; the rule_head auxiliary classifier is
    on by default for mixed, AUX_CLASSIFY=false turns it off). NUM_MASK
    default is
    order-1. Old spellings (mixed_ab/mixed_abc + AB_PAIRS/ABC_PAIRS +
    MAX_UNIQUE_RATIO/MIXED_AB_MAX_UNIQUE_RATIOS) arrive already translated
    by scenarios.normalize_task_config.
    """
    cfg = dict(config.get('main', {}))
    P = cfg['P']
    D_MODEL = cfg['D_MODEL']
    N_HEAD = cfg['N_HEAD']
    N_LAYER = cfg['N_LAYER']
    BATCH_SIZE = cfg['BATCH_SIZE']
    TRAIN_LEN = cfg['TRAIN_LEN']
    OOD_LEN = cfg['OOD_LEN']
    DROPOUT = cfg['DROPOUT']
    order = norm['order']
    field = field_from_cfg(cfg, P)
    rules = rules_from_config(norm['rules'], P, field=field)
    state_space_size = P ** order
    if field is not None:
        print(f"[Field] GF(2^{field.n}) order={field.order} "
              f"irr={poly_str(field.irreducible)} "
              f"(value tokens 0..{P - 1}, pad={P})")
        for r in rules:
            print(f"[Field] rule {r.name}: {format_recurrence_name(r.coeffs, P, field=field)}")

    BLOCK_SIZE = _round_up_pow2(max(TRAIN_LEN, OOD_LEN))

    # Explicit NUM_TRAIN_SAMPLES overrides the ratio-derived counts: a scalar
    # gives every rule the same count, a list sets per-rule counts.
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
        ratios = norm['exposure_ratios']
        NUM_TRAIN_SAMPLES = [max(1, int(state_space_size * r)) for r in ratios]

    # NUM_MASK unset (None) means: the first num_mask targets are initial
    # values and are not evaluated; default order-1.
    num_mask_cfg = cfg.get('NUM_MASK')
    num_mask = (order - 1) if num_mask_cfg is None else num_mask_cfg
    # An all-zero loss mask yields a grad-less constant loss and crashes
    # backward() far from the cause; require at least one evaluated position.
    assert num_mask < TRAIN_LEN - 1, \
        f"NUM_MASK ({num_mask}) must be < TRAIN_LEN - 1 ({TRAIN_LEN - 1})"

    missing_prob = cfg.get('MISSING_PROB', 0.0)
    ds = MixedRecurrenceDataset(rules=rules, num_samples=NUM_TRAIN_SAMPLES,
                                length=TRAIN_LEN, verbose=True)
    train_dataset = ds.train_data
    test_dataset = ds.test_data

    _print_task_banner(BLOCK_SIZE, TRAIN_LEN, f"Rules: {[r.name for r in rules]}")

    # Explicit collate selection: with MISSING_PROB, corruption is applied
    # on the fly in the collate (fresh randomness per batch, predict regime).
    if missing_prob > 0:
        mixed_collate = make_mixed_missing_collate(
            p=P, order=order, n_rules=len(rules),
            missing_prob=missing_prob, miss_len=cfg.get('MISS_LEN', 1))
    else:
        mixed_collate = mixed_ab_collate_fn           # (seq, label) -> MIXED_AB
    train_loader, test_loader = _make_loaders(
        train_dataset, test_dataset, BATCH_SIZE, mixed_collate,
        reshuffle_each_epoch=cfg.get('RESHUFFLE_EACH_EPOCH', False))

    # FRESH_TEST_PER_EVAL (mixed): at every eval, sample NUM_TEST_SAMPLES fresh
    # initial states PER RULE from each rule's full state space and roll out
    # OOD_LEN windows (num_samples=0 sends them all to the test split).
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

    # AUX_CLASSIFY (mixed only, validated by the normalization layer): attach
    # the rule-classification head and feed per-sample rule labels
    # (ab_indices) to the model via extra_kwargs_fn. Default ON for mixed
    # (2026-10-10, user decision); AUX_CLASSIFY=false passes no extra kwargs
    # and stays bitwise identical to pre-feature runs (rule_head init
    # consumes torch RNG).
    aux_classify = cfg.get('AUX_CLASSIFY', True)
    model_kwargs = {'n_rules': len(rules), 'order': order} if aux_classify else {}
    model = FibonacciTransformer(p=P, d_model=D_MODEL, n_head=N_HEAD,
                                 n_layer=N_LAYER, block_size=BLOCK_SIZE,
                                 dropout=DROPOUT, mlp_ratio=cfg.get('MLP_RATIO', 8),
                                 **model_kwargs)
    if aux_classify:
        extra_kwargs_fn = lambda ab_indices: {'ab_labels': ab_indices.to(device)}
    else:
        extra_kwargs_fn = None
    save_config = {
        'p': P,
        'scenario': 'mixed',
        'order': order,
        'rules': [list(r.coeffs) for r in rules],
        # canonical == legacy mixed convention, safe for old analysis tools
        'ab_pairs': [list(r.coeffs) for r in rules],
        'field': ('gf2n' if field is not None else None),
        'rules_poly': [coeffs_poly_list(r.coeffs, field) for r in rules],
        'irreducible_poly': (poly_str(field.irreducible) if field is not None else None),
        'exposure_ratios': ratios,
        'aux_classify': aux_classify,
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'mlp_ratio': cfg.get('MLP_RATIO', 8),
        'vocab_size': P + 1,
        'pad_token_id': P,
        # for analyze_attention.py: in-dist length and corruption settings
        'train_len': TRAIN_LEN,
        'missing_prob': cfg.get('MISSING_PROB', 0.0),
        'miss_len': cfg.get('MISS_LEN', 1),
    }
    return {
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


def _prepare_action(config, norm):
    """Prepare dataset, model, loaders and training params for the action
    scenario (rule can change at every step), any order >= 2.

    norm['rules'] is canonical (first coefficient multiplies the NEWEST
    value). The ACTION data/layout convention is the opposite (first
    coefficient multiplies the OLDEST value), so the rules are reversed ONCE
    here — the only other reversal site is scenarios._legacy_rules (legacy
    AB_PAIRS/ABC_PAIRS -> canonical). Do not add a third one.
    """
    # batch_run already merges the action section into main with the correct
    # precedence (main -> task defaults -> experiment override). Do NOT
    # re-apply the section here.
    cfg = dict(config.get('main', {}))

    order = norm['order']
    AB_PAIRS = [tuple(reversed(r)) for r in norm['rules']]  # canonical -> action
    missing_prob = cfg.get('MISSING_PROB', 0.0)
    miss_len = cfg.get('MISS_LEN', 1)
    P = cfg['P']
    field = field_from_cfg(cfg, P)
    if field is not None:
        print(f"[Field] GF(2^{field.n}) order={field.order} "
              f"irr={poly_str(field.irreducible)} "
              f"(value tokens 0..{P - 1}, pad={P})")
        for coeffs in norm['rules']:
            print(f"[Field] rule: {format_recurrence_name(coeffs, P, field=field)}")
    D_MODEL = cfg['D_MODEL']
    N_HEAD = cfg['N_HEAD']
    N_LAYER = cfg['N_LAYER']
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROPOUT = cfg['DROPOUT']
    NUM_TRAIN_SAMPLES = cfg.get('NUM_TRAIN_SAMPLES', 10000)
    NUM_TEST_SAMPLES = cfg.get('NUM_TEST_SAMPLES', 2000)  # default matches src/config.json
    TRAIN_LEN = cfg.get('TRAIN_LEN', 16)
    OOD_LEN = cfg.get('OOD_LEN', 16)

    # In action, each generated token is preceded by a flag token,
    # so the actual sequence length is 2*length - order.
    max_seq_len = 2 * max(TRAIN_LEN, OOD_LEN) - order
    BLOCK_SIZE = _round_up_pow2(max_seq_len)

    train_dataset = ActionDataset(
        p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TRAIN_SAMPLES,
        length=TRAIN_LEN, seed=cfg.get('RANDOM_SEED', 42), order=order,
        field=field,
    )
    test_dataset = ActionDataset(
        p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TEST_SAMPLES,
        length=OOD_LEN, seed=cfg.get('RANDOM_SEED', 42) + 1, order=order,
        field=field,
    )

    _print_task_banner(BLOCK_SIZE, TRAIN_LEN, f"Dynamic mixed rules: {AB_PAIRS}")

    # With MISSING_PROB, corruption is applied on the fly in the collate
    # (fresh randomness per batch; datasets store clean samples only).
    collate = (make_action_missing_collate(p=P, missing_prob=missing_prob,
                                           miss_len=miss_len, order=order)
               if missing_prob > 0 else action_collate_fn)
    train_loader, test_loader = _make_loaders(
        train_dataset, test_dataset, BATCH_SIZE, collate,
        reshuffle_each_epoch=cfg.get('RESHUFFLE_EACH_EPOCH', False))

    # FRESH_TEST_PER_EVAL: rebuild the test set at every eval with a fresh seed.
    test_loader_fn = None
    if cfg.get('FRESH_TEST_PER_EVAL', False):
        fresh_rng = random.Random(cfg.get('RANDOM_SEED', 42) + 10 ** 6 + 7)

        def test_loader_fn():  # noqa: F811 (intentional closure name)
            fresh_seed = fresh_rng.randrange(2 ** 31)
            ds = ActionDataset(
                p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TEST_SAMPLES,
                length=OOD_LEN, seed=fresh_seed, order=order, field=field)
            print(f"[FreshTest] rebuilt test set: n={NUM_TEST_SAMPLES}, len={OOD_LEN}, seed={fresh_seed}")
            return _make_test_loader(ds, BATCH_SIZE, collate)

    model = FibonacciTransformer(
        p=P, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER,
        block_size=BLOCK_SIZE, dropout=DROPOUT,
        mlp_ratio=cfg.get('MLP_RATIO', 8)
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
        'scenario': 'action',
        'order': order,
        'rules': [list(r) for r in norm['rules']],  # canonical
        'ab_pairs': [list(pair) for pair in AB_PAIRS],  # action 旧约定，供旧分析工具
        'field': (f'gf2n' if field is not None else None),
        'rules_poly': [coeffs_poly_list(r, field) for r in norm['rules']],
        'irreducible_poly': (poly_str(field.irreducible) if field is not None else None),
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'mlp_ratio': cfg.get('MLP_RATIO', 8),
        'vocab_size': model.vocab_size,
        'pad_token_id': model.pad_token_id,
        'recurrence': 'action',
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
        'missing_prob': missing_prob,
        'miss_len': miss_len,
    }
    return {
        'model': model,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'test_loader_fn': test_loader_fn,
        'num_mask': 0,  # unused; loss_mask comes from the dataset
        'save_config': save_config,
        'cfg': cfg,
        'p': P,
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
    }


def _prepare_single_recurrence(config, norm):
    """Prepare dataset, model, loaders and training params for the single
    scenario (linear, any order via LinearRecurrenceRule) and the nonlinear
    exception tasks (hardcoded recurrences)."""
    cfg = dict(config.get('main', {}))
    P = cfg['P']
    D_MODEL = cfg['D_MODEL']
    N_HEAD = cfg['N_HEAD']
    N_LAYER = cfg['N_LAYER']
    BATCH_SIZE = cfg['BATCH_SIZE']
    TRAIN_LEN = cfg['TRAIN_LEN']
    OOD_LEN = cfg['OOD_LEN']
    DROPOUT = cfg['DROPOUT']
    scenario = norm['scenario']

    BLOCK_SIZE = _round_up_pow2(max(TRAIN_LEN, OOD_LEN))

    field = field_from_cfg(cfg, P)
    if field is not None:
        print(f"[Field] GF(2^{field.n}) order={field.order} "
              f"irr={poly_str(field.irreducible)} "
              f"(value tokens 0..{P - 1}, pad={P})")
    if scenario == 'single':
        init_len, recurrence_fn, recurrence_name = single_rule_from_task(
            'single', {
                'P': P, 'RULES': norm['rules'], 'CONST': norm.get('const', 0),
                'FIELD': field if field is not None else cfg.get('FIELD'),
            })
        default_num_mask = norm['order'] - 1
        max_unique_ratio = norm['exposure_ratios']
    else:
        # Nonlinear exception tasks keep their hardcoded recurrences.
        if field is not None:
            raise ValueError(
                f"FIELD=gf2n is not supported for exception task {scenario!r}")
        init_len, recurrence_fn, recurrence_name = single_rule_from_task(scenario, cfg)
        default_num_mask = {'multiplication': 1, 'nonlinear': 1,
                            'nonlinear_mul': 1}[scenario]
        max_unique_ratio = cfg.get('MAX_UNIQUE_RATIO', 0.7)
    save_extra_config = save_config_extra(norm, cfg)
    if field is not None:
        save_extra_config = {
            **save_extra_config,
            'field': 'gf2n',
            'rules_poly': [coeffs_poly_list(norm['rules'][0], field)],
            'irreducible_poly': poly_str(field.irreducible),
            'recurrence_poly': recurrence_name,
        }

    # Explicit data-generation strategy.
    #   full_split: traverse the full state space, then split into train/test.
    #   sampled_fresh_test: draw a fixed unique training set without materializing
    #     the state space; every evaluation draws a fresh test set with replacement.
    data_mode = cfg.get('DATA_MODE')
    if data_mode is None:
        # Backward compatibility for existing experiment files. Legacy
        # STATE_SPACE_CAP was only needed when full enumeration was too large,
        # which corresponds to the new sampled_fresh_test regime.
        legacy_sampled = (
            cfg.get('FRESH_TEST_PER_EVAL', False)
            or cfg.get('STATE_SPACE_CAP') is not None
        )
        data_mode = 'sampled_fresh_test' if legacy_sampled else 'full_split'
    if data_mode not in ('full_split', 'sampled_fresh_test'):
        raise ValueError(
            f"DATA_MODE must be 'full_split' or 'sampled_fresh_test', got {data_mode!r}")
    print(f"[Data mode] {data_mode}")

    state_space_size = P ** init_len
    NUM_TRAIN_SAMPLES = cfg.get('NUM_TRAIN_SAMPLES')
    if data_mode == 'full_split':
        if NUM_TRAIN_SAMPLES is None:
            NUM_TRAIN_SAMPLES = max(1, int(state_space_size * max_unique_ratio))
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
            miss_len=cfg.get('MISS_LEN', 1))
    else:
        collate = collate_fn              # plain tensors -> PLAIN
    train_sampler = BucketBatchSampler(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        reshuffle_each_epoch=cfg.get('RESHUFFLE_EACH_EPOCH', False))
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
        mlp_ratio=cfg.get('MLP_RATIO', 8)
    )
    save_config = {
        'p': P,
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'mlp_ratio': cfg.get('MLP_RATIO', 8),
        # for analyze_attention.py: in-dist length and corruption settings
        'train_len': TRAIN_LEN,
        'missing_prob': cfg.get('MISSING_PROB', 0.0),
        'miss_len': cfg.get('MISS_LEN', 1),
        'data_mode': data_mode,
        'num_train_samples': NUM_TRAIN_SAMPLES,
    }
    save_config.update(save_extra_config)
    return {
        'model': model,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'test_loader_fn': test_loader_fn,
        'num_mask': num_mask,
        'save_config': save_config,
        'cfg': cfg,
        'p': P,
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
    }


def _prepare_single_variant(config, norm):
    """Prepare single-scenario task variants (TASK_VARIANT ∈
    {'corrupt_fix', 'fill_blank'}; see scenarios.TASK_VARIANTS).

    The data pipeline is identical to _prepare_single_recurrence's single
    branch (RecurrenceDataset + full_split/sampled_fresh_test, same exposure
    logic); only the collate, the token layout and the model vocab differ:

      - corrupt_fix: collate corrupts exactly ONE position with a random
        wrong value; x = [corrupted window, EOS, pos_token], targets =
        [.., pos_token, correct value], loss on the last two positions.
        vocab = values [0..P-1] | missing P (unused) | EOS P+1 |
        position tokens P+2+k, k < max(TRAIN_LEN, OOD_LEN).
      - fill_blank: collate replaces exactly ONE position with the blank
        token (id = P); x = [corrupted window, EOS], targets = [.., correct
        value], loss on the EOS slot only. vocab = values | blank P |
        EOS P+1, i.e. vocab_size = P + 2.

    num_mask = 0 in both: the loss mask comes from the collate.
    """
    variant = norm['variant']
    cfg = dict(config.get('main', {}))
    P = cfg['P']
    D_MODEL = cfg['D_MODEL']
    N_HEAD = cfg['N_HEAD']
    N_LAYER = cfg['N_LAYER']
    BATCH_SIZE = cfg['BATCH_SIZE']
    TRAIN_LEN = cfg['TRAIN_LEN']
    OOD_LEN = cfg['OOD_LEN']
    DROPOUT = cfg['DROPOUT']

    max_len = max(TRAIN_LEN, OOD_LEN)
    # corrupt_fix: 输入 = 窗口 + EOS + 位置 token；fill_blank: 窗口 + EOS
    tail_tokens = 2 if variant == 'corrupt_fix' else 1
    BLOCK_SIZE = _round_up_pow2(max_len + tail_tokens)

    field = field_from_cfg(cfg, P)
    if field is not None:
        print(f"[Field] GF(2^{field.n}) order={field.order} "
              f"irr={poly_str(field.irreducible)} "
              f"(value tokens 0..{P - 1}, pad={P})")
    init_len, recurrence_fn, recurrence_name = single_rule_from_task(
        'single', {
            'P': P, 'RULES': norm['rules'], 'CONST': norm.get('const', 0),
            'FIELD': field if field is not None else cfg.get('FIELD'),
        })
    max_unique_ratio = norm['exposure_ratios']
    save_extra_config = save_config_extra(norm, cfg)
    if field is not None:
        save_extra_config = {
            **save_extra_config,
            'field': 'gf2n',
            'rules_poly': [coeffs_poly_list(norm['rules'][0], field)],
            'irreducible_poly': poly_str(field.irreducible),
            'recurrence_poly': recurrence_name,
        }

    # Data-generation strategy: same semantics as _prepare_single_recurrence.
    data_mode = cfg.get('DATA_MODE')
    if data_mode is None:
        legacy_sampled = (
            cfg.get('FRESH_TEST_PER_EVAL', False)
            or cfg.get('STATE_SPACE_CAP') is not None
        )
        data_mode = 'sampled_fresh_test' if legacy_sampled else 'full_split'
    if data_mode not in ('full_split', 'sampled_fresh_test'):
        raise ValueError(
            f"DATA_MODE must be 'full_split' or 'sampled_fresh_test', got {data_mode!r}")
    print(f"[Data mode] {data_mode}")

    state_space_size = P ** init_len
    NUM_TRAIN_SAMPLES = cfg.get('NUM_TRAIN_SAMPLES')
    if data_mode == 'full_split':
        if NUM_TRAIN_SAMPLES is None:
            NUM_TRAIN_SAMPLES = max(1, int(state_space_size * max_unique_ratio))
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

    # FILL_BLANK_ALL_POSITIONS（仅 fill_blank）：每条序列的每个位置挖空各成
    # 一条样本，样本数 × 窗口长度；空白位置由样本携带（(seq, k) 元组），
    # collate 不再随机选位（空白替换仍由 collate 完成）。
    blank_all = (variant == 'fill_blank'
                 and cfg.get('FILL_BLANK_ALL_POSITIONS', False))
    if blank_all:
        n_base = len(train_dataset)
        train_dataset = [(seq, k) for seq in train_dataset for k in range(len(seq))]
        if data_mode == 'full_split':
            test_dataset = [(seq, k) for seq in test_dataset for k in range(len(seq))]
        print(f"[Variant] FILL_BLANK_ALL_POSITIONS: {n_base} windows -> "
              f"{len(train_dataset)} train samples (x{TRAIN_LEN})")
    # TEST_ALL_POSITIONS（仅 fill_blank）：只把测试集改成逐位置枚举挖空
    # （确定性评估口径），训练侧保持在线随机挖空。blank_all 时测试已枚举，跳过。
    # 静态切分下 NUM_TEST_SAMPLES 作为测试窗口数上限（默认 256）：确定性子抽样，
    # 控制 ×16 后的 eval 开销。
    test_all = (variant == 'fill_blank' and not blank_all
                and cfg.get('TEST_ALL_POSITIONS', False))
    if test_all and data_mode == 'full_split':
        test_windows = test_dataset
        cap = cfg.get('NUM_TEST_SAMPLES', 256)
        if cap is not None and len(test_windows) > cap:
            sub_rng = random.Random(cfg.get('RANDOM_SEED', 42) + 77)
            test_windows = sub_rng.sample(test_windows, cap)
        print(f"[Variant] TEST_ALL_POSITIONS: {len(test_windows)} windows -> "
              f"{len(test_windows) * TRAIN_LEN} deterministic test samples")
        test_dataset = [(seq, k) for seq in test_windows for k in range(len(seq))]

    eos_id = P + 1
    if variant == 'corrupt_fix':
        # CORRUPT_FIX_PARALLEL：并行解码，值预测看不到位置答案（双 EOS 槽）
        parallel = cfg.get('CORRUPT_FIX_PARALLEL', False)
        collate = make_corrupt_fix_collate(p=P, eos_id=eos_id, pos_base=P + 2,
                                           parallel=parallel)
        vocab_size = P + 2 + max_len
        variant_save = {'eos_token_id': eos_id, 'pos_token_base': P + 2,
                        'parallel_decode': parallel}
    else:  # fill_blank
        collate = make_fill_blank_collate(p=P, eos_id=eos_id)
        vocab_size = P + 2
        variant_save = {'eos_token_id': eos_id, 'blank_token_id': P}
    _print_task_banner(BLOCK_SIZE, TRAIN_LEN,
                       f"{recurrence_name} [{variant}]")
    train_sampler = BucketBatchSampler(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        reshuffle_each_epoch=cfg.get('RESHUFFLE_EACH_EPOCH', False))
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
            if blank_all or test_all:
                fresh_samples = [(seq, k) for seq in fresh_samples
                                 for k in range(len(seq))]
            print(f"[FreshTest] sampled with replacement: "
                  f"n={NUM_TEST_SAMPLES}, len={OOD_LEN}, seed={fresh_seed}")
            return _make_test_loader(fresh_samples, BATCH_SIZE, collate)

    # POS_PROBE（变体默认开，`POS_PROBE: false` 关闭）：固定探针窗口集，
    # 每个 eval 点按挖空/污损位置 k 分桶统计 acc 并打进日志（逐位置热力图
    # 的数据源）。探针集固定，跨 epoch 可比。
    probe_fn = None
    if cfg.get('POS_PROBE', True):
        probe_n = min(256, cfg.get('NUM_TEST_SAMPLES', 256) * 16)
        if data_mode == 'full_split':
            pool = [item[0] if isinstance(item, (list, tuple)) else item
                    for item in test_dataset]
            probe_windows = random.Random(cfg.get('RANDOM_SEED', 42) + 55).sample(
                pool, min(probe_n, len(pool)))
            probe_len = TRAIN_LEN
        else:
            probe_windows = ds.sample_random_windows(
                probe_n, length=OOD_LEN,
                rng=random.Random(cfg.get('RANDOM_SEED', 42) + 56))
            probe_len = OOD_LEN
        probe_fn = _make_pos_probe(probe_windows, collate, probe_len, BATCH_SIZE)
        print(f"[Probe] POS_PROBE: {len(probe_windows)} windows x "
              f"{probe_len} positions, fixed across evals")

    model = FibonacciTransformer(
        p=P, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER,
        block_size=BLOCK_SIZE, dropout=DROPOUT,
        mlp_ratio=cfg.get('MLP_RATIO', 8), vocab_size=vocab_size
    )
    save_config = {
        'p': P,
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'mlp_ratio': cfg.get('MLP_RATIO', 8),
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
        'data_mode': data_mode,
        'num_train_samples': NUM_TRAIN_SAMPLES,
        # variant block
        'variant': variant,
        **variant_save,
        'vocab_size': vocab_size,
        'missing_prob': 0.0,
        'miss_len': cfg.get('MISS_LEN', 1),
    }
    save_config.update(save_extra_config)
    return {
        'model': model,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'test_loader_fn': test_loader_fn,
        'num_mask': 0,  # unused; loss_mask comes from the collate
        'extra_eval_fn': probe_fn,
        'save_config': save_config,
        'cfg': cfg,
        'p': P,
        'train_len': TRAIN_LEN,
        'ood_len': OOD_LEN,
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

    # TF32 恒开（2026-10-08 写死；对训练数值影响可忽略，L20 上 ~1.3x）
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[Config] TF32 enabled (matmul + cudnn)")

    if not config.get(BATCH_RUN_MERGED_FLAG):
        print("[Error] Config not merged. Please run via batch_run.py or merge config manually.")
        sys.exit(2)  # non-zero so batch_run records failure instead of a silent "success"

    # Print the effective config table so direct core.py runs (launched
    # without the batch_run wrapper) still carry a full parameter table in
    # their logs.
    print("=" * 60)
    print("Merged Config (all keys)")
    print("=" * 60)
    for key in sorted(cfg_main):
        print(f"{key:<25} {cfg_main[key]}")
    print("=" * 60)

    # ========================================================================
    # Stage 1: Scenario dispatch -- prepare dataset, model, loaders
    # ========================================================================
    try:
        norm = normalize_task_config(TASK, cfg_main)
    except ValueError as exc:
        print(f"[Error] {exc}")
        sys.exit(2)  # non-zero so batch_run records failure instead of a silent "success"
    scenario = norm['scenario']
    if norm.get('variant') in ('corrupt_fix', 'fill_blank'):
        # scenarios._normalize_variant already rejects non-single scenarios;
        # re-check defensively so a future variant never falls through silently.
        if scenario != 'single':
            print(f"[Error] variant {norm['variant']!r} not supported for scenario {scenario!r}")
            sys.exit(2)
        ctx = _prepare_single_variant(config, norm)
    elif scenario == 'mixed':
        ctx = _prepare_mixed_recurrence(config, device, norm)
    elif scenario == 'action':
        ctx = _prepare_action(config, norm)
    else:  # 'single' or a nonlinear exception task name
        ctx = _prepare_single_recurrence(config, norm)

    model = ctx['model']
    train_dataset = ctx['train_dataset']
    train_loader = ctx['train_loader']
    test_loader = ctx['test_loader']
    num_mask = ctx['num_mask']
    save_config = ctx['save_config']
    cfg = ctx['cfg']
    save_config['reshuffle_each_epoch'] = cfg.get('RESHUFFLE_EACH_EPOCH', False)
    print(f"[Sampler] reshuffle_each_epoch={save_config['reshuffle_each_epoch']}")
    P = ctx['p']
    TRAIN_LEN = ctx['train_len']
    OOD_LEN = ctx['ood_len']

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
        num_mask=num_mask,
        extra_kwargs_fn=ctx.get('extra_kwargs_fn'),
        first_task_weight=cfg.get('FIRST_TASK_WEIGHT', 1.0),
        max_train_hours=cfg.get('MAX_TRAIN_HOURS'),
        resume_state=resume_state,
        test_loader_fn=ctx.get('test_loader_fn'),
        grad_accum_steps=cfg.get('GRAD_ACCUM_STEPS', 1),
        grad_clip_norm=cfg.get('GRAD_CLIP_NORM', 1.0),
        opt_diag_interval=cfg.get('OPT_DIAG_INTERVAL', 0),
        extra_epochs_after_high_acc=cfg.get('EARLY_STOP_EXTRA_EPOCHS', 0),
        epoch_cap=cfg.get('EPOCH_CAP'),
        extra_eval_fn=ctx.get('extra_eval_fn')
    )

    if timed_out:
        # Wall-clock limit hit: resume checkpoint already saved by the engine.
        # Exit 42 so batch_run reports it as timeout (not success, not crash).
        print(f"[TIMEOUT] Resume later with RESUME_FROM="
              f"{resume_checkpoint_path(SAVE_PATH)}")
        sys.exit(42)
