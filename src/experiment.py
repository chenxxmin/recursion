"""Unified experiment entry: config parsing, scenario dispatch, training.

Extracted from core.py (core.py split, step 4). run_experiment is a slim
orchestrator: parse config -> normalize_task_config -> scenario dispatch
(_prepare_single_recurrence / _prepare_mixed_recurrence / _prepare_action
/ _prepare_single_variant（corrupt_fix/fill_blank/clean_value_set）)
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
from itertools import combinations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rules import rules_from_config, single_rule_from_task, save_config_extra
from protocol import BATCH_RUN_MERGED_FLAG
from scenarios import normalize_task_config
from models import FibonacciTransformer
from datasets import (RecurrenceDataset, MixedRecurrenceDataset,
                      ActionDataset, ActionOrbitView, BucketBatchSampler,
                      collate_fn, make_missing_collate,
                      mixed_ab_collate_fn, make_mixed_missing_collate,
                      action_collate_fn, make_action_missing_collate,
                      make_corrupt_fix_collate, make_fill_blank_collate,
                      make_clean_value_set_collate)
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


def _normalize_experiment_config(task, cfg):
    """Normalize configs, including the experiment-local clean_value_set variant.

    The shared legacy normalization layer intentionally remains untouched;
    clean_value_set is normalized here and then uses the common single-variant
    preparation path.  All other variants retain the exact legacy behavior.
    """
    if cfg.get('TASK_VARIANT') != 'clean_value_set':
        if 'CLEAN_VALUE_SET_CORRUPT_RATIO' in cfg:
            raise ValueError("CLEAN_VALUE_SET_CORRUPT_RATIO 仅 "
                             "TASK_VARIANT='clean_value_set' 支持")
        return normalize_task_config(task, cfg)
    ratio = cfg.get('CLEAN_VALUE_SET_CORRUPT_RATIO', 0.0)
    if not isinstance(ratio, (int, float)) or not 0.0 <= float(ratio) <= 1.0:
        raise ValueError("CLEAN_VALUE_SET_CORRUPT_RATIO 必须在 [0, 1] 内")
    if cfg.get('MISSING_PROB'):
        raise ValueError("clean_value_set uses clean windows and is incompatible "
                         "with MISSING_PROB")
    base_cfg = dict(cfg)
    base_cfg['TASK_VARIANT'] = None
    # clean_value_set shares the variant BODY_SUPERVISION switch.  The legacy
    # normalizer only knows the older variants, so validate it without this key.
    base_cfg.pop('BODY_SUPERVISION', None)
    norm = normalize_task_config(task, base_cfg)
    if norm['scenario'] != 'single':
        raise ValueError("TASK_VARIANT='clean_value_set' is supported only for single")
    norm['variant'] = 'clean_value_set'
    return norm


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


def _make_pos_probe(windows, collate, batch_size, variant, blank_lens=None,
                    blank_counts=None, value_only=False):
    """固定窗口上的逐位置探针；扩展填空按段长或空格数分桶。"""
    blank_lens = [1] if blank_lens is None else list(blank_lens)
    blank_counts = [1] if blank_counts is None else list(blank_counts)
    multi_blank = variant == 'fill_blank' and blank_lens != [1]
    multi_count = variant == 'fill_blank' and blank_counts != [1]
    by_len = {}
    for w in windows:
        by_len.setdefault(w.numel(), []).append(w)

    REL_ORDER = ['first', 'second', 'mid', 'last2', 'last']

    def _rel_class(k, length):
        if k == 0:
            return 'first'
        if k == 1:
            return 'second'
        if k == length - 1:
            return 'last'
        if k == length - 2:
            return 'last2'
        return 'mid'

    def fmt(c, t):
        return ' '.join(f"k{k:02d}={c[k] / t[k] * 100:.1f}" for k in sorted(t))

    def fmt_rel(c, t):
        return ' '.join(f"{r}={c[r] / t[r] * 100:.1f}" for r in REL_ORDER if r in t)

    def _add(c, t, key, correct, total):
        c[key] = c.get(key, 0.0) + correct
        t[key] = t.get(key, 0) + total

    def multi_count_probe(model, device):
        names = ('val_c', 'val_t', 'val_rel_c', 'val_rel_t')
        stats = {count: {name: {} for name in names} for count in blank_counts}
        with torch.no_grad():
            for window_len, ws in sorted(by_len.items()):
                for count in blank_counts:
                    if count > window_len:
                        raise ValueError(
                            f"blank count {count} exceeds probe window {window_len}")
                    s = stats[count]
                    for positions in combinations(range(window_len), count):
                        batch = [(seq, positions) for seq in ws]
                        loader = DataLoader(batch, batch_size=batch_size,
                                            collate_fn=collate)
                        _, _, per_pos, _ = evaluate(
                            model, loader, device, num_mask=0)
                        output_pos = list(range(window_len, window_len + count))
                        missing = [pos for pos in output_pos if pos not in per_pos]
                        if missing:
                            raise RuntimeError(
                                f"multi-count probe missing output positions {missing}")
                        n = len(ws)
                        for output, original_pos in zip(output_pos, positions):
                            rc = _rel_class(original_pos, window_len)
                            _add(s['val_c'], s['val_t'], original_pos,
                                 per_pos[output] * n, n)
                            _add(s['val_rel_c'], s['val_rel_t'], rc,
                                 per_pos[output] * n, n)
        lines = []
        for count in blank_counts:
            s = stats[count]
            lines.extend([
                f"  PosProbe val_n{count}| " + fmt(s['val_c'], s['val_t']),
                f"  PosProbe valR_n{count}| " + fmt_rel(s['val_rel_c'], s['val_rel_t']),
            ])
        return lines

    def multi_probe(model, device):
        names = ('val_c', 'val_t', 'val_rel_c', 'val_rel_t')
        stats = {blank_l: {name: {} for name in names} for blank_l in blank_lens}
        with torch.no_grad():
            for window_len, ws in sorted(by_len.items()):
                for blank_l in blank_lens:
                    m = window_len - blank_l + 1
                    if m < 1:
                        raise ValueError(
                            f"blank length {blank_l} exceeds probe window {window_len}")
                    s = stats[blank_l]
                    for k in range(m):
                        batch = [(seq, k, blank_l) for seq in ws]
                        loader = DataLoader(batch, batch_size=batch_size,
                                            collate_fn=collate)
                        _, _, per_pos, _ = evaluate(
                            model, loader, device, num_mask=0)
                        output_pos = list(range(m, m + blank_l))
                        missing = [pos for pos in output_pos if pos not in per_pos]
                        if missing:
                            raise RuntimeError(
                                f"multi-blank probe missing output positions {missing}")
                        n = len(ws)
                        rc = _rel_class(k, window_len)
                        # EOS(l) 已给定段长；所有值槽逐槽累加为 token 微平均。
                        for pos in output_pos:
                            _add(s['val_c'], s['val_t'], k, per_pos[pos] * n, n)
                            _add(s['val_rel_c'], s['val_rel_t'], rc,
                                 per_pos[pos] * n, n)
        lines = []
        for blank_l in blank_lens:
            s = stats[blank_l]
            lines.extend([
                f"  PosProbe val_l{blank_l}| " + fmt(s['val_c'], s['val_t']),
                f"  PosProbe valR_l{blank_l}| " + fmt_rel(s['val_rel_c'], s['val_rel_t']),
            ])
        return lines

    def probe(model, device):
        model.eval()
        if multi_count:
            return multi_count_probe(model, device)
        if multi_blank:
            return multi_probe(model, device)
        first_c, first_t, second_c, second_t = {}, {}, {}, {}
        rel_c1, rel_t1, rel_c2, rel_t2 = {}, {}, {}, {}
        with torch.no_grad():
            for length, ws in sorted(by_len.items()):
                for k in range(length):
                    batch = [(seq, k) for seq in ws]
                    loader = DataLoader(batch, batch_size=batch_size,
                                        collate_fn=collate)
                    _, _, per_pos, _ = evaluate(model, loader, device, num_mask=0)
                    n = len(ws)
                    rc = _rel_class(k, length)
                    if length in per_pos:
                        _add(first_c, first_t, k, per_pos[length] * n, n)
                        _add(rel_c1, rel_t1, rc, per_pos[length] * n, n)
                    if length + 1 in per_pos:
                        _add(second_c, second_t, k, per_pos[length + 1] * n, n)
                        _add(rel_c2, rel_t2, rc, per_pos[length + 1] * n, n)

        if variant == 'corrupt_fix' and not value_only:
            return ['  PosProbe pos| ' + fmt(first_c, first_t),
                    '  PosProbe val| ' + fmt(second_c, second_t),
                    '  PosProbe posR| ' + fmt_rel(rel_c1, rel_t1),
                    '  PosProbe valR| ' + fmt_rel(rel_c2, rel_t2)]
        if variant == 'fill_blank' or value_only:
            return ['  PosProbe val| ' + fmt(first_c, first_t),
                    '  PosProbe valR| ' + fmt_rel(rel_c1, rel_t1)]
        raise ValueError(f"POS_PROBE does not support variant={variant!r}")

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
    rules = rules_from_config(norm['rules'], P)
    state_space_size = P ** order

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
    action_sampling = dict(rule_mode=cfg.get('ACTION_RULE_MODE','per_step'),
                           active_rule_ids=cfg.get('ACTION_ACTIVE_RULE_IDS'))
    AB_PAIRS = [tuple(reversed(r)) for r in norm['rules']]  # canonical -> action
    missing_prob = cfg.get('MISSING_PROB', 0.0)
    miss_len = cfg.get('MISS_LEN', 1)
    P = cfg['P']
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

    orbit_mode = action_sampling['rule_mode'] == 'per_sequence_orbit'
    if orbit_mode:
        if TRAIN_LEN <= order or OOD_LEN <= order:
            raise ValueError("action windows require TRAIN_LEN/OOD_LEN > ORDER")
        if action_sampling['active_rule_ids'] is not None:
            raise ValueError("per_sequence_orbit currently requires all RULES active")
        orbit_rules = rules_from_config(norm['rules'], P)
        counts = cfg.get('NUM_TRAIN_SAMPLES')
        if counts is None:
            counts = [max(1, int(P ** order * r)) for r in norm['exposure_ratios']]
        elif isinstance(counts, (int, float)):
            counts = [counts] * len(orbit_rules)
        if len(counts) != len(orbit_rules) or any(
                int(n) != n or not 0 <= n <= P ** order for n in counts):
            raise ValueError("orbit NUM_TRAIN_SAMPLES must give a valid count per rule")
        ds = MixedRecurrenceDataset(orbit_rules, num_samples=[int(n) for n in counts],
                                    length=TRAIN_LEN)
        train_dataset = ActionOrbitView(ds.train_data, P, order)
        # Preserve held-out initial states when OOD windows have another length.
        test_data = ds.test_data
        if OOD_LEN != TRAIN_LEN:
            test_data = []
            for values, rule_id in ds.test_data:
                window = values[:order].tolist()
                next_fn = orbit_rules[rule_id].next_fn()
                while len(window) < OOD_LEN:
                    window.append(next_fn(window[-order:], P))
                test_data.append((torch.tensor(window, dtype=torch.long), rule_id))
        test_dataset = ActionOrbitView(test_data, P, order)
    else:
        train_dataset = ActionDataset(
            p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TRAIN_SAMPLES,
            length=TRAIN_LEN, seed=cfg.get('RANDOM_SEED', 42), order=order, **action_sampling
        )
        test_dataset = ActionDataset(
            p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TEST_SAMPLES,
            length=OOD_LEN, seed=cfg.get('RANDOM_SEED', 42) + 1, order=order, **action_sampling
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
            if orbit_mode:
                # Match mixed fresh-test semantics: NUM_TEST_SAMPLES per rule.
                rng_state = random.getstate()
                try:
                    random.seed(fresh_seed)
                    samples = []
                    for idx, rule in enumerate(orbit_rules):
                        fresh_ds = RecurrenceDataset(
                            p=P, recurrence_fn=rule.next_fn(), init_len=order,
                            num_samples=0, length=OOD_LEN, verbose=False,
                            state_cap=NUM_TEST_SAMPLES)
                        fresh_ds.run()
                        samples.extend((seq, idx) for seq in fresh_ds.test_samples)
                finally:
                    random.setstate(rng_state)
                ds = ActionOrbitView(samples, P, order)
            else:
                ds = ActionDataset(
                    p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TEST_SAMPLES,
                    length=OOD_LEN, seed=fresh_seed, order=order, **action_sampling)
            print(f"[FreshTest] rebuilt action test set: n={len(ds)}, len={OOD_LEN}, seed={fresh_seed}")
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
            # Scale only newly added rule rows; preserve RNG consumption and all
            # other initial weights for a paired initialization comparison.
            rule_init_std = float(cfg.get('ACTION_RULE_TOKEN_INIT_STD', 1.0))
            if not 0 < rule_init_std < float('inf'):
                raise ValueError('ACTION_RULE_TOKEN_INIT_STD must be finite and positive')
            new_wte.weight.data[old_wte.weight.size(0):].mul_(rule_init_std)
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
        'd_model': D_MODEL,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'mlp_ratio': cfg.get('MLP_RATIO', 8),
        'vocab_size': model.vocab_size,
        'pad_token_id': model.pad_token_id,
        'recurrence': 'action',
        'action_rule_mode': action_sampling['rule_mode'],
        'action_active_rule_ids': action_sampling['active_rule_ids'],
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

    if scenario == 'single':
        init_len, recurrence_fn, recurrence_name = single_rule_from_task(
            'single', {'P': P, 'RULES': norm['rules'], 'CONST': norm.get('const', 0)})
        default_num_mask = norm['order'] - 1
        max_unique_ratio = norm['exposure_ratios']
    else:
        # Nonlinear exception tasks keep their hardcoded recurrences.
        init_len, recurrence_fn, recurrence_name = single_rule_from_task(scenario, cfg)
        default_num_mask = {'multiplication': 1, 'nonlinear': 1,
                            'nonlinear_mul': 1}[scenario]
        max_unique_ratio = cfg.get('MAX_UNIQUE_RATIO', 0.7)
    save_extra_config = save_config_extra(norm, cfg)

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
    {'corrupt_fix', 'fill_blank', 'clean_value_set'}).

    The data pipeline is identical to _prepare_single_recurrence's single
    branch (RecurrenceDataset + full_split/sampled_fresh_test, same exposure
    logic); only the collate, the token layout and the model vocab differ:

      - corrupt_fix: collate corrupts exactly ONE position with a random
        wrong value; x = [corrupted window, EOS, pos_token], targets =
        [.., pos_token, correct value], loss on the last two positions.
        vocab = values [0..P-1] | missing P (unused) | EOS P+1 |
        position tokens P+2+k, k < max(TRAIN_LEN, OOD_LEN).
      - fill_blank: BLANK_LENS compresses one contiguous segment and EOS(l)
        encodes its length. BLANK_COUNTS selects independent unit positions;
        EOS(k) encodes their count and the tail contains their values in
        increasing position order. Extended vocab = P+1+max(length/count).
      - clean_value_set: no corruption; body CE plus a unique-window-value
        set objective at the EOS input slot. vocab = P + 2 (P is unused).

    num_mask = 0 in all variants: the loss mask comes from the collate.
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

    # SEQ_LEN_MIN/MAX（仅变体，归一化层校验）：窗口长度逐样本均匀随机
    # （不定长）；缺省为定长 TRAIN_LEN/OOD_LEN。不定长时数据集按最大长度
    # 生成再随机截断，分桶 sampler 按长度分组 batch。
    seq_min = cfg.get('SEQ_LEN_MIN')
    seq_max = cfg.get('SEQ_LEN_MAX')
    var_len = seq_min is not None
    blank_lens = list(cfg.get('BLANK_LENS', [1]))
    blank_counts = list(cfg.get('BLANK_COUNTS', [1]))
    multi_blank = variant == 'fill_blank' and blank_lens != [1]
    multi_count = variant == 'fill_blank' and blank_counts != [1]

    max_len = seq_max if var_len else max(TRAIN_LEN, OOD_LEN)
    # corrupt_fix：窗口 + EOS + 位置 token；fill_blank：窗口 + EOS(l/k) + 值前缀
    tail_tokens = (2 if variant == 'corrupt_fix' else
                   max(blank_counts) if multi_count else 1)
    BLOCK_SIZE = _round_up_pow2(max_len + tail_tokens)

    init_len, recurrence_fn, recurrence_name = single_rule_from_task(
        'single', {'P': P, 'RULES': norm['rules'], 'CONST': norm.get('const', 0)})
    if var_len and seq_min <= init_len:
        raise ValueError(
            f"SEQ_LEN_MIN ({seq_min}) 必须 > 阶数 init_len ({init_len})，"
            "否则主体没有可监督的递推位置")
    max_unique_ratio = norm['exposure_ratios']
    save_extra_config = save_config_extra(norm, cfg)

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
        length=seq_max if var_len else TRAIN_LEN,
    )
    if data_mode == 'full_split':
        ds.run_full_split()
        train_dataset = ds.train_samples
        test_dataset = ds.test_samples
    else:
        train_dataset = ds.sample_unique_train()
        test_dataset = []

    # 不定长：逐样本随机截断到 [SEQ_LEN_MIN, SEQ_LEN_MAX]（独立 RNG，不扰动主流）
    if var_len:
        cut_rng = random.Random(cfg.get('RANDOM_SEED', 42) + 31)

        def _cut(samples):
            return [s[:cut_rng.randint(seq_min, seq_max)] for s in samples]

        train_dataset = _cut(train_dataset)
        if data_mode == 'full_split':
            test_dataset = _cut(test_dataset)
        print(f"[Variant] variable length: windows cut to [{seq_min}, {seq_max}]")

    # 枚举连续段起点，或按空格数枚举全部独立位置组合。
    def _enumerate_blanks(samples):
        if multi_count:
            return [(seq, positions) for seq in samples for k in blank_counts
                    for positions in combinations(range(len(seq)), k)]
        if multi_blank:
            return [(seq, k, l) for seq in samples for l in blank_lens
                    for k in range(len(seq) - l + 1)]
        return [(seq, k) for seq in samples for k in range(len(seq))]

    blank_all = (variant == 'fill_blank'
                 and cfg.get('FILL_BLANK_ALL_POSITIONS', False))
    if blank_all:
        n_base = len(train_dataset)
        train_dataset = _enumerate_blanks(train_dataset)
        if data_mode == 'full_split':
            test_dataset = _enumerate_blanks(test_dataset)
        print(f"[Variant] FILL_BLANK_ALL_POSITIONS: {n_base} windows -> "
              f"{len(train_dataset)} train samples")
    # CORRUPT_FIX_ALL_POSITIONS（仅 corrupt_fix）：每条序列的每个位置各污损
    # 一次（(seq, k) 元组，错误值仍由 collate 在线随机），训练/测试样本数
    # × 窗口长度。与 FILL_BLANK_ALL_POSITIONS 同构。
    corrupt_all = (variant == 'corrupt_fix'
                   and cfg.get('CORRUPT_FIX_ALL_POSITIONS', False))
    if corrupt_all:
        n_base = len(train_dataset)
        train_dataset = [(seq, k) for seq in train_dataset for k in range(len(seq))]
        if data_mode == 'full_split':
            test_dataset = [(seq, k) for seq in test_dataset for k in range(len(seq))]
        print(f"[Variant] CORRUPT_FIX_ALL_POSITIONS: {n_base} windows -> "
              f"{len(train_dataset)} train samples")
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
        test_dataset = _enumerate_blanks(test_windows)
        print(f"[Variant] TEST_ALL_POSITIONS: {len(test_windows)} windows -> "
              f"{len(test_dataset)} deterministic test samples")

    # BODY_SUPERVISION（变体默认开）：递推主体也计入 loss/acc
    # （predict 口径，targets 前缀用干净序列）。
    body_sup = cfg.get('BODY_SUPERVISION', True)
    body_from = init_len - 1
    eos_id = P + 1
    if variant == 'clean_value_set':
        corrupt_ratio = float(cfg.get('CLEAN_VALUE_SET_CORRUPT_RATIO', 0.0))
        if corrupt_ratio:
            print(f'[Variant] CLEAN_VALUE_SET_CORRUPT_RATIO={corrupt_ratio}: '
                  '每 batch 随机混合干净集合预测行与单点纠错行')
        collate = make_clean_value_set_collate(
            p=P, eos_id=eos_id, body_from=body_from,
            body_supervision=body_sup, corrupt_ratio=corrupt_ratio)
        vocab_size = P + 2
        variant_save = {'eos_token_id': eos_id, 'body_from': body_from,
                        'corrupt_ratio': corrupt_ratio}
    elif variant == 'corrupt_fix' and cfg.get('CORRUPT_FIX_VALUE_ONLY', False):
        # VALUE_ONLY：不输出位置、只监督 EOS 真值槽（fill_blank 版面 + 污损）
        fixed_pos = cfg.get('CORRUPT_FIX_FIXED_POS')
        if fixed_pos is not None:
            fixed_pos = int(fixed_pos)
            print(f"[Variant] CORRUPT_FIX_FIXED_POS={fixed_pos}: 污损位置固定")
        print('[Variant] CORRUPT_FIX_VALUE_ONLY: 不告知/不预测位置，'
              '只监督 EOS 正确值（自行检测污损）')
        collate = make_fill_blank_collate(p=P, eos_id=eos_id,
                                          body_supervision=body_sup,
                                          body_from=body_from,
                                          fixed_k=fixed_pos, corrupt=True,
                                          last_n=cfg.get('CORRUPT_FIX_LAST_N'),
                                          pos_set=cfg.get('CORRUPT_FIX_POS_SET'))
        vocab_size = P + 2
        variant_save = {'eos_token_id': eos_id, 'value_only': True,
                        'fixed_pos': fixed_pos,
                        'last_n': cfg.get('CORRUPT_FIX_LAST_N'),
                        'pos_set': cfg.get('CORRUPT_FIX_POS_SET')}
    elif variant == 'corrupt_fix':
        # CORRUPT_FIX_PARALLEL：并行解码，值预测看不到位置答案（双 EOS 槽）
        parallel = cfg.get('CORRUPT_FIX_PARALLEL', False)
        # CORRUPT_FIX_PAIRED_CLEAN：配对双样本（干净窗续写 + 污损窗只答题）
        paired_clean = cfg.get('CORRUPT_FIX_PAIRED_CLEAN', False)
        # CORRUPT_FIX_MASK_AROUND：动态剔除污损牵连的主体预测位 {k-1,k,k+1}
        mask_around = cfg.get('CORRUPT_FIX_MASK_AROUND', False)
        # CORRUPT_FIX_POS_WEIGHT：pos 槽 loss 权重（默认 1.0）
        pos_weight = cfg.get('CORRUPT_FIX_POS_WEIGHT', 1.0)
        collate = make_corrupt_fix_collate(p=P, eos_id=eos_id, pos_base=P + 2,
                                           parallel=parallel,
                                           body_supervision=body_sup,
                                           body_from=body_from,
                                           paired_clean=paired_clean,
                                           mask_around=mask_around,
                                           pos_weight=pos_weight)
        vocab_size = P + 2 + max_len
        variant_save = {'eos_token_id': eos_id, 'pos_token_base': P + 2,
                        'parallel_decode': parallel,
                        'paired_clean': paired_clean,
                        'mask_around': mask_around,
                        'pos_weight': pos_weight,
                        'all_positions': corrupt_all}
    else:  # fill_blank
        fixed_pos = cfg.get('FILL_BLANK_FIXED_POS')
        if fixed_pos is not None:
            fixed_pos = int(fixed_pos)
            print(f"[Variant] FILL_BLANK_FIXED_POS={fixed_pos}: 挖空位置固定"
                  f"（负数从末尾数，-1 = 最后一位）")
        if multi_count:
            print(f"[Variant] BLANK_COUNTS={blank_counts}: 独立位置各留一个 BLANK，"
                  "EOS(k) 编码空格数，随后按位置升序输出 k 个原值")
        elif multi_blank:
            print(f"[Variant] BLANK_LENS={blank_lens}: 连续段压缩为单个 BLANK，"
                  "EOS(l) 在输入中编码段长，随后输出被挖值序列")
        collate = make_fill_blank_collate(p=P, eos_id=eos_id,
                                          body_supervision=body_sup,
                                          body_from=body_from,
                                          fixed_k=fixed_pos,
                                          blank_lens=blank_lens,
                                          blank_counts=blank_counts)
        max_eos_variant = (max(blank_counts) if multi_count else
                           max(blank_lens) if multi_blank else 1)
        vocab_size = P + 1 + max_eos_variant
        variant_save = {'eos_token_id': eos_id, 'blank_token_id': P,
                        'fixed_pos': fixed_pos, 'blank_lens': blank_lens,
                        'blank_counts': blank_counts,
                        'blank_mode': ('multi_single' if multi_count else
                                       'contiguous' if multi_blank else 'single'),
                        'eos_variant_base': eos_id if multi_blank or multi_count else None}
    if variant == 'corrupt_fix' and cfg.get('CORRUPT_FIX_PAIRED_CLEAN', False):
        print('[Variant] CORRUPT_FIX_PAIRED_CLEAN: 干净续写行 + 污损答题行'
              '（主体监督由干净行承担，污损行只留 pos/val 两槽）')
    if variant == 'corrupt_fix' and cfg.get('CORRUPT_FIX_MASK_AROUND', False):
        print('[Variant] CORRUPT_FIX_MASK_AROUND: 动态剔除污损牵连的主体预测位'
              ' {k-1,k,k+1}（预测 x_k/x_{k+1}/x_{k+2} 不计 loss/acc）')
    if variant == 'corrupt_fix' and cfg.get('CORRUPT_FIX_POS_WEIGHT', 1.0) != 1.0:
        print(f"[Variant] CORRUPT_FIX_POS_WEIGHT: pos 槽 loss 权重 x"
              f"{cfg['CORRUPT_FIX_POS_WEIGHT']}")
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
                NUM_TEST_SAMPLES, length=seq_max if var_len else OOD_LEN,
                rng=eval_rng)
            if var_len:
                cut = random.Random(fresh_seed + 1)
                fresh_samples = [s[:cut.randint(seq_min, seq_max)]
                                 for s in fresh_samples]
            if blank_all or test_all:
                fresh_samples = _enumerate_blanks(fresh_samples)
            elif corrupt_all:
                fresh_samples = [(seq, k) for seq in fresh_samples
                                 for k in range(len(seq))]
            print(f"[FreshTest] sampled with replacement: "
                  f"n={NUM_TEST_SAMPLES}, len={seq_max if var_len else OOD_LEN}, seed={fresh_seed}")
            return _make_test_loader(fresh_samples, BATCH_SIZE, collate)

    # POS_PROBE（变体默认开，`POS_PROBE: false` 关闭）：固定探针窗口集，
    # 每个 eval 点按挖空/污损位置 k 分桶统计 acc 并打进日志（逐位置热力图
    # 的数据源）。探针集固定，跨 epoch 可比。
    probe_fn = None
    if variant != 'clean_value_set' and cfg.get('POS_PROBE', True):
        probe_n = min(256, cfg.get('NUM_TEST_SAMPLES', 256) * 16)
        if data_mode == 'full_split':
            # 枚举数据集会对同一 tensor 对象重复引用；先按对象身份去重，
            # 避免探针重复 forward 及不定长窗口被组合数额外加权。
            pool, seen_windows = [], set()
            for item in test_dataset:
                window = item[0] if isinstance(item, (list, tuple)) else item
                if id(window) not in seen_windows:
                    seen_windows.add(id(window))
                    pool.append(window)
            probe_windows = random.Random(cfg.get('RANDOM_SEED', 42) + 55).sample(
                pool, min(probe_n, len(pool)))
        else:
            probe_windows = ds.sample_random_windows(
                probe_n, length=seq_max if var_len else OOD_LEN,
                rng=random.Random(cfg.get('RANDOM_SEED', 42) + 56))
            if var_len:
                probe_windows = _cut(probe_windows)
        probe_fn = _make_pos_probe(
            probe_windows, collate, BATCH_SIZE, variant=variant,
            blank_lens=blank_lens, blank_counts=blank_counts,
            value_only=bool(cfg.get('CORRUPT_FIX_VALUE_ONLY', False)))
        probe_lens = sorted({w.numel() for w in probe_windows})
        print(f"[Probe] POS_PROBE: {len(probe_windows)} windows, "
              f"lengths={probe_lens}, fixed across evals")

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
        'body_supervision': body_sup,
        'seq_len_min': seq_min,
        'seq_len_max': seq_max,
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
        norm = _normalize_experiment_config(TASK, cfg_main)
    except ValueError as exc:
        print(f"[Error] {exc}")
        sys.exit(2)  # non-zero so batch_run records failure instead of a silent "success"
    scenario = norm['scenario']
    if norm.get('variant') in ('corrupt_fix', 'fill_blank', 'clean_value_set'):
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
        extra_eval_fn=ctx.get('extra_eval_fn'),
        save_checkpoints=cfg.get('SAVE_CHECKPOINTS', True)
    )

    if timed_out:
        # Wall-clock limit hit: resume checkpoint already saved by the engine.
        # Exit 42 so batch_run reports it as timeout (not success, not crash).
        print(f"[TIMEOUT] Resume later with RESUME_FROM="
              f"{resume_checkpoint_path(SAVE_PATH)}")
        sys.exit(42)
