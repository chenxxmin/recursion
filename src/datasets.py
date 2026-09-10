"""Datasets, samplers and collate functions for the recurrence experiments.

Extracted from core.py (core.py split, step 2). MixedRecurrenceDataset was
merged in from the legacy mixed-rule dataset module, which eliminated the
circular import between that module and core. core.py re-exports the
externally referenced symbols defined here, so existing `from core import
...` users (analysis scripts, tests) are unaffected.
"""
import random
from enum import IntEnum

import torch
from torch.utils.data import Dataset, Sampler


# ==================== Data Generation ====================
class RecurrenceDataset(Dataset):
    def __init__(self, p=127, recurrence_fn=None, recurrence_name="X(k)=?", init_len=2, num_samples=1000, length=10,
                 verbose=True, missing_prob=0.0, num_mask=0, first_task_weight=1.0,
                 missing_token=None, miss_len=1, miss_second=False,
                 predict_missing=False, state_cap=None):
        self.length = length
        self.p = p
        self.recurrence_fn = recurrence_fn
        self.recurrence_name = recurrence_name
        self.init_len = init_len
        self.num_samples = num_samples
        self.verbose = verbose
        # MISSING_PROB: when > 0, corruption is applied ON THE FLY in the
        # collate (make_missing_collate): windows are stored CLEAN here, and
        # every batch gets fresh random missing positions, so each epoch (and
        # each eval) sees a different corruption pattern. The fields below are
        # still used by _corrupt (the single implementation, shared by the
        # collate factories and the corrupt_window analysis helper).
        # num_mask/first_task_weight feed the loss-mask prefix in _corrupt.
        assert miss_len >= 1, f"miss_len must be >= 1, got {miss_len}"
        self.missing_prob = missing_prob
        self.num_mask = num_mask
        self.first_task_weight = first_task_weight
        self.missing_token = p if missing_token is None else missing_token
        self.miss_len = miss_len
        self.miss_second = miss_second
        # PREDICT_MISSING: when true, items are (corrupted_view, clean_seq)
        # tuples — the model trains on the corrupted input but the loss/accuracy
        # targets are the clean values (missing positions must be filled in).
        # When false (default), items are (corrupted_seq, loss_mask) and the
        # missing positions are masked out of the metrics.
        self.predict_missing = predict_missing
        # STATE_SPACE_CAP: when p**init_len exceeds this, subsample state_cap
        # distinct initial states uniformly (one window per state, rolled out
        # step by step) instead of full cycle traversal; the train/test quota
        # split is unchanged (first num_samples sampled states -> train).
        self.state_cap = state_cap
        self.train_samples = []
        self.test_samples = []
        self.seen_indices = set()
        self.split = 'train'
        
    def index_to_values(self, index):
        vals = []
        for _ in range(self.init_len):
            vals.insert(0, index % self.p)
            index //= self.p
        return vals
    
    def values_to_index(self, vals):
        assert(len(vals) == self.init_len)
        index = 0
        w = 1
        for val in vals:
            index += val * w
            w *= self.p
        return index
    
    def generate_cycle(self, start_idx):
        cycle_vals = self.index_to_values(start_idx)
        while True:
            next_val = self.recurrence_fn(cycle_vals[-self.init_len:], self.p)
            current_state_idx = 0
            for val in cycle_vals[-self.init_len:]:
                current_state_idx = current_state_idx * self.p + val
            next_idx = (current_state_idx % (self.p ** (self.init_len - 1))) * self.p + next_val

            # Break when the trajectory closes back to the start state (a pure
            # cycle; array_len = cycle_len + (state_len-1)), or when it runs
            # into an already-seen state. The latter happens for transient
            # states (e.g. states containing 0 under multiplication, whose
            # recurrence is not bijective): the trajectory flows into a
            # previously processed cycle, and without this check the loop
            # would never terminate.
            if next_idx == start_idx or next_idx in self.seen_indices:
                break

            self.seen_indices.add(next_idx)
            cycle_vals.append(next_val)
        # Return the full trajectory (init_len seed values + one value per step).
        # The number of NEW states on it is len(vals) - (init_len - 1); the
        # trailing init_len-1 values are needed as seed context for extending.
        return cycle_vals

    def run(self):
        state_space = self.p ** self.init_len
        capped = self.state_cap is not None and state_space > self.state_cap

        if capped:
            # random.sample on a range has a fast path (no full enumeration)
            # and returns its picks in random order, so no extra shuffle.
            all_indices = random.sample(range(state_space), self.state_cap)
        else:
            # Randomly shuffle all state indices
            all_indices = list(range(state_space))
            random.shuffle(all_indices)

        for idx_pos, start_idx in enumerate(all_indices):
            if capped:
                n_before = idx_pos
                # One window per sampled state, rolled out step by step.
                seq = self.index_to_values(start_idx)
                num_inits = 1
                while len(seq) < self.length:
                    seq.append(self.recurrence_fn(seq[-self.init_len:], self.p))
            else:
                if start_idx in self.seen_indices:
                    continue

                # MAX_UNIQUE_RATIO is the proportion of initial states exposed to training.
                # The first num_samples states (in shuffled order) go to train, the rest to test.
                # The quota is enforced PER STATE (not per cycle): n_before is the number
                # of states processed before this traversal, and window i comes from the
                # (n_before + i)-th processed state. This keeps the exposed fraction exact
                # even when one cycle spans a large part of the state space.
                n_before = len(self.seen_indices)

                # Traverse the entire cycle
                self.seen_indices.add(start_idx)
                seq = self.generate_cycle(start_idx)

                # One training window per new state on the trajectory.
                num_inits = len(seq) - (self.init_len - 1)

                # Extend the sequence to num_inits + length - 1 values by continuing
                # the recurrence step by step. (Periodic doubling would be equivalent
                # for pure cycles, but wrong for transient trajectories truncated at
                # an already-seen state: the tail is not periodic.)
                while len(seq) < num_inits + self.length - 1:
                    seq.append(self.recurrence_fn(seq[-self.init_len:], self.p))

            # Append to train/test set. Windows are always stored CLEAN;
            # missing-value corruption is applied on the fly per batch in the
            # collate (make_missing_collate), never baked in here.
            for i in range(num_inits):
                is_train = n_before + i < self.num_samples
                target = self.train_samples if is_train else self.test_samples
                target.append(torch.tensor(seq[i:i + self.length], dtype=torch.long))

        # Shuffle sample order
        random.shuffle(self.train_samples)

        if self.verbose:
            cov = len(self.train_samples) / state_space
            print(f"[Dataset] Cycle traverse + sliding window: {len(self.train_samples)} + {len(self.test_samples)} samples, length {self.length}, mod {self.p}")
            print(f"  - Initial state coverage: {len(self.train_samples)}/{state_space} ({cov*100:.1f}%)")
            if capped:
                print(f"  - State space subsampled (STATE_SPACE_CAP): {self.state_cap}/{state_space} initial states, one window per state")
            if self.missing_prob > 0:
                print(f"  - Missing-value corruption: ON THE FLY in collate (fresh randomness per "
                      f"batch), prob={self.missing_prob}, miss_len={self.miss_len}, "
                      f"miss_second={self.miss_second}, positions >= {self.init_len}, token id {self.missing_token}")
            print(f"Recurrence: {self.recurrence_name}")
            print("-" * 50)
            print("-" * 50)

    def _corrupt(self, window, is_train):
        """Return the per-position loss mask for a window, corrupting it in place.

        Corruption applies to train AND test windows alike (is_train is kept
        only for signature compatibility): the metric of interest is bridging
        over missing tokens, which must be measurable on held-out states.
        The mask reproduces the train_epoch default (zeros up to num_mask, ones
        after, first_task_weight at num_mask); a corrupted token at position pos
        additionally zeroes its prediction target at index pos-1.

        Corruption model: when miss_second is set, positions 1..miss_len are
        corrupted unconditionally, the position right after that forced run
        stays clean, and scanning starts at position miss_len+2 (or init_len
        if later). Otherwise scanning starts at init_len. During the scan each
        position independently hits with probability missing_prob; a hit
        corrupts a RUN of random length in [1, miss_len], the position right
        after a run is always left clean, and scanning resumes from the
        position after that. A run may be truncated at the end of the window.
        If no random run in the window reaches length miss_len, the whole
        random scan is redone until it does (miss_second's forced run already
        satisfies this, so no redo happens in that mode).
        """
        mask = torch.zeros(self.length - 1, dtype=torch.float)
        if self.length - 1 > self.num_mask:
            mask[self.num_mask:] = 1.0
            if self.first_task_weight != 1.0:
                mask[self.num_mask] = self.first_task_weight
        pos = self.init_len
        if self.miss_second:
            end = min(1 + self.miss_len, self.length)
            for q in range(1, end):
                window[q] = self.missing_token
                mask[q - 1] = 0.0
            # like any corrupted run, the forced run is followed by one
            # guaranteed-clean position, so blocks never merge
            pos = max(pos, end + 1)
        # random scan; redo until the longest random run equals miss_len
        # (miss_second's forced run already counts as that maximum)
        retries = 0
        while True:
            trial_w = window.clone()
            trial_m = mask.clone()
            max_run = self.miss_len if self.miss_second else 0
            scan = pos
            while scan < self.length:
                if random.random() < self.missing_prob:
                    run = random.randint(1, self.miss_len)
                    end = min(scan + run, self.length)
                    for q in range(scan, end):
                        trial_w[q] = self.missing_token
                        trial_m[q - 1] = 0.0
                    max_run = max(max_run, end - scan)
                    scan = end + 1  # the position right after a run stays clean
                else:
                    scan += 1
            if max_run in (0, self.miss_len) or retries >= 1000:
                window.copy_(trial_w)
                mask.copy_(trial_m)
                if retries >= 1000:
                    print(f"WARNING: _corrupt gave up matching max run length "
                          f"{self.miss_len} after {retries} retries")
                break
            retries += 1
        return mask

    def __len__(self):
        data = self.train_samples if self.split == 'train' else self.test_samples
        return len(data)

    def __getitem__(self, idx):
        data = self.train_samples if self.split == 'train' else self.test_samples
        return data[idx]


def missing_token_id(config, p, is_mixed):
    """Missing-token id for a (merged or save) config dict.

    Follows MixedRecurrenceDataset's convention: p + n_rules in mixed tag
    mode (plain p would collide with rule 0's flag token), else p.
    """
    use_tag = config.get('use_ab_tag', config.get('USE_AB_TAG', False))
    if not (is_mixed and use_tag):
        return p
    pairs = config.get('ab_pairs') or config.get('AB_PAIRS') or config.get('ABC_PAIRS') or [1]
    return p + len(pairs)


def corrupt_window(window, is_train, *, p, init_len, missing_prob, miss_len=1,
                   miss_second=False, missing_token=None):
    """Corrupt `window` (LongTensor) in place with the dataset's missing-value
    rule, without constructing a full dataset. Returns the missing token id.

    Single source for analysis scripts that need the same corruption the
    training data went through (see RecurrenceDataset._corrupt).
    """
    token = p if missing_token is None else missing_token
    helper = RecurrenceDataset(p=p, init_len=init_len, length=len(window), verbose=False,
                               missing_prob=missing_prob, miss_len=miss_len,
                               miss_second=miss_second, missing_token=token)
    helper._corrupt(window, is_train)
    return token

# ==================== BucketBatchSampler (group by same length) ====================
class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group sample indices by sequence length
        length_to_indices = {}
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, (list, tuple)):
                length = len(item[0])
            else:
                length = len(item)
            if length not in length_to_indices:
                length_to_indices[length] = []
            length_to_indices[length].append(i)
        
        self.batches = []
        for indices in length_to_indices.values():
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i:i+batch_size])
        
        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

# Every collate_fn returns (x, tag, *payload); the tag tells consumers how to
# interpret the payload. See _unpack_batch for the layout of each tag.
# Routing is EXPLICIT: the prepare functions select the collate that matches
# the dataset's item layout (plain tensor / (seq, mask) / (view, clean) etc.);
# collates never guess from item types.
class BatchTag(IntEnum):
    MIXED_AB = 0       # payload: (ab_indices,)  -- per-sample rule index
    LOSS_MASK = 1      # payload: (loss_mask,)   -- per-position loss mask
    PLAIN = 2          # payload: ()             -- sequences only
    MIXED_AB_MASKED = 3  # payload: (ab_indices, loss_mask) -- rule index + per-position loss mask
    PLAIN_TARGET = 4     # payload: (clean_seqs,) -- clean sequences; targets = clean[:, 1:]
    MIXED_AB_TARGET = 5  # payload: (ab_indices, clean_seqs) -- rule index + clean sequences
    ACTION_MISS = 6    # payload: (clean_seqs, loss_mask) -- action + missing (predict mode)


def collate_fn(batch):
    """Plain tensor samples -> PLAIN."""
    return torch.stack(batch, dim=0), BatchTag.PLAIN


def _corrupt_helper(p, init_len, length, missing_prob, miss_len, miss_second,
                    num_mask, first_task_weight, missing_token):
    """A RecurrenceDataset instance used only for its _corrupt implementation."""
    return RecurrenceDataset(p=p, init_len=init_len, length=length, verbose=False,
                             missing_prob=missing_prob, miss_len=miss_len,
                             miss_second=miss_second, missing_token=missing_token,
                             num_mask=num_mask, first_task_weight=first_task_weight)


def make_missing_collate(*, p, init_len, missing_prob, miss_len=1, miss_second=False,
                         num_mask=0, first_task_weight=1.0, missing_token=None,
                         predict_missing=False):
    """Collate with ON-THE-FLY missing-value corruption (2026-09-09 regime).

    Input: a batch of CLEAN windows (plain LongTensors). Every call corrupts
    with fresh randomness, so each epoch and each eval sees different missing
    positions. Mask mode -> (views, LOSS_MASK, masks); predict mode ->
    (views, PLAIN_TARGET, cleans).
    """
    token = p if missing_token is None else missing_token

    def collate(batch):
        helper = _corrupt_helper(p, init_len, len(batch[0]), missing_prob,
                                 miss_len, miss_second, num_mask,
                                 first_task_weight, token)
        views, cleans, masks = [], [], []
        for seq in batch:
            view = seq.clone()
            mask = helper._corrupt(view, True)
            views.append(view)
            masks.append(mask)
            cleans.append(seq)
        if predict_missing:
            return torch.stack(views), BatchTag.PLAIN_TARGET, torch.stack(cleans)
        return torch.stack(views), BatchTag.LOSS_MASK, torch.stack(masks)

    return collate


# ==================== Mixed AB Experiment (mixed training with multiple recurrence params) ====================

def mixed_ab_collate_fn(batch):
    """(seq, rule_idx) samples -> MIXED_AB."""
    sequences = [item[0] for item in batch]
    ab_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return torch.stack(sequences, dim=0), BatchTag.MIXED_AB, ab_indices


def make_mixed_missing_collate(*, p, order, n_rules, use_ab_tag, missing_prob,
                               miss_len=1, miss_second=False, num_mask=2,
                               first_task_weight=1.0, predict_missing=False):
    """Mixed-rule on-the-fly corruption: (clean_seq, rule_idx) batches.

    With use_ab_tag the leading flag token stays clean; corruption and the
    mask prefix are computed in pre-tag coordinates (inner num_mask =
    num_mask - 1, mask regrown with a leading 0), matching the legacy
    semantics exactly. Mask mode -> (views, MIXED_AB_MASKED, labels, masks);
    predict mode -> (views, MIXED_AB_TARGET, labels, cleans).
    """
    token = p + n_rules if use_ab_tag else p
    inner_num_mask = max(0, num_mask - 1) if use_ab_tag else num_mask

    def collate(batch):
        seqs = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        core_len = len(seqs[0]) - (1 if use_ab_tag else 0)
        helper = _corrupt_helper(p, order, core_len, missing_prob, miss_len,
                                 miss_second, inner_num_mask, first_task_weight,
                                 token)
        views, cleans, masks = [], [], []
        zero = torch.zeros(1, dtype=torch.float)
        for seq in seqs:
            tag, core = (seq[:1], seq[1:]) if use_ab_tag else (None, seq)
            view_core = core.clone()
            mask_core = helper._corrupt(view_core, True)
            if tag is not None:
                # The prepended flag shifts every target by one; grow the mask
                # with a leading 0 (predicting x0 from the flag alone is
                # unsupervisable).
                view = torch.cat([tag, view_core], dim=0)
                mask = torch.cat([zero, mask_core], dim=0)
            else:
                view, mask = view_core, mask_core
            views.append(view)
            masks.append(mask)
            cleans.append(seq)
        if predict_missing:
            # clean target sequences keep the leading flag so that
            # targets = clean[:, 1:] stay aligned with the input view
            return (torch.stack(views), BatchTag.MIXED_AB_TARGET, labels,
                    torch.stack(cleans))
        return (torch.stack(views), BatchTag.MIXED_AB_MASKED, labels,
                torch.stack(masks))

    return collate


def generate_action_sample(p, ab_pairs, flag_start_id, length, rng):
    """Generate one action sequence [x1, x2, flag_3, x3, ..., flag_L, x_L].

    rng: a random.Random instance owned by the caller (seeding decides the
    sequence). Shared by ActionDataset and the attention analysis.
    """
    x1 = rng.randrange(p)
    x2 = rng.randrange(p)
    seq = [x1, x2]
    values = [x1, x2]  # Only numeric values, used for recurrence
    for _ in range(2, length):
        rule_idx = rng.randrange(len(ab_pairs))
        a, b = ab_pairs[rule_idx]
        x_next = (a * values[-2] + b * values[-1]) % p
        seq.append(flag_start_id + rule_idx)
        seq.append(x_next)
        values.append(x_next)

    # Sanity check: flag positions should be >= flag_start_id, value positions < p
    for i, tok in enumerate(seq):
        if i % 2 == 0 and i >= 2:
            assert tok >= flag_start_id, \
                f"Position {i} should be a flag token (>= {flag_start_id}), got {tok}"
        else:
            assert tok < p, \
                f"Position {i} should be a value token (< {p}), got {tok}"
    return seq


class ActionDataset(Dataset):
    """Dataset where the recurrence rule can change at every step.

    Input format: [x1, x2, flag_3, x3, flag_4, x4, ..., flag_L, x_L]
    where flag_k indicates which rule is used to generate x_k.
    Loss is computed only on x3..x_L; x1, x2 and all flags are masked.

    With missing_prob > 0, corruption is applied ON THE FLY in
    make_action_missing_collate (fresh randomness per batch): samples are
    always stored CLEAN as (seq, loss_mask) pairs.
    """

    def __init__(self, p, ab_pairs, num_samples, length, seed=0):
        super().__init__()
        self.p = p
        self.ab_pairs = [tuple(pair) for pair in ab_pairs]
        self.num_ab_pairs = len(self.ab_pairs)
        self.num_samples = num_samples
        self.length = length
        self.pad_token_id = p
        self.flag_start_id = p + 1
        self.rng = random.Random(seed)
        self.samples = [self._generate_sample() for _ in range(num_samples)]

    def _generate_sample(self):
        """Generate one CLEAN sequence with per-step random rules.

        Input format: [x1, x2, f3, x3, flag_4, x4, ..., f_L, x_L]
        where flag_k indicates which rule is used to generate x_k.
        Loss is computed only on x3..x_L; x1, x2 and all flags are masked.
        Missing-value corruption happens on the fly in
        make_action_missing_collate, never here.
        """
        seq = generate_action_sample(self.p, self.ab_pairs, self.flag_start_id,
                                      self.length, self.rng)

        # Build loss mask aligned to targets = seq[1:]
        # Input: [x1, x2, f3, x3, f4, x4, ..., f_L, x_L]
        # Target: [x2, f3, x3, f4, x4, ..., f_L, x_L]
        # Only x3, x4, ..., x_L should contribute to loss; x2 and all flags are masked.
        # Loop index k=2 generates x_3 (target index 2), k=3 generates x_4 (target index 4),
        # so x_{k+1} is at target index 2*(k-1).
        loss_mask = torch.zeros(len(seq) - 1, dtype=torch.float)
        for k in range(2, self.length):
            target_idx = 2 * (k - 1)
            loss_mask[target_idx] = 1.0

        return torch.tensor(seq, dtype=torch.long), loss_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def corrupt_action_view(view, *, p, missing_prob, miss_len):
    """Corrupt value positions x3..x_L (odd seq indices >= 3) of an action
    sequence in place with the missing token p. Same run model as the legacy
    ActionDataset._corrupt_values (scan from x3; a hit corrupts a run of
    random length in [1, miss_len]; the value right after a run stays clean;
    redo until the longest run equals miss_len), but with fresh global
    randomness per call (on-the-fly regime).
    """
    L = (len(view) + 2) // 2  # number of values: seq has 2 + 2*(L-2) tokens
    retries = 0
    while True:
        trial = view.clone()
        max_run = 0
        k = 3
        while k <= L:
            if random.random() < missing_prob:
                run = random.randint(1, miss_len)
                end = min(k + run, L + 1)  # exclusive value index
                for q in range(k, end):
                    trial[2 * q - 3] = p
                max_run = max(max_run, end - k)
                k = end + 1  # the value right after a run stays clean
            else:
                k += 1
        if max_run in (0, miss_len) or retries >= 1000:
            view.copy_(trial)
            if retries >= 1000:
                print(f"WARNING: corrupt_action_view gave up matching max run length "
                      f"{miss_len} after {retries} retries")
            return
        retries += 1


def action_collate_fn(batch):
    sequences = [item[0] for item in batch]
    loss_masks = [item[1] for item in batch]
    return torch.stack(sequences, dim=0), BatchTag.LOSS_MASK, torch.stack(loss_masks, dim=0)


def make_action_missing_collate(*, p, missing_prob, miss_len=1):
    """Action on-the-fly corruption: items are (clean_seq, loss_mask); every
    batch gets fresh corruption of value positions -> (views, ACTION_MISS,
    cleans, masks). Predict mode: loss/accuracy targets are the clean values.
    """
    def collate(batch):
        seqs = [item[0] for item in batch]
        loss_masks = torch.stack([item[1] for item in batch], dim=0)
        views = []
        for seq in seqs:
            view = seq.clone()
            corrupt_action_view(view, p=p, missing_prob=missing_prob, miss_len=miss_len)
            views.append(view)
        return (torch.stack(views, dim=0), BatchTag.ACTION_MISS,
                torch.stack(seqs, dim=0), loss_masks)

    return collate


# ==================== MixedRecurrenceDataset (merged from the legacy
# mixed-rule dataset module; its module docstring follows verbatim)
#
# Mixed-rule recurrence dataset.
#
# Generates data for several LinearRecurrenceRule rules (same modulus p, same
# order), each with its own RecurrenceDataset cycle traversal, then concatenates
# the per-rule samples and tags every sample with its rule index. Mirrors the
# legacy MixedABDataset pipeline step for step (same global-random call
# sequence), so given the same rules and the same seed the produced samples are
# identical to the legacy implementation (kept in tests/test_mixed_ab_compat.py
# as the golden master).
class MixedRecurrenceDataset(Dataset):
    """Dataset mixed from several linear recurrence rules over Z/pZ.

    Samples are CLEAN (seq, rule_idx) pairs; with use_ab_tag=True a leading
    flag token p + rule_idx is prepended to every sequence. The prepare
    function (experiment._prepare_mixed_recurrence) selects the collate:
    mixed_ab_collate_fn, or make_mixed_missing_collate when MISSING_PROB > 0
    (on-the-fly corruption per batch; missing token id is p, or
    p + len(rules) in tag mode where plain p would collide with rule 0's
    flag token).
    """

    def __init__(self, rules, num_samples=1000, length=10, verbose=True,
                 use_ab_tag=False):
        assert len(rules) >= 1, "MixedRecurrenceDataset needs at least one rule"
        self.p = rules[0].p
        self.order = rules[0].order
        for rule in rules:
            assert rule.p == self.p, "all rules must share the same modulus p"
            assert rule.order == self.order, "all rules must share the same order"
        coeffs_list = [rule.coeffs for rule in rules]
        assert len(set(coeffs_list)) == len(coeffs_list), \
            "duplicate rule coeffs are not allowed"
        self.rules = list(rules)
        self.length = length
        self.use_ab_tag = use_ab_tag
        self.split = 'train'

        if isinstance(num_samples, (list, tuple)):
            num_samples_list = list(num_samples)
        else:
            num_samples_list = [num_samples] * len(self.rules)
        assert len(num_samples_list) == len(self.rules), \
            f"num_samples list length ({len(num_samples_list)}) must equal number of rules ({len(self.rules)})"

        all_train, all_test = [], []
        all_labels_train, all_labels_test = [], []
        per_rule_stats = []
        for idx, rule in enumerate(self.rules):
            ds = RecurrenceDataset(
                p=self.p, recurrence_fn=rule.next_fn(),
                recurrence_name=rule.name, init_len=rule.order,
                num_samples=num_samples_list[idx], length=length,
                verbose=False,
            )
            ds.run()
            train_seqs, test_seqs = ds.train_samples, ds.test_samples
            all_train.extend(train_seqs)
            all_test.extend(test_seqs)
            all_labels_train.extend([idx] * len(train_seqs))
            all_labels_test.extend([idx] * len(test_seqs))
            per_rule_stats.append((rule, len(train_seqs), len(test_seqs)))

        # Prepend rule token if use_ab_tag (DataLoader indexes the flat lists, not __getitem__)
        if self.use_ab_tag:
            def _prepend(seq, l):
                return torch.cat([torch.tensor([self.p + l], dtype=torch.long), seq], dim=0)
            all_train = [_prepend(seq, l) for seq, l in zip(all_train, all_labels_train)]
            all_test = [_prepend(seq, l) for seq, l in zip(all_test, all_labels_test)]

        # One global shuffle of train (seq, rule_idx) pairs (same seed semantics
        # as the legacy MixedABDataset).
        train_pairs = list(zip(all_train, all_labels_train))
        random.shuffle(train_pairs)
        self.train_samples = [s for s, _ in train_pairs]
        self.rule_labels_train = [l for _, l in train_pairs]
        self.test_samples = all_test
        self.rule_labels_test = all_labels_test

        # Flat lists that can be fed directly to DataLoader / BucketBatchSampler
        self.train_data = list(zip(self.train_samples, self.rule_labels_train))
        self.test_data = list(zip(self.test_samples, self.rule_labels_test))

        if verbose:
            total_states = self.p ** self.order
            print(f"[Dataset] Mixed mode: generated {len(self.train_samples)} + {len(self.test_samples)} samples, length {length}")
            print(f"Rules: {[r.name for r in self.rules]}")
            for rule, n_train, n_test in per_rule_stats:
                print(f"  - {rule.name} {rule.coeffs}: train {n_train} | test {n_test} | exposed ~{n_train/total_states*100:.1f}%")
            print("-" * 50)

    def __len__(self):
        seqs = self.train_samples if self.split == 'train' else self.test_samples
        return len(seqs)

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.train_samples[idx], self.rule_labels_train[idx]
        return self.test_samples[idx], self.rule_labels_test[idx]
