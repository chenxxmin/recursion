import sys
import os
import re
import torch
import torch.nn.functional as F
import math
import random

# Allow running from repo root as: python src/analyze_attention.py <pth>
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import datasets
import models
from rules import LinearRecurrenceRule, single_rule_from_task, task_from_save_config
from report_front_back_attention import parse_log, split_k

# Display/analysis constants
HEADER_WIDTH = 70       # width of printed section separators
TOP_K = 3               # how many strongest-attended positions to report per query
MAX_DISTANCE = 63       # max lag for per-distance attention statistics
ENTROPY_EPS = 1e-12     # numerical epsilon inside log() for attention entropy
MIN_PRINT_VAL = 0.05    # per-position/per-distance values below this are omitted from printout
ITEMS_PER_LINE = 8      # per-distance entries printed per line


def _print_header(title):
    sep = '=' * HEADER_WIDTH
    print(f"\n{sep}")
    print(title)
    print(sep)


def load_model(pth_path, device='cpu'):
    """Load saved model and config."""
    checkpoint = torch.load(pth_path, map_location=device)
    config = checkpoint['config']
    # Filter out params not accepted by FibonacciTransformer (e.g. a, b)
    valid_keys = {'p', 'd_model', 'n_head', 'n_layer', 'block_size', 'dropout', 'use_learnable_pe', 'mlp_ratio', 'vocab_size', 'pad_token_id', 'cond_wte_shared_ratio'}
    # backward compat: old checkpoints saved 'recurrence' or 'a','b' in config
    filtered_config = {k: v for k, v in config.items() if k in valid_keys}
    
    # backward compat: old checkpoints without use_learnable_pe, infer from state_dict keys.
    # The two position-encoding schemes leave different fingerprints in the checkpoint:
    # learnable PE (or its alibi variant) stores per-position parameters ('alibi_slopes'),
    # while RoPE registers its frequency table as a buffer ('rope.inv_freq').
    state_keys = set(checkpoint['model_state_dict'].keys())
    if 'use_learnable_pe' not in filtered_config:
        if any('alibi_slopes' in k for k in state_keys):
            filtered_config['use_learnable_pe'] = True
        elif any('rope.inv_freq' in k for k in state_keys):
            filtered_config['use_learnable_pe'] = False
    
    # backward compat: old checkpoints without mlp_ratio, infer from first MLP weight shape
    if 'mlp_ratio' not in filtered_config:
        for k in state_keys:
            if k.endswith('.mlp.0.weight'):
                out_dim, in_dim = checkpoint['model_state_dict'][k].shape
                if in_dim > 0:
                    filtered_config['mlp_ratio'] = out_dim // in_dim
                break
    
    # Detect MixedABTransformer checkpoint and instantiate accordingly
    if any('ab_emb' in k or 'rule_head' in k or 'cond_wte' in k for k in state_keys):
        num_ab_pairs = len(config.get('ab_pairs', [])) if config.get('ab_pairs') else 1
        mixedab_config = dict(filtered_config)
        mixedab_config['use_ab_tag'] = config.get('use_ab_tag', True)
        mixedab_config['use_conditional_wte'] = config.get('use_conditional_wte', any('cond_wte' in k for k in state_keys))
        mixedab_config['cond_wte_shared_ratio'] = config.get('cond_wte_shared_ratio', 0.0)
        mixedab_config['order'] = config.get('order', 2)
        model = models.MixedABTransformer(num_ab_pairs=num_ab_pairs, **mixedab_config)
    else:
        model = models.FibonacciTransformer(**filtered_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def _build_input_embedding(model, input_ids, ab_label=None):
    """
    Build the input embedding used by the model during forward pass.

    Handles standard wte, learnable position embeddings, and conditional WTE
    (used by MixedABTransformer in cond_wte mode).
    """
    device = input_ids.device
    b, t = input_ids.size()

    if getattr(model, 'use_conditional_wte', False):
        if ab_label is None:
            # Default to the first rule. For attention analysis the rule choice
            # only affects which conditional embedding segment is used.
            ab_label = torch.zeros(b, dtype=torch.long, device=device)
        tok_emb = torch.zeros(b, t, model.d_model, device=device)
        shared_size = model.cond_wte_shared_size
        shared_mask = (input_ids < shared_size)
        if shared_mask.any():
            tok_emb[shared_mask] = model.cond_wte(input_ids[shared_mask])
        rule_mask = ~shared_mask
        if rule_mask.any():
            rule_idx = input_ids[rule_mask] - shared_size
            shifted_rule_idx = (
                rule_idx
                + ab_label.unsqueeze(1).expand(b, t)[rule_mask] * model.cond_wte_rule_size
                + shared_size
            )
            tok_emb[rule_mask] = model.cond_wte(shifted_rule_idx)
    else:
        tok_emb = model.transformer.wte(input_ids)

    if getattr(model, 'wpe', None) is not None:
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = tok_emb + model.wpe(pos)

    return model.transformer.drop(tok_emb)


def get_attention_weights(model, input_ids, ab_label=None):
    """
    Manually compute attention map for each layer and head given input.

    Returns: list of (n_head, T, T) tensors, one per layer.
    """
    # extract_qk_raw_scores runs the same manual forward pass and its
    # 'attn_weights' field is exactly this function's output.
    layer_outputs = extract_qk_raw_scores(model, input_ids, ab_label=ab_label)
    return [layer['attn_weights'] for layer in layer_outputs]


def _forward_per_layer(model, input_ids, ab_label=None):
    """Single-source manual forward pass for analysis scripts.

    Replays the model layer by layer (LN -> QKV -> RoPE -> causal mask ->
    softmax -> weighted sum -> proj -> residuals -> MLP) and returns a list
    of per-layer dicts:
        - 'q', 'k': (B, n_head, T, head_size)
        - 'raw_scores', 'attn_weights': (B, n_head, T, T)
        - 'hidden_post_attn': (B, T, C) residual stream right after attention
    Tensors keep the batch dimension and stay on the model's device.
    """
    x = _build_input_embedding(model, input_ids, ab_label=ab_label)

    layer_outputs = []

    for block in model.transformer.h:
        ln_x = block.ln_1(x)
        attn_module = block.attn
        B, T, C = ln_x.size()

        qkv = attn_module.c_attn(ln_x)
        q, k, v = qkv.split(attn_module.n_embd, dim=2)

        q = q.view(B, T, attn_module.n_head, attn_module.head_size).transpose(1, 2)
        k = k.view(B, T, attn_module.n_head, attn_module.head_size).transpose(1, 2)
        v = v.view(B, T, attn_module.n_head, attn_module.head_size).transpose(1, 2)

        # RoPE
        if attn_module.rope is not None:
            cos, sin = attn_module.rope(q, seq_len=T)
            q = models.apply_rotary_emb(q, cos, sin)
            k = models.apply_rotary_emb(k, cos, sin)

        # Raw QK scores (before softmax)
        raw_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn_module.head_size))

        # Attention weights (after softmax)
        att = raw_scores.masked_fill(attn_module.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        row_all_inf = torch.isinf(att).all(dim=-1, keepdim=True)
        att = att.masked_fill(row_all_inf, 0.0)
        att = F.softmax(att, dim=-1)

        # Continue forward
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = attn_module.c_proj(y)
        x = x + y
        layer_outputs.append({
            'q': q,
            'k': k,
            'raw_scores': raw_scores,
            'attn_weights': att,
            'hidden_post_attn': x,
        })
        x = x + block.mlp(block.ln_2(x))

    return layer_outputs


def extract_qk_raw_scores(model, input_ids, ab_label=None):
    """
    Extract Q, K and raw QK scores (before softmax) for each layer.

    Returns: list of dict, each layer contains:
        - 'q': (n_head, T, head_size)
        - 'k': (n_head, T, head_size)
        - 'raw_scores': (n_head, T, T), i.e. q @ k^T / sqrt(d)
        - 'attn_weights': (n_head, T, T), attention after softmax
    """
    return [{key: layer[key][0].detach().cpu() for key in ('q', 'k', 'raw_scores', 'attn_weights')}
            for layer in _forward_per_layer(model, input_ids, ab_label=ab_label)]


def summarize_attention_for_sequence(model, seq, query_mask=None):
    """
    Compute attention summary for a single sequence.

    seq: list[int] or torch.LongTensor, length T
    query_mask: optional list/tensor of length T-1 indicating which target/query
                positions should be included in the summary (1 = include).
                If None, all query positions are included.
    Returns: dict containing attention concentration analysis per layer.
    """
    if isinstance(seq, list):
        seq = torch.tensor([seq], dtype=torch.long)
    seq = seq.to(next(model.parameters()).device)

    attn_maps = get_attention_weights(model, seq)
    T = seq.size(1)

    if query_mask is None:
        query_positions = list(range(T))
    else:
        if isinstance(query_mask, (list, tuple)):
            query_mask = torch.tensor(query_mask, dtype=torch.bool)
        query_positions = query_mask.nonzero(as_tuple=True)[0].tolist()
        # Pad/truncate to T in case query_mask was shorter than T
        query_positions = [i for i in query_positions if 0 <= i < T]

    results = []

    for layer_idx, att in enumerate(attn_maps):
        n_head = att.shape[0]
        layer_info = {
            'layer': layer_idx,
            'heads': []
        }

        for h in range(n_head):
            attn_matrix = att[h]  # (T, T), attn_matrix[i, j] is attention from position i to j

            # Filter to valid query positions for statistics
            valid_attn = attn_matrix[query_positions] if query_positions else attn_matrix[:0]

            # For each valid query position i, find top K positions it attends to most
            if query_positions:
                topk_vals, topk_idx = torch.topk(valid_attn, k=min(TOP_K, T), dim=-1)
            else:
                topk_vals = torch.empty(0, min(TOP_K, T))
                topk_idx = torch.empty(0, min(TOP_K, T), dtype=torch.long)

            # Compute attention entropy (lower = more concentrated) over valid query positions
            if query_positions:
                entropy = -(valid_attn * (valid_attn + ENTROPY_EPS).log()).sum(dim=-1)
                mean_entropy = entropy.mean().item()
            else:
                mean_entropy = 0.0

            # Compute diagonal/sub-diagonal attention ratio over valid query positions
            focus_prev1 = 0.0
            focus_prev2 = 0.0
            if query_positions and T >= 3:
                for i in query_positions:
                    if i >= 1:
                        focus_prev1 += attn_matrix[i, i-1].item()
                    if i >= 2:
                        focus_prev2 += attn_matrix[i, i-2].item()
                focus_prev1 /= len(query_positions)
                focus_prev2 /= len(query_positions)

            # Compute average attention for distances 0..MAX_DISTANCE over valid query positions
            focus_by_distance = {}
            max_d = min(MAX_DISTANCE, T - 1)
            for d in range(0, max_d + 1):
                total = 0.0
                count = 0
                for i in query_positions:
                    if i - d >= 0:
                        total += attn_matrix[i, i - d].item()
                        count += 1
                if count > 0:
                    focus_by_distance[d] = total / count

            layer_info['heads'].append({
                'head': h,
                'mean_entropy': mean_entropy,
                'focus_prev1': focus_prev1,      # Average attention to previous token
                'focus_prev2': focus_prev2,      # Average attention to token before previous
                'focus_by_distance': focus_by_distance,
                'topk_positions': topk_idx.cpu().tolist(),
                'topk_values': topk_vals.cpu().tolist(),
                'attention_matrix': attn_matrix.cpu().tolist(),
                'query_positions': query_positions,
            })

        results.append(layer_info)

    return results


def print_attention_summary(summary, seq=None):
    """Print text summary of attention analysis."""
    if seq is not None and isinstance(seq, torch.Tensor):
        seq = seq[0].tolist()
    
    _print_header("Attention Layer Training Result Summary")
    if seq is not None:
        print(f"Input sequence: {seq}")
    print()
    
    for layer in summary:
        print(f"--- Layer {layer['layer']} ---")
        for head in layer['heads']:
            h = head['head']
            ent = head['mean_entropy']
            
            print(f"  Head {h}: Entropy={ent:.3f}")
            
            attn_mat = head['attention_matrix']
            T = len(attn_mat)
            query_positions = head.get('query_positions', list(range(T)))
            print(f"    Attention coefficients from each query position to previous positions (>= {MIN_PRINT_VAL}):")
            for i in query_positions:
                if i < T:
                    parts = [f"j={j}:{attn_mat[i][j]:.3f}" for j in range(i+1)
                             if attn_mat[i][j] >= MIN_PRINT_VAL]
                    line = ", ".join(parts) if parts else f"(all < {MIN_PRINT_VAL})"
                    print(f"      i={i:2d} -> " + line)
            
            # Average attention per distance (no filtering, print all for debugging)
            if head.get('focus_by_distance'):
                print(f"    Average attention per distance:")
                max_d = max(head['focus_by_distance'].keys())
                line_parts = []
                for d in range(0, max_d+1):
                    val = head['focus_by_distance'].get(d, 0.0)
                    if val >= MIN_PRINT_VAL:
                        line_parts.append(f"d={d}:{val:.3f}")
                    if len(line_parts) == ITEMS_PER_LINE or d == max_d:
                        print("      " + ", ".join(line_parts))
                        line_parts = []
        print()


def compute_front_split(log_path):
    """Compute the front/back segment split from a training log.

    Parses the per-position accuracy lines (`from xK: ...`). Rule: skip
    epoch 0 (untrained init noise); at the first epoch showing a low-high
    separation — a leading run of positions < 0.1 with the position right
    after >= 0.15 — the split k is the length of that leading run.

    Returns (start_x, k): start_x is the K of the first `from xK` label (the
    log line's first value corresponds to predicting x_K); k is the number of
    leading positions (in log-line order) forming the front segment, 0 means
    no shortcut was found, None means the positions never separated.
    """
    start_x, series, _ = parse_log(log_path)
    return start_x, split_k(series)


def print_attention_overview_table(summary):
    """Print a head x layer overview table: each cell lists the attention
    distances whose average value is >= MIN_PRINT_VAL (d = positions back)."""
    _print_header(f"Attention overview: significant targets (>= {MIN_PRINT_VAL}) per head x layer")
    n_heads = max(len(layer['heads']) for layer in summary)
    col_w = 28
    print('head \\ layer'.ljust(13) + ''.join(
        f"layer {layer['layer']}".rjust(col_w) for layer in summary))
    for h in range(n_heads):
        cells = []
        for layer in summary:
            if h < len(layer['heads']):
                fbd = layer['heads'][h].get('focus_by_distance', {})
                parts = [f"d{d}:{v:.3f}" for d, v in sorted(fbd.items()) if v >= MIN_PRINT_VAL]
                cells.append(' '.join(parts) if parts else '-')
            else:
                cells.append('-')
        print(f"head {h}".ljust(13) + ''.join(c.rjust(col_w) for c in cells))
    print()


def _resolve_recurrence(config):
    """Inspect the saved config and describe the recurrence the model was trained on.

    Returns a dict with:
      init_len:         number of initial values the recurrence needs
      next_val:         callable seq -> next value (None for dynamic_mixed/mixed)
      is_dynamic_mixed: whether the rule can change at every step
      ab_pairs, flag_start_id, dynamic_seq_len: only for dynamic_mixed
      is_mixed, rules, order, use_ab_tag: only for mixed_ab/mixed_abc
        (mixed checkpoints save ab_pairs + order and no 'recurrence' key)
    """
    p = config['p']
    recurrence = config.get('recurrence', 'addition')

    if recurrence == 'dynamic_mixed':
        print(f"Model config: dynamic_mixed, ab_pairs={config['ab_pairs']}, p={p}")
        return {'init_len': 2, 'is_dynamic_mixed': True, 'next_val': None,
                'ab_pairs': config['ab_pairs'], 'flag_start_id': p + 1,
                'dynamic_seq_len': config.get('train_len') or config.get('ood_len', 32)}
    if 'ab_pairs' in config and 'order' in config:
        # mixed_ab / mixed_abc checkpoint: one analysis per rule (see
        # _analyze_mixed_rules); there is no single next_val.
        order = config['order']
        rules = [tuple(pair) for pair in config['ab_pairs']]
        use_tag = config.get('use_ab_tag', False)
        print(f"Model config: mixed order-{order}, rules={rules}, p={p}, use_ab_tag={use_tag}")
        return {'init_len': order, 'is_dynamic_mixed': False, 'is_mixed': True,
                'next_val': None, 'rules': rules, 'order': order, 'use_ab_tag': use_tag}
    # Single-rule checkpoint (addition/multiplication/tribonacci/nonlinear,
    # including old a/b-only or minimal configs defaulting to addition).
    task = task_from_save_config(config)
    init_len, next_fn, name = single_rule_from_task(task, config)
    print(f"Model config: {name}")
    return {'init_len': init_len, 'is_dynamic_mixed': False,
            'next_val': lambda seq: next_fn(seq, p)}


def _make_dynamic_seq(p, ab_pairs, flag_start_id, length, seed):
    """Generate one dynamic_mixed sequence: [x1, x2, flag_3, x3, ..., flag_L, x_L]."""
    return datasets.generate_dynamic_sample(p, ab_pairs, flag_start_id, length,
                                            random.Random(seed))


def _make_dynamic_query_mask(length):
    # Attention matrix is over the input sequence, which has length 2*length - 2.
    # Valid prediction queries for x_k are the input positions of x_{k-1} and flag_k
    # immediately preceding x_k. For x3 (k=3), x3 is at input index 3 (0-based),
    # so we use input position 3 as the query for predicting x3.
    # General: x_k is at input index 2*(k-1) - 1, so the query position is 2*(k-1) - 1.
    total_len = 2 * length - 2
    mask = [0] * total_len
    for k in range(3, length + 1):
        query_pos = 2 * (k - 1) - 1
        mask[query_pos] = 1
    return mask


def _make_mixed_seq(p, coeffs, rule_idx, n_rules, order, use_ab_tag, length, config):
    """Generate one mixed-task sequence following a single rule.

    Mirrors MixedRecurrenceDataset's sample layout: missing-value corruption
    (when configured) is applied first, then the flag token p + rule_idx is
    prepended in tag mode.
    """
    rule = LinearRecurrenceRule(coeffs=tuple(coeffs), p=p)
    next_fn = rule.next_fn()
    seq = [random.randint(0, p - 1) for _ in range(order)]
    for _ in range(order, length):
        seq.append(next_fn(seq, p))
    if config.get('missing_prob', 0.0) > 0:
        window = torch.tensor(seq, dtype=torch.long)
        datasets.corrupt_window(window, True, p=p, init_len=order,
                                missing_prob=config['missing_prob'],
                                miss_len=config.get('miss_len', 1),
                                miss_second=config.get('miss_second', False),
                                missing_token=datasets.missing_token_id(config, p, is_mixed=True))
        seq = window.tolist()
    if use_ab_tag:
        seq = [p + rule_idx] + seq
    return rule.name, seq


def _analyze_mixed_rules(model, config, p, rec):
    """Per-rule attention analysis for mixed_ab/mixed_abc checkpoints.

    Each rule gets its own sequence (flag token prepended in tag mode) and a
    full attention summary, so rule-specific circuits can be compared.
    """
    max_len = config.get('train_len') or config.get('block_size', 20)
    n_rules = len(rec['rules'])
    for k, coeffs in enumerate(rec['rules']):
        rule_name, seq = _make_mixed_seq(p, coeffs, k, n_rules, rec['order'],
                                         rec['use_ab_tag'], max_len, config)
        print(f"\n{'#' * HEADER_WIDTH}")
        header = f"# Rule {k + 1}/{n_rules}: {rule_name}"
        if rec['use_ab_tag']:
            header += f" (flag token {p + k})"
        print(header)
        print('#' * HEADER_WIDTH)
        print(f"Test sequence (length {len(seq)}): {seq[:20]}{'...' if len(seq) > 20 else ''}")
        summary = summarize_attention_for_sequence(model, seq, query_mask=None)
        print_attention_summary(summary, seq)
        print_attention_overview_table(summary)


def analyze_model_attention(pth_path, device=None, log_path=None):
    """Main entry: load model and analyze attention on one test sequence.

    mixed_ab/mixed_abc checkpoints are analyzed per rule (see
    _analyze_mixed_rules). When log_path points to the training log and it
    shows the shortcut pattern (early positions failing while later ones
    succeed), the analysis is split into FRONT/BACK segments (see
    compute_front_split) so the two position groups' attention can be
    compared.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model(pth_path, device=device)
    config = checkpoint['config']
    p = config['p']
    rec = _resolve_recurrence(config)

    print(f"Best test accuracy: {checkpoint.get('best_accuracy', 'N/A')}")
    print(f"Training epochs: {checkpoint.get('final_epoch', 'N/A')}")

    if rec.get('is_mixed'):
        # mixed_ab/mixed_abc: one full analysis per rule, then done.
        _analyze_mixed_rules(model, config, p, rec)
        return

    # Front/back split from the training log (single-rule tasks only).
    start_x, k = (None, None)
    if log_path and os.path.exists(log_path) and not rec['is_dynamic_mixed']:
        start_x, k = compute_front_split(log_path)
        if k:
            print(f"Front/back split from log: front = first {k} prediction position(s) "
                  f"(predicting x{start_x}..x{start_x + k - 1})")
        elif k == 0:
            print("Front/back split from log: no shortcut phase found (k=0)")
        else:
            print("Front/back split from log: positions never separated (no split)")

    # Randomly generate one test sequence for attention visualization.
    # (QK property verification was removed; use verify_circle.py instead.)
    if rec['is_dynamic_mixed']:
        test_seq = _make_dynamic_seq(p, rec['ab_pairs'], rec['flag_start_id'],
                                     rec['dynamic_seq_len'], seed=0)
        query_mask = _make_dynamic_query_mask(rec['dynamic_seq_len'])
    else:
        # In-distribution length only (train_len); old checkpoints without it
        # fall back to block_size.
        max_len = config.get('train_len') or config.get('block_size', 20)
        test_seq = [random.randint(0, p - 1) for _ in range(rec['init_len'])]
        for _ in range(rec['init_len'], max_len):
            test_seq.append(rec['next_val'](test_seq))
        # Missing-value experiments: corrupt the analysis input with the same
        # rule as the dataset (RecurrenceDataset._corrupt).
        if config.get('missing_prob', 0.0) > 0:
            n_rules = len(config.get('ab_pairs') or [1])
            use_tag = config.get('use_ab_tag', False)
            missing_token = p + n_rules if use_tag else p
            helper = datasets.RecurrenceDataset(
                p=p, init_len=rec['init_len'], length=len(test_seq), verbose=False,
                missing_prob=config['missing_prob'],
                miss_len=config.get('miss_len', 1),
                miss_second=config.get('miss_second', False),
                missing_token=missing_token)
            window = torch.tensor(test_seq, dtype=torch.long)
            helper._corrupt(window, True)
            test_seq = window.tolist()
            print(f"Missing-value corruption applied to the analysis input: "
                  f"prob={config['missing_prob']}, miss_len={config.get('miss_len', 1)}, "
                  f"miss_second={config.get('miss_second', False)}")
        query_mask = None

    # Attention visualization uses this single sequence
    print(f"\nRandom test sequence (length {len(test_seq)}): {test_seq[:20]}{'...' if len(test_seq) > 20 else ''}")

    if k:
        # Segmented analysis: front/back query positions in model-input
        # coordinates. Log value j (0-based) = predicting x_{start_x+j},
        # whose query position is start_x + j - 1.
        T = len(test_seq)
        segments = [
            ('FRONT', [1 if start_x - 1 <= t < start_x - 1 + k else 0 for t in range(T)]),
            ('BACK', [1 if start_x - 1 + k <= t <= T - 2 else 0 for t in range(T)]),
        ]
        for seg_name, qm in segments:
            n_q = sum(qm)
            print(f"\n{'#' * 70}")
            print(f"# Segment {seg_name}: {n_q} query positions")
            print('#' * 70)
            seg_summary = summarize_attention_for_sequence(model, test_seq, query_mask=qm)
            print_attention_summary(seg_summary, test_seq)
            print_attention_overview_table(seg_summary)
    else:
        summary = summarize_attention_for_sequence(model, test_seq, query_mask=query_mask)
        print_attention_summary(summary, test_seq)
        print_attention_overview_table(summary)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_attention.py <path_to_pth> [training_log]")
        print("Example: python analyze_attention.py fibonacci_transformer.pth")
        sys.exit(1)

    pth_path = sys.argv[1]
    log_path = sys.argv[2] if len(sys.argv) > 2 else None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyze_model_attention(pth_path, device=device, log_path=log_path)
