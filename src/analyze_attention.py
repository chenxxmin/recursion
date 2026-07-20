import sys
import os
import torch
import torch.nn.functional as F
import math

# Allow running from repo root as: python src/analyze_attention.py <pth>
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import core as main

# Display/analysis constants
HEADER_WIDTH = 70       # width of printed section separators
TOP_K = 3               # how many strongest-attended positions to report per query
MAX_DISTANCE = 63       # max lag for per-distance attention statistics
ENTROPY_EPS = 1e-12     # numerical epsilon inside log() for attention entropy
CV_MEAN_EPS = 1e-6      # guard against division by a ~0 mean in the CV
MIN_PRINT_VAL = 0.01    # per-distance values below this are omitted from printout
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
    
    # backward compat: old checkpoints without use_learnable_pe, infer from state_dict keys
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
        model = main.MixedABTransformer(num_ab_pairs=num_ab_pairs, **mixedab_config)
    else:
        model = main.FibonacciTransformer(**filtered_config)
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
    elif getattr(model, 'use_embedding', True):
        tok_emb = model.transformer.wte(input_ids)
    else:
        tok_emb = F.one_hot(input_ids, num_classes=model.vocab_size).float()

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


def extract_qk_raw_scores(model, input_ids, ab_label=None):
    """
    Extract Q, K and raw QK scores (before softmax) for each layer.
    
    Returns: list of dict, each layer contains:
        - 'q': (n_head, T, head_size)
        - 'k': (n_head, T, head_size)
        - 'raw_scores': (n_head, T, T), i.e. q @ k^T / sqrt(d)
        - 'attn_weights': (n_head, T, T), attention after softmax
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
            q = main.apply_rotary_emb(q, cos, sin)
            k = main.apply_rotary_emb(k, cos, sin)
        
        # Raw QK scores (before softmax)
        raw_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(attn_module.head_size))
        
        # Attention weights (after softmax)
        att = raw_scores.masked_fill(attn_module.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        row_all_inf = torch.isinf(att).all(dim=-1, keepdim=True)
        att = att.masked_fill(row_all_inf, 0.0)
        att = F.softmax(att, dim=-1)
        
        layer_outputs.append({
            'q': q[0].detach().cpu(),
            'k': k[0].detach().cpu(),
            'raw_scores': raw_scores[0].detach().cpu(),
            'attn_weights': att[0].detach().cpu(),
        })
        
        # Continue forward
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = attn_module.c_proj(y)
        x = x + y
        x = x + block.mlp(block.ln_2(x))
    
    return layer_outputs


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
        n_head, _, _ = att.shape
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


def _mean_std(vals):
    """Return (mean, population std) of a non-empty list of numbers."""
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    return mean, std


def verify_qk_properties(model, test_sequences, init_len, query_masks=None, device='cpu'):
    """
    Verify two QK properties:
    P1: raw_score ≈ 0 when |i-j| > init_len (orthogonality)
    P2: raw_score constant across positions i for fixed distance d (time-homogeneous)

    test_sequences: list[list[int]], test sequences
    query_masks: optional list of masks, each of length T (input sequence
                 length), indicating which query positions should be included.
    """
    _print_header("QK Property Verification (Raw Scores, before Softmax)")
    print(f"Recurrence order init_len={init_len}, num test sequences={len(test_sequences)}")
    print()

    # Collect QK data for all sequences
    all_seq_data = []
    for seq_idx, seq in enumerate(test_sequences):
        input_ids = torch.tensor([seq], dtype=torch.long).to(device)
        qk_data = extract_qk_raw_scores(model, input_ids)
        mask = query_masks[seq_idx] if query_masks is not None else None
        if mask is not None and isinstance(mask, (list, tuple)):
            mask = torch.tensor(mask, dtype=torch.bool)
        all_seq_data.append({
            'seq': seq,
            'qk': qk_data,
            'T': len(seq),
            'query_mask': mask,
        })
    
    n_layers = len(all_seq_data[0]['qk'])
    n_heads = all_seq_data[0]['qk'][0]['raw_scores'].shape[0]
    
    # ==================== P1: Long-distance Orthogonality ====================
    print("--- [P1] Long-distance Orthogonality Verification ---")
    print("Ideal: raw_score mean ≈ 0 and small std when d > init_len")
    print()
    
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            # Collect raw scores by distance d
            scores_by_d = {}
            for seq_data in all_seq_data:
                T = seq_data['T']
                raw = seq_data['qk'][layer_idx]['raw_scores'][h]  # (T, T)
                mask = seq_data.get('query_mask')
                for i in range(T):
                    if mask is not None and not mask[i]:
                        continue
                    for j in range(i+1):
                        d = i - j
                        if d not in scores_by_d:
                            scores_by_d[d] = []
                        scores_by_d[d].append(raw[i, j].item())
            
            print(f"  Layer {layer_idx}, Head {h}:")
            # Effective distances
            for d in range(1, init_len + 1):
                vals = scores_by_d.get(d, [])
                if vals:
                    mean, std = _mean_std(vals)
                    print(f"    d={d:2d} (effective): mean={mean:7.3f}, std={std:6.3f}, n={len(vals)}")
            # long-distance
            far_vals = []
            for d, vals in scores_by_d.items():
                if d > init_len:
                    far_vals.extend(vals)
            if far_vals:
                mean_far, std_far = _mean_std(far_vals)
                print(f"    d>{init_len} (long-distance): mean={mean_far:7.3f}, std={std_far:6.3f}, n={len(far_vals)}")
            print()
    
    # ==================== P2: Time-homogeneity ====================
    print("--- [P2] Time-homogeneous Verification ---")
    print("Ideal: CV of raw_score across positions i ≈ 0 for fixed distance d")
    print("(CV = std / |mean|, smaller = more constant)")
    print()
    
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            print(f"  Layer {layer_idx}, Head {h}:")
            for d in range(1, init_len + 1):
                # Collect raw scores at distance d across all sequences and positions i
                vals = []
                for seq_data in all_seq_data:
                    T = seq_data['T']
                    raw = seq_data['qk'][layer_idx]['raw_scores'][h]
                    mask = seq_data.get('query_mask')
                    for i in range(d, T):
                        if mask is not None and not mask[i]:
                            continue
                        vals.append(raw[i, i - d].item())
                
                if vals:
                    mean_v, std_v = _mean_std(vals)
                    cv = std_v / abs(mean_v) if abs(mean_v) > CV_MEAN_EPS else float('inf')
                    print(f"    d={d:2d}: mean={mean_v:7.3f}, std={std_v:6.3f}, CV={cv:5.3f}, n={len(vals)}")
            print()
    
    
    print("=" * HEADER_WIDTH)
    print("QK Property Verification Completed")
    print("=" * HEADER_WIDTH)


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
            print(f"    Attention coefficients from each query position to previous positions:")
            for i in query_positions:
                if i < T:
                    parts = [f"j={j}:{attn_mat[i][j]:.3f}" for j in range(i+1)]
                    print(f"      i={i:2d} -> " + ", ".join(parts))
            
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


def analyze_model_attention(pth_path, device=None):
    """Main entry: load model and randomly generate a test sequence for attention analysis."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import random
    model, checkpoint = load_model(pth_path, device=device)
    config = checkpoint['config']
    p = config['p']
    recurrence = config.get('recurrence', 'addition')
    is_dynamic_mixed = False

    if 'c' in config:
        a, b, c = config['a'], config['b'], config['c']
        print(f"Model config: tribonacci, a={a}, b={b}, c={c}, p={p}")
        init_len = 3
        def next_val(seq):
            return (a * seq[-1] + b * seq[-2] + c * seq[-3]) % p
    elif recurrence == 'multiplicative':
        print(f"Model config: multiplication, p={p}")
        init_len = 2
        def next_val(seq):
            return (seq[-1] * seq[-2]) % p
    elif recurrence == 'dynamic_mixed':
        print(f"Model config: dynamic_mixed, ab_pairs={config['ab_pairs']}, p={p}")
        init_len = 2
        is_dynamic_mixed = True
        ab_pairs = config['ab_pairs']
        flag_start_id = p + 1
        dynamic_seq_len = config.get('ood_len', 32)
    elif 'a' in config and 'b' in config:
        a, b = config['a'], config['b']
        print(f"Model config: addition, a={a}, b={b}, p={p}")
        init_len = 2
        def next_val(seq):
            return (a * seq[-1] + b * seq[-2]) % p
    else:
        # Fallback: old checkpoint or minimal config defaults to addition
        print(f"Model config: addition (default), p={p}")
        init_len = 2
        def next_val(seq):
            return (seq[-1] + seq[-2]) % p
    
    print(f"Best test accuracy: {checkpoint.get('best_accuracy', 'N/A')}")
    print(f"Training epochs: {checkpoint.get('final_epoch', 'N/A')}")
    
    # Randomly generate one test sequence for attention visualization.
    # (QK property verification lives in qk_verification.py, which reuses
    # verify_qk_properties from this module.)
    max_len = config.get('block_size', 20)
    test_sequences = []
    query_masks = None

    if is_dynamic_mixed:
        def make_dynamic_seq(seed):
            rng = random.Random(seed)
            x1 = rng.randint(0, p - 1)
            x2 = rng.randint(0, p - 1)
            seq = [x1, x2]
            values = [x1, x2]  # Only numeric values, used for recurrence
            for _ in range(2, dynamic_seq_len):
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

        def make_dynamic_query_mask(length):
            # Attention matrix is over the input sequence, which has length 2*length - 2.
            # Valid prediction queries for x_k are the input positions of x_{k-1} and flag_k
            # immediately preceding x_k. For x3 (k=3), x3 is at input index 3 (0-based),
            # so we use input position 3 as the query for predicting x3.
            # General: x_k is at input index 2*(k-1), so the query position is 2*(k-1).
            total_len = 2 * length - 2
            mask = [0] * total_len
            for k in range(3, length + 1):
                query_pos = 2 * (k - 1)
                mask[query_pos] = 1
            return mask

        test_sequences.append(make_dynamic_seq(seed=0))
        query_masks = [make_dynamic_query_mask(dynamic_seq_len)]
    else:
        init = [random.randint(0, p - 1) for _ in range(init_len)]
        seq = init[:]
        for i in range(init_len, max_len):
            seq.append(next_val(seq))
        test_sequences.append(seq)

    # Attention visualization uses this single sequence
    print(f"\nRandom test sequence (length {len(test_sequences[0])}): {test_sequences[0][:20]}{'...' if len(test_sequences[0]) > 20 else ''}")
    summary = summarize_attention_for_sequence(
        model, test_sequences[0],
        query_mask=query_masks[0] if query_masks is not None else None)
    print_attention_summary(summary, test_sequences[0])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_attention.py <path_to_pth>")
        print("Example: python analyze_attention.py fibonacci_transformer.pth")
        sys.exit(1)
    
    pth_path = sys.argv[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyze_model_attention(pth_path, device=device)
