import sys
import math
import torch
import torch.nn.functional as F

# Import shared helpers from analyze_attention.py.
from analyze_attention import load_model, _build_input_embedding


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
            import core as main
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


def verify_qk_properties(model, test_sequences, init_len, query_masks=None, device='cpu'):
    """
    Verify two QK properties:
    P1: raw_score ≈ 0 when |i-j| > init_len (orthogonality)
    P2: raw_score constant across positions i for fixed distance d (time-homogeneous)

    test_sequences: list[list[int]], test sequences
    query_masks: optional list of masks, each of length T, indicating which
                 query positions should be included.
    """
    print(f"\n{'='*70}")
    print("QK Property Verification (Raw Scores, before Softmax)")
    print(f"{'='*70}")
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
                    print(f"    d={d:2d} (effective): mean={sum(vals)/len(vals):7.3f}, std={math.sqrt(sum((v-sum(vals)/len(vals))**2 for v in vals)/len(vals)):6.3f}, n={len(vals)}")
            # long-distance
            far_vals = []
            for d, vals in scores_by_d.items():
                if d > init_len:
                    far_vals.extend(vals)
            if far_vals:
                mean_far = sum(far_vals) / len(far_vals)
                std_far = math.sqrt(sum((v - mean_far)**2 for v in far_vals) / len(far_vals))
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
                    mean_v = sum(vals) / len(vals)
                    std_v = math.sqrt(sum((v - mean_v)**2 for v in vals) / len(vals))
                    cv = std_v / abs(mean_v) if abs(mean_v) > 1e-6 else float('inf')
                    print(f"    d={d:2d}: mean={mean_v:7.3f}, std={std_v:6.3f}, CV={cv:5.3f}, n={len(vals)}")
            print()

    print("="*70)
    print("QK Property Verification Completed")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qk_verification.py <path_to_pth> [init_len]")
        print("Example: python qk_verification.py fibonacci_transformer.pth 2")
        sys.exit(1)

    pth_path = sys.argv[1]
    init_len = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, checkpoint = load_model(pth_path, device=device)
    config = checkpoint['config']
    p = config['p']

    import random
    max_len = config.get('block_size', 20)
    seq = [random.randint(0, p - 1) for _ in range(init_len)]
    for i in range(init_len, max_len):
        if 'a' in config and 'b' in config:
            a, b = config['a'], config['b']
            seq.append((a * seq[-1] + b * seq[-2]) % p)
        else:
            seq.append((seq[-1] + seq[-2]) % p)

    verify_qk_properties(model, [seq], init_len=init_len, device=device)
