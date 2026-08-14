import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== RoPE Definition ====================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    @torch.no_grad()
    def forward(self, x, seq_len=None):
        # x: (B, n_head, T, head_size) only used to get device and seq_len
        if seq_len is None:
            seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
        cos = emb.cos()[None, None, :, :]  # (1, 1, T, dim)
        sin = emb.sin()[None, None, :, :]  # (1, 1, T, dim)
        return cos, sin

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_emb(x, cos, sin):
    # x: (B, n_head, T, head_size)
    # cos, sin: (1, 1, T, head_size)
    return (x * cos) + (rotate_half(x) * sin)

# ==================== Model Definition ====================
# Finite large-negative value for attention masking. Using -inf would produce
# NaN in softmax for fully-masked rows (e.g. padded queries).
ATTN_MASK_NEG = -1e9


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0, rope=None, entropy_penalty_weight=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.rope = rope
        self.entropy_penalty_weight = entropy_penalty_weight
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size))
                                    .view(1, 1, block_size, block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        if self.rope is not None:
            cos, sin = self.rope(q, seq_len=T)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, ATTN_MASK_NEG)
        att = F.softmax(att, dim=-1)
        
        penalty = torch.tensor(0.0, device=x.device)
        if self.training and self.entropy_penalty_weight > 0:
            valid_mask = self.causal_mask[:, :, :T, :T].float()
            att_clamped = att.clamp(min=1e-9)
            entropy = -(att_clamped * torch.log(att_clamped) * valid_mask).sum(dim=-1)
            n_visible = valid_mask.sum(dim=-1).clamp(min=1)
            max_entropy = torch.log(n_visible)
            normalized_entropy = torch.where(
                n_visible > 1,
                entropy / max_entropy,
                torch.zeros_like(entropy)
            )
            penalty = penalty + self.entropy_penalty_weight * normalized_entropy.mean()
        
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, penalty

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0, rope=None, entropy_penalty_weight=0.0, mlp_ratio=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, rope, entropy_penalty_weight)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_ratio * n_embd),
            nn.GELU(),
            nn.Linear(mlp_ratio * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        attn_out, penalty = self.attn(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, penalty


def _lm_loss(logits, targets, loss_mask, total_penalty):
    """Masked next-token cross-entropy shared by both model forwards.

    logits/targets are truncated to their common length; positions with
    loss_mask == 0 are excluded and the loss is normalized by the mask mass
    (an all-zero mask yields a plain 0.0). total_penalty is added at the end.
    """
    b = logits.size(0)
    t_min = min(logits.size(1), targets.size(1))
    logits = logits[:, :t_min, :]
    targets = targets[:, :t_min]

    loss_all = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                               reduction='none')
    loss_all = loss_all.view(b, t_min)

    if loss_mask is not None:
        masked_loss = (loss_all * loss_mask.float()).sum()
        num_loss_positions = loss_mask.float().sum()
        if num_loss_positions > 0:
            loss = masked_loss / num_loss_positions
        else:
            loss = torch.tensor(0.0, device=logits.device)
    else:
        loss = loss_all.mean()
    return loss + total_penalty


class FibonacciTransformer(nn.Module):
    def __init__(self, p=53, d_model=64, n_head=1, n_layer=1, block_size=128, dropout=0.0, 
                 entropy_penalty_weight=0.0, use_learnable_pe=False, mlp_ratio=4, 
                 pad_token_id=None, vocab_size=None, restricted_token_ids=None):
        super().__init__()
        self.p = p
        self.pad_token_id = pad_token_id if pad_token_id is not None else p
        self.block_size = block_size 
        self.vocab_size = vocab_size if vocab_size is not None else p + 1
        self.restricted_token_ids = restricted_token_ids
        self.d_model = d_model
        self.use_learnable_pe = use_learnable_pe        # choose position embedding
        
        if use_learnable_pe:
            self.rope = None
            self.wpe = nn.Embedding(block_size, d_model)
        else:
            self.rope = RotaryEmbedding(d_model // n_head)
            self.wpe = None
        
        wte = nn.Embedding(self.vocab_size, d_model)
        
        self.transformer = nn.ModuleDict(dict(
            wte = wte,
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([TransformerBlock(d_model, n_head, block_size, dropout, self.rope, entropy_penalty_weight, mlp_ratio)
                               for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(d_model),
        ))
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, loss_mask=None, **kwargs):
        device = idx.device
        b, t = idx.size()
        
        tok_emb = self.transformer.wte(idx)
        
        if self.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.wpe(pos)
            tok_emb = tok_emb + pos_emb
        
        x = self.transformer.drop(tok_emb)
        
        total_penalty = torch.tensor(0.0, device=device)
        for block in self.transformer.h:
            x, p = block(x)
            total_penalty = total_penalty + p
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = _lm_loss(logits, targets, loss_mask, total_penalty)
        
        return logits, loss


# Weight of the rule-classification auxiliary loss relative to the LM loss.
RULE_LOSS_WEIGHT = 0.5


class MixedABTransformer(FibonacciTransformer):
    def __init__(self, num_ab_pairs=1, use_ab_tag=True,
                 use_conditional_wte=False, cond_wte_shared_ratio=0.0, order=2, **kwargs):
        assert not (use_conditional_wte and use_ab_tag), \
            "use_conditional_wte and use_ab_tag cannot both be True"
        p = kwargs.get('p', 53)
        if use_ab_tag:
            kwargs['vocab_size'] = p + 1 + num_ab_pairs
            kwargs['pad_token_id'] = p + num_ab_pairs
            kwargs['restricted_token_ids'] = list(range(p, p + num_ab_pairs))
        super().__init__(**kwargs)
        self.num_ab_pairs = num_ab_pairs
        self.use_ab_tag = use_ab_tag
        self.use_conditional_wte = use_conditional_wte
        self.cond_wte_shared_ratio = cond_wte_shared_ratio
        self.order = order  # recurrence order; drives rule_start_offset

        if self.use_conditional_wte:
            shared_size = int(self.cond_wte_shared_ratio * self.vocab_size)
            rule_size = self.vocab_size - shared_size
            self.cond_wte_shared_size = shared_size
            self.cond_wte_rule_size = rule_size
            # shared rows + num_ab_pairs rule-specific segments
            self.cond_wte = nn.Embedding(
                shared_size + num_ab_pairs * rule_size, self.d_model)
            torch.nn.init.normal_(self.cond_wte.weight, mean=0.0, std=0.02)
            
        self.rule_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, num_ab_pairs)
        )
        for module in self.rule_head.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, loss_mask=None, ab_labels=None):
        device = idx.device
        b, t = idx.size()
        
        if self.use_conditional_wte and ab_labels is not None:
            tok_emb = torch.zeros(b, t, self.d_model, device=device)
            # Shared tokens: all rules use the same embedding rows
            shared_mask = (idx < self.cond_wte_shared_size)
            if shared_mask.any():
                tok_emb[shared_mask] = self.cond_wte(idx[shared_mask])
            # Rule-specific tokens: each rule has its own segment
            rule_mask = ~shared_mask
            if rule_mask.any():
                rule_idx = idx[rule_mask] - self.cond_wte_shared_size
                shifted_rule_idx = (
                    rule_idx
                    + ab_labels.unsqueeze(1).expand(b, t)[rule_mask] * self.cond_wte_rule_size
                    + self.cond_wte_shared_size
                )
                tok_emb[rule_mask] = self.cond_wte(shifted_rule_idx)
        else:
            tok_emb = self.transformer.wte(idx)

        if self.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.wpe(pos)
            tok_emb = tok_emb + pos_emb
        # use_ab_tag: rule information is now carried by the leading rule token in idx,
        # no longer added to token embeddings.
        x = self.transformer.drop(tok_emb)
        total_penalty = torch.tensor(0.0, device=device)
        for block in self.transformer.h:
            x, p = block(x)
            total_penalty = total_penalty + p
        x = self.transformer.ln_f(x)
        if self.use_conditional_wte and ab_labels is not None:
            logits = torch.zeros(b, t, self.vocab_size, device=device)
            # Shared token logits (rule-independent)
            if self.cond_wte_shared_size > 0:
                logits[:, :, :self.cond_wte_shared_size] = torch.matmul(
                    x, self.cond_wte.weight[:self.cond_wte_shared_size].t()
                )
            # Rule-specific token logits
            if self.cond_wte_rule_size > 0:
                for r in range(self.num_ab_pairs):
                    mask = (ab_labels == r)
                    if mask.any():
                        start = self.cond_wte_shared_size + r * self.cond_wte_rule_size
                        end = start + self.cond_wte_rule_size
                        weight = self.cond_wte.weight[start:end]
                        logits[mask, :, self.cond_wte_shared_size:] = torch.matmul(
                            x[mask], weight.t()
                        )
        else:
            logits = self.lm_head(x)
        
        rule_logits = None
        # Rule head reads positions from the first transition that reveals the
        # recurrence rule: x_{order+1} onward; with a leading rule token, shift
        # the window by one position.
        rule_start_offset = self.order + 1 + (1 if self.use_ab_tag else 0)
        if ab_labels is not None and t > rule_start_offset:
            rule_features = x[:, rule_start_offset:, :]
            rule_logits = self.rule_head(rule_features)
        
        loss = None
        if targets is not None:
            loss = _lm_loss(logits, targets, loss_mask, total_penalty)

            if rule_logits is not None:
                B, num_pos, _ = rule_logits.shape
                rule_logits_flat = rule_logits.reshape(-1, self.num_ab_pairs)
                ab_labels_expanded = ab_labels.unsqueeze(1).expand(B, num_pos).reshape(-1)
                rule_loss = F.cross_entropy(rule_logits_flat, ab_labels_expanded)
                loss = loss + RULE_LOSS_WEIGHT * rule_loss
        return logits, loss, rule_logits
