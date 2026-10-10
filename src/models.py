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
    def __init__(self, n_embd, n_head, block_size, dropout=0.0, rope=None):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.rope = rope
        
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
        # Clamp the mask value to the dtype's range (defensive; fp32 下是 no-op)。
        neg = max(ATTN_MASK_NEG, torch.finfo(att.dtype).min)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, neg)
        att = F.softmax(att, dim=-1)
        
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0, rope=None, mlp_ratio=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, rope)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_ratio * n_embd),
            nn.GELU(),
            nn.Linear(mlp_ratio * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def _lm_loss(logits, targets, loss_mask):
    """Masked next-token cross-entropy shared by both model forwards.

    logits/targets are truncated to their common length; positions with
    loss_mask == 0 are excluded and the loss is normalized by the mask mass
    (an all-zero mask yields a plain 0.0).
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
    return loss


# Weight of the rule-classification auxiliary loss relative to the LM loss
# (AUX_CLASSIFY, mixed scenario only; restored from pre-2026-10-08
# MixedABTransformer, tag/cond_wte variants stay removed).
RULE_LOSS_WEIGHT = 0.5


class FibonacciTransformer(nn.Module):
    def __init__(self, p=53, d_model=64, n_head=1, n_layer=1, block_size=128, dropout=0.0,
                 mlp_ratio=4, pad_token_id=None, vocab_size=None, n_rules=0, order=None):
        super().__init__()
        self.p = p
        self.pad_token_id = pad_token_id if pad_token_id is not None else p
        self.block_size = block_size 
        self.vocab_size = vocab_size if vocab_size is not None else p + 1
        self.d_model = d_model
        self.rope = RotaryEmbedding(d_model // n_head)
        
        wte = nn.Embedding(self.vocab_size, d_model)
        
        self.transformer = nn.ModuleDict(dict(
            wte = wte,
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([TransformerBlock(d_model, n_head, block_size, dropout, self.rope, mlp_ratio)
                               for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(d_model),
        ))
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        
        self.transformer.wte.weight = self.lm_head.weight

        # Optional rule-classification head (n_rules > 0). Created before
        # self.apply so _init_weights initializes it with the same scheme as
        # everything else (normal 0.02 / zero bias), matching the historical
        # MixedABTransformer rule_head init.
        self.n_rules = n_rules
        if n_rules > 0:
            if order is None:
                raise ValueError("n_rules > 0 requires order (rule_head read offset = order + 1)")
            self.order = order
            self.rule_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, n_rules)
            )
        else:
            self.rule_head = None

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, loss_mask=None, ab_labels=None):
        t = idx.size(1)
        tok_emb = self.transformer.wte(idx)
        
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Rule head reads positions from the first transition that reveals the
        # recurrence rule: x_{order+1} onward. Tag mode is permanently removed,
        # so the offset never shifts by a leading rule token.
        rule_logits = None
        if self.rule_head is not None and ab_labels is not None:
            rule_start_offset = self.order + 1
            if t > rule_start_offset:
                rule_logits = self.rule_head(x[:, rule_start_offset:, :])

        loss = None
        if targets is not None:
            loss = _lm_loss(logits, targets, loss_mask)
            if rule_logits is not None:
                B, num_pos, _ = rule_logits.shape
                rule_logits_flat = rule_logits.reshape(-1, self.n_rules)
                ab_labels_expanded = ab_labels.unsqueeze(1).expand(B, num_pos).reshape(-1)
                rule_loss = F.cross_entropy(rule_logits_flat, ab_labels_expanded)
                loss = loss + RULE_LOSS_WEIGHT * rule_loss
        return logits, loss
