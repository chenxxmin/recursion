import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import itertools
from torch.utils.data import Dataset, DataLoader, Sampler

# ==================== Data Generation ====================
class RecurrenceDataset(Dataset):
    def __init__(self, p=127, recurrence_fn=None, recurrence_name="X(k)=?", init_len=2, num_samples=1000, len=10,
                 verbose=True):
        self.length = len
        self.p = p
        self.recurrence_fn = recurrence_fn
        self.recurrence_name = recurrence_name
        self.init_len = init_len
        self.num_samples = num_samples
        self.verbose = verbose
        self.part1 = []
        self.part2 = []
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
            
            if next_idx == start_idx:
                break
            # break when next_state=start_state, so array_len = cycle_len + (state_len-1)
            
            self.seen_indices.add(next_idx)
            cycle_vals.append(next_val)
        return cycle_vals[:1 - self.init_len]

    def run(self):
        state_space = self.p ** self.init_len

        # Randomly shuffle all state indices
        all_indices = list(range(state_space))
        random.shuffle(all_indices)

        for start_idx in all_indices:
            if start_idx in self.seen_indices:
                continue

            # MAX_UNIQUE_RATIO is the proportion of initial states exposed to training.
            # The first num_samples states (in shuffled order) go to train, the rest to test.
            where = 1 if len(self.seen_indices) < self.num_samples else 2

            # Traverse the entire cycle
            self.seen_indices.add(start_idx)
            seq = self.generate_cycle(start_idx)

            # Extend cycle by self.length
            num_inits = len(seq)
            len_to_be_extended = num_inits + self.length - 1
            while len(seq) > 0 and len(seq) < len_to_be_extended:
                seq = seq + seq

            # Append to train/test set
            if where == 1:
                for i in range(num_inits):
                    self.part1.append(torch.tensor(seq[i:i + self.length], dtype=torch.long))
            else:
                for i in range(num_inits):
                    self.part2.append(torch.tensor(seq[i:i + self.length], dtype=torch.long))

        # Shuffle sample order
        random.shuffle(self.part1)

        if self.verbose:
            cov = len(self.part1) / state_space
            print(f"[Dataset] Cycle traverse + sliding window: {len(self.part1)} + {len(self.part2)} samples, length {self.length}, mod {self.p}")
            print(f"  - Initial state coverage: {len(self.part1)}/{state_space} ({cov*100:.1f}%)")
            print(f"Recurrence: {self.recurrence_name}")
            print("-" * 50)
            print("-" * 50)

    def __len__(self):
        data = self.part1 if self.split == 'train' else self.part2
        return len(data)

    def __getitem__(self, idx):
        data = self.part1 if self.split == 'train' else self.part2
        return data[idx]

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

# ==================== Collate fn (same length within batch, direct stack) ====================
def collate_fn(batch):
    return torch.stack(batch, dim=0)

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
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, -1e9)
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
            t_logits = logits.size(1)
            t_targets = targets.size(1)
            t_min = min(t_logits, t_targets)
            logits_for_loss = logits[:, :t_min, :]
            targets = targets[:, :t_min]
            
            logits_flat = logits_for_loss.reshape(-1, self.vocab_size)
            targets_flat = targets.reshape(-1)
            
            loss_all = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            loss_all = loss_all.view(b, t_targets)
            
            if loss_mask is not None:
                masked_loss = (loss_all * loss_mask.float()).sum()
                num_loss_positions = loss_mask.float().sum()
                if num_loss_positions > 0:
                    loss = masked_loss / num_loss_positions
                else:
                    loss = torch.tensor(0.0, device=device)
            else:
                loss = loss_all.mean()
            loss = loss + total_penalty
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=10, use_greedy_generate=None, **kwargs):
        if use_greedy_generate is None:
            use_greedy_generate = getattr(self, 'use_greedy_generate', True)
        self.eval()
        
        pad_id = getattr(self, 'pad_token_id', getattr(self, 'p', None))
        if pad_id is None:
            raise AttributeError("Model must have pad_token_id or p attribute")
        
        batch_size = idx.size(0)
        
        for _ in range(max_new_tokens):
            # 上下文截断
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # 安全地提取 logits
            output = self(idx_cond, **kwargs)
            if isinstance(output, (tuple, list)):
                logits = output[0]
            elif hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
            
            # 取最后一个时间步
            logits = logits[:, -1, :]  # (B, V)
            
            # 屏蔽 PAD
            logits[:, pad_id] = float('-inf')
            # 屏蔽其他受限 token（如 rule tokens）
            restricted = getattr(self, 'restricted_token_ids', None)
            if restricted is not None:
                for tid in restricted:
                    logits[:, tid] = float('-inf')
            
            if use_greedy_generate:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==================== Training Functions ====================
def freeze_partial(model, cond_fix):
    """
    Freeze part of the model for conditional_wte analysis.
    cond_fix: 'WTE'    -> freeze embedding-related params (cond_wte, ab_emb, wte, wpe)
              'LINEAR' -> freeze transformer backbone + rule head
    Returns a list of (param, saved_value) for restoring after optimizer steps.
    """
    if cond_fix is None:
        return []
    
    frozen = []
    for name, param in model.named_parameters():
        freeze = False
        if cond_fix == "WTE":
            if any(k in name for k in ('cond_wte', 'ab_emb', 'transformer.wte', 'transformer.wpe')):
                freeze = True
        elif cond_fix == "LINEAR":
            if (name.startswith('transformer.h.') or 
                name.startswith('transformer.ln_f.') or
                name.startswith('rule_head.')):
                freeze = True
        if freeze:
            param.requires_grad = False
            frozen.append((param, param.detach().clone()))
    
    print(f"[CondFix] Frozen {cond_fix}: {len(frozen)} param groups")
    return frozen


def train_epoch(model, dataloader, optimizer, device, num_mask=1, extra_kwargs_fn=None, first_task_weight=1.0, frozen_param_states=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    pos_correct = {}
    pos_total = {}
    
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)  # (B, L)
            # If the dataset supplies a float loss_mask, use it directly.
            if len(batch) > 1 and batch[1].dtype == torch.float:
                loss_mask = batch[1].to(device)
                kwargs = {}
            else:
                extra = batch[1:]
                kwargs = extra_kwargs_fn(*extra) if extra_kwargs_fn is not None else {}
                loss_mask = None
        else:
            x = batch.to(device)
            kwargs = {}
            loss_mask = None
        B = x.size(0)
        
        # Construct targets (no padding, no PAD token)
        # Input [a,b,c,d,e], targets [b,c,d,e], aligned to L-1
        targets = x[:, 1:]  # (B, L-1)
        if loss_mask is None:
            loss_len = targets.size(1)
            loss_mask = torch.zeros(B, loss_len, dtype=torch.float, device=device)
            if loss_len > num_mask:
                loss_mask[:, num_mask:] = 1.0
                if first_task_weight != 1.0:
                    loss_mask[:, num_mask] = first_task_weight
        
        optimizer.zero_grad()
        logits, loss, *_ = model(x, targets, loss_mask, **kwargs)
        
        if loss is not None and not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Restore frozen parameters (AdamW weight decay would otherwise drift them)
            if frozen_param_states:
                with torch.no_grad():
                    for param, saved_value in frozen_param_states:
                        param.copy_(saved_value)
        
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            preds = preds[:, :targets.size(1)]  # Align length with targets
            valid_mask = loss_mask > 0
            match = ((preds == targets) & valid_mask).float()
            for pos in range(valid_mask.size(1)):
                if valid_mask[:, pos].any():
                    pos_correct[pos] = pos_correct.get(pos, 0) + match[:, pos].sum().item()
                    pos_total[pos] = pos_total.get(pos, 0) + valid_mask[:, pos].sum().item()
            total_correct += match.sum().item()
            total_samples += valid_mask.float().sum().item()
        
        total_loss += loss.item() if loss is not None else 0
    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    per_pos_acc = {pos: pos_correct[pos] / pos_total[pos] for pos in sorted(pos_total.keys())}
    return total_loss / len(dataloader), overall_acc, per_pos_acc

def evaluate(model, dataloader, device, num_mask=1, extra_kwargs_fn=None):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    pos_correct = {}
    pos_total = {}
    group_correct = {}
    group_total = {}
    
    debug_printed = False
    with torch.no_grad():
        for batch in dataloader:
            ab_labels = None
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)  # (B, L)
                # Dataset-supplied loss_mask takes priority.
                if len(batch) > 1 and batch[1].dtype == torch.float:
                    loss_mask = batch[1].to(device)
                    kwargs = {}
                else:
                    extra = batch[1:]
                    kwargs = extra_kwargs_fn(*extra) if extra_kwargs_fn is not None else {}
                    ab_labels = batch[1].to(device)
            else:
                x = batch.to(device)
                kwargs = {}
                loss_mask = None
            
            B = x.size(0)
            
            targets = x[:, 1:]  # (B, L-1)
            if loss_mask is None:
                loss_len = targets.size(1)
                loss_mask = torch.zeros(B, loss_len, dtype=torch.float, device=device)
                if loss_len > num_mask:
                    loss_mask[:, num_mask:] = 1.0  # Start calculating from predicting item (num_mask+2)
            
            logits, loss, *_ = model(x, targets, loss_mask, **kwargs)
            
            preds = logits.argmax(dim=-1)
            preds = preds[:, :targets.size(1)]  # Align length with targets
            valid_mask = loss_mask > 0
            match = ((preds == targets) & valid_mask).float()
            for pos in range(loss_mask.size(1)):
                if valid_mask[:, pos].any():
                    pos_correct[pos] = pos_correct.get(pos, 0) + match[:, pos].sum().item()
                    pos_total[pos] = pos_total.get(pos, 0) + valid_mask[:, pos].sum().item()
            total_correct += match.sum().item()
            total_samples += valid_mask.float().sum().item()
            total_loss += loss.item()
            
            # Per-rule/group accuracy for mixed_ab
            if ab_labels is not None:
                for b_idx in range(B):
                    g = ab_labels[b_idx].item()
                    group_correct[g] = group_correct.get(g, 0) + match[b_idx].sum().item()
                    group_total[g] = group_total.get(g, 0) + loss_mask[b_idx].float().sum().item()
    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    per_pos_acc = {pos: pos_correct[pos] / pos_total[pos] for pos in sorted(pos_total.keys())}
    group_acc = {g: group_correct[g] / group_total[g] for g in sorted(group_total.keys())}
    return total_loss / len(dataloader), overall_acc, per_pos_acc, group_acc


def run_training_engine(model, train_loader, test_loader, optimizer, scheduler, device,
                        epochs, eval_interval, early_stop_accuracy, early_stop_no_improve,
                        save_path, save_config, num_mask=1, extra_kwargs_fn=None, first_task_weight=1.0,
                        cond_fix=None, cond_fix_start=None,
                        cond_fix_start_a1=None, cond_fix_start_a2=None):
    print(f"\nStart training...")
    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    frozen_param_states = None
    cond_fix_triggered = False

    for epoch in range(epochs):
        train_loss, train_acc, train_pos_acc = train_epoch(
            model, train_loader, optimizer, device,
            num_mask=num_mask, extra_kwargs_fn=extra_kwargs_fn,
            first_task_weight=first_task_weight,
            frozen_param_states=frozen_param_states)
        scheduler.step()
        
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            _, test_acc, test_pos_acc, test_group_acc = evaluate(
                model, test_loader, device,
                num_mask=num_mask, extra_kwargs_fn=extra_kwargs_fn)

            # Conditional freeze: freeze cond_fix params when the start condition is met.
            # New behavior: max per-rule acc >= a1 AND min per-rule acc >= a2.
            # Old behavior (backward compat): any per-rule acc >= cond_fix_start.
            if cond_fix is not None and not cond_fix_triggered:
                use_new_cond = (cond_fix_start_a1 is not None) and (cond_fix_start_a2 is not None)
                use_old_cond = cond_fix_start is not None

                if use_new_cond and test_group_acc:
                    max_acc = max(test_group_acc.values())
                    min_acc = min(test_group_acc.values())
                    trigger = (max_acc >= cond_fix_start_a1) and (min_acc >= cond_fix_start_a2)
                    trigger_desc = f"max>=a1 ({cond_fix_start_a1:.2f}) & min>=a2 ({cond_fix_start_a2:.2f})"
                elif use_old_cond:
                    if test_group_acc:
                        trigger = any(acc >= cond_fix_start for acc in test_group_acc.values())
                    else:
                        trigger = test_acc >= cond_fix_start
                    trigger_desc = f"threshold {cond_fix_start:.2f}"
                else:
                    trigger = False
                    trigger_desc = ""

                if trigger:
                    frozen_param_states = freeze_partial(model, cond_fix)
                    cond_fix_triggered = True
                    no_improve = 0  # reset patience after freeze
                    print(f"[CondFix] Epoch {epoch}: triggered at {trigger_desc}")
                    accs = ' '.join(f"{test_group_acc[g]:.2f}" for g in sorted(test_group_acc.keys()))
                    print(f"  per-rule acc: {accs}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += eval_interval
            
            print(f"Epoch {epoch:3d} | Train: Loss={train_loss:.4f} Acc={train_acc:.1%} | "
                  f"Test: Acc={test_acc:.1%} | Best={best_acc:.1%}(@{best_epoch})")
            
            # Print per-position accuracy for test (compact single line)
            if test_pos_acc:
                positions = sorted(test_pos_acc.keys())
                # Skip the first num_mask positions (initial values, not evaluated)
                positions = [p for p in positions if p >= num_mask]
                if positions:
                    start_pos = positions[0]
                    # use_ab_tag=True: targets[pos] predicts x_pos; otherwise predicts x_{pos+1}
                    offset = 0 if getattr(model, 'use_ab_tag', False) else 1
                    start_x = start_pos + offset
                    accs = ' '.join(f"{test_pos_acc[p]:.2f}" for p in positions)
                    print(f"  from x{start_x}: {accs}")
            
            # Print per-rule/group accuracy for mixed_ab (compact single line)
            if test_group_acc:
                accs = ' '.join(f"{test_group_acc[g]:.2f}" for g in sorted(test_group_acc.keys()))
                print(f"  per-rule acc: {accs}")
            
            if test_acc >= early_stop_accuracy:
                print(f"[Early stop] Epoch {epoch}: test set reached high accuracy")
                break
            if no_improve >= early_stop_no_improve:
                print(f"[Early stop] No improvement for {early_stop_no_improve} consecutive epochs")
                break
    
    print(f"\n{'='*50}")
    print("Saving model...")
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': save_config,
        'best_accuracy': best_acc,
        'final_epoch': epoch
    }
    torch.save(save_dict, save_path)
    print(f"[OK] Model saved to: {os.path.abspath(save_path)}")
    print(f"  File size: {os.path.getsize(save_path)/1024:.1f} KB")
    print(f"  Best test accuracy: {best_acc:.2%}")
    
    return best_acc, epoch



# ==================== Mixed AB Experiment (mixed training with multiple recurrence params) ====================

class MixedABDataset(Dataset):
    def __init__(self, p=127, ab_pairs=None, fixed_ab_idx=None, num_samples=1000, len=10,
                 verbose=True, use_ab_tag=False):
        self.p = p
        self.use_ab_tag = use_ab_tag
        self.ab_pairs = ab_pairs if ab_pairs is not None else [(3, 5), (7, 11)]
        self.part1 = []
        self.part2 = []
        self.ab_labels_part1 = []
        self.ab_labels_part2 = []
        self.split = 'train'

        if fixed_ab_idx is not None:
            a, b = self.ab_pairs[fixed_ab_idx]
            ns = num_samples[fixed_ab_idx] if isinstance(num_samples, (list, tuple)) else num_samples
            self._build_with_ab(a, b, ns, len, verbose)
        else:
            self._build_mixed(num_samples, len, verbose)

    def _build_with_ab(self, a, b, num_samples, len, verbose):
        import builtins
        _len = builtins.len
        length = len
        def recurrence_fn(seq, p):
            return (a * seq[-1] + b * seq[-2]) % p

        ds = RecurrenceDataset(
            p=self.p, recurrence_fn=recurrence_fn,
            recurrence_name=f"X(k)={a}*X(k-1)+{b}*X(k-2)",
            init_len=2, num_samples=num_samples, len=length,
            verbose=verbose
        )
        ds.run()
        self.part1 = ds.part1
        self.part2 = ds.part2
        self.ab_labels_part1 = [(a, b)] * _len(ds.part1)
        self.ab_labels_part2 = [(a, b)] * _len(ds.part2)

    def _build_mixed(self, num_samples, len, verbose):
        import builtins
        _len = builtins.len
        length = len
        if isinstance(num_samples, (list, tuple)):
            num_samples_list = list(num_samples)
        else:
            num_samples_list = [num_samples] * _len(self.ab_pairs)
        all_part1 = []
        all_part2 = []
        all_labels_part1 = []
        all_labels_part2 = []
        per_rule_stats = []
        for idx, (a, b) in enumerate(self.ab_pairs):
            self._build_with_ab(a, b, num_samples_list[idx], length, False)
            all_part1.extend(self.part1)
            all_part2.extend(self.part2)
            all_labels_part1.extend(self.ab_labels_part1)
            all_labels_part2.extend(self.ab_labels_part2)
            per_rule_stats.append((a, b, _len(self.part1), _len(self.part2)))

        # Prepend rule token if use_ab_tag (DataLoader indexes the flat list, not __getitem__)
        if self.use_ab_tag:
            all_part1 = [
                torch.cat([torch.tensor([self.p + self.ab_pairs.index(l)], dtype=torch.long), seq], dim=0)
                for seq, l in zip(all_part1, all_labels_part1)
            ]
            all_part2 = [
                torch.cat([torch.tensor([self.p + self.ab_pairs.index(l)], dtype=torch.long), seq], dim=0)
                for seq, l in zip(all_part2, all_labels_part2)
            ]

        self.combined1 = list(zip(all_part1, all_labels_part1))
        self.combined2 = list(zip(all_part2, all_labels_part2))
        random.shuffle(self.combined1)
        self.part1 = [s for s, _ in self.combined1]
        self.ab_labels_part1 = [l for _, l in self.combined1]
        self.part2 = all_part2
        self.ab_labels_part2 = all_labels_part2

        # 构建可直接喂给 DataLoader / BucketBatchSampler 的扁平列表
        self.train_data = list(zip(self.part1, [self.ab_pairs.index(l) for l in self.ab_labels_part1]))
        self.test_data  = list(zip(self.part2, [self.ab_pairs.index(l) for l in self.ab_labels_part2]))

        if verbose:
            total_states = self.p * self.p
            print(f"[Dataset] Mixed mode: generated {_len(self.part1)} + {_len(self.part2)} samples, length {length}")
            print(f"AB parameter pairs: {self.ab_pairs}")
            for a, b, n_train, n_test in per_rule_stats:
                print(f"  - AB=({a},{b}): train {n_train} | test {n_test} | exposed ~{n_train/total_states*100:.1f}%")
            print("-" * 50)

    def __len__(self):
        seqs = self.part1 if self.split == 'train' else self.part2
        return len(seqs)

    def __getitem__(self, idx):
        seqs = self.part1 if self.split == 'train' else self.part2
        labels = self.ab_labels_part1 if self.split == 'train' else self.ab_labels_part2
        ab_pair = labels[idx]
        ab_idx = self.ab_pairs.index(ab_pair)
        seq = seqs[idx]
        return seq, ab_idx


def mixed_ab_collate_fn(batch):
    sequences = [item[0] for item in batch]
    ab_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return torch.stack(sequences, dim=0), ab_indices


class DynamicMixedDataset(Dataset):
    """Dataset where the recurrence rule can change at every step.

    Input format: [x1, x2, flag_3, x3, flag_4, x4, ..., flag_L, x_L]
    where flag_k indicates which rule is used to generate x_k.
    Loss is computed only on x3..x_L; x1, x2 and all flags are masked.
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
        """Generate one sequence with per-step random rules."""
        x1 = self.rng.randrange(self.p)
        x2 = self.rng.randrange(self.p)
        seq = [x1, x2]
        # loss_mask aligned to targets = seq[1:]
        loss_mask = [0.0, 0.0]  # skip x2 prediction and first flag
        for _ in range(2, self.length):
            rule_idx = self.rng.randrange(self.num_ab_pairs)
            a, b = self.ab_pairs[rule_idx]
            x_next = (a * seq[-2] + b * seq[-1]) % self.p
            flag_id = self.flag_start_id + rule_idx
            seq.append(flag_id)
            seq.append(x_next)
            # flag position -> mask, value position -> keep
            loss_mask.extend([0.0, 1.0])
        # The last flag has no following value in targets, so drop its mask entry
        # Actually seq length is 2 + 2*(length-2), target length is seq_len - 1.
        # loss_mask length should equal target length = len(seq) - 1.
        # Last element of seq is x_L; its target position is the second-to-last mask.
        # The flag before x_L contributes a target (flag), masked.
        # So loss_mask should already be len(seq)-1.
        assert len(loss_mask) == len(seq) - 1
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        mask_tensor = torch.tensor(loss_mask, dtype=torch.float)
        return seq_tensor, mask_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def dynamic_mixed_collate_fn(batch):
    sequences = [item[0] for item in batch]
    loss_masks = [item[1] for item in batch]
    return torch.stack(sequences, dim=0), torch.stack(loss_masks, dim=0)


class MixedABTransformer(FibonacciTransformer):
    def __init__(self, num_ab_pairs=1, use_greedy_generate=True, use_ab_tag=True,
                 use_conditional_wte=False, cond_wte_shared_ratio=0.0, **kwargs):
        assert not (use_conditional_wte and use_ab_tag), \
            "use_conditional_wte and use_ab_tag cannot both be True"
        p = kwargs.get('p', 53)
        if use_ab_tag:
            kwargs['vocab_size'] = p + 1 + num_ab_pairs
            kwargs['pad_token_id'] = p + num_ab_pairs
            kwargs['restricted_token_ids'] = list(range(p, p + num_ab_pairs))
        super().__init__(**kwargs)
        self.num_ab_pairs = num_ab_pairs
        self.use_greedy_generate = use_greedy_generate
        self.use_ab_tag = use_ab_tag
        self.use_conditional_wte = use_conditional_wte
        self.cond_wte_shared_ratio = cond_wte_shared_ratio

        if self.use_conditional_wte:
            shared_size = int(self.cond_wte_shared_ratio * self.vocab_size)
            rule_size = self.vocab_size - shared_size
            self.cond_wte_shared_size = shared_size
            self.cond_wte_rule_size = rule_size
            # shared rows + num_ab_pairs rule-specific segments
            self.cond_wte = nn.Embedding(
                shared_size + num_ab_pairs * rule_size, self.d_model)
            torch.nn.init.normal_(self.cond_wte.weight, mean=0.0, std=0.02)
            
        self.ab_emb = nn.Embedding(num_ab_pairs, self.d_model)
        self.rule_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, num_ab_pairs)
        )
        torch.nn.init.normal_(self.ab_emb.weight, mean=0.0, std=0.02)
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
        # With leading rule token, shift rule-head window by one position.
        rule_start_offset = 4 if self.use_ab_tag else 3
        if ab_labels is not None and t > rule_start_offset:
            rule_features = x[:, rule_start_offset:, :]
            rule_logits = self.rule_head(rule_features)
        
        loss = None
        if targets is not None:
            t_logits = logits.size(1)
            t_targets = targets.size(1)
            t_min = min(t_logits, t_targets)
            logits_for_loss = logits[:, :t_min, :]
            targets = targets[:, :t_min]
            logits_flat = logits_for_loss.reshape(-1, self.vocab_size)
            targets_flat = targets.reshape(-1)
            loss_all = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            loss_all = loss_all.view(b, t_targets)
            if loss_mask is not None:
                masked_loss = (loss_all * loss_mask.float()).sum()
                num_loss_positions = loss_mask.float().sum()
                if num_loss_positions > 0:
                    loss = masked_loss / num_loss_positions
                else:
                    loss = torch.tensor(0.0, device=device)
            else:
                loss = loss_all.mean()
            loss = loss + total_penalty
            
            if rule_logits is not None:
                B, num_pos, _ = rule_logits.shape
                rule_logits_flat = rule_logits.reshape(-1, self.num_ab_pairs)
                ab_labels_expanded = ab_labels.unsqueeze(1).expand(B, num_pos).reshape(-1)
                rule_loss = F.cross_entropy(rule_logits_flat, ab_labels_expanded)
                loss = loss + 0.5 * rule_loss
        return logits, loss, rule_logits

# ==================== Unified Experiment Entry ====================
def run_experiment(config_path=None):
    import json
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    cfg_main = config.get('main', {})
    TASK = cfg_main.get('TASK', 'addition')
    P = cfg_main['P']
    D_MODEL = cfg_main['D_MODEL']
    N_HEAD = cfg_main['N_HEAD']
    N_LAYER = cfg_main['N_LAYER']
    BATCH_SIZE = cfg_main['BATCH_SIZE']
    EPOCHS = cfg_main['EPOCHS']
    LR = cfg_main['LR']
    SAVE_PATH = cfg_main['SAVE_PATH']
    TRAIN_LEN = cfg_main['TRAIN_LEN']
    OOD_LEN = cfg_main['OOD_LEN']
    DROPOUT = cfg_main['DROPOUT']
    ENTROPY_PENALTY_WEIGHT = cfg_main.get('ENTROPY_PENALTY_WEIGHT', 0.0)
    MAX_UNIQUE_RATIO = cfg_main.get('MAX_UNIQUE_RATIO', 0.5)

    # Determine if we need extra room for rule token in mixed_ab label mode.
    # We cannot compute final BLOCK_SIZE until we know the task and use_ab_tag,
    # so defer block_size calculation to the task branches below.
    USE_LEARNABLE_PE = cfg_main.get('USE_LEARNABLE_PE', False)
    seed = cfg_main['RANDOM_SEED']
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # ========================================================================
    # Stage 1: Task branch -- prepare dataset, model, loader, training params
    # ========================================================================
    if TASK == 'mixed_ab':
        if not config.get('_BATCH_RUN_MERGED'):
            print("[Error] Config not merged. Please run via batch_run.py or merge config manually.")
            return
        cfg = dict(cfg_main)
        ab_pairs_raw = cfg.get('AB_PAIRS', [(3, 5), [7, 11]])
        AB_PAIRS = [tuple(pair) for pair in ab_pairs_raw] if ab_pairs_raw else [(3, 5), (7, 11)]
        state_space_size = P ** 2

        # block_size must accommodate max sequence length plus an optional leading rule token
        max_seq_len = max(TRAIN_LEN, OOD_LEN)
        if cfg.get('USE_AB_TAG', True):
            max_seq_len += 1
        BLOCK_SIZE = 2 ** (max_seq_len - 1).bit_length()

        # Per-rule exposure ratio for mixed_ab. MAX_UNIQUE_RATIO means exposed (train) proportion.
        if 'MIXED_AB_MAX_UNIQUE_RATIOS' in cfg:
            ratios = cfg['MIXED_AB_MAX_UNIQUE_RATIOS']
        else:
            ratios = [MAX_UNIQUE_RATIO] * len(AB_PAIRS)
        if isinstance(ratios, (int, float)):
            ratios = [ratios] * len(AB_PAIRS)
        assert len(ratios) == len(AB_PAIRS), \
            f"MIXED_AB_MAX_UNIQUE_RATIOS length ({len(ratios)}) must equal AB_PAIRS length ({len(AB_PAIRS)})"
        NUM_TRAIN_SAMPLES = [max(1, int(state_space_size * r)) for r in ratios]

        ds = MixedABDataset(p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TRAIN_SAMPLES, len=TRAIN_LEN,
                            verbose=True, use_ab_tag=cfg.get('USE_AB_TAG', True))
        train_dataset = ds.train_data
        test_dataset = ds.test_data
        
        print(f"[Model config] block_size: {BLOCK_SIZE}, train length: {TRAIN_LEN}")
        print(f"[Number theory] AB parameter pairs: {AB_PAIRS}")
        print()
        
        train_sampler = BucketBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=mixed_ab_collate_fn)
        test_sampler = BucketBatchSampler(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=mixed_ab_collate_fn)
        
        model = MixedABTransformer(p=P, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER, block_size=BLOCK_SIZE,
                                   dropout=DROPOUT, 
                                   entropy_penalty_weight=cfg['ENTROPY_PENALTY_WEIGHT'],
                                   num_ab_pairs=len(AB_PAIRS),
                                   use_greedy_generate=cfg.get('USE_GREEDY_GENERATE', True),
                                   use_learnable_pe=cfg.get('USE_LEARNABLE_PE', False),
                                   mlp_ratio=cfg.get('MLP_RATIO', 4),
                                   use_ab_tag=cfg.get('USE_AB_TAG', True),
                                   use_conditional_wte=cfg.get('USE_CONDITIONAL_WTE', False),
                                   cond_wte_shared_ratio=cfg.get('COND_WTE_SHARED_RATIO', 0.0),
)
        NUM_MASK = cfg.get('NUM_MASK', 0)
        if NUM_MASK == 0:
            # x0 and x1 are initial values; start evaluating from x2 (the first generated token)
            num_mask = 2
        else:
            num_mask = NUM_MASK
        extra_kwargs_fn = lambda ab_indices: {'ab_labels': ab_indices.to(device)}
        save_config = {
            'p': P,
            'ab_pairs': AB_PAIRS,
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
            'vocab_size': P + 1 + len(AB_PAIRS) if cfg.get('USE_AB_TAG', True) else P + 1,
            'pad_token_id': P + len(AB_PAIRS) if cfg.get('USE_AB_TAG', True) else P,
        }
        post_train_mode = 'mixed_ab'

    elif TASK == 'dynamic_mixed':
        if not config.get('_BATCH_RUN_MERGED'):
            print("[Error] Config not merged. Please run via batch_run.py or merge config manually.")
            return

        cfg = dict(cfg_main)
        cfg.update(config.get('dynamic_mixed', {}))

        P = cfg['P']
        AB_PAIRS = [tuple(pair) for pair in cfg.get('AB_PAIRS', [[1, 1], [1, 2]])]
        NUM_TRAIN_SAMPLES = cfg.get('NUM_TRAIN_SAMPLES', 10000)
        NUM_TEST_SAMPLES = cfg.get('NUM_TEST_SAMPLES', 1000)
        TRAIN_LEN = cfg.get('TRAIN_LEN', 16)
        OOD_LEN = cfg.get('OOD_LEN', 32)

        # In dynamic_mixed, each generated token is preceded by a flag token,
        # so the actual sequence length is 2*length - 2.
        max_seq_len = 2 * max(TRAIN_LEN, OOD_LEN) - 2
        BLOCK_SIZE = 2 ** (max_seq_len - 1).bit_length()

        train_dataset = DynamicMixedDataset(
            p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TRAIN_SAMPLES,
            length=TRAIN_LEN, seed=cfg.get('RANDOM_SEED', 42)
        )
        test_dataset = DynamicMixedDataset(
            p=P, ab_pairs=AB_PAIRS, num_samples=NUM_TEST_SAMPLES,
            length=OOD_LEN, seed=cfg.get('RANDOM_SEED', 42) + 1
        )

        print(f"[Model config] block_size: {BLOCK_SIZE}, train length: {TRAIN_LEN}")
        print(f"[Number theory] Dynamic mixed rules: {AB_PAIRS}")
        print()

        train_sampler = BucketBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_sampler = BucketBatchSampler(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=dynamic_mixed_collate_fn)
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=dynamic_mixed_collate_fn)

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
        # Expand lm_head and transformer.wte to accommodate flags
        if model.lm_head.weight.size(0) < model.vocab_size:
            with torch.no_grad():
                old_head = model.lm_head
                old_wte = model.transformer.wte
                new_head = nn.Linear(old_head.in_features, model.vocab_size, bias=False)
                new_head.weight.data[:old_head.weight.size(0)] = old_head.weight.data
                model.lm_head = new_head
                new_wte = nn.Embedding(model.vocab_size, old_wte.embedding_dim)
                new_wte.weight.data[:old_wte.weight.size(0)] = old_wte.weight.data
                model.transformer.wte = new_wte

        num_mask = 0  # unused; loss_mask comes from the dataset
        extra_kwargs_fn = None
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
        }
        post_train_mode = 'dynamic_mixed'
        
    elif TASK == 'addition':
        config_key = 'main'
        default_num_mask = 1
        init_len = 2
    elif TASK == 'multiplication':
        config_key = 'multiplicative'
        default_num_mask = 1
        init_len = 2
    elif TASK == 'tribonacci':
        config_key = 'tribonacci'
        default_num_mask = 2
        init_len = 3
    else:
        print(f"Unknown task: {TASK}")
        return

    # ========================================================================
    # Single recurrence task data preparation (mixed_ab handled above)
    # ========================================================================
    if TASK not in ('mixed_ab', 'dynamic_mixed'):
        if not config.get('_BATCH_RUN_MERGED'):
            print("[Error] Config not merged. Please run via batch_run.py or merge config manually.")
            return

        BLOCK_SIZE = max(TRAIN_LEN, OOD_LEN)
        BLOCK_SIZE = 2 ** (BLOCK_SIZE - 1).bit_length()
        
        cfg = dict(cfg_main)

        if TASK == 'addition':
            a = cfg['A']
            b = cfg['B']
            def recurrence_fn(seq, p):
                return (a * seq[-1] + b * seq[-2]) % p
            recurrence_name = f"X(k)=({a}*X(k-1)+{b}*X(k-2)) mod {P}"
            save_extra_config = {'a': a, 'b': b, 'recurrence': 'addition'}
        elif TASK == 'multiplication':
            def recurrence_fn(seq, p):
                return (seq[-1] * seq[-2]) % p
            recurrence_name = f"X(k)=(X(k-1)*X(k-2)) mod {P}"
            save_extra_config = {'recurrence': 'multiplicative'}
        elif TASK == 'tribonacci':
            a = cfg.get('A', 1)
            b = cfg.get('B', 1)
            c = cfg.get('C', 1)
            def recurrence_fn(seq, p):
                return (a * seq[-1] + b * seq[-2] + c * seq[-3]) % p
            recurrence_name = f"X(k)=({a}*X(k-1)+{b}*X(k-2)+{c}*X(k-3)) mod {P}"
            save_extra_config = {'a': a, 'b': b, 'c': c, 'recurrence': 'tribonacci'}

        state_space_size = P ** init_len
        NUM_TRAIN_SAMPLES = max(1, int(state_space_size * MAX_UNIQUE_RATIO))
        NUM_MASK = cfg.get('NUM_MASK', 0)
        if NUM_MASK == 0:
            num_mask = default_num_mask
        else:            num_mask = NUM_MASK

        ds = RecurrenceDataset(
            p=P, recurrence_fn=recurrence_fn, recurrence_name=recurrence_name,
            init_len=init_len, num_samples=NUM_TRAIN_SAMPLES,
            len=TRAIN_LEN
        )
        ds.run()
        train_dataset = ds.part1
        test_dataset = ds.part2
        print(f"[Model config] block_size: {BLOCK_SIZE}, train length: {TRAIN_LEN}")
        print(f"[Number theory] {recurrence_name}")
        print()

        train_sampler = BucketBatchSampler(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_sampler = BucketBatchSampler(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

        model = FibonacciTransformer(
            p=P, d_model=D_MODEL, n_head=N_HEAD, n_layer=N_LAYER,
            block_size=BLOCK_SIZE, dropout=DROPOUT,
            entropy_penalty_weight=ENTROPY_PENALTY_WEIGHT,
            use_learnable_pe=USE_LEARNABLE_PE,
            mlp_ratio=cfg.get('MLP_RATIO', 4)
        )
        extra_kwargs_fn = None
        save_config = {
            'p': P,
            'd_model': D_MODEL,
            'n_head': N_HEAD,
            'n_layer': N_LAYER,
            'block_size': BLOCK_SIZE,
            'use_learnable_pe': USE_LEARNABLE_PE,
            'mlp_ratio': cfg.get('MLP_RATIO', 4),
        }
        if save_extra_config:
            save_config.update(save_extra_config)
        post_train_mode = 'single_recurrence'

    # ========================================================================
    # Stage 2: Common training
    # ========================================================================
    model = model.to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_acc, epoch = run_training_engine(
        model, train_loader, test_loader, optimizer, scheduler, device,
        epochs=EPOCHS, eval_interval=cfg['EVAL_INTERVAL'],
        early_stop_accuracy=cfg.get('EARLY_STOP_ACCURACY', 0.99),
        early_stop_no_improve=cfg.get('EARLY_STOP_NO_IMPROVE', 100),
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
        print(f"\n{'='*50}")
        print("Final generation test (batched teacher forcing, validate from position 3)")
        print(f"{'='*50}")
        model.eval()
        test_cases = list(itertools.product(range(P), repeat=2))

        # Collect exposed initial states per AB pair from training set
        train_seen_inits = {idx: set() for idx in range(len(AB_PAIRS))}
        for i in range(len(train_dataset)):
            seq, ab_idx = train_dataset[i]
            if model.use_ab_tag:
                x0, x1 = seq[1].item(), seq[2].item()
            else:
                x0, x1 = seq[0].item(), seq[1].item()
            train_seen_inits[ab_idx].add((x0, x1))

        batch_size = 1024

        for ab_idx, (a, b) in enumerate(AB_PAIRS):
            print(f"\n--- AB pair {ab_idx+1}: ({a}, {b}) ---")
            seen = train_seen_inits[ab_idx]

            # Build full true sequences for all initial states under this rule
            full_sequences = []
            for x0, x1 in test_cases:
                seq = [x0, x1]
                while len(seq) < OOD_LEN:
                    seq.append((a * seq[-1] + b * seq[-2]) % P)
                if model.use_ab_tag:
                    seq = [P + ab_idx] + seq
                full_sequences.append(seq)
            full_sequences = torch.tensor(full_sequences, dtype=torch.long, device=device)

            ab_labels = torch.full((len(test_cases),), ab_idx, dtype=torch.long, device=device)
            exposed_list = [(x0, x1) in seen for x0, x1 in test_cases]
            exposed = torch.tensor(exposed_list, dtype=torch.bool, device=device)

            seq_len = full_sequences.size(1)
            target_len = seq_len - 1

            # Teacher-forced forward in batches
            all_preds = []
            with torch.no_grad():
                for start in range(0, len(test_cases), batch_size):
                    end = min(start + batch_size, len(test_cases))
                    batch_seq = full_sequences[start:end]
                    batch_labels = ab_labels[start:end]
                    logits, _, _ = model(batch_seq, ab_labels=batch_labels)
                    preds = logits.argmax(dim=-1)[:, :-1]
                    all_preds.append(preds)
            all_preds = torch.cat(all_preds, dim=0)

            targets = full_sequences[:, 1:]
            correct = (all_preds == targets).float()

            # Loss mask: ignore first num_mask=2 positions
            loss_mask = torch.zeros(len(test_cases), target_len, dtype=torch.float, device=device)
            if target_len > 2:
                loss_mask[:, 2:] = 1.0

            # Position masks (absolute position i = t + 1)
            in_dist_mask = torch.arange(target_len, device=device) < (TRAIN_LEN - 1)
            ood_mask = torch.arange(target_len, device=device) >= (TRAIN_LEN - 1)

            masked_correct = correct * loss_mask
            exp_corr = masked_correct[exposed]
            unexp_corr = masked_correct[~exposed]
            exp_mask = loss_mask[exposed]
            unexp_mask = loss_mask[~exposed]

            exp_id_correct = int((exp_corr * in_dist_mask).sum().item())
            exp_id_total = int((exp_mask * in_dist_mask).sum().item())
            exp_ood_correct = int((exp_corr * ood_mask).sum().item())
            exp_ood_total = int((exp_mask * ood_mask).sum().item())

            unexp_id_correct = int((unexp_corr * in_dist_mask).sum().item())
            unexp_id_total = int((unexp_mask * in_dist_mask).sum().item())
            unexp_ood_correct = int((unexp_corr * ood_mask).sum().item())
            unexp_ood_total = int((unexp_mask * ood_mask).sum().item())

            def _safe_div(a, b):
                return a / b if b > 0 else 0

            print(f"\n--- Statistics for AB=({a},{b}) ---")
            print(f"Exposed    samples: {len([tc for tc in test_cases if tc in seen]):5d} | In-dist: {exp_id_correct:5d}/{exp_id_total:5d} ({_safe_div(exp_id_correct, exp_id_total)*100:5.1f}%) | OOD: {exp_ood_correct:5d}/{exp_ood_total:5d} ({_safe_div(exp_ood_correct, exp_ood_total)*100:5.1f}%)")
            print(f"Unexposed  samples: {len([tc for tc in test_cases if tc not in seen]):5d} | In-dist: {unexp_id_correct:5d}/{unexp_id_total:5d} ({_safe_div(unexp_id_correct, unexp_id_total)*100:5.1f}%) | OOD: {unexp_ood_correct:5d}/{unexp_ood_total:5d} ({_safe_div(unexp_ood_correct, unexp_ood_total)*100:5.1f}%)")

            # Per-position accuracy for this rule
            print(f"\n--- Per-position accuracy for AB=({a},{b}) ---")
            for pos in range(2, target_len):
                i = pos + 1
                loss_pos = loss_mask[:, pos]
                corr_pos = correct[:, pos]
                exp_pos = exposed & (loss_pos > 0)
                unexp_pos = (~exposed) & (loss_pos > 0)
                is_id = i < TRAIN_LEN
                parts = []
                for label, mask in [('Exposed-ID' if is_id else 'Exposed-OOD', exp_pos),
                                     ('Unexposed-ID' if is_id else 'Unexposed-OOD', unexp_pos)]:
                    if mask.any():
                        cnt = corr_pos[mask].sum().item()
                        tot = mask.sum().item()
                        parts.append(f"{label}: {cnt}/{tot} ({_safe_div(cnt, tot)*100:5.1f}%)")
                if parts:
                    print(f"  Position {i:3d}: " + " | ".join(parts))

    elif post_train_mode == 'single_recurrence':
        print(f"\n{'='*50}")
        print(f"Final generation test (batched teacher forcing, validate from position {num_mask+1})")
        print(f"{'='*50}")
        model.eval()
        # Collect exposed initial states from training set
        train_seen_inits = set()
        for i in range(len(train_dataset)):
            seq = train_dataset[i]
            init_state = tuple(seq[:init_len].tolist())
            train_seen_inits.add(init_state)

        all_inits = list(itertools.product(range(P), repeat=init_len))
        total_states = len(all_inits)

        # Build full true sequences of length OOD_LEN for every initial state
        full_sequences = []
        for init_vals in all_inits:
            seq = list(init_vals)
            while len(seq) < OOD_LEN:
                seq.append(recurrence_fn(seq, P))
            full_sequences.append(seq)
        full_sequences = torch.tensor(full_sequences, dtype=torch.long, device=device)

        # Teacher-forced forward in batches
        batch_size = 1024
        all_preds = []
        with torch.no_grad():
            for start in range(0, total_states, batch_size):
                end = min(start + batch_size, total_states)
                batch_seq = full_sequences[start:end]
                logits, _ = model(batch_seq)
                preds = logits.argmax(dim=-1)[:, :-1]
                all_preds.append(preds)
        all_preds = torch.cat(all_preds, dim=0)

        targets = full_sequences[:, 1:]
        correct = (all_preds == targets).float()

        # Loss mask: ignore first num_mask positions
        loss_mask = torch.zeros(total_states, OOD_LEN - 1, dtype=torch.float, device=device)
        if OOD_LEN - 1 > num_mask:
            loss_mask[:, num_mask:] = 1.0

        # Position masks (t indexes targets, absolute position i = t + 1)
        in_dist_mask = torch.arange(OOD_LEN - 1, device=device) < (TRAIN_LEN - 1)
        ood_mask = torch.arange(OOD_LEN - 1, device=device) >= (TRAIN_LEN - 1)

        # Exposed / unexposed mask
        exposed_list = [init_vals in train_seen_inits for init_vals in all_inits]
        exposed = torch.tensor(exposed_list, dtype=torch.bool, device=device)

        # Accumulate statistics
        masked_correct = correct * loss_mask
        exp_corr = masked_correct[exposed]
        unexp_corr = masked_correct[~exposed]
        exp_mask = loss_mask[exposed]
        unexp_mask = loss_mask[~exposed]

        exp_id_correct = int((exp_corr * in_dist_mask).sum().item())
        exp_id_total = int((exp_mask * in_dist_mask).sum().item())
        exp_ood_correct = int((exp_corr * ood_mask).sum().item())
        exp_ood_total = int((exp_mask * ood_mask).sum().item())

        unexp_id_correct = int((unexp_corr * in_dist_mask).sum().item())
        unexp_id_total = int((unexp_mask * in_dist_mask).sum().item())
        unexp_ood_correct = int((unexp_corr * ood_mask).sum().item())
        unexp_ood_total = int((unexp_mask * ood_mask).sum().item())

        # Per-position statistics
        pos_exp_id = {}
        pos_exp_ood = {}
        pos_unexp_id = {}
        pos_unexp_ood = {}
        for pos in range(num_mask, OOD_LEN - 1):
            i = pos + 1
            loss_pos = loss_mask[:, pos]
            corr_pos = correct[:, pos]
            exp_pos = exposed & (loss_pos > 0)
            unexp_pos = (~exposed) & (loss_pos > 0)
            if i < TRAIN_LEN:
                pos_exp_id[pos] = (int(corr_pos[exp_pos].sum().item()), int(exp_pos.sum().item()))
                pos_unexp_id[pos] = (int(corr_pos[unexp_pos].sum().item()), int(unexp_pos.sum().item()))
            else:
                pos_exp_ood[pos] = (int(corr_pos[exp_pos].sum().item()), int(exp_pos.sum().item()))
                pos_unexp_ood[pos] = (int(corr_pos[unexp_pos].sum().item()), int(unexp_pos.sum().item()))

        def _safe_div(a, b):
            return a / b if b > 0 else 0

        print(f"\n--- Statistics for {recurrence_name} ---")
        print(f"Exposed    samples: {exposed.sum().item():5d} | In-dist: {exp_id_correct:5d}/{exp_id_total:5d} ({_safe_div(exp_id_correct, exp_id_total)*100:5.1f}%) | OOD: {exp_ood_correct:5d}/{exp_ood_total:5d} ({_safe_div(exp_ood_correct, exp_ood_total)*100:5.1f}%)")
        print(f"Unexposed  samples: {(~exposed).sum().item():5d} | In-dist: {unexp_id_correct:5d}/{unexp_id_total:5d} ({_safe_div(unexp_id_correct, unexp_id_total)*100:5.1f}%) | OOD: {unexp_ood_correct:5d}/{unexp_ood_total:5d} ({_safe_div(unexp_ood_correct, unexp_ood_total)*100:5.1f}%)")

        print(f"\n--- Per-position accuracy ---")
        all_positions = sorted(set().union(pos_exp_id, pos_exp_ood, pos_unexp_id, pos_unexp_ood))
        for pos in all_positions:
            i = pos + 1
            parts = []
            for label, d in [('Exposed-ID', pos_exp_id), ('Exposed-OOD', pos_exp_ood),
                             ('Unexposed-ID', pos_unexp_id), ('Unexposed-OOD', pos_unexp_ood)]:
                cnt, tot = d.get(pos, (0, 0))
                if tot > 0:
                    parts.append(f"{label}: {cnt}/{tot} ({_safe_div(cnt, tot)*100:5.1f}%)")
            if parts:
                print(f"  Position {i:3d}: " + " | ".join(parts))
