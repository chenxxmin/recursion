# mixed_ab 模式实验设置与模型设置总结

## 1. 任务概述

`mixed_ab` 任务用于研究**多组递推规则混合训练**场景。模型同时学习多组 $(a,b)$ 参数下的二阶线性递推：

$$
X(k) = a \cdot X(k-1) + b \cdot X(k-2) \pmod P
$$

默认配置使用模数 `P = 53`，常见 AB 参数对为 `[(1,1), (1,2)]`（两规则）或 `[(1,1), (1,2), (1,3)]`（三规则）。

混合训练通过 `MixedABDataset` 实现：分别为每条规则生成训练/测试样本，按规则比例暴露初始状态，然后将所有规则的数据合并后打乱训练。

---

## 2. 公共实验设置

以下设置对 `basic`、`label`、`cond_wte` 三个子实验均适用（除非子实验配置显式覆盖）。

| 配置项 | 默认值/说明 |
|--------|-------------|
| 任务类型 | `TASK: "mixed_ab"` |
| 模数 | `P = 53` |
| 序列长度 | `TRAIN_LEN = 16`（训练），`OOD_LEN = 32`（OOD 测试） |
| Batch size | `BATCH_SIZE = 128` |
| 优化器 | AdamW，`LR = 0.0003`，`WEIGHT_DECAY = 1` |
| 学习率调度 | CosineAnnealingLR，周期 `T_max = EPOCHS` |
| 训练轮数 | `EPOCHS = 10000`（含早停） |
| 评估间隔 | `EVAL_INTERVAL = 20` |
| 早停条件 | 测试准确率 `>= 0.99` 或 `EARLY_STOP_NO_IMPROVE` 轮无提升 |
| 随机种子 | `RANDOM_SEED = 42` |
| AB 参数对 | 默认 `[(1,1), (1,2)]` |
| 训练暴露比例 | `MIXED_AB_MAX_UNIQUE_RATIOS = [0.5, 0.5]`，即每条规则约 50% 的初始状态暴露给训练 |
| 位置编码 | 使用 RoPE（`USE_LEARNABLE_PE: false`） |
| Dropout | `DROPOUT = 0.0` |
| 生成方式 | 贪婪生成（`USE_GREEDY_GENERATE = true`） |

---

## 3. 子实验一：basic

### 3.1 实验设置

| 配置项 | 设置 |
|--------|------|
| 实验名前缀 | `mixed_basic_*` |
| 规则标识方式 | **不添加 rule token**，也不使用条件 WTE |
| 关键配置 | `USE_AB_TAG: false`，`USE_CONDITIONAL_WTE: false` |
| AB 规则数 | 常见 2 条（默认）或扩展为 3/4 条（文件名中 `n3`、`n4`） |
| 训练长度变体 | 默认 `TRAIN_LEN = 16`，扩展实验有 `t16`、`t32` |
| 暴露比例变体 | 默认 0.5，扩展实验有 `c0.5`、`c1.0` |

`basic` 是混合训练的最基础形式：模型仅通过数据本身的统计特征区分不同递推规则，输入序列就是普通递推序列 `[x0, x1, x2, ...]`。

### 3.2 模型设置

使用 `MixedABTransformer` 模型，关键参数：

| 配置项 | 常见取值 |
|--------|----------|
| `D_MODEL` | 64 / 128 / 256 / 512 / 1024 / 2048 |
| `N_LAYER` | 1 / 2 / 4（文件名 `l1`、`l2`、`l4`） |
| `N_HEAD` | 1 |
| `MLP_RATIO` | 4 / 8 / 16 / 32（文件名 `r4`、`r8`、`r16`、`r32`） |
| `block_size` | 32（由 `max(TRAIN_LEN, OOD_LEN)` 向上取整到 2 的幂） |

**模型结构特点**：

- 共享词嵌入 `transformer.wte` 与输出 `lm_head`（权重绑定）。
- 不使用 rule token，也不在输入中显式传入规则信息。
- 新增 `ab_emb` 与 `rule_head`：
  - `ab_emb`：规则索引的可学习嵌入。
  - `rule_head`：从隐藏状态预测当前序列属于哪条规则，辅助损失权重为 `0.5`。
- 由于 `USE_AB_TAG = false`，`rule_head` 的输入窗口起始位置为 `3`（即读取位置 3 及之后的隐藏状态）。

---

## 4. 子实验二：label

### 4.1 实验设置

| 配置项 | 设置 |
|--------|------|
| 实验名前缀 | `mixed_label_*` |
| 规则标识方式 | **在序列开头添加 rule token** |
| 关键配置 | `USE_AB_TAG: true`，`USE_CONDITIONAL_WTE: false` |
| AB 规则数 | 2 条（默认）或 3/4 条（文件名 `n3`、`n4`） |
| 特殊变体 | `e{value}` 系列、`e{a1}_{a2}` 系列用于探索冻结/条件触发阈值 |

`label` 子实验显式给每条序列打上规则标签：输入序列变为 `[rule_token, x0, x1, x2, ...]`，其中 `rule_token = P + rule_idx`，属于新增的特殊 token。

### 4.2 模型设置

| 配置项 | 常见取值 |
|--------|----------|
| `D_MODEL` | 128 / 256 |
| `N_LAYER` | 1 |
| `N_HEAD` | 1 |
| `MLP_RATIO` | 8 / 16 |
| `vocab_size` | `P + 1 + num_ab_pairs`（多出 `num_ab_pairs` 个 rule token） |
| `pad_token_id` | `P + num_ab_pairs` |
| `restricted_token_ids` | `range(P, P + num_ab_pairs)`（生成时禁止输出 rule token） |

**模型结构特点**：

- 使用 `MixedABTransformer`，`use_ab_tag = true`。
- 共享词嵌入与 lm_head 仍然绑定，但词表扩展以容纳 rule token。
- rule token 被拼接到序列最前面，**不再通过 `ab_emb` 加到 token embedding 上**。
- `rule_head` 输入窗口起始位置变为 `4`（因为 rule token 占用了位置 0，需预测的位置整体后移）。
- 不使用条件 WTE，所有规则共享同一套词嵌入。

---

## 5. 子实验三：cond_wte

### 5.1 实验设置

| 配置项 | 设置 |
|--------|------|
| 实验名前缀 | `mixed_cond_*` |
| 规则标识方式 | **使用条件词嵌入（conditional WTE）** |
| 关键配置 | `USE_AB_TAG: false`，`USE_CONDITIONAL_WTE: true` |
| 共享比例 | `COND_WTE_SHARED_RATIO`（默认 `0.0`，overlap 系列探索 `0.0 ~ 1.0`） |
| AB 规则数 | 2 条（默认）或 3/4 条 |

`cond_wte` 子实验不再添加显式 rule token，而是让**每个数值 token 根据当前规则拥有不同的嵌入**。输入序列仍为普通序列 `[x0, x1, x2, ...]`，但训练时传入 `ab_labels` 以选择对应规则的条件嵌入。

### 5.2 模型设置

| 配置项 | 常见取值 |
|--------|----------|
| `D_MODEL` | 128 / 256 |
| `N_LAYER` | 1 |
| `N_HEAD` | 1 |
| `MLP_RATIO` | 8 / 16 |
| `COND_WTE_SHARED_RATIO` | 0.0（完全规则专属）或 0.0~1.0（overlap 系列） |

**模型结构特点**：

- 使用 `MixedABTransformer`，`use_conditional_wte = true`。
- `cond_wte` 嵌入矩阵大小为：
  - `shared_size + num_ab_pairs * rule_size`
  - 其中 `shared_size = int(COND_WTE_SHARED_RATIO * vocab_size)`，`rule_size = vocab_size - shared_size`。
- **共享部分**：token id 小于 `shared_size` 的使用同一套嵌入。
- **规则专属部分**：token id 大于等于 `shared_size` 的，根据 `ab_labels` 选择对应规则段。
- 输出 logits 也使用 `cond_wte` 的权重按规则分段计算：
  - 共享 token 的输出使用共享嵌入段。
  - 规则专属 token 的输出使用对应规则的嵌入段。
- `transformer.wte` 与 `lm_head` 的绑定在 conditional WTE 模式下被**解除**，转而使用 `cond_wte` 同时作为输入嵌入与输出分类权重。

### 5.3 重要变体

#### 5.3.1 overlap 系列

位于 `results/multitask-overlap/logs/mixed_cond_*_overlapXX.log`，用于研究共享嵌入与规则专属嵌入的比例对多任务学习的影响。

- `COND_WTE_SHARED_RATIO` 从 `0.0` 到 `1.0`，步长 `0.1`。
- 当 `SHARED_RATIO = 1.0` 时，所有规则完全共享嵌入，等价于 basic。
- 当 `SHARED_RATIO = 0.0` 时，每个规则拥有完全独立的嵌入段。

#### 5.3.2 freeze 系列

位于 `results/freeze/logs/mixed_cond_*_fixWTE_*` 与 `mixed_cond_*_fixLINEAR_*`，用于研究训练过程中部分参数冻结的影响。

| 变体 | `COND_FIX` | 冻结参数 | 触发条件 |
|------|-----------|----------|----------|
| fixWTE | `"WTE"` | `cond_wte`、`ab_emb`、`transformer.wte`、`transformer.wpe` | `max per-rule acc >= a1` 且 `min per-rule acc >= a2` |
| fixLINEAR | `"LINEAR"` | `transformer.h.*`、`transformer.ln_f.*`、`rule_head.*` | 同上 |

触发阈值示例：

- `start95_90`：`a1 = 0.95, a2 = 0.90`
- `start95_80`：`a1 = 0.95, a2 = 0.80`
- `start95`：旧版阈值，只要任意规则准确率 `>= 0.95` 即触发

冻结后，优化器仍会对被冻结参数计算梯度，但每个 epoch 结束时会将被冻结参数恢复为触发时刻的值，防止 AdamW 的 weight decay 使其漂移。

---

## 6. 数据集生成细节

`MixedABDataset` 对每条规则分别执行：

1. 构建状态空间 `P^2` 中所有初始状态。
2. 随机打乱后遍历每个状态，生成完整递推循环。
3. 对循环做滑动窗口，得到长度为 `TRAIN_LEN` 的序列。
4. 根据暴露比例决定该序列进入训练集还是测试集。

当 `exposed_ratio_mode = true` 时，`MIXED_AB_MAX_UNIQUE_RATIOS` 表示**训练暴露比例**；否则为**测试未暴露比例**。所有 mixed_ab 子实验均使用 exposed_ratio_mode（即比例 = 训练暴露比例）。

---

## 7. 评估指标

训练过程中记录：

- 训练/测试 loss 与 overall accuracy。
- 每个预测位置 `x3, x4, ...` 的 per-position accuracy。
- 每条规则的 per-rule accuracy。

训练结束后执行 final generation test：

- 枚举所有 `P^2` 个初始状态。
- 区分 exposed / unexposed。
- 区分 in-distribution（位置 `< TRAIN_LEN`）与 OOD（位置 `>= TRAIN_LEN`）。
- 输出 exposed/unexposed 在 in-dist 与 OOD 上的准确率。

部分实验设置 `SKIP_FINAL_GENERATION_TEST: true` 以跳过最终生成测试。

---

## 8. 模型结构详解

`mixed_ab` 的模型主干继承自 `FibonacciTransformer`，多规则版本由 `MixedABTransformer` 扩展而来。下面按数据流从输入到输出逐层拆解。

### 8.1 总体架构图

```
Input token ids (B, T)
        │
        ▼
┌─────────────────┐
│   Embedding     │  wte / cond_wte + optional wpe / RoPE
└────────┬────────┘
         │ (B, T, D_MODEL)
         ▼
┌─────────────────┐
│  Dropout        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ TransformerBlock│ ──▶ │ TransformerBlock│  × N_LAYER
│  (Attention+MLP)│     │  (Attention+MLP)│
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  LayerNorm      │  ln_f
            └────────┬────────┘
                     │ (B, T, D_MODEL)
                     ▼
            ┌─────────────────┐
            │   LM Head       │  lm_head / cond_wte 线性分类
            └────────┬────────┘
                     │ (B, T, VOCAB_SIZE)
                     ▼
              logits + loss

辅助分支：
隐藏状态 ──▶ rule_head ──▶ 规则分类 logits ──▶ rule_loss (weight=0.5)
```

### 8.2 Token Embedding 层

#### 8.2.1 `transformer.wte`（basic / label 模式）

- **定义**：`nn.Embedding(vocab_size, d_model)`
- **词表大小**：
  - basic 模式：`vocab_size = P + 1`（数值 0~P-1 + PAD token P）
  - label 模式：`vocab_size = P + 1 + num_ab_pairs`（额外增加 rule token）
- **权重绑定**：`self.transformer.wte.weight = self.lm_head.weight`，输入嵌入与输出分类矩阵共享参数。
- **初始化**：均值 0、标准差 0.02 的正态分布。

#### 8.2.2 `cond_wte`（cond_wte 模式）

- **定义**：`nn.Embedding(shared_size + num_ab_pairs * rule_size, d_model)`
- **共享段**：前 `shared_size` 行对所有规则共享。
- **规则专属段**：后续按 `num_ab_pairs` 分段，每段大小为 `rule_size`。
- **索引计算**：
  - 共享 token（`idx < shared_size`）：直接使用 `cond_wte(idx)`。
  - 规则专属 token（`idx >= shared_size`）：
    ```
    rule_idx = idx - shared_size
    shifted = rule_idx + ab_label * rule_size + shared_size
    ```
- **输出 logits**：`cond_wte` 同时作为输入嵌入与输出分类权重，按共享段 / 规则段分别计算 logits。

### 8.3 位置编码

模型支持两种位置编码，通过 `USE_LEARNABLE_PE` 切换：

| 类型 | 配置 | 实现 | 说明 |
|------|------|------|------|
| RoPE | `USE_LEARNABLE_PE = false` | `RotaryEmbedding(d_model // n_head)` | 在 attention 内部对 q、k 施加旋转位置编码 |
| 可学习 PE | `USE_LEARNABLE_PE = true` | `nn.Embedding(block_size, d_model)` | 在 embedding 层直接加到 token embedding 上 |

当前所有实验均使用 **RoPE**。RoPE 的核心实现：

- `inv_freq = 1.0 / (10000 ** (arange(0, dim, 2) / dim))`
- 对每个位置 `t` 计算旋转角 `freqs = outer(t, inv_freq)`
- 对 query/key 向量分前后两半，执行复数旋转：
  ```python
  rotate_half(x) = cat([-x[..., half:], x[..., :half]], dim=-1)
  apply_rotary_emb(x, cos, sin) = x * cos + rotate_half(x) * sin
  ```

### 8.4 Causal Self-Attention 层

类 `CausalSelfAttention` 实现因果自注意力。

#### 8.4.1 参数与形状

| 组件 | 定义 | 输出形状 |
|------|------|----------|
| `c_attn` | `nn.Linear(d_model, 3 * d_model)` | `(B, T, 3D)` |
| q, k, v | 沿最后一维三等分 | 各 `(B, T, D)` |
| reshape | `(B, T, D) → (B, n_head, T, head_size)` | `(B, H, T, HS)`，其中 `head_size = D // H` |
| RoPE | 对 q、k 应用旋转位置编码 | `(B, H, T, HS)` |
| `c_proj` | `nn.Linear(d_model, d_model)` | `(B, T, D)` |

#### 8.4.2 注意力计算

```python
att = (q @ k.transpose(-2, -1)) / sqrt(head_size)
att = att.masked_fill(causal_mask == 0, -1e9)
att = softmax(att, dim=-1)
out = att @ v
```

- 因果掩码 `causal_mask` 为下三角矩阵，保证每个位置只能 attend 到当前及之前位置。
- 可选 **entropy penalty**：计算注意力分布的归一化熵并加入总 loss，权重由 `ENTROPY_PENALTY_WEIGHT` 控制（当前实验通常为 0）。

#### 8.4.3 Dropout 与残差

- `attn_dropout` 作用于 attention 权重。
- `resid_dropout` 作用于投影输出。
- 在 `TransformerBlock` 中：`x = x + attn(ln_1(x))`，`x = x + mlp(ln_2(x))`。

### 8.5 Transformer Block

```python
TransformerBlock(
    ln_1,           # LayerNorm(d_model)
    attn,           # CausalSelfAttention
    ln_2,           # LayerNorm(d_model)
    mlp = Sequential(
        Linear(d_model, mlp_ratio * d_model),
        GELU(),
        Linear(mlp_ratio * d_model, d_model),
        Dropout(dropout)
    )
)
```

- `MLP_RATIO` 常见取 4 / 8 / 16 / 32，决定隐藏层宽度。
- 每个 block 输出隐藏状态以及一个可选的 entropy penalty。
- 多个 block 堆叠形成 `transformer.h` ModuleList。

### 8.6 最终层归一化与输出头

#### 8.6.1 LayerNorm

- `transformer.ln_f = nn.LayerNorm(d_model)`
- 应用于最后一个 transformer block 的输出。

#### 8.6.2 LM Head

- **basic / label 模式**：`lm_head = nn.Linear(d_model, vocab_size, bias=False)`，且与 `wte` 共享权重。
- **cond_wte 模式**：不使用 `lm_head`，而是直接用 `cond_wte` 的嵌入权重做线性分类：
  - 共享 token：`logits[..., :shared_size] = x @ cond_wte[:shared_size].T`
  - 规则专属 token：对每条规则 r，`logits[ab_label==r, ..., shared_size:] = x[ab_label==r] @ cond_wte[rule_segment_r].T`

### 8.7 辅助规则预测头

`MixedABTransformer` 额外添加了规则识别分支：

```python
self.ab_emb = nn.Embedding(num_ab_pairs, d_model)
self.rule_head = Sequential(
    Linear(d_model, d_model // 2),
    ReLU(),
    Linear(d_model // 2, num_ab_pairs)
)
```

- `ab_emb` 在 basic/label 早期实现中用于给 embedding 加规则信息，当前代码中基本不再使用（被 rule token 或 cond_wte 取代），但仍保留在模型中。
- `rule_head` 从隐藏状态预测当前序列属于哪条规则：
  - `use_ab_tag = false`：读取位置 `3` 及之后的状态。
  - `use_ab_tag = true`：读取位置 `4` 及之后的状态（因为 rule token 占用了位置 0）。
- 损失函数：`loss = loss + 0.5 * rule_loss`，其中 `rule_loss = CrossEntropy(rule_logits, ab_labels)`。

### 8.8 前向传播与损失计算

```python
def forward(idx, targets=None, loss_mask=None, ab_labels=None):
    # 1. embedding
    tok_emb = embedding_layer(idx)  # (B, T, D)
    # 2. 可选 position embedding
    # 3. dropout
    # 4. transformer blocks
    for block in transformer.h:
        x, penalty = block(x)
    # 5. final ln
    x = ln_f(x)
    # 6. logits
    logits = output_head(x)
    # 7. loss
    if targets is not None:
        loss = masked_cross_entropy(logits, targets, loss_mask)
        loss = loss + entropy_penalty
        if rule_logits is not None:
            loss = loss + 0.5 * rule_loss
    return logits, loss, rule_logits
```

- `targets` 是输入序列右移一位：`targets = x[:, 1:]`。
- `loss_mask` 默认屏蔽前 `num_mask` 个位置，使模型只学习生成后续 token：
  - mixed_ab 中 `num_mask = 2`（屏蔽 x0、x1 两个初始值，从预测 x2 开始计算 loss）。
- 使用 `F.cross_entropy` 计算每个位置的分类损失，再按 mask 求平均。

### 8.9 生成函数

```python
def generate(idx, max_new_tokens=10, use_greedy_generate=True, ab_labels=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = forward(idx_cond, ab_labels=ab_labels)[0][:, -1, :]
        logits[:, pad_id] = -inf
        logits[:, restricted_token_ids] = -inf
        idx_next = argmax(logits)  # greedy
        idx = cat([idx, idx_next], dim=1)
    return idx
```

- 每次取最后一个时间步的 logits。
- 屏蔽 PAD token 与 rule token（避免生成非法 token）。
- 支持贪婪生成与采样生成。

---

## 9. 源码关键位置

- 数据生成：`src/core.py::MixedABDataset`
- 模型定义：`src/core.py::MixedABTransformer`
- 训练流程：`src/core.py::run_training_engine`
- 参数冻结：`src/core.py::freeze_partial`
- 配置合并与批量运行：`src/batch_run.py`
