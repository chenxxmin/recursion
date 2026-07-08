# Recursion 训练任务：模型结构与训练设置说明

本文档汇总 `src/` 中 `addition`、`triponacci`（三阶线性递推）、`multiply`、`mixed` 四种递归训练任务的**模型结构**与**训练设置**。四种任务共享同一套 Transformer 主干，仅在递推规则、状态空间大小、输入格式以及多规则扩展上有所区别。

---

## 1. 任务定义

### 1.1 addition
二阶线性递推：

```
X(k) = a * X(k-1) + b * X(k-2)  (mod P)
```

- 默认参数：`P = 53`，`A = 1`，`B = 1`
- 初始状态长度：`init_len = 2`
- 状态空间大小：`P^2 = 53^2 = 2809`
- 模型：基础 `FibonacciTransformer`

### 1.2 multiply
二阶乘法递推：

```
X(k) = X(k-1) * X(k-2)  (mod P)
```

- 默认参数：`P = 53`
- 初始状态长度：`init_len = 2`
- 状态空间大小：`P^2`
- 模型：基础 `FibonacciTransformer`

### 1.3 tribonacci
三阶线性递推：

```
X(k) = a * X(k-1) + b * X(k-2) + c * X(k-3)  (mod P)
```

- 默认参数：`P = 23`，`A = 1`，`B = 2`，`C = 3`
- 初始状态长度：`init_len = 3`
- 状态空间大小：`P^3 = 23^3 = 12167`
- 模型：基础 `FibonacciTransformer`

### 1.4 mixed_ab
多组二阶线性递推混合训练。对每一组参数 `(a, b)`：

```
X(k) = a * X(k-1) + b * X(k-2)  (mod P)
```

- 默认参数：`P = 53`，`AB_PAIRS = [[1,1], [1,2]]`
- 初始状态长度：`init_len = 2`
- 每组规则状态空间：`P^2 = 2809`
- 模型：`MixedABTransformer`（在 `FibonacciTransformer` 基础上扩展多规则能力）

---

## 2. 公共训练设置

以下设置来自 `src/config.json` 中的 `main` 字段，除非被子任务配置或 `experiments.json` 覆盖。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `TASK` | `addition` | 当前任务类型 |
| `P` | `53` | 模数 |
| `D_MODEL` | `64` | 模型隐藏维度 |
| `N_HEAD` | `1` | 注意力头数 |
| `N_LAYER` | `1` | Transformer block 层数 |
| `BATCH_SIZE` | `128` | 每个 batch 的序列数 |
| `EPOCHS` | `10000` | 最大训练 epoch 数 |
| `LR` | `0.0003` | AdamW 学习率 |
| `WEIGHT_DECAY` | `1` | AdamW weight decay |
| `TRAIN_LEN` | `16` | 训练序列长度 |
| `OOD_LEN` | `32` | OOD 测试序列长度 |
| `DROPOUT` | `0.0` | Dropout 概率 |
| `USE_LEARNABLE_PE` | `false` | 是否使用可学习位置编码（false 使用 RoPE） |
| `MLP_RATIO` | `4` | MLP 隐藏层相对 d_model 的倍数 |
| `USE_GREEDY_GENERATE` | `true` | 生成时是否使用贪心解码 |
| `MAX_UNIQUE_RATIO` | `0.5` | 暴露给训练的初始状态比例（单任务） |
| `ENTROPY_PENALTY_WEIGHT` | `0.0` | 注意力熵惩罚权重 |
| `FIRST_TASK_WEIGHT` | `1.0` | 第一个预测位置的损失权重 |
| `NUM_MASK` | `0` | 0 表示使用任务默认的 mask 起始位置 |
| `EVAL_INTERVAL` | `20` | 每隔多少 epoch 评估一次 |
| `EARLY_STOP_NO_IMPROVE` | `1000` | 测试准确率多久未提升则早停 |
| `EARLY_STOP_ACCURACY` | `0.99` | 测试准确率达到该值则早停 |
| `RANDOM_SEED` | `42` | 随机种子 |
| `SAVE_PATH` | `fibonacci_transformer.pth` | 模型保存路径 |

### 2.1 各任务对公共配置的覆盖

| 任务 | 覆盖项 |
|------|--------|
| `addition` | 无覆盖，使用 `main` 默认配置 |
| `multiplication` | 无覆盖，使用 `main` 默认配置 |
| `tribonacci` | `P: 23`，`A: 1`，`B: 2`，`C: 3` |
| `mixed_ab` | `USE_AB_TAG: false`，`USE_CONDITIONAL_WTE: false`，`COND_WTE_SHARED_RATIO: 0.0`，`MIXED_AB_MAX_UNIQUE_RATIOS: [0.5, 0.5]` |

### 2.2 训练流程通用设置

- **优化器**：`torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)`
- **学习率调度**：`CosineAnnealingLR(optimizer, T_max=EPOCHS)`
- **梯度裁剪**：`clip_grad_norm_(model.parameters(), 1.0)`
- **损失函数**：每个时间步的交叉熵，按 `loss_mask` 平均后加上可选的熵惩罚和 rule loss
- **mask 策略**：
  - `addition` / `multiply`：默认屏蔽前 `1` 个位置（从预测第 3 项开始）
  - `tribonacci`：默认屏蔽前 `2` 个位置（从预测第 4 项开始）
  - `mixed_ab`：默认屏蔽前 `2` 个位置（从预测第 3 项开始）
- **block_size 计算**：`max(TRAIN_LEN, OOD_LEN)` 向上取整到最近的 2 的幂，默认 `32`

---

## 3. 模型结构

### 3.1 总体架构

所有任务共享一个 **Pre-LayerNorm** 的 decoder-only Transformer 架构。每个 TransformerBlock 内部都是 **先 LayerNorm，再 Attention/MLP，最后加残差**；最终输出头 `lm_head` 之前还有一次 LayerNorm（`ln_f`），属于 Pre-LN 架构的标准收尾，而非 Post-LN。

```
Input token ids (B, T)
        │
        ▼
┌─────────────────────────────────────┐
│   Embedding                         │  wte / cond_wte + optional wpe / RoPE
└─────────────────┬───────────────────┘
                  │ (B, T, D_MODEL)
                  ▼
┌─────────────────────────────────────┐
│  Dropout                            │
└─────────────────┬───────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │  LayerNorm      │  ln_1
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ CausalSelfAttention
        └────────┬────────┘
                 │
                 ├── (残差连接) ──▶ ⊕
                 │                  │
                 ▼                  ▼
        ┌─────────────────┐  ┌─────────────┐
        │  LayerNorm      │  │  x + attn   │
        │      ln_2       │  └──────┬──────┘
        └────────┬────────┘         │
                 │                  │
                 ▼                  ▼
        ┌─────────────────┐  ┌─────────────┐
        │      MLP        │  │  x + mlp    │
        └────────┬────────┘  └──────┬──────┘
                 │                  │
                 └── (残差连接) ──▶ ⊕
                                    │
        ╔═══════════════════════════╝
        ║  以上 TransformerBlock 重复 N_LAYER 次
        ╚═══════════════════════════╗
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  LayerNorm      │  ln_f（位于最后一层之后、输出头之前）
                           └────────┬────────┘
                                    │ (B, T, D_MODEL)
                                    ▼
                           ┌─────────────────┐
                           │   LM Head       │  lm_head / cond_wte 线性分类
                           └────────┬────────┘
                                    │ (B, T, VOCAB_SIZE)
                                    ▼
                             logits + loss
```

对于 `mixed_ab` 任务，额外包含一个辅助规则预测分支：

```
隐藏状态 ──▶ rule_head ──▶ 规则分类 logits ──▶ rule_loss (weight=0.5)
```

### 3.2 FibonacciTransformer（单任务模型）

类：`src/core.py::FibonacciTransformer`

| 组件 | 定义 | 说明 |
|------|------|------|
| `transformer.wte` | `nn.Embedding(vocab_size, d_model)` | 词嵌入，vocab_size = P + 1（含 PAD token） |
| `transformer.wpe` | `nn.Embedding(block_size, d_model)` | 可学习位置编码，仅在 `USE_LEARNABLE_PE=true` 时使用 |
| `rope` | `RotaryEmbedding(d_model // n_head)` | 旋转位置编码，默认开启 |
| `transformer.h` | `ModuleList[TransformerBlock]` | N_LAYER 个 Transformer block |
| `transformer.ln_f` | `nn.LayerNorm(d_model)` | 最终层归一化 |
| `lm_head` | `nn.Linear(d_model, vocab_size, bias=False)` | 输出分类头，与 `wte` 共享权重 |

#### 3.2.1 TransformerBlock

```python
TransformerBlock(
    ln_1 = LayerNorm(d_model),
    attn = CausalSelfAttention(...),
    ln_2 = LayerNorm(d_model),
    mlp  = Sequential(
        Linear(d_model, mlp_ratio * d_model),
        GELU(),
        Linear(mlp_ratio * d_model, d_model),
        Dropout(dropout)
    )
)
```

#### 3.2.2 CausalSelfAttention

```python
CausalSelfAttention(
    c_attn = Linear(d_model, 3 * d_model),  # QKV 投影
    c_proj = Linear(d_model, d_model),      # 输出投影
    attn_dropout = Dropout(dropout),
    resid_dropout = Dropout(dropout),
    causal_mask = 下三角矩阵 (block_size, block_size)
)
```

计算流程：

```python
q, k, v = c_attn(x).split(d_model, dim=-1)
q, k, v = reshape to (B, n_head, T, head_size)
if use_rope:
    q, k = apply_rotary_emb(q, k)
att = (q @ k.T) / sqrt(head_size)
att = att.masked_fill(causal_mask == 0, -1e9)
att = softmax(att, dim=-1)
out = att @ v
out = c_proj(out)
```

#### 3.2.3 RoPE 实现

- 频率：`inv_freq = 1.0 / (10000 ** (arange(0, dim, 2) / dim))`
- 对每个位置 `t` 计算 `freqs = outer(t, inv_freq)`
- 对 query/key 分前后两半执行旋转：`x * cos + rotate_half(x) * sin`

### 3.3 MixedABTransformer（多规则模型）

类：`src/core.py::MixedABTransformer`，继承自 `FibonacciTransformer`，用于 `mixed_ab` 任务。

#### 3.3.1 三种规则标识方式

| 模式 | 配置 | 输入序列 | 说明 |
|------|------|----------|------|
| `basic` | `USE_AB_TAG=false`, `USE_CONDITIONAL_WTE=false` | `[x0, x1, x2, ...]` | 无显式规则信息，模型从数据统计中区分规则 |
| `label` | `USE_AB_TAG=true`, `USE_CONDITIONAL_WTE=false` | `[rule_token, x0, x1, x2, ...]` | 序列开头添加 rule token |
| `cond_wte` | `USE_AB_TAG=false`, `USE_CONDITIONAL_WTE=true` | `[x0, x1, x2, ...]` | 同一数值在不同规则下使用不同嵌入 |

#### 3.3.2 扩展组件

| 组件 | 定义 | 说明 |
|------|------|------|
| `ab_emb` | `nn.Embedding(num_ab_pairs, d_model)` | 规则索引嵌入（当前主要保留作历史兼容） |
| `rule_head` | `Linear(d_model, d_model // 2) → ReLU → Linear(d_model // 2, num_ab_pairs)` | 从隐藏状态预测当前序列属于哪条规则 |
| `cond_wte` | `nn.Embedding(shared_size + num_ab_pairs * rule_size, d_model)` | 条件词嵌入，仅在 `cond_wte` 模式下使用 |

#### 3.3.3 label 模式下的词表扩展

```
vocab_size = P + 1 + num_ab_pairs
pad_token_id = P + num_ab_pairs
restricted_token_ids = range(P, P + num_ab_pairs)  # 生成时禁止输出 rule token
```

#### 3.3.4 cond_wte 模式

- `shared_size = int(COND_WTE_SHARED_RATIO * vocab_size)`
- `rule_size = vocab_size - shared_size`
- 共享 token：`idx < shared_size`，所有规则使用同一嵌入
- 规则专属 token：`idx >= shared_size`，按 `ab_labels` 选择对应规则段
- 输出 logits 也按共享段 / 规则段分别计算，`cond_wte` 同时作为输入嵌入和输出分类权重

### 3.4 参数初始化

- 所有 `nn.Linear` 和 `nn.Embedding`：均值 0、标准差 0.02 的正态分布
- `nn.Linear` 的 bias：初始化为 0

---

## 4. 数据集生成

### 4.1 单任务数据集（RecurrenceDataset）

类：`src/core.py::RecurrenceDataset`

1. 枚举所有 `P^init_len` 个初始状态。
2. 随机打乱顺序后遍历每个初始状态，生成完整递推循环。
3. 对循环做滑动窗口，得到长度为 `TRAIN_LEN` 的序列。
4. 根据 `MAX_UNIQUE_RATIO` 决定前多少个初始状态进入训练集，其余进入测试集。

### 4.2 多规则数据集（MixedABDataset）

类：`src/core.py::MixedABDataset`

1. 对每条 `(a, b)` 规则分别调用 `RecurrenceDataset` 生成训练/测试样本。
2. 每条规则的暴露比例由 `MIXED_AB_MAX_UNIQUE_RATIOS` 控制。
3. 合并所有规则样本后打乱训练顺序。
4. 若启用 `USE_AB_TAG`，在序列开头拼接 rule token。

---

## 5. 损失函数与评估

### 5.1 训练损失

```python
loss = masked_cross_entropy(logits, targets, loss_mask)
loss = loss + entropy_penalty
if mixed_ab:
    loss = loss + 0.5 * rule_loss
```

- `targets = x[:, 1:]`，即输入序列右移一位
- `loss_mask` 默认屏蔽前 `num_mask` 个位置，只计算后续生成位置的损失
- `rule_loss` 是规则预测头的交叉熵，权重为 0.5

### 5.2 评估指标

训练过程中每个评估 epoch 输出：

- 训练 loss 与 overall accuracy
- 测试 overall accuracy
- 每个预测位置（per-position）的准确率
- `mixed_ab` 任务额外输出每条规则的 per-rule accuracy

训练结束后执行 final generation test：

- 枚举所有初始状态
- 区分 **Exposed**（训练集中见过的初始状态）和 **Unexposed**
- 区分 **In-distribution**（位置 `< TRAIN_LEN`）和 **OOD**（位置 `>= TRAIN_LEN`）
- 输出四组准确率：Exposed-ID、Exposed-OOD、Unexposed-ID、Unexposed-OOD

---

## 6. 各任务模型与设置对比

| 项目 | addition | multiply | tribonacci | mixed_ab |
|------|----------|----------|------------|----------|
| 递推公式 | `a*x1 + b*x0 mod P` | `x1 * x0 mod P` | `a*x2 + b*x1 + c*x0 mod P` | 多组 `(a,b)` 二阶线性递推 |
| 默认 `P` | 53 | 53 | 23 | 53 |
| `init_len` | 2 | 2 | 3 | 2 |
| 状态空间 | `P^2` | `P^2` | `P^3` | 每组 `P^2` |
| 默认参数 | `A=1, B=1` | 无 | `A=1, B=2, C=3` | `AB_PAIRS=[[1,1],[1,2]]` |
| `num_mask` | 1 | 1 | 2 | 2 |
| 模型 | `FibonacciTransformer` | `FibonacciTransformer` | `FibonacciTransformer` | `MixedABTransformer` |
| 词表扩展 | 无 | 无 | 无 | 可能有 rule token / cond_wte |
| 规则预测头 | 无 | 无 | 无 | 有 `rule_head` |
| 条件冻结 | 支持 `COND_FIX` | 支持 | 支持 | 常用 `COND_FIX=WTE/LINEAR` |

---

## 7. 源码关键位置

| 功能 | 文件与位置 |
|------|------------|
| 数据生成 | `src/core.py::RecurrenceDataset`，`src/core.py::MixedABDataset` |
| 模型定义 | `src/core.py::FibonacciTransformer`，`src/core.py::MixedABTransformer` |
| 注意力层 | `src/core.py::CausalSelfAttention` |
| RoPE 实现 | `src/core.py::RotaryEmbedding`，`src/core.py::apply_rotary_emb` |
| 训练流程 | `src/core.py::run_training_engine`，`src/core.py::train_epoch`，`src/core.py::evaluate` |
| 参数冻结 | `src/core.py::freeze_partial` |
| 配置入口 | `src/core.py::run_experiment`，`src/main.py` |
| 批量运行 | `src/batch_run.py` |
| 注意力分析 | `src/analyze_attention.py` |
| 可视化 | `src/visualize.py` |

---

## 8. 备注

- 当前所有实验默认使用 **RoPE**（`USE_LEARNABLE_PE=false`），可切换为可学习位置编码。
- `mixed_ab` 的详细子实验设置（basic / label / cond_wte / freeze / overlap 等）可参考已有的 `mixed_ab_experiments_summary.md`。
- `experiments.json` 中定义的实验会覆盖 `config.json` 的对应字段，`batch_run.py` 负责配置合并与批量执行。
