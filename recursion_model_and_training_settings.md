# Recursion 训练任务：模型结构与训练设置说明

本文档汇总 `src/` 中 `addition`、`tribonacci`、`multiplication`、`nonlinear`、`mixed_ab`、`dynamic_mixed` 六种递归训练任务的**模型结构**与**训练设置**。所有任务共享同一套 Transformer 主干，仅在递推规则、状态空间大小、输入格式以及多规则扩展上有所区别。

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

### 1.2 multiplication
二阶乘法递推：

```
X(k) = X(k-1) * X(k-2)  (mod P)
```

- 默认参数：`P = 53`
- 初始状态长度：`init_len = 2`
- 状态空间大小：`P^2`
- 模型：基础 `FibonacciTransformer`
- 注意：P 为质数时，非零初始状态不会陷入 `(0,0)` 不动点。

### 1.3 tribonacci
三阶线性递推：

```
X(k) = a * X(k-1) + b * X(k-2) + c * X(k-3)  (mod P)
```

- 默认参数：`P = 23`，`A = 1`，`B = 2`，`C = 3`
- 初始状态长度：`init_len = 3`
- 状态空间大小：`P^3 = 23^3 = 12167`
- 模型：基础 `FibonacciTransformer`

### 1.4 nonlinear
二阶非线性递推：

```
X(k) = X(k-1)^2 + X(k-2)  (mod P)
```

- 默认参数：`P = 53`
- 初始状态长度：`init_len = 2`
- 状态空间大小：`P^2`
- 模型：基础 `FibonacciTransformer`
- 注意：状态映射 `(x, y) -> (y, y^2 + x)` 对任意 P 都是双射（反解 `x = z - y^2`），所有状态都在纯循环上，暴露率配额精确生效。

### 1.5 mixed_ab
多组二阶线性递推混合训练。对每一组参数 `(a, b)`：

```
X(k) = a * X(k-1) + b * X(k-2)  (mod P)
```

- 默认参数：`P = 53`，`AB_PAIRS = [[1,1], [1,2]]`
- 初始状态长度：`init_len = 2`
- 每组规则状态空间：`P^2 = 2809`
- 模型：`MixedABTransformer`（在 `FibonacciTransformer` 基础上扩展多规则能力）

### 1.6 dynamic_mixed
每步规则可变的二阶线性递推。序列格式：

```
[x1, x2, flag_3, x3, flag_4, x4, ..., flag_L, x_L]
```

其中 `flag_k` 表示生成 `x_k` 所用的规则索引。每步从 `AB_PAIRS` 中随机选一条规则：

```
x_k = a * x_{k-2} + b * x_{k-1}  (mod P)
```

- 默认参数：`P = 53`，`AB_PAIRS = [[1,1], [1,2]]`
- 训练/测试样本数：`NUM_TRAIN_SAMPLES = 10000`，`NUM_TEST_SAMPLES = 2000`
- 序列长度：`TRAIN_LEN = 16`，`OOD_LEN = 32`
- 实际输入序列长度：`2 * L - 2`
- 模型：`FibonacciTransformer`（词表扩展以容纳 flag token）
- loss 只计算 `x3, x4, ..., x_L`，flag token 与 `x2` 被 mask。

---

## 2. 公共训练设置

以下设置来自 `src/config.json` 中的 `main` 字段，除非被子任务配置或 `experiments.json` 覆盖。
**默认值一律以 `src/config.json` 为准**（下表只列含义，不再复制数值，避免双源漂移）。

| 配置项 | 说明 |
|--------|------|
| `TASK` | 当前任务类型 |
| `P` | 模数 |
| `D_MODEL` | 模型隐藏维度 |
| `N_HEAD` | 注意力头数 |
| `N_LAYER` | Transformer block 层数 |
| `BATCH_SIZE` | 每个 batch 的序列数 |
| `EPOCHS` | 最大训练 epoch 数 |
| `LR` | AdamW 学习率 |
| `WEIGHT_DECAY` | AdamW weight decay |
| `TRAIN_LEN` | 训练序列长度 |
| `OOD_LEN` | OOD 测试序列长度 |
| `DROPOUT` | Dropout 概率 |
| `USE_LEARNABLE_PE` | 是否使用可学习位置编码（false 使用 RoPE） |
| `MLP_RATIO` | MLP 隐藏层相对 d_model 的倍数 |
| `MAX_UNIQUE_RATIO` | 暴露给训练的初始状态比例（单任务 / mixed_ab 默认 fallback） |
| `ENTROPY_PENALTY_WEIGHT` | 注意力熵惩罚权重 |
| `FIRST_TASK_WEIGHT` | 第一个预测位置的损失权重 |
| `NUM_MASK` | `null` 表示使用任务默认的 mask 起始位置。**注意：当前代码中 `0` 是字面值**（首位置也计入损失与评估）；旧代码（≤2026-07-08，如 05fc707）把 `0` 当"未设置"回退到任务默认值。**今后配置不要再写 `NUM_MASK: 0`**——想排除"只看 x0 预测 x1"这个不可预测的首位置时，应显式写 `1`（参考 addition_p127_tr64_ood128 与 addition-p127w64 的口径差异） |
| `MISSING_PROB` | >0 时启用缺失值污损：train/test 窗口中 index ≥ init_len 的位置按该概率触发污损段，被污损位置的预测损失与正确率均不计入（其后一位仍计入，用于测跨缺口补全能力）。仅单规则与 mixed_ab/mixed_abc 任务支持 |
| `MISS_LEN` | 污损段最大长度：命中后污损**随机长度 ∈ [1, MISS_LEN]** 的连续段，段后第一个位置强制干净；若整个窗口没有任何一段达到 MISS_LEN 则重摇直至达到；段尾可在窗口末尾截断 |
| `MISS_SECOND` | true 时第 2 项起连续 MISS_LEN 项**必定**缺失，段后第一个位置强制干净（不会与随机段合并），随机扫描从第 MISS_LEN+2 项开始；false 为原逻辑。注意：此模式下训练样本前 init_len 个 token 不再是真实初始状态，post-train 生成测试的 exposed/unexposed 统计不可用 |
| `PREDICT_MISSING` | 污损位的指标口径：false = mask 掉不计入；true = 输入污损序列但以**污损前的值**为目标，污损位必须填出原值并计入损失与正确率（单规则与 mixed_ab/mixed_abc 支持） |
| `EVAL_INTERVAL` | 每隔多少 epoch 评估一次 |
| `EARLY_STOP_NO_IMPROVE` | 测试准确率多久未提升则早停 |
| `EARLY_STOP_ACCURACY` | 测试准确率达到该值则早停 |
| `RANDOM_SEED` | 随机种子 |
| `SAVE_PATH` | 模型保存路径；由 `batch_run.py` 自动覆盖为 `{model-base-dir}/{批次名}/{实验名}.pth` |

### 2.1 各任务对公共配置的覆盖

| 任务 | 覆盖项 |
|------|--------|
| `addition` | 无覆盖，使用 `main` 默认配置 |
| `multiplication` | 无覆盖，使用 `main` 默认配置 |
| `nonlinear` | 无覆盖，使用 `main` 默认配置 |
| `tribonacci` | `P: 23`，`A: 1`，`B: 2`，`C: 3` |
| `mixed_ab` | `USE_AB_TAG: false`，`USE_CONDITIONAL_WTE: false`，`COND_WTE_SHARED_RATIO: 0.0`，`MIXED_AB_MAX_UNIQUE_RATIOS: [0.7, 0.7]` |
| `dynamic_mixed` | `P: 53`，`AB_PAIRS: [[1,1],[1,2]]`，`NUM_TRAIN_SAMPLES: 10000`，`NUM_TEST_SAMPLES: 2000`，`TRAIN_LEN: 16`，`OOD_LEN: 32` |

### 2.2 训练流程通用设置

- **运行入口**：`src/batch_run.py` 是唯一的运行入口，已删除 `src/main.py`。
- **优化器**：`torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)`
- **学习率调度**：`CosineAnnealingLR(optimizer, T_max=EPOCHS)`
- **梯度裁剪**：`clip_grad_norm_(model.parameters(), 1.0)`
- **损失函数**：每个时间步的交叉熵，按 `loss_mask` 平均后加上可选的熵惩罚和 rule loss
- **mask 策略**：
  - `addition` / `multiplication` / `nonlinear`：默认屏蔽前 `1` 个位置（从预测第 3 项开始）
  - `tribonacci`：默认屏蔽前 `2` 个位置（从预测第 4 项开始）
  - `mixed_ab`：默认屏蔽前 `2` 个位置（从预测第 3 项开始，受 rule token 影响）
  - `dynamic_mixed`：由数据集生成 2D loss_mask，只计算 `x3, x4, ...`
- **block_size 计算**：`max(TRAIN_LEN, OOD_LEN)` 向上取整到最近的 2 的幂；`dynamic_mixed` 按 `2*L - 2` 计算。

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

### 3.2 FibonacciTransformer（单任务 / dynamic_mixed 模型）

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

### 3.4 dynamic_mixed 的词表扩展

`dynamic_mixed` 使用基础 `FibonacciTransformer`，但在 `run_experiment` 中扩展词表以容纳 flag token：

```python
vocab_size = P + 1 + len(AB_PAIRS)
pad_token_id = P
```

扩展后的 `transformer.wte` 与 `lm_head` 保持权重共享。

### 3.5 参数初始化

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

### 4.2 多规则数据集（MixedRecurrenceDataset）

类：`src/mixed_dataset.py::MixedRecurrenceDataset`（旧实现 `MixedABDataset` 已从 core.py 移出，保留在 `tests/test_mixed_ab_compat.py` 作为 golden master 对照）

1. 对每条 `(a, b)` 规则分别调用 `RecurrenceDataset` 生成训练/测试样本。
2. 每条规则的暴露比例由 `MIXED_AB_MAX_UNIQUE_RATIOS` 控制。
3. 合并所有规则样本后打乱训练顺序。
4. 若启用 `USE_AB_TAG`，在序列开头拼接 rule token。

### 4.3 动态混合数据集（DynamicMixedDataset）

类：`src/core.py::DynamicMixedDataset`

1. 使用独立的 `random.Random(seed)` 确定性生成样本。
2. 每步随机选择一条规则，生成 `x_k` 并在前面插入 `flag_k`。
3. 返回序列和 1D loss_mask，collate 后扩展为 2D。
4. 训练/测试通过 `NUM_TRAIN_SAMPLES` / `NUM_TEST_SAMPLES` 控制，使用不同 seed 区分。

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
- NaN 检测：`train_epoch` 中检测到 NaN loss 时，会将诊断信息打印到 stderr（进入 `.err` 日志），该 batch 不计入 epoch 平均 loss。

### 5.2 batch 类型区分

`train_epoch` 与 `evaluate` 使用显式的 `BatchTag`（IntEnum）区分不同 collate 模式：

| Tag | 来源 | 额外张量 |
|-----|------|----------|
| `BatchTag.PLAIN` | `collate_fn` | 无 |
| `BatchTag.DYNAMIC_MIXED` | `collate_fn_masked` / `dynamic_mixed_collate_fn` | `loss_mask` |
| `BatchTag.PLAIN_TARGET` | `collate_fn_predict` | 干净目标序列（PREDICT_MISSING） |
| `BatchTag.MIXED_AB` | `mixed_ab_collate_fn` | `ab_indices` |
| `BatchTag.MIXED_AB_MASKED` | `mixed_ab_collate_fn_masked` | `ab_indices` + `loss_mask` |
| `BatchTag.MIXED_AB_TARGET` | `mixed_ab_collate_fn_predict` | `ab_indices` + 干净目标序列 |

不再通过张量 `ndim` 推断 batch 类型。

### 5.3 评估指标

训练过程中每个评估 epoch 输出：

- 训练 loss 与 overall accuracy
- 测试 overall accuracy
- 每个预测位置（per-position）的准确率
- `mixed_ab` 任务额外输出每条规则的 per-rule accuracy
- 保存模型前打印 `Training epochs: {epoch}`，供 `batch_run.py` 汇总表提取

训练结束后执行 final generation test：

- 枚举所有初始状态
- 区分 **Exposed**（训练集中见过的初始状态）和 **Unexposed**
- 区分 **In-distribution**（位置 `< TRAIN_LEN`）和 **OOD**（位置 `>= TRAIN_LEN`）
- 输出四组准确率：Exposed-ID、Exposed-OOD、Unexposed-ID、Unexposed-OOD
- `dynamic_mixed` 没有 final generation test。

---

## 6. 各任务模型与设置对比

| 项目 | addition | multiplication | nonlinear | tribonacci | mixed_ab | dynamic_mixed |
|------|----------|----------------|-----------|------------|----------|---------------|
| 递推公式 | `a*x1 + b*x0 mod P` | `x1 * x0 mod P` | `x1^2 + x0 mod P` | `a*x2 + b*x1 + c*x0 mod P` | 多组 `(a,b)` 二阶线性递推 | 每步随机 `(a,b)` 二阶线性递推 |
| 默认 `P` | 53 | 53 | 53 | 23 | 53 | 53 |
| `init_len` | 2 | 2 | 2 | 3 | 2 | 2 |
| 状态空间 | `P^2` | `P^2` | `P^2` | `P^3` | 每组 `P^2` | 不枚举状态空间 |
| 默认参数 | `A=1, B=1` | 无 | 无 | `A=1, B=2, C=3` | `AB_PAIRS=[[1,1],[1,2]]` | `AB_PAIRS=[[1,1],[1,2]]` |
| `num_mask` | 1 | 1 | 1 | 2 | 2 | 来自数据集 |
| 模型 | `FibonacciTransformer` | `FibonacciTransformer` | `FibonacciTransformer` | `FibonacciTransformer` | `MixedABTransformer` | `FibonacciTransformer` |
| 词表扩展 | 无 | 无 | 无 | 无 | rule token / cond_wte | flag tokens |
| 规则预测头 | 无 | 无 | 无 | 无 | 有 `rule_head` | 无 |
| 条件冻结 | 支持 `COND_FIX` | 支持 | 支持 | 支持 | 常用 `COND_FIX=WTE/LINEAR` | 支持 |
| final generation test | 有 | 有 | 有 | 有 | 有 | 无 |

---

## 7. 源码关键位置

| 功能 | 文件与位置 |
|------|------------|
| 数据生成 | `src/core.py::RecurrenceDataset`，`src/mixed_dataset.py::MixedRecurrenceDataset`，`src/core.py::DynamicMixedDataset` |
| 模型定义 | `src/core.py::FibonacciTransformer`，`src/core.py::MixedABTransformer` |
| 注意力层 | `src/core.py::CausalSelfAttention` |
| RoPE 实现 | `src/core.py::RotaryEmbedding`，`src/core.py::apply_rotary_emb` |
| 训练流程 | `src/core.py::run_training_engine`，`src/core.py::train_epoch`，`src/core.py::evaluate` |
| 参数冻结 | `src/core.py::freeze_partial` |
| 配置入口 | `src/core.py::run_experiment`（由 `src/batch_run.py` 以 `python src/core.py <config>` 子进程方式调用） |
| 批量运行 | `src/batch_run.py` |
| 注意力可视化 | `src/analyze_attention.py` |
| 圆结构验证（傅里叶/单位圆） | `src/verify_circle.py`（独立脚本，不在默认流程中运行） |
| 可视化 | `src/visualize.py` |

---

## 8. 备注

- 当前所有实验默认使用 **RoPE**（`USE_LEARNABLE_PE=false`），可切换为可学习位置编码。
- `experiments.json` 中定义的实验会覆盖 `config.json` 的对应字段，`batch_run.py` 负责配置合并与批量执行；实验路由的唯一来源是每个实验条目的顶层 `task` 字段。
- `src/main.py` 已删除，所有运行必须通过 `src/batch_run.py`。
- `.gitignore` 已忽略训练产物：`*.log`、`*.err`、`*.pth`、`src/nohup.out`、`config_tmp_*.json` 等。
- 模型默认保存路径：`/data/cxm/models/{批次名}/{实验名}.pth`（由 `batch_run.py` 自动设置，可在 `experiments.json` 中通过 `SAVE_PATH` 覆盖）。
- 日志与绘图输出路径：`/data/cxm/recursion/{批次名}/logs/` 与 `/data/cxm/recursion/{批次名}/plots/`（批次名 = 实验 JSON 文件名去扩展名）。
- 可通过 `--base-dir` 和 `--model-base-dir` 参数修改这两个根目录，默认分别为 `/data/cxm/recursion` 和 `/data/cxm/models`。
