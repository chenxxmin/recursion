# Recursion 训练任务：模型结构与训练设置说明

本文档汇总 `src/` 中递归训练任务的**模型结构**与**训练设置**。

**2026-10-08 场景统一**（设计规格：`docs/superpowers/specs/2026-10-08-scenario-unification-design.md`）后，task 体系收敛为**三个场景**（`single` / `mixed` / `action`，由 `ORDER` × 污损两维参数化）加**三个非线性例外 task**；全部共享唯一模型类 `FibonacciTransformer`，旧 task 名与旧键由归一化层 `src/scenarios.py::normalize_task_config` 兼容翻译。

---

## 1. 任务定义

### 1.1 场景矩阵总览

| 场景（`TASK`） | 语义 | `ORDER` | `RULES` | `NUM_MASK` 默认 | 词表 |
|---|---|---|---|---|---|
| `single` | 单条线性递推 | ≥2 | 恰好 1 条 | `order-1` | `P+1` |
| `mixed` | 多规则混合（每条样本固定一条规则，逐样本带规则标签） | ≥2 | ≥1 条 | `order-1` | 恒 `P+1` |
| `action` | 每步规则可变（flag 交错布局） | ≥2 | ≥1 条 | 数据集生成 loss_mask | `P+1+N规则` |

例外 task（不进场景矩阵，无 `ORDER`/`RULES`，归一化层原样透传）：`multiplication` / `nonlinear` / `nonlinear_mul`。

三个场景统一的两个配置维度：

- **阶数**：`ORDER`（≥2 整数，新格式必须显式写）。
- **污损**：`MISSING_PROB` + `MISS_LEN`（0 / 不写 = 干净）。**predict 是唯一口径**：污损位以原值为目标并计入损失与正确率；mask 口径（污损位不计入）已于 2026-10-08 删除。污损为在线模式：数据集只存干净样本，`make_missing_collate` / `make_mixed_missing_collate` / `make_action_missing_collate` 逐 batch 新鲜随机污损（每 epoch 缺失位置都不同，train/test 同样）。

`RULES` 方向三场景统一为 **canonical：第一个系数乘最新的值**（`x_k = c1·x_{k-1} + c2·x_{k-2} + ...`，与 `LinearRecurrenceRule` 一致），详见 §1.6。

### 1.2 single

任意阶单条线性递推：

```
X(k) = c1*X(k-1) + c2*X(k-2) + ... + c_o*X(k-o)  (mod P)
```

- `ORDER` ≥2，`RULES` 恰好 1 条（canonical）。
- 初始状态长度 `init_len = ORDER`；状态空间大小 `P^order`。
- 暴露率由 `EXPOSURE_RATIOS`（标量，默认 0.7）控制；`NUM_TRAIN_SAMPLES` 显式设置时覆盖。
- 高阶大 P（如 order=4 @ P=127，`P^4 ≈ 2.6 亿`）必须配 `STATE_SPACE_CAP` 或用 `DATA_MODE: sampled_fresh_test`（见 §4.1），否则全状态枚举爆炸。
- 旧 task 名兼容别名：`addition`（order=2，默认 `P=53`、`[1,1]`）、`tribonacci`（order=3，默认 `P=23`、`[1,2,3]`）、`tetranacci`（order=4，默认 `P=23`、`[1,1,1,1]`）。新实验一律写 `single` + `ORDER` + `RULES`。

### 1.3 mixed

多组同阶线性递推混合训练，每条样本固定属于一条规则：

```
X(k) = c1*X(k-1) + c2*X(k-2) + ... + c_o*X(k-o)  (mod P)   # 每条样本取 RULES 中一条
```

- `ORDER` ≥2，`RULES` ≥1 条（canonical）；每条规则状态空间 `P^order`。
- 每规则样本数由 `EXPOSURE_RATIOS`（标量 = 各规则同一比例；列表 = 逐规则，长度必须等于规则数；默认各 0.7）控制；`NUM_TRAIN_SAMPLES`（标量或逐规则列表）显式设置时覆盖。
- 词表恒 `P+1`；模型直接用 `FibonacciTransformer`（rule token、cond_wte 均已于 2026-10-08 删除；rule_head 规则分类辅助头可通过 `AUX_CLASSIFY: true` 可选启用，见 §2.1）。样本仍带逐样本规则标签（`ab_indices`），评估输出 per-rule accuracy。
- 旧 task 名兼容别名：`mixed_ab`（order=2，旧键 `AB_PAIRS`，本就是 canonical 直接用）、`mixed_abc`（order=3，旧键 `ABC_PAIRS`，同上）。

### 1.4 action

每步规则可变的线性递推。序列格式（o = ORDER，L = 窗口长度）：

```
[x0, x1, ..., x_{o-1}, flag_o, x_o, flag_{o+1}, x_{o+1}, ..., flag_{L-1}, x_{L-1}]
```

其中 `flag_k` 表示生成 `x_k` 所用的规则索引，每步从 `RULES` 中随机选一条。窗口共 **L 个值**（`x0..x_{L-1}`，含 o 个初值 + L−o 个生成值），**实际输入序列长度为 `2L - order`**。

- 训练/测试样本数：`NUM_TRAIN_SAMPLES`（默认 10000）、`NUM_TEST_SAMPLES`（默认 2000）；不枚举状态空间。
- 词表：`P+1+N规则`（值 0..P-1，missing token = `P`，flag token = `P+1..P+N`）；`run_experiment` 中扩展 `wte`/`lm_head`（保持权重共享，见 §3.3）。
- loss 只计算 `x_o..x_{L-1}`（数据集生成 loss_mask），flag token 与初值被 mask；逐位置 loss mask 语义不随场景统一改变。
- 污损：`MISSING_PROB > 0` 时经 `make_action_missing_collate` 在线污损——每个值位置独立以 `MISSING_PROB` 概率命中并污损随机长度 ∈ [1, `MISS_LEN`] 的连续段，段后第一个值强制干净；若整段扫描没有任何 run 达到 `MISS_LEN` 则重摇（全净样本也接受）。flag token 永不污损；predict 口径（污损位以原值为目标并计入损失）。
- 旧 task 名兼容别名：`action`（order=2，旧键 `AB_PAIRS`，⚠ 反向约定见 §1.6）、`action_trib`（order=3，旧键 `ABC_PAIRS`，同上）、`dynamic_mixed`（已废弃，按 `action` 处理）。

### 1.5 非线性例外 task

不走场景矩阵，规则硬编码在 `rules.py::single_rule_from_task`，均 order=2、`init_len=2`、状态空间 `P^2`、默认 `P=53`、`NUM_MASK` 默认 1：

- **`multiplication`**：`X(k) = X(k-1) * X(k-2) (mod P)`。P 为质数时，非零初始状态不会陷入 `(0,0)` 不动点。
- **`nonlinear`**：`X(k) = X(k-1)^2 + X(k-2) (mod P)`。状态映射 `(x, y) -> (y, y^2 + x)` 对任意 P 都是双射（反解 `x = z - y^2`），所有状态都在纯循环上，暴露率配额精确生效。
- **`nonlinear_mul`**：`X(k) = X(k-2) * X(k-1)^2 (mod P)`。状态映射 `(x, y) -> (y, x*y^2)` 不是双射（`y=0` 流入 `(0,0)` 不动点），存在瞬态轨迹，`RecurrenceDataset` 内部处理（截断轨迹逐步扩展）。

### 1.6 旧 task 名兼容与系数方向约定（⚠ 重要）

归一化层 `src/scenarios.py::normalize_task_config` 把任何合法 task 写法翻译成统一结构 `{scenario, order, rules(canonical), exposure_ratios, source}`：

| 旧 task | scenario | order | rules 来源 |
|---|---|---|---|
| `addition` | single | 2 | `[A, B]`（canonical） |
| `tribonacci` | single | 3 | `[A, B, C]` |
| `tetranacci` | single | 4 | `[A, B, C, D]` |
| `mixed_ab` | mixed | 2 | `AB_PAIRS`（本就是 canonical，直接用） |
| `mixed_abc` | mixed | 3 | `ABC_PAIRS`（同上） |
| `action` / `dynamic_mixed` | action | 2 | `AB_PAIRS` **逐条 reverse** |
| `action_trib` | action | 3 | `ABC_PAIRS` **逐条 reverse** |

- **新键 `RULES` 三场景同一方向（canonical）：第一个系数乘最新的值。**
- **action 家族旧 JSON 的 `AB_PAIRS`/`ABC_PAIRS` 是相反约定（第一个系数乘最老的值），仅作兼容输入**，归一化层负责反转。全仓库的语义反转只有两处：`scenarios._legacy_rules`（旧键 → canonical）与 `experiment._prepare_action` 入口（canonical → action 内部约定）；另有一处反转仅用于新旧键一致性比较（`scenarios._resolve_conflicts`），不产生新语义。不要再增加新的反转点。
- 新旧键同时出现：一致 → warning + 新键优先；不一致 → 报错。
- 显式启用已删除功能（如 `PREDICT_MISSING: false`、`USE_AB_TAG: true`、`USE_AMP: true`、`COND_FIX*`）→ 启动时报错并指出替代写法（完整清单见 §2.3）。

---

## 2. 配置

以下设置来自 `src/config.json` 的 `main` 字段，除非被场景默认段或实验 JSON 覆盖。
**默认值一律以 `src/config.json` 为准**（下表只列含义，不再复制数值，避免双源漂移）。

合并顺序（`batch_run.py::build_merged_config`）：`main` → 场景默认段 →（旧 task 同名段）→ 实验 config 覆盖；默认段查找顺序为**场景段 → 旧 task 段**（两者同名时去重，如 `action`）。

### 2.1 保留的配置键（新体系的全部可配面）

**任务/规则**

| 配置项 | 说明 |
|--------|------|
| `TASK` | 场景名（`single`/`mixed`/`action`）或三个非线性例外 task 名；旧 task 名经归一化层兼容（见 §1.6），新实验一律用场景名 |
| `P` | 模数 |
| `ORDER` | 递推阶数，≥2 整数；新格式必须显式写 |
| `RULES` | 系数列表，canonical 方向（第一个系数乘最新的值）。`single` 恰好 1 条；`mixed`/`action` ≥1 条；每条长度 == `ORDER`，系数 ∈ [0, P) |
| `NUM_MASK` | `null` 表示场景默认（single/mixed = `order-1`；action 由数据集生成 loss_mask）。**注意：`0` 是字面值**（首位置也计入损失与评估）；旧代码（≤2026-07-08）把 `0` 当"未设置"。**今后配置不要再写 `NUM_MASK: 0`**——想排除"只看 x0 预测 x1"这个不可预测的首位置时，应显式写 `1` |

**数据**

| 配置项 | 说明 |
|--------|------|
| `TRAIN_LEN` | 训练序列长度 |
| `OOD_LEN` | OOD 测试序列长度。**默认 2026-10-08 起从 32 改为 16**（action 场景默认段仍为 32） |
| `EXPOSURE_RATIOS` | 暴露率统一键（取代 `MAX_UNIQUE_RATIO` / `MIXED_AB_MAX_UNIQUE_RATIOS`）。标量 = 所有规则同一比例（single 用标量）；列表 = 逐规则比例，长度必须等于规则数。`NUM_TRAIN_SAMPLES` 显式设置时覆盖它。action 不使用（规则逐步随机选择，无曝光比例概念；设置了会警告忽略） |
| `NUM_TRAIN_SAMPLES` | 显式训练样本数（默认 `null` → 按 `EXPOSURE_RATIOS × 状态空间` 计算）；single 为标量；mixed 支持标量（每规则同样数量）或列表（逐规则数量，长度必须 = 规则数）；action 一直用此键，默认 10000。`sampled_fresh_test` 模式下必须显式设置 |
| `NUM_TEST_SAMPLES` | 测试集大小（single/mixed 新鲜测试默认 256，action 默认 2000） |
| `STATE_SPACE_CAP` | 状态空间上限（默认 `null` 不限）。旧语义：`P^init_len` 超过它时数据集均匀无放回抽样这么多个初始状态。`DATA_MODE` 缺省时，设置它或 `FRESH_TEST_PER_EVAL` 会映射到 `sampled_fresh_test` 模式。高阶递推（如 order=4 @ P=127）必须设置 |
| `DATA_MODE` | single 场景数据生成策略：`full_split`（遍历全状态空间后切分 train/test）或 `sampled_fresh_test`（无放回抽固定训练集，每 eval 有放回抽新鲜测试）。`null` 时按旧键映射（见 `STATE_SPACE_CAP`） |
| `FRESH_TEST_PER_EVAL` | `true` 时每个 eval 用新鲜随机种子重建测试集（single：从全状态空间抽 `NUM_TEST_SAMPLES` 个初始状态生成 `OOD_LEN` 窗口；mixed：每规则各抽 `NUM_TEST_SAMPLES` 个；action：同长度重生成）。配 `EVAL_INTERVAL=1` 即"每 epoch 测 256 个随机案例"；日志打 `[FreshTest]` 行 |
| `RESHUFFLE_EACH_EPOCH` | 默认 false 保留历史固定 batch/顺序。true 时训练 sampler 每次 epoch 迭代都会在各长度桶内重洗所有样本，再重新分 batch 并打乱 batch 顺序；同长度 batch、每条样本恰好一次、不丢尾 batch 均保持。测试 sampler 不受影响。使用 checkpoint 已保存的 Python RNG，使 epoch 边界续跑可复现。仅改配置不改变训练样本集合（2026-09-22） |

**污损**

| 配置项 | 说明 |
|--------|------|
| `MISSING_PROB` | >0 时启用缺失值污损（在线模式：数据集只存干净窗口，collate 逐 batch 新鲜随机污损，每 epoch 缺失位置都重摇，train/test 同样）。**predict 是唯一口径**：污损位以原值为目标并计入损失与正确率。三场景均支持 |
| `MISS_LEN` | 污损段最大长度：命中后污损**随机长度 ∈ [1, MISS_LEN]** 的连续段，段后第一个位置强制干净；若整个窗口没有任何一段达到 MISS_LEN 则重摇直至达到；段尾可在窗口末尾截断 |

**模型**

| 配置项 | 说明 |
|--------|------|
| `D_MODEL` / `N_HEAD` / `N_LAYER` / `MLP_RATIO` / `DROPOUT` | 模型隐藏维度 / 注意力头数 / Transformer block 层数 / MLP 隐藏层相对 d_model 的倍数 / Dropout 概率 |

**优化**

| 配置项 | 说明 |
|--------|------|
| `LR` / `WEIGHT_DECAY` | AdamW 学习率 / weight decay |
| `BATCH_SIZE` | 每个 batch 的序列数（默认 512） |
| `GRAD_CLIP_NORM` | 梯度裁剪范数（默认 1.0） |
| `GRAD_ACCUM_STEPS` | 梯度累积步数（默认 1；显存受限时配合减小 `BATCH_SIZE` 使用） |
| `EPOCHS` | 最大训练 epoch 数（同时是 cosine 调度的 `T_max`） |
| `EPOCH_CAP` | 可选硬停 epoch 上限（先于 `EPOCHS` 生效；LR 调度保持完整 `T_max`） |

**训练控制**

| 配置项 | 说明 |
|--------|------|
| `EVAL_INTERVAL` | 每隔多少 epoch 评估一次 |
| `EARLY_STOP_NO_IMPROVE` | 测试准确率多久未提升则早停 |
| `EARLY_STOP_ACCURACY` | 早停阈值：静态切分制（无 fresh test）测试 ≥ 该值直接停；新鲜测试制（`FRESH_TEST_PER_EVAL`）最近 10 个 eval 均值 > 该值才停 |
| `EARLY_STOP_EXTRA_EPOCHS` | **默认 200 → 0（2026-10-08）**。键保留以兼容旧配置，但当前无作用（`high_acc_epoch` 仅为 resume checkpoint 兼容保留，不再影响停止） |
| `MAX_TRAIN_HOURS` | 墙钟时间上限（小时），`null` 为不限。在每个 eval 点检查，超时则把完整训练状态（模型+优化器+调度器+RNG）存到 `{SAVE_PATH去后缀}_resume.pth` 并以退出码 42 退出（batch_run 记为 timed out，不写最终模型） |

**初始化/续跑**

| 配置项 | 说明 |
|--------|------|
| `RANDOM_SEED` | 随机种子 |
| `INIT_FROM` | 课程/迁移学习：从指定 checkpoint 初始化权重（仅拷贝名字与形状都匹配的张量，其余保持随机初始化，日志打印加载/跳过报告）；`null` 为不启用。优化器与调度器始终全新开始 |
| `RESUME_FROM` | 续跑：从 `MAX_TRAIN_HOURS` 超时保存的 resume checkpoint 恢复完整训练状态（模型/优化器/调度器/RNG/当前 epoch/best 等），`null` 为不启用。与 `INIT_FROM` 同时设置时 `RESUME_FROM` 后生效。存档会覆盖配置里的 LR；若要在保留 Adam 状态与 cosine 进度的同时降低 LR，先用 `scripts/fork_training_lr.py <source> <new-checkpoint> --scale 0.2` 生成独立完整存档，再以它为 `RESUME_FROM`（同时把配置 LR 写成对应值以便记录）。该脚本仅缩放 optimizer/scheduler 的 LR 字段，保留 epoch、T_max、权重、moments 和 RNG，不覆盖源存档 |

**损失**

| 配置项 | 说明 |
|--------|------|
| `FIRST_TASK_WEIGHT` | 第一个预测位置的损失权重 |
| `AUX_CLASSIFY` | **仅 mixed**（默认 false）。true 时模型加规则分类辅助头 rule_head（`Linear(d_model, d_model//2) → ReLU → Linear(d_model//2, n_rules)`，读 `x_{order+1}` 起的位置），辅助交叉熵以权重 0.5 加进训练 loss；其他场景写 `true` 会在归一化层报错。checkpoint 的 save_config 记录 `aux_classify` |

**诊断**

| 配置项 | 说明 |
|--------|------|
| `OPT_DIAG_INTERVAL` | 优化器诊断（裁剪前梯度范数等）打印间隔，0 = 关闭 |

**其他**

| 配置项 | 说明 |
|--------|------|
| `SAVE_PATH` | 模型保存路径；由 `batch_run.py` 自动覆盖为 `{model-base-dir}/{批次名}/{实验名}.pth` |

### 2.2 写死的配置（2026-10-08 起代码内固定，不再可配）

| 键 | 写死值 | 后果 |
|---|---|---|
| `PREDICT_MISSING` | true | 污损位永远以原值为目标并计入损失与正确率；mask 口径（不计入）连同对应 collate 分支移除 |
| `USE_LEARNABLE_PE` | false | 全部用 RoPE；`wpe` 代码路径移除 |
| `ALLOW_TF32` | true | 常开（仅 CUDA 生效） |
| `USE_AMP` | false | AMP/autocast/GradScaler 代码移除，训练恒为 fp32（依据 `docs/AMP_PRECISION_NOTES.md`） |
| `ENTROPY_PENALTY_WEIGHT` | 0 | 注意力熵惩罚从损失中移除 |

### 2.3 已于 2026-10-08 删除的键

**被新键取代、归一化层仍作兼容输入翻译**（出现在旧 task JSON 中可继续工作；与新键并存时一致 → warning + 新键优先，不一致 → 报错）：

| 旧键 | 取代者 |
|---|---|
| `A` / `B` / `C` / `D` | `RULES`（single 场景一条规则） |
| `AB_PAIRS` / `ABC_PAIRS` | `RULES`（mixed 本就是 canonical；action 家族是反向约定，归一化层反转，见 §1.6） |
| `MAX_UNIQUE_RATIO` / `MIXED_AB_MAX_UNIQUE_RATIOS` | `EXPOSURE_RATIOS` |

**功能整体删除**（归一化层对显式启用者**报错**并指出替代写法；值与写死值一致（如 `USE_AMP: false`、`ALLOW_TF32: true`）则**警告后忽略**）：

| 键 | 删除内容 | 替代写法 |
|---|---|---|
| `PREDICT_MISSING` | mask 污损口径 | 恒 predict，去掉该键 |
| `MISS_SECOND` | 固定第二段污损 | `MISSING_PROB` + `MISS_LEN` |
| `USE_LEARNABLE_PE` | 可学习位置编码（wpe） | 恒 RoPE |
| `USE_AB_TAG` | mixed label 模式（rule token / 词表扩展） | mixed 只剩 basic 模式 |
| `USE_CONDITIONAL_WTE` / `COND_WTE_SHARED_RATIO` | cond_wte 条件嵌入 | — |
| `COND_FIX` / `COND_FIX_START` / `COND_FIX_START_A1` / `COND_FIX_START_A2` | 部分参数冻结（freeze_partial） | — |
| `USE_AMP` / `AMP_DTYPE` | 混合精度训练 | 恒 fp32 |
| `ALLOW_TF32` | TF32 开关 | 恒开 |
| `ENTROPY_PENALTY_WEIGHT` | 注意力熵惩罚 | 恒 0 |
| `SKIP_FINAL_GENERATION_TEST` | final generation test 整体消失 | 直接忽略（无需替代） |

连带删除的实现：`MixedABTransformer` 类（rule_head 后以可选形式恢复为 `FibonacciTransformer` 的 `n_rules` 参数，由 `AUX_CLASSIFY` 开关控制；tag/cond_wte 变体不恢复）、`src/final_eval.py`、`training.py::freeze_partial`、AMP/autocast/GradScaler 路径。

### 2.4 `src/config.json` 的场景默认段

- `single` / `mixed`：空段（无覆盖）。
- `action`：`P: 53`，`NUM_TRAIN_SAMPLES: 10000`，`NUM_TEST_SAMPLES: 2000`，`TRAIN_LEN: 16`，`OOD_LEN: 32`。
- 旧 task 段保留供旧名兼容：`tribonacci`（`P:23, A:1, B:2, C:3`）、`tetranacci`（`P:23, A:B:C:D=1`）、`mixed_ab` / `mixed_abc`（`MIXED_AB_MAX_UNIQUE_RATIOS: [0.7, 0.7]`）、`multiplication` / `nonlinear` / `nonlinear_mul` / `action_trib`（空段）。

### 2.5 训练流程通用设置

- **运行入口**：`src/batch_run.py` 是唯一的运行入口，已删除 `src/main.py`。
- **优化器**：`torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)`
- **学习率调度**：`CosineAnnealingLR(optimizer, T_max=EPOCHS)`
- **梯度裁剪**：`clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)`（默认 1.0）
- **滚动 checkpoint**：每个 eval 点更新 `<model>_latest.pth`（完整训练状态，可直接 `RESUME_FROM`）；best 更新时写 `<model>_best.pth`（按 test accuracy 选择的纯权重，与最终保存同格式）。latest、resume、best 和最终权重均先写同目录临时文件，成功后原子替换；写入失败最多尝试 3 次，保留上一次完整存档（2026-09-21）。
- **checkpoint save_config**：2026-10-08 起新 checkpoint 在原有字段外追加 `scenario` / `order` / `rules`（canonical）三键；仍能无歧义表达的旧字段（`ab_pairs`、小写 `a/b/c/d`、`recurrence` 名）照写，方便旧分析工具读取新 checkpoint。`rules.py::task_from_save_config` 优先读新块，没有则走旧反推逻辑（旧 checkpoint 全兼容）。
- **损失函数**：每个时间步的交叉熵，按 `loss_mask` 平均（熵惩罚已于 2026-10-08 移除；mixed 的 rule loss 仅在 `AUX_CLASSIFY: true` 时以权重 0.5 加回）。
- **mask 策略**：
  - `single` / 例外 task：默认屏蔽前 `order-1` 个位置（例外 task 固定 1）
  - `mixed`：默认屏蔽前 `order-1` 个位置
  - `action`：由数据集生成 loss_mask，只计算 `x_o, x_{o+1}, ...`
- **block_size 计算**：`max(TRAIN_LEN, OOD_LEN)` 向上取整到最近的 2 的幂；`action` 按 `2*L - order` 计算。

---

## 3. 模型结构

### 3.1 总体架构

**唯一模型类是 `FibonacciTransformer`**（`MixedABTransformer` 已于 2026-10-08 删除）。所有任务共享一个 **Pre-LayerNorm** 的 decoder-only Transformer 架构。每个 TransformerBlock 内部都是 **先 LayerNorm，再 Attention/MLP，最后加残差**；最终输出头 `lm_head` 之前还有一次 LayerNorm（`ln_f`），属于 Pre-LN 架构的标准收尾，而非 Post-LN。

```
Input token ids (B, T)
        │
        ▼
┌─────────────────────────────────────┐
│   Embedding                         │  wte + RoPE（位置编码恒为 RoPE）
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
                           │   LM Head       │  lm_head 线性分类（与 wte 权重共享）
                           └────────┬────────┘
                                    │ (B, T, VOCAB_SIZE)
                                    ▼
                             logits + loss
```

### 3.2 FibonacciTransformer

类：`src/models.py::FibonacciTransformer`

| 组件 | 定义 | 说明 |
|------|------|------|
| `transformer.wte` | `nn.Embedding(vocab_size, d_model)` | 词嵌入，vocab_size = P + 1（含 PAD token）；action 场景运行时扩展（见 §3.3） |
| `rope` | `RotaryEmbedding(d_model // n_head)` | 旋转位置编码，恒用（可学习 `wpe` 路径已于 2026-10-08 删除） |
| `transformer.h` | `ModuleList[TransformerBlock]` | N_LAYER 个 Transformer block |
| `transformer.ln_f` | `nn.LayerNorm(d_model)` | 最终层归一化 |
| `lm_head` | `nn.Linear(d_model, vocab_size, bias=False)` | 输出分类头，与 `wte` 共享权重 |
| `rule_head`（可选） | `Linear(d_model, d_model//2) → ReLU → Linear(d_model//2, n_rules)` | 仅 `AUX_CLASSIFY: true` 的 mixed 实验创建（构造参数 `n_rules>0` + `order`）；读 `x_{order+1}` 起的隐状态做规则分类，辅助损失权重 0.5 |

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
q, k = apply_rotary_emb(q, k)   # RoPE 恒用
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

### 3.3 action 的词表扩展

`action` 场景使用同一个 `FibonacciTransformer`，但在 `run_experiment` 中扩展词表以容纳 flag token：

```python
vocab_size = P + 1 + len(rules)
pad_token_id = P
```

扩展后的 `transformer.wte` 与 `lm_head` 保持权重共享。

### 3.4 参数初始化

- 所有 `nn.Linear` 和 `nn.Embedding`：均值 0、标准差 0.02 的正态分布
- `nn.Linear` 的 bias：初始化为 0

---

## 4. 数据集生成

### 4.1 单任务数据集（RecurrenceDataset）

类：`src/datasets.py::RecurrenceDataset`，两种数据模式（`DATA_MODE`）：

- **`full_split`**（默认）：枚举所有 `P^init_len` 个初始状态（若设了 `STATE_SPACE_CAP` 且 `P^init_len` 超过它，则改为均匀无放回抽样 `STATE_SPACE_CAP` 个状态，每状态直接逐步递推得到一个窗口，跳过循环遍历）；随机打乱后遍历每个初始状态生成完整递推循环，对循环做滑动窗口得到长度为 `TRAIN_LEN` 的序列；按 `EXPOSURE_RATIOS`（或显式 `NUM_TRAIN_SAMPLES`）切分训练/测试集。
- **`sampled_fresh_test`**：无放回抽样固定训练集（必须显式 `NUM_TRAIN_SAMPLES`），不物化状态空间；每个 eval 从全状态空间**有放回**抽 `NUM_TEST_SAMPLES` 个初始状态生成 `OOD_LEN` 窗口作为新鲜测试。

`DATA_MODE` 缺省时按旧键映射：`FRESH_TEST_PER_EVAL` 或 `STATE_SPACE_CAP` 任一设置 → `sampled_fresh_test`，否则 → `full_split`。

### 4.2 多规则数据集（MixedRecurrenceDataset）

类：`src/datasets.py::MixedRecurrenceDataset`（旧实现 `MixedABDataset` 保留在 `tests/test_mixed_ab_compat.py` 作为 golden master 对照）

1. 对每条规则分别调用 `RecurrenceDataset` 生成训练/测试样本。
2. 每条规则的样本数由 `EXPOSURE_RATIOS` 控制；显式 `NUM_TRAIN_SAMPLES`（标量或逐规则列表）设置后覆盖。
3. 合并所有规则样本后打乱训练顺序。
4. 每条样本带逐样本规则标签（`ab_indices`），仅供 per-rule accuracy 统计；**序列本身不含 rule token**（tag 模式与 cond_wte 已于 2026-10-08 删除，词表恒 `P+1`）。

### 4.3 动态混合数据集（ActionDataset）

类：`src/datasets.py::ActionDataset`

1. 使用独立的 `random.Random(seed)` 确定性生成样本。
2. 每步随机选择一条规则，生成 `x_k` 并在前面插入 `flag_k`（`order` 参数支持任意阶数）。
3. 返回序列和 1D loss_mask，collate 后扩展为 2D。
4. 训练/测试通过 `NUM_TRAIN_SAMPLES` / `NUM_TEST_SAMPLES` 控制，使用不同 seed 区分。

---

## 5. 损失函数与评估

### 5.1 训练损失

```python
loss = masked_cross_entropy(logits, targets, loss_mask)
```

- `targets = x[:, 1:]`，即输入序列右移一位
- `loss_mask` 默认屏蔽前 `num_mask` 个位置，只计算后续生成位置的损失
- 注意力熵惩罚已于 2026-10-08 移除；mixed 的规则分类辅助损失（rule loss，权重 0.5）仅在 `AUX_CLASSIFY: true` 时启用
- NaN 检测：`train_epoch` 中检测到 NaN loss 时，会将诊断信息打印到 stderr（进入 `.err` 日志），该 batch 不计入 epoch 平均 loss。

### 5.2 batch 类型区分

`train_epoch` 与 `evaluate` 使用显式的 `BatchTag`（IntEnum，`src/datasets.py`）区分不同 collate 模式：

| Tag | 来源 collate | 额外张量 |
|-----|------|----------|
| `BatchTag.PLAIN` | `collate_fn` | 无 |
| `BatchTag.LOSS_MASK` | `action_collate_fn` | `loss_mask` |
| `BatchTag.PLAIN_TARGET` | `make_missing_collate` | 干净目标序列（predict 污损口径） |
| `BatchTag.MIXED_AB` | `mixed_ab_collate_fn` | `ab_indices`（逐样本规则标签） |
| `BatchTag.MIXED_AB_TARGET` | `make_mixed_missing_collate` | `ab_indices` + 干净目标序列 |
| `BatchTag.ACTION_MISS` | `make_action_missing_collate` | 干净目标序列 + `loss_mask` |

不再通过张量 `ndim` 推断 batch 类型。

### 5.3 评估指标

训练过程中每个评估 epoch 输出：

- 训练 loss 与 overall accuracy，以及 `TrainPos` 逐位置训练准确率（`target[i]` 为 `x[:,1:]` 中的零基索引；order-4 序列的 `target[3]` 对应首个预测项 x4）
- 测试 overall accuracy
- 每个预测位置（per-position）的准确率
- `mixed` 场景额外输出每条规则的 per-rule accuracy
- 保存模型前打印 `Training epochs: {epoch}`，供 `batch_run.py` 汇总表提取

**训练结束后的 final generation test（Exposed/Unexposed × ID/OOD 表格）已于 2026-10-08 整体删除**（`src/final_eval.py` 移除）。结果口径 = 训练日志的 best test acc + `scripts/eval_clean_action.py`（clean acc 评估，已 order 参数化）。

---

## 6. 场景对比

| 项目 | single | mixed | action | 例外 task |
|------|--------|-------|--------|-----------|
| 递推公式 | `RULES` 一条，canonical | `RULES` 多条，每条样本固定一条，canonical | `RULES` 多条，每步随机一条，canonical | 硬编码非线性 |
| 默认 `P` | 53（旧别名 trib/tetra 为 23） | 53 | 53 | 53 |
| `init_len` | `ORDER` | `ORDER` | `ORDER` | 2 |
| 状态空间 | `P^order` | 每规则 `P^order` | 不枚举 | `P^2` |
| `num_mask` 默认 | `order-1` | `order-1` | 来自数据集 loss_mask | 1 |
| 模型 | `FibonacciTransformer`（唯一模型类） | 同左 | 同左 | 同左 |
| 词表 | `P+1` | 恒 `P+1` | `P+1+N规则`（flag tokens） | `P+1` |
| final generation test | 无（2026-10-08 起全场景删除） | 无 | 无 | 无 |

---

## 7. 源码关键位置

| 功能 | 文件与位置 |
|------|------------|
| 归一化层（旧名映射 / action 反转 / 已删功能报错） | `src/scenarios.py::normalize_task_config`（纯配置翻译，无 torch 依赖） |
| 递推规则 | `src/rules.py::LinearRecurrenceRule`（任意阶任意系数，canonical），`src/rules.py::single_rule_from_task`（single/例外 task → (init_len, next_fn, name)），`src/rules.py::rules_from_config(rules_raw, p)`（消费归一化结果建规则列表），`src/rules.py::save_config_extra` / `task_from_save_config`（checkpoint save_config 读写） |
| 数据生成 | `src/datasets.py::RecurrenceDataset`，`src/datasets.py::MixedRecurrenceDataset`，`src/datasets.py::ActionDataset`，collate 与 `BatchTag` 同文件 |
| 模型定义 | `src/models.py::FibonacciTransformer`（唯一模型类；`MixedABTransformer` 已删） |
| 注意力层 | `src/models.py::CausalSelfAttention` |
| RoPE 实现 | `src/models.py::RotaryEmbedding`，`src/models.py::apply_rotary_emb` |
| 训练流程 | `src/training.py::run_training_engine`，`src/training.py::train_epoch`，`src/training.py::evaluate`（AMP、freeze_partial、entropy penalty 均已移除） |
| 配置入口 | `src/experiment.py::run_experiment`（开头经 `normalize_task_config` 归一化后按 scenario 分派 `_prepare_single_recurrence` / `_prepare_mixed_recurrence` / `_prepare_action`；`src/core.py` 为入口壳并 re-export，由 `src/batch_run.py` 以 `python src/core.py <config>` 子进程方式调用） |
| 批量运行 | `src/batch_run.py` |
| 注意力可视化 | `src/analyze_attention.py`（action/mixed 已按 order 参数化；含 cond_wte/ab_emb 键的旧 checkpoint 明确报错，含 rule_head 键的可加载） |
| 训练后最终生成测试 | **已删除**（`src/final_eval.py` 移除，2026-10-08） |
| 圆结构验证（傅里叶/单位圆） | `src/verify_circle.py`（独立脚本，不在默认流程中运行） |
| 可视化 | `src/visualize.py` |

---

## 8. 备注

- 位置编码恒为 **RoPE**（`USE_LEARNABLE_PE` 写死 false，`wpe` 代码路径已删除）。
- `experiments.json` 中定义的实验会覆盖 `config.json` 的对应字段，`batch_run.py` 负责配置合并与批量执行；实验路由的唯一来源是每个实验条目的顶层 `task` 字段（新场景名与旧 task 名均可）。
- `src/main.py` 已删除，所有运行必须通过 `src/batch_run.py`。
- `.gitignore` 已忽略训练产物：`*.log`、`*.err`、`*.pth`、`src/nohup.out`、`config_tmp_*.json` 等。
- 模型默认保存路径：`/mnt/workspace/hujiachen/models/{批次名}/{实验名}.pth`（由 `batch_run.py` 自动设置，可在 `experiments.json` 中通过 `SAVE_PATH` 覆盖）。
- 日志与绘图输出路径：`/mnt/workspace/hujiachen/recursion_results/{批次名}/logs/` 与 `/mnt/workspace/hujiachen/recursion_results/{批次名}/plots/`（批次名 = 实验 JSON 文件名去扩展名）。
- 可通过 `--base-dir` 和 `--model-base-dir` 参数修改这两个根目录，默认分别为 `/mnt/workspace/hujiachen/recursion_results` 和 `/mnt/workspace/hujiachen/models`。

跨机定时运行包：每个常规 eval 保留 latest 完整状态；定时、手动停止、正常完成和早停均保存 resume 完整状态。保存先写临时文件，再原子替换，避免写失败覆盖旧存档。
