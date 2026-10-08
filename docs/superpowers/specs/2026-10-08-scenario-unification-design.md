# 训练场景统一设计：single / mixed / action × order × miss_len

> 日期：2026-10-08
> 状态：方案与参数去留已与用户确认（方案 A + 参数决策表）
> 参数决策明细：`docs/superpowers/specs/2026-10-08-config-params-review.md`

## 0. 目标与范围

把当前散落在十余个 task 名里的训练场景收敛为**三个统一场景**，每个场景由两个配置维度参数化，
同时大幅精简配置面与代码路径：

| 场景 | task 值 | order 维度 | miss_len 维度 |
|---|---|---|---|
| 单任务线性递推 | `single` | `ORDER` ≥ 2 | `MISSING_PROB` + `MISS_LEN`（0/不写 = 干净） |
| 多规则混合 | `mixed` | `ORDER` ≥ 2 | 同上 |
| 每步规则可变 | `action` | `ORDER` ≥ 2 | 同上 |

**范围内**：配置体系、归一化层、experiment.py 路由、被删除功能的代码移除、
checkpoint save_config 扩展、分析/评估工具的 order 参数化、测试与文档。

**明确不做**（YAGNI）：

- 数据集类不合并（ActionDataset 的 flag 交错布局与单任务差异本质大，RNG 调用序被 compat 测试钉死）
- `multiplication` / `nonlinear` / `nonlinear_mul` 保留为例外 task，不进统一矩阵
- `experiments/` 下约 3000 条历史 JSON 不做任何改动

## 1. 配置体系（参数去留已确认）

### 1.1 保留的配置键（新体系的全部可配面）

| 类别 | 键 |
|---|---|
| 任务/规则 | `TASK`（single/mixed/action/三个非线性例外）、`P`、`ORDER`、`RULES`、`NUM_MASK`（默认自动推导） |
| 数据 | `TRAIN_LEN`、`OOD_LEN`（**默认改为 16**）、`EXPOSURE_RATIOS`、`NUM_TRAIN_SAMPLES`、`NUM_TEST_SAMPLES`、`STATE_SPACE_CAP`、`DATA_MODE`、`FRESH_TEST_PER_EVAL`、`RESHUFFLE_EACH_EPOCH` |
| 污损 | `MISSING_PROB`、`MISS_LEN` |
| 模型 | `D_MODEL`、`N_LAYER`、`N_HEAD`、`MLP_RATIO`、`DROPOUT` |
| 优化 | `LR`、`WEIGHT_DECAY`、`BATCH_SIZE`、`GRAD_CLIP_NORM`、`GRAD_ACCUM_STEPS`、`EPOCHS`、`EPOCH_CAP` |
| 训练控制 | `EVAL_INTERVAL`、`EARLY_STOP_ACCURACY`、`EARLY_STOP_NO_IMPROVE`、`EARLY_STOP_EXTRA_EPOCHS`（保留，**默认 0**）、`MAX_TRAIN_HOURS` |
| 初始化/续跑 | `RANDOM_SEED`、`INIT_FROM`、`RESUME_FROM` |
| 损失 | `FIRST_TASK_WEIGHT` |
| 诊断 | `OPT_DIAG_INTERVAL` |

### 1.2 新键语义

- **`ORDER`**：递推阶数，任意 ≥2 整数。新格式必须显式写。
- **`RULES`**：系数列表。`single` 恰好 1 条；`mixed`/`action` ≥1 条。
  **统一约定：第一个系数乘最新的值**（`x_k = c1·x_{k-1} + c2·x_{k-2} + ...`，与 `LinearRecurrenceRule` 一致）。
  ⚠ action 场景历史 JSON 的 `AB_PAIRS`/`ABC_PAIRS` 是相反约定（第一个乘最老），仅作兼容输入；
  新键 RULES 三个场景同一方向。
- **`EXPOSURE_RATIOS`**：暴露率统一键（取代 `MAX_UNIQUE_RATIO` 和 `MIXED_AB_MAX_UNIQUE_RATIOS`）。
  标量 = 所有规则同一比例（single 场景用标量）；列表 = 逐规则比例，长度必须等于规则数。
  与 `NUM_TRAIN_SAMPLES` 的标量/列表规则对齐；`NUM_TRAIN_SAMPLES` 显式设置时覆盖它。

### 1.3 写死的配置（代码内固定，不再可配）

| 键 | 写死值 | 后果 |
|---|---|---|
| `PREDICT_MISSING` | true | 污损位永远以原值为目标并计入损失；mask 模式（不计入）连同对应 collate 分支移除 |
| `USE_LEARNABLE_PE` | false | 全部用 RoPE；`wpe` 代码路径移除 |
| `ALLOW_TF32` | true | 常开 |
| `USE_AMP` | false | AMP/autocast/GradScaler 代码移除（bf16 已被实证判决，见 AMP_PRECISION_NOTES） |
| `ENTROPY_PENALTY_WEIGHT` | 0 | 注意力熵惩罚从损失中移除 |

### 1.4 删除的功能（连代码逻辑一起移除）

| 功能 | 影响面 |
|---|---|
| `A/B/C/D`、`AB_PAIRS`、`ABC_PAIRS` 键 | 被 `RULES` 取代；归一化层仍读旧键做兼容翻译 |
| `MAX_UNIQUE_RATIO`、`MIXED_AB_MAX_UNIQUE_RATIOS` 键 | 被 `EXPOSURE_RATIOS` 取代；归一化层兼容翻译 |
| `MISS_SECOND` | 从 datasets/collate 移除 |
| `SKIP_FINAL_GENERATION_TEST` 与**整个 final generation test** | final_eval.py 移除；训练结束不再跑 Exposed/Unexposed 生成评估（结果口径由训练日志 best acc + `eval_clean_action.py` 承担） |
| `COND_FIX` / `COND_FIX_START` / `COND_FIX_START_A1` / `COND_FIX_START_A2` | training.py 的 freeze_partial 及相关逻辑移除 |
| `AMP_DTYPE` | 随 USE_AMP 移除 |
| `USE_AB_TAG`（label 模式） | mixed 只剩 basic 模式；rule token、词表扩展、`restricted_token_ids` 移除 |
| `USE_CONDITIONAL_WTE` / `COND_WTE_SHARED_RATIO`（cond_wte 模式） | 条件嵌入移除 |
| **rule_head** | mixed 场景直接使用 `FibonacciTransformer`；**`MixedABTransformer` 类整体移除**（含 rule_loss） |

### 1.5 兼容层对删除功能的策略：明确报错

历史旧 JSON / 旧 checkpoint 若用到已删除或写死的功能
（`PREDICT_MISSING: false`、`MISS_SECOND: true`、`USE_LEARNABLE_PE: true`、`USE_AB_TAG: true`、
`USE_CONDITIONAL_WTE: true`、`COND_FIX*`、`USE_AMP: true` 等），
启动时**明确报错并说明原因**，不静默改变实验语义。用户此时应改写为新格式或放弃该配置。

### 1.6 NUM_MASK 默认值

- single = `order-1`；mixed = `order-1`（tag 模式已删，无需 tag 自适应）；action 不变（数据集生成 loss_mask）。

### 1.7 实验命名约定

`{scenario}_o{order}_...`，例如 `single_o3_d1024l2_P127_tr64_miss0.1len1_seed17996`。
seed 后缀（`_seed<N>`）规则不变（课程链 `@prev` 配对依赖它）。

## 2. 内部实现

### 2.1 新增 `src/scenarios.py`（无 torch 依赖，纯配置翻译）

```python
def normalize_task_config(task: str, cfg: dict) -> dict
    # 返回 {'scenario': 'single'|'mixed'|'action'|<例外task名>,
    #       'order': int, 'rules': [[c1, ...], ...],   # canonical 方向
    #       'exposure_ratios': [...], 'source': 'new' | 旧task名}
```

旧格式映射表（显式字典，每种旧 task 一行）：

| 旧 task | scenario | order | rules 来源 |
|---|---|---|---|
| `addition` | single | 2 | `[A, B]`（canonical） |
| `tribonacci` | single | 3 | `[A, B, C]` |
| `tetranacci` | single | 4 | `[A, B, C, D]` |
| `mixed_ab` | mixed | 2 | `AB_PAIRS`（本就是 canonical，直接用） |
| `mixed_abc` | mixed | 3 | `ABC_PAIRS`（同上） |
| `action` / `dynamic_mixed` | action | 2 | `AB_PAIRS` **逐条 reverse** |
| `action_trib` | action | 3 | `ABC_PAIRS` **逐条 reverse** |
| `multiplication` / `nonlinear` / `nonlinear_mul` | 原样透传 | — | — |

- 归一化层同时负责：旧暴露率键 → `EXPOSURE_RATIOS`；§1.5 的删除功能报错检查。
- `rules_from_config` 的 `{2:'AB_PAIRS',3:'ABC_PAIRS'}` 写死映射废弃，改为消费归一化结果（任意 order）。

### 2.2 experiment.py 路由收敛

`run_experiment` 开头调 `normalize_task_config`，之后按 `scenario` 分派：

- `_prepare_single_recurrence`：线性单任务改走 `LinearRecurrenceRule`；非线性特例维持现有硬编码分支。
  `single_rule_from_task` 保留给 rule_fit/verify_sample 等旧调用方。
- `_prepare_mixed_recurrence(config, device, order)`：模型改为 `FibonacciTransformer`（无 tag/cond_wte/rule_head）；
  NUM_MASK 默认 `order-1`；词表恒为 `P+1`。
- `_prepare_action(config, order)`：内部维持现有 action 约定不动，
  仅在入口处把 canonical rules 逐条 `reversed()` 喂入（一处反转 + 注释钉死原因）。
- stage-3 final generation test 调用整体删除。

### 2.3 模型层（models.py）

- 删除 `MixedABTransformer`、`rule_head`、`cond_wte`、`ab_emb`、`USE_AB_TAG` 相关分支、`RULE_LOSS_WEIGHT`。
- 删除 `wpe` 可学习位置编码路径（RoPE 恒用）。
- `FibonacciTransformer` 成为唯一模型类。

### 2.4 训练层（training.py）

- 移除：AMP/autocast/GradScaler、freeze_partial 及 COND_FIX 系列、entropy penalty。
- 污损 collate 只保留 predict 口径：`_unpack_batch` 的 mask 模式分支（LOSS_MASK 用于污损的场景）收缩；
  BatchTag 枚举相应精简（具体保留哪些 tag 由实现计划定）。

### 2.5 checkpoint save_config

新 checkpoint 在现有字段之外追加 `scenario` / `order` / `rules`（canonical）三键；
仍能无歧义表达的旧字段（`ab_pairs` 等）照写，方便旧分析工具读取新 checkpoint。
`task_from_save_config`：有新块优先读新块；没有则走现有反推逻辑（旧 checkpoint 全兼容）。

### 2.6 batch_run.py

- 顶层 `task` 字段接受新场景名；config.json 默认段查找顺序 = 新场景段 → 旧 task 段。
- `dynamic_mixed`→`action` 别名逻辑挪进归一化层。
- attention 分析跳过逻辑随 analyze_attention 参数化后取消（见 §3）。

## 3. 校验与错误处理（集中在归一化层）

- `RULES` 每条规则长度必须 == `ORDER`，系数 ∈ [0, P)；`single` 恰好 1 条；`mixed`/`action` ≥1 条。
- `EXPOSURE_RATIOS` 为列表时长度必须等于规则数。
- 新格式缺 `ORDER` → 报错。
- 新旧键同时出现且值不一致 → 报错；一致则以新键为准并打 warning。
- 命中已删除功能（§1.5）→ 报错并指出替代写法。
- 未知 task 名 → 报错并列出全部合法新旧名称。

## 4. 配套工具跟进

- `scripts/eval_clean_action.py`：order 从 checkpoint save_config 读（新块优先），去掉 order-2 硬编码
  （misslen 评估 pipeline 的必需环节）。
- `src/analyze_attention.py`：action 分支按 order 参数化（`2*L-order` 布局、query mask 公式）；
  `load_model` 移除 MixedABTransformer 重建分支前需确认旧 checkpoint 加载路径（必要时保留只读兼容）；
  完成后 batch_run.py 取消对 `action_trib` 的跳过。
- `src/rule_fit.py`：维持只支持 single 场景；`task_from_save_config` 新块反解后补一条 `single` 映射。
- `scripts/experiment_pipeline.py` 的 SINGLE/TASKS 表加入新 task 名（交付 pipeline 需要）。
- `src/final_eval.py`：删除。
- 引用被删功能的课程脚本（`curriculum_chain_*.py` 硬编码 `mixed_ab` 段合并）：同步更新到新键。

## 5. 测试

本机用 `C:/Users/Chen/anaconda3/envs/torch/python.exe` 直接跑 runner（无 pytest）。

- **新增 `tests/test_scenarios.py`**：
  - 映射表全条目单测（每个旧 task → 期望 scenario/order/rules，含 action 方向反转）
  - 非法输入报错（规则长度不符、系数越界、single 多规则、缺 ORDER、未知 task、EXPOSURE_RATIOS 长度不符）
  - 删除功能报错（§1.5 每条各一例）
  - 新旧键冲突处理（一致 → warning+新键优先；冲突 → 报错）
- **等价性金标准**：同一实验用旧写法和新写法各跑一次 CPU 冒烟（2 epoch），
  断言同 seed 下数据集样本逐点相同、模型结构相同。覆盖三组对照：
  `single_o3` vs `tribonacci`、`mixed_o2` vs `mixed_ab`（注意：旧 mixed 带 rule_head，模型结构不同——
  等价断言只比数据集与 loss 数值口径，模型结构以新代码为准）、`action_o3` vs `action_trib`
  （专门验证系数反转正确性）。
- 现有测试：凡引用被删功能（USE_AB_TAG=true、cond_wte、rule_head、AMP、COND_FIX、
  final generation test、MISS_SECOND、learnable PE）的用例同步删除或改写；
  其余必须不做改动地通过。

## 6. 已知语义变化（相对旧体系，需在交付时说明）

1. 污损全部 predict 口径：single/mixed 历史默认是 mask 模式（PREDICT_MISSING=false），
   新体系恒为 predict（污损位计入损失与正确率）。新旧实验的 test acc 口径不可直接比较。
2. mixed 模型不再有 rule_head 辅助损失，也无 tag/cond_wte 变体——同配置新跑与历史
   mixed 结果会有架构差异。
3. 训练结束不再有 final generation test（Exposed/Unexposed × ID/OOD 表格不再产出）。
4. OOD_LEN 默认从 32 改为 16。

## 7. 文档同步

- `docs/recursion_model_and_training_settings.md`：§1 改为场景矩阵（三场景 × order × miss_len），
  §2 配置表按 §1.1 全量重写，删除的功能标注移除日期。
- `AGENTS.md` §1/§3 更新场景描述与语义改动清单。
- `.agents/skills/recursion-experiment-runner/SKILL.md` §1 命名约定补新格式。
- `docs/AMP_PRECISION_NOTES.md` 保留作历史教训，但注明 AMP 代码已移除。
- 本文档：`docs/superpowers/specs/2026-10-08-scenario-unification-design.md`。
