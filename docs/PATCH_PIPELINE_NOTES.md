# 模型组件 Patch 管线交接文档

**日期**：2026-08-28
**适用范围**：mixed_ab（无 ab_tag）秩序列模型的可解释性分析。
**前置阅读**：`reports/multi_pos_patch_summary.md`（方法论与全部结论）、
`reports/activation_patch_summary.md`（第一轮 HSS，含已修正的结论）。

## 0. 仓库与数据位置

| 内容 | 位置 |
|---|---|
| 主代码库（本仓库） | `~/recursion`（工作副本）/ GitHub `recursion.git` main |
| 模型 checkpoint | `/data/cxm/models/<批次名>/` |
| 训练日志 | `/data/cxm/recursion/<批次名>/logs/`（也是 `recursion_results.git` 的 checkout） |
| 分析图/报告 | `reports/`（已入 git） |
| 一次性分析脚本（未固化） | `.chain_tmp/`（gitignored，仅供翻查） |

常用模型组：
- `mixed_ab_downscale_l1h4/`：[(1,1),(1,2)]，d256l1r8h2/h4 等，3 seeds；
- `mixed_basic_d256l1r8h2_p127_rules12345/`：[(1,1),(2,3)] N=2..5；
- `mixed_basic_n2_ladder_rules12/`、`mixed_basic_scale_matrix_h4_rules12/`：规模阶梯；
- `curriculum_chains/`：课程链各阶段模型。

## 1. 管线总览（按分析顺序）

```
① load_model（analyze_attention.py）
      ↓
② 注意力解剖：get_attention_weights → 逐头 lag 分布
      ↓ 确定每个头的角色（lag0/lag1=操作数, lag2=证据, 混合=惰性）
③ HSS / activation patching：activation_patch.py
      ↓ 单组件单位置翻转率低——需要④
④ 多位置 + 翻转猎手：multi_pos_patch.py（5 目标分类）
      ↓ 找到证据头后——
⑤ 证据值扫描：evidence_sweep.py（合成 V patch，画判定函数）
⑥ 头消融：head_ablation.py（必要性 vs 被读取）
⑦ 逐样本分析：见 .chain_tmp/per_sample_*.py（曝光集重建等）
```

## 2. 各工具用法

### ② 注意力解剖
```python
from analyze_attention import load_model, get_attention_weights
model, ck = load_model(pth, device='cuda')
w = get_attention_weights(model, seqs)  # list of (n_head,T,T) per layer
```
对每条规则分别生成序列测 pattern；pattern 与规则无关是常态（l1），
l2 的 layer-1 头 pattern 可随规则变化。参考 `.chain_tmp/check_attn_pattern_v2.py`。

### ③ activation_patch.py（HSS 原始探针）
```
python src/activation_patch.py model.pth --pairs 64 --out report.json
```
- 组件：每层每 head 的 c_proj 输入切片 + MLP 隐藏层；双向、全位置扫描。
- **已知的坑（第一轮踩过）**：patch 位置 q 只影响预测位置 k=q+1（l1），
  对全部 k 平均会把 0.4 的翻转稀释成 0.01——必须用④的对齐口径。

### ④ multi_pos_patch.py（翻转猎手，推荐的主力工具）
```
python src/multi_pos_patch.py model.pth --pairs 128 --out report.json
```
- 支持多位置同时 patch（pos 可为列表）、site 类型：
  `('attn',layer,head)` / `('attn_all',layer)` / `('mlp',layer)` /
  `('emb',)` / `('resid',layer)`；
- **5 目标分类**（关键创新，缺一就会误读）：
  - `match_a/b`：对方规则 × 输入自己的历史（**规则翻转**，真正的目标）；
  - `match_src`：源运行的实际续写（**状态转移**，整向量 patch 的平凡结果）；
  - `match_hyb_a/b`：规则 × 杂交操作数对（区分"真崩坏"与"杂交自洽"）；
  - `neither`：四不像（**大部分是自回声**，见 §3 坑 2）。
- 判定原则：规则翻转 = match_对方↑ 且 match_src≈0 且 neither≈0。

### ⑤ evidence_sweep.py（证据值扫描，本次最强工具）
```
python src/evidence_sweep.py model.pth --q 4 --pairs 512 --out report.json
```
- 用模型自身权重合成"读到任意 token"的头输出，扫遍全部 p 个证据值，
  直接画出规则判定函数；
- 报告含：baseline、保真度（truthful evidence 应≈1）、各规则盆地捕获率、
  逐证据值的 match 曲线；
- 横轴用**模 p 比值** ĉ2 = Δ/x̂（Δ = x_(q+1)−x_q）聚合才能看到盆地结构，
  数值距离空间没有任何结构（教训：度量在比值空间）。

### ⑥ head_ablation.py
```
python src/head_ablation.py model.pth --pairs 256
```
- patch 证明"被读取"，消融证明"被需要"；两者必须分开测；
- 典型签名：操作数头→崩；证据头→部分崩且规则间不对称；混合头→无感。

## 3. 必须知道的坑（血泪清单）

1. **聚合稀释**：单位置 patch 只对齐到 k=q+1 评估，切勿摊平；
2. **回声抵消伪影（B1）**：绑定 embedding 使残差流对当前 token 有 +1.36
   自回声投影，MLP 恒以 −1.35 抵消；patch 掉抵消器 → 模型复读当前 token。
   规则对 [(1,1),(1,2)] 有恒等式 x_4^B ≡ x_5^A，会在 k4 AtoB 产生
   match_src=1.0 的**幻影状态转移**。分析新现象时先检查回声；
3. **neither ≈ 回声**：patch 后的四不像输出先验证是不是"复读当前 token"；
4. **规则 tuple 约定**：(c1,c2) ⇒ X(k)=c1·X(k-1)+c2·X(k-2)，与
   `LinearRecurrenceRule` 一致（activation_patch.py 里曾搞反过，已修）；
5. **退化样本**：生成配对样本时排除 x_1=0（b 不可识别）、x_2=0
   （x_4 对所有 b 相同）；
6. **pattern ≠ 内容**：头读哪里不说明它转发什么（V/O 矩阵决定内容），
   功能判定必须过 patch/消融；
7. **逐模型验证解剖**：同一任务存在不同解法（rules23 批 seed18318 的
   证据-操作数纠缠解，无 lag2 纯证据头、无翻转位点）；
8. **GPU/会话**：长任务用 `setsid nohup ... &` 脱离会话，否则会话关闭
   进程被杀（课程调度器 src/curriculum_chain.py 可作模板）。

## 4. 当前结论一句话版

模型判规则 = attention 取三数（lag0/lag1 操作数 + lag2 证据）→
对每条记忆规则做精确残差检验 r_X = x_(q+1)−c1·x_q−c2·x_(q-1) → 归零则按
该规则精确续写，都不归零则按固定先验选一条（或崩坏，取决于模型）。
执行环节的模乘精度是大系数规则的短板；规模/深度决定能装几条规则
（l1 最多 2 条，l2d512 可到 5 条需课程，l4 最优）。

## 5. 代码架构（pipeline 涉及的模块）

### 5.1 分层总览

```
模型层    models.py
规则层    rules.py
数据层    datasets.py
训练编排  experiment.py ← training.py ← final_eval.py
批运行    batch_run.py（experiments/*.json → config_tmp_*.json → core.py）
分析工具  analyze_attention.py（基础设施）
          activation_patch.py / multi_pos_patch.py /
          evidence_sweep.py / head_ablation.py / rule_generalization.py
测试      tests/
```

### 5.2 模型层 `models.py`

- `FibonacciTransformer`：基类。结构：`wte`（与 `lm_head` **权重绑定**——
  回声现象的根源）→（可选 `wpe`，默认 RoPE）→ `transformer.h`（block 列表）
  → `ln_f` → `lm_head`；
- `TransformerBlock`：pre-norm。`x += attn(ln_1(x))`，`x += mlp(ln_2(x))`；
  `mlp = Sequential(Linear, GELU, Linear, Dropout)`——patch 点 `mlp[2]` 是
  第二层 Linear 的输入（post-GELU 隐藏层）；
- `CausalSelfAttention`：`c_attn`（d→3d，按 q,k,v 切分）→ RoPE（只作用 q,k）
  → causal mask → softmax → `att @ v` → `c_proj`（d→d）。
  **patch 点 `c_proj` 的输入** = 各头 att@v 拼接，按 head_size 切列；
- `MixedABTransformer(FibonacciTransformer)`：多规则变体，`rule_head`
  （规则分类头，尺寸随规则数变，INIT_FROM 时会跳过）+ 可选 `ab_emb`/
  `cond_wte`。`use_ab_tag=False` 时 forward 与基类相同；
- 输出协议：`model(idx)` 返回 `(logits, ...)`，分析代码统一取 `out[0]`。

### 5.3 规则层 `rules.py`

- `LinearRecurrenceRule(coeffs, p)`：**约定 (c1,c2) ⇒ X(k)=c1·X(k-1)+c2·X(k-2)**
  （曾在此搞反过，activation_patch 的 bug 之源）；`next_fn()` 供数据集用；
- `rules_from_config(cfg, order)`：order=2 读 `AB_PAIRS`，order=3 读
  `ABC_PAIRS`。

### 5.4 数据层 `datasets.py`

- `RecurrenceDataset`（单规则）：环遍历 + 滑动窗口；**曝光拆分**按初始状态
  ——`num_samples` 之前（shuffle 后顺序）的初始态进 train（曝光），其余进
  test（未曝光）。`MAX_UNIQUE_RATIO`/per-rule ratios 控制曝光率；
- `MixedRecurrenceDataset`：多规则混合；每条规则独立跑一个
  RecurrenceDataset 再拼接。**重建曝光集**：`random.seed(RANDOM_SEED)` →
  逐规则构造（顺序敏感），见 `.chain_tmp/b2_exposure.py`；
- collate 族：普通 / masked（缺失值不计 loss）/ predict（缺失值要预测）；
  `BatchTag` 路由；
- 退化样本注意：x_1=0（证据不可识别）、x_2=0（x_4 不可区分规则）。

### 5.5 训练编排

- `core.py`：**兼容壳**（实际实现已拆到 experiment/training/final_eval）；
- `experiment.py`：`run_experiment(config_path)` 主入口。设种子 →
  `_prepare_mixed_recurrence`（建数据集）→ 建模型 → **`INIT_FROM` 暖启动**
  （`_load_partial_checkpoint`：按 name+shape 匹配拷贝，rule_head 等不匹配的
  自动跳过）→ 训练；
- `training.py`：训练循环。早停双规则：`EARLY_STOP_ACCURACY=0.99`（到线再跑
  200 epoch 收尾）+ `EARLY_STOP_NO_IMPROVE=3000`；
- `final_eval.py`：最终教师强制评估，按位置 × 曝光/未曝光拆分
  （"Exposed-ID / Unexposed-ID" 表格的出处）。

### 5.6 批运行 `batch_run.py`

- 输入：`experiments/*.json`（`{"concurrency": N, "experiments": [...]}`，
  每条含 name/task/config 覆盖）；
- `build_merged_config`：main → task 默认段 → 实验覆盖，自动填 SAVE_PATH，
  打 `_BATCH_RUN_MERGED` 标记（`protocol.py`）；写出 `config_tmp_<name>.json`
  后 spawn `core.py`；
- GPU：`get_idle_gpus()`（nvidia-smi 探测）+ 队列分配，**会覆盖外层
  CUDA_VISIBLE_DEVICES**——要钉卡需直接 spawn core.py（curriculum_chain.py
  就是这么做的）；
- 输出：`/data/cxm/recursion/<批次名>/logs|plots`，`/data/cxm/models/<批次名>/`
  （PATH_CONVENTIONS.md）；
- **长任务务必 `setsid nohup ... &`**，会话关闭会杀后台任务。

### 5.7 分析工具层

| 模块 | 提供 | 关键接口 |
|---|---|---|
| `analyze_attention.py` | 基础设施 | `load_model`（checkpoint→模型，自动识别 MixedAB/PE 方案）、`get_attention_weights`（手动重放前向算 pattern） |
| `activation_patch.py` | HSS 原语 | `generate_paired_samples`、`rule_targets`、`capture_hidden`（缓存 attn/mlp/emb/resid）、`patched_logits`（多位置、多 site 类型）、`classify` |
| `multi_pos_patch.py` | 翻转猎手 | 条件梯子 + 5 目标分类（match_a/b、match_src、match_hyb、neither） |
| `evidence_sweep.py` | 判定函数测绘 | 合成 V 内容 patch，证据值全扫描 + 盆地捕获率 |
| `head_ablation.py` | 必要性测试 | 逐头清零消融，按规则拆分准确率 |
| `rule_generalization.py` | 行为泛化 | 新 b 值续写归属测试（b=1/b=2/真值/随机） |

### 5.8 测试

`tests/test_activation_patch.py`（10 个，含强不变量：末层 resid patch 必须
复现源 logits）等；直接 `python tests/xxx.py` 运行，无需 pytest。

## 6. 待办 / 开放问题

- h2 矩阵批（40 run）收尾后的 h2 vs h4 对照图；
- B4：lag2/lag3 混合头为何惰性（组合 patch 判别）；
- B5：h4 翻转率低于 h2（平台更低的机制原因）；
- l2 模型的判定函数扫描（证据通道在 layer 1，读打包向量）；
- 课程链为什么 l1 走不通（单层放不下第三台乘法器？）；
- d128l2 h4 全崩、head_size 调制的机制诊断。
