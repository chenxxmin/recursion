# rule_fit 专项工作交接文档

> 写给专门做 `src/rule_fit.py` 工作的会话。
> 基于 2026-08-14 全库代码审查 + 后续 core.py 拆分（已由另一会话完成）后的仓库现状。
> 行号以当前工作区为准。

## 1. 这个脚本是干什么的

`rule_fit.py`（613 行）对**训练好的单规则模型**做"规则发现"式分析：不假设模型学会了
哪条递推，而是从模型的预测和注意力出发，在 F_p 上**拟合**线性公式，验证模型实际使用
的计算规则。两种分析模式：

- **常规实验**（无 MISSING_PROB）：按 ATTENTION SIGNATURE（显著注意力距离集合，阈值
  0.10）把预测位置分段，每段用**完全随机**的探针序列（off-manifold 设计）拟合系数；
  最后把拟合公式相同的相邻段合并成组输出。
- **missing 实验**（MISSING_PROB>0，且未加 --clean）：走**队列驱动 BFS**——根为
  全部 2^init_len 个长度 init_len 的 pattern（order-2 即 (x,x)/(x,M)/(M,x)/(M,M)）。
  每个 pattern 报告 model-vs-truth 一致率（真实污损序列，不再量真实注意力），
  假设集 C 从**探针分布本身**量注意力得到（`probe_attention`：单窗口强制 mask、
  深文干净随机，与拟合同分布，避免子 pattern 混合污染），再用随机探针
  （off-manifold，只强制窗口内 mask）拟合 C 的系数。展开规则（`mask_children`/`refine_children`，查重，长度上限
  --depth 默认 8）：拟合可信（agreement≥0.8）→ 公式非零系数距离是规则用到的
  位置，每个位置单独入队一个"该位置也被 M"的子 pattern；不可信 → 该 pattern
  可能是更深子情形的混合，窗口加深一位，(x,) 与 (M,) 两种状态分别入队。
  打印：只有拟合可信的 pattern 才打条目（头行 + 公式行，0 系数项不打），其余
  静默展开。加 `--influence` 后，拟合失败的 pattern 改测逐位置影响率
  （`position_influence`，反事实扰动，不假设线性）并打 influence 行、以影响率
  ≥0.5 的位置驱动展开——但真实模型上扰动会连带移动注意力，噪音地板高，输出
  偏糊，故默认关闭。

**只支持单规则任务**（addition/multiplication/tribonacci/nonlinear）；mixed_ab/mixed_abc
在 `run_one` 里直接跳过（`rule_fit.py:371`），dynamic_mixed 会走 build_single_rule 报错路径。

## 2. 仓库现状（先读这个，和旧认知不同）

core.py 已拆分完毕（CORE_SPLIT_PLAN.md 步骤 1-4 已执行）：

```
src/
  models.py     (337)  RotaryEmbedding/CausalSelfAttention/TransformerBlock/_lm_loss/
                       FibonacciTransformer/MixedABTransformer
  datasets.py   (623)  RecurrenceDataset/MixedRecurrenceDataset/DynamicMixedDataset/
                       BatchTag+collates/BucketBatchSampler/corrupt_window/missing_token_id
  training.py   (371)  train_epoch/evaluate/run_training_engine/_unpack_batch/_sample_seq 等
  final_eval.py (220)  stage-3 曝光统计/_teacher_forced_correct
  experiment.py (484)  三个 _prepare_* + run_experiment
  core.py       (27)   兼容壳：re-export 测试引用的符号 + __main__ 入口
  rules.py      (123)  LinearRecurrenceRule/rules_from_config/single_rule_from_task/
                       save_config_extra/task_from_save_config
  protocol.py   (15)   BATCH_RUN_MERGED_FLAG（跨进程协议常量，无依赖）
```

- `batch_run.py` 以 `python src/core.py <merged_config>` 派生训练子进程（不再是 -c 字符串）
- 测试：8 个文件全绿，每个 `tests/test_*.py` 都有 `__main__` runner，可直接
  `python tests/test_x.py` 跑。**本机环境注意**：默认 anaconda base 无 torch；用
  `C:/Users/Chen/anaconda3/envs/torch/python.exe`（torch 2.5.1+cu124，无 pytest，用 runner 跑）
- 真实 checkpoint 和训练日志在**服务器** `/data/cxm/...`，本机（Windows）没有

## 3. rule_fit.py 结构与数据流

### CLI（`main()`, :522-613）

```
python src/rule_fit.py <name>                 # 批次模式：迭代 experiments/<name>.json
python src/rule_fit.py <name> <exp_name>      # 单实验模式：<name> 是模型目录名也是 json 名，
                                              # <exp_name> 是 .pth 文件名也是 json 条目名
python src/rule_fit.py <model.pth> <cfg.json> # 单模型模式：按 pth 文件名 stem 匹配实验
  [--samples N=500] [--seed 0] [--length L] [--min-seg N=1]
  [--depth D=8] [--min-n N=20] [--clean]
```

输出 tee 到 `rule_fit_output.log`（CWD）。

### 主流程（批次模式）

```
experiments/<name>.json 逐实验
  → 拼模型路径 /data/cxm/models/<name>/<exp>.pth
  → 在 LOG_BASE_CANDIDATES 两个候选目录找 <exp>.log
  → run_one(args, exp_cfg, task, exp_name, pth, log)
       load_model(pth) 重建模型（analyze_attention.py，处理 MixedABTransformer 检测、
                       PE/mlp_ratio 推断；rule_fit 实际只接 FibonacciTransformer）
       task 回退：task_from_save_config(checkpoint['config'])（rules.py，
                  会把 checkpoint 词汇 'multiplicative' 归一为 'multiplication'）
       build_single_rule(task, cfg)（verify_sample 包装 rules.single_rule_from_task）
       ├─ MISSING_PROB>0 且无 --clean → run_missing_categories() (:227)
       └─ 否则 → 注意力模式 (:385-503)：
            parse_log(log) 拿 (start_x, series, attn)
            segment_by_attention(attn, start_pos=num_mask-1, threshold=0.10) 分段
            [--min-seg>1 时 _merge_short_segments]
            dump_matrices 打印注意力符号网格
            全随机探针 × args.samples → 每段每位置一条方程
            fit_and_score（先 solve_mod_p 精确解，失败则 RANSAC 300 轮）
            相邻同式段合并成组（norm_C 丢弃零系数）输出
  → 末尾 MISSING REPORT（缺模型/缺日志/报错的实验清单）
```

### 关键函数

| 函数 | 位置 | 作用 |
|---|---|---|
| `solve_mod_p` | :59 | F_p 上高斯消元精确解，无唯一解返回 None |
| `fit_and_score` | :90 | 先精确解，失败回退 RANSAC（返回 (coeffs, agreement, exact)） |
| `fmt_equation` | :221 | 公式排版 `x_{t+1} = c1*x_{t-d1} + ... (mod p)` |
| `run_missing_categories` | :227 | missing 模式：BFS 队列编排 |
| `run_one` | :360 | 单实验编排（task 回退、规则解析、模式分派） |
| `_merge_short_segments` | :504 | 短段并入邻段 |

missing 模式辅助函数组：`visible_distance` / `patt_label` / `child_patterns` /
`aggregate_buckets` / `select_C` / `rule_coeffs` / `probe_equations`（:113-226，见
`run_missing_categories` docstring 与 REFACTOR_LOG #53）。

### 注意力数据从哪来（重要，别找错地方）

rule_fit 自己**不算**注意力矩阵——它解析的是训练日志末尾的 `Attention Analysis` 段：
`batch_run.run_attention_analysis` 在每个实验成功后追加
`analyze_attention.analyze_model_attention` 的输出（`:326-329` 打印的
`i= N -> j=M:val ...` 矩阵行），`report_front_back_attention.parse_log` 的正则
（LAYER_RE/HEAD_RE/IROW_RE/JPAIR_RE）与之精确匹配。**注意口径**：该矩阵来自
analyze_model_attention 选的**单条测试序列**，不是训练集统计——签名分段反映的是单序列
注意力结构。改 analyze 的打印格式会静默打断 parse_log（正则耦合，无测试钉住）。

### 外部接口（改动这些文件会影响 rule_fit）

- `analyze_attention.load_model` / `extract_qk_raw_scores`（后者是 `_forward_per_layer`
  的薄包装，返回每层 attn_weights 等，cpu 张量）
- `report_front_back_attention.parse_log / split_k / segment_by_attention / dump_matrices`
- `verify_sample.build_single_rule / _Tee`
- `rules.task_from_save_config`
- `datasets.corrupt_window`（污损探针用，单规则 missing_token=p）

## 4. 已知问题（审查发现，仍未修）

1. **配置不合并（最重要）**：`run_one` (:362) `cfg = dict(exp_cfg)` 直接用实验覆盖配置，
   不合并 `src/config.json` 的 main/任务段。后果：tribonacci 批次（实验不写 P，靠 base 段
   P=23）每个都 `TypeError` 崩；`TRAIN_LEN` 缺省静默用 16 (:242/:397)。**建议**：复用
   `batch_run.build_merged_config`（去掉 SAVE_PATH 填充）拿合并后配置。
2. **路径硬编码**：`MODEL_BASE='/data/cxm/models'` (:54)、`LOG_BASE_CANDIDATES` (:55)
   无 CLI 覆盖；离开服务器即不可用时是静默 skip（missing report 里一堆 "model not found"）。
3. **`out_path` 重复赋值**：:542 和 :575 各写一遍 `'rule_fit_output.log'`。
4. **报错不带 traceback**：:566/:601 只打 `type(exc).__name__: exc`，批处理排障困难。
   建议加 `--verbose` 开关打印 traceback。
5. **正则耦合无测试**：parse_log 的正则与 analyze_attention 打印格式之间的匹配没有任何
   测试钉住（missing 模式已由 `tests/test_rule_fit_missing.py` 覆盖，常规注意力模式仍无）。
6. 小：`verify_sample.py:26` 注释是从 analyze_attention 复制的，CLI 描述不对。

## 5. 修改时的注意事项

- **seed 语义**：`run_one` :378 `random.seed(args.seed)` 之后才生成探针/污损；在此之前的
  任何新增 random 调用都会改变全部下游随机序列（输出可比性）。探针（probe_equations）
  在 seed 之后消耗 RNG，属合法位置。
- **两套配置词汇**：checkpoint save_config 用小写（`p`/`a`/`b`/`recurrence`），实验覆盖
  配置用大写（`P`/`A`/`B`/`TRAIN_LEN`）。读取时用现有的归一化入口
  （`task_from_save_config`/`single_rule_from_task` 已内建大小写容忍），不要再手写映射。
- **验证方式**：本机无真实数据，可造合成 checkpoint 冒烟（参考
  `tests/test_training_smoke.py` 的模式：models.FibonacciTransformer + torch.save 带
  config dict）；端到端验证必须在服务器用真实批次跑一次，检查拟合输出是否合理
  （拟合出的公式应与训练规则一致或揭示 shortcut）。
- **不要顺手改的**：`evaluate` 的 loss mask 有意不带 first_task_weight；BucketBatchSampler
  每 epoch batch 固定（是否疏漏未定论）；这些与 rule_fit 无关但同属本库"有意差异"。

## 6. 相关文档

- `REFACTOR_LOG.md`：52+ 条历史等价重构记录（含多次行为修复的根因分析，值得先读）
- `CORE_SPLIT_PLAN.md`：core.py 拆分计划（已执行完毕，含"不要顺手改进"清单）
- `recursion_model_and_training_settings.md`：模型结构与训练设置（已同步到拆分前状态，
  §7 源码位置表仍指 core.py——拆分后的模块映射以本文档 §2 为准）
- `PATH_CONVENTIONS.md`：日志/模型/图片路径约定
