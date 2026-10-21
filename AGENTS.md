# AGENTS.md — recursion 实验仓库交接

模运算递推/grokking 实验仓库。代码在 `/mnt/workspace/hujiachen/recursion`（git main），结果库在 `/mnt/workspace/hujiachen/recursion_results`（独立 git master，云同步）。

## 0. 必读的现有文档

- `.agents/skills/recursion-experiment-runner/SKILL.md`：实验运行全流程 + 四个工具脚本（逐卡队列状态、全量汇总、存档点停止、临时定时）。**跑实验前必读**。
- `docs/recursion_model_and_training_settings.md`：所有配置键的完整说明（TRAIN_LEN/MISSING_PROB/NUM_MASK/MAX_TRAIN_HOURS/...），改配置语义后必须同步它。
- `docs/AMP_PRECISION_NOTES.md`：精度教训——结论性实验必须 fp32（默认），bf16 只能粗筛。

## 1. 代码结构（src/）

- `core.py`：薄入口（re-export），实际逻辑在 `experiment.py`（run_experiment 主流程 + 三个 prepare）。
- `experiment.py`：`run_experiment` 开头经 `scenarios.normalize_task_config` 归一化后按 scenario/variant 分派（`_prepare_single_recurrence`（single + 三个非线性例外）/ `_prepare_mixed_recurrence`（mixed）/ `_prepare_action`（action，order 参数区分阶数）/ `_prepare_single_variant`（TASK_VARIANT=corrupt_fix/fill_blank，仅 single））。
- `scenarios.py`：归一化层 `normalize_task_config`（纯配置翻译，无 torch 依赖）：旧 task 名映射（addition/tribonacci/tetranacci→single、mixed_ab/mixed_abc→mixed、action_trib/dynamic_mixed→action）、action 旧键 AB_PAIRS/ABC_PAIRS 反转为 canonical RULES、旧暴露率键→EXPOSURE_RATIOS、已删功能显式启用则报错。
- `rules.py`：递推规则。`LinearRecurrenceRule`（任意阶任意系数的通用入口，canonical 方向）、`single_rule_from_task`（task 名 → (init_len, next_fn, name)）、`rules_from_config(rules_raw, p)`（新签名：消费归一化结果的 canonical 系数列表）。
- `datasets.py`：RecurrenceDataset（全状态枚举+循环遍历，或 STATE_SPACE_CAP 抽样）、MixedRecurrenceDataset、ActionDataset、所有 collate（含变体的 `make_corrupt_fix_collate` / `make_fill_blank_collate`）。
- `training.py`：训练/评估循环、早停、SIGTERM 优雅退出、checkpoint。
- `batch_run.py`：批量编排（读 experiments/*.json，合并配置，一卡一实验）；`chain_rounds.sh` + `prepare_chain_round.py`：多轮续跑。`chain_run.py`：通用课程链（分 stage 串行，`INIT_FROM:"@prev"/@<stage>` 暖启动，gate 门槛，顶层 concurrency 透传）；交接文档 `docs/CURRICULUM_NTASK_HANDOVER.md`。`curriculum_chain_Ntask.py` 是旧的 N=2→5 专用调度器（硬编码，勿用作通用 pipeline）。
- 测试：`tests/test_*.py` 是纯 assert 脚本，**没有 pytest**；本机运行 `python tests/test_xxx.py`（test_sample_gen.py 等需要 `PYTHONPATH=src`）。

## 2. 配置合并与实验 JSON

- **实验交付流程（用户要求，2026-09-22）**：用户提出实验及参数后，先按 `docs/EXPERIMENT_DELIVERY_PIPELINE.md` 打印完整实际参数表（包括默认/参考来源、样本数、mask、seed、初始化/续跑、LR/WD/clip、重洗牌、精度、scheduler horizon、评估间隔和定时），等待用户明确确认该版本；确认后才生成可执行实验 JSON 与可分发运行包。参数或代码版本变化后重新出表确认。默认交付给用户在其他机器运行，不在本机自动启动实验。构建 pipeline 本身和 CPU 测试不属于批准新的正式实验。

- 合并顺序：`src/config.json` 的 main → 场景默认段（场景段→旧 task 同名段去重，batch_run.py:129-134）→ 实验 config 覆盖。仓库内传统实验 JSON 只写与默认不同的键；跨机交付包展开完整有效配置，避免目标机器默认值改变实验。
- 命名约定见 SKILL.md §1；本机 v2 制度（30万样本/fresh 256 测试）写 `experiments/v2/` 并用 `--base-dir /mnt/workspace/hujiachen/recursion_results_v2 --model-base-dir /mnt/workspace/hujiachen/models_v2` 启动。（注意：云端机器 2026-09-11 起已废弃 v2 隔离目录并入主库；本机仍保留该目录。）

## 3. 最近的语义改动（改代码时注意一致性）

0. **场景统一（2026-10-08）**：task 体系收敛为 TASK ∈ {single, mixed, action} + 三个非线性例外 task（multiplication/nonlinear/nonlinear_mul），新键 ORDER / RULES（canonical：第一个系数乘最新的值）/ EXPOSURE_RATIOS（标量或逐规则列表）；写死 PREDICT_MISSING=true / USE_LEARNABLE_PE=false（RoPE）/ ALLOW_TF32=true / USE_AMP=false / ENTROPY_PENALTY_WEIGHT=0；删除 MixedABTransformer/USE_AB_TAG/cond_wte/MISS_SECOND/COND_FIX 系列/final generation test（final_eval.py）/mask 污损口径；rule_head 已恢复为 mixed 场景的默认组件（`AUX_CLASSIFY`，默认开，仅 mixed，2026-10-10）。旧 task 名与旧键全部经归一化层（scenarios.py）兼容，已删功能显式启用则报错；config.json 新增场景默认段（查找顺序：场景段→旧 task 段）；checkpoint save_config 追加 scenario/order/rules 三键。设计规格：`docs/superpowers/specs/2026-10-08-scenario-unification-design.md`。**已知语义变化四条**：①污损恒 predict 口径（single/mixed 历史默认 mask，新旧 test acc 口径不可直接比较）；②mixed 无 tag/cond_wte，rule_head 默认开启（`AUX_CLASSIFY: false` 可关，仅 mixed；同配置新跑与历史 mixed 有架构差异）；③无 final generation test（Exposed/Unexposed × ID/OOD 表格不再产出，结果口径 = 训练日志 best acc + `eval_clean_action.py`）；④OOD_LEN 默认 32→16（EARLY_STOP_EXTRA_EPOCHS 默认 200→0）。
1. **污损全面改为在线模式**：数据集只存干净样本，`make_missing_collate` / `make_mixed_missing_collate` / `make_action_missing_collate` 逐 batch 新鲜随机污损（每 epoch 缺失位置都不同，train/test 同样）。旧的"生成时固定污损"已删除，无开关。
2. **早停分两制**：静态切分制（无 fresh test）测试 ≥ `EARLY_STOP_ACCURACY`(0.99) 直接停；新鲜测试制（`FRESH_TEST_PER_EVAL`）最近 10 个 eval 均值 >99% 才停。`EARLY_STOP_EXTRA_EPOCHS` 保留但默认 0、当前无作用（`high_acc_epoch` 仅为 resume checkpoint 兼容保留）。
3. **滚动 checkpoint**：每个 eval 点覆盖写 `<model>_latest.pth`（完整状态可 RESUME_FROM），best 更新时写 `<model>_best.pth`（纯权重）。
4. **tetranacci task**：4 阶递推（旧写法 A/B/C/D 键，init_len=4；新写法 `single` + `ORDER:4` + `RULES`）。**P⁴ 状态空间必须配 `STATE_SPACE_CAP` 或 `DATA_MODE: sampled_fresh_test`**（否则枚举爆炸）。
5. **单规则任务支持** `NUM_TRAIN_SAMPLES` 显式覆盖 和 `FRESH_TEST_PER_EVAL`（每 eval 从全状态空间抽 NUM_TEST_SAMPLES 个全新状态生成 OOD_LEN 窗口）。**mixed 也支持 `NUM_TRAIN_SAMPLES` 显式覆盖**（2026-09-10）：标量=每规则同样数量，列表=逐规则数量，设置后覆盖 `EXPOSURE_RATIOS`（旧键 `MIXED_AB_MAX_UNIQUE_RATIOS`）的反推。
6. **batch_run 直接文件重定向**（不再管道转发）：杀编排器不会误杀训练进程。
7. **action_trib task**（2026-09-10，现为 action 场景的 order=3 兼容别名）：`[x0 x1 x2 flag1 x3 flag2 x4 ...]` 逐位置插 flag，`x_k = a·x_{k-3}+b·x_{k-2}+c·x_{k-1}`。与 action 共用 `ActionDataset`/`_prepare_action`（`order` 参数），旧 JSON 规则读 `ABC_PAIRS`（反向约定，归一化层反转；文档见 docs/recursion_model_and_training_settings.md §1.6）。attention 分析已按 order 参数化，action 家族全阶数可用。
8. **corrupt_fix 任务变体**（2026-10-14）：新正交键 `TASK_VARIANT`（场景下一级，默认 null）。`corrupt_fix` 目前仅 single：collate 在线把恰好一个位置（含初值）改成随机错误值（≠原值，非 missing token），尾部接 EOS（id=P+1），模型自回归输出位置 token（id=P+2+k）与正确值（teacher forcing）；词表 `P+2+max(TRAIN_LEN,OOD_LEN)`，loss/acc 只算最后两个输出位（复用 targets_override 约定，training.py 零改动）。与 `MISSING_PROB>0` 互斥（报错），`NUM_MASK` 无意义（警告忽略）；mixed/action 组合报“尚未实现”。规格见 docs/recursion_model_and_training_settings.md §1.7。
9. **MLP_RATIO 默认 4→8 / N_HEAD 默认 1→4**（2026-10-14）：`src/config.json` main 段与代码内全部 fallback（experiment.py / models.py / eval_clean_action.py）同步改。未显式写这些键的旧实验 JSON 重跑会得到不同架构（历史实验 JSON 多数显式写了，不受影响）。
10. **ORDER=1 开放给 single**（2026-10-14）：等比递推 `x_k = c·x_{k-1}`（注意不是等差——框架无常数项）。状态空间 `P^1 = P`，建议大 P；mixed/action 仍要求 ORDER≥2（归一化层报错）。
11. **fill_blank 任务变体**（2026-10-14）：`TASK_VARIANT='fill_blank'`（仅 single）：恰好一位替换成空白 token（id=P），尾部接 EOS，模型只输出正确值一个 token（loss/acc 仅 EOS 槽）；词表 `P+2`，prepare 与 corrupt_fix 共用 `_prepare_single_variant`。约束同 corrupt_fix（与 MISSING_PROB 互斥、NUM_MASK 忽略、mixed/action 报错）。规格见 settings 文档 §1.8。
12. **CONST 常数项 + FILL_BLANK_ALL_POSITIONS + TEST_ALL_POSITIONS**（2026-10-14）：①`CONST`（默认 0，仅 single 可非零）给递推公式加常数项 `x_k = ... + CONST mod P`；等差数列 = `ORDER=1, RULES=[[1]], CONST=d`。键名不能用 C（A/B/C/D 已被旧系数占用）。②`FILL_BLANK_ALL_POSITIONS: true`（仅 fill_blank，其他情况报错）：每条序列的每个位置挖空各成一条样本（(seq,k) 元组，样本数 × 窗口长度），collate 用样本携带的位置。③`TEST_ALL_POSITIONS: true`（仅 fill_blank）：只把测试集改成逐位置枚举（确定性评估口径，eval 计算 × 窗口长度），训练侧保持在线随机挖空；静态切分制下 `NUM_TEST_SAMPLES` 变为测试窗口数上限（默认 256，确定性子抽样）。checkpoint save_config 的 single 块新增 `const` 键。
13. **CORRUPT_FIX_PARALLEL 并行解码**（2026-10-14）：仅 corrupt_fix。true 时输入末位用第二个 EOS 代替真实位置 token（双 EOS 槽），位置/值并行预测，值看不到位置答案（默认 false = teacher forcing，值 acc 是“给定点位置算值”的条件准确率，端到端能力≈位置 acc）。save_config 记 `parallel_decode`。
14. **POS_PROBE 逐位置探针**（2026-10-14，变体默认开）：corrupt_fix/fill_blank 每个 eval 点在固定 256 窗口探针集上按挖空/污损位置 k 分桶评估，日志打 `PosProbe pos|/val|` 行（corrupt_fix 两行：位置/值；fill_blank 一行：值）。实现：`experiment._make_pos_probe`（(seq,k) 元组固定 k，两个 collate 均支持元组样本；不定长时逐 (l,k) 再按 k 加权聚合）+ 引擎 `extra_eval_fn` 钩子。epoch×k 热力图的数据源。
15. **变体主体监督 + 不定长**（2026-10-15）：①`BODY_SUPERVISION`（变体默认 true）：targets 前缀改用干净序列、loss_mask 覆盖递推主体 [order-1, L-1) + 输出位，不再只监督最后输出位。②`SEQ_LEN_MIN`/`SEQ_LEN_MAX`（仅变体，必须成对且 min > 阶数）：窗口逐样本随机截断成不定长，分桶 sampler 按长度组 batch；词表/block 按 max。两个键在非变体下设置会报错。save_config 记 `body_supervision`/`seq_len_min`/`seq_len_max`。
16. **CORRUPT_FIX_PAIRED_CLEAN**（2026-10-20，仅 corrupt_fix，与 PARALLEL 互斥）：配对双样本——裸窗口产 干净续写行（主体监督，正确值暴露）+ 污损答题行（只 pos/val 两槽）两行；(seq,k) 元组只产答题行，POS_PROBE 口径不变；批内右补齐。save_config 记 `paired_clean`。
17. **CORRUPT_FIX_MASK_AROUND**（2026-10-21，仅 corrupt_fix）：动态主体 mask——污损位 k 牵连的预测位 {k-1,k,k+1}（预测 x_k/x_{k+1}/x_{k+2}）不计主体 loss/acc；pos/val 槽不动；评估同口径（0.99 早停恢复可达）。paired_clean 下无作用。

15. **SAVE_CHECKPOINTS**（2026-10-19，默认 true）：false 时训练全程不落任何存档（`_best`/`_latest`/`_resume`/最终 `model.pth` 都不写），仅日志，用于大批量粗扫省磁盘；超时/停掉的实验不可续跑，attention 后处理自动跳过。文档：settings §训练循环与早停。

## 4. 本机环境的坑（反复踩过）

- **GPU（2026-10-10 更新）**：本机 8 卡全部可用（0-7；旧的"只用 0-3"限制是另一台机器的规定，已删除）。选卡时先用 `nvidia-smi` 看占用；启动后核对日志里的 `on cuda:N` 是你要的卡。
- **队列制度（2026-09-10 起）**：一卡一个队列，用 `scripts/start_queue.sh <gpu 0-7> <experiments.json>` 启动（内部 `BATCH_RUN_GPUS=<gpu>` 钉卡，同卡已有队列会拒绝启动）。batch_run.py 的 GPU 白名单默认 0-7。
- **BATCH_SIZE 默认 512**：bench 结论（`bench_batch_size/bench_action_bs_results.txt`）——action misslen2 l4 fp32 下 512 与 1024 同速（~372s/epoch @30万样本），显存 26.5GB vs 40.7GB，bs≥1280 OOM。所有实验就用默认 512。
- **misslen 实验的评估 pipeline（2026-09-15 起固定）**：每条 misslen 实验训练完毕后，结果必须报两个维度——**clean acc**（规则学习能力）：用 `scripts/eval_clean_action.py --eval-len <TRAIN_LEN> --num-samples 1000` 在新生成的无污损 train_len 窗口样本上评（取该实验最后一轮的 `_best.pth`，缺则 `_latest.pth`；先确认该轮没发散——latest 明显比历史 best 差时改用未发散轮的 best）；**total acc**（填补污损能力）：就是训练日志里的 best test acc。已知事实：正常收敛的 action+miss 模型 clean acc 在训练窗内都是 100%，差距全在 total acc；续跑轮可能发散把模型跑废，采用续跑结果前必须先验证。
- **Python 环境（2026-10-09 更新）**：用 conda base `/home/cxm/miniconda3/bin/python`（3.13，torch 2.6.0+cu124，cuda 可用，matplotlib/numpy 齐全），裸 `python`/`python3` 默认就指向它。**系统 `/usr/bin/python3`（3.10）已无 torch，别再用**。另有 `torch_env` 环境有 torch 但缺 matplotlib，测试/画图不要用。chain_rounds.sh 内部的裸 `python` 现在直接可用。
- **GPFS 写坑（2026-09-11 踩过）**：/mnt/workspace 是 GPFS，曾出现约 15 分钟的瞬时写故障，torch.save 写 checkpoint 直接失败（`PytorchStreamWriter failed writing`，stderr 在批次 `logs/*.err`），正在写的 `_resume.pth` 被截断成 4MB 废文件。**教训：resume 前先 torch.load 校验 checkpoint；`_resume.pth` 损坏时改用 `_latest.pth`（滚动全状态，通常完好）；查训练报错要看 logs/ 下的 .err 不是 .log。**
- **pgrep/pkill 自匹配**：模式会匹配到自己 shell 的命令行，用括号写法 `[c]ore.py` 或先核对 PID。
- **停止实验**：用 skill 的 stop_experiment.sh（SIGTERM → 下一 eval 点存盘），或 timer.sh 设定时。不要 kill -9 正在跑且进度有用的实验。
- GPU 检测回退到 CPU 时实际会跑上 cuda:0 撞车——启动后务必核对日志里的 `on cuda:N` 是你要的卡。

## 5. 当前在跑（会话结束时快照，以 gpu_queue_status.py 为准）

- 本机 GPU 0-7 均可用；以 gpu_queue_status.py 为准查在跑实验。
