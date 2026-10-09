# AGENTS.md — recursion 实验仓库交接

模运算递推/grokking 实验仓库。代码在 `/mnt/workspace/hujiachen/recursion`（git main），结果库在 `/mnt/workspace/hujiachen/recursion_results`（独立 git master，云同步）。

## 0. 必读的现有文档

- `.agents/skills/recursion-experiment-runner/SKILL.md`：实验运行全流程 + 四个工具脚本（逐卡队列状态、全量汇总、存档点停止、临时定时）。**跑实验前必读**。
- `docs/recursion_model_and_training_settings.md`：所有配置键的完整说明（TRAIN_LEN/MISSING_PROB/NUM_MASK/MAX_TRAIN_HOURS/...），改配置语义后必须同步它。
- `docs/AMP_PRECISION_NOTES.md`：精度教训——结论性实验必须 fp32（默认），bf16 只能粗筛。

## 1. 代码结构（src/）

- `core.py`：薄入口（re-export），实际逻辑在 `experiment.py`（run_experiment 主流程 + 三个 prepare）。
- `experiment.py`：`run_experiment` 开头经 `scenarios.normalize_task_config` 归一化后按 scenario 分派（`_prepare_single_recurrence`（single + 三个非线性例外）/ `_prepare_mixed_recurrence`（mixed）/ `_prepare_action`（action，order 参数区分阶数））。
- `scenarios.py`：归一化层 `normalize_task_config`（纯配置翻译，无 torch 依赖）：旧 task 名映射（addition/tribonacci/tetranacci→single、mixed_ab/mixed_abc→mixed、action_trib/dynamic_mixed→action）、action 旧键 AB_PAIRS/ABC_PAIRS 反转为 canonical RULES、旧暴露率键→EXPOSURE_RATIOS、已删功能显式启用则报错。
- `rules.py`：递推规则。`LinearRecurrenceRule`（任意阶任意系数的通用入口，canonical 方向）、`single_rule_from_task`（task 名 → (init_len, next_fn, name)）、`rules_from_config(rules_raw, p)`（新签名：消费归一化结果的 canonical 系数列表）。
- `datasets.py`：RecurrenceDataset（全状态枚举+循环遍历，或 STATE_SPACE_CAP 抽样）、MixedRecurrenceDataset、ActionDataset、所有 collate。
- `training.py`：训练/评估循环、早停、SIGTERM 优雅退出、checkpoint。
- `batch_run.py`：批量编排（读 experiments/*.json，合并配置，一卡一实验）；`chain_rounds.sh` + `prepare_chain_round.py`：多轮续跑。`chain_run.py`：通用课程链（分 stage 串行，`INIT_FROM:"@prev"/@<stage>` 暖启动，gate 门槛，顶层 concurrency 透传）；交接文档 `docs/CURRICULUM_NTASK_HANDOVER.md`。`curriculum_chain_Ntask.py` 是旧的 N=2→5 专用调度器（硬编码，勿用作通用 pipeline）。
- 测试：`tests/test_*.py` 是纯 assert 脚本，**没有 pytest**；本机运行 `/usr/bin/python3 tests/test_xxx.py`（test_sample_gen.py 需要 `PYTHONPATH=src`）。

## 2. 配置合并与实验 JSON

- **实验交付流程（用户要求，2026-09-22）**：用户提出实验及参数后，先按 `docs/EXPERIMENT_DELIVERY_PIPELINE.md` 打印完整实际参数表（包括默认/参考来源、样本数、mask、seed、初始化/续跑、LR/WD/clip、重洗牌、精度、scheduler horizon、评估间隔和定时），等待用户明确确认该版本；确认后才生成可执行实验 JSON 与可分发运行包。参数或代码版本变化后重新出表确认。默认交付给用户在其他机器运行，不在本机自动启动实验。构建 pipeline 本身和 CPU 测试不属于批准新的正式实验。

- 合并顺序：`src/config.json` 的 main → 场景默认段（场景段→旧 task 同名段去重，batch_run.py:129-134）→ 实验 config 覆盖。仓库内传统实验 JSON 只写与默认不同的键；跨机交付包展开完整有效配置，避免目标机器默认值改变实验。
- 命名约定见 SKILL.md §1；本机 v2 制度（30万样本/fresh 256 测试）写 `experiments/v2/` 并用 `--base-dir /mnt/workspace/hujiachen/recursion_results_v2 --model-base-dir /mnt/workspace/hujiachen/models_v2` 启动。（注意：云端机器 2026-09-11 起已废弃 v2 隔离目录并入主库；本机仍保留该目录。）

## 3. 最近的语义改动（改代码时注意一致性）

0. **场景统一（2026-10-08）**：task 体系收敛为 TASK ∈ {single, mixed, action} + 三个非线性例外 task（multiplication/nonlinear/nonlinear_mul），新键 ORDER / RULES（canonical：第一个系数乘最新的值）/ EXPOSURE_RATIOS（标量或逐规则列表）；写死 PREDICT_MISSING=true / USE_LEARNABLE_PE=false（RoPE）/ ALLOW_TF32=true / USE_AMP=false / ENTROPY_PENALTY_WEIGHT=0；删除 MixedABTransformer/rule_head/USE_AB_TAG/cond_wte/MISS_SECOND/COND_FIX 系列/final generation test（final_eval.py）/mask 污损口径。旧 task 名与旧键全部经归一化层（scenarios.py）兼容，已删功能显式启用则报错；config.json 新增场景默认段（查找顺序：场景段→旧 task 段）；checkpoint save_config 追加 scenario/order/rules 三键。设计规格：`docs/superpowers/specs/2026-10-08-scenario-unification-design.md`。**已知语义变化四条**：①污损恒 predict 口径（single/mixed 历史默认 mask，新旧 test acc 口径不可直接比较）；②mixed 无 rule_head/tag/cond_wte（同配置新跑与历史 mixed 有架构差异）；③无 final generation test（Exposed/Unexposed × ID/OOD 表格不再产出，结果口径 = 训练日志 best acc + `eval_clean_action.py`）；④OOD_LEN 默认 32→16（EARLY_STOP_EXTRA_EPOCHS 默认 200→0）。
1. **污损全面改为在线模式**：数据集只存干净样本，`make_missing_collate` / `make_mixed_missing_collate` / `make_action_missing_collate` 逐 batch 新鲜随机污损（每 epoch 缺失位置都不同，train/test 同样）。旧的"生成时固定污损"已删除，无开关。
2. **早停分两制**：静态切分制（无 fresh test）测试 ≥ `EARLY_STOP_ACCURACY`(0.99) 直接停；新鲜测试制（`FRESH_TEST_PER_EVAL`）最近 10 个 eval 均值 >99% 才停。`EARLY_STOP_EXTRA_EPOCHS` 保留但默认 0、当前无作用（`high_acc_epoch` 仅为 resume checkpoint 兼容保留）。
3. **滚动 checkpoint**：每个 eval 点覆盖写 `<model>_latest.pth`（完整状态可 RESUME_FROM），best 更新时写 `<model>_best.pth`（纯权重）。
4. **tetranacci task**：4 阶递推（旧写法 A/B/C/D 键，init_len=4；新写法 `single` + `ORDER:4` + `RULES`）。**P⁴ 状态空间必须配 `STATE_SPACE_CAP` 或 `DATA_MODE: sampled_fresh_test`**（否则枚举爆炸）。
5. **单规则任务支持** `NUM_TRAIN_SAMPLES` 显式覆盖 和 `FRESH_TEST_PER_EVAL`（每 eval 从全状态空间抽 NUM_TEST_SAMPLES 个全新状态生成 OOD_LEN 窗口）。**mixed 也支持 `NUM_TRAIN_SAMPLES` 显式覆盖**（2026-09-10）：标量=每规则同样数量，列表=逐规则数量，设置后覆盖 `EXPOSURE_RATIOS`（旧键 `MIXED_AB_MAX_UNIQUE_RATIOS`）的反推。
6. **batch_run 直接文件重定向**（不再管道转发）：杀编排器不会误杀训练进程。
7. **action_trib task**（2026-09-10，现为 action 场景的 order=3 兼容别名）：`[x0 x1 x2 flag1 x3 flag2 x4 ...]` 逐位置插 flag，`x_k = a·x_{k-3}+b·x_{k-2}+c·x_{k-1}`。与 action 共用 `ActionDataset`/`_prepare_action`（`order` 参数），旧 JSON 规则读 `ABC_PAIRS`（反向约定，归一化层反转；文档见 docs/recursion_model_and_training_settings.md §1.6）。attention 分析已按 order 参数化，action 家族全阶数可用。

## 4. 本机环境的坑（反复踩过）

- **GPU 硬性规定（长期有效）：只能用 GPU 0-3，GPU 4-7 绝对禁用**。启动实验前必须确认 `CUDA_VISIBLE_DEVICES` 或日志里的 `on cuda:N` 只落在 0-3；选卡时先用 `nvidia-smi` 看占用。
- **队列制度（2026-09-10 起）**：一卡一个队列，用 `scripts/start_queue.sh <gpu 0-3> <experiments.json>` 启动（内部 `BATCH_RUN_GPUS=<gpu>` 钉卡，同卡已有队列会拒绝启动）。batch_run.py 的 GPU 白名单默认就是 0,1,2,3。
- **BATCH_SIZE 默认 512**：bench 结论（`bench_batch_size/bench_action_bs_results.txt`）——action misslen2 l4 fp32 下 512 与 1024 同速（~372s/epoch @30万样本），显存 26.5GB vs 40.7GB，bs≥1280 OOM。所有实验就用默认 512。
- **misslen 实验的评估 pipeline（2026-09-15 起固定）**：每条 misslen 实验训练完毕后，结果必须报两个维度——**clean acc**（规则学习能力）：用 `scripts/eval_clean_action.py --eval-len <TRAIN_LEN> --num-samples 1000` 在新生成的无污损 train_len 窗口样本上评（取该实验最后一轮的 `_best.pth`，缺则 `_latest.pth`；先确认该轮没发散——latest 明显比历史 best 差时改用未发散轮的 best）；**total acc**（填补污损能力）：就是训练日志里的 best test acc。已知事实：正常收敛的 action+miss 模型 clean acc 在训练窗内都是 100%，差距全在 total acc；续跑轮可能发散把模型跑废，采用续跑结果前必须先验证。
- **Python 环境**：用系统 `/usr/bin/python3`（3.10，torch 2.4.0+cu121，matplotlib/numpy 齐全）。旧机器的 `/home/cxm/miniconda3` 不存在；chain_rounds.sh 内部用裸 `python`，调它前确认 `python` 指向正确（必要时 `export PATH` 使 `python` 可用或改脚本）。
- **GPFS 写坑（2026-09-11 踩过）**：/mnt/workspace 是 GPFS，曾出现约 15 分钟的瞬时写故障，torch.save 写 checkpoint 直接失败（`PytorchStreamWriter failed writing`，stderr 在批次 `logs/*.err`），正在写的 `_resume.pth` 被截断成 4MB 废文件。**教训：resume 前先 torch.load 校验 checkpoint；`_resume.pth` 损坏时改用 `_latest.pth`（滚动全状态，通常完好）；查训练报错要看 logs/ 下的 .err 不是 .log。**
- **pgrep/pkill 自匹配**：模式会匹配到自己 shell 的命令行，用括号写法 `[c]ore.py` 或先核对 PID。
- **停止实验**：用 skill 的 stop_experiment.sh（SIGTERM → 下一 eval 点存盘），或 timer.sh 设定时。不要 kill -9 正在跑且进度有用的实验。
- GPU 检测回退到 CPU 时实际会跑上 cuda:0 撞车——启动后务必核对日志里的 `on cuda:N` 是你要的卡。

## 5. 当前在跑（会话结束时快照，以 gpu_queue_status.py 为准）

- 本机（2026-09-10 迁移后）暂无在跑实验；8×L40 当前全空闲，但只可用 0-3 号卡。
