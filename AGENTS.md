# AGENTS.md — recursion 实验仓库交接

模运算递推/grokking 实验仓库。代码在 `/home/cxm/recursion`（git main），结果库在 `/data/cxm/recursion`（独立 git master，云同步）。

## 0. 必读的现有文档

- `.agents/skills/recursion-experiment-runner/SKILL.md`：实验运行全流程 + 四个工具脚本（逐卡队列状态、全量汇总、存档点停止、临时定时）。**跑实验前必读**。
- `docs/recursion_model_and_training_settings.md`：所有配置键的完整说明（TRAIN_LEN/MISSING_PROB/NUM_MASK/MAX_TRAIN_HOURS/...），改配置语义后必须同步它。
- `docs/AMP_PRECISION_NOTES.md`：精度教训——结论性实验必须 fp32（默认），bf16 只能粗筛。

## 1. 代码结构（src/）

- `core.py`：薄入口（re-export），实际逻辑在 `experiment.py`（run_experiment 主流程 + 三个 prepare）。
- `experiment.py`：`_prepare_single_recurrence`（addition/multiplication/tribonacci/tetranacci/nonlinear*）、`_prepare_mixed_recurrence`（mixed_ab/mixed_abc）、`_prepare_action`（action）。
- `rules.py`：递推规则。`LinearRecurrenceRule`（任意阶任意系数的通用入口）、`single_rule_from_task`（task 名 → (init_len, next_fn, name)）。
- `datasets.py`：RecurrenceDataset（全状态枚举+循环遍历，或 STATE_SPACE_CAP 抽样）、MixedRecurrenceDataset、ActionDataset、所有 collate。
- `training.py`：训练/评估循环、早停、SIGTERM 优雅退出、checkpoint。
- `batch_run.py`：批量编排（读 experiments/*.json，合并配置，一卡一实验）；`chain_rounds.sh` + `prepare_chain_round.py`：多轮续跑。
- 测试：`tests/test_*.py` 是纯 assert 脚本，**没有 pytest**；运行 `/home/cxm/miniconda3/bin/python tests/test_xxx.py`（test_sample_gen.py 需要 `PYTHONPATH=src`）。

## 2. 配置合并与实验 JSON

- 合并顺序：`src/config.json` 的 main → task 默认段 → 实验 config 覆盖。实验 JSON 只写与默认不同的键。
- 命名约定见 SKILL.md §1；v2 制度（30万样本/fresh 256 测试）写 `experiments/v2/` 并用 `--base-dir /data/cxm/recursion_v2 --model-base-dir /data/cxm/models_v2` 启动。

## 3. 最近的语义改动（2026-09-08/09，改代码时注意一致性）

1. **污损全面改为在线模式**：数据集只存干净样本，`make_missing_collate` / `make_mixed_missing_collate` / `make_action_missing_collate` 逐 batch 新鲜随机污损（每 epoch 缺失位置都不同，train/test 同样）。旧的"生成时固定污损"已删除，无开关。
2. **早停分两制**：静态切分制（无 fresh test）测试 ≥ `EARLY_STOP_ACCURACY`(0.99) 直接停；新鲜测试制（`FRESH_TEST_PER_EVAL`）最近 10 个 eval 均值 >99% 才停。`EARLY_STOP_EXTRA_EPOCHS` 已废弃。
3. **滚动 checkpoint**：每个 eval 点覆盖写 `<model>_latest.pth`（完整状态可 RESUME_FROM），best 更新时写 `<model>_best.pth`（纯权重）。
4. **tetranacci task**：4 阶递推（A/B/C/D 键，init_len=4）。**P⁴ 状态空间必须配 `STATE_SPACE_CAP`**（否则枚举爆炸），抽样路径同时用于数据集和 final eval。
5. **单规则任务支持** `NUM_TRAIN_SAMPLES` 显式覆盖 和 `FRESH_TEST_PER_EVAL`（每 eval 从全状态空间抽 NUM_TEST_SAMPLES 个全新状态生成 OOD_LEN 窗口）。
6. **batch_run 直接文件重定向**（不再管道转发）：杀编排器不会误杀训练进程。

## 4. 本机环境的坑（反复踩过）

- **PATH 里有别人的 venv**（`/home/cxm/Dan/.venv`，CPU 版 torch 且无 matplotlib）：所有启动/测试必须用**绝对路径** `/home/cxm/miniconda3/bin/python`；chain_rounds.sh 内部用裸 `python`，调它前 `export PATH=/home/cxm/miniconda3/bin:$PATH`。
- **pgrep/pkill 自匹配**：模式会匹配到自己 shell 的命令行，用括号写法 `[c]ore.py` 或先核对 PID。
- **共享 GPU 机器**（8×46GB，有 liping/yyx/asleepx 等用户）：启动时加 `BATCH_RUN_IDLE_MEM_MB=4096`（或更高）放宽空闲阈值；时刻先 `nvidia-smi` 看占用再选卡。
- **停止实验**：用 skill 的 stop_experiment.sh（SIGTERM → 下一 eval 点存盘），或 timer.sh 设定时。不要 kill -9 正在跑且进度有用的实验。
- GPU 检测回退到 CPU 时实际会跑上 cuda:0 撞车——启动后务必核对日志里的 `on cuda:N` 是你要的卡。

## 5. 当前在跑（会话结束时快照，以 gpu_queue_status.py 为准）

- 卡 1/5：mixed N2 l4 len64 online（len1→len2，各 6h）；卡 3：addition len64 online 系列（len1→8 × 2 种子）；卡 7：mixed N4 l8（12h）。
- 关键在途结论：在线污损使 addition len3 从 87%→99%+；64 长度让 mixed+miss 变难一个量级；tetra 只有等比系数 (1,2,4,8) 能 grok。
