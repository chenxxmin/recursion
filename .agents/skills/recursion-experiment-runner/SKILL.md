---
name: recursion-experiment-runner
description: 在本仓库（/mnt/workspace/hujiachen/recursion，模运算递推/grokking 实验）批量运行训练实验的完整流程。当用户给出实验类型和超参要求跑实验时使用。覆盖：生成配置、找空卡、等待并固定 N 张物理卡、用 batch_run.py/chain_rounds.sh 启动、日志与 checkpoint 落盘、查进度、关停实验/旧等待器、定时、汇总与结果库同步。触发词示例：跑一批实验/找空卡/用4卡并行/限制只用某几张卡/等待空卡启动/替换旧启动脚本/排个队列/查进度/停止实验/汇总结果/同步结果库。
---

# Recursion 实验运行器

仓库：`/mnt/workspace/hujiachen/recursion`（代码，git main）；
结果库：`/mnt/workspace/hujiachen/recursion_results`（独立 git 库，master，remote: recursion_results，云同步用）。

## 1. 生成实验配置 JSON

写到 `experiments/`。格式：

```json
{"concurrency": 2,
 "experiments": [{"name": "...", "task": "action", "config": {...覆盖键...}}]}
```

- 合并顺序：`src/config.json` 的 main → 场景默认段（场景段→旧 task 同名段去重）→ 实验 config 覆盖。只写与默认不同的键。
- 配置键完整表：读 `docs/recursion_model_and_training_settings.md`（含 MISS_LEN/MISSING_PROB/MAX_TRAIN_HOURS/RESUME_FROM/GRAD_ACCUM_STEPS 等）。
- 参考模板：`experiments/` 下现有 JSON。

**命名约定**（从现有实验沿用，保证日志可分组成图）：
- 新体系（2026-10-08 场景统一后）：`{scenario}_o{order}_...`（如 `single_o3_d1024l2_P127_tr64_miss0.1len1_seed17996`），seed 后缀规则不变（课程链 `@prev` 配对依赖）；旧 task 名（tribonacci/mixed_ab/action_trib 等）仍可跑（归一化层兼容），但新实验一律用场景名。
- v1 制度：`{task}_d{D}l{L}r{R}h{H}_P{P}_tr{T}ood{O}_{N规则}_randmiss{prob}len{len}_seed{S}`
- v2 命名（30万训练集/新鲜测试）：`action_v2_d{D}l{L}r{R}h{H}_P127_tr127_{N}_miss{prob}len{len}_seed{S}`（只是命名风格，输出目录与其他实验相同）

### 1.1 多长度连续填空 `BLANK_LENS`（2026-11-05）

`TASK_VARIANT='fill_blank'` 时可用 `BLANK_LENS` 配置连续挖空长度。默认 `[1]`，旧 JSON 无需修改；集合必须非空、元素为不重复正整数，且最大长度不能超过最短实际窗口（不定长时看 `SEQ_LEN_MIN`）。示例（`len3to5` 是窗口长度，`blank1to2` 是空缺长度，命名不要混淆）：

```json
{"concurrency": 1,
 "experiments": [{
   "name": "single_o2_fillblank_a25b56_d512l4r8h4_P127_len3to5_blank1to2_seed42",
   "task": "single",
   "config": {
     "TASK_VARIANT": "fill_blank", "ORDER": 2, "RULES": [[25, 56]], "P": 127,
     "TRAIN_LEN": 5, "OOD_LEN": 5, "SEQ_LEN_MIN": 3, "SEQ_LEN_MAX": 5,
     "BLANK_LENS": [1, 2]
   }
 }]}
```

- 采样时先均匀抽 `l`，再均匀抽连续段起点 `k∈[0,L-l]`，整段只保留一个 BLANK。最终布局是 `x=[V,EOS(l),H[:-1]]`、`target_seq=[B,EOS(l),H]`，其中 `H` 为被挖值、`V` 为压缩窗口、`EOS(l)=P+1+(l-1)`。EOS(l) 在输入中直接给出长度，EOS 槽立即输出首值；**不要使用已废弃的独立 `LEN(l)` 输出方案**。
- `EOS(1)` 就是旧 EOS；`[1]` 完整兼容旧实验，混合 `[1,2]` 中的 `l=1` 样本也保持旧布局。扩展词表为 `P+1+max(BLANK_LENS)`，block 按 `max_len+1`。
- `FILL_BLANK_ALL_POSITIONS` / `TEST_ALL_POSITIONS` 在多长度时枚举 `(seq,k,l)`，每窗口样本数为 `sum_l(L-l+1)`；写 JSON 时注意评估规模。PosProbe 输出 `val_l*`/`valR_l*`，没有长度准确率。
- **课程链暖启动限制**：从 `[1]` 到含 `l>1` 的配置会扩大词表。现有 `INIT_FROM` 只复制名称和形状都匹配的权重，形状变化的 `wte/lm_head` 会跳过并重新初始化；`RESUME_FROM` 不能跨词表使用。若要求完整保留旧嵌入，必须先实现并验证嵌入扩容/复制，不能在参数表中写成“完整暖启动”。相同 `[1]` 布局的旧 checkpoint 可直接加载。

## 2. 启动

```bash
# 单批（日志→<base>/<批次名>/logs/，模型→<model-base>/<批次名>/，结束自动画图到 plots/）
setsid nohup python -u src/batch_run.py experiments/<批次>.json > logs/run_<批次>.log 2>&1 < /dev/null &

# 长实验用链式多轮（每轮自动续 checkpoint、跳过已完成、日志追加）
setsid nohup bash scripts/chain_rounds.sh experiments/<批次>.json <gpus> <rounds> > logs/run_<批次>.log 2>&1 &
```

- GPU 分配：`BATCH_RUN_GPUS=0,1`（白名单）；`BATCH_RUN_IDLE_MEM_MB=4096`（共享卡时放宽空闲阈值，默认 100MB）。空闲 GPU 用 `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` 查。
- 默认目录（本机）：`/mnt/workspace/hujiachen/recursion_results` + `/mnt/workspace/hujiachen/models`。本机 v2 隔离制度用 `--base-dir/--model-base-dir`（batch_run）或 `CHAIN_DATA_BASE/CHAIN_MODEL_BASE`（chain_rounds.sh），见 `reports/v2/README.md`。（云端机器 2026-09-11 起已废弃 v2 目录，一律用其默认 `/data/cxm/recursion` + `/data/cxm/models`。）
- 并发上限 = min(concurrency, 空闲卡数)，一卡一实验。

### 2.1 找空卡、钉 N 卡并启动（2026-11-04 固化）

仅对已确认配置的实验执行。使用 `scripts/gpu_batch_launch.py`，默认 dry-run；它要求显式输出目录，避免误写默认的 `/mnt/workspace`：

```bash
CFG=experiments/<批次>.json
N=4
GPUS=2,3,4,5   # 有序候选池，也可以写 0,1,2,3,4,5,6,7
python .agents/skills/recursion-experiment-runner/scripts/gpu_batch_launch.py \
  "$CFG" --required "$N" --gpus "$GPUS" \
  --base-dir /data/cxm/recursion --model-base-dir /data/cxm/models
```

空卡必须同时满足：显存 ≤100 MiB、利用率=0、没有 compute 进程。候选池空闲数达到 N 后，工具按候选顺序选择**恰好 N 张**，用 `BATCH_RUN_GPUS` 钉卡，并设置 `BATCH_RUN_REQUIRE_GPU=1` 禁止 CPU/cuda:0 回退。JSON 的 `concurrency` 大于 N 不会扩卡；实际 worker 数仍受钉卡数限制。

确认 dry-run 后后台启动控制器（控制器先等卡，再启动 batch；`--execute` 不可省略）：

```bash
LOG=logs/run_<批次>_${N}gpu_controller_$(date +%Y%m%d_%H%M%S).log
setsid nohup python .agents/skills/recursion-experiment-runner/scripts/gpu_batch_launch.py \
  "$CFG" --required "$N" --gpus "$GPUS" \
  --base-dir /data/cxm/recursion --model-base-dir /data/cxm/models \
  --execute > "$LOG" 2>&1 < /dev/null &
echo $! > logs/run_<批次>_${N}gpu_controller.pid
```

控制器日志给出 `[GPU] selected`、`[BATCH_PID]` 和独立 batch 日志路径。启动后必须继续核验，不能把“等待器存在”当成“训练已启动”：

1. batch 日志的 `GPU whitelist` 只能包含所选卡，`Effective concurrency` 必须等于 N；
2. `core.py` 进程和 `nvidia-smi` 显存/利用率与卡号一致；
3. 至少看到首个 eval/`Start training...`，排除配置归一化错误；多阶段实验还要检查 gate，batch rc=0 不等于 gate 通过。

## 3. 关键运行行为（影响决策）

- **6h 存盘**：`MAX_TRAIN_HOURS` 到点后写 `<model>_resume.pth`（含优化器/调度器/RNG），退出码 42，batch 标记 timed out。
- **滚动 checkpoint**：每个 eval 点覆盖写 `<model>_latest.pth`（完整状态可续跑），best 更新时写 `<model>_best.pth`（纯权重）。意外死亡也能保住最佳权重和最新进度（2026-09-08 起）。
- **手动关停保存**：对 core.py 进程 `kill <pid>`（SIGTERM）→ 下一个 eval 点存盘退出（码 42）。不要 kill -9（丢进度）。**教训：永远先等存盘/用 SIGTERM，不要在存盘点前强杀。**
- **精度**：恒为 fp32+TF32（AMP 代码路径已于 2026-10-08 随场景统一移除，`USE_AMP` 显式启用会报错）。历史教训仍有效：bf16 只能从零粗筛取阳性，续跑接近收敛的模型会摧毁它；结论性实验必须 fp32。细节见 `docs/AMP_PRECISION_NOTES.md`。
- **曲线密度**：要画学习曲线就设 `EVAL_INTERVAL=1`（评估便宜）；默认 20 太稀。
- 显存：bs2048 时 l2 已贴边（44GB 卡），l4 必须 `BATCH_SIZE=1024 + GRAD_ACCUM_STEPS=2`（等效 2048）。OOM 即照此处理。

## 4. 查进度

```bash
python .agents/skills/recursion-experiment-runner/scripts/check_progress.py <批次名> [--data-base /mnt/workspace/hujiachen/recursion_results_v2]
```

输出每实验 `[完成/运行/存盘待续] epoch best%`。注意日志只在 eval 点写：慢实验（分钟级 epoch）几十分钟没新行是正常的，用 GPU 利用率确认存活。

## 5. 关停

- **定时关停**：配 `MAX_TRAIN_HOURS`；或给 chain_rounds.sh 限定轮数。
- **手动关停且保存**：**先对训练进程（core.py）发 SIGTERM**，等日志出现 `[STOP] Checkpoint saved`（下一个 eval 点，EVAL_INTERVAL=1 时秒级），**再杀编排器**（chain_rounds.sh / batch_run）。顺序不能反！
- **手动关停整条链**：先 `kill` 各个 `core.py config_tmp_<实验名>`（SIGTERM 存盘），确认存盘后 `pkill -f 'chain_rounds.sh experiments/<批次>'`。pkill 模式别写成能匹配到自己命令行的形式（会误杀 shell，用 `pgrep -af '[c]ore.py ...'` 括号写法或先核对 PID 再 kill）。
- 2026-09-08 教训：旧版 batch_run 用管道转发 core.py 的 stdout，先杀编排器会让训练进程下次 print 时 BrokenPipeError 静默死掉、丢 checkpoint（trib/tetra 跑了 12h 的进度因此丢失）。已修复为直接重定向到文件（编排器死亡不再影响训练进程），但仍建议按上面的顺序操作。
- **替换旧的等卡等待器**：先 `ps -o pid,ppid,pgid,sid,stat,args -p <PIDs>` 并检查子进程，确认只有 bash/python/tee、没有 batch/core 训练后，再 `kill -TERM -- -<PGID>` 关闭整个独立等待进程组。不要凭名字 `pkill -f`（会自匹配/误杀）；若已有训练，等待器可单独停，但 detached batch 会继续且后续 gate/阶段不会执行，训练本身仍须用 stop_experiment.sh 存盘停止。
- **取消排队实验**：batch_run 把队列读进内存，改 JSON 不影响在跑的编排器。正确做法：直接杀编排器（训练进程因文件重定向不受影响，会继续跑完并正常存最终模型），不要让训练进程先退出——否则编排器会在 watcher 轮询间隙（秒级）内抢跑下一个实验（两次踩坑）。缺 plots 时用 `python -c "import batch_run; batch_run.generate_grouped_plots('<logs>', '<plots>')"` 手动补画。

## 6. 五个常用技能（2026-11-04 更新，脚本在 scripts/ 下）

**技能 1 — 逐卡队列状态**（用户说“每张卡队列/在跑什么”）：
```bash
python .agents/skills/recursion-experiment-runner/scripts/gpu_queue_status.py
```
每张卡一行：在跑实验、累计 epoch（含续跑轮）、当前/best 正确率、已跑时长、定时（含 timer.sh 临时定时的剩余时间）；排队列（含定时）。排队语义：batch_run=JSON 里当前实验之后的条目；chain_rounds=未完成的实验（最后一轮则不再排）。

**技能 2 — 全量实验汇总**（用户说“汇总所有实验”）：
```bash
python .agents/skills/recursion-experiment-runner/scripts/experiment_summary.py [--since-days 3] [--match 子串]
```
按批次分组列出每个实验：状态（RUN/DONE/STOP）、累计 epoch、best 正确率、每 epoch 耗时（含启动开销，短任务会被摊高）。

**技能 3 — 下一个存档点退出某条实验**（用户说“停止/退出某实验”）：
```bash
bash .agents/skills/recursion-experiment-runner/scripts/stop_experiment.sh <实验名唯一子串>
```
对训练进程发 SIGTERM → 下一个 eval 点存 `_resume.pth` 退出（码 42）。子串必须恰好匹配一个进程，否则脚本拒绝执行。编排器不动，队列照常前进。

**技能 4 — 临时定时**（用户说“给某实验设 N 小时定时/取消定时”）：
```bash
bash .agents/skills/recursion-experiment-runner/scripts/timer.sh set <实验名唯一子串> <小时>
bash .agents/skills/recursion-experiment-runner/scripts/timer.sh cancel <子串>
bash .agents/skills/recursion-experiment-runner/scripts/timer.sh list
```
原理：后台 watcher（setsid 分离，状态文件在 logs/timers/）到点后按技能 3 发 SIGTERM。只适用于**正在运行**的实验；未开始的实验直接在 JSON 配 MAX_TRAIN_HOURS。

**技能 5 — 找空卡并钉 N 卡启动**（用户说“找空卡/用 N 卡/只用这几张卡/等待空卡启动”）：
```bash
python .agents/skills/recursion-experiment-runner/scripts/gpu_batch_launch.py \
  experiments/<批次>.json --required <N> --gpus <有序候选池> \
  --base-dir /data/cxm/recursion --model-base-dir /data/cxm/models [--execute]
```
默认 dry-run；实际后台启动、日志核验和替换旧等待器的完整流程见 §2.1 与 §5。


## 7. 汇总与上传

- 结果汇总到状态文档（本机 v1 口径：`reports/ACTION_MISSLEN_STATUS.md`，用 `scripts/update_status_tables.py` 重建表格；v2：`reports/v2/ACTION_V2_STATUS.md`，用 `scripts/update_v2_status.py`）。
- 上传结果库（本机 `/mnt/workspace/hujiachen/recursion_results`，master 分支）：把批次 logs/plots 复制进 `action_v2/<批次名>/`（v2 用 `v2/` 前缀子目录），checkpoint 放 `checkpoints/` 子目录（**不要叫 models/，会被 .gitignore 忽略**），`git add` + commit + `git push origin HEAD`。
- **GitHub 单文件 100MB 限制**：大 checkpoint 先提取纯权重转 bf16（实证无损，见 AMP_PRECISION_NOTES.md §2.6），超 100MB 的完整 resume ckpt 只留本地。
- 代码库推送用 main 分支；两个库远端有新提交时先 `git pull --no-rebase` 合并（历史上均无冲突）。
