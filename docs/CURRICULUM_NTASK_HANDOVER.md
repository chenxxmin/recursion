# 课程学习交接文档：N-task 阶梯（单任务 → k 任务，全程带 missing）

> 面向接收方的 agent。本课程由 `src/chain_run.py` 驱动，**不需要改任何代码**，
> 所有机器相关的路径都通过命令行参数/环境变量指定。

## 0. 课程设计一览

- 第 1 步（N1）：单任务 addition（规则 (1,1)，即 `A=1,B=1`），带 missing；
- 第 k 步（Nk，k=2..5）：`mixed_ab`，规则集 = 前 k 条规则 `[[1,1],[2,3],[3,5],...]`
  （第 i 条 = `(i, 2i-1)`），带 missing，从第 k-1 步的 checkpoint 暖启动（`INIT_FROM`）。
- 超参对齐参考实验 `mixed_basic_d1024l2r8h4_P127_rules12_N2_e0.7_randmiss0.1len1_seed18318`，
  唯一改动：`TRAIN_LEN 16 → 64`（`OOD_LEN` 配套为 128）：
  `P=127, D_MODEL=1024, N_LAYER=2, N_HEAD=4, MLP_RATIO=8,
   MISSING_PROB=0.1, MISS_LEN=1, 每规则暴露比例 0.7`。
- 两个种子（17996、18318）成对推进：Nk 的 seed17996 从 N(k-1) 的 seed17996 暖启动，
  靠实验名末尾的 `_seed<N>` 后缀自动配对。
- 已写好的课程文件：`experiments/curriculum_ntask_len64_p127.json`（5 个 stage，N1..N5）。
  想加 N6+：往 `stages` 里追加同构条目，规则按 `(i, 2i-1)` 递增即可。

## 1. 拿到代码后：依赖与环境检查

```bash
git clone git@ssh.github.com:chenxxmin/recursion.git && cd recursion

# 1) Python 环境：需要 torch(CUDA 版) + matplotlib + numpy。
#    本仓库历史上用 /home/cxm/miniconda3/bin/python；你的机器上换成自己的解释器，
#    下文所有 python 均指它。
python -c "import torch, matplotlib; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

# 2) GPU 情况（共享机器必查）
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 3) 测试自检（纯 assert 脚本，无 pytest；全过即环境 OK）
for t in tests/test_*.py; do echo "== $t"; PYTHONPATH=src python "$t" | tail -1; done
```

## 2. 路径修改清单（按你的机器情况改）

代码里**不需要**改任何路径。涉及路径的只有三处，全部在启动命令上：

| 项 | 默认值 | 怎么改 |
|---|---|---|
| 日志/图根目录 | `/data/cxm/recursion` | `chain_run.py --base-dir <你的目录>` |
| 模型根目录 | `/data/cxm/models` | `chain_run.py --model-base-dir <你的目录>` |
| 编排器自身日志 | 无（stdout） | 启动命令里 shell 重定向，如 `> logs/run_curr.log 2>&1` |

自动产生、无需干预的路径：
- 每个 stage 的产物：`<base-dir>/curriculum_ntask_len64_p127_<stage>/logs|plots/`，
  模型 `<model-base-dir>/curriculum_ntask_len64_p127_<stage>/`；
- `chain_run` 每 stage 把解析好 `INIT_FROM` 的临时 batch JSON 写到仓库根 `.chain_tmp/`（自动建）；
- 训练进程写 `config_tmp_<实验名>.json` 到仓库根（自动清理）。
- GPU 相关环境变量（按需）：`BATCH_RUN_GPUS=0,1` 限定可用卡；
  `BATCH_RUN_IDLE_MEM_MB=4096` 放宽"空闲卡"判定阈值（共享机器默认 100MB 太严）。

## 3. 课程 JSON 字段说明（experiments/curriculum_ntask_len64_p127.json）

```json
{"concurrency": 2,          // 每个 stage 内并行实验数（2 = 两个种子各占一卡并行；
                            // 删掉则串行）。由 chain_run 透传给每 stage 的 batch_run
 "stages": [
  {"name": "N1",            // stage 名；产物目录 = <chain文件名>_<stage名>
   "gate": {"min_best_accuracy": 0.95},   // 门槛：该 stage 所有实验 best acc ≥ 0.95
                                          // 才放行下一 stage，否则整条链中止。可调/可删
   "experiments": [
    {"name": "curr_N1_add11_P127_tr64_miss01_seed17996",
     "task": "addition",                  // N1 单任务；N2+ 为 "mixed_ab"
     "config": {
        "P": 127, "D_MODEL": 1024, "N_LAYER": 2, "N_HEAD": 4, "MLP_RATIO": 8,
        "TRAIN_LEN": 64, "OOD_LEN": 128,
        "MISSING_PROB": 0.1, "MISS_LEN": 1,   // 在线污损：逐 batch 新鲜随机，flag/初始值不污损
        "MAX_TRAIN_HOURS": 12,                // 单实验墙钟上限（见 §5 注意事项 1）
        "RANDOM_SEED": 17996,                 // 每个实验一个种子；@prev 按 _seed<N> 后缀配对
        // 仅 N2+ 有：
        // "AB_PAIRS": [[1,1],[2,3],...],   前 k 条规则（mixed_ab 二阶规则 (a,b)）
        // "MIXED_AB_MAX_UNIQUE_RATIOS": [0.7]*k,   每规则暴露 70% 初始状态作训练集
        // "INIT_FROM": "@prev"              从上一 stage 同种子实验的 checkpoint 暖启动；
                                            // 也可写 "@N1" 引用任意更早 stage
     }}]}
 ]
}
```

机制要点（接收方理解用，不用改）：
- 配置合并顺序：`src/config.json` 的 main → task 默认段 → 实验 config。JSON 里只写与默认不同的键。
- 暖启动是**部分加载**（`experiment.py::_load_partial_checkpoint`）：按名字+shape 匹配拷贝。
  addition 与 mixed_ab（`USE_AB_TAG=false`）词表都是 P+1=128，WTE/主干全部迁移；
  mixed 的 rule_head 形状不符自动跳过、重新初始化，属预期行为。
- 早停：静态切分制，某次 eval 测试正确率 ≥ `EARLY_STOP_ACCURACY`（默认 0.99）即停并存最终模型。
- 完整配置键表见 `docs/recursion_model_and_training_settings.md`。

## 4. 启动、监控、停止、恢复

```bash
cd recursion   # 必须在仓库根运行

# 启动（setsid 脱离会话；按 §2 改两个路径）
setsid nohup python src/chain_run.py experiments/curriculum_ntask_len64_p127.json \
    --base-dir <日志根> --model-base-dir <模型根> \
    > logs/run_curriculum_ntask.log 2>&1 &

# 可选环境变量（共享机器建议）
#   BATCH_RUN_GPUS=0,1 BATCH_RUN_IDLE_MEM_MB=4096 加在命令最前面
```

监控：
```bash
python .agents/skills/recursion-experiment-runner/scripts/gpu_queue_status.py   # 逐卡：在跑什么/正确率/排队
python .agents/skills/recursion-experiment-runner/scripts/check_progress.py curriculum_ntask_len64_p127_N2 \
    --data-base <日志根>    # 查某 stage 批次（注意批次名带 _N<k> 后缀）
tail -f logs/run_curriculum_ntask.log   # 编排器日志：stage 切换、INIT_FROM 解析、gate 判定
```

停止（顺序不能反：先训练进程存盘，后编排器）：
```bash
bash .agents/skills/recursion-experiment-runner/scripts/stop_experiment.sh <实验名唯一子串>  # SIGTERM→下一 eval 点存盘退出
# 确认日志出现 [STOP] Checkpoint saved 后，再杀编排器：
pkill -f 'chain_run.py experiments/curriculum_ntask_len64_p127'
```

恢复 / 续跑：
```bash
# 从某个 stage 接着跑（前面 stage 不重跑，但会校验其 checkpoint 存在作为 @ 引用锚点）
python src/chain_run.py experiments/curriculum_ntask_len64_p127.json \
    --base-dir <日志根> --model-base-dir <模型根> --from-stage N3
```

## 5. 注意事项

1. **stage 必须"跑完"才算数**：`chain_run` 在每个 stage 后校验 `<model根>/<批次>/<实验名>.pth`
   存在。这个最终模型只在训练正常结束（早停或跑满 EPOCHS）时写出；
   若撞上 `MAX_TRAIN_HOURS` 超时（存的是 `_resume.pth`），链会中止。
   恢复办法：用多轮续跑脚本把该 stage 跑完（`bash scripts/chain_rounds.sh
   .chain_tmp/curriculum_ntask_len64_p127_<stage>.json <gpus> 3`，会自动接 `_latest`
   checkpoint），再 `--from-stage` 续链。`MAX_TRAIN_HOURS` 可按卡时改大或删掉。
2. **显存**：d1024/l2/bs512（默认）在 46GB 卡上很宽松；若换更大模型 OOM，
   设 `BATCH_SIZE=256 + GRAD_ACCUM_STEPS=2`。
3. **精度**：默认 fp32+TF32，结论性实验保持默认；不要开 `USE_AMP`（bf16 只能粗筛，
   且暖启动续跑会被 bf16 摧毁，见 `docs/AMP_PRECISION_NOTES.md`）。
4. **种子配对是名字后缀匹配**：新增种子/阶段时保持 `_seed<N>` 结尾，且每个 stage
   内同一后缀只能有一个实验，否则 `@prev` 解析报错。
5. 仓库里另有 `src/curriculum_chain_Ntask.py`（旧的 N=2→5 专用调度器，硬编码路径与卡号），
   与本课程**无关**，不要用它；`scripts/chain_rounds.sh` 是多轮续跑工具，也不是课程。
6. 结果口径：每个 stage 结束自动画分组图到 `<base>/<批次>/plots/`；
   结果库同步流程见 `.agents/skills/recursion-experiment-runner/SKILL.md` §7（按需）。
