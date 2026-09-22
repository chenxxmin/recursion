# 实验参数确认、JSON 生成与跨机器运行 pipeline

适用于本仓库的单规则、mixed、action 等实验。用户先提出实验与参数；助手负责解析、出表、等待确认、制作可分发运行包。正式实验由用户分发并启动。

## 1. 对话流程

1. **接收要求**：任务/系数、模型、样本机制、样本量、训练超参、seed、从头或续跑、实验数量、并发数、每条定时、参考配置。未提到的值从明确参考或当前代码默认推导，注明来源。
2. **先打印完整参数表**：既展示不同实验之间的差异，也展示全部共同参数。每个值注明用户指定、参考实验、任务默认、main 默认或代码推导。打印实际样本数、有效 mask、系数方向、初始化方式、定时语义和早停条件。待澄清项单独列出。
3. **等待用户明确确认**：用户确认当前表格版本后，才允许生成可执行实验 JSON 和运行包。参数、存档或代码改变后重新出表。不能自行把“生成审阅表”当作用户确认。
4. **生成与冻结**：写入完整有效配置，打包当前代码快照、所需存档、运行脚本、参数表、依赖说明和文件校验清单。
5. **交付**：给出压缩包、参数表、启动/检查/停止命令、输出位置和定时行为。默认不在本机启动实验。
6. **目标机器运行**：用户选择本机 GPU、输出目录及实验子集，脚本检查依赖、文件一致性、GPU 空闲和重复运行后启动。每张 GPU 同时一个实验。

用户说“确认”即可，助手在构建命令中填写对应审阅版本 ID；这是版本校验，不是要求用户手抄哈希。

## 2. 每次确认必须展示的参数

| 类别 | 必须展示 |
| --- | --- |
| 实验身份 | 实验名、task、递推阶数与完整系数/规则列表、系数方向、P |
| 模型 | d_model、层数、头数、MLP ratio、位置编码、dropout |
| 数据 | 数据模式、训练样本总数及每规则数量、暴露率、train_len、ood_len、NUM_MASK 的实际作用、污损参数 |
| 测试 | 静态/fresh test、样本量（总量或每规则）、评估间隔、全部早停条件 |
| 优化 | AdamW、LR、WD、clip/不裁剪、batch、梯度累积、每轮重洗牌 |
| 数值与复现 | seed、fp32/AMP、TF32、初始化方式、参考配置、代码快照 |
| 续跑 | 存档来源/大小、下一 epoch、实际当前 LR、scheduler base LR 与 T_max |
| 运行 | 实验数、最大并发、每条实验时限、计时起点、到时保存方式、日志与模型路径 |

seed 同时影响数据采样和初始化；相同 seed 的两份运行属于同条件复现，不是不同随机种子的稳健性实验。

## 3. 工具入口

- `scripts/experiment_pipeline.py`：只负责审阅和打包，绝不启动训练。
- `scripts/experiment_package_runner.py`：随每个运行包分发，执行检查、运行、状态查询及存档停止。
- 现有 `src/batch_run.py` 仍是训练入口；运行包使用其当前代码快照并显式传入目标机输出目录。

请求草稿可从 stdin 传入，不必在确认前写任何可执行实验 JSON。草稿结构：

```json
{
  "batch_name": "tetra_example",
  "concurrency": 2,
  "common": {
    "P": 127, "A": 3, "B": 6, "C": 1, "D": 4,
    "D_MODEL": 4096, "N_LAYER": 2, "N_HEAD": 16, "MLP_RATIO": 4,
    "NUM_TRAIN_SAMPLES": 300000, "NUM_MASK": 3,
    "TRAIN_LEN": 16, "OOD_LEN": 32, "BATCH_SIZE": 512,
    "DATA_MODE": "sampled_fresh_test", "STATE_SPACE_CAP": 2048383,
    "NUM_TEST_SAMPLES": 256,
    "LR": 0.00002, "WEIGHT_DECAY": 0, "GRAD_CLIP_NORM": 5,
    "RESHUFFLE_EACH_EPOCH": true,
    "USE_AMP": false, "ALLOW_TF32": true,
    "EPOCHS": 6000, "MAX_TRAIN_HOURS": 12, "EVAL_INTERVAL": 1,
    "INIT_FROM": null, "RESUME_FROM": null
  },
  "experiments": [
    {"name": "tetra_seed0", "task": "tetranacci", "config": {"RANDOM_SEED": 0}},
    {"name": "tetra_seed1", "task": "tetranacci", "config": {"RANDOM_SEED": 1}}
  ],
  "unresolved": []
}
```

这个例子仅说明输入格式，不代表已经批准的新实验。也可在某条实验中添加：
`"reference": {"path": "experiments/已有批次.json", "name": "其中一条实验名"}`。
多条目参考文件必须指定 name；task 必须一致。

合并顺序：代码 fallback → main 默认 → task 默认 → 参考 config → 用户公共参数 → 用户逐实验参数。
最终可执行 JSON 会展开所有有效值；目标机器不会重新引用旧仓库默认值。SAVE_PATH 由目标输出目录生成。
mixed 的 NUM_TRAIN_SAMPLES 标量表示每规则数量；审阅表另外展示逐规则和总样本量、实际测试样本量及测试序列长度。
静态测试集大小由状态空间减去训练配额决定，不能把 NUM_TEST_SAMPLES 默认值误读为静态测试集大小。

```bash
# 只输出审阅表；--request - 表示从 stdin 接收请求草稿
python3 scripts/experiment_pipeline.py review --request - --out reviews/tetra_v1.md

# 用户明确确认表格以后，助手使用该表打印的审阅 ID 打包
python3 scripts/experiment_pipeline.py build \
  --review reviews/tetra_v1.md --confirm REVIEW_ID \
  --out bundles/tetra_confirmed
```

第一步只产生 Markdown 审阅文件。第二步产生目录和同名 `.tar.gz`。
工具会拒绝错误确认 ID、人工修改过的表、未澄清参数、审阅后变化的代码或存档、已有同名产物。
Markdown 末尾保存机器可读审阅快照，用于保证表格与生成配置一致。

## 4. 运行包与目标机命令

```text
tetra_confirmed/
  PARAMETERS.md       # 已确认的完整参数与来源
  experiments.json    # 已展开的实验 JSON
  run.sh              # 包含每个实验的 TIMERS_JSON
  runner.py           # 可移植的运行控制器
  manifest.json       # 确认版本、定时、文件校验
  requirements.txt
  README.md
  code/src/           # 当前实际训练代码（包含未提交但已确认的修改）
  checkpoints/        # 若 INIT_FROM/RESUME_FROM 需要存档，自动附带
```

目标机器需要 Linux、Python 3.10+、与该机器 CUDA 匹配的 PyTorch，以及 requirements.txt 中的依赖。
不自动安装软件，不依赖打包机器的绝对目录。可以设置 `EXPERIMENT_PYTHON=/path/to/python`。

```bash
tar -xzf tetra_confirmed.tar.gz
cd tetra_confirmed
bash run.sh check
bash run.sh run --gpus 0,1 --output /data/runs/tetra_run1 --background
bash run.sh status --output /data/runs/tetra_run1
bash run.sh stop --output /data/runs/tetra_run1
```

分发同一个包到多台机器时，分别指定 `--only tetra_seed0`、`--only tetra_seed1` 等实验子集。
默认运行包内全部实验，最大并发受确认的 concurrency、指定 GPU 数和实验数限制。
GPU 参数使用物理编号，并尊重已有 CUDA_VISIBLE_DEVICES 分配。在本打包机器上仍只允许 GPU 0–3。
GPU 缺失/占用时直接失败，不会悄悄落到 CPU 或 cuda:0；`--cpu` 仅供显式 CPU 小规模检查。
后台启动只代表控制器已启动，需查看 status/controller.log 确认训练已进入步骤。

输出目录必须全新。它包含 `controller.log`（后台时）、`state.json`、
`results/<实验名>/logs`、`results/<实验名>/plots`、
`models/<实验名>` 和每条实验的独立工作目录。
机器环境与调度资源可以调整；训练参数变更则重新走审阅确认流程。

已批准实验需要等待空闲卡时，使用后台排队模式：

```bash
bash run.sh run --gpus 2,3,0,1 --output /data/runs/tetra_queue \
  --wait-for-idle --poll-seconds 300 --background
```

GPU 列表在此模式下是候选池，按列表顺序选择；最大同时运行数仍受 concurrency 限制。
启动时立即检查一次，此后每 300 秒检查；只空出一张卡就先启动一条实验。
空闲要求显存低于 100 MiB、利用率为 0、没有 compute 进程，并能取得该 GPU 的运行锁。
查询失败或运行锁仍被占用时继续等待。state.json 记录检查次数、上次/下次检查时间和 GPU 快照。
每条只启动一次，运行失败不会自动重跑；等待时间不计入 MAX_TRAIN_HOURS。
原有 status / stop 命令同样适用；stop 取消未启动任务并让已启动任务在存档点退出。

## 5. 定时和停止的准确语义

- 每条实验独立使用已确认的 `MAX_TRAIN_HOURS`。null 表示不限；0 表示首个 eval 点存档退出，主要用于测试。
- 定时同时写入 JSON 和 `run.sh` 的 `TIMERS_JSON`；启动前检查一致性，训练引擎执行定时。
- 从进入训练循环开始计时，排队和初始化不计入；每 EVAL_INTERVAL 个 epoch 的 eval 点检查。
- 到时先保存完整模型、Adam、scheduler、RNG、下一 epoch，再退出；可能超过名义时限一个评估间隔及存盘时间。
- 不把 EPOCHS 改成短跑 epoch 数，不重置 cosine horizon；不使用 sleep 后 SIGKILL 的方式。
- 原有早停条件仍生效；时限是评估点检查的训练预算，不是精确到秒的总进程硬截止时间。
- 正常完成时沿用原代码的 final generation（除非 SKIP_FINAL_GENERATION_TEST=true）及 attention 后处理，这部分不计入训练预算；定时/手动存档退出则跳过。
- `stop` 先取消未开始的任务，对正在训练的子进程发 SIGTERM，等待下一个 eval 点保存；不会自动排新轮次。
- 退出码 42 在控制器中记录为正常的定时/手动存档退出，区别于失败；状态文件保留真实 trainer return code。
- 初始数据准备阶段尚无训练进度时收到停止请求，可能没有新存档。实际保存情况见 state.json 与训练日志。

## 6. 续跑及结果口径

RESUME_FROM 必须是本机可读取的完整存档。审阅时用 CPU 核对来源 epoch、实际 LR/WD、scheduler horizon；
若配置 LR 与存档 base LR 不一致，先显式生成改变 LR 的新存档，再重新出表确认，避免 JSON 的 LR 被存档覆盖。
INIT_FROM 则是仅初始化匹配权重，优化器和 scheduler 全新开始，表中明确区分。

存档会按内容摘要放入包内 checkpoints，JSON 使用包内相对路径，运行时转换到解压位置；
因此包移到其他路径/机器后仍可运行。大存档会增加压缩包体积，审阅表会显示总大小。

在线 train loss/accuracy 只用于跟踪训练过程。完整训练集固定权重复评、额外诊断、后续续跑轮次若需要，应作为明确要求另行确认配置；运行包不会自动增加这些任务。

## 7. 验证

```bash
python3 tests/test_experiment_pipeline.py
```

CPU 集成验证覆盖：先审阅后打包、确认版本/配置与代码漂移、定时一致性、包含空格的目录、
实际数据集样本量核对、新鲜训练超时保存、打包续训存档后迁移路径、无定时完成与子集选择、后台启动/存档停止/取消待运行实验、GPU 不可用时禁止静默回退。


## 8. 固定卡数的跨机 action 运行包

已确认请求可增加 delivery 字段：required_gpus、output_root、clean_eval_samples、
clean_eval_seed。required_gpus 必须与 concurrency 相等。此模式使用
scripts/remote_experiment_runner.py，启动前一次性取得全部预设空闲卡；
不足则整批退出，不自动降低卡数，也不启动后台等待器。终端打印卡号、型号和任务。
后台控制器通过 pass_fds 继承已锁定文件描述符，避免分离时释放 GPU 锁。

启动：bash run.sh run --background（默认自动选卡）。
候选池：--gpus 0,2,4,6，仍要满足预设空闲卡数。
进度/停止：bash run.sh status 或 stop --batch <批次目录>。
续跑：bash run.sh resume --resume-batch <已结束批次目录> --background。
仅恢复有完整存档的未完成任务；显式 --only 可选择已完成任务。

所有输出放在 output_root；每个实验建立独立目录，按 configs/logs/checkpoints/plots/metrics 分类。
批次控制目录另存 configs/logs/state。续跑建立新目录，保留旧文件和历史 best 权重。
每组独立计时；正常完成/早停也保存完整 resume。结束后同卡执行 clean 评估、
记录 total acc、生成曲线，之后才接续下一个任务。评估不计入训练时限。

验证：python3 tests/test_remote_experiment_runner.py（CPU 与模拟 GPU 状态，无正式 GPU 训练）。
