# 路径约定与一致性说明

> 生成时间：2026-07-20  
> 基于 commit：`849dd7e`（恢复 `logs`/`plots` 子文件夹结构后）

## 1. 默认根目录

| 用途 | 默认值 | 对应代码 |
|---|---|---|
| 日志 / 图片 | `/data/cxm/recursion` | `batch_run.py: DEFAULT_BASE_DIR`  <br> `visualize.py: --base-dir`  <br> `prepare_rerun.py: DEFAULT_BASE_DIR` |
| 模型 | `/data/cxm/models` | `batch_run.py: DEFAULT_MODEL_BASE_DIR` |

两个目录都通过 `--base-dir` / `--model-base-dir` 参数可覆盖。

## 2. 实验目录结构

假设 `experiments.json` 的文件名为 `experiments`，则最终结构为：

```text
/data/cxm/recursion/experiments/
├── logs/
│   ├── mixed_basic_d1024l2r8_N3_e0.7_seed0.log
│   ├── mixed_basic_d1024l2r8_N3_e0.7_seed0.err
│   └── ...
└── plots/
    ├── mixed_basic_d1024l2r8_N3_curve.png
    └── ...

/data/cxm/models/experiments/
├── mixed_basic_d1024l2r8_N3_e0.7_seed0.pth
└── ...
```

## 3. 各工具读写路径对照

### 3.1 `batch_run.py`（主入口）

| 文件类型 | 写入路径 | 代码位置 |
|---|---|---|
| 训练日志 `.log` | `<base-dir>/<exp-name>/logs/<name>.log` | `run_single()` |
| 错误日志 `.err` | `<base-dir>/<exp-name>/logs/<name>.err` | `run_single()` |
| 模型 `.pth` | `<model-base-dir>/<exp-name>/<name>.pth` | `run_single()` 自动填充 `SAVE_PATH` |
| 汇总图片 `.png` | `<base-dir>/<exp-name>/plots/` | `generate_grouped_plots()` |

其中 `exp-name` 来自实验 JSON 文件名（去掉 `.json`）。

### 3.2 `core.py`（训练子进程）

| 操作 | 路径来源 | 说明 |
|---|---|---|
| 读取配置 | `config_path` 参数 | 由 `batch_run.py` 生成临时 `config_tmp_<name>.json` |
| 保存模型 | `SAVE_PATH` | 由 `batch_run.py` 自动填充为 `<model-base-dir>/<exp-name>/<name>.pth` |

`core.py` 本身不直接写日志文件，所有 stdout/stderr 由 `batch_run.py` 收集进 `.log`/`.err`。

### 3.3 `analyze_attention.py`（attention 分析）

| 操作 | 路径 |
|---|---|
| 读取模型 | 由 `batch_run.py` 传入的 `pth_path`，即 `<model-base-dir>/<exp-name>/<name>.pth` |
| 输出 | 直接打印到 stdout，由 `batch_run.py` 追加到对应 `.log` |

### 3.4 `prepare_rerun.py`（补跑配置生成）

| 操作 | 路径 |
|---|---|
| 读取日志 | 默认 `<base-dir>/<exp-name>/logs/` |
| 输出 JSON | 默认 `<exp-name>_rerun.json` |

与 `batch_run.py` 当前结构一致。

### 3.5 `visualize.py`（可视化）

| 操作 | 路径 |
|---|---|
| 读取日志 | `--all` 时默认 `<base-dir>/<exp-name>/logs/` |
| 输出图片 | `--all` 时默认 `<base-dir>/<exp-name>/plots/` |

与 `batch_run.py` 当前结构一致。

### 3.6 `fix_logs.py`（日志清理）

| 操作 | 路径 |
|---|---|
| 读取/修改日志 | 通过 `folder` 参数传入，无硬编码默认；实际使用应指向 `<base-dir>/<exp-name>/logs/` |

### 3.7 `verify_circle.py`（圆结构验证）

| 操作 | 路径 |
|---|---|
| 读取模型 | 通过 `pth_path` 参数传入 `.pth` 路径 |
| 保存图片 | 通过 `--plot` 显式指定，无默认路径 |

## 4. 一致性检查结果

| 工具 | 默认日志路径 | 默认图片路径 | 默认模型路径 | 是否一致 |
|---|---|---|---|---|
| `batch_run.py` | `<base-dir>/<exp>/logs/` | `<base-dir>/<exp>/plots/` | `<model-base-dir>/<exp>/` | ✅ 基准 |
| `prepare_rerun.py` | `<base-dir>/<exp>/logs/` | - | - | ✅ |
| `visualize.py` | `<base-dir>/<exp>/logs/` | `<base-dir>/<exp>/plots/` | - | ✅ |
| `analyze_attention.py` | 接收传入的 `.pth` | - | 接收传入的 `.pth` | ✅ |
| `fix_logs.py` | 需手动指定 | - | - | ⚠️ 注意指向 `logs/` 子文件夹 |
| `verify_circle.py` | 需手动指定 `.pth` | `--plot` 显式指定 | 需手动指定 `.pth` | ⚠️ 手动 |

## 5. 注意事项

1. **`config.json` 中的 `SAVE_PATH` 默认是 `fibonacci_transformer.pth`**  
   这是项目相对路径，仅在直接运行 `core.run_experiment()` 而不经过 `batch_run.py` 时生效。正常使用应通过 `batch_run.py` 自动填充为 `<model-base-dir>/<exp-name>/<name>.pth`。

2. **`fix_logs.py` 默认文件夹是 `logs`**  
   这是项目相对路径。对于默认的 `/data/cxm/recursion/experiments/` 实验，应运行：
   ```bash
   python src/fix_logs.py /data/cxm/recursion/experiments/logs
   ```

3. **不要混用新旧路径结构**  
   2026-07-17 的 commit `de9b336` 曾把日志直接放到 `<base-dir>/<exp>/` 下（没有 `logs/` 子文件夹），已在后续 commit 中恢复。请确保所有工具版本一致。

4. **权限问题**  
   `batch_run.py` 会调用 `os.makedirs(..., exist_ok=True)` 创建所需目录。如果运行时报 `Permission denied`，请检查 `/data/cxm/recursion` 和 `/data/cxm/models` 的写入权限。
