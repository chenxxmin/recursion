# 画图流程：从 log 到 PNG（src/visualize.py）

## 1. 数据来源：log 格式约定

训练日志里三类被解析的行（正则匹配，见 `parse_log`）：

```
Epoch  20 | Train: Loss=0.0198 Acc=99.9% | Test: Acc=96.7% | Best=96.7%(@20)
  from x3: 0.90 0.94 0.96 ...        # 该行所属 epoch 的逐位置 acc（from x<起始位置>）
  per-rule acc: 0.99 0.98 0.94       # 该行所属 epoch 的逐规则 acc（多任务时才有）
```

- `Epoch ... | Train: Loss=.. Acc=..% | Test: Acc=..% | Best=..%(@N)` → epochs / train_loss / train_acc / test_acc / best_acc / best_epoch
- `from x<k>: <float>...` → 当前 epoch 的 per_pos = (起始位置 k, acc 列表)
- `per-rule acc: <float>...` → 当前 epoch 的 per_rule 列表
- per_pos / per_rule 通过"读到下一个 Epoch 行时 flush 上一段"的方式归属到对应 epoch
- 百分比字符串自动归一化到 [0,1]（`parse_percent`）

注意：拼接续跑日志时（`=== RESUME ROUND ===` 分隔），Epoch 行号延续即可被正确解析，中间的非 Epoch 行会被跳过。

## 2. 分组：setting × seed

`extract_setting_and_seed`：文件名 `xxx_seed<digits>` → setting=`xxx`、seed=`<digits>`；无 seed 后缀则自成一组。

`collect_log_groups(names, log_dir)`：
1. names 里每个名字先按完整实验名找 `<log_dir>/<name>.log`
2. 找不到且有 seed 后缀 → 跳过（兼容部分运行的批次）
3. 找不到且无 seed 后缀 → 当作 setting 前缀，glob `<name>_seed*.log` 收集所有种子

输出：`{setting: [(seed_label, log_path), ...]}`，按 seed 数值排序（`seed_sort_key`）。

## 3. 三种图（每个 setting 一组，同图多 seed 分面）

由 `plot_setting_group` 产出到 `<out_dir>/<setting>_*.png`：

| 文件 | 函数 | 内容 |
|---|---|---|
| `*_curve.png` | `plot_learning_curve` | 每 seed 一个子图：左轴 train/test acc + best（绿虚线），右轴 train loss（红）；loss 轴上限 = 95% 分位数 × 1.2（忽略首轮尖峰），acc 轴固定 [-0.05, 1.05] |
| `*_per_pos.png` | `plot_per_position` | 每 seed 一个子图：`imshow` 热图（epoch × 位置），`RdYlGn` colormap，vmin=0/vmax=1；不同起始位置/长度的行用 NaN 对齐补齐（`_regularize`），epoch 倒序（新的在上） |
| `*_per_rule.png` | `plot_per_rule` | 每 seed 一个子图：每条规则一条 acc 曲线（tab10 配色），并在每条曲线最高点标注数值；无 per-rule 数据时跳过 |

通用样式：matplotlib `Agg` 后端（无显示环境可用）、`FIG_DPI=150`、多 seed 时 `_make_subplot_grid`（最多 4 列）自动排版。

单实验模式（data 为单个 dict 而非 list）时布局不同：curve 是 1×2（acc 左 / loss 右），per_pos 是大张热图。当前批处理流程都用多 seed 模式。

## 4. 入口与 CLI

```bash
python src/visualize.py [names...] [--log-dir DIR] [--out-dir DIR]
                        [--all] [--base-dir /data/cxm/recursion]
                        [--no-per-pos] [--no-per-rule] [--no-group]
```

- 不带 names：用 `--log-dir` 下**所有 .log**（本项目合并整理后的标准用法）
- names 可以是完整实验名或 setting 前缀（自动收集 `前缀_seed*.log`）
- `--all`：读 `experiments/experiments.json` 的全部实验，默认 log/plot 目录为 `<base-dir>/experiments/{logs,plots}`
- `--no-group`：老行为，每个 log 单独出图（不分 seed 聚合）
- 默认（不传 --log-dir/--out-dir）：`./logs` → `./plots`

### 本项目惯例

```bash
python src/visualize.py \
  --log-dir /data/cxm/recursion/<实验目录>/logs \
  --out-dir /data/cxm/recursion/<实验目录>/plots
```

合并/拼接实验后需要重画时：先 `rm -f <plots>/*.png` 再跑上面命令（否则旧图残留）。

## 5. 关键实现位置（src/visualize.py）

- `parse_log` (L23)：日志 → metrics dict
- `extract_setting_and_seed` (L94)：文件名拆 setting/seed
- `collect_log_groups` (L437)：按 setting 聚合 log 文件
- `plot_learning_curve` (L144) / `plot_per_position` (L236) / `plot_per_rule` (L340)
- `plot_setting_group` (L473)：一个 setting 出全套图
- `_resolve_inputs` (L502) / `main` (L539)：CLI 参数解析
