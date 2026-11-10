# B/C 二十规则补种子：已确认运行包

**最新交付：** 已追加四条三规则 A′，并按用户要求显式设置 B 的 `NUM_MASK=1`，见 [11条合并运行包](inter_intra_combined_20261110.md)。B/C的有效监督仍为q3–q16；本页及旧包保留为历史记录。

用户于 2026-11-10 确认参数并要求上传。审阅版本：`0b67b85805959d9d0bba`。
本批新增七条训练；与已有日志合并后，B/C 各覆盖五个种子。本机没有启动训练。

## 文件

- [完整实验 JSON](../experiments/inter_intra_bc_n20_5seeds_20261110_v1.json)
- [跨机器运行包](../bundles/inter_intra_bc_n20_5seeds_20261110_v1.tar.gz)
- [压缩包 SHA256](../bundles/inter_intra_bc_n20_5seeds_20261110_v1.tar.gz.sha256)
- [完整参数审阅表](../reviews/inter_intra_bc_n20_5seeds_20261110_v1.md)

建议使用运行包：它包含确认时的实际代码、展开默认值的 JSON、运行器和文件校验清单，
不依赖目标机器仓库的默认参数。审阅表保留原审阅文本；本文件和包内 manifest 记录交付状态。

## 种子与配置

| 组别 | 已有日志的种子 | 本批新增种子 |
| --- | --- | --- |
| B：同一条序列固定规则及标签 | 17996、18318 | 34789、44536、66601 |
| C：序列内逐步选择规则及标签 | 17996 | 18318、34789、44536、66601 |

两组固定 P127、二阶、D1024/L2/H4/MLP4、数值长度16（含标签30 tokens），
监督 q3–q16。二十条规则按 canonical 方向为 `RULES[i-1]=[2*i-1,i]`，i=1,…,20，
即第一个系数乘最新值。无污损；AdamW LR3e-4/WD1/clip1；batch1024，梯度累积1，
不逐轮重洗牌；从头初始化；fp32+TF32；cosine T_max6000轮；无墙钟时限。

- B：轨道切分，训练每规则11290条、共225800条；静态测试每规则4839条、共96780条。
- C：固定300000条训练，每轮fresh256条测试。
- 每轮评估；B单次测试≥99%或C最近10次均值>99%成功停止；连续3000轮无提升也停止，最多6000轮。
- 最大并发2，一卡一条；可只选择一张GPU顺序执行。保存最佳权重、latest和完整续跑存档。

完整参数及逐项来源以审阅表为准。B/C 沿用各自的数据制度，不是训练样本和更新次数完全匹配的实验。
相同种子编号也不表示两组共享相同样本。历史 B17996/C17996 均在6000轮前被外部停止，汇总时注明实际预算；
C18318 缺失日志不算失败，本批重新从头运行。预先固定本批全部种子，完整报告结果。

## 目标机器运行

在代码仓库根目录更新并解压：

```bash
git pull --ff-only
tar -xzf bundles/inter_intra_bc_n20_5seeds_20261110_v1.tar.gz -C bundles
cd bundles/inter_intra_bc_n20_5seeds_20261110_v1
bash run.sh check
bash run.sh run --gpus 0,1 --output /data/runs/inter_intra_bc_n20_5seeds_20261110_v1 --background
```

将 `0,1` 改成目标机器实际允许使用的 GPU；输出路径须为尚未使用的新目录。
目标机器需 Linux、Python3.10+、匹配 CUDA 的 PyTorch 及包内 requirements.txt 中的依赖。
可设置 `EXPERIMENT_PYTHON=/path/to/python` 选择环境。

查看进度或在存档点停止：

```bash
bash run.sh status --output /data/runs/inter_intra_bc_n20_5seeds_20261110_v1
bash run.sh stop --output /data/runs/inter_intra_bc_n20_5seeds_20261110_v1
```

分配到多台机器时，用 `--only 实验名[,实验名]` 限定各机子集，避免重复训练。
实验名如下：

```text
inter_intra_b_o2_P127_N20_d1024l2h4r4_len16_seed34789
inter_intra_b_o2_P127_N20_d1024l2h4r4_len16_seed44536
inter_intra_b_o2_P127_N20_d1024l2h4r4_len16_seed66601
inter_intra_c_o2_P127_N20_d1024l2h4r4_len16_seed18318
inter_intra_c_o2_P127_N20_d1024l2h4r4_len16_seed34789
inter_intra_c_o2_P127_N20_d1024l2h4r4_len16_seed44536
inter_intra_c_o2_P127_N20_d1024l2h4r4_len16_seed66601
```

## 输出与交付校验

- 运行状态：`<output>/state.json`、`<output>/controller.log`。
- 训练日志：`<output>/results/<experiment>/logs/<experiment>.log`；启动日志位于 `<output>/logs/`。
- 模型存档：`<output>/models/<experiment>/`。
- 曲线：`<output>/results/<experiment>/plots/`。

审阅表中的 log_path 是概括路径，实际训练日志以此处运行器布局为准。

已通过运行包完整性检查、所有 manifest 文件哈希校验、七条配置/种子/规则/数据模式核对及压缩包内外 JSON 一致性检查。
此前的 CPU 数据集核对和实验交付集成测试均通过；没有执行正式 GPU 训练。

压缩包 SHA256：`3b33a40bf8679742adbc4a80dfa971b11f0ceba512b1bb04c040b74d2e971537`。
