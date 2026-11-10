# A′ / B / C 补充实验：合并运行包

用户已确认三规则 A′ 参数，并明确要求 B 的 `NUM_MASK=1`。
本批共 **11 条**，最大并发2，在外机运行；本机没有启动正式训练。
最终参数版本：`4ccaaade9a9f91451930`。

## 交付文件

- [合并实验 JSON](../experiments/inter_intra_aprime_n3_bc_n20_20261110_v3.json)
- [完整运行包](../bundles/inter_intra_aprime_n3_bc_n20_20261110_v3.tar.gz)
- [运行包 SHA256](../bundles/inter_intra_aprime_n3_bc_n20_20261110_v3.tar.gz.sha256)
- [确认后的完整参数表](../reviews/inter_intra_aprime_n3_bc_n20_20261110_v3_confirmed.md)
- [用户确认与修订记录](../reports/inter_intra_combined_delivery_20261110/approval.json)

建议使用运行包，其中包含确认时的代码、完整配置、运行器、审计结果及文件校验清单。
它支持三规则 NULL 布局，不依赖目标机器仓库的默认参数。
单独 JSON 需要匹配的三规则 NULL 实现；请使用本次运行包启动 A′。
这是此前七条 B/C 包的合并后续版本，旧包保留作为历史交付记录。

## 实验组成

| 组别 | 本批种子 | 条数 | 架构 | 训练 / 测试 |
| --- | --- | ---: | --- | --- |
| B：标签固定 | 34789、44536、66601 | 3 | D1024/L2/H4/MLP4 | 225800 / 静态96780 |
| C：逐步选择规则及标签 | 18318、34789、44536、66601 | 4 | D1024/L2/H4/MLP4 | 300000 / 每次fresh256 |
| A′：所有规则共用NULL | 17996、18318、34789、44536 | 4 | D2048/L2/H8/MLP8 | 33870 / 静态14517 |

B/C 为二十规则任务，新增结果与已有日志合并后各覆盖五个种子。
A′ 为三规则任务，种子对应历史 A 的全部四条运行。

共同设置：P127、二阶、数值长度16（插入标签/NULL后30tokens）、RoPE、无污损、dropout0，
AdamW LR3e-4/WD1/clip1、梯度累积1、不逐轮重洗牌、从零初始化、fp32+TF32。
cosine T_max6000轮、上限6000轮、无墙钟时限、连续3000轮无提升早停。

- B/C：batch1024，每轮评估；B单次静态准确率≥99%停止，C最近10次fresh均值>99%停止。
- A′：batch512，每20轮评估（也评估epoch0和末轮），单次静态准确率≥99%停止；保留权重0.5的规则辅助分类损失。
- B/C canonical规则为 `RULES[i-1]=[2*i-1,i]`，i=1,…,20；A′规则为 `[[1,1],[2,3],[3,5]]`。
  两者均是第一个系数乘最新值。

## NUM_MASK 与有效监督

| 组别 | JSON中的NUM_MASK | 实际监督数值位置（从1编号） | 每样本目标数 |
| --- | --- | --- | ---: |
| B | **1**（按用户修订） | q3–q16 | 14 |
| C | null（保留原配置） | q3–q16 | 14 |
| A′ | 2 | q4–q16 | 13 |

`action` 场景的 B/C 使用数据集提供的 `loss_mask`，不读取配置中的 `NUM_MASK` 来选择监督位置。
因此 B 改写为1后仍与 C 对齐；这次改写没有改变 B 的训练目标。
[CPU核对](../reports/inter_intra_combined_delivery_20261110/action_mask_audit.json)验证了 B 修改前后的输入、mask、logits、loss完全一致。
A′ 属于 `mixed`，其 `NUM_MASK=2` 实际生效，并与历史 A 的13个监督位置一致；NULL和标签均不作为数值预测目标。

## 外机启动

目标机器需要 Linux、Python3.10+、匹配 CUDA 的 PyTorch，以及包内 requirements.txt 中的依赖。
可设置 `EXPERIMENT_PYTHON=/path/to/python`。在代码仓库根目录执行：

```bash
git pull --ff-only
tar -xzf bundles/inter_intra_aprime_n3_bc_n20_20261110_v3.tar.gz -C bundles
cd bundles/inter_intra_aprime_n3_bc_n20_20261110_v3
bash run.sh check
bash run.sh run --gpus 0,1 --output /data/runs/inter_intra_aprime_n3_bc_n20_20261110_v3 --background
```

把 `0,1` 改为目标机器允许使用的 GPU，输出路径使用全新目录；一张卡也可顺序执行。
如需等空闲卡，在 run 命令增加 `--wait-for-idle --poll-seconds 300`。

```bash
bash run.sh status --output /data/runs/inter_intra_aprime_n3_bc_n20_20261110_v3
bash run.sh stop --output /data/runs/inter_intra_aprime_n3_bc_n20_20261110_v3
```

分发到多台机器或已有部分 B/C 在运行时，用 `--only 实验名[,实验名]` 选择本机子集。
本批沿用原 B/C 实验名。新增 A′ 名称为：

```text
mixed_o2_null_d2048l2r8h8_P127_N3_e07_tr16_nm2_aux05_seed17996
mixed_o2_null_d2048l2r8h8_P127_N3_e07_tr16_nm2_aux05_seed18318
mixed_o2_null_d2048l2r8h8_P127_N3_e07_tr16_nm2_aux05_seed34789
mixed_o2_null_d2048l2r8h8_P127_N3_e07_tr16_nm2_aux05_seed44536
```

## 输出与核对

- 状态：`<output>/state.json`、`<output>/controller.log`。
- 训练日志：`<output>/results/<experiment>/logs/<experiment>.log`。
- 存档：`<output>/models/<experiment>/`，保存best、latest和完整resume。
- A′精确计数、逐规则/逐位置指标及更新数：`<output>/models/<experiment>/<experiment>_metrics.jsonl`。
- B/C结束后曲线：`<output>/results/<experiment>/plots/`；A′跳过不兼容的旧attention后处理。

已通过两/三规则 NULL 的配对、损失、因果性、精确计数、续跑和交付CPU测试，以及 action 轨道数据回归。
[三规则全量审计](../reports/inter_intra_aprime_n3_audit_20261110/audit.json)核对了四个种子的全量数据、批次顺序、日志样本和完整D2048初始化。
历史参考是提交01feadd中的实现，原远端冻结源码不可得。

结果汇总时保留各运行的实际预算。历史 A/B/C 部分未成功种子提前停止，不当作完整6000轮失败；
A/A′按共同已观测预算比较。B/C仍沿用各自的数据与测试制度；同一JSON只统一交付，不表示所有条件完全匹配。

合并配置、运行包检查、全部 manifest/压缩包文件哈希与独立 JSON 一致性检查均已通过。
压缩包 SHA256：`f2f2f12c8c39eaf6cb1b762006a39511435b672ac8d292cd389b2bb7b8a06401`。
