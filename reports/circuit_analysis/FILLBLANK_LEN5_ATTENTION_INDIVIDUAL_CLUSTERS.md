# (25,56) len5：逐 base 注意力与距离分组

2026-11-06。按本次确认范围分析 **s=1、len=5、x0–x4 挖空**，每个位置复用历史 32 个 base，共 160 个输入。已生成每个 base 的完整注意力、576 维向量和单独热图，并在同一挖空位置内按欧氏距离分组。

**结果：完整注意力的组间分离较弱；x2 的最后一层 EOS 注意力与历史扰动的 F/M 倾向关联更明显。现有证据支持继续区分不同读出模式，尚不能把聚类直接命名为确定的计算电路。**

直接查看：[全部 base 图册](plots/fillblank_len5_attention_individual_a25b56/index.html)、[x2 完整注意力距离树](plots/fillblank_len5_attention_individual_a25b56/x2_overview.png)、[x2 最后一层 EOS 的 32 个 base](plots/fillblank_len5_attention_individual_a25b56/x2_last_layer_eos_bases.png)、[全部数据与分类元信息](../data/fillblank_len5_attention_individual_a25b56/summary.json)。

## 数据与模型

- checkpoint：`/mnt/workspace/hujiachen/circuit_analysis/checkpoints/single_o2_fillblank_a25b56_d512l4r8h4_P127_len3to5_randpos_noaux_lr0.0003_wd2_ckpt_seed42_best.pth`。
- P=127，递推 `x[t+2] = 25*x[t+1] + 56*x[t] mod 127`；D512/L4/H4，MLP ratio=8，body supervision=false，训练长度3–5。
- checkpoint epoch=710，记录的 best accuracy=99.710684%；checkpoint 与历史基底文件的配置、epoch、best accuracy 一致，SHA256 存于 summary。
- 基底：[历史扰动数据](../data/fillblank_len5_all_blank_linear_a25b56.json)，每个位置的32个 held-out 且原预测正确的窗口。它们不能用于无偏估计模型准确率；不同挖空位置的相同 base_row 也不是同一输入。
- 输入 `[x0,...,BLANK(xk),...,x4,EOS]`，BLANK=127，EOS=128。全部分析使用 CPU、fp32。

## 向量和距离

每个 base 保存 `A[layer,head,query,key]`，形状 `4×4×6×6`。按 C-order 展平为576维，不平均 base、层或头，不标准化、不按向量范数归一化。纵轴为 query、横轴为 key，BLANK/EOS 均保留，颜色固定 viridis 0–1。

距离为 `d(i,j) = sqrt(sum((A_i - A_j)^2))`，也就是整个注意力张量差的 Frobenius 范数。未来位置的恒零元素、首 query 的恒定自注意力对距离没有贡献。

每个挖空位置独立做 complete-linkage 层次聚类。在阈值 t 处切树后，同组任意两个 base 的距离均不超过 t。当前自动比较可实现的2–8组切分，选择 silhouette 最大的切分；所有候选阈值、得分、成员、完整树、距离矩阵均保留。这是便于浏览的探索性分组方法，低分时不能认为模型存在对应数量的自然类别。组号只在同一位置、同一特征集内有效。

| 挖空位置 | 完整576维分组大小 | 距离阈值 | silhouette |
|---|---|---:|---:|
| x0 | 11 / 16 / 5 | 2.1594 | 0.0713 |
| x1 | 2 / 30 | 2.2003 | 0.1299 |
| x2 | 2 / 30 | 2.2683 | 0.1325 |
| x3 | 4 / 3 / 4 / 5 / 4 / 4 / 8 | 1.4842 | 0.0705 |
| x4 | 18 / 14 | 1.7187 | 0.0561 |

这些分离度都偏弱。尤其 x0/x3/x4 的历史扰动分别一致偏 R/F/F，仍会被完整注意力距离切出多个组，说明注意力差异也包含输入内容等变化，不能全部解释成推导路径变化。

图册按完整576维分组排列，每张图都对应一个真实 base。每组的 medoid 是组内到其他成员距离之和最小的真实 base，保存在元信息中；没有生成组均值来替代个体。

## x2：和推导路径的关系

聚类时完全没有使用历史扰动标签。事后对照三种局部公式，均在 mod127 下：

| 标签 | 操作数 | x2 的公式 |
|---|---|---|
| F | x0、x1 | `56*x0 + 25*x1` |
| M | x1、x3 | `13*x1 + 61*x3` |
| R | x3、x4 | `88*x3 + 93*x4` |

每个 base 的候选公式得分来自历史全部 `4个非空位置 × 126个非零扰动`：操作数位置要求输出差 `D=系数*d mod127`，其余位置要求 `D=0`；完整词表 argmax 为非数值 token 时记错。得分与逐位置得分均单独保存。

取三种得分最大的标签后，x2 有 F=23、M=8、R=1。但最佳拟合度只有46.4%–76.6%，没有一个达到预设的强证据门槛（拟合≥80%，与次名差≥10个百分点）。所以 F/M/R 在这里都是**扰动倾向**；扰动输入已离开干净递推分布，标签也不能直接证明原输入采用哪个公式。

完整576维聚类把 b30、b31 单独分为一组，另30个 base 为一组，后者仍混有23个F和7个M，不能可靠分离F/M。

为了查看答案读出，另保存两组补充距离：所有层/头的 EOS query（96维）、最后一层各头的 EOS query（24维）。这两组也是原始权重的直接子向量，没有平均。32个 base 的 L4 EOS 聚类分离度提高到0.4104，分成9和23个：

| L4 EOS 分组 | F倾向 | M倾向 | R倾向 | medoid |
|---|---:|---:|---:|---|
| C1，9个 | 2 | 6 | 1 | b25 |
| C2，23个 | 21 | 2 | 0 | b28 |

参见 [L4 EOS 距离树](plots/fillblank_len5_attention_individual_a25b56/x2_last_layer_eos_overview.png) 和 [逐 base 读出热图](plots/fillblank_len5_attention_individual_a25b56/x2_last_layer_eos_bases.png)。这里的组号与完整576维分组不同。

为检查是否只是唯一R样本造成差异，排除 b31，只比较23个F倾向和8个M倾向。相同倾向有281对，不同倾向有184对；下表是这些配对距离的算术均值，不是对注意力求均值：

| 特征 | 相同倾向距离均值 | F–M距离均值 | 相对增幅 |
|---|---:|---:|---:|
| 完整576维 | 1.5461 | 1.5863 | 2.6% |
| 所有层 EOS，96维 | 0.7690 | 0.8821 | 14.7% |
| 最后一层 EOS，24维 | 0.4291 | 0.5862 | 36.6% |

[距离分布图](plots/fillblank_len5_attention_individual_a25b56/x2_FM_distance_comparison.png) 显示分布仍有重叠。用1999次 base 标签置换（保留组大小，不把配对当独立样本）检验“异倾向距离更大”，单侧探索性 p 分别为0.089、0.0005、0.0005；未做多重比较校正，样本只有这31个预选正确基底，不能外推为总体电路结论。原始比较保存在 [排除R的结果](../data/fillblank_len5_attention_individual_a25b56/x2_excluding_single_R.json)。

一个可见差异在 L4H4 的 EOS query：

- [b25，偏M，窗口 `[28,24,9,45,105]`](plots/fillblank_len5_attention_individual_a25b56/x2/base_25.png)：对 BLANK(x2) 的权重0.176，对 x4 的权重0.787。
- [b28，偏F，窗口 `[66,108,46,86,27]`](plots/fillblank_len5_attention_individual_a25b56/x2/base_28.png)：对 BLANK(x2) 的权重0.593，对 x4 的权重0.339。

这是两个 L4 EOS 簇的 medoid，选择没有使用路径标签。此处读的是第四层的隐藏表示：读取 x4 并不等于直接把原始 x4 当算术操作数；该位置可能已经汇集前面位置的信息。因此能确认的是读出位置/权重模式有变化，确切计算路径还需要激活干预来验证。

## 文件和复现

数据目录：`reports/data/fillblank_len5_attention_individual_a25b56/`。

- `xK/base_NN.json`：单个 base 的干净窗口、输入 token、state index、预测/真值、完整注意力、576维向量、组号和扰动得分；共160份。
- `xK_arrays.npz`：attention `(32,4,4,6,6)`、vectors `(32,576)`、distances `(32,32)`、inputs、state indices、predictions、cluster labels、linkage。
- `xK_distances.csv`：完整距离矩阵，行列都按 base_row=0..31 排列；`xK_membership.csv`：成员与标签表。
- `xK_all_eos_queries_distances.npy` / `xK_last_layer_eos_distances.npy`：两个补充特征的距离矩阵。
- `summary.json`：模型/基底路径与哈希、完整聚类树、全部候选切分、所有成员和校验记录。不同特征的聚类结果分别保存。

图目录：`reports/circuit_analysis/plots/fillblank_len5_attention_individual_a25b56/`，含160张完整单base热图、每个位置的完整/补充聚类图与EOS热图、x2距离分布图，以及可按分组浏览的 `index.html`。

脚本：[probe_fillblank_attention_clusters.py](../../scripts/probe_fillblank_attention_clusters.py)。测试：[test_fillblank_attention_clusters.py](../../tests/test_fillblank_attention_clusters.py)。运行日志：[log](../data/fillblank_len5_attention_individual_a25b56.log)。

```bash
cd /mnt/workspace/hujiachen/recursion
python tests/test_fillblank_attention_clusters.py
python tests/test_fillblank_attention_maps.py
CUDA_VISIBLE_DEVICES='' python scripts/probe_fillblank_attention_clusters.py
# 只重画补充图/图册，直接读取已保存的每个 base 数据，无需模型 forward：
python scripts/probe_fillblank_attention_clusters.py --plots-only
```

完整分析支持 `--distance-threshold T` 以明确的距离上限替代自动选切分；另指定 `--data-dir` 和 `--plot-dir` 可保留当前版本。只想快速重分组时，可直接调用 `cluster_vectors(np.load(...)["vectors"], distance_threshold=T)`，无需再次提取注意力。

## 校验

- 160/160个 EOS 全词表预测与历史 base outputs 及真值一致。
- hook 与独立 forward 对拍92,160个注意力元素，最大绝对差 `2.384185791015625e-07`，满足 rtol=1e-5、atol=1e-6。
- 每个 base 都通过有限值、0–1范围、行和为1、未来位置为0、首 query 自注意力为1检查；JSON/NPZ 重新读取后验证无损及向量展开顺序。
- 作为历史对照，本次个体注意力的 batch 均值与旧图数据五组均一致，最大绝对差 `1.1920928955078125e-07`。本次交付和聚类使用的始终是个体数据。
- 全部PNG重新读取验证，逐 base 图通过五张联系表检查，另检查完整分辨率图和 x2 补充图的轴、标签、层头布局、色阶。
- 新增3项CPU数学/绘图测试与既有6项attention测试通过。顺带将旧依赖脚本中一处打印语句改为兼容当前Python 3.10的f-string写法；不改变模型或计算公式。
