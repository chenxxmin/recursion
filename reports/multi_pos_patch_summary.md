# Multi-Position Patching / Flip Hunt — 结果汇总

**日期**：2026-08-24
**前置**：`activation_patch_summary.md`（第一轮 HSS 全 null）。
本文档修正其方法论缺陷后的完整重测结果。

## 1. 方法论修正（相对第一轮 HSS 的三处关键改动）

1. **patch 位置 × 预测位置对齐**：l1 模型中 patch 位置 q 只影响预测位置
   k=q+1，第一轮对全部预测位置取平均把 0.4 的真实翻转稀释成 0.013。
   现在逐 (patch 位置, 预测位置) 报告；
2. **三目标分类**：区分 规则翻转（源规则 × 输入自己的历史，match_a/b）、
   状态转移（源运行的实际续写，match_src）、崩坏（neither）。
   缺一就会把 resid 整向量 patch 的 trivial 状态转移误读成规则位点；
3. **多位置同时 patch 与新 site 类型**：`('emb',)`、`('resid', layer)`、
   `('attn_all', layer)`，`pos` 接受列表。
   代码：`src/activation_patch.py`（扩展）+ `src/multi_pos_patch.py`（flip hunt
   梯子），单测 10/10（`tests/test_activation_patch.py`）。

## 2. Attention pattern 事实（全部 9 个 l1 模型 + 3 个 l2 模型）

- pattern 与规则无关（A/B 逐位一致）→ pattern patching 无意义，已排除；
- l1 h2 模型：head 分工 = lag1 头（递推操作数）+ lag2 头（规则证据），
  当前 token 走 skip connection，无 lag0 头；
- l1 h4 模型：lag0 头 + lag1 头 + lag2 主导头 + 一个 lag2/lag3 混合头；
- l2 h2 模型：layer0 有较干净的 lag1 头，layer1 pattern 较弥散。

## 3. 主要发现：规则证据通道可定位

**patch 后果按通道角色整齐分层**（对 6 个 l1 模型 × 2 方向稳定）：

| patch 对象 | 后果 |
|---|---|
| lag1 头（操作数） | 崩坏（neither ≈ 1.0） |
| lag0 头 / emb（当前 token） | 崩坏 |
| MLP 隐藏层 | 崩坏（l1 中 MLP 是负载组件！） |
| **lag2 头 @ query 位置** | **规则翻转 Δ+0.13~0.36（max 0.65），src≈0** |
| lag2/lag3 混合头 | 无影响（惰性） |
| resid 整向量 | 状态转移（src≈1.0，trivial 天花板） |

- **翻转头恒为 lag2 主导头**，head 编号随 seed 置换、功能角色不变（6/6）；
- **结构性边界**：k=3,4（预测 x_4, x_5）lag2 头读的 x_1/x_2 为 A/B 共享，
  patch 为恒等操作，Δ 严格为 0；k≥5（读到分叉的 x_3 起）翻转出现。
  推论：预测 x_4 时规则身份与 x_3 的值不可分离，不存在可对调的中间表示；
- **翻转是部分的**（max ~0.65）：规则判定是多通道证据整合，非单一开关；
- **l2 模型（d256l2r8h2_P127，3 seeds）**：翻转通道上移到 **layer 1 的某个
  head**（编号随 seed 置换），AtoB Δ+0.27~0.44；layer 0 各头无效。

## 4. A2 验证（block 输出 patch 在 l2 上的重做）

对 `mixed_basic_d256l2r8h2_P127_N2_e0.7`（3 seeds）：

- **resid1@q（末层 block 输出 / ln_f 输入）**：match_src ≈ 0.998——
  状态转移，数学上必然（logits 是它的逐位置确定性函数）；
- **resid0@q（第 0 层 block 输出）**：match_src ≈ 0.73，第 1 层部分重算；
- **resid0@W（第 0 层整窗口）**：match_src ≈ 1.0。

结论：第一轮"block 输出 patch 不翻转"的记录若指末层输出则与数学矛盾，
实际应是第 0 层输出（冗余可解释）。本轮数据与此自洽。

## 5. 修正第一轮 summary 的两处结论

1. ~~"MLP 对规则区分几乎无贡献"~~ → l1 中 MLP 隐藏层 patch 使预测完全
   崩坏（neither≈1.0），MLP 是递推计算的负载组件；l2 中"第 0 层 MLP 输出
   patch 无影响"只说明该层冗余、可被第 1 层补救；
2. ~~"无法定位承载规则区分的 hidden state"~~ → 在新口径下可定位：
   l1 为 lag2 头 @ query 位置，l2 为 layer 1 的对应证据头。

## 6. 未解决问题

- **B3 翻转率天花板 ~0.65**：拉锯假说 vs 数值巧合假说，需逐样本数据裁决
  （当前 report 只有聚合率，需改脚本存逐样本后重跑）；
- **B1 k4 AtoB 状态转移异常**：lag0 头/mlp 在 k4 仅 AtoB 方向出现
  match_src≈1.0（其余条件皆崩坏），原因不明；
- **B2 方向不对称**：BtoA 恒强于 AtoB（h4 达 2~3 倍，l2 更极端），
  不能用 k2 先验解释；
- **B4 lag2/lag3 混合头为何惰性**（冗余备份 vs 旁观者，可做组合 patch 判别）；
- **B5 h4 的 AtoB 翻转率系统性低于 h2**（~0.16 vs ~0.44），疑证据通道稀释。

## 7. 跨规则对验证（2026-08-25，[(1,1),(2,3)] N=2 模型）

在 `mixed_basic_d256l1r8h2_P127_N2_e0.7`（rules12345 批次，规则对
**[(1,1),(2,3)]**，a≠1，4 seeds，best acc 99.0~99.4%）上复测：

- **3/4 模型结论完全成立**：翻转头 = lag2 主导头（seed17996 h1:0.86、
  seed34789 h0:0.74、seed44536 h1:0.79），AtoB/BtoA Δ+0.25~0.63，
  match_src≈0；k3/k4 结构性零值、k≥5 翻转出现的边界不变；
  resid→状态转移、其余组件→崩坏/无效的格局不变；
- **1/4 模型（seed18318）反例**：head 解剖不同（h1=纯 lag1 操作数头，
  h0=lag0+lag1+lag2 混合头，**无 lag2 主导头**），任何 head patch 都
  崩坏（neither≈0.9~1.0），**不存在可翻转位点**。说明该 seed 学到的
  解决方案把规则证据与操作数纠缠在同一通道里——"lag2 纯证据通道"
  是常见解（迄今 l1 模型 9/10），但不是唯一解；
- 新模型的操作数通道解剖随 a≠1 改变（出现 lag0 头 / lag0+lag1 混合头），
  但证据通道的角色定义不变；
- **B2 方向不对称在此批中方向不定**（seed17996 AtoB 强、seed44536 BtoA 强、
  seed34789 对称）——不对称依赖具体规则对/seed，非普遍现象。

报告：`reports/flip_hunt_rules23_seed{17996,18318,34789,44536}.json`。

## 8. 报告文件

`reports/flip_hunt_{h2,h4}_seed{0,999,12345}.json`（l1，d256l1r8h{2,4}_P127）、
`reports/flip_hunt_l2h2_seed{0,999,12345}.json`（l2，d256l2r8h2_P127）。
