# Activation Patching Probe 交接文档

**日期**：2026-08-19
**目的**：在 mixed basic（mixed_ab，**无 ab_tag**）已训练到 100% 正确率的 checkpoint 上，定位"模型用哪个 hidden state 区分当前规则"。

## 1. 任务背景

paired 样本：共享初始值 x1, x2，A 样本按规则 A 递推、B 样本按规则 B 递推。把 A 样本某组件某位置的 hidden state（HS_A）覆盖到 B 样本前向的同一位置（反向也做），看输出变得符合规则 A、规则 B、还是都不符合——翻转率高的组件即承载规则区分的候选。

## 2. 已实现的代码（本仓库，已推送）

- `src/activation_patch.py` — probe 主程序 + CLI：
  ```
  python src/activation_patch.py <model.pth> [--pairs 64] [--len L] [--seed 0] \
      [--rule-a 0] [--rule-b 1] [--out report.json]
  ```
- `tests/test_activation_patch.py` — 7 个单测（随机初始化小模型），已全绿：
  `python tests/test_activation_patch.py`（无需 pytest，直接跑）。

### Hook 点（未改 models.py，全部 forward pre-hook）

| 组件 | 位置 | 形状 |
|---|---|---|
| attention 每头 | `block.attn.c_proj` 的输入（`att@v` reshape 后），按头切列 `[h*hs:(h+1)*hs]` | (B,T,head_size) |
| MLP 隐藏层 | `block.mlp[2]` 的输入（GELU 之后） | (B,T,mlp_ratio*d) |

每层有 n_head 个头 + 1 个 MLP，逐 token 位置单独 patch，双向（AtoB = 输入 B 注入 HS_A；BtoA = 输入 A 注入 HS_B）。

### 判定口径（重要，用户明确改过）

- 每个预测位 k（≥2，用输入样本自身历史）分别算 v_A、v_B；
- **不跳过任何位置**：记录 match_A / match_B 两个布尔，v_A==v_B 时两者可同时为真；argmax 两边都不中也计数（neither = 1 - matchA - matchB + both）；
- 报告 `match_a_rate` / `match_b_rate`；翻转效果看 AtoB 方向的 match_a_rate 相对 baseline 的增量。
- baseline（不 patch）在 report['baseline'] 里，100% 模型应接近 match 1.0（input_A→A、input_B→B），跑真模型时**先核对这一项**，不对就说明环境/加载有问题。

## 3. 服务器上要做的事

1. 选一个 100% 正确率的 mixed_ab（无 tag）pth（models/ 下，config 里 `use_ab_tag: false`、`order: 2`、`ab_pairs` 两条规则）。
2. 先小规模试跑确认流程：
   ```
   python src/activation_patch.py models/<ckpt>.pth --pairs 8 --len 32 --out .chain_tmp/probe_smoke.json
   ```
   检查：baseline 两个方向都 ≈1.0；无 NaN/报错。
3. 正式跑：`--pairs 64`（或更多），长度用 checkpoint 的 train_len（默认就是）。
4. 看 `report.json`：results 是 方向 × 组件 × 位置 的逐点记录，可自行聚合画热图（位置 × 组件 的 match_a_rate 增量）。

### 预期规模 / 耗时

前向次数 = 2 方向 × n_layer × (n_head+1) × seq_len。l2h4、len 64 约 1280 次前向（每次 batch=pairs），单卡分钟级。l4 时翻倍。

## 4. 已知限制 / 可能需要微调的点

- **只支持无 tag 的 mixed_ab**（`use_ab_tag=false`）。带 tag 或 conditional_wte 的 checkpoint 没测过；若要支持，注意输入首 token 是 flag、位置偏移 +1，且 forward 需传 ab_labels。
- `load_model` 复用自 `src/analyze_attention.py`，能识别 MixedABTransformer checkpoint。
- 现在 patch 的是"单组件单位置"。若结果太弱（单点 patch 被残差流旁路绕过），自然的下一步是：同层所有头一起 patch、或相邻位置窗口 patch——改 `patched_logits` 的 slice 即可。
- 样本对生成 seed 固定（--seed），A/B 共享初始值；如需"同初始值但历史不同"之外的配对方式（如完全独立样本），改 `generate_paired_samples`。
- 分类口径若想看 per-position 细分（如"x3 位置最依赖哪个组件"），report.json 的逐位置数据已够，直接聚合即可。

## 5. 验收标准

- smoke 跑通、baseline ≈1.0；
- 正式 report 写出；
- 改动（若有微调）补进 `tests/test_activation_patch.py` 并保持全绿。
