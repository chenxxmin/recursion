# Fill Blank 全位置激活补丁汇总：x0–x4

- 日期：2026-11-04
- 模型：a25b56、len3～5 训练、四层 best checkpoint
- 每个 blank：32 个各自 held-out clean-correct 基底，全部 `d=1..126`
- 补丁：5 个 residual 边界 × 合法的 `k-2,k-1,k+1,k+2` 位置，排除 blank 自身

## 总体结论

**强恢复节点不是固定的 x2，而是随 blank 和污损来源移动。** 早中期 future-operand residual 在多个位置具有很强的局部充分性，但深层保持程度、input 依赖和是否存在单点恢复都随场景变化，不能概括为统一电路。

- x0：x2 的早中期 residual 强，R2 降至约 9%。
- x1：强节点移到 x2/x3；x0→x2 到 R2 仍有 97.52%。
- x2：x3 最稳健，x4 次之；x1→x3 为中等强度。
- x3：只有 x4→x4 是大分母外仍需谨慎的 100% 小样本节点；x2→x4 很弱。
- x4：9 个可解释 cell 全部 0 恢复，未发现单点局部充分性节点。

以上均是 clean→corrupted 单点补丁的**局部充分性**结果，不证明必要性、唯一性或完整电路。

## 1. Changed 样本分母

主指标为 changed-only `Restore_full`：corrupted full-vocab 输出确实不同于原答案时，patch 后是否恢复原 `x_k`。unchanged 样本另行统计保持率，不混入恢复分母。

| Blank | source | changed / 4032 | Blank | source | changed / 4032 |
|---|---|---:|---|---|---:|
| x0 | x1 | 4032 | x2 | x3 | 2006 |
| x0 | x2 | 4032 | x2 | x4 | **103** |
| x1 | x0 | 2781 | x3 | x1 | 3915 |
| x1 | x2 | 4026 | x3 | x2 | 4029 |
| x1 | x3 | 2687 | x3 | x4 | **235** |
| x2 | x0 | 1541 | x4 | x2 | 4032 |
| x2 | x1 | 3786 | x4 | x3 | 4032 |

因此，x2 的 x4→x4 和 x3 的 x4→x4 虽然都是 100% 恢复，实际分别只有 **103** 和 **235** 个 changed 样本，不能与 4032 个 changed 样本上的 100% 等同解释。

## 2. 主要有效节点

R0/R1/R2 分别是 L1/L2/L3 后的 residual；表中均为 changed-only Restore_full。

### x0：x2 早中期强，R2 衰减

| 污损→补丁 | R0 | R1 | R2 |
|---|---:|---:|---:|
| x1 → x2 | 87.0536% | 75.7937% | 9.5238% |
| x2 → x2 | 99.4544% | 97.4206% | 8.8046% |

直接 x1 patch 只有 2 次和 4 次恢复。详细结果见 `reports/circuit_analysis/FILLBLANK_X0_ACTIVATION_PATCH.md`。

### x1：x0→x2 与 x3→x3 最强

| 污损→补丁 | R0 | R1 | R2 |
|---|---:|---:|---:|
| x0 → x2 | 99.7483% | 99.2089% | 97.5189% |
| x3 → x3 | 100% | 99.9628% | 90.9192% |
| x2 → x2 | 27.2727% | 26.0556% | 5.6135% |
| x2 → x3 | 23.5718% | 19.7963% | 1.9622% |

x0→x2 的 R2 没有像 x0 场景那样崩落，仍有 97.52%。x3→x3 的 R0 在全部 32 个 base 上恢复；x0→x2 在 R0/R1/R2 分别有 29/27/21 个 full base。x2 污损下的部分恢复则有明显 base 离群，例如 x2→x2 的 base14 在 R0/R1 都只有 1/126。

### x2：x3 最稳健，x4 次之

| 污损→补丁 | R0 | R1 | R2 |
|---|---:|---:|---:|
| x3 → x3 | 94.2173% | 93.6690% | 84.2473% |
| x3 → x4 | 77.8664% | 77.8166% | 35.9920% |
| x1 → x3 | 49.1548% | 30.9297% | 17.6704% |
| x4 → x4 | **100%（N=103）** | **100%（N=103）** | **100%（N=103）** |

x0 污损没有强单点节点，最高只有 R0@x0 的 9.0850%。x4→x4 的 103 个 changed 样本散布在 10 个 base，且 base31/30 贡献 45/26 个，不能推广成全输入规律。

x3→x3 的突出负离群是 base31：R0/R1/R2 分别只有 20/125、17/125、3/125；R0/R1 其余多数 base 全恢复，R2 则只有 13/32 个 full base。x1→x3 的 base25 达到 100%/97.14%/80%，base20 则为 4/126、0/126、0/126，进一步说明 input 依赖性。

### x3：直接 x4 小样本全恢复，其他节点弱

| 污损→补丁 | R0 | R1 | R2 |
|---|---:|---:|---:|
| x4 → x4 | **100%（N=235）** | **100%（N=235）** | **100%（N=235）** |
| x2 → x4 | 3.5989% | 1.2162% | 0.2978% |

x4 污损只有 5.8284% 会改变答案。235 个 changed 样本分布于 23 个 base，这些 base 均逐 base 100% 恢复；另外 9 个 base 没有 changed，不能记为 0% 或 100%。其余 cell 最高仅 0.5875%。x2→x4 的 R0 恢复分布于 21 个 base，base29 为 24%，其余 base 合计仅 2.95%，呈明显离群。

### x4：没有可解释的单点恢复

x2、x3 两种污损在 R0–R2 的直接和间接 patch 共 9 个可解释 cell，**全部 0/4032 恢复**。所有 base 都没有恢复例外，最大 mean Δmargin 只有 +0.133834，persistence 接近 100%。

这不是结构性 no-op，而是实际的阴性结果；但它仍只否定这些位置的单点局部充分性，不排除信息位于其他位置、分散表示或需要联合补丁。

## 3. 跨位置比较与解释边界

1. **早中期强恢复模式复现，但位置改变。** x0 的 x2、x1 的 x2/x3、x2 的 x3/x4，说明不存在固定的“x2 总是答案载体”。
2. **R2 不是统一的信息消失点。** x0 的 x2 patch 降至约 9%，而 x1 的 x0→x2、x3→x3 和 x2 的 x3→x3 在 R2 仍分别为 97.52%、90.92%、84.25%。
3. **小 changed 分母必须随结果一起报告。** x2/x3 的 x4→x4 都是 100%，但只有 103/235 个 changed；unchanged 不能算作恢复。
4. **route_exact 不是唯一电路标签。** 多路由匹配和零响应路由都存在；unchanged 也可 route_exact。分 F/M/R 时应回连 `corrupted_baseline.matched_routes`。
5. embedding ceiling/identity、`source>patch` 的 causal no-op、R3 非 EOS→EOS no-op 均排除在电路解释之外。no-op 若原本正确，继续保持正确是预期，不是恢复。
6. 每个 blank 的 32 个基底不是跨 blank 配对样本；126 个 d 是聚簇扫描，不是独立随机试验。结果条件于该 checkpoint 和 clean-correct 基底。

## 4. 数据、图和审计

原始 JSON 统一使用 gzip 压缩（`.json.gz`），以控制仓库文件大小；审计时已验证压缩内容与原 JSON 逐字节一致。

每个 blank 均有：

```text
reports/data/fillblank_xK_activation_patch_a25b56.json.gz
reports/data/fillblank_xK_activation_patch_a25b56_audit.json
reports/data/fillblank_xK_activation_patch_a25b56.log
reports/circuit_analysis/plots/fillblank_xK_activation_patch_a25b56_*.png
```

规模：

| Blank | patched raw | 总体 cell | per-base cell | 热图 |
|---|---:|---:|---:|---:|
| x0 | 80,640 | 20 | 640 | 4 |
| x1 | 181,440 | 45 | 1,440 | 6 |
| x2 | 322,560 | 80 | 2,560 | 8 |
| x3 | 181,440 | 45 | 1,440 | 6 |
| x4 | 80,640 | 20 | 640 | 4 |

独立审计已逐一核对上游输出、D、matched_routes、baseline_changed、唯一 raw 键、全部总体和 per-base 聚合；五个 blank 的指标差均为 **0**。所有 mandatory controls 通过。x1–x4 的全部 24 张热图已连同 x0 的 4 张图一起检查。
