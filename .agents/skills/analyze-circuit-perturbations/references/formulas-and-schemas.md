# 候选公式与数据结构

## 1. 局部三项递推及敏感性

所有公式均在模 P 下计算，且 `a' = a^{-1}`、`b' = b^{-1}`。

| 角色 | 公式 | 扰动响应 |
|---|---|---|
| 最新项 F | `x[k]=a*x[k-1]+b*x[k-2]` | `k-1 -> a`，`k-2 -> b` |
| 中间项 M | `x[k]=a'*x[k+1]-a'*b*x[k-1]` | `k-1 -> -a'b`，`k+1 -> a'` |
| 最早项 R | `x[k]=b'*x[k+2]-b'*a*x[k+1]` | `k+1 -> -b'a`，`k+2 -> b'` |

只在相应位置存在时使用公式：F 要求 `k>=2`，M 要求 `1<=k<=L-2`，R 要求 `k<=L-3`。未出现在公式中的位置预期响应为 0，但这是待检验假设，不是预设结论。

len5 的典型分类：

- `k=0`：只有 R；
- `k=1`：M 与 R 可选；
- `k=2`：F/M/R 均可选；
- `k=3`：F 与 M 可选；
- `k=4`：只有 F。

## 2. len5 “其他”的排除口径

文件 `reports/data/fillblank_len5_all_blank_linear_a25b56.json` 中：

| 空格/扰动 | 排除斜率 | 未解释数量 |
|---|---|---:|
| `x1 / x2` | `61,88`（a' 与 -b'a） | 1023/4032 = 25.37% |
| `x2 / x0` | `56,0`（F 与 M/R 的 0） | 399/4032 = 9.90% |
| `x2 / x1` | `25,13,0`（F/M 与 R 的 0） | 1222/4032 = 30.31% |
| `x2 / x3` | `0,61,88`（F 的 0、M 与 R） | 1184/4032 = 29.37% |

不要把不同排除口径的“其他”合并比较。

## 3. len5 raw 结构

`by_blank.xK` 下包含：

- `base_clean_windows`：选中的干净窗口；
- `base_outputs`：基线数值答案；
- `summaries.xJ.fits`：七候选 exact_count/exact_match；
- `raw_records.xJ`：逐扰动列式记录。

`raw_records.xJ` 的主要数组：

```text
base_row, d, baseline_output, perturbed_numeric_output,
perturbed_full_vocab_output, D
```

第 `i` 条记录的基底为 `base_clean_windows[base_row[i]]`。筛选 `D != k*d (mod P)` 后可统计：

- 输出：`perturbed_numeric_output`
- 变化：`D`
- 等效斜率：`D * d^{-1} mod P`
- 基底集中度：`base_row`

d 必须非零才能计算逆元。

## 4. len8–64 raw 结构

长窗口脚本使用：

```text
by_length[L].by_blank[K].by_relative[R]
```

raw 字段为：

```text
base_row, d, y0, y1, full1, D
```

`base_row` 对应该长度固定的 128 个 held-out 窗口。聚合位于：

```text
aggregates.by_length / aggregates.all
comparison.by_length / comparison.all
```

len5 对照位于 `len5_reference`。dominant_match 只表示 argmax 相同；拟合分布差异另看 `fit_vector_mae_pp`，不能仅凭一个最佳系数宣称电路等价。

## 5. 远位置与纠错补充

远位置扫描默认只给异常单元嵌入 raw，但所有单元都保留七候选精确计数。检查绝对全零时，另看最小 zero_rate、输出改变总数和最大非零候选率，不要只看阈值异常数。

纠错 `by_corrupt_position.xK` 的错误位 raw 还包含 `clean_hit` 与 `clean_delta_per_trial`。拆分时使用各自的 `clean_n/nonclean_n` 作分母，不能误用总数 4032。
