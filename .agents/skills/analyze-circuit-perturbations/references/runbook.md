# 运行手册

统一使用 `/home/cxm/miniconda3/bin/python`，从仓库根目录执行。现有分析脚本默认 CPU；检查任务时不要误停其他用户进程。

## 1. len5 填空全位置

```bash
/home/cxm/miniconda3/bin/python scripts/probe_fillblank_all_blank_linear.py \
  --ckpt /data/cxm/models/variant_o2_fillblank_noaux_randpos_d512l4_P127_len3to5_wd2_ckpt/single_o2_fillblank_a25b56_d512l4r8h4_P127_len3to5_randpos_noaux_lr0.0003_wd2_ckpt_seed42_best.pth \
  --device cpu --threads 16 --n-bases 32 --batch-size 1024 \
  --out reports/data/fillblank_len5_all_blank_linear_a25b56.json
```

五位置共 80,640 次扰动，每位置 32 个基底、全部 126 个非零 d。

## 2. len5 纠错全位置

```bash
/home/cxm/miniconda3/bin/python scripts/probe_corruptfix_len5_all_positions_linear.py \
  --ckpt /data/cxm/models/variant_o2_corruptfix_valueonly_d512l4_P127_len5_initPos024/single_o2_corruptfix_a25b56_d512l4r8h4_P127_len5_valueonly_initPos024_seed42_best.pth \
  --device cpu --threads 16 --n-bases 32 --batch-size 512 \
  --out reports/data/corruptfix_len5_all_positions_linear_a25b56.json
```

五个错误位置 × 五个扰动位置，共 100,800 次扰动。错误位每个基底恰有一次 clean-hit，拆分率使用 nonclean=4000、clean=32 的分母。

## 3. len8–64 的 ±1/±2 随机扰动

```bash
/home/cxm/miniconda3/bin/python scripts/probe_fillblank_len8to64_stride8_random.py \
  --device cpu --threads 16 --n-bases 32 --n-deltas 8 --batch-size 512 \
  --out reports/data/fillblank_len8to64_stride8_random_a25b56.json
```

共 288 个挖空位置、1104 个近邻组合、282,624 次扰动。len8 路径特例已内置。输出含 len5_reference 与逐长度 comparison。

## 4. 全部远位置扫描与异常复核

```bash
/home/cxm/miniconda3/bin/python scripts/probe_fillblank_len8to64_far_random.py \
  --device cpu --threads 16 --n-bases 16 --n-deltas 4 --batch-size 512 \
  --out reports/data/fillblank_len8to64_far_random_a25b56.json

/home/cxm/miniconda3/bin/python scripts/confirm_fillblank_far_outliers.py \
  --device cpu --threads 16 --batch-size 512
```

扫描覆盖 11,664 个 `|r|>=3` 组合、746,496 次扰动。异常阈值为 zero<95% 或 best_nonzero>=10%；阈值外仍有少量变化时，继续逐单元检查。复核脚本对 zero<100% 的 12 个单元枚举全部 d，共 24,192 次。

## 5. 分布提取示例

```bash
# 空格 x1，扰动 x2，排除 a' 和 -b'a
python .agents/skills/analyze-circuit-perturbations/scripts/extract_unmatched_distribution.py \
  --data reports/data/fillblank_len5_all_blank_linear_a25b56.json \
  --blank x1 --position x2 --slopes 61,88 --top 12

# 空格 x2 的三类“其他”
python .agents/skills/analyze-circuit-perturbations/scripts/extract_unmatched_distribution.py \
  --data reports/data/fillblank_len5_all_blank_linear_a25b56.json \
  --blank x2 --position x0 --slopes 56,0

python .agents/skills/analyze-circuit-perturbations/scripts/extract_unmatched_distribution.py \
  --data reports/data/fillblank_len5_all_blank_linear_a25b56.json \
  --blank x2 --position x1 --slopes 25,13,0

python .agents/skills/analyze-circuit-perturbations/scripts/extract_unmatched_distribution.py \
  --data reports/data/fillblank_len5_all_blank_linear_a25b56.json \
  --blank x2 --position x3 --slopes 0,61,88
```

加 `--out <json>` 保存完整计数；不加时只打印摘要。

## 6. 已有报告作为格式参考

- `reports/circuit_analysis/FILLBLANK_LEN5_LAST_BLANK_LINEAR.md`：文件名保留 last_blank，但内容已扩展为五位置。
- `reports/circuit_analysis/CORRUPTFIX_LEN5_ALL_POSITIONS_LINEAR.md`：含 clean-hit/nonclean 和双不一致说明。
- `reports/circuit_analysis/FILLBLANK_LEN8TO64_STRIDE8_RANDOM.md`：含逐长度及 len5 对照。
- `reports/circuit_analysis/FILLBLANK_LEN8TO64_FAR_RANDOM.md`：含远位置稀有异常和定向复核。

相关测试：`tests/test_fillblank_stride8_random.py`、`tests/test_fillblank_far_random.py`。项目测试为纯 assert 脚本，直接 `python tests/test_xxx.py`，不使用 pytest。
