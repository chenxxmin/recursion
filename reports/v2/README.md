# v2 实验制度（2026-09-03 起）

新一批实验与旧实验**完全隔离**，超参制度如下：

## 训练制度变更

| 项 | v1（旧） | v2（新） |
|---|---|---|
| 训练集 | 10,000 条 | **300,000 条** |
| 序列长度 | 视批次（tr64 等） | **127**（L=64 个 value + 63 个 flag） |
| 测试集 | 固定 2000 条，长度 OOD | **每次评估临时新采 256 条**，长度同训练（127），不再区分 OOD |
| 其他 | — | 不变（EPOCHS=6000、BATCH_SIZE=512、EVAL_INTERVAL=20、早停规则同） |

实现：`FRESH_TEST_PER_EVAL=true`（每次 eval 重建 ActionDataset，seed 从
`RANDOM_SEED + 10^6 + 7` 的独立 RNG 流派生，日志打 `[FreshTest]` 行）。
注意：256 条评估在 99% 附近噪声约 ±1.2%，best/早停判定会比 v1 更"毛躁"。

## 目录隔离约定

| 内容 | v1 | v2 |
|---|---|---|
| 启动配置 | `experiments/*.json` | **`experiments/v2/*.json`** |
| 日志/图 | `/mnt/workspace/hujiachen/recursion_results/<批次名>/` | **`/mnt/workspace/hujiachen/recursion_results_v2/<批次名>/`** |
| 模型/checkpoint | `/mnt/workspace/hujiachen/models/<批次名>/` | **`/mnt/workspace/hujiachen/models_v2/<批次名>/`** |
| 实验记录文档 | `reports/*.md` | **`reports/v2/*.md`** |

## 启动方式

```bash
# 单批
python src/batch_run.py --base-dir /mnt/workspace/hujiachen/recursion_results_v2 \
    --model-base-dir /mnt/workspace/hujiachen/models_v2 experiments/v2/<批次>.json

# 链式多轮（自动续 checkpoint / 跳过已完成）
CHAIN_DATA_BASE=/mnt/workspace/hujiachen/recursion_results_v2 CHAIN_MODEL_BASE=/mnt/workspace/hujiachen/models_v2 \
    bash scripts/chain_rounds.sh experiments/v2/<批次>.json <gpus> <rounds>
```

## 默认精度

沿用 2026-09-01 结论：fp32 + TF32（`USE_AMP=false, ALLOW_TF32=true`）。
bf16 仅限明确只取阳性的粗筛，且需在实验配置里显式开启。
