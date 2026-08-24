# 同事用样本生成模块设计

日期：2026-08-24
状态：已与用户确认（输出结构=复刻本地、暴露比例=传入参数、返回 torch.Tensor、train 保留整体 shuffle）

## 目标

给同事一个独立模块：输入 `p`、样本长度 `length`、递推系数 `(a, b)`（仅 addition），
输出 train/test 两个数据集。生成逻辑与本仓库 `src/datasets.py` 的
`RecurrenceDataset`（无污损、无混合规则）逐点一致。

## 关键事实（已实证确认）

- addition 递推：`X(k) = (a*X(k-1) + b*X(k-2)) mod p`，`init_len = 2`。
- train/test 划分按**状态计数**：打乱 `0..p²-1` 状态索引后遍历轨道，
  前 `num_samples = max(1, int(p² * exposed_ratio))` 个处理到的状态的窗口进 train，其余进 test。
- 因此同轨道的窗口**尽量**在同一集合，但跨越边界的那一条轨道会被从中拆分
  （p=53、ratio=0.7、seed=0 实测：27 条轨道中恰 1 条被拆；p=127 同理恰 1 条）。
  这是本地既有行为，模块保持一致，需告知同事。
- 本地 run() 末尾对 train 做整体 shuffle（test 不打乱）。模块保留。

## 接口

`src/sample_gen.py`（无仓库内部依赖，可单独拷走；依赖仅 torch + 标准库 random）：

```python
def generate_datasets(p, length, a, b, exposed_ratio=0.7, seed=0):
    """返回 (train, test)：两个 LongTensor，形状 (N, length)，dtype=torch.long。

    完全复刻 RecurrenceDataset 的干净样本生成路径：
    1. random.seed(seed)；num_samples = max(1, int(p**2 * exposed_ratio))
    2. 打乱全部 p² 个状态索引
    3. 对未访问状态遍历整条轨道（回到起点或撞上已见状态时截断），
       延拓到 num_inits + length - 1 个值后滑窗，每个状态产出一个长 length 的窗口
    4. 按状态序号 n_before + i < num_samples 分 train / test
    5. train 整体 shuffle
    """
```

## 等价性验证

写一个对照测试：同一 (p, length, a, b, ratio, seed) 下，模块输出与
`RecurrenceDataset(..., num_samples=max(1,int(p²*ratio))).run()` 的
train/test 样本逐点一致（train 为打乱后顺序一致、test 顺序一致）。

## 不做（YAGNI）

- 缺失值污损、混合规则、其他任务类型——同事只要 addition 干净样本。
- 不改 `src/datasets.py`；新模块自带实现，不从 datasets 导入（保证可单独拷走）。
