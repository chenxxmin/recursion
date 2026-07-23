# mixed_abc（tribonacci 混合规则）架构设计

日期：2026-07-22
状态：已与用户逐节确认

## 背景与目标

当前代码库支持单规则递推任务（addition/tribonacci/nonlinear 等）和二阶混合任务
`mixed_ab`（多个 `(a,b)` 二阶递推规则各生成数据后混合训练，支持 label flag 与
cond_wte 两种规则提示方式）。目标是支持 **tribonacci（三阶递推）的混合规则实验
`mixed_abc`**：多条 `(a,b,c)` 规则各自生成数据后拼接训练，复用现有训练/评估代码。

已确认的决策：

- 规则抽象只做到**线性递推规则**（系数元组 + 阶数），不做通用 callable 规则（YAGNI）。
- 代码组织：先把规则提成独立的 **Rule 模块**，再写 **mixed_dataset 模块**，其输入是
  `list[Rule]`；core.py 只做接线。
- 本轮只做架构，不写实验配置 JSON。
- 同一混合实验内所有规则必须同 P、同阶（assert 限制），跨阶混合留待以后。

## 总体结构

```
src/rules.py          新：LinearRecurrenceRule + rules_from_config()
src/mixed_dataset.py  新：MixedRecurrenceDataset（输入 list[LinearRecurrenceRule]）
src/core.py           改：接线与泛化（任务分发、prepare、生成测试、rule_start_offset）
src/config.json       改：新增 mixed_abc section
```

## 第 1 节：规则模块 `src/rules.py`

```python
@dataclass(frozen=True)
class LinearRecurrenceRule:
    coeffs: tuple[int, ...]   # X(n) = (c1*X(n-1) + c2*X(n-2) + ...) mod p
    p: int
    name: str = ""            # 如 "a1b2c4"，缺省自动生成

    @property
    def order(self) -> int:   # = len(coeffs)
        ...

    def next_fn(self):
        """返回与 RecurrenceDataset 兼容的闭包 fn(seq, p)。"""
```

- `next_fn()` 语义与现有闭包**逐值一致**：`(a,b)` → `a*seq[-1]+b*seq[-2]`，
  `(a,b,c)` → `a*seq[-1]+b*seq[-2]+c*seq[-3]`（mod p）。保证旧实验数值不变。
- `rules_from_config(cfg, order)`：
  - `mixed_ab`（order=2）：从 `AB_PAIRS`（二元组列表）造规则；
  - `mixed_abc`（order=3）：从新配置键 `ABC_PAIRS`（三元组列表）造规则；
  - 校验：规则数量 ≥ 1、每条系数个数 == order。

## 第 2 节：混合数据集模块 `src/mixed_dataset.py`

`MixedRecurrenceDataset`：

- 输入：`rules: list[LinearRecurrenceRule]`、per-rule 样本数（由
  `MIXED_AB_MAX_UNIQUE_RATIOS` / `MAX_UNIQUE_RATIO` 推出，沿用现有语义）、`length`、
  `use_ab_tag` 等，与现 `MixedABDataset` 参数对齐。
- 生成流程（与现 `_build_mixed` 一致，仅把硬编码二阶闭包换成规则对象）：
  1. 逐规则调用现有 `RecurrenceDataset`（`init_len=rule.order`、
     `recurrence_fn=rule.next_fn()`）完成环遍历 + train/test 状态划分；
  2. 各规则样本 extend 成扁平列表，标签 = 规则索引 int；
  3. train 的 (seq, rule_idx) 对做一次 shuffle（用全局 random，保持 seed 语义）。
- `use_ab_tag=True` 时在样本头部前插 flag token = `p + rule_idx`（与现机制相同）。
- `__getitem__` 返回 `(seq, rule_idx)`，直接复用现有 `mixed_ab_collate_fn` /
  `BatchTag.MIXED_AB` / `_unpack_batch` 管线。
- assert：所有规则同 P、同阶。

## 第 3 节：模型与训练（复用，仅两处小改）

- `MixedABTransformer` 原样承载三阶规则：`vocab_size = p + 1 + N_rules`、pad /
  restricted token ids、cond_wte 分段 embedding（`COND_WTE_SHARED_RATIO`）、
  rule_head 辅助分类、`freeze_partial` 冻结逻辑——全部只依赖规则数量 N，与阶数
  无关。label flag 与 cond_wte 两种模式对三阶规则零改动可用。
- 训练/评估管线零改动：`BatchTag.MIXED_AB` → `extra_kwargs_fn` 传 `ab_labels`；
  evaluate 的 per-rule group_acc、`run_training_engine`、cond_fix 触发、
  per-rule 打印全部复用。
- 仅改两处：
  1. `rule_start_offset`（core.py:1017）由硬编码 3/4 改为按
     `order + 1 (+1 if use_ab_tag)` 计算；
  2. NUM_MASK：保持可配置；`mixed_abc` 默认 2（与 tribonacci 单任务实验一致，
     便于对比）；旧 mixed_ab 实验的显式 `NUM_MASK: 2` 不受影响。

## 第 4 节：core.py 接线、配置与验证

- 任务分发（`run_experiment`）：新增 `mixed_abc` task；`_prepare_mixed_ab` 泛化为
  `_prepare_mixed_recurrence(cfg, order)`，`mixed_ab`（order=2）与
  `mixed_abc`（order=3）共用；`AB_PAIRS`/`ABC_PAIRS` 经 `rules_from_config`
  统一成规则列表。
- 泛化点：
  - `state_space_size = P ** order`；
  - `MIXED_AB_MAX_UNIQUE_RATIOS` 长度校验改为 `len(rules)`；
  - 训练后生成测试的 mixed_ab 分支：`itertools.product(range(P), repeat=order)`、
    exposed 初值取前 order 个、真值递推改用 `rule.next_fn()`
    （single_recurrence 分支已是 init_len 参数化，作模板）；
  - `_print_task_banner` 文案按规则列表打印。
- 配置：
  - 旧 mixed_ab 实验 JSON **零改动**，行为不变；
  - `src/config.json` 新增 `mixed_abc` section（默认值同 mixed_ab：
    `USE_AB_TAG=false, USE_CONDITIONAL_WTE=false, COND_WTE_SHARED_RATIO=0.0,
    MIXED_AB_MAX_UNIQUE_RATIOS=[0.7, ...]`）。
- 验证：
  1. `next_fn()` 与现有二阶/三阶闭包逐值单元测试；
  2. 兼容性冒烟：同 seed 下用旧 `AB_PAIRS` 配置走新管线，生成的 train/test 数据
     与旧实现**逐样本一致**；
  3. 三阶端到端小跑：P=23、2 条规则、少量 epoch，走通训练 + per-rule 评估 +
     生成测试。

## 明确不做（YAGNI）

- 通用 callable / nonlinear 规则抽象；
- 跨阶混合（不同 order 的规则混在同一实验）；
- 实验配置 JSON（等架构落地后另行决定规则组合）；
- `TARGET_INIT_COVERAGE`（死键，不引入新代码路径）。
