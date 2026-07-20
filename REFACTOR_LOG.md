# 等价重构记录

本文档逐条记录代码可读性重构：问题是什么、改了什么、为什么。所有修改均为**行为等价**修改（除特别注明外），每条对应一个 commit。

---

## 0-1. core.py 三个 collate_fn 被重复定义（commit 2a5ff14 的一部分）

**问题**：`collate_fn` / `mixed_ab_collate_fn` / `dynamic_mixed_collate_fn` 在 core.py 中各被定义 2~3 次（原 :149/:153/:159、:173/:177/:183、:851/:922），后者静默覆盖前者。读者无法判断哪个版本生效；其中带 BatchTag 的中间版给人"tag 机制已生效"的错觉。

**修改**：删除被覆盖的旧定义，每个函数只保留最终生效的一个。

**原因**：同名重复定义是严重的误导源，且死代码会让后续维护者在错误的版本上修改。

---

## 0-2. 补全 BatchTag 机制（commit 2a5ff14）

**问题**：BatchTag 机制只实现了一半——`mixed_ab`/`dynamic_mixed` 的 collate_fn 不带 tag，`train_epoch`/`evaluate` 里的 tag 分支永远走不到，实际靠 `batch[1].ndim == 2` 这种维度猜测区分数据格式。畸形 batch 会被静默误判。

**修改**：
- `BatchTag` 由 dict 改为 `enum.IntEnum`（大写成员），引用拼写错误在 import 期即暴露；
- 三个 collate_fn 统一返回 `(x, tag, *payload)`；
- 新增 `_unpack_batch` 统一解包，校验 payload 数量，畸形输入立即 `ValueError`；
- `train_epoch`/`evaluate` 各约 30 行的解包+fallback 逻辑替换为一行调用，删除不可达的裸张量分支。

**原因**：显式标记替代形状猜测，数据格式错误在第一时间报错而非被静默路由到错误分支；同时消除了 train/evaluate 之间的大段重复。

**验证**：CPU smoke 测试——三种任务各跑一步 train/evaluate（loss 正常、mixed_ab 分组准确率正常）；5 种畸形 batch 全部被拒绝。

---

## 0-3. qk_verification.py 整体复制 analyze_attention.py（commit d6b0ed1）

**问题**：`extract_qk_raw_scores`（57 行）和 `verify_qk_properties`（104 行）从 analyze_attention.py 逐字复制（仅差行尾空白、一处函数内 import、一处 docstring）。两份副本会随时间漂移；且两处 docstring 对 query mask 长度描述矛盾（T vs T-1）。

**修改**：qk_verification.py 删除两个本地函数，改为 `from analyze_attention import load_model, verify_qk_properties`；顺带删除不再使用的 `math`/`F` import。修正 analyze_attention.py:311 docstring 为 "length T"（mask 按输入位置 `i in range(T)` 索引，原 "T-1" 描述错误）。

**原因**：单一实现源，消除副本漂移。

**验证**：两文件编译通过；`qk_verification.verify_qk_properties is analyze_attention.verify_qk_properties` 断言成立。

---

## 1. analyze_attention.py 前向骨架重复（get_attention_weights vs extract_qk_raw_scores）

**问题**：两个函数各自手写了一遍相同的逐层前向逻辑（LayerNorm → QKV → RoPE → mask → softmax → 残差 → MLP，约 35 行），仅"每层收集什么"不同。模型结构改动时两处必须同步修改，漏改不会报错但分析结果会错。

**修改**：`extract_qk_raw_scores` 的返回 dict 中 `'attn_weights'` 字段即为 `get_attention_weights` 的全部输出（前者是后者的超集），故 `get_attention_weights` 改为一行包装：调用 `extract_qk_raw_scores` 并提取 `'attn_weights'`。删除约 40 行重复。

**原因**：前向逻辑只保留一份，所有分析函数共享同一实现，消除副本漂移风险。

**验证**：小模型上断言两函数输出的 attention weights 逐元素相等（`torch.equal`）。

---

## 2. analyze_attention.py 死参数 p（summarize_attention_for_sequence）

**问题**：`summarize_attention_for_sequence(model, seq, p=None, query_mask=None)` 的参数 `p` 在函数体内从未被引用，但调用处（`analyze_model_attention`）还在传 `p=p`，给人"模数会参与汇总计算"的错觉。

**修改**：从签名和唯一调用处删除该参数。

**原因**：死参数误导读者去理解一个不存在的数据流。

**验证**：grep 确认函数体内无 `p` 引用；py_compile 通过。

---

## 3. analyze_attention.py load_model 的两处逻辑噪音

**问题**：
1. `:35` 条件 `'transformer.h.0.mlp.0.weight' in k or k.endswith('.mlp.0.weight')` 中，前半句恒被后半句覆盖（该字符串本身就以 `.mlp.0.weight` 结尾），是无效冗余。
2. `:43-46` `num_ab_pairs = len(...) if config.get('ab_pairs') else 1` 的结果最小为 1，紧随的 `if num_ab_pairs == 0` 分支不可达；且该分支引用的 `ab_emb.weight` 在当前模型结构中并不存在，若真走到反而会 KeyError。

**修改**：1 的条件只保留 `endswith`；2 删除不可达分支。

**原因**：冗余条件和不可达分支会让读者误以为存在需要兼容的历史情况。

**验证**：py_compile 通过；逻辑等价性由条件包含关系直接得证。

---

## 4. analyze_attention.py 生成 5 条测试序列但只用第 0 条

**问题**：`analyze_model_attention` 中 dynamic_mixed 分支生成 5 条测试序列和 5 个 query mask（原 `:500-502`），但函数体只使用 `test_sequences[0]`（QK 验证已移至 qk_verification.py）。其余 4 条是纯粹的计算浪费，且 `:457` 注释 "1 for attention visualization, 5 for QK property verification" 描述的是已不存在的行为，打印的 `1/{len}` 序号也暗示存在多条序列。

**修改**：只生成 1 条序列和 1 个 mask；注释改为说明 QK 验证在 qk_verification.py；打印文案去掉 `1/N` 序号。

**原因**：消除死计算和过时注释，避免读者去找"另外 4 条序列用在哪"。

**验证**：py_compile 通过；唯一消费点 `test_sequences[0]` 的取值不变（seed=0 即原第 0 条）。
