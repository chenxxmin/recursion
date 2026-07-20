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

---

## 5. analyze_attention.py 手写 mean/std 重复三处

**问题**：`verify_qk_properties` 中均值/总体标准差的手写计算重复三次（原 `:322` 还在一条 130+ 字符的 f-string 里内联两遍 `sum(vals)/len(vals)`），可读性差且容易改漏一处。

**修改**：提取模块级辅助函数 `_mean_std(vals)`，三处改为调用。

**原因**：命名即文档；f-string 行长度恢复正常。

**验证**：`_mean_std` 输出与原手算结果数值断言一致。

---

## 6. analyze_attention.py 魔法数字常量化

**问题**：分析脚本中散布多个无语义字面量——top-k 的 `3`（还在一条 130+ 字符的三元表达式里重复 3 次）、距离上限 `63`、熵 epsilon `1e-12`、CV 保护阈值 `1e-6`、打印阈值 `0.01`、每行 `8` 项、分隔线 `'='*70` 重复 7 处。

**修改**：模块顶部定义常量 `TOP_K / MAX_DISTANCE / ENTROPY_EPS / CV_MEAN_EPS / MIN_PRINT_VAL / ITEMS_PER_LINE / HEADER_WIDTH`（各带注释）；两处标题打印提取为 `_print_header()`；过长的 topk 三元表达式拆为 if/else。

**原因**：字面量有了名字就有了含义；调整参数时改一处即可。

**验证**：py_compile + 小模型实跑 `summarize_attention_for_sequence` / `print_attention_summary` / `verify_qk_properties`，输出格式与之前一致。

---

## 7. analyze_attention.py 命名与 import 风格清理

**问题**：
1. `import core as main`——别名 `main` 通常指程序入口函数，这里却是模块，易误读；
2. `import random` 写在 `analyze_model_attention` 函数体内；
3. `n_head, _, _ = att.shape` 解包两个弃值，不如直接 `att.shape[0]`；
4. `for i in range(init_len, max_len)` 的循环变量 `i` 未使用。

**修改**：别名改回 `import core`（4 处 `main.` 引用同步改为 `core.`）；`import random` 上移到文件顶部；`n_head = att.shape[0]`；循环变量改 `_`。

**原因**：符合 Python 惯例，消除"main 是什么"的认知负担。

**验证**：py_compile + 小模型实跑 summarize 流程正常；qk_verification 同步编译通过。

---

## 8. analyze_attention.py analyze_model_attention 拆分

**问题**：`analyze_model_attention` 约 100 行，混杂三类职责：从 config 推断递推规则（5 个分支）、dynamic_mixed 测试序列与 query mask 生成（2 个嵌套函数）、汇总调用。嵌套函数依赖闭包变量，阅读时需要在 100 行内追踪多个隐式状态。

**修改**：提取三个模块级函数——`_resolve_recurrence(config)`（返回描述递推规则的 dict）、`_make_dynamic_seq(p, ab_pairs, flag_start_id, length, seed)`、`_make_dynamic_query_mask(length)`；主函数只保留"加载模型 → 生成序列 → 汇总打印"的流程骨架，原来的列表包装（`test_sequences[0]`）简化为标量。

**原因**：每个函数一个职责，参数显式传递取代闭包隐式依赖。

**验证**：`_resolve_recurrence` 五个分支单测；小 checkpoint 端到端跑通 addition 路径。

**⚠️ 发现的既有 bug（未修，保持等价）**：`_make_dynamic_query_mask` 中 `query_pos = 2*(k-1)` 与其注释矛盾——注释说"x3 在输入下标 3"，但公式给出 4。按输入布局 `[x1,x2,f3,x3,...]`，x_k 的下标应为 `2*(k-1)-1`。该 bug 在重构前即存在（见 commit d6b0ed1 第 539 行），且最后一个 k 必然越界——dynamic_mixed 的注意力分析路径实际上从未成功运行过。是否修复待确认（修复属于行为变更，超出等价重构范围）。

---

## 9. analyze_attention.py 补充 PE 方案推断注释

**问题**：`load_model` 通过 state_dict 键名反推 `use_learnable_pe`（看到 `alibi_slopes` 判 True、看到 `rope.inv_freq` 判 False），这条因果链不显然，读者需要知道两种位置编码在 checkpoint 里留下的"指纹"才能理解。

**修改**：补充注释说明两种位置编码各自在 checkpoint 中留下什么键。

**原因**：推断逻辑的有效性依赖模型实现细节，必须把这条隐式知识写显式。

**验证**：仅注释变更，py_compile 通过。

---

## 10. visualize.py plot_setting_group 是死代码，main 内联重复了同样逻辑

**问题**：`plot_setting_group`（`:454-480`）已实现了"过滤空数据 → 拼 base 路径 → 调三个 plot"的完整分组绘图流程，但 `main()` 的分组分支没有调用它，而是内联重写了同样的逻辑。两份实现会漂移。

**修改**：`main()` 分组分支保留逐日志解析（保留每条空日志的提示信息），尾部改为调用 `plot_setting_group`，删除内联的 base 拼接和三个 plot 调用。

**原因**：单一实现；`plot_setting_group` 本就是为此设计的。

**行为差异（仅日志输出）**：某 setting 所有日志都无 epoch 数据时，现在会额外打印一行 "No epoch data for setting X, skipping."（来自 plot_setting_group 自身的检查），信息更明确。

**验证**：构造两个假日志跑 `visualize.py`，分组输出的曲线图/热力图正常生成。

---

## 11. visualize.py 子图网格样板代码重复三次

**问题**：三个绘图函数的多 seed 分支各自写了一遍相同的网格样板：`ncols=4` / nrows 计算 / `figsize=(3.2*ncols, 2.5*nrows)` / flatten / 结尾"关闭多余子图"循环。

**修改**：提取 `_make_subplot_grid(n, ncols=4)`（创建网格并隐藏多余子图），三处改为调用，删除三处收尾的 `axis('off')` 循环。

**原因**：网格布局规则只保留一份；各绘图函数只关心自己的内容。

**验证**：5 个 seed 的假日志出图正常（2 行网格、3 个空位正确隐藏），三种图均生成。

---

## 12. visualize.py single_mode 判定重复三处且依赖隐式约定

**问题**：`len(items) == 1 and items[0][0] == ''` 在三个绘图函数中重复出现，其正确性依赖 `_data_items` 用空串标记"无标签单实验"这一隐式约定，读者不跨函数对照就无法理解空串的含义。

**修改**：提取 `_is_single(items)`；在 `_data_items` docstring 中写明空串约定。

**原因**：把隐式约定变成有名字的显式判断。

**验证**：py_compile 通过；replace_all 精确替换 3 处。
