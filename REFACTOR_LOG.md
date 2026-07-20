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

---

## 13. visualize.py 魔法数字常量化

**问题**：`set_ylim(-0.05, 1.05)` 重复 4 次；loss 轴上限的 `0.95` 分位数、`*1.2` 余量、`0.5` 下限三个数字挤在一行；epoch 刻度数量单图用 `//10`、网格图用 `//5`，差异无说明；`dpi=150` 重复 3 次；`os.path.basename(lp)[:-4]` 手写去扩展名。

**修改**：提取常量 `ACC_YLIM / FIG_DPI / LOSS_YLIM_QUANTILE / LOSS_YLIM_MARGIN / LOSS_YLIM_MIN / MAX_YTICKS_SINGLE / MAX_YTICKS_GRID`（各带注释，两个刻度常量的命名直接体现"单图 vs 网格图"的差异）；`[:-4]` 改为 `os.path.splitext`。

**原因**：同 AA-6；刻度数量不一致从"可疑的笔误"变成"有意的差异"。

**验证**：分组与 --no-group 两种模式出图均正常。

---

## 14. visualize.py parse_log 的 flush 块重复

**问题**：遇到新 Epoch 行时的"把暂存的 per_pos/per_rule 写入 data 并重置"与文件末尾的 flush 是同样的逻辑，写了两遍。

**修改**：提取闭包 `flush_current(cur)`，两处调用。

**原因**：解析状态机的写入规则只保留一份。

**验证**：构造含 per_pos/per_rule 及缺失项的假日志，断言解析结果逐项相等（含末尾 flush 的 None 填充）。

---

## 15. visualize.py exp_name 计算恒为常量，具有误导性

**问题**：`--all` 分支中 `exp_path` 硬编码为 `'experiments/experiments.json'`，却用 `os.path.splitext(os.path.basename(exp_path))[0]` 计算 `exp_name`——结果恒为 `'experiments'`，两行计算给人"路径可变"的错觉。

**修改**：直接写 `'experiments'`，并加注释说明输出目录与 experiments/ 目录的固定对应关系。

**原因**：恒定的值就应该写成常量，计算过程是噪音。

**验证**：py_compile 通过；替换前后 `default_log/default_out` 的取值字符串相等（恒等推导）。

---

## 16. visualize.py colorbar 依赖循环遗留变量 im

**问题**：`plot_per_position` 多 seed 分支的 `fig.colorbar(im, ...)` 使用的 `im` 是 imshow 循环的遗留变量——仅在 `valid_items` 非空时才有定义（恰好由前面的检查保证），这种跨作用域隐式依赖很脆弱：将来有人改动循环或检查顺序就会 NameError。

**修改**：在 colorbar 调用处加注释，显式说明 `im` 来自最后一个面板、非空由上面的检查保证、以及所有面板共享 vmin/vmax 所以共用一个 colorbar。

**原因**：把"恰好成立"的隐式前提写成显式契约，改代码时能立刻看到依赖关系。（此处不加防御性代码，因为前提已由函数内检查保证。）

**验证**：py_compile 通过。

---

## 17. visualize.py main() 输入解析与流程混杂

**问题**：`main()` 约 80 行，参数解析、`--all`/默认/fallback 三套 log/out 目录与 names 的推导、两种绘图模式混在一起。

**修改**：把 names/log_dir/out_dir 的推导（约 30 行）提取为 `_resolve_inputs(args)`，`main()` 只保留流程骨架：解析参数 → 解析输入 → 分组 → 绘图。

**原因**：输入推导是一个有独立语义的单元（docstring 可以说清三套默认规则），提出来之后 main 的结构一眼可读。

**验证**：py_compile + 单日志跑通出图。

---

## 17b. visualize.py _seed_sort_key 改公开名并说明 -1 语义

**问题**：`batch_run.py` 跨模块调用了 visualize 的私有函数 `_seed_sort_key`——下划线前缀向读者声明"这是模块内部细节"，与实际的跨模块使用矛盾。另外空 label 返回 -1（使无 seed 后缀的独立实验排最前）的语义没有说明。

**修改**：改名 `seed_sort_key`（去掉下划线），docstring 补充 -1 语义；visualize.py 内 2 处、batch_run.py 1 处调用同步更新。

**原因**：被跨模块使用的函数就应该是公开 API。

**验证**：排序断言通过（'' < '0' < '2' < '10'，数值序而非字典序）；两文件 py_compile 通过。

---

## 18. batch_run.py 死变量 prefix

**问题**：`run_single` 中 `prefix = f"[{name}] " if concurrency > 1 else ""`（原 `:169`）定义后全文件再无引用，疑似多进程打印改造的遗留。

**修改**：删除该行。

**原因**：死代码。（连带影响：`concurrency` 参数现仅作为签名保留，由调用处传入；不改变任何输出。）

**验证**：py_compile 通过。

---

## 19. batch_run.py 死赋值 returncode = None

**问题**：`returncode = None`（原 `:190`）之后没有任何读取，唯一赋值来源是后面的 `returncode = process.wait()`（`:225`），死赋值。

**修改**：删除该行。

**原因**：死代码；保留会让读者怀疑存在"wait 之前的 returncode 语义"。

**验证**：py_compile 通过。

---

## 20. batch_run.py format_config_table 三段重复与无效分支

**问题**：三个配置段的循环结构完全重复；更离谱的是每段内部 `isinstance(float) / isinstance(bool) / else` 三个分支的函数体一模一样（`f"{key:<25} {value}"`）——类型判断完全不产生差异，是纯噪音。

**修改**：键列表提升为模块常量 `NET_KEYS / DATA_KEYS / TRAIN_KEYS`；段输出提取为 `_append_config_section()`；删除所有无效 isinstance 分支，统一一行格式化。Training 段因多 OPTIMIZER/SCHEDULER 两行保留内联。

**原因**：60 行缩到 35 行；无效分支会让读者停下来找"不同类型到底有什么区别"。

**验证**：构造含各类型值的 config，断言输出段落、分隔线数量与键均正确。

---

## 21. batch_run.py gpu_label 重复计算且内层遮蔽外层

**问题**：`run_single` 外层已算好 `gpu_label`，`read_stdout` 闭包内又用同一表达式重算一遍并遮蔽外层变量。两处独立计算同一逻辑值，将来改一处忘另一处会产生日志不一致。

**修改**：删除闭包内的重算，直接使用外层 `gpu_label`。

**原因**：单一计算源；遮蔽同名变量是经典的维护陷阱。

**验证**：py_compile 通过；表达式逐字相同，等价性显然。

---

## 22. batch_run.py run_single 拆分

**问题**：`run_single` 约 140 行，混杂五个职责：配置合并、临时文件管理、子进程输出泵（两个线程）、注意力分析子进程、清理与汇报。两处 `subprocess.Popen` 的 `python -c` 命令行构造也重复。

**修改**：纯提取式拆分，行为不变——
- `build_merged_config(exp, base_config)`：配置合并 + SAVE_PATH 自动填充，返回 `(name, task, merged_main, merged)`；
- `run_attention_analysis(name, pth_path, log_path, env)`：注意力分析子进程；
- `_spawn_python(statement, env, stderr=...)`：统一的 `python -c` 启动器（两处 Popen 调用点共用）；
- `_decode` 从 run_single 内层提升为模块级（read_stdout/read_stderr/分析三处共用）。

**原因**：每个函数一个职责；`run_single` 剩下约 90 行的线性流程，一眼可读。

**验证**：`build_merged_config` 合并顺序/自动填充断言；`_spawn_python` 实际启动子进程跑通。

---

## 23. batch_run.py 魔法值常量化（含 core.py 的合并标记）

**问题**：`'_BATCH_RUN_MERGED'` 字面量横跨 batch_run.py（写入）和 core.py（3 处消费），拼写漂移不会报错只会静默拒绝运行；`'fibonacci_transformer.pth'` 兜底默认值与 config.json 重复；分隔线宽度 `50`、汇总表宽 `140` 和列宽 `45/6/8` 散布多处，表头与数据行的格式串各自手写数字，改一处忘另一处表格就错位。

**修改**：batch_run.py 定义 `BATCH_RUN_MERGED_FLAG / DEFAULT_SAVE_PATH / SEP_WIDTH / SUMMARY_WIDTH / NAME_COL / EPOCH_COL / NUM_COL`；core.py 定义同名 `BATCH_RUN_MERGED_FLAG`（两模块互为生产者-消费者，各自定义并注释来源，避免训练核心依赖批跑器）。表头与数据行格式串共用列宽常量。

**原因**：跨模块协议（标记键名）必须有名字；表格宽度单一来源。

**验证**：两文件编译通过；断言两模块标记常量相等；format_config_table 输出分隔线宽度不变。

---

## 24. batch_run.py [:-4] 手写去扩展名

**问题**：`generate_grouped_plots` 和 `summarize_experiments` 用 `os.path.basename(log_path)[:-4]` 去掉 `.log`，与 prepare_rerun.py 已在用的 `os.path.splitext` 风格不一致；手写数字 4 依赖"扩展名恰好 4 字符"的隐含假设。

**修改**：两处改为 `os.path.splitext(os.path.basename(log_path))[0]`。

**原因**：语义化 API，与项目内其他文件一致。

**验证**：py_compile 通过。

---

## 25. batch_run.py 跳过逻辑在串行/并行两分支重复

**问题**：`main()` 串行分支和并行分支各写了一遍相同的"resource_stop 时打印跳过原因并记录失败"块。

**修改**：提取闭包 `skip_exp(exp)`，两分支各调一次。

**原因**：跳过行为（打印格式 + 记录方式）单一来源。

**验证**：py_compile 通过。

---

## 26. batch_run.py 汇总表 N/A 格式化重复 7 行

**问题**：`summarize_experiments` 中 7 列各自手写 `f"{...:.1f}" if ... is not None else 'N/A'`，结构完全相同。

**修改**：提取 `fmt(value, spec)` 闭包，列值改为列表推导，数据行用统一的列宽常量 join 输出。

**原因**：7 行重复缩为 1 个辅助函数 + 3 行列表；列的增删只需改列表。

**验证**：构造可解析/不可解析两份假日志实跑 `summarize_experiments`，表格列宽与 N/A 填充正确。

---

## 27. batch_run.py 模块级可变全局目录改显式传参

**问题**：`LOG_DIR / MODEL_DIR / PLOT_DIR` 是模块级可变全局，`main()` 用 `global` 改写，读者需要跨 `main`/`run_single`/`build_merged_config`/`generate_grouped_plots`/`summarize_experiments` 五个函数追踪隐式状态；且原注释 "updated per experiment" 不准确（是每个**批次**更新一次）。

**修改**：删除三个全局；`main()` 计算 `dirs = {'log', 'plot', 'model'}` 字典并显式下传——`run_single(exp, base_config, dirs, ...)`、`build_merged_config(..., model_dir)`、`summarize_experiments(log_dir)`、`generate_grouped_plots(log_dir, plot_dir)`。

**原因**：数据流显式化；函数签名即依赖声明，可独立测试（如本次验证直接以临时目录调用 summarize/plots）。

**验证**：断言模块级全局已不存在；临时目录实跑 summarize_experiments 与 generate_grouped_plots 正常。

**签名变更说明**：以上四个函数均为 batch_run.py 内部函数（无其他模块调用），签名调整不影响外部。

---

## 28. batch_run.py 失效/历史注释清理

**问题**：
1. 原 `:30` 注释 "These are updated per experiment in main()" 不准确（实为每个批次更新一次）——已随第 27 条全局变量的删除一并移除；
2. 原 `:181` "NOTE: main.py was removed; batch_run.py is the only supported entry point." 是历史变更记录，对理解当前代码无帮助（git 历史已保留该信息）。

**修改**：删除第 2 条注释。

**原因**：注释应描述现状，不应承担变更日志职能。

**验证**：py_compile 通过。

---

## 29. prepare_rerun.py help 文本与实际默认值不符

**问题**：`--log-dir` 的 help 写 "default: /data/cxm/<exp_name>/logs"，但实际默认由 `DEFAULT_BASE_DIR = '/data/cxm/recursion'` 推出，是 `/data/cxm/recursion/<exp_name>/logs`。按 help 拼路径会找到错误的目录。

**修改**：help 改为 f-string 引用 `DEFAULT_BASE_DIR` 常量，消除两处来源。

**原因**：文档与代码必须同源。

**验证**：py_compile 通过。

---

## 30. prepare_rerun.py 状态字符串字面量散落

**问题**：`'missing'/'success'/'failed'` 在 `log_status` 的返回值和 `main` 的判断处重复出现，拼写漂移不会产生报错。

**修改**：定义 `STATUS_MISSING / STATUS_SUCCESS / STATUS_FAILED` 常量（各带注释），全部引用处替换。

**原因**：协议性字符串应有单一来源。

**验证**：四种情况（缺文件/返回码 0/非 0/无标记）断言正确。

---

## 31. prepare_rerun.py 与 batch_run.py 的重复定义

**问题**：`load_json`/`save_json` 与 batch_run.py 逐字重复；`DEFAULT_BASE_DIR = '/data/cxm/recursion'` 两处硬编码，改一处忘另一处会导致日志路径错位。

**修改**：prepare_rerun.py 删除本地副本，改为 `from batch_run import load_json, save_json, DEFAULT_BASE_DIR`（batch_run 的 `main()` 有 `__main__` 保护，导入无副作用；唯一代价是会连带导入 visualize→matplotlib，已在注释中说明）。

**原因**：路径常量与 IO 辅助必须单一来源。

**验证**：端到端——构造假 experiments（成功/失败/缺失各一）跑 `main()`，rerun JSON 内容正确、metadata 保留、成功实验的 .err 被清理。

---

## 32. fix_logs.py target_prefix 三处重复且定义在循环内

**问题**：字面量 `"per-rule acc: {0:"` 出现在函数体（循环内每次迭代重复赋值）、docstring、argparse description 三处。

**修改**：提取模块级常量 `BROKEN_LINE_PREFIX`；docstring 改述常量名，description 用 f-string 引用，删除循环内的重复赋值和中间变量 `target_prefix`。

**原因**：修复目标字符串必须单一来源——改 prefix 时漏一处就会清错行。

**验证**：py_compile 通过。

---

## 33. fix_logs.py 命名含糊与参数重赋值

**问题**：`total_files` 实际统计"被修复的文件数"（只在 removed>0 时 +1），与"处理的文件总数"（`len(log_files)`）易混淆；`folder = os.path.abspath(folder)` 直接覆盖入参，函数签名语义被就地改变。

**修改**：`total_files` → `fixed_files`（与输出文案对齐）；绝对路径改用新变量 `folder_abs`。

**原因**：名字应该说出变量的真实含义；不重赋值参数是基本纪律。

**验证**：构造含损坏行/正常两份假日志实跑，损坏行被精确移除、输出计数正确。
