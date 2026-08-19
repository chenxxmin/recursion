# rule_fit missing 模式改队列驱动分析 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `src/rule_fit.py` 的 missing 实验分析（`run_missing_categories`）从"静态全枚举 + on-manifold 拟合"改为"从 (x,x) 出发的 BFS 队列 + 随机探针假设检验式拟合"，同时消除 on-manifold 窗口秩亏导致的 `no linear fit (-100%)` 问题。

**Architecture:** 两遍结构。Pass 1 不变（真实递推序列自然污损，按深度 d_max 的 pattern 分桶，累计 n / model-vs-truth / 实时注意力）。Pass 2 改为 BFS：根为全可见 pattern `(x,)*init_len`；每个 pattern 汇总后缀匹配桶的统计，注意力显著（≥0.10）且可见的距离构成假设集 C；用**随机探针**（iid 序列，只强制窗口内的 M 位，其余全可见）拟合 C 的系数；拟合公式的非零系数距离是"前提"，把"前提中某些距离也被污损"的更深 pattern 入队（去重，长度上限 d_max=5）。

**Tech Stack:** Python 3 + torch（仅前向）。测试环境：`C:/Users/Chen/anaconda3/envs/torch/python.exe`（无 pytest，用 `__main__` runner，仓库惯例）。

## Global Constraints

- 本机无真实 checkpoint（在服务器 `/data/cxm/...`），验证靠合成/stub + 小模型冒烟；端到端需用户在服务器跑真实批次。
- **seed 语义**：`run_one` 在 `random.seed(args.seed)` 之后才调用本函数；本设计的探针消耗 RNG 在该点之后，合法，但输出与旧版不可比（方法论已变，可接受）。
- 只做单层模型的解读；`len(model.transformer.h) > 1` 时打印 warning 继续跑（多层 off-manifold 解读以后另议）。
- nonlinear/multiplication 规则非线性，线性拟合本就无意义（旧版同样如此），不特殊处理。
- 用户已授权**直接在 main 分支上逐任务 git commit**（2026-08-18 确认）；每个 Task 完成后提交一次。
- 现有测试 8 个文件全绿，改完不得破坏；rule_fit 本身此前零测试，本计划新增 `tests/test_rule_fit_missing.py`。

## 背景：为什么旧输出是 "no linear fit (-100%)"

旧版在真实递推序列上拟合 k 个距离的系数。递推阶数为 2 的序列，任意窗口在 F_p 上最多张成 2 维 → k>2 的设计矩阵必然秩亏 → `solve_mod_p` 在整体解和全部 300 轮 RANSAC 中都返回 None → `fit_and_score` 返回 `(None, -1.0, False)` → 打印 `-100.0%`（sentinel 未更新）。随机探针特征天然满秩，新设计从根上消除该问题。

---

### Task 1: 纯函数助手 + 单测

**Files:**
- Modify: `src/rule_fit.py`（在 `fit_and_score` 之后新增函数；文件顶部 import 增加 `itertools`）
- Test: `tests/test_rule_fit_missing.py`（新建）

**Interfaces:**
- Produces（后续任务依赖的确切签名）:
  - `visible_distance(P, d) -> bool`：P 为 oldest-first 的 mask tuple（True=污损）；d≥len(P) 恒 True（探针只强制窗口内 mask）。
  - `patt_label(P) -> str`：`(x,M)` 格式。
  - `child_patterns(P, deps, d_max) -> list[tuple]`：deps 中任意非空子集被追加污损得到的子 pattern；窗口不够长时向左扩展（新位默认可见= False）；长度超 d_max 的子 pattern 丢弃。
  - `aggregate_buckets(buckets, P, d_max) -> (n, agree, attn)`：合并所有长度 d_max 桶中长度 len(P) 后缀 == P 的记录；attn 结构 `{(li,h): {d: [sum, cnt]}}`。
  - `select_C(attn, P, threshold=0.10) -> list[int]`：任一 (layer,head) 合并均值 ≥ threshold 的距离，过滤到 P 中可见者，升序。
  - `rule_coeffs(next_fn, init_len, p) -> (coeffs, base)`：one-hot 探测线性规则系数，`coeffs[d]` 是 x_{t-d} 的系数（d 从 0 起）。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_rule_fit_missing.py`：

```python
"""Unit tests for the queue-driven missing-pattern analysis in rule_fit.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from rule_fit import (aggregate_buckets, child_patterns, patt_label,
                      rule_coeffs, select_C, visible_distance)


def test_visible_distance():
    P = (False, True)  # (x,M): d0 masked, d1 visible
    assert not visible_distance(P, 0)
    assert visible_distance(P, 1)
    assert visible_distance(P, 5)  # beyond the window: always visible in probes


def test_patt_label():
    assert patt_label((False, True)) == '(x,M)'
    assert patt_label((True, False, True)) == '(M,x,M)'


def test_child_patterns_root():
    # (x,x) with deps {0,1} -> mask any non-empty subset
    ch = set(child_patterns((False, False), {0, 1}, 5))
    assert ch == {(False, True), (True, False), (True, True)}, ch


def test_child_patterns_extends_window():
    # (x,M) with deps {1,2}: d1 -> (M,M); d2 outside window -> extend to (M,x,M);
    # both -> (M,M,M)
    ch = set(child_patterns((False, True), {1, 2}, 5))
    assert ch == {(True, True), (True, False, True), (True, True, True)}, ch


def test_child_patterns_depth_cap():
    # dep at d5 needs a length-6 pattern -> dropped when d_max=5
    assert child_patterns((False, True), {5}, 5) == []
    # dep at d4 fits exactly in length 5
    ch = child_patterns((False, True), {4}, 5)
    assert ch == [(True, False, False, False, True)], ch


def test_aggregate_buckets_and_select_C():
    buckets = {
        (False, False, False, False, True): {'n': 100, 'agree': 99,
            'attn': {(0, 0): {0: [90.0, 100], 1: [80.0, 100], 2: [5.0, 100]}}},
        (True, False, False, False, True): {'n': 50, 'agree': 40,
            'attn': {(0, 0): {1: [45.0, 50], 3: [20.0, 50]}}},
        (False, False, False, True, False): {'n': 70, 'agree': 70,
            'attn': {(0, 0): {0: [70.0, 70]}}},
    }
    n, agree, attn = aggregate_buckets(buckets, (False, True), 5)
    assert (n, agree) == (150, 139)
    # merged means: d0 = 90/150 = 0.60 (but masked in (x,M) -> excluded),
    # d1 = 125/150 >= 0.10, d2 = 5/150 < 0.10, d3 = 20/150 = 0.133 >= 0.10
    assert select_C(attn, (False, True)) == [1, 3]
    # suffix match works for full-depth pattern too
    n2, agree2, _ = aggregate_buckets(buckets, (False, False, False, True, False), 5)
    assert (n2, agree2) == (70, 70)


def test_rule_coeffs():
    next_fn = lambda s: (s[-1] + 2 * s[-2]) % 127
    coeffs, base = rule_coeffs(next_fn, 2, 127)
    assert coeffs == [1, 2] and base == 0
    next_fn3 = lambda s: (3 * s[-1] + s[-2] + 2 * s[-3]) % 127
    coeffs3, _ = rule_coeffs(next_fn3, 3, 127)
    assert coeffs3 == [3, 1, 2]


if __name__ == '__main__':
    test_visible_distance()
    test_patt_label()
    test_child_patterns_root()
    test_child_patterns_extends_window()
    test_child_patterns_depth_cap()
    test_aggregate_buckets_and_select_C()
    test_rule_coeffs()
    print("ALL TESTS PASSED: test_rule_fit_missing.py")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_rule_fit_missing.py`
Expected: ImportError（`visible_distance` 等尚不存在）

- [ ] **Step 3: 实现助手函数**

`src/rule_fit.py` 顶部 import 块加 `import itertools`。在 `fit_and_score` 之后插入：

```python
EXPAND_MIN_AGREEMENT = 0.9   # only expand BFS children from a fit at least this good
FIT_MAX_ROWS = 2000          # cap equations fed to fit_and_score (RANSAC scoring cost)


def visible_distance(P, d):
    """True if distance d (0 = last token) is visible in pattern P (oldest-first
    mask tuple). Distances beyond the window are always visible: probes only
    force masks inside the window."""
    L = len(P)
    return d >= L or not P[L - 1 - d]


def patt_label(P):
    return '(' + ','.join('M' if m else 'x' for m in P) + ')'


def child_patterns(P, deps, d_max):
    """Patterns obtained by additionally masking any non-empty subset of deps
    (dependency distances of P's fitted formula). The window extends left (new
    positions default to visible) when a dependency lies outside it; children
    longer than d_max are dropped."""
    L = len(P)
    children = []
    for r in range(1, len(deps) + 1):
        for S in itertools.combinations(sorted(deps), r):
            new_len = max(L, max(S) + 1)
            if new_len > d_max:
                continue
            child = [False] * new_len
            for d in range(L):
                child[new_len - 1 - d] = P[L - 1 - d]
            for d in S:
                child[new_len - 1 - d] = True
            children.append(tuple(child))
    return children


def aggregate_buckets(buckets, P, d_max):
    """Merge depth-d_max bucket records whose length-len(P) suffix equals P.
    Returns (n, agree, attn) with attn {(li,h): {d: [sum, cnt]}}."""
    L = len(P)
    n, agree = 0, 0
    attn = {}
    for B, rec in buckets.items():
        if B[d_max - L:] != P:
            continue
        n += rec['n']
        agree += rec['agree']
        for key, dd in rec['attn'].items():
            acc = attn.setdefault(key, {})
            for d, (s, c) in dd.items():
                if d in acc:
                    acc[d][0] += s
                    acc[d][1] += c
                else:
                    acc[d] = [s, c]
    return n, agree, attn


def select_C(attn, P, threshold=SIG_THRESHOLD):
    """Hypothesis set: distances whose merged mean attention reaches `threshold`
    in any (layer, head), restricted to distances visible in P, sorted."""
    sig = set()
    for dd in attn.values():
        for d, (s, c) in dd.items():
            if s / c >= threshold:
                sig.add(d)
    return sorted(d for d in sig if visible_distance(P, d))


def rule_coeffs(next_fn, init_len, p):
    """Extract a linear rule's coefficients by probing next_fn with one-hot
    histories. Returns ([c_d0, c_d1, ...], base); meaningful for linear rules
    (addition/tribonacci) only."""
    base = next_fn([0] * init_len)
    coeffs = []
    for d in range(init_len):
        v = [0] * init_len
        v[init_len - 1 - d] = 1
        coeffs.append((next_fn(v) - base) % p)
    return coeffs, base
```

- [ ] **Step 4: 跑测试确认通过**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_rule_fit_missing.py`
Expected: `ALL TESTS PASSED: test_rule_fit_missing.py`

---

### Task 2: 随机探针生成 `probe_equations` + stub 模型测试

**Files:**
- Modify: `src/rule_fit.py`（Task 1 助手之后新增）
- Test: `tests/test_rule_fit_missing.py`（追加）

**Interfaces:**
- Consumes: `fit_and_score(X, y, p)`（既有）、`visible_distance`（Task 1）
- Produces: `probe_equations(model, P, C, length, n_probes, p) -> (X, y)`
  - X 的每行是 `[view[t-d] for d in C]`，y 是 `preds[t]`；保证 X 中无 mask token。
  - 窗口间距规则：`t0 = max(L-1, max(C))`，`ts = range(t0, length-1, L + max(C))`——保证其它窗口强制的 M 不会落入本窗口起 max(C) 范围内（否则特征会被邻近窗口的 mask 污染）。

- [ ] **Step 1: 追加失败测试**

`tests/test_rule_fit_missing.py` 追加（顶部 import 增加 `import random`、`import torch`，from rule_fit 增加 `probe_equations, fit_and_score`）：

```python
class _StubModel:
    """A perfect textbook model: x_{t+1} = x_t + x_{t-1} when x_t is visible,
    else the unrolled 2*x_{t-1} + x_{t-2}. Mimics the (logits, loss) return."""

    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        view = x[0].tolist()
        T = len(view)
        logits = torch.zeros(1, T, self.p + 1)
        for t in range(1, T):
            if view[t] == self.p:
                pred = (2 * view[t - 1] + view[t - 2]) % self.p
            else:
                pred = (view[t] + view[t - 1]) % self.p
            logits[0, t, pred] = 1.0
        return (logits, None)


def test_probe_equations_recovers_stub_formulas():
    random.seed(0)
    model = _StubModel(127)
    # (x,M): fit on d1,d2 must recover the unrolled 2*x_{t-1} + x_{t-2}
    X, y = probe_equations(model, (False, True), [1, 2], 32, 50, 127)
    assert len(X) > 100
    assert all(v != 127 for row in X for v in row), "masked token leaked into features"
    coeffs, acc, exact = fit_and_score(X, y, 127)
    assert exact and coeffs == [2, 1], (coeffs, acc)
    # (x,x): fit on d0,d1 recovers the base rule
    X, y = probe_equations(model, (False, False), [0, 1], 32, 50, 127)
    coeffs, acc, exact = fit_and_score(X, y, 127)
    assert exact and coeffs == [1, 1], (coeffs, acc)


def test_probe_equations_neighbour_window_masks_do_not_leak():
    # C includes distances beyond the window: neighbours' forced masks must not
    # land inside [t - max(C), t]
    random.seed(0)
    model = _StubModel(127)
    X, y = probe_equations(model, (False, True), [1, 2, 3, 4, 5], 32, 20, 127)
    assert len(X) > 0
    assert all(v != 127 for row in X for v in row)
```

`__main__` runner 追加两行调用。

- [ ] **Step 2: 跑测试确认失败**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_rule_fit_missing.py`
Expected: ImportError（`probe_equations` 不存在）

- [ ] **Step 3: 实现 `probe_equations`**

`src/rule_fit.py`（Task 1 助手之后）：

```python
def probe_equations(model, P, C, length, n_probes, p):
    """Random off-manifold probes for pattern P: iid uniform sequences with only
    P's own window positions forced to the missing token; everything else stays
    visible. Random features are full-rank, so fitting C's coefficients is
    always solvable (unlike on-manifold windows, which are rank-deficient for
    k > recurrence order). Returns (X, y): features at distances C, target =
    model prediction at each window end."""
    L = len(P)
    c_max = max(C)
    # window ends are spaced so other windows' forced masks can never land
    # within [t - c_max, t] of a sampled window end
    t0 = max(L - 1, c_max)
    ts = list(range(t0, length - 1, L + c_max))
    X, y = [], []
    for _ in range(n_probes):
        view = [random.randrange(p) for _ in range(length)]
        for t in ts:
            for k in range(L):
                if P[k]:
                    view[t - L + 1 + k] = p
        x = torch.tensor([view], dtype=torch.long)
        with torch.no_grad():
            preds = model(x)[0][0].argmax(dim=-1).tolist()
        for t in ts:
            X.append([view[t - d] for d in C])
            y.append(preds[t])
    return X, y
```

- [ ] **Step 4: 跑测试确认通过**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_rule_fit_missing.py`
Expected: `ALL TESTS PASSED: test_rule_fit_missing.py`

---

### Task 3: 重写 `run_missing_categories`（BFS 主流程）+ 冒烟测试

**Files:**
- Modify: `src/rule_fit.py:113-221`（整函数重写）、模块 docstring 第 4-6 条、`--depth` 的 help 文本
- Test: `tests/test_rule_fit_missing.py`（追加冒烟测试）

**Interfaces:**
- Consumes: Task 1/2 全部函数；既有 `corrupt_window`、`extract_qk_raw_scores`、`fit_and_score`、`fmt_equation`、`SIG_THRESHOLD`。
- Produces: `run_missing_categories(args, model, cfg, next_fn, init_len, p, exp_name, desc)`——签名不变，`run_one` (:247) 调用点不动。

**要删除的旧逻辑**：`cat4`/`CATS` 顶层四类分组及其打印；`set_A`/`dists_C_full`；pass 1 里的 `'fits'` 累计；旧的 "Deep patterns" 打印循环。`lo`/`start` 语义保留（`start = max(max(num_mask,1), d_max-1)`）。

- [ ] **Step 1: 追加失败测试（小模型端到端冒烟）**

`tests/test_rule_fit_missing.py` 追加（import 增加 `import contextlib, io`，`from types import SimpleNamespace`，`from models import FibonacciTransformer`，from rule_fit 增加 `run_missing_categories`）：

```python
def test_run_missing_categories_smoke():
    """Untrained tiny model: the BFS machinery must run end-to-end and report
    the root pattern. Fit content is meaningless for an untrained model."""
    random.seed(0)
    torch.manual_seed(0)
    model = FibonacciTransformer(p=7, d_model=32, n_head=2, n_layer=1, block_size=16)
    model.eval()
    args = SimpleNamespace(length=12, min_n=5, samples=30, depth=None)
    cfg = {'MISSING_PROB': 0.3, 'MISS_LEN': 2, 'TRAIN_LEN': 12}
    next_fn = lambda s: (s[-1] + s[-2]) % 7
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_missing_categories(args, model, cfg, next_fn, 2, 7, 'smoke', 'X(k)=(1*X(k-1)+1*X(k-2)) mod 7')
    out = buf.getvalue()
    assert 'Model: smoke' in out
    assert '== pattern (x,x)' in out, out
    assert 'model-vs-truth' in out
```

`__main__` runner 追加调用。

- [ ] **Step 2: 跑测试确认失败**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_rule_fit_missing.py`
Expected: 冒烟测试失败（新版函数体尚未写入；旧版输出无 `== pattern (x,x)` 格式或直接报错——确认失败信息来自断言而非环境）

- [ ] **Step 3: 重写 `run_missing_categories`**

整体替换 `src/rule_fit.py:113-221` 的函数：

```python
def run_missing_categories(args, model, cfg, next_fn, init_len, p, exp_name, desc):
    """Missing-experiment mode, queue-driven edition.

    BFS over corruption patterns rooted at the all-visible pattern
    (x,)*init_len. Pass 1 buckets positions of real corrupted recurrence
    sequences by the depth-d_max pattern (shorter patterns are derived by
    suffix merges). Each queued pattern with n >= min-n reports:
      - model-vs-truth agreement and live per-distance attention (real data)
      - a coefficient fit over the attention-significant visible distances
        (set C), done on RANDOM off-manifold probes (probe_equations)
    A fitted formula's nonzero-coefficient distances are its premises: every
    pattern obtained by masking some of them (child_patterns) is enqueued, up
    to pattern length d_max (default miss_len + 2; --depth overrides).
    Probe fitting targets single-layer models (a warning is printed otherwise).
    """
    length = args.length or cfg.get('TRAIN_LEN', 16)
    num_mask = cfg.get('NUM_MASK') or 0
    missing_prob = cfg['MISSING_PROB']
    miss_len = cfg.get('MISS_LEN', 1)
    d_max = args.depth or (miss_len + 2)
    min_n = args.min_n
    start = max(max(num_mask, 1), d_max - 1)

    if len(model.transformer.h) > 1:
        print('[warn] probe fitting targets single-layer models; multi-layer '
              'off-manifold results need separate interpretation')

    # --- pass 1: real recurrence sequences, naturally corrupted; bucket every
    # position by the corruption pattern of its last d_max tokens
    buckets = {}  # pattern tuple (len d_max) -> {'n','agree','attn'}
    for _ in range(args.samples):
        seq = [random.randrange(p) for _ in range(init_len)]
        while len(seq) < length:
            seq.append(next_fn(seq))
        window = torch.tensor(seq, dtype=torch.long)
        # single-rule only here, so the missing token is plain p
        corrupt_window(window, True, p=p, init_len=init_len, missing_prob=missing_prob,
                       miss_len=miss_len, miss_second=cfg.get('MISS_SECOND', False))
        view = window.tolist()
        x = torch.tensor([view], dtype=torch.long)
        with torch.no_grad():
            preds = model(x)[0][0].argmax(dim=-1).tolist()
            attn_layers = [lo_['attn_weights'] for lo_ in extract_qk_raw_scores(model, x)]

        for t in range(start, length - 1):
            patt = tuple(view[t - d_max + 1 + k] == p for k in range(d_max))
            rec = buckets.get(patt)
            if rec is None:
                rec = buckets[patt] = {'n': 0, 'agree': 0, 'attn': {}}
            rec['n'] += 1
            if preds[t] == seq[t + 1]:
                rec['agree'] += 1
            for li, att in enumerate(attn_layers):
                row_all = att[:, t, :]  # (H, T)
                for h in range(att.shape[0]):
                    acc = rec['attn'].setdefault((li, h), {})
                    row = row_all[h]
                    for d in range(t + 1):
                        v = row[t - d].item()
                        if d in acc:
                            acc[d][0] += v
                            acc[d][1] += 1
                        else:
                            acc[d] = [v, 1]

    print(f'Model: {exp_name}')
    print(f'Rule : {desc} | length={length}, missing prob={missing_prob}, '
          f'miss_len={miss_len}, miss_second={cfg.get("MISS_SECOND", False)}, '
          f'max pattern depth={d_max}')
    print(f'Queue-driven analysis: root {"(x,)" if init_len == 1 else "(x," * 0 + patt_label((False,) * init_len)}; '
          f'children = patterns masking the fitted formula\'s dependency distances')

    # --- pass 2: BFS over patterns
    root = (False,) * init_len
    visited = set()
    queue = [root]
    while queue:
        P = queue.pop(0)
        if P in visited or len(P) > d_max:
            continue
        visited.add(P)
        label = patt_label(P)
        n, agree, attn = aggregate_buckets(buckets, P, d_max)
        if n < min_n:
            print(f'\n== pattern {label}  n={n} (< --min-n {min_n}), skipped')
            continue
        print(f'\n== pattern {label}  n={n}, model-vs-truth {agree / n:.1%}')
        attn_parts = []
        for (li, h), acc in sorted(attn.items()):
            sig = [(d, s / c) for d, (s, c) in sorted(acc.items()) if s / c >= SIG_THRESHOLD]
            if sig:
                attn_parts.append(f'L{li}H{h} ' + ' '.join(f'd{d}:{v:.2f}' for d, v in sig))
        if attn_parts:
            print('  attn: ' + ' | '.join(attn_parts))

        C = select_C(attn, P)
        if not C:
            print('  no significant attention on visible distances; no fit, not expanding')
            continue
        X, y = probe_equations(model, P, C, length, args.samples, p)
        if len(X) > FIT_MAX_ROWS:
            idx = random.sample(range(len(X)), FIT_MAX_ROWS)
            X = [X[i] for i in idx]
            y = [y[i] for i in idx]
        coeffs, acc_fit, exact = fit_and_score(X, y, p)
        if coeffs is None:
            print(f'  set C {C}: no linear fit on random probes '
                  f'(best agreement {acc_fit:.1%}, n={len(X)})')
            continue
        tag = 'EXACT' if exact else f'agreement {acc_fit:.1%}'
        print(f'  set C {C}: {fmt_equation(C, coeffs, p)}  [{tag}, probe n={len(X)}]')

        if P == root:
            rc, base = rule_coeffs(next_fn, init_len, p)
            fit_map = {d: c % p for d, c in zip(C, coeffs)}
            ok = (base % p == 0
                  and all(fit_map.get(d, 0) == rc[d] for d in range(init_len))
                  and all(c % p == 0 for d, c in fit_map.items() if d >= init_len))
            print(f'  root check: training rule coeffs {rc} -> {"MATCH" if ok else "MISMATCH"}')

        if acc_fit < EXPAND_MIN_AGREEMENT:
            print(f'  agreement < {EXPAND_MIN_AGREEMENT:.0%}; not expanding')
            continue
        deps = {d for d, c in zip(C, coeffs) if c % p != 0}
        children = [c for c in child_patterns(P, deps, d_max)
                    if c not in visited and c not in queue]
        if children:
            print(f'  deps {sorted(deps)} -> enqueue '
                  + ', '.join(patt_label(c) for c in children))
            queue.extend(children)
    print()
```

同时：
1. 模块 docstring 第 4-6 条改为：
   ```
   4. MISSING-MODE (experiments with MISSING_PROB > 0): queue-driven BFS over
      corruption patterns rooted at (x,)*init_len. Per pattern: model-vs-truth
      agreement + live attention (real corrupted sequences), then a
      coefficient fit on attention-significant visible distances using RANDOM
      off-manifold probes (only the pattern's own window is forced masked).
      Children = patterns masking the fitted formula's dependency distances,
      up to pattern length --depth (default miss_len + 2)
   5. otherwise positions are segmented by ATTENTION SIGNATURE (unchanged)
   6. PROBE DATA: fully random sequences in both modes (missing mode forces
      the pattern's window masks; attention mode is uncorrupted)
   ```
2. `--depth` help 改为 `'missing mode: max pattern length (default miss_len + 2)'`。

注意：那个 `root ...` 打印行里我写的条件表达式太花哨，直接简化为
`print(f'Queue-driven analysis: root {patt_label(root)}; children = ...')`，把该行放在 `root = (False,) * init_len` 之后。

- [ ] **Step 4: 跑全部测试**

Run: `C:/Users/Chen/anaconda3/envs/torch/python.exe tests/test_rule_fit_missing.py`
Expected: `ALL TESTS PASSED: test_rule_fit_missing.py`

- [ ] **Step 5: 回归——既有 8 个测试文件 + 编译检查**

Run:
```bash
C:/Users/Chen/anaconda3/envs/torch/python.exe -m py_compile src/rule_fit.py
for f in tests/test_*.py; do C:/Users/Chen/anaconda3/envs/torch/python.exe "$f" || exit 1; done
```
Expected: 全部 `ALL TESTS PASSED`（rule_fit 未被这些测试 import，但确认无连带破坏）

---

### Task 4: 文档同步

**Files:**
- Modify: `RULE_FIT_NOTES.md`（§1 第二条、§3 关键函数表）
- Modify: `REFACTOR_LOG.md`（追加 #53）

- [ ] **Step 1: 更新 RULE_FIT_NOTES.md**

- §1 第二条（missing 实验描述）改为：missing 实验走**队列驱动 BFS**：根为全可见 pattern，每个 pattern 报告 model-vs-truth 一致率与实时注意力（真实污损序列），再用随机探针（off-manifold，只强制窗口内 mask）拟合注意力显著可见距离的系数；拟合公式的非零系数距离生成子 pattern 入队，长度上限 `--depth`（默认 miss_len+2）。
- §3 关键函数表 `run_missing_categories` 行改为 "missing 模式：BFS 队列编排"；表下补一行新函数组：`visible_distance/child_patterns/aggregate_buckets/select_C/rule_coeffs/probe_equations`。
- §5 seed 语义一条末尾补：探针（probe_equations）在 seed 之后消耗 RNG，属合法位置。

- [ ] **Step 2: REFACTOR_LOG.md 追加 #53**

```markdown
## 53. 【行为变更】rule_fit missing 模式改队列驱动分析，消除 on-manifold 秩亏失明

**问题**：旧 missing 模式在真实递推序列（on-manifold）上拟合 k 个距离的系数。递推阶数为 2 的序列任意窗口在 F_p 上最多张成 2 维，k>2 的设计矩阵必然秩亏——`solve_mod_p` 整体解与全部 300 轮 RANSAC 都返回 None，输出 `no linear fit (best agreement -100.0%)`（-100% 是未更新的 sentinel）。后果：model-vs-truth 99.9% 的 pattern（如 (x,x,x,x,M)）给不出公式，set A 退化时只剩 1.3% 的噪音拟合。

**修改**：`run_missing_categories` 重写为队列驱动 BFS——根为 (x,)*init_len，每 pattern 用随机探针（off-manifold，只强制窗口内 mask，特征满秩）在注意力显著可见距离（set C）上拟合；拟合公式的非零系数距离作为前提，其非空子集被污损生成的子 pattern 入队（长度上限 --depth=miss_len+2）。新增 `visible_distance/child_patterns/aggregate_buckets/select_C/rule_coeffs/probe_equations`；删除 cat4 顶层四类分组、set_A/dists_C_full 与 pass 1 的 fits 累计。root pattern 增加与训练规则系数的 MATCH/MISMATCH 校验。多层模型打印 warning（off-manifold 解读仅限单层）。

**验证**：`tests/test_rule_fit_missing.py`（纯函数单测 + stub 模型端到端拟合恢复 2x_{t-1}+x_{t-2} + 小模型冒烟）；真实批次端到端需在服务器复跑确认。
```

---

## Self-Review 记录

- Spec 覆盖：队列 BFS ✓（Task 3）、注意力显著距离建 C ✓（select_C）、随机探针只强制窗口内 ✓（probe_equations 间距规则 + Task 2 泄漏测试）、真实序列统计保留 ✓（pass 1 不变）、pattern 长度上限 5 ✓（d_max 默认 miss_len+2）、min_n 防护 ✓、单层限定 ✓（warning）、nonlinear 不考虑 ✓（Global Constraints）。
- 类型一致性：`aggregate_buckets` 返回的 attn 结构与 pass 1 写入结构一致（{(li,h): {d:[sum,cnt]}}）；`select_C` 消费同一结构；`probe_equations` 的 model 调用方式 `model(x)[0][0].argmax(dim=-1)` 与现有代码及 stub 一致；`rule_coeffs` 的 next_fn 单参签名与 `build_single_rule` 的闭包一致。
- 已发现并在计划中修正：初稿 `root` 打印行的花哨条件表达式 → 简化为 `patt_label(root)`。
