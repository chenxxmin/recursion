#!/usr/bin/env python3
"""overnight_h123_p127_d512_600 系数的代数分类（独立重算 + 与 metadata 交叉校验）。

分类标准 1（特征根类型，优先级 double > root1 > rootm1 > other）：
  特征方程 x^2 = a*x + b (mod p)，判别式 D = a^2 + 4b。
  - D == 0            : double（重根，Jordan 块）
  - 1  是特征根 (a+b=1): root1
  - -1 是特征根 (b=a+1): rootm1
  - 其余（含 F_p 不可约）: other

分类标准 2（循环轨道模式）：
  - split      : 两个不同特征根 -> 类 = 无序对 {ord(λ), ord(μ)}，周期 = lcm
  - double     : 类 = ('J', ord(λ))，Jordan 块 -> 周期 = p * ord(λ)
  - irreducible: F_{p^2} 中的根 -> 类 = ord(r)（= 伴随矩阵 M 在 GL_2(F_p) 的阶）

输出：终端统计表 + reports/data/overnight_h123_coef_classification.json
"""
import json
import math
import sys
from collections import Counter, defaultdict

P = 127
META = 'experiments/overnight_h123_p127_d512_600_metadata.json'
OUT = 'reports/data/overnight_h123_coef_classification.json'


def sqrt_mod(d, p):
    """返回 d 在 F_p 的一个平方根；d 为非二次剩余返回 None。"""
    d %= p
    for x in range(p):
        if x * x % p == d:
            return x
    return None


def mul_ord(x, p):
    """x (≠0) 在 F_p^* 的乘法阶。"""
    x %= p
    assert x != 0
    n, v = 1, x
    while v != 1:
        v = v * x % p
        n += 1
        assert n <= p  # 阶整除 p-1
    return n


def mat_pow_ord(a, b, p):
    """伴随矩阵 M=[[a,b],[1,0]] 在 GL_2(F_p) 的阶（暴力迭代，p=127 上界 16002 步）。"""
    # (m00,m01,m10,m11)，单位阵起
    m = (a % p, b % p, 1 % p, 0)
    cur = (1, 0, 0, 1)
    for n in range(1, p * p):
        cur = ((cur[0] * m[0] + cur[1] * m[2]) % p,
               (cur[0] * m[1] + cur[1] * m[3]) % p,
               (cur[2] * m[0] + cur[3] * m[2]) % p,
               (cur[2] * m[1] + cur[3] * m[3]) % p)
        if cur == (1, 0, 0, 1):
            return n
    raise AssertionError('matrix order not found')


def classify(a, b, p=P):
    """返回 (root_class, roots_info, cycle_class, cycle_period)。"""
    a %= p
    b %= p
    D = (a * a + 4 * b) % p
    inv2 = (p + 1) // 2
    if D == 0:
        lam = a * inv2 % p
        o = mul_ord(lam, p)
        period = p * o  # Jordan 块贡献因子 p
        return 'double', {'roots': [lam], 'root_orders': [o]}, ('J', o), period
    s = sqrt_mod(D, p)
    if s is not None:
        lam = (a + s) * inv2 % p
        mu = (a - s) * inv2 % p
        o_lam, o_mu = mul_ord(lam, p), mul_ord(mu, p)
        if 1 in (lam, mu):
            rc = 'root1'
        elif (p - 1) in (lam, mu):
            rc = 'rootm1'
        else:
            rc = 'other'
        orders = tuple(sorted((o_lam, o_mu)))
        return rc, {'roots': sorted((lam, mu)), 'root_orders': list(orders)}, ('split',) + orders, math.lcm(o_lam, o_mu)
    # F_p 不可约：根 r ∈ F_{p^2}，ord(r) = ord(M)（M 在 F_{p^2} 上对角化为 diag(r, r^p)，同阶）
    period = mat_pow_ord(a, b, p)
    return 'other', {'roots': None, 'root_orders': None}, ('irred', period), period


def fmt_cycle_class(cc):
    if cc[0] == 'J':
        return f'double_J{cc[1]}'
    if cc[0] == 'split':
        return f'split_{{{cc[1]},{cc[2]}}}'
    return f'irred_{cc[1]}'


def main():
    meta = json.load(open(META))
    assert meta['p'] == P
    coefs = meta['coefficients']
    records = []
    mismatches = []
    for c in coefs:
        a, b = c['coefficients']
        rc, info, cc, period = classify(a, b)
        # ---- 交叉校验 metadata ----
        if c['root_type'] != rc:
            mismatches.append((a, b, 'root_type', c['root_type'], rc))
        if rc == 'double':
            if c['root_orders'] != info['root_orders']:
                mismatches.append((a, b, 'root_orders', c['root_orders'], info['root_orders']))
        elif cc[0] == 'split':
            if c['root_orders'] != info['root_orders']:
                mismatches.append((a, b, 'root_orders', c['root_orders'], info['root_orders']))
        else:
            if c['irreducible_root_order'] != period:
                mismatches.append((a, b, 'irreducible_root_order', c['irreducible_root_order'], period))
        records.append({
            'a': a, 'b': b, 'group': c['group'],
            'root_class': rc,
            'roots': info['roots'], 'root_orders': info['root_orders'],
            'cycle_class': fmt_cycle_class(cc),
            'cycle_period': period,
            'shortest_one_term_lag': c['shortest_one_term_lag'],
        })

    print(f'系数 {len(records)} 组（实验 600 = 150 系数 x 4 seeds），p={P}')
    # ab=0 与一元捷径排除校验
    assert all(r['a'] * r['b'] % P != 0 for r in records), '存在 ab=0!'
    min_lag = min(r['shortest_one_term_lag'] for r in records)
    print(f'ab=0 已排除 ✓ ; shortest_one_term_lag 最小值 = {min_lag}（>16 阶内一元捷径已排除: {min_lag > 16}）')
    if mismatches:
        print(f'\n!! metadata 交叉校验不符 {len(mismatches)} 处:')
        for m in mismatches:
            print('  ', m)
    else:
        print('metadata 交叉校验全部一致 ✓\n')

    # ---- 分类标准 1 ----
    print('== 分类标准 1：特征根类型 ==')
    c1 = Counter(r['root_class'] for r in records)
    for k in ['double', 'root1', 'rootm1', 'other']:
        print(f'  {k:7s}: {c1[k]:3d} 组系数 ({c1[k]*4:3d} 实验)')
    # double 与 root1/rootm1 的理论重叠检查（重根为 ±1）
    overlap = [(r['a'], r['b']) for r in records
               if r['root_class'] == 'double' and (r['a'] + r['b']) % P == 1]
    print(f'  （重根同时为 λ=1 的边界情形: {overlap or "无"}）')

    # ---- 分类标准 2 ----
    print('\n== 分类标准 2：循环轨道模式（每种子结构一类，跨 H 合并）==')
    c2 = defaultdict(list)
    for r in records:
        c2[r['cycle_class']].append(r)
    for cls in sorted(c2, key=lambda k: (k.split('_')[0], c2[k][0]['cycle_period'])):
        rs = c2[cls]
        period = rs[0]['cycle_period']
        groups = sorted(set(r['group'] for r in rs))
        print(f'  {cls:18s} 周期={period:6d} : {len(rs):3d} 组系数 ({len(rs)*4:3d} 实验)  来源 {groups}')

    out = {
        'p': P, 'n_coefficients': len(records), 'n_experiments': len(records) * 4,
        'class1_root_type': {k: [[r['a'], r['b']] for r in records if r['root_class'] == k]
                             for k in ['double', 'root1', 'rootm1', 'other']},
        'class2_cycle': {cls: {'cycle_period': c2[cls][0]['cycle_period'],
                               'coefficients': [[r['a'], r['b']] for r in c2[cls]]}
                         for cls in sorted(c2)},
        'records': records,
    }
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f'\n已保存 -> {OUT}')
    if mismatches:
        sys.exit(1)


if __name__ == '__main__':
    main()
