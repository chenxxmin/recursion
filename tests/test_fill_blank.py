"""Tests for the fill_blank task variant (TASK_VARIANT='fill_blank').

Same shape as test_corrupt_fix.py: collate layout, fresh randomness,
_unpack_batch + evaluate integration, end-to-end smoke (CPU), config errors.
"""
import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from datasets import BatchTag, make_fill_blank_collate
from models import FibonacciTransformer
from training import evaluate, _unpack_batch
from test_corrupt_fix import _addition_windows, _run_experiment


def test_collate_layout():
    p, eos_id, L = 7, 8, 6
    seqs = _addition_windows(p=p, n=16, length=L)
    random.seed(123)
    collate = make_fill_blank_collate(p=p, eos_id=eos_id)
    x, tag, targets, mask = collate(seqs)
    assert tag == BatchTag.FILL_BLANK
    B = len(seqs)
    assert x.shape == (B, L + 1) and targets.shape == (B, L + 2)
    assert mask.shape == (B, L + 1)
    clean = torch.stack(seqs)
    for i in range(B):
        # exactly one position replaced by the BLANK token (= p)
        diff = (x[i, :L] != clean[i]).nonzero().flatten().tolist()
        assert len(diff) == 1, f"sample {i}: expected exactly 1 blank, got {diff}"
        k = diff[0]
        assert int(x[i, k]) == p, "blanked position must carry the blank token id (= p)"
        assert int(x[i, L]) == eos_id
        # target: correct value at the EOS slot; targets = tgt[:, 1:]
        assert int(targets[i, L + 1]) == int(clean[i, k])
        ones = mask[i].nonzero().flatten().tolist()
        assert ones == [L], f"sample {i}: loss mask at {ones}"
        assert torch.equal(targets[i, 1:L + 1], x[i, 1:L + 1])


def test_collate_fresh_randomness():
    p, L = 11, 8
    seqs = _addition_windows(p=p, n=32, length=L)
    collate = make_fill_blank_collate(p=p, eos_id=p + 1)
    random.seed(1)
    x1, _, _, _ = collate(seqs)
    random.seed(2)
    x2, _, _, _ = collate(seqs)
    # blank positions differ between calls (compare against clean to locate)
    clean = torch.stack(seqs)
    k1 = [(x1[i, :L] != clean[i]).nonzero().item() for i in range(32)]
    k2 = [(x2[i, :L] != clean[i]).nonzero().item() for i in range(32)]
    assert k1 != k2, "blank positions must be fresh per call"


def test_unpack_and_evaluate_integration():
    p, L = 7, 6
    eos_id = p + 1
    vocab_size = p + 2
    seqs = _addition_windows(p=p, n=16, length=L)
    collate = make_fill_blank_collate(p=p, eos_id=eos_id)
    batch = collate(seqs)
    x, loss_mask, kwargs, ab_labels, targets_override = _unpack_batch(batch, 'cpu', None)
    assert kwargs == {} and ab_labels is None
    assert loss_mask is not None and targets_override is not None
    assert torch.equal(targets_override[:, :-1], x)

    torch.manual_seed(0)
    model = FibonacciTransformer(p=p, d_model=32, n_head=1, n_layer=1,
                                 block_size=8, dropout=0.0, vocab_size=vocab_size)
    loader = DataLoader(seqs, batch_size=8, collate_fn=collate)
    loss, acc, per_pos_acc, group_acc = evaluate(model, loader, 'cpu', num_mask=0)
    assert math.isfinite(loss) and loss > 0
    assert 0.0 <= acc <= 1.0
    # accuracy counted only at the single output position (EOS slot)
    assert set(per_pos_acc) == {L}, f"unexpected acc positions: {per_pos_acc}"


def test_run_experiment_fill_blank_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {'TASK_VARIANT': 'fill_blank'})
        assert os.path.exists(save_path), "checkpoint was not saved"
        assert 'Train: Loss=' in out, "training loop output missing"
        assert 'nan' not in out.lower(), "NaN appeared in training"
        assert '[fill_blank]' in out, "task banner missing variant tag"
        # POS_PROBE：fill_blank 每个 eval 打一行 val，16 个 k
        probe_lines = [l for l in out.splitlines() if l.startswith('  PosProbe val|')]
        assert probe_lines and probe_lines[0].count('k') == 8, "PosProbe line missing"  # TRAIN_LEN=8
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['variant'] == 'fill_blank'
        assert sc['scenario'] == 'single' and sc['rules'] == [[1, 1]]
        assert sc['eos_token_id'] == 8 and sc['blank_token_id'] == 7
        assert sc['vocab_size'] == 9          # P + 2
        assert sc['block_size'] == 16         # round_up_pow2(max(8,10) + 1)


def test_run_experiment_fill_blank_rejects_missing_prob():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'fill_blank',
                                     'MISSING_PROB': 0.1})
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected SystemExit(2) for MISSING_PROB conflict")


def test_collate_fixed_position_tuples():
    # (seq, k) 元组样本（FILL_BLANK_ALL_POSITIONS 模式）：空白位置由样本携带
    p, eos_id, L = 7, 8, 6
    seqs = _addition_windows(p=p, n=4, length=L)
    batch = [(seq, k) for seq in seqs for k in range(L)]
    collate = make_fill_blank_collate(p=p, eos_id=eos_id)
    x, tag, targets, mask = collate(batch)
    assert x.shape == (4 * L, L + 1)
    for j, (seq, k) in enumerate(batch):
        assert int(x[j, k]) == p, "blank must be at the carried position k"
        diff = (x[j, :L] != seq).nonzero().flatten().tolist()
        assert diff == [k]
        assert int(targets[j, L + 1]) == int(seq[k])
        assert mask[j].nonzero().flatten().tolist() == [L]


def test_run_experiment_arithmetic_fill_blank_all_positions():
    # 等差数列（ORDER=1 + CONST=1）+ fill_blank + 逐位置枚举挖空：端到端冒烟
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'fill_blank', 'FILL_BLANK_ALL_POSITIONS': True,
            'P': 17, 'ORDER': 1, 'RULES': [[1]], 'CONST': 1})
        assert os.path.exists(save_path), "checkpoint was not saved"
        assert 'Train: Loss=' in out and 'nan' not in out.lower()
        assert 'X(k)=(1*X(k-1)+1) mod 17' in out  # 递推名含常数项
        # 17 个状态 × 0.7 = 11 train 窗口 × TRAIN_LEN=8 位置 = 88 条训练样本
        assert '11 windows -> 88 train samples' in out
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['variant'] == 'fill_blank' and sc['const'] == 1
        assert sc['order'] == 1 and sc['rules'] == [[1]]


def test_run_experiment_fill_blank_test_all_positions():
    # TEST_ALL_POSITIONS：只扩测试集（训练侧保持在线随机挖空）
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'fill_blank', 'TEST_ALL_POSITIONS': True,
            'P': 17, 'ORDER': 1, 'RULES': [[1]], 'CONST': 1})
        assert os.path.exists(save_path)
        # 6 个测试窗口 × TRAIN_LEN=8 位置 = 48 条确定性测试样本
        assert 'TEST_ALL_POSITIONS: 6 windows -> 48 deterministic test samples' in out
        assert 'FILL_BLANK_ALL_POSITIONS' not in out  # 训练侧未开枚举
        # NUM_TEST_SAMPLES 作为测试窗口数上限（确定性子抽样）
        out2, _ = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'fill_blank', 'TEST_ALL_POSITIONS': True,
            'NUM_TEST_SAMPLES': 4,
            'P': 17, 'ORDER': 1, 'RULES': [[1]], 'CONST': 1})
        assert 'TEST_ALL_POSITIONS: 4 windows -> 32 deterministic test samples' in out2


def test_collate_body_supervision():
    # BODY_SUPERVISION：targets 前缀为干净序列，mask 覆盖主体 [body_from, L-1) + 输出槽
    p, eos_id, L = 7, 8, 6
    seqs = _addition_windows(p=p, n=8, length=L)
    random.seed(5)
    collate = make_fill_blank_collate(p=p, eos_id=eos_id,
                                      body_supervision=True, body_from=1)
    x, tag, targets, mask = collate(seqs)
    clean = torch.stack(seqs)
    for i in range(8):
        k = (x[i, :L] != clean[i]).nonzero().flatten().item()
        # targets 前缀是干净序列（targets[:, :L+1] = [clean[1:], EOS]）
        assert torch.equal(targets[i, 1:L], clean[i, 1:])
        # mask：body_from=1 -> target[1..L-2] 主体 + target[L] 输出槽；
        # target[0]（预测 x1，初值位）与 target[L-1]（预测 EOS）不算
        ones = mask[i].nonzero().flatten().tolist()
        assert ones == list(range(1, L - 1)) + [L], f"mask: {ones}"


def test_run_experiment_var_len_smoke():
    # 不定长 SEQ_LEN_MIN/MAX + 主体监督，端到端冒烟
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'fill_blank', 'P': 17, 'ORDER': 1,
            'RULES': [[1]], 'CONST': 1,
            'SEQ_LEN_MIN': 3, 'SEQ_LEN_MAX': 5})
        assert os.path.exists(save_path)
        assert 'nan' not in out.lower()
        assert 'variable length: windows cut to [3, 5]' in out
        # 探针按最大长度出 5 个 k（不定长下逐 (l,k) 聚合）
        probe_lines = [l for l in out.splitlines() if l.startswith('  PosProbe val|')]
        assert probe_lines and probe_lines[0].count('k') == 5
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['seq_len_min'] == 3 and sc['seq_len_max'] == 5
        assert sc['body_supervision'] is True


def test_collate_fixed_k():
    # FILL_BLANK_FIXED_POS：纯张量输入时挖空位置固定，负数从末尾数
    p, eos_id, L = 7, 8, 6
    seqs = _addition_windows(p=p, n=16, length=L)
    collate = make_fill_blank_collate(p=p, eos_id=eos_id, fixed_k=-1)
    x, tag, targets, mask = collate(seqs)
    clean = torch.stack(seqs)
    for i in range(len(seqs)):
        assert int(x[i, L - 1]) == p, "fixed_k=-1 must blank the last position"
        assert torch.equal(x[i, :L - 1], clean[i, :L - 1])
        assert int(targets[i, L + 1]) == int(clean[i, L - 1])
    # 非负 fixed_k
    collate2 = make_fill_blank_collate(p=p, eos_id=eos_id, fixed_k=0)
    x2, _, _, _ = collate2(seqs)
    for i in range(len(seqs)):
        assert int(x2[i, 0]) == p
    # 元组输入携带的 k 优先（不受 fixed_k 影响）
    collate3 = make_fill_blank_collate(p=p, eos_id=eos_id, fixed_k=-1)
    xt, _, _, _ = collate3([(seqs[0], 2)])
    assert int(xt[0, 2]) == p and int(xt[0, L - 1]) != p


def test_fixed_pos_gates():
    # 仅 fill_blank；与 FILL_BLANK_ALL_POSITIONS 互斥
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'corrupt_fix',
                                     'FILL_BLANK_FIXED_POS': -1})
            assert False, 'corrupt_fix + FILL_BLANK_FIXED_POS should fail'
        except SystemExit:
            pass
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'fill_blank',
                                     'FILL_BLANK_FIXED_POS': -1,
                                     'FILL_BLANK_ALL_POSITIONS': True})
            assert False, 'FIXED_POS + ALL_POSITIONS should fail'
        except SystemExit:
            pass


def test_run_experiment_fixed_pos_smoke():
    # 端到端：L=4 固定挖末位，save_config 记录 fixed_pos
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'fill_blank', 'P': 17, 'ORDER': 1,
            'RULES': [[1]], 'CONST': 1, 'TRAIN_LEN': 4, 'OOD_LEN': 4,
            'FILL_BLANK_FIXED_POS': -1})
        assert os.path.exists(save_path)
        assert 'FILL_BLANK_FIXED_POS=-1' in out
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        assert latest['config']['fixed_pos'] == -1


def test_collate_multi_blank_exact_layout():
    # [x0..x4] 挖掉 [x2,x3]：单个 BLANK + 输入 EOS(2) + teacher forcing。
    p, eos_id = 7, 8
    seq = torch.tensor([0, 1, 2, 3, 4])
    collate = make_fill_blank_collate(p=p, eos_id=eos_id, blank_lens=[2])
    x, tag, targets, mask = collate([(seq, 2, 2)])
    assert tag == BatchTag.FILL_BLANK
    assert x.tolist() == [[0, 1, p, 4, eos_id + 1, 2]]
    assert targets.tolist() == [[0, 1, p, 4, eos_id + 1, 2, 3]]
    assert mask[0].nonzero().flatten().tolist() == [4, 5]
    assert torch.equal(targets[:, :-1], x)
    shifted = targets[:, 1:]
    assert shifted[0, 4].item() == 2  # EOS(2) -> x2；长度由输入 EOS 已知
    assert shifted[0, 5].item() == 3  # fed-back x2 -> x3

    # 主体监督：BLANK 对应第一个被挖值，可见后缀保持干净。
    body_collate = make_fill_blank_collate(
        p=p, eos_id=eos_id, blank_lens=[2], body_supervision=True, body_from=1)
    xb, _, tb, mb = body_collate([(seq, 2, 2)])
    assert torch.equal(xb, x)
    assert tb.tolist() == [[0, 1, 2, 4, eos_id + 1, 2, 3]]
    assert mb[0].nonzero().flatten().tolist() == [1, 2, 4, 5]

    # 段首/段尾和 l=L 均是合法边界；l=1 在混合集合中仍是旧 EOS 布局。
    mixed = make_fill_blank_collate(p=p, eos_id=eos_id, blank_lens=[1, 2, 5])
    for l in (1, 2, 5):
        for k in ((0, seq.numel() - l) if l < seq.numel() else (0,)):
            xe, _, te, me = mixed([(seq, k, l)])
            m = seq.numel() - l + 1
            expected_view = seq[:k].tolist() + [p] + seq[k + l:].tolist()
            assert xe.shape == (1, seq.numel() + 1)
            assert te.shape == (1, seq.numel() + 2)
            assert xe[0, :m].tolist() == expected_view
            assert xe[0, m].item() == eos_id + l - 1
            assert te[0, m + 1:].tolist() == seq[k:k + l].tolist()
            assert me[0].nonzero().flatten().tolist() == list(range(m, m + l))
    legacy = make_fill_blank_collate(p=p, eos_id=eos_id)
    mixed_single = mixed([(seq, 2, 1)])
    legacy_single = legacy([(seq, 2)])
    assert all(torch.equal(a, b) if torch.is_tensor(a) else a == b
               for a, b in zip(mixed_single, legacy_single))


def test_collate_multi_blank_sampling_fixed_and_evaluate():
    p, eos_id, L = 7, 8, 5
    seqs = _addition_windows(p=p, n=32, length=L)
    collate = make_fill_blank_collate(p=p, eos_id=eos_id, blank_lens=[1, 2])
    random.seed(1)
    x1, _, t1, m1 = collate(seqs)
    random.seed(2)
    x2, _, _, _ = collate(seqs)
    assert not torch.equal(x1, x2), "length/start sampling must be fresh"
    assert x1.shape == (32, L + 1) and t1.shape == (32, L + 2)
    seen_lens = set()
    for row in x1:
        eos_pos = ((row >= eos_id) & (row <= eos_id + 1)).nonzero().flatten().item()
        blank_pos = (row[:eos_pos] == p).nonzero().flatten().item()
        blank_l = int(row[eos_pos]) - eos_id + 1
        seen_lens.add(blank_l)
        assert blank_l in (1, 2)
        assert eos_pos == L - blank_l + 1
        assert 0 <= blank_pos <= L - blank_l
    assert seen_lens == {1, 2}
    assert m1.shape == (32, L + 1)

    # 负数 fixed_k 从合法段起点倒计数：-1 表示段尾贴住窗口末尾。
    fixed = make_fill_blank_collate(p=p, eos_id=eos_id, blank_lens=[2], fixed_k=-1)
    xf, _, tf, mf = fixed(seqs)
    for i, seq in enumerate(seqs):
        assert xf[i, 3].item() == p and xf[i, 4].item() == eos_id + 1
        assert tf[i, -2:].tolist() == seq[3:5].tolist()
        assert mf[i].nonzero().flatten().tolist() == [4, 5]

    # 训练引擎的 shift/masked CE/accuracy 无需特殊分支。
    items = [(seq, i % (L - 1), 2) for i, seq in enumerate(seqs)]
    fixed_items = make_fill_blank_collate(p=p, eos_id=eos_id, blank_lens=[1, 2])
    batch = fixed_items(items)
    x, loss_mask, kwargs, ab_labels, targets_override = _unpack_batch(batch, 'cpu', None)
    assert kwargs == {} and ab_labels is None
    assert torch.equal(targets_override[:, :-1], x)
    torch.manual_seed(0)
    model = FibonacciTransformer(p=p, d_model=32, n_head=1, n_layer=1,
                                 block_size=8, dropout=0.0, vocab_size=p + 3)
    loader = DataLoader(items, batch_size=8, collate_fn=fixed_items)
    loss, acc, per_pos_acc, _ = evaluate(model, loader, 'cpu', num_mask=0)
    assert math.isfinite(loss) and 0.0 <= acc <= 1.0
    assert set(per_pos_acc) == {4, 5}


def test_run_experiment_multi_blank_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'fill_blank', 'BLANK_LENS': [1, 2]})
        assert os.path.exists(save_path)
        assert 'BLANK_LENS=[1, 2]' in out and 'nan' not in out.lower()
        for blank_l, n_positions in ((1, 8), (2, 7)):
            val_lines = [line for line in out.splitlines()
                         if line.startswith(f'  PosProbe val_l{blank_l}|')]
            assert val_lines, f'missing multi-blank probe l={blank_l}'
            assert val_lines[0].count('k') == n_positions
        assert not any(line.startswith('  PosProbe len_') for line in out.splitlines())
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['blank_lens'] == [1, 2] and sc['eos_variant_base'] == 8
        assert 'length_token_base' not in sc
        assert sc['vocab_size'] == 10       # P + 1 + max(BLANK_LENS)
        assert sc['block_size'] == 16       # round_up_pow2(max(8,10) + 1)


def test_run_experiment_multi_blank_enumeration():
    common = {'TASK_VARIANT': 'fill_blank', 'BLANK_LENS': [1, 2],
              'P': 17, 'ORDER': 1, 'RULES': [[1]], 'CONST': 1,
              'TRAIN_LEN': 4, 'OOD_LEN': 4}
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {
            **common, 'FILL_BLANK_ALL_POSITIONS': True})
        assert os.path.exists(save_path)
        # L=4: l=1 有 4 段，l=2 有 3 段；11 窗口 × 7 = 77。
        assert '11 windows -> 77 train samples' in out
        assert '[Probe] POS_PROBE: 6 windows' in out, "枚举引用必须先去重再探针"
        out2, _ = _run_experiment(tmpdir, {**common, 'TEST_ALL_POSITIONS': True})
        assert 'TEST_ALL_POSITIONS: 6 windows -> 42 deterministic test samples' in out2
        assert '[Probe] POS_PROBE: 6 windows' in out2


def test_run_experiment_multi_blank_var_len():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'fill_blank', 'BLANK_LENS': [1, 2, 3],
            'P': 17, 'ORDER': 1, 'RULES': [[1]], 'CONST': 1,
            'SEQ_LEN_MIN': 3, 'SEQ_LEN_MAX': 5})
        assert os.path.exists(save_path) and 'nan' not in out.lower()
        assert 'variable length: windows cut to [3, 5]' in out
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['blank_lens'] == [1, 2, 3] and sc['block_size'] == 8
        assert sc['vocab_size'] == 21       # P17 + 1 + max_l3


def test_collate_multi_count_exact_layout():
    p, eos_id = 7, 8
    seq = torch.tensor([0, 1, 2, 3, 4])
    collate = make_fill_blank_collate(p=p, eos_id=eos_id, blank_counts=[1, 2, 3])
    x, tag, targets, mask = collate([(seq, (1, 3))])
    assert tag == BatchTag.FILL_BLANK
    assert x.tolist() == [[0, p, 2, p, 4, eos_id + 1, 1]]
    assert targets.tolist() == [[0, p, 2, p, 4, eos_id + 1, 1, 3]]
    assert mask[0].nonzero().flatten().tolist() == [5, 6]
    assert torch.equal(targets[:, :-1], x)
    assert targets[0, 6].item() == 1 and targets[0, 7].item() == 3

    body = make_fill_blank_collate(
        p=p, eos_id=eos_id, blank_counts=[1, 2], body_supervision=True, body_from=1)
    xb, _, tb, mb = body([(seq, (1, 3))])
    assert torch.equal(xb, x)
    assert tb.tolist() == [[0, 1, 2, 3, 4, eos_id + 1, 1, 3]]
    assert mb[0].nonzero().flatten().tolist() == [1, 2, 3, 5, 6]

    legacy = make_fill_blank_collate(p=p, eos_id=eos_id)
    single = collate([(seq, (2,))])
    legacy_single = legacy([(seq, 2)])
    assert all(torch.equal(a, b) if torch.is_tensor(a) else a == b
               for a, b in zip(single, legacy_single))


def test_collate_multi_count_sampling_padding_and_evaluate():
    p, eos_id, L = 7, 8, 5
    seqs = _addition_windows(p=p, n=48, length=L)
    collate = make_fill_blank_collate(p=p, eos_id=eos_id, blank_counts=[1, 2, 3])
    random.seed(1)
    x1, _, t1, m1 = collate(seqs)
    random.seed(2)
    x2, _, _, _ = collate(seqs)
    assert not torch.equal(x1, x2)
    assert x1.shape == (48, L + 3) and t1.shape == (48, L + 4)
    seen_counts = set()
    for i, seq in enumerate(seqs):
        k = int(x1[i, L]) - eos_id + 1
        seen_counts.add(k)
        positions = (x1[i, :L] == p).nonzero().flatten().tolist()
        assert len(positions) == k
        assert t1[i, L + 1:L + k + 1].tolist() == seq[positions].tolist()
        assert m1[i].nonzero().flatten().tolist() == list(range(L, L + k))
        assert not m1[i, L + k:].any()
    assert seen_counts == {1, 2, 3}

    items = [(seq, tuple(range(1 + i % 3))) for i, seq in enumerate(seqs)]
    batch = collate(items)
    x, loss_mask, kwargs, ab_labels, targets_override = _unpack_batch(batch, 'cpu', None)
    assert kwargs == {} and ab_labels is None
    torch.manual_seed(0)
    model = FibonacciTransformer(p=p, d_model=32, n_head=1, n_layer=1,
                                 block_size=8, dropout=0.0, vocab_size=p + 4)
    loader = DataLoader(items, batch_size=8, collate_fn=collate)
    loss, acc, per_pos_acc, _ = evaluate(model, loader, 'cpu', num_mask=0)
    assert math.isfinite(loss) and 0.0 <= acc <= 1.0
    assert set(per_pos_acc) == {5, 6, 7}


def test_run_experiment_multi_count_smoke_and_enumeration():
    common = {'TASK_VARIANT': 'fill_blank', 'BLANK_COUNTS': [1, 2],
              'P': 17, 'ORDER': 1, 'RULES': [[1]], 'CONST': 1,
              'TRAIN_LEN': 4, 'OOD_LEN': 4}
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, common)
        assert os.path.exists(save_path) and 'nan' not in out.lower()
        assert 'BLANK_COUNTS=[1, 2]' in out
        for count in (1, 2):
            lines = [line for line in out.splitlines()
                     if line.startswith(f'  PosProbe val_n{count}|')]
            assert lines and lines[0].count('k') == 4
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['blank_counts'] == [1, 2] and sc['blank_mode'] == 'multi_single'
        assert sc['vocab_size'] == 20 and sc['block_size'] == 8
        assert sc['eos_variant_base'] == 18

        out2, _ = _run_experiment(tmpdir, {
            **common, 'FILL_BLANK_ALL_POSITIONS': True})
        assert '11 windows -> 110 train samples' in out2
        assert '[Probe] POS_PROBE: 6 windows' in out2


def test_collate_corrupt_value_only():
    # corrupt=True：BLANK 换成随机错误值，版面与 fill_blank 完全一致
    p, eos_id, L = 7, 8, 5
    seqs = _addition_windows(p=p, n=16, length=L)
    clean = torch.stack(seqs)
    for fixed in (None, -1):
        collate = make_fill_blank_collate(p=p, eos_id=eos_id, corrupt=True,
                                          fixed_k=fixed)
        x, tag, targets, mask = collate(seqs)
        assert tag == BatchTag.FILL_BLANK
        assert x.shape == (16, L + 1) and targets.shape == (16, L + 2)
        for i in range(16):
            diff = (x[i, :L] != clean[i]).nonzero().flatten().tolist()
            assert len(diff) == 1
            k = diff[0]
            if fixed == -1:
                assert k == L - 1
            wrong = int(x[i, k])
            assert wrong != int(clean[i, k]), "corrupt value must differ from orig"
            assert 0 <= wrong < p
            assert int(targets[i, L + 1]) == int(clean[i, k])
            assert mask[i].nonzero().flatten().tolist() == [L]


def test_value_only_smoke_and_gates():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'corrupt_fix', 'P': 17, 'ORDER': 1,
            'RULES': [[1]], 'CONST': 1, 'TRAIN_LEN': 4, 'OOD_LEN': 4,
            'CORRUPT_FIX_VALUE_ONLY': True, 'CORRUPT_FIX_FIXED_POS': -1})
        assert os.path.exists(save_path)
        assert 'CORRUPT_FIX_VALUE_ONLY' in out
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        assert latest['config']['value_only'] is True
        assert latest['config']['fixed_pos'] == -1
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'corrupt_fix',
                                     'CORRUPT_FIX_FIXED_POS': -1})
            assert False, 'FIXED_POS without VALUE_ONLY should fail'
        except SystemExit:
            pass
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'fill_blank',
                                     'CORRUPT_FIX_VALUE_ONLY': True})
            assert False, 'VALUE_ONLY on fill_blank should fail'
        except SystemExit:
            pass


def test_collate_last_n():
    # last_n=2：在线污损位置只落在最后两位
    p, eos_id, L = 7, 8, 5
    seqs = _addition_windows(p=p, n=64, length=L)
    collate = make_fill_blank_collate(p=p, eos_id=eos_id, corrupt=True, last_n=2)
    clean = torch.stack(seqs)
    seen = set()
    for _ in range(8):
        x, _, _, _ = collate(seqs)
        for i in range(len(seqs)):
            diff = (x[i, :L] != clean[i]).nonzero().flatten().tolist()
            assert len(diff) == 1
            k = diff[0]
            assert k in (L - 2, L - 1), f"k={k} out of last-2 range"
            seen.add(k)
    assert seen == {L - 2, L - 1}
    # 与 FIXED_POS 互斥的门
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'corrupt_fix',
                                     'CORRUPT_FIX_VALUE_ONLY': True,
                                     'CORRUPT_FIX_FIXED_POS': -1,
                                     'CORRUPT_FIX_LAST_N': 2})
            assert False, 'FIXED_POS + LAST_N should fail'
        except SystemExit:
            pass


def test_collate_pos_set():
    # pos_set=[0,2,4]：污损位置只在给定集合内
    p, eos_id, L = 7, 8, 5
    seqs = _addition_windows(p=p, n=64, length=L)
    collate = make_fill_blank_collate(p=p, eos_id=eos_id, corrupt=True,
                                      pos_set=[0, 2, 4])
    clean = torch.stack(seqs)
    seen = set()
    for _ in range(8):
        x, _, _, _ = collate(seqs)
        for i in range(len(seqs)):
            diff = (x[i, :L] != clean[i]).nonzero().flatten().tolist()
            assert len(diff) == 1
            k = diff[0]
            assert k in (0, 2, 4), f"k={k} not in pos_set"
            seen.add(k)
    assert seen == {0, 2, 4}


if __name__ == '__main__':
    test_collate_multi_blank_exact_layout()
    print('PASS test_collate_multi_blank_exact_layout')
    test_collate_multi_blank_sampling_fixed_and_evaluate()
    print('PASS test_collate_multi_blank_sampling_fixed_and_evaluate')
    test_run_experiment_multi_blank_smoke()
    print('PASS test_run_experiment_multi_blank_smoke')
    test_run_experiment_multi_blank_enumeration()
    print('PASS test_run_experiment_multi_blank_enumeration')
    test_run_experiment_multi_blank_var_len()
    print('PASS test_run_experiment_multi_blank_var_len')
    test_collate_multi_count_exact_layout()
    print('PASS test_collate_multi_count_exact_layout')
    test_collate_multi_count_sampling_padding_and_evaluate()
    print('PASS test_collate_multi_count_sampling_padding_and_evaluate')
    test_run_experiment_multi_count_smoke_and_enumeration()
    print('PASS test_run_experiment_multi_count_smoke_and_enumeration')
    test_collate_pos_set()
    print('PASS test_collate_pos_set')
    test_collate_last_n()
    print('PASS test_collate_last_n')
    test_collate_corrupt_value_only()
    print('PASS test_collate_corrupt_value_only')
    test_value_only_smoke_and_gates()
    print('PASS test_value_only_smoke_and_gates')
    test_collate_fixed_k()
    print('PASS test_collate_fixed_k')
    test_fixed_pos_gates()
    print('PASS test_fixed_pos_gates')
    test_run_experiment_fixed_pos_smoke()
    print('PASS test_run_experiment_fixed_pos_smoke')
    test_collate_layout()
    print('PASS test_collate_layout')
    test_collate_fresh_randomness()
    print('PASS test_collate_fresh_randomness')
    test_unpack_and_evaluate_integration()
    print('PASS test_unpack_and_evaluate_integration')
    test_run_experiment_fill_blank_smoke()
    print('PASS test_run_experiment_fill_blank_smoke')
    test_run_experiment_fill_blank_rejects_missing_prob()
    print('PASS test_run_experiment_fill_blank_rejects_missing_prob')
    test_collate_fixed_position_tuples()
    print('PASS test_collate_fixed_position_tuples')
    test_run_experiment_arithmetic_fill_blank_all_positions()
    print('PASS test_run_experiment_arithmetic_fill_blank_all_positions')
    test_run_experiment_fill_blank_test_all_positions()
    print('PASS test_run_experiment_fill_blank_test_all_positions')
    test_collate_body_supervision()
    print('PASS test_collate_body_supervision')
    test_run_experiment_var_len_smoke()
    print('PASS test_run_experiment_var_len_smoke')
    print('ALL fill_blank tests PASSED')
    print('PASS test_run_experiment_var_len_smoke')
    print('ALL TESTS PASSED: test_fill_blank.py')
