"""Tests for the corrupt_fix task variant (TASK_VARIANT='corrupt_fix').

Covers: make_corrupt_fix_collate corruption/layout semantics, _unpack_batch
routing + evaluate integration, and an end-to-end run_experiment smoke test
(CPU, tiny config). Style follows test_training_smoke.py / test_missing_values.py.
"""
import contextlib
import io
import json
import math
import os
import random
import re
import sys
import tempfile

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from datasets import BatchTag, make_corrupt_fix_collate
from models import FibonacciTransformer
from training import evaluate, _unpack_batch


def _addition_windows(p=7, n=8, length=6, seed=0):
    """A few clean addition (order-2) windows as plain LongTensors."""
    rng = random.Random(seed)
    seqs = []
    for _ in range(n):
        a, b = rng.randrange(p), rng.randrange(p)
        seq = [a, b]
        while len(seq) < length:
            seq.append((seq[-1] + seq[-2]) % p)
        seqs.append(torch.tensor(seq, dtype=torch.long))
    return seqs


def test_collate_layout():
    p, eos_id, pos_base, L = 7, 8, 9, 6
    seqs = _addition_windows(p=p, n=16, length=L)
    random.seed(123)
    collate = make_corrupt_fix_collate(p=p, eos_id=eos_id, pos_base=pos_base)
    x, tag, targets, mask = collate(seqs)
    assert tag == BatchTag.CORRUPT_FIX
    B = len(seqs)
    assert x.shape == (B, L + 2) and targets.shape == (B, L + 3)
    assert mask.shape == (B, L + 2)
    clean = torch.stack(seqs)
    for i in range(B):
        diff = (x[i, :L] != clean[i]).nonzero().flatten().tolist()
        assert len(diff) == 1, f"sample {i}: expected exactly 1 corrupted position, got {diff}"
        k = diff[0]
        assert int(x[i, k]) != int(clean[i, k]), "wrong value must differ from the original"
        assert 0 <= int(x[i, k]) < p, "wrong value must stay a value token"
        # EOS + teacher-forced position token at the tail of the input
        assert int(x[i, L]) == eos_id
        assert int(x[i, L + 1]) == pos_base + k
        # targets = targets_override[:, 1:]: position token at slot L, correct
        # value (the ORIGINAL clean value) at slot L+1
        assert int(targets[i, L + 1]) == pos_base + k
        assert int(targets[i, L + 2]) == int(clean[i, k])
        # loss mask: exactly the two output positions
        ones = mask[i].nonzero().flatten().tolist()
        assert ones == [L, L + 1], f"sample {i}: loss mask at {ones}"
        # targets alignment: targets[:, 1:] == x shifted, with value appended
        assert torch.equal(targets[i, 1:L + 1], x[i, 1:L + 1])


def test_collate_fresh_randomness():
    # Two calls on the same batch corrupt (with high probability) different
    # positions/values — corruption is online, not baked into the dataset.
    p, L = 11, 8
    seqs = _addition_windows(p=p, n=32, length=L)
    collate = make_corrupt_fix_collate(p=p, eos_id=p + 1, pos_base=p + 2)
    random.seed(1)
    x1, _, _, _ = collate(seqs)
    random.seed(2)
    x2, _, _, _ = collate(seqs)
    assert not torch.equal(x1, x2), "corruption must be fresh per call"


def test_unpack_and_evaluate_integration():
    p, L = 7, 6
    eos_id, pos_base = p + 1, p + 2
    vocab_size = p + 2 + L
    seqs = _addition_windows(p=p, n=16, length=L)
    collate = make_corrupt_fix_collate(p=p, eos_id=eos_id, pos_base=pos_base)
    batch = collate(seqs)
    x, loss_mask, kwargs, ab_labels, targets_override = _unpack_batch(batch, 'cpu', None)
    assert kwargs == {} and ab_labels is None
    assert loss_mask is not None and targets_override is not None
    assert torch.equal(targets_override[:, :-1], x)

    torch.manual_seed(0)
    model = FibonacciTransformer(p=p, d_model=32, n_head=1, n_layer=1,
                                 block_size=16, dropout=0.0, vocab_size=vocab_size)
    loader = DataLoader(seqs, batch_size=8, collate_fn=collate)
    loss, acc, per_pos_acc, group_acc = evaluate(model, loader, 'cpu', num_mask=0)
    assert math.isfinite(loss) and loss > 0
    assert 0.0 <= acc <= 1.0
    # accuracy is counted only at the two output positions
    assert set(per_pos_acc) <= {L, L + 1}, f"unexpected acc positions: {per_pos_acc}"
    assert set(per_pos_acc) == {L, L + 1}


def _run_experiment(tmpdir, overrides):
    from core import run_experiment
    main = {
        'P': 7, 'ORDER': 2, 'RULES': [[1, 1]], 'TASK': 'single',
        'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
        'BATCH_SIZE': 32, 'EPOCHS': 2, 'LR': 0.001, 'RANDOM_SEED': 42,
        'TRAIN_LEN': 8, 'OOD_LEN': 10, 'DROPOUT': 0.0, 'MAX_UNIQUE_RATIO': 0.7,
        'WEIGHT_DECAY': 0.1, 'FIRST_TASK_WEIGHT': 1.0,
        'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0, 'EARLY_STOP_NO_IMPROVE': 3000,
        'MLP_RATIO': 4, 'NUM_MASK': None, 'MISSING_PROB': 0.0,
    }
    main.update(overrides)
    main['SAVE_PATH'] = os.path.join(tmpdir, 'model.pth')
    cfg = {'main': main, '_BATCH_RUN_MERGED': True}
    cfg_path = os.path.join(tmpdir, 'config.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_experiment(cfg_path)
    return buf.getvalue(), main['SAVE_PATH']


def test_run_experiment_corrupt_fix_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {'TASK_VARIANT': 'corrupt_fix'})
        assert os.path.exists(save_path), "checkpoint was not saved"
        assert 'Train: Loss=' in out, "training loop output missing"
        assert 'nan' not in out.lower(), "NaN appeared in training"
        assert '[corrupt_fix]' in out, "task banner missing variant tag"
        # POS_PROBE：corrupt_fix 每个 eval 打 pos/val 两行，各 16 个 k
        pos_lines = [l for l in out.splitlines() if l.startswith('  PosProbe pos|')]
        val_lines = [l for l in out.splitlines() if l.startswith('  PosProbe val|')]
        assert pos_lines and val_lines, "PosProbe lines missing"
        assert pos_lines[0].count('k') == 8 and val_lines[0].count('k') == 8  # TRAIN_LEN=8
        # 0.0% acc on a fresh model never beats best=0.0, so _best.pth is not
        # written this early; check the rolling full-state checkpoint instead.
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['variant'] == 'corrupt_fix'
        assert sc['scenario'] == 'single' and sc['rules'] == [[1, 1]]
        assert sc['eos_token_id'] == 8 and sc['pos_token_base'] == 9
        assert sc['vocab_size'] == 7 + 2 + 10  # max(TRAIN_LEN=8, OOD_LEN=10)
        assert sc['block_size'] == 16  # round_up_pow2(10 + 2)


def test_run_experiment_corrupt_fix_rejects_missing_prob():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'corrupt_fix',
                                     'MISSING_PROB': 0.1})
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected SystemExit(2) for MISSING_PROB conflict")


def test_run_experiment_corrupt_fix_order1_smoke():
    # ORDER=1（等比递推 x_k = c*x_{k-1}）：init_len=1，状态空间 = P。
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {'TASK_VARIANT': 'corrupt_fix',
                                                  'P': 17, 'ORDER': 1,
                                                  'RULES': [[3]]})
        assert os.path.exists(save_path), "checkpoint was not saved"
        assert 'Train: Loss=' in out and 'nan' not in out.lower()
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['variant'] == 'corrupt_fix' and sc['order'] == 1
        assert sc['rules'] == [[3]]


def test_collate_layout_parallel():
    # CORRUPT_FIX_PARALLEL：x 末位是第二个 EOS（不是位置 token），
    # targets/loss_mask 与 teacher forcing 版完全一致
    p, eos_id, pos_base, L = 7, 8, 9, 6
    seqs = _addition_windows(p=p, n=16, length=L)
    collate_tf = make_corrupt_fix_collate(p=p, eos_id=eos_id, pos_base=pos_base)
    collate_par = make_corrupt_fix_collate(p=p, eos_id=eos_id, pos_base=pos_base,
                                           parallel=True)
    random.seed(123)
    x_tf, tag, targets_tf, mask_tf = collate_tf(seqs)
    random.seed(123)
    x_par, _, targets_par, mask_par = collate_par(seqs)
    assert torch.equal(targets_tf, targets_par) and torch.equal(mask_tf, mask_par)
    assert torch.equal(x_tf[:, :-1], x_par[:, :-1])  # 仅末位不同
    assert (x_par[:, -1] == eos_id).all(), "parallel 模式末位必须是 EOS"
    assert (x_tf[:, -1] >= pos_base).all()  # 对照：TF 版末位是位置 token


def test_run_experiment_corrupt_fix_parallel_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {'TASK_VARIANT': 'corrupt_fix',
                                                  'CORRUPT_FIX_PARALLEL': True})
        assert os.path.exists(save_path)
        assert 'Train: Loss=' in out and 'nan' not in out.lower()
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['variant'] == 'corrupt_fix' and sc['parallel_decode'] is True


def test_collate_fixed_position_tuples():
    # (seq, k) 元组样本（探针/枚举用）：污损位置由样本携带
    p, eos_id, pos_base, L = 7, 8, 9, 6
    seqs = _addition_windows(p=p, n=4, length=L)
    batch = [(seq, k) for seq in seqs for k in range(L)]
    collate = make_corrupt_fix_collate(p=p, eos_id=eos_id, pos_base=pos_base)
    x, tag, targets, mask = collate(batch)
    for j, (seq, k) in enumerate(batch):
        diff = (x[j, :L] != seq).nonzero().flatten().tolist()
        assert diff == [k], f"corruption must be at carried k: {diff}"
        assert int(targets[j, L + 1]) == pos_base + k
        assert int(targets[j, L + 2]) == int(seq[k])


def test_collate_layout_paired_clean():
    # 配对双样本：裸序列产 [干净续写行, 污损答题行] 两行；元组只产答题行
    p, eos_id, pos_base, L = 7, 8, 9, 6
    seqs = _addition_windows(p=p, n=8, length=L)
    random.seed(123)
    collate = make_corrupt_fix_collate(p=p, eos_id=eos_id, pos_base=pos_base,
                                       body_supervision=True, body_from=1,
                                       paired_clean=True)
    x, tag, targets, mask = collate(seqs)
    assert tag == BatchTag.CORRUPT_FIX
    B = len(seqs)
    max_t = L + 3  # 污损行 tgt 长
    assert x.shape == (2 * B, max_t - 1)
    assert targets.shape == (2 * B, max_t)
    assert mask.shape == (2 * B, max_t - 1)
    for i in range(B):
        seq = seqs[i]
        xc, tc, mc = x[2 * i], targets[2 * i], mask[2 * i]          # 干净行
        xd, td, md = x[2 * i + 1], targets[2 * i + 1], mask[2 * i + 1]  # 污损行
        # 干净行：输入就是干净窗，tgt = [clean, EOS]
        assert torch.equal(xc[:L], seq)
        assert torch.equal(tc[:L], seq) and int(tc[L]) == eos_id
        # 干净行 mask：主体 [1, L-1) + EOS 槽 L-1，其余（含补齐位）为 0
        assert mc[L - 1] == 1.0 and mc[1].item() == 1.0
        assert mc[0].item() == 0.0 and mc[L:].sum().item() == 0.0
        # 污损行：恰一位污损，mask 只有 pos/val 两实槽（主体交给干净行）
        diff = (xd[:L] != seq).nonzero().flatten().tolist()
        assert len(diff) == 1
        k = diff[0]
        assert int(xd[L]) == eos_id and int(xd[L + 1]) == pos_base + k
        assert int(td[L + 1]) == pos_base + k and int(td[L + 2]) == int(seq[k])
        assert md.sum().item() == 2.0 and md[L] == 1.0 and md[L + 1] == 1.0
        assert md[:L].sum().item() == 0.0, "paired 污损行不应监督主体"
    # 元组路径：只产一行（探针语义不变）
    batch = [(seqs[0], 2), (seqs[1], 4)]
    _, _, targets2, mask2 = collate(batch)
    assert targets2.shape[0] == 2
    assert mask2[0].sum().item() == 2.0


def test_run_experiment_corrupt_fix_paired_clean_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'corrupt_fix',
            'CORRUPT_FIX_PAIRED_CLEAN': True})
        assert os.path.exists(save_path)
        assert 'CORRUPT_FIX_PAIRED_CLEAN' in out
        assert 'Train: Loss=' in out and 'nan' not in out.lower()
        assert 'PosProbe pos|' in out and 'PosProbe val|' in out
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        sc = latest['config']
        assert sc['variant'] == 'corrupt_fix' and sc['paired_clean'] is True


def test_paired_clean_gates():
    # 仅 corrupt_fix；与 PARALLEL 互斥（归一化错误 -> SystemExit(2)）
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'fill_blank',
                                     'CORRUPT_FIX_PAIRED_CLEAN': True})
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError('paired_clean on fill_blank must exit(2)')
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'corrupt_fix',
                                     'CORRUPT_FIX_PAIRED_CLEAN': True,
                                     'CORRUPT_FIX_PARALLEL': True})
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError('paired_clean + parallel must exit(2)')


def test_collate_layout_mask_around():
    # 动态主体 mask：污损位 k 牵连预测位 {k-1,k,k+1}（预测 x_k/x_{k+1}/x_{k+2}）
    p, eos_id, pos_base, L = 7, 8, 9, 8
    seqs = _addition_windows(p=p, n=4, length=L)
    collate = make_corrupt_fix_collate(p=p, eos_id=eos_id, pos_base=pos_base,
                                       body_supervision=True, body_from=1,
                                       mask_around=True)
    # 用元组固定 k，逐一验证（k=3 中部 / k=0 边界 / k=L-1 边界）
    for k, expect_masked in [(3, {2, 3, 4}), (0, {1}), (L - 1, {L - 2})]:
        batch = [(seq, k) for seq in seqs]
        _, _, _, mask = collate(batch)
        for i in range(len(seqs)):
            body = {j for j in range(1, L - 1)}  # 主体监督区间 [1, L-2]
            want = body - expect_masked
            got = {j for j in range(L + 2) if mask[i, j] > 0} - {L, L + 1}
            assert got == want, f'k={k}: masked body slots {got} != {want}'
            assert mask[i, L] == 1.0 and mask[i, L + 1] == 1.0  # pos/val 不动


def test_run_experiment_corrupt_fix_mask_around_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'corrupt_fix',
            'CORRUPT_FIX_MASK_AROUND': True})
        assert os.path.exists(save_path)
        assert 'CORRUPT_FIX_MASK_AROUND' in out
        assert 'Train: Loss=' in out and 'nan' not in out.lower()
        # 相对位置五桶探针行
        assert re.search(r'PosProbe posR\| first=[\d.]+ second=[\d.]+ mid=[\d.]+ last2=[\d.]+ last=[\d.]+', out)
        assert 'PosProbe valR|' in out
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        assert latest['config']['mask_around'] is True


def test_run_experiment_corrupt_fix_all_positions_smoke():
    # 逐位置枚举污损：样本数 × 窗口长度（不定长下按截断后长度）
    with tempfile.TemporaryDirectory() as tmpdir:
        out, save_path = _run_experiment(tmpdir, {
            'TASK_VARIANT': 'corrupt_fix',
            'CORRUPT_FIX_ALL_POSITIONS': True,
            'TRAIN_LEN': 8, 'OOD_LEN': 8})
        assert os.path.exists(save_path)
        m = re.search(r'CORRUPT_FIX_ALL_POSITIONS: (\d+) windows -> (\d+) train samples', out)
        assert m, out[-2000:]
        n_win, n_samp = int(m.group(1)), int(m.group(2))
        assert n_samp == n_win * 8, f'{n_samp} != {n_win}*8'
        assert 'Train: Loss=' in out and 'nan' not in out.lower()
        latest = torch.load(os.path.splitext(save_path)[0] + '_latest.pth',
                            weights_only=False)
        assert latest['config']['all_positions'] is True


def test_all_positions_gates():
    # 仅 corrupt_fix；与 PAIRED_CLEAN 互斥（归一化错误 -> SystemExit(2)）
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'fill_blank',
                                     'CORRUPT_FIX_ALL_POSITIONS': True})
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError('all_positions on fill_blank must exit(2)')
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _run_experiment(tmpdir, {'TASK_VARIANT': 'corrupt_fix',
                                     'CORRUPT_FIX_ALL_POSITIONS': True,
                                     'CORRUPT_FIX_PAIRED_CLEAN': True})
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError('all_positions + paired_clean must exit(2)')


if __name__ == '__main__':
    test_collate_layout()
    print('PASS test_collate_layout')
    test_collate_fixed_position_tuples()
    print('PASS test_collate_fixed_position_tuples')
    test_collate_fresh_randomness()
    print('PASS test_collate_fresh_randomness')
    test_unpack_and_evaluate_integration()
    print('PASS test_unpack_and_evaluate_integration')
    test_run_experiment_corrupt_fix_smoke()
    print('PASS test_run_experiment_corrupt_fix_smoke')
    test_run_experiment_corrupt_fix_rejects_missing_prob()
    print('PASS test_run_experiment_corrupt_fix_rejects_missing_prob')
    test_run_experiment_corrupt_fix_order1_smoke()
    print('PASS test_run_experiment_corrupt_fix_order1_smoke')
    test_collate_layout_parallel()
    print('PASS test_collate_layout_parallel')
    test_run_experiment_corrupt_fix_parallel_smoke()
    print('PASS test_run_experiment_corrupt_fix_parallel_smoke')
    test_collate_layout_paired_clean()
    print('PASS test_collate_layout_paired_clean')
    test_run_experiment_corrupt_fix_paired_clean_smoke()
    print('PASS test_run_experiment_corrupt_fix_paired_clean_smoke')
    test_paired_clean_gates()
    print('PASS test_paired_clean_gates')
    test_collate_layout_mask_around()
    print('PASS test_collate_layout_mask_around')
    test_run_experiment_corrupt_fix_mask_around_smoke()
    print('PASS test_run_experiment_corrupt_fix_mask_around_smoke')
    test_run_experiment_corrupt_fix_all_positions_smoke()
    print('PASS test_run_experiment_corrupt_fix_all_positions_smoke')
    test_all_positions_gates()
    print('PASS test_all_positions_gates')
    print('ALL TESTS PASSED: test_corrupt_fix.py')
