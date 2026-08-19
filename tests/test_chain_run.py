"""Tests for INIT_FROM partial checkpoint loading and chain_run stage wiring."""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import torch

from experiment import _load_partial_checkpoint
from models import FibonacciTransformer, MixedABTransformer


def test_load_partial_checkpoint():
    p = 7
    src = FibonacciTransformer(p=p, d_model=32, n_head=1, n_layer=1, block_size=32)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, 'src.pth')
        torch.save({'model_state_dict': src.state_dict(),
                    'config': {'p': p, 'a': 1, 'b': 1, 'recurrence': 'addition'}}, ckpt)

        torch.manual_seed(123)
        dst = MixedABTransformer(num_ab_pairs=2, order=2, p=p, d_model=32, n_head=1,
                                 n_layer=1, block_size=32, use_ab_tag=False)
        rule_head_before = dst.rule_head[0].weight.detach().clone()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _load_partial_checkpoint(dst, ckpt, 'cpu')
        out = buf.getvalue()

        # shared tensors copied, rule_head untouched (still its own fresh init)
        assert torch.equal(dst.transformer.wte.weight, src.transformer.wte.weight)
        assert torch.equal(dst.transformer.h[0].attn.c_attn.weight,
                           src.transformer.h[0].attn.c_attn.weight)
        assert torch.equal(dst.rule_head[0].weight, rule_head_before)
        # source (single-rule model) is a strict subset of the target: all its
        # tensors load, nothing is skipped
        assert 'loaded' in out and 'skipped 0' in out


def _tiny_chain(tmpdir, gate):
    """A 2-stage chain with one seed; stage 2 resumes from stage 1."""
    main = {'P': 7, 'A': 1, 'B': 1, 'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
            'BATCH_SIZE': 32, 'EPOCHS': 2, 'LR': 0.001, 'TRAIN_LEN': 8,
            'OOD_LEN': 10, 'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0}
    chain = {'stages': [
        {'name': 's1', 'gate': {'min_best_accuracy': gate},
         'experiments': [{'name': 'c_A_seed0', 'task': 'addition',
                          'config': dict(main)}]},
        {'name': 's2',
         'experiments': [{'name': 'c_AB_seed0', 'task': 'mixed_ab',
                          'config': {**main, 'AB_PAIRS': [[1, 1], [1, 2]],
                                     'USE_AB_TAG': False,
                                     'MIXED_AB_MAX_UNIQUE_RATIOS': [0.7, 0.7],
                                     'INIT_FROM': '@prev'}}]},
    ]}
    path = os.path.join(tmpdir, 'tiny_chain.json')
    with open(path, 'w') as f:
        json.dump(chain, f)
    return path


def _run_chain(chain_path, tmpdir):
    import chain_run
    argv = sys.argv
    sys.argv = ['chain_run.py', chain_path,
                '--base-dir', os.path.join(tmpdir, 'out'),
                '--model-base-dir', os.path.join(tmpdir, 'models')]
    try:
        chain_run.main()
    finally:
        sys.argv = argv


def test_chain_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        chain_path = _tiny_chain(tmpdir, gate=0.0)  # gate always passes
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _run_chain(chain_path, tmpdir)
        out = buf.getvalue()
        stem = os.path.splitext(os.path.basename(chain_path))[0]
        ckpt2 = os.path.join(tmpdir, 'models', f'{stem}_s2', 'c_AB_seed0.pth')
        assert os.path.exists(ckpt2), 'stage-2 checkpoint missing'
        assert 'INIT_FROM <- s1/c_A_seed0' in out, '@prev was not resolved'
        # the phase-2 training log must show the partial-load report
        log2 = os.path.join(tmpdir, 'out', f'{stem}_s2', 'logs', 'c_AB_seed0.log')
        with open(log2, encoding='utf-8') as f:
            log = f.read()
        assert '[INIT_FROM]' in log and 'loaded' in log


def test_chain_gate_aborts():
    with tempfile.TemporaryDirectory() as tmpdir:
        chain_path = _tiny_chain(tmpdir, gate=1.01)  # unreachable -> must abort
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _run_chain(chain_path, tmpdir)
            assert False, 'expected RuntimeError from the accuracy gate'
        except RuntimeError as e:
            assert 'gate' in str(e)
        stem = os.path.splitext(os.path.basename(chain_path))[0]
        ckpt2 = os.path.join(tmpdir, 'models', f'{stem}_s2', 'c_AB_seed0.pth')
        assert not os.path.exists(ckpt2), 'stage 2 ran despite the gate abort'


def test_freeze_from_start():
    """COND_FIX_START=0 must freeze WTE before the first gradient step."""
    import experiment
    from training import run_training_engine
    cfg = {'main': {'P': 7, 'A': 1, 'B': 1, 'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
                    'BATCH_SIZE': 32, 'TRAIN_LEN': 8, 'OOD_LEN': 10, 'DROPOUT': 0.0,
                    'ENTROPY_PENALTY_WEIGHT': 0.0, 'WEIGHT_DECAY': 0.1,
                    'MAX_UNIQUE_RATIO': 0.7}}
    ctx = experiment._prepare_single_recurrence(cfg, 'addition')
    model = ctx['model']
    wte_before = model.transformer.wte.weight.detach().clone()
    attn_before = model.transformer.h[0].attn.c_attn.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        with contextlib.redirect_stdout(io.StringIO()):
            run_training_engine(
                model, ctx['train_loader'], ctx['test_loader'], optimizer, scheduler, 'cpu',
                epochs=2, eval_interval=1, early_stop_accuracy=2.0,
                early_stop_no_improve=3000,
                save_path=os.path.join(tmpdir, 'm.pth'), save_config=ctx['save_config'],
                num_mask=ctx['num_mask'], extra_kwargs_fn=ctx['extra_kwargs_fn'],
                first_task_weight=1.0,
                cond_fix='WTE', cond_fix_start=0)
    # WTE frozen from the start: unchanged (lm_head is tied, same tensor);
    # attention weights must still have trained
    assert torch.equal(model.transformer.wte.weight, wte_before)
    assert not torch.equal(model.transformer.h[0].attn.c_attn.weight, attn_before)


def test_chain_named_stage_ref():
    """INIT_FROM '@<stage>' resolves to the named earlier stage, not just the
    previous one (stage 3 referencing stage 1)."""
    main = {'P': 7, 'A': 1, 'B': 1, 'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
            'BATCH_SIZE': 32, 'EPOCHS': 2, 'LR': 0.001, 'TRAIN_LEN': 8,
            'OOD_LEN': 10, 'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0}
    mixed = {**main, 'AB_PAIRS': [[1, 1], [1, 2]], 'USE_AB_TAG': False,
             'MIXED_AB_MAX_UNIQUE_RATIOS': [0.7, 0.7]}
    chain = {'stages': [
        {'name': 's1', 'experiments': [{'name': 'c_A_seed0', 'task': 'addition',
                                        'config': dict(main)}]},
        {'name': 's2', 'experiments': [{'name': 'c_AB_seed0', 'task': 'mixed_ab',
                                        'config': {**mixed, 'INIT_FROM': '@s1'}}]},
        {'name': 's3', 'experiments': [{'name': 'c_ABF_seed0', 'task': 'mixed_ab',
                                        'config': {**mixed, 'INIT_FROM': '@s1'}}]},
    ]}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'named_chain.json')
        with open(path, 'w') as f:
            json.dump(chain, f)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _run_chain(path, tmpdir)
        out = buf.getvalue()
        stem = os.path.splitext(os.path.basename(path))[0]
        for stage, exp in (('s2', 'c_AB_seed0'), ('s3', 'c_ABF_seed0')):
            ckpt = os.path.join(tmpdir, 'models', f'{stem}_{stage}', f'{exp}.pth')
            assert os.path.exists(ckpt), f'{stage} checkpoint missing'
        # both downstream stages must reference the stage-1 checkpoint
        assert out.count('INIT_FROM <- s1/c_A_seed0') == 2, out


def test_chain_from_stage_resume():
    """--from-stage skips earlier stages (verifying their checkpoints exist)
    and resolves @refs against them without re-running them."""
    main_cfg = {'P': 7, 'A': 1, 'B': 1, 'D_MODEL': 32, 'N_HEAD': 1, 'N_LAYER': 1,
                'BATCH_SIZE': 32, 'EPOCHS': 2, 'LR': 0.001, 'TRAIN_LEN': 8,
                'OOD_LEN': 10, 'EVAL_INTERVAL': 1, 'EARLY_STOP_ACCURACY': 2.0}
    mixed = {**main_cfg, 'AB_PAIRS': [[1, 1], [1, 2]], 'USE_AB_TAG': False,
             'MIXED_AB_MAX_UNIQUE_RATIOS': [0.7, 0.7]}
    chain = {'stages': [
        {'name': 's1', 'experiments': [{'name': 'r_A_seed0', 'task': 'addition',
                                        'config': dict(main_cfg)}]},
        {'name': 's2', 'experiments': [{'name': 'r_AB_seed0', 'task': 'mixed_ab',
                                        'config': {**mixed, 'INIT_FROM': '@s1'}}]},
    ]}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'resume_chain.json')
        with open(path, 'w') as f:
            json.dump(chain, f)
        stem = 'resume_chain'
        # fake a completed s1: an empty state dict loads as "0 tensors" but is
        # a valid INIT_FROM source
        s1_dir = os.path.join(tmpdir, 'models', f'{stem}_s1')
        os.makedirs(s1_dir)
        torch.save({'model_state_dict': {}}, os.path.join(s1_dir, 'r_A_seed0.pth'))

        import chain_run
        argv = sys.argv
        sys.argv = ['chain_run.py', path,
                    '--base-dir', os.path.join(tmpdir, 'out'),
                    '--model-base-dir', os.path.join(tmpdir, 'models'),
                    '--from-stage', 's2']
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                chain_run.main()
            out = buf.getvalue()
        finally:
            sys.argv = argv

        assert 'resuming at stage' in out
        assert os.path.exists(os.path.join(tmpdir, 'models', f'{stem}_s2',
                                           'r_AB_seed0.pth'))
        # s1 must NOT have been re-run (no log dir for its batch)
        assert not os.path.exists(os.path.join(tmpdir, 'out', f'{stem}_s1'))


if __name__ == '__main__':
    test_load_partial_checkpoint()
    test_chain_end_to_end()
    test_chain_gate_aborts()
    test_freeze_from_start()
    test_chain_named_stage_ref()
    test_chain_from_stage_resume()
    print('ALL TESTS PASSED: test_chain_run.py')
