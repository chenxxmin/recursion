"""Pure-CPU tests for probe_fillblank_attention_maps.py (no checkpoint loaded)."""
import contextlib
import io
import os
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import probe_fillblank_attention_maps as probe
from analyze_attention import _forward_per_layer
from models import FibonacciTransformer


def _assert_raises(exception_type, fn):
    try:
        fn()
    except exception_type:
        return
    assert False, f'expected {exception_type.__name__}'


def _causal_attention(shape):
    torch.manual_seed(20261104)
    scores = torch.randn(shape)
    mask = torch.tril(torch.ones(shape[-1], shape[-1], dtype=torch.bool))
    return torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)


def _synthetic_by_blank():
    attention = torch.zeros(4, 4, 6, 6)
    for query in range(6):
        attention[:, :, query, :query + 1] = 1.0 / (query + 1)
    return {f'x{k}': {
        'k': k,
        'N': 32,
        'attention_mean': (attention * (1.0 - 0.05 * k)).tolist(),
    } for k in range(5)}


def test_input_6x6_head_preserving_mean_and_blank_eos_labels():
    windows = torch.arange(10).reshape(2, 5)
    original = windows.clone()
    inputs = probe.make_blank_inputs(windows, 3, blank_token=127, eos_token=128)
    assert inputs.shape == (2, 6)
    assert inputs[:, 3].tolist() == [127, 127]
    assert inputs[:, 5].tolist() == [128, 128]
    assert torch.equal(windows, original)
    for k in range(5):
        labels = probe.token_labels(k)
        assert len(labels) == 6 and labels[k] == f'BLANK(x{k})'
        assert labels[-1] == 'EOS'
        assert labels[:5] == [f'BLANK(x{k})' if j == k else f'x{j}' for j in range(5)]
    _assert_raises(ValueError, lambda: probe.token_labels(5))

    raw = torch.zeros(2, 4, 6, 6)
    for batch in range(2):
        for head in range(4):
            raw[batch, head] = (batch + 1) * (head + 1)
    mean = probe.mean_over_batch(raw)
    assert mean.shape == (4, 6, 6)
    assert mean[:, 0, 0].tolist() == [1.5, 3.0, 4.5, 6.0]
    torch.testing.assert_close(mean, raw.mean(dim=0), rtol=0, atol=0)
    head_averaged = raw.mean(dim=(0, 1))
    assert not torch.equal(mean[0], head_averaged)
    assert not torch.equal(mean[1], mean[2])


def test_causal_row_sum_upper_triangle_and_query0_validation():
    attention = _causal_attention((2, 3, 6, 6))
    metrics = probe.validate_attention(attention)
    assert metrics['shape'] == [2, 3, 6, 6]
    assert metrics['all_passed'] and metrics['row_sum_passed']
    assert metrics['upper_triangle_passed'] and metrics['query0_self_passed']
    assert metrics['row_sum_max_abs_error'] <= probe.ATTENTION_ATOL
    assert metrics['upper_triangle_max_abs'] == 0.0
    assert metrics['query0_self_max_abs_error'] == 0.0

    bad_rows = attention.clone()
    bad_rows[0, 0, 2, 2] += 0.1
    _assert_raises(RuntimeError, lambda: probe.validate_attention(bad_rows))
    bad_future = attention.clone()
    bad_future[0, 0, 1, 2] = 0.1
    _assert_raises(RuntimeError, lambda: probe.validate_attention(bad_future))
    bad_query0 = attention.clone()
    bad_query0[..., 0, 0] = 0.5
    _assert_raises(RuntimeError, lambda: probe.validate_attention(bad_query0))
    nonfinite = attention.clone()
    nonfinite[0, 0, 0, 0] = torch.nan
    _assert_raises(RuntimeError, lambda: probe.validate_attention(nonfinite))


def test_tiny_real_transformer_hook_matches_independent_forward():
    torch.manual_seed(20261104)
    model = FibonacciTransformer(
        p=5, d_model=8, n_head=2, n_layer=2, block_size=6,
        dropout=0.0, mlp_ratio=2, vocab_size=7).eval()
    windows = torch.tensor([
        [1, 2, 3, 4, 0],
        [2, 1, 4, 3, 2],
        [0, 3, 1, 4, 2],
    ], dtype=torch.long)
    inputs = probe.make_blank_inputs(windows, 2, blank_token=5, eos_token=6)
    logits, hooked, validation = probe.capture_batch(model, inputs)
    assert len(hooked) == 2
    assert all(att.shape == (3, 2, 6, 6) for att in hooked)
    assert validation['allclose']
    assert validation['n_elements_compared'] == 2 * 3 * 2 * 6 * 6

    replayed = _forward_per_layer(model, inputs)
    reference_logits, _ = model(inputs)
    torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
    for layer, (hook_att, replay, check) in enumerate(
            zip(hooked, replayed, validation['layers'])):
        replay_att = replay['attn_weights'].detach().cpu()
        assert check['layer_index'] == layer and check['allclose']
        torch.testing.assert_close(
            hook_att, replay_att, rtol=probe.REPLAY_RTOL, atol=probe.REPLAY_ATOL)
        assert check['max_abs_diff'] <= probe.REPLAY_ATOL
        assert check['causal']['all_passed']
        probe.validate_attention(hook_att)
    assert all(not block.attn._forward_hooks for block in model.transformer.h)


def test_four_head_validation_without_checkpoint():
    blocks = [SimpleNamespace(attn=SimpleNamespace(n_head=4)) for _ in range(4)]
    model = SimpleNamespace(transformer=SimpleNamespace(h=blocks))
    probe.validate_four_heads(model, {'n_head': 4})
    _assert_raises(ValueError, lambda: probe.validate_four_heads(model, {'n_head': 2}))
    bad_blocks = [SimpleNamespace(attn=SimpleNamespace(n_head=4)) for _ in range(4)]
    bad_blocks[2].attn.n_head = 8
    bad_model = SimpleNamespace(transformer=SimpleNamespace(h=bad_blocks))
    _assert_raises(ValueError, lambda: probe.validate_four_heads(bad_model, {'n_head': 4}))


def test_five_plot_smoke_and_unified_color_scale():
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg

    by_blank = _synthetic_by_blank()
    fig = probe.make_attention_figure(by_blank['x2']['attention_mean'], 2, 32)
    try:
        images = [image for ax in fig.axes for image in ax.images]
        assert len(images) == 16
        assert all(image.get_array().shape == (6, 6) for image in images)
        assert all(image.get_clim() == (0.0, 1.0) for image in images)
        assert all(image.get_cmap().name == 'viridis' for image in images)
        panel_axes = [ax for ax in fig.axes if ax.images]
        assert len(panel_axes) == 16
        for ax in panel_axes:
            assert ax.get_xlabel() == 'Key token' and ax.get_ylabel() == 'Query token'
            assert [tick.get_text() for tick in ax.get_xticklabels()] == probe.token_labels(2)
            assert [tick.get_text() for tick in ax.get_yticklabels()] == probe.token_labels(2)
    finally:
        plt.close(fig)

    with tempfile.TemporaryDirectory() as temp_dir:
        prefix = os.path.join(temp_dir, 'attention')
        paths = probe.make_plots(by_blank, prefix)
        assert len(paths) == 5
        assert [os.path.basename(path) for path in paths] == [
            f'attention_blank_x{k}.png' for k in range(5)]
        for path in paths:
            assert os.path.isfile(path) and os.path.getsize(path) > 1000
            image = mpimg.imread(path)
            assert image.ndim == 3 and image.shape[0] > 100 and image.shape[1] > 100
        assert sorted(os.listdir(temp_dir)) == sorted(os.path.basename(path) for path in paths)


def test_cli_defaults_and_dry_run_plan():
    args = probe.parse_args(['--dry-run'])
    assert args.device == 'cpu' and args.threads == 16 and args.batch_size == 32
    assert args.ckpt == probe.DEFAULT_CKPT and args.base_data == probe.DEFAULT_BASE_DATA
    assert args.out == 'reports/data/fillblank_len5_attention_a25b56.json'
    assert args.plot_prefix == 'reports/circuit_analysis/plots/fillblank_len5_attention_a25b56'
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        probe.print_dry_run(args)
    text = output.getvalue()
    assert 'no data or model loaded' in text
    assert 'output shape [layer][head][query][key]=4x4x6x6' in text
    assert 'rtol=1e-05, atol=1e-06' in text
    for k in range(5):
        assert f'BLANK(x{k})' in text
        assert f'attention_a25b56_blank_x{k}.png' in text


if __name__ == '__main__':
    torch.set_num_threads(1)
    tests = [value for name, value in sorted(globals().items())
             if name.startswith('test_') and callable(value)]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'\nALL TESTS PASSED: {len(tests)}/{len(tests)} (pure CPU; no checkpoint loaded)')
