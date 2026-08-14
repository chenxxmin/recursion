"""Verify circular (Fourier) structure of token representations after attention.

For every token value v in [0, p-1], extract its post-attention hidden vector
and test whether the p points lie uniformly on a circle under SOME linear
projection -- the geometric signature of a linear layer having learned modular
addition. Two context modes:

  pairs : input [w, v] for every w, and analyze the p position-1 vectors per
          context w separately (the model's real usage: v as the second
          initial value; position 1 is the first position with attention
          mixing). This is the primary mode.
  single: input [v] alone and analyze the p position-0 vectors (pure
          embedding + attention projection, no context mixing).

Method (per group of p vectors H):
  1. Center H and DFT along the token axis, giving a per-frequency power
     spectrum. A circular representation concentrates power in one frequency.
  2. The dominant frequency's complex coefficient vector (Re, Im) spans the
     projection plane; on it the points are approximately circular by
     construction if that frequency dominates.
  3. Metrics per group:
     - top1/top5 power share: frequency concentration
     - radius_cv: CV of point radii in the projection plane (0 = on a circle)
     - angle_coherence: |mean_v exp(i (theta_v - 2 pi k* v / p))|, 1.0 when
       angles form a perfect arithmetic progression in v (modular addition)
     - shuffled baseline: angle_coherence after permuting the token order
       (destroys the arithmetic structure; the gap is the signal)

Usage: python src/verify_circle.py <model.pth> [--context pairs|single] [--plot out.png]
"""
import argparse
import math
import os
import sys

import torch

# Allow running from repo root as: python src/verify_circle.py <pth>
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import core
from analyze_attention import load_model, _forward_per_layer, HEADER_WIDTH


def extract_hidden_after_attention(model, input_ids, ab_label=None):
    """Return per-layer post-attention residuals from the shared manual forward.

    input_ids: (B, T). Returns a list with one (B, T, d) tensor per layer,
    each the residual stream right after x = x + attn(ln_1(x)).
    """
    return [layer['hidden_post_attn']
            for layer in _forward_per_layer(model, input_ids, ab_label=ab_label)]


def _held_out_coherence(H, k_star):
    """Angle coherence with a held-out projection direction (unbiased).

    Fit the projection direction on one half of the tokens, then test whether
    the other half's angles continue the arithmetic progression 2*pi*k*v/p.
    Averaged over both split directions. (Using the same data to both pick the
    projection and score it injects the target phase via the Gram diagonal,
    biasing coherence upward even for random matrices.)
    """
    p = H.shape[0]
    v = torch.arange(p, dtype=torch.float32)
    vals = []
    for fit_idx, test_idx in ((torch.arange(0, p, 2), torch.arange(1, p, 2)),
                              (torch.arange(1, p, 2), torch.arange(0, p, 2))):
        Hf = H[fit_idx] - H[fit_idx].mean(dim=0, keepdim=True)
        basis = torch.exp(-2j * math.pi * k_star * v[fit_idx] / p).unsqueeze(1)
        u = (Hf * basis).sum(dim=0).conj()
        Ht = H[test_idx] - H[test_idx].mean(dim=0, keepdim=True)
        proj = torch.stack([Ht @ u.real, Ht @ u.imag], dim=-1)
        theta = torch.atan2(proj[:, 1], proj[:, 0])
        phase = theta - 2 * math.pi * k_star * v[test_idx] / p
        vals.append(torch.complex(torch.cos(phase).mean(),
                                  torch.sin(phase).mean()).abs().item())
    return sum(vals) / len(vals)


def circle_metrics(H, k_star=None):
    """Compute circle-structure metrics for one group of p vectors.

    H: (p, d) — one hidden vector per token value 0..p-1.
    k_star: optionally force the analysis frequency (default: argmax power).
    Returns a dict with the metrics plus the 2D projection for plotting.
    """
    p = H.shape[0]
    Hc = H - H.mean(dim=0, keepdim=True)
    spec = torch.fft.fft(Hc, dim=0)                 # (p, d) complex
    power = spec.abs().square().sum(dim=1)          # (p,)
    total = power.sum().clamp(min=1e-12)

    # Real input: frequencies k and p-k are conjugate pairs with equal power.
    # Fold them so a single circular harmonic shows up as one peak.
    half = p // 2  # p is an odd prime; no Nyquist bin
    idx = torch.arange(1, half + 1)
    folded = power[idx] + power[p - idx]            # index k-1 -> frequency k
    topk = folded.topk(min(5, half)).values
    if k_star is None:
        k_star = int(folded.argmax()) + 1

    # spec[k*] picks up the e^{-i theta} component, so its conjugate gives the
    # projection plane in which angles rotate the positive way (+2 pi k* v / p).
    u = spec[k_star].conj()                         # (d,) complex coefficients
    proj = torch.stack([Hc @ u.real, Hc @ u.imag], dim=-1)  # (p, 2)

    r = proj.norm(dim=1)
    radius_cv = (r.std() / r.mean().clamp(min=1e-12)).item()

    return {
        'k_star': k_star,
        'top1_share': (topk[0] / total).item(),
        'top5_share': (topk.sum() / total).item(),
        'radius_cv': radius_cv,
        'angle_coherence': _held_out_coherence(H, k_star),
        'proj': proj,
    }


def shuffled_coherence(H, k_star, seed):
    """angle_coherence after permuting the token order (baseline)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(H.shape[0], generator=g)
    return circle_metrics(H[perm], k_star=k_star)['angle_coherence']


def collect_hidden_vectors(model, p, context, device):
    """Collect post-attention vectors for all tokens, per layer.

    Returns a list over layers of (p, p, d) tensors ([w][v]) for context
    'pairs', or (1, p, d) for context 'single'.
    """
    v_all = torch.arange(p, device=device)
    if context == 'single':
        with torch.no_grad():
            hs = extract_hidden_after_attention(model, v_all.unsqueeze(1))
        return [h[:, 0, :].float().cpu().unsqueeze(0) for h in hs]

    layers = None
    for w in range(p):
        ids = torch.stack([torch.full_like(v_all, w), v_all], dim=1)  # (p, 2)
        with torch.no_grad():
            hs = extract_hidden_after_attention(model, ids)
        if layers is None:
            d = hs[0].shape[-1]
            layers = [torch.empty(p, p, d) for _ in hs]
        for li, h in enumerate(hs):
            layers[li][w] = h[:, 1, :].float().cpu()
    return layers


def summarize_layer(H, layer_idx, seed):
    """Print the metric distribution over contexts for one layer."""
    p = H.shape[1]
    per_ctx = [circle_metrics(H[w]) for w in range(H.shape[0])]

    k_stars = [m['k_star'] for m in per_ctx]
    k_common, k_count = max(((k, k_stars.count(k)) for k in set(k_stars)),
                            key=lambda t: t[1])
    shares = [m['top1_share'] for m in per_ctx]
    cvs = [m['radius_cv'] for m in per_ctx]
    cohs = [m['angle_coherence'] for m in per_ctx]
    baselines = [shuffled_coherence(H[w], k_common, seed + w)
                 for w in range(H.shape[0])]

    mean = lambda xs: sum(xs) / len(xs)
    print(f"\n--- Layer {layer_idx} ---")
    print(f"dominant frequency k*: {k_common} (in {k_count}/{len(k_stars)} contexts)")
    print(f"top1 power share : mean {mean(shares):.3f} | min {min(shares):.3f}")
    print(f"radius CV        : mean {mean(cvs):.3f} | max {max(cvs):.3f}")
    print(f"angle coherence  : mean {mean(cohs):.3f} | min {min(cohs):.3f} "
          f"| shuffled baseline {mean(baselines):.3f}")

    return per_ctx, k_common


def plot_projections(per_ctx, H, k_common, layer_idx, title_prefix, ax):
    """Scatter the projection of the median-coherence context, colored by v."""
    import matplotlib.pyplot as plt
    mid = sorted(range(len(per_ctx)), key=lambda i: per_ctx[i]['angle_coherence'])[len(per_ctx) // 2]
    m = circle_metrics(H[mid], k_star=k_common)
    proj = m['proj'].numpy()
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=range(H.shape[1]),
                    cmap='viridis', s=12)
    ax.set_title(f"{title_prefix} L{layer_idx} ctx#{mid}\n"
                 f"k*={m['k_star']} CV={m['radius_cv']:.3f} coh={m['angle_coherence']:.3f}",
                 fontsize=9)
    ax.set_aspect('equal')
    return sc


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('pth_path', help='Path to the model checkpoint (.pth).')
    parser.add_argument('--context', choices=['pairs', 'single'], default='pairs',
                        help='Context mode for collecting token vectors (default: pairs).')
    parser.add_argument('--plot', default=None, help='Optional path to save projection scatter PNG.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the shuffled baseline.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint = load_model(args.pth_path, device=device)
    config = checkpoint['config']
    p = config['p']

    if isinstance(model, core.MixedABTransformer):
        print("[Warning] verify_circle targets single-recurrence checkpoints; "
              "mixed_ab models have rule tokens in the input layout and the "
              "results will not be meaningful.")

    print("=" * HEADER_WIDTH)
    print(f"Circle structure verification: {os.path.basename(args.pth_path)}")
    print(f"p={p}, context={args.context}, device={device}")
    print("=" * HEADER_WIDTH)

    layers = collect_hidden_vectors(model, p, args.context, device)

    per_layer = []
    for li, H in enumerate(layers):
        per_ctx, k_common = summarize_layer(H, li, args.seed)
        per_layer.append((per_ctx, k_common))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5),
                                 squeeze=False)
        for li, H in enumerate(layers):
            per_ctx, k_common = per_layer[li]
            sc = plot_projections(per_ctx, H, k_common, li,
                                  os.path.basename(args.pth_path), axes[0][li])
            fig.colorbar(sc, ax=axes[0][li], label='token value')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nSaved projection plot to {args.plot}")

    print("\n" + "=" * HEADER_WIDTH)
    print("Interpretation: high top1 share + low radius CV + coherence >> shuffled")
    print("baseline means a circular (modular-addition) representation exists.")
    print("=" * HEADER_WIDTH)


if __name__ == '__main__':
    main()
