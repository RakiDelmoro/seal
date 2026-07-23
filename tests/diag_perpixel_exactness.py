"""Diagnostic 4-alt.6.1 — exactness invariant under PerPixelThreshold (move A).

The non-negotiable safety check (Stage-1 Test A): with theta=0, the event
layer's running output MUST equal a dense layer applied to the full frame
sequence, to within 1e-5. This must hold under the new per-element threshold,
otherwise move A breaks the encoder's core guarantee.

PerPixelThreshold starts with theta=0 (cold start, var=0 -> theta=0 -> all
fire == dense), so exactness must hold at init. We also check:
  - after running with theta>0 (k=3), resetting theta to 0 restores exactness
  - reset_cache (episode boundary) still re-seeds exactly
  - EventLinear exactness under PerPixelThreshold

Also a sanity check: with k=3, per-pixel theta rises above 0 and the event
rate settles well clear of both 0 (not dead) and 1 (not dense).
"""
import numpy as np
import torch
import torch.nn.functional as F

from model.event_layers import EventConv2d, EventLinear
from model.thresholds import PerPixelThreshold
from tests.test_stage1_encoder import _load_frames


def test_perpixel_exactness_conv():
    frames = _load_frames(50)
    th = PerPixelThreshold(theta0=0.0, warmup_steps=10_000)
    ev = EventConv2d(1, 8, 8, 5, th)
    ref = F.conv2d(frames, ev.weight, ev.bias, 5)
    outs = [ev(frames[i:i + 1]).detach() for i in range(frames.shape[0])]
    out = torch.cat(outs, dim=0)
    err = (out - ref).abs().max().item()
    print(f"  [A-pp] EventConv2d + PerPixel exactness (theta0=0): max err {err:.2e}")
    assert err < 1e-5, f"PerPixel exactness failed: {err}"


def test_perpixel_exactness_linear():
    torch.manual_seed(0)
    x_seq = torch.randn(40, 1, 16)
    th = PerPixelThreshold(theta0=0.0, warmup_steps=10_000)
    ev = EventLinear(16, 32, th)
    ref = F.linear(x_seq, ev.weight, ev.bias)
    outs = [ev(x_seq[i:i + 1]).detach() for i in range(x_seq.shape[0])]
    out = torch.cat(outs, dim=0)
    err = (out - ref).abs().max().item()
    print(f"  [A-pp] EventLinear + PerPixel exactness (theta0=0): max err {err:.2e}")
    assert err < 1e-5, f"PerPixel linear exactness failed: {err}"


def test_perpixel_reset_to_zero_restores_exactness():
    """After running with theta>0, forcing theta=0 must restore dense."""
    frames = _load_frames(60)
    th = PerPixelThreshold(theta0=0.0, warmup_steps=10)
    ev = EventConv2d(1, 8, 8, 5, th)
    for i in range(40):
        ev(frames[i:i + 1])
    # theta is now > 0 per-element
    mean_theta = float(th._theta.mean().item()) if isinstance(th._theta, torch.Tensor) else 0.0
    assert mean_theta > 0, "theta did not rise above 0"
    # force theta=0 (dense mode) and verify exactness against dense conv
    th._theta = torch.zeros_like(th._theta)
    th.theta = th._theta
    # reset_cache to re-seed out_prev from the next frame, then check one frame
    ev.reset_cache()
    out = ev(frames[40:41]).detach()
    ref = F.conv2d(frames[40:41], ev.weight, ev.bias, 5)
    err = (out - ref).abs().max().item()
    print(f"  [A-pp] theta->0 restores exactness: mean_theta_was={mean_theta:.2e} err {err:.2e}")
    assert err < 1e-5


def test_perpixel_event_rate_not_dead_not_dense():
    """Per-element scale-following theta should keep every layer's event rate
    clear of 0 and 1, including deeper layers (the structural fix for Bug 4)."""
    frames = _load_frames(2000)
    th = PerPixelThreshold(k=2.0, warmup_steps=50)
    ev = EventConv2d(1, 16, 8, 5, th)
    rates = []
    for i in range(frames.shape[0]):
        ev(frames[i:i + 1])
        rates.append(ev.last_event_rate)
    final = float(np.mean(rates[-200:]))
    overall = float(np.mean(rates))
    print(f"  [B-pp] PerPixel event rate: overall={overall:.4f} last200={final:.4f} "
          f"mean_theta={float(th._theta.mean()):.2e}")
    # Not dead (>>0) and not dense (<<1). k=3 on ~Gaussian deltas gives ~0.3%
    # tail, but Pong deltas are heavy-tailed so expect a few %.
    assert final > 1e-4, f"layer went dead: {final}"
    assert final < 0.5, f"layer too dense: {final}"


if __name__ == "__main__":
    print("Diagnostic 4-alt.6.1 (PerPixelThreshold exactness):")
    test_perpixel_exactness_conv()
    test_perpixel_exactness_linear()
    test_perpixel_reset_to_zero_restores_exactness()
    test_perpixel_event_rate_not_dead_not_dense()
    print("All PerPixel exactness checks passed.")
