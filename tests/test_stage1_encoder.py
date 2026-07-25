"""Stage 1 acceptance tests: event encoder only, no RL.

  Test A (exactness): theta=0 => EventConv2d & EventLinear running output
                       matches a dense layer applied to the full frame
                       sequence, allclose 1e-5. NON-NEGOTIABLE.
  Test B (sparsity):   per-element theta => event rate settles in a sparse
                       regime (not 0%, not 100%).
  Test C (heatmap):    save a heatmap figure of event masks; lit on
                       ball/paddles only (lit fraction < 20%).
  Test D (reconstruction): drift vs exact recompute is bounded (< 1e-4).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

from model.event_layers import EventConv2d, EventLinear
from model.thresholds import PerPixelThreshold
from model.metrics import flops_event_layers
from model.gvf import compute_cumulants

FRAMES_PATH = "results/stage1_pong_frames.npy"


def _load_frames(n=200):
    arr = np.load(FRAMES_PATH)[:n]  # [N,1,84,84]
    return torch.from_numpy(arr).float()


# ---------------------------------------------------------------------------
# Test A: exactness (theta=0 via large warmup so theta stays at 0)
# ---------------------------------------------------------------------------
def test_eventconv2d_exactness():
    frames = _load_frames(50)
    in_ch, out_ch, k, stride = 4, 8, 8, 5
    th = PerPixelThreshold(theta0=0.0, warmup_steps=10_000)
    ev = EventConv2d(in_ch, out_ch, k, stride, th)
    ref = F.conv2d(frames, ev.weight, ev.bias, stride)
    outs = [ev(frames[i:i + 1]).detach() for i in range(frames.shape[0])]
    out = torch.cat(outs, dim=0)
    assert torch.allclose(out, ref, atol=1e-5), \
        f"EventConv2d exactness failed: max err {(out-ref).abs().max().item()}"
    print(f"  [A] EventConv2d max abs err: {(out-ref).abs().max().item():.2e}")


def test_eventlinear_exactness():
    torch.manual_seed(0)
    x_seq = torch.randn(40, 1, 16)
    th = PerPixelThreshold(theta0=0.0, warmup_steps=10_000)
    ev = EventLinear(16, 32, th)
    ref = F.linear(x_seq, ev.weight, ev.bias)
    outs = [ev(x_seq[i:i + 1]).detach() for i in range(x_seq.shape[0])]
    out = torch.cat(outs, dim=0)
    assert torch.allclose(out, ref, atol=1e-5), \
        f"EventLinear exactness failed: max err {(out-ref).abs().max().item()}"
    print(f"  [A] EventLinear max abs err: {(out-ref).abs().max().item():.2e}")


def test_eventconv_reset_cache_exactness():
    """After reset_cache, the next frame must equal its dense conv (re-seed)."""
    frames = _load_frames(30)
    th = PerPixelThreshold(theta0=0.0, warmup_steps=10_000)
    ev = EventConv2d(4, 8, 8, 5, th)
    for i in range(15):
        ev(frames[i:i + 1])
    ev.reset_cache()
    out = ev(frames[15:16]).detach()
    ref = F.conv2d(frames[15:16], ev.weight, ev.bias, 5)
    assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Test B: sparsity (per-element theta keeps event rate sparse but not dead)
# ---------------------------------------------------------------------------
def test_eventconv_sparsity_settles():
    frames = _load_frames(2000)
    th = PerPixelThreshold(k=2.0, warmup_steps=50)
    ev = EventConv2d(4, 16, 8, 5, th)
    rates = []
    for i in range(frames.shape[0]):
        ev(frames[i:i + 1])
        rates.append(ev.last_event_rate)
    final = float(np.mean(rates[-200:]))
    print(f"  [B] final event rate (last 200): {final:.4f}")
    assert final > 1e-4, f"layer went dead: {final}"
    assert final < 0.20, f"event rate too high: {final}"


# ---------------------------------------------------------------------------
# Test C: heatmap (lit fraction small & concentrated)
# ---------------------------------------------------------------------------
def test_event_heatmap_concentrated():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = _load_frames(200)
    th = PerPixelThreshold(k=2.0, warmup_steps=50)
    ev = EventConv2d(4, 16, 8, 5, th)
    masks = []
    for i in range(frames.shape[0]):
        ev(frames[i:i + 1])
        if ev.last_mask is not None and i >= 50:
            masks.append(ev.last_mask.float().sum(dim=1)[0].numpy())
    masks = np.stack(masks, 0)
    mean_mask = masks.mean(0)
    lit_frac = float((mean_mask > 0.05).mean())
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(mean_mask, cmap="hot", vmin=0, vmax=max(0.1, mean_mask.max()))
    ax.set_title("event heatmap (avg over 150 frames)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig("results/stage1_event_heatmap.png", dpi=80)
    plt.close(fig)
    print(f"  [C] mean lit fraction: {lit_frac:.4f} (expect small/concentrated)")
    assert lit_frac < 0.20, f"events not concentrated: {lit_frac}"
    aux = compute_cumulants(ev.last_mask, 0.0).numpy()
    assert aux.shape == (4,) and np.isfinite(aux).all(), aux
    print(f"  [C] gvf cumulants (motion_density, pos_reward, neg_reward, motion_spread): {aux}")


# ---------------------------------------------------------------------------
# Test D: reconstruction drift vs exact recompute
# ---------------------------------------------------------------------------
def test_reconstruction_drift_bounded():
    frames = _load_frames(300)
    th = PerPixelThreshold(theta0=0.0, warmup_steps=10_000)
    ev = EventConv2d(4, 8, 8, 5, th)
    max_drift = 0.0
    for i in range(frames.shape[0]):
        out = ev(frames[i:i + 1]).detach()
        ref = F.conv2d(frames[i:i + 1], ev.weight, ev.bias, 5)
        drift = (out - ref).abs().max().item()
        max_drift = max(max_drift, drift)
        if (i + 1) % 50 == 0:
            ev.reset_cache()
    print(f"  [D] max reconstruction drift over 300 frames: {max_drift:.2e}")
    assert max_drift < 1e-4, f"drift too large: {max_drift}"


# ---------------------------------------------------------------------------
def _ensure_frames():
    if not os.path.exists(FRAMES_PATH):
        from env.envs import make_env, FrameRecorder
        env, spec = make_env("ALE/Pong-v5", seed=0)
        obs, _ = env.reset(seed=0)
        rec = FrameRecorder(n=3000, obs_shape=spec.obs_shape)
        while not rec.full:
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            rec.add(obs, bool(term or trunc))
            if term or trunc:
                obs, _ = env.reset()
        rec.save(FRAMES_PATH)
        env.close()
        print(f"  recorded fixture: {FRAMES_PATH}")


if __name__ == "__main__":
    _ensure_frames()
    print("Stage 1 tests:")
    test_eventconv2d_exactness()
    test_eventlinear_exactness()
    test_eventconv_reset_cache_exactness()
    test_eventconv_sparsity_settles()
    test_event_heatmap_concentrated()
    test_reconstruction_drift_bounded()
    print("Stage 1 tests passed.")
