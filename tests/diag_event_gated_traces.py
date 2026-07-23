"""Diagnostic 4-alt.6.3 — event-gated trace accumulation (move C).

Premise of move C: eligibility traces accumulate the FULL gradient every step
(incl. the straight-through estimator's non-zero-everywhere contribution), so
z_sum scales with TOTAL param count, not active-param count -> explodes ->
ObGD step size collapses (Bug 5). Gating the trace increment by the hard event
mask should make z_sum scale with EVENT ACTIVITY instead, bounding it without
a hard clip.

This diagnostic runs the real encoder + a synthetic-loss backward over an
episode and compares two trace accumulators:
  - UNGATED:   trace += grad                 (current behaviour, incl. ST)
  - GATED:     trace += grad * mask_broadcast (hard-event-mask zeroing)
for the trunk EventLinear layer (the dominant trace contributor: ~33k params).
We report the z_sum trajectory of each over ~1500 steps.

PASS = gated z_sum stays bounded (tracks event activity, not param count)
while ungated z_sum grows substantially larger. If gated ~ ungated, move C's
premise is wrong and we do not implement it.

NOTE: this is a diagnostic. We do NOT touch the real optimizer here.
"""
import numpy as np
import torch
import torch.nn.functional as F

from config import config_from_preset
from model.event_layers import EventConv2d, EventLinear
from model.thresholds import PerPixelThreshold, HomeostaticThreshold
from tests.test_stage1_encoder import _load_frames


def _build_encoder(in_ch=1, use_perpixel=True):
    cfg = config_from_preset("ALE/Pong-v5")
    ths = ([PerPixelThreshold(k=2.0, warmup_steps=50) if use_perpixel
            else HomeostaticThreshold(target_lo=0.005, target_hi=0.03,
                                      adapt_rate=1e-2, theta0=1e-4)
            for _ in range(3)])
    evs = [EventConv2d(1, 16, 8, 5, ths[0]),
           EventConv2d(16, 32, 4, 3, ths[1]),
           EventConv2d(32, 32, 3, 2, ths[2])]
    fc_th = PerPixelThreshold(k=2.0, warmup_steps=50) if use_perpixel \
        else HomeostaticThreshold(target_lo=0.005, target_hi=0.03,
                                  adapt_rate=1e-2, theta0=1e-4)
    fc = EventLinear(32 * 2 * 2, 256, fc_th)
    head = torch.nn.Linear(256, 6)
    return evs, fc, head, ths + [fc_th]


def main(n=1500, lam=0.8, gamma=0.99):
    frames = _load_frames(n + 50)
    evs, fc, head, ths = _build_encoder(use_perpixel=True)
    ln = lambda x: F.layer_norm(x, x.shape[1:])
    lk = lambda x: F.leaky_relu(x)

    # two trace accumulators for the fc weight [256, 128]
    z_ungated = torch.zeros_like(fc.weight)
    z_gated = torch.zeros_like(fc.weight)
    history = []
    for i in range(n):
        x = frames[i:i + 1]
        # forward through conv stack (no grad on convs for this diagnostic;
        # we only need fc grad wrt a synthetic loss)
        with torch.no_grad():
            h = lk(ln(evs[0](x)))
            for L in [1, 2]:
                h = lk(ln(evs[L](h)))
                ths[L].update(evs[L].last_event_rate)
            ths[0].update(evs[0].last_event_rate)
            h_flat = h.flatten(1)
            # hard event mask on the fc INPUT (trunk features delta > theta)
            # fc.x_prev holds previous input; delta = h_flat - fc.x_prev
            if bool(fc._initialized[0]):
                fc_delta = (h_flat - fc.x_prev).abs()
                fc_mask = (fc_delta > fc.threshold.theta).float()  # [1,128]
            else:
                fc_mask = torch.ones_like(h_flat)
        # fc forward WITH grad (so we get fc.weight grad)
        fc_out = fc(h_flat)
        feats = lk(ln(fc_out))
        logits = head(feats)
        # synthetic loss (random target) -> nonzero grad on fc.weight
        target = torch.tensor([float(i % 6)])
        loss = F.cross_entropy(logits, target.long())
        grads = torch.autograd.grad(loss, [fc.weight], retain_graph=False)
        g = grads[0].detach()  # [256, 128]
        # trace accumulation
        z_ungated.mul_(lam * gamma).add_(g)
        # gated: zero the trace increment for inactive input elements
        # fc_mask is [1,128]; broadcast over the 256 output rows
        g_gated = g * fc_mask  # broadcasts [256,128]*[1,128]
        z_gated.mul_(lam * gamma).add_(g_gated)
        if i % 100 == 0 or i == n - 1:
            zu = float(z_ungated.abs().sum().item())
            zg = float(z_gated.abs().sum().item())
            er = float(fc_mask.mean().item())
            history.append((i, zu, zg, er))
    print(f"steps={n} lam={lam} gamma={gamma}  fc.weight shape={tuple(fc.weight.shape)}")
    print(f"{'step':>6} {'z_ungated':>12} {'z_gated':>12} {'ratio':>7} {'fc_event_rate':>14}")
    for (i, zu, zg, er) in history:
        r = zu / (zg + 1e-12)
        print(f"{i:>6} {zu:>12.2f} {zg:>12.2f} {r:>7.2f}x {er:>14.4f}")
    final_zu, final_zg = history[-1][1], history[-1][2]
    ratio = final_zu / (final_zg + 1e-12)
    print(f"\nfinal ratio ungated/gated = {ratio:.2f}x")
    # PASS: gated z_sum is substantially smaller (traces scale with event
    # activity, not param count). >3x means gating removes most of the mass.
    assert ratio > 3.0, (
        f"gating does not bound z_sum (ratio {ratio:.2f}x) -- move C premise "
        f"FAILS, do not implement")
    print(f"PASS: gated z_sum is {ratio:.1f}x smaller -- move C premise confirmed.")


if __name__ == "__main__":
    main()
