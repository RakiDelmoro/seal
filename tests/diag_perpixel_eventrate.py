"""Diagnostic 4-alt.6.2 — per-pixel event rate distribution.

Premise of move A (per-pixel variance theta): ALE Pong is *bimodal* per-pixel
-- ~90% of pixels are static background (event rate ~0), the ball+paddles are
the entire task signal (event rate ~high). A single per-layer theta cannot
separate these two modes; it picks one threshold for a bimodal distribution,
which is exactly why deeper layers go dead (Bug 4) -- the global theta tracks
the background mode and the object pixels get gated away, or vice versa.

This diagnostic runs the CURRENT per-layer-homeostat EventConv2d (layer 0)
over the Stage-1 fixture and measures the per-pixel event-rate distribution.
It is pure instrumentation -- no behavior change, no new code paths in the
model. It produces:
  - results/diag_perpixel_eventrate_hist.png  (histogram of per-pixel rates)
  - results/diag_perpixel_eventrate_map.png   (spatial map of per-pixel rates)
  - printed summary stats + a bimodality assertion

PASS = distribution is bimodal: a large background mass near 0 AND a distinct
object mass well above 0. If unimodal, move A's premise is wrong and we stop.
"""
import os
import numpy as np
import torch

from model.event_layers import EventConv2d
from model.thresholds import HomeostaticThreshold
from tests.test_stage1_encoder import _load_frames

OUT = "results"


def main(n=2000, warmup=200):
    frames = _load_frames(n)  # [N,1,84,84]
    # Use the SAME homeostat config as Test B / the real encoder (layer 0).
    th = HomeostaticThreshold(target_lo=0.005, target_hi=0.03,
                              adapt_rate=2e-3, theta0=1e-4)
    ev = EventConv2d(1, 16, 8, 5, th)

    masks = []
    thetas = []
    for i in range(frames.shape[0]):
        ev(frames[i:i + 1])
        th.update(ev.last_event_rate)
        if i >= warmup and ev.last_mask is not None:
            # last_mask is [1,1,H,W] hard mask in input space
            masks.append(ev.last_mask[0, 0].numpy().astype(np.float32))
        thetas.append(th.theta)

    masks = np.stack(masks, 0)  # [T, H, W]
    per_pixel_rate = masks.mean(axis=0)  # [H, W] in [0,1]
    flat = per_pixel_rate.ravel()

    # ---- summary stats ----
    frac_near_zero = float((flat < 0.01).mean())   # background mode
    frac_object    = float((flat > 0.20).mean())   # object mode (fires >20% of frames)
    frac_middle    = 1.0 - frac_near_zero - frac_object
    mean_theta = float(np.mean(thetas[warmup:]))
    print(f"frames={n} warmup={warmup}  mean_theta(post-warmup)={mean_theta:.2e}")
    print(f"per-pixel event rate distribution:")
    print(f"  background (rate<0.01): {frac_near_zero*100:5.1f}%  {int(frac_near_zero*flat.size)} px")
    print(f"  object     (rate>0.20): {frac_object*100:5.1f}%  {int(frac_object*flat.size)} px")
    print(f"  middle     (0.01-0.20): {frac_middle*100:5.1f}%")
    print(f"  global mean event rate: {flat.mean():.4f}  (cf. Test B: ~0.058)")

    # ---- bimodality assertion (premise of move A) ----
    bimodal = frac_near_zero > 0.6 and frac_object > 0.02
    print(f"  BIMODAL: {bimodal}  (need bg>60% AND object>2%)")

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(flat, bins=50, range=(0, 1), color="steelblue")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("per-pixel event rate")
    ax[0].set_ylabel("count (log)")
    ax[0].set_title("per-pixel event rate distribution")
    ax[0].axvline(0.01, color="green", ls="--", lw=1, label="bg threshold")
    ax[0].axvline(0.20, color="red", ls="--", lw=1, label="object threshold")
    ax[0].legend(fontsize=8)
    im = ax[1].imshow(per_pixel_rate, cmap="hot", vmin=0, vmax=max(0.3, flat.max()))
    ax[1].set_title("spatial map of per-pixel event rate")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{OUT}/diag_perpixel_eventrate.png", dpi=90)
    plt.close(fig)
    print(f"saved: {OUT}/diag_perpixel_eventrate.png")

    assert bimodal, (
        f"per-pixel event rate is NOT bimodal (bg={frac_near_zero:.2f}, "
        f"object={frac_object:.2f}) -- move A premise FAILS, do not implement"
    )
    print("PASS: distribution is bimodal -- move A premise confirmed.")


if __name__ == "__main__":
    main()
