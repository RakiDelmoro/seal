"""Per-element event threshold (move A).

Each event layer's threshold is per-element (one theta per pixel/unit), not a
single per-layer scalar. The formula is scale-following:

    adelta[e] = beta * adelta[e] + (1-beta) * |delta[e]|   (EWMA of |delta|)
    theta[e]  = clip( k * adelta[e], floor, ceil )

An element fires when |delta[e]| > k * (its own typical |delta|). This is:
  - per-element from the first step (adelta differs across elements), so it
    does NOT degenerate to a per-layer scalar;
  - robust to heavy tails (tracks the mean |delta|, not the variance, so
    outliers do not dominate);
  - self-calibrating: static elements (delta ~ 0) get theta -> floor and are
    ARMED (fire the instant real signal arrives); active elements get theta
    proportional to their own motion scale and fire at a stable tail rate;
  - cannot overshoot to dead: theta tracks |delta| directly, so it falls with
    the signal and rises with it. Structurally prevents dead layers (Bug 4).

Cold start: theta=0 during `warmup_steps` (dense / exact) while adelta primes,
preserving the exactness invariant (theta=0 => output == dense conv).

`observe(delta, mask)` is called from inside each event layer's forward pass.
"""
from __future__ import annotations
import torch


class PerPixelThreshold:
    def __init__(self, k: float = 2.0, beta: float = 0.99,
                 floor: float = 1e-6, ceil: float = 1.0,
                 theta0: float = 0.0, warmup_steps: int = 50):
        self.k = float(k)
        self.beta = float(beta)
        self.floor = float(floor)
        self.ceil = float(ceil)
        self.warmup_steps = int(warmup_steps)
        self._adelta = None     # EWMA of |delta|, per-element
        self._theta = torch.tensor(float(theta0))
        self.theta = self._theta
        self._steps = 0

    def observe(self, delta: torch.Tensor, mask: torch.Tensor):
        """Update per-element EWMA of |delta| and recompute theta.

        delta: [1, C, H, W] or [1, D] (the layer's input delta, detached).
        During warmup, theta stays at 0 (dense / exact) while adelta primes.
        """
        with torch.no_grad():
            d = delta.detach()[0]            # [C,H,W] or [D]
            ad = d.abs()
            if self._adelta is None or self._adelta.shape != ad.shape:
                self._adelta = ad.clone()
                self._theta = torch.zeros_like(ad)
            else:
                self._adelta.mul_(self.beta).add_((1.0 - self.beta) * ad)
            self._steps += 1
            if self._steps <= self.warmup_steps:
                self._theta = torch.zeros_like(ad)   # dense / exact
            else:
                self._theta = (self.k * self._adelta).clamp(self.floor, self.ceil)
            self.theta = self._theta

    @property
    def value(self):
        return self._theta

    def reset(self):
        """Reset theta to floor (cold start). adelta persists across episode
        boundaries (the input distribution does not change)."""
        if isinstance(self._theta, torch.Tensor) and self._theta.numel() > 1:
            self._theta = torch.full_like(self._theta, self.floor)
        self.theta = self._theta
