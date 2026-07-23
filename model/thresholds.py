"""Homeostatic event threshold (spec §2.3).

One scalar theta per event layer. Updated every step from the layer's event
rate to keep activity inside a target band [target_lo, target_hi].

  rate < lo -> theta *= (1 - adapt_rate)   (too quiet: lower the bar)
  rate > hi -> theta *= (1 + adapt_rate)   (too loud: raise the bar)

Per-channel theta is a v2 ablation (not implemented).

DEAD-LAYER RECOVERY (fix for the overshoot failure mode observed in Stage 3):
  The spec's adapt_rate=1e-3 was too slow: deeper layers that overshot to
  theta >> delta went silent (rate=0) and never recovered within the run. We
  apply two spec-faithful fixes (the homeostat's STATED purpose is to keep
  rate in the band; dead layers fail that purpose, so fixing them is alignment,
  not a deviation):
    1. adapt_rate defaults to 1e-2 (10x faster) -> recovery from overshoot in
       ~700 frames, not ~7000.
    2. Dead-layer safety net: if rate==0 for `dead_steps` consecutive steps,
       cut theta in half immediately. Breaks the overshoot deadlock without
       affecting healthy layers (which never hit 0% for long).
"""
from __future__ import annotations
import torch


class HomeostaticThreshold:
    def __init__(self, target_lo=0.03, target_hi=0.10,
                 adapt_rate=1e-2, theta0=1e-4,
                 dead_steps=100, dead_reset_mult=0.3):
        self.target_lo = float(target_lo)
        self.target_hi = float(target_hi)
        self.adapt_rate = float(adapt_rate)
        self.theta = float(theta0)
        # Dead-layer safety net (see module docstring). Fires every `dead_steps`
        # of silence, cutting theta by `dead_reset_mult` each time until the
        # layer wakes. 0.3 (not 0.5) so a 1000x overshoot recovers in ~7 fires
        # (~700 steps) instead of ~10 fires.
        self.dead_steps = int(dead_steps)
        self.dead_reset_mult = float(dead_reset_mult)
        self._silent_count = 0

    def update(self, event_rate: float) -> float:
        """Adapt theta from the latest per-layer event rate. Returns new theta."""
        r = float(event_rate)
        if r <= 0.0:
            self._silent_count += 1
            if self._silent_count >= self.dead_steps:
                # layer is dead -- force theta down hard to wake it up.
                # Fires repeatedly (every dead_steps) until rate > 0.
                self.theta *= self.dead_reset_mult
                self._silent_count = 0  # reset counter so it fires again if still dead
                return self.theta
        else:
            self._silent_count = 0
        if r < self.target_lo:
            self.theta *= (1.0 - self.adapt_rate)
        elif r > self.target_hi:
            self.theta *= (1.0 + self.adapt_rate)
        return self.theta

    @property
    def value(self) -> float:
        return self.theta

    # --- uniform threshold protocol (see PerPixelThreshold.observe) ---
    def observe(self, delta: torch.Tensor, mask: torch.Tensor):
        """Homeostat is driven by the scalar event rate, not per-element stats.
        The agent calls `update(event_rate)` after forward; this is a no-op here."""
        return


class PerPixelThreshold:
    """Per-element adaptive theta via a per-element homeostat (move A).

    Motivation (validated in tests/diag_perpixel_eventrate.py + the per-element
    std-spread diagnostic): ALE Pong per-pixel event rates are bimodal (72%
    background near 0, 8% object >20%) and per-element delta std varies ~128x
    in layer 0 and 3-6x in deeper layers. A single per-layer theta cannot
    track this spread -- it picks one threshold for a multi-scale distribution,
    overshoots on the small-scale elements, and drives the layer dead (Bug 4).

    A variance-based per-element theta (theta = k*sqrt(EWMA(delta^2))) was
    tried first and REJECTED: in deeper layers / the trunk, the delta
    distribution is heavy-tailed, so the variance is dominated by the active
    minority and theta is too high for the static majority -> the layer still
    goes dead (the same failure mode, one level down).

    A per-element HOMEOSTAT (each element adapts theta to hit a target firing
    rate) was tried second and REJECTED: after warmup theta starts uniform
    across elements and the multiplicative adapt rule keeps it uniform, so it
    degenerates to the per-layer homeostat (all elements fire identically under
    a uniform theta; differentiation never emerges).

    The fix that generalizes is a per-element SCALE-FOLLOWING threshold:

        adelta[e] = beta * adelta[e] + (1-beta) * |delta[e]|   (EWMA of |delta|)
        theta[e]  = clip( k * adelta[e], floor, ceil )

    An element fires when |delta[e]| > k * (its own typical |delta|). This is:
      - per-element from the first step (adelta differs across elements), so
        it does NOT degenerate to a per-layer scalar;
      - robust to heavy tails (tracks the mean |delta|, not the variance, so
        outliers do not dominate);
      - self-calibrating: static elements (delta ~ 0) get theta -> floor and
        are ARMED (fire the instant real signal arrives); active elements get
        theta proportional to their own motion scale and fire at a stable tail
        rate (~P(|x|>k*E|x|), e.g. ~11% at k=2, ~2% at k=3 for Gaussian).
      - cannot overshoot to dead: theta tracks |delta| directly, so it falls
        with the signal and rises with it. The structural fix for Bug 4.

    The cost: static elements sit at the floor and fire on any nonzero delta
    (noise in noisy envs; in deterministic ALE Pong, only on real signal). The
    companion move C (event-gated traces) ensures those low-signal events do
    not inflate credit assignment. A and C are designed to work together.

    `observe(delta, mask)` is called from inside the layer forward (sees every
    delta). The legacy `update(event_rate)` scalar call from the agent is a
    no-op for this threshold (its state is updated in forward).
    """
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

    # legacy scalar interface (agent calls this; no-op for per-pixel)
    def update(self, event_rate: float) -> float:
        return float(self._theta.mean().item()) if isinstance(self._theta, torch.Tensor) else float(self._theta)

    @property
    def value(self):
        return self._theta

    def reset(self):
        """Reset theta to floor (cold start). adelta persists across episode
        boundaries (the input distribution does not change)."""
        if isinstance(self._theta, torch.Tensor) and self._theta.numel() > 1:
            self._theta = torch.full_like(self._theta, self.floor)
        self.theta = self._theta
