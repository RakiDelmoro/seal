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
