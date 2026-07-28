"""Plasticity: dormant spiking-unit regeneration (ReDo-style).

A spiking neuron is "dormant" if it has not fired for `dormant_silence_ms`
milliseconds. Every `regen_every` env steps, the bottom `regen_frac` of
dormant units are reinitialized: incoming weights (Win, Wrec rows) get fresh
random values, outgoing weights (Wrec columns, readout rows) are zeroed.

This combats the loss-of-plasticity problem in online learning and is
especially important for spiking nets, where dead neurons never contribute
eligibility traces and thus never recover on their own.
"""
from __future__ import annotations
import torch
import numpy as np


class UtilityTracker:
    """Tracks spike-silence per LSNN neuron and regenerates dormant units.

    Args:
        n_total: LSNN population size
        regen_every: env steps between regeneration sweeps
        dormant_silence_ms: ms of silence to count a neuron dormant
        regen_frac: fraction of dormant units to regenerate each sweep
        win_scale, wrec_scale: init scales for regenerated incoming weights
    """
    def __init__(self, n_total: int, regen_every: int = 25_000,
                 dormant_silence_ms: float = 10_000.0, regen_frac: float = 0.01,
                 win_scale: float = 0.02, wrec_scale: float = 0.01):
        self.n_total = n_total
        self.regen_every = int(regen_every)
        self.dormant_silence_ms = float(dormant_silence_ms)
        self.regen_frac = float(regen_frac)
        self.win_scale = float(win_scale)
        self.wrec_scale = float(wrec_scale)
        self.since_spike_ms = np.zeros(n_total, dtype=np.float64)
        self.last_n_regen = 0

    def observe(self, z: torch.Tensor, sim_ms: int):
        """Update silence counters from a step's spike vector."""
        fired = (z.numpy() > 0.5)
        self.since_spike_ms[fired] = 0.0
        self.since_spike_ms[~fired] += float(sim_ms)

    def dormant_units(self) -> np.ndarray:
        """Indices of neurons silent longer than the threshold."""
        return np.where(self.since_spike_ms >= self.dormant_silence_ms)[0]

    def maybe_regen(self, step: int, core, readout) -> int:
        """If due, regenerate a fraction of dormant units. Returns n regenerated."""
        if step == 0 or step % self.regen_every != 0:
            return 0
        dormant = self.dormant_units()
        if len(dormant) == 0:
            return 0
        n_regen = max(1, int(self.regen_frac * self.n_total))
        n_regen = min(n_regen, len(dormant))
        # pick the longest-silent dormant units
        order = np.argsort(-self.since_spike_ms[dormant])[:n_regen]
        dead = torch.from_numpy(dormant[order]).long()
        with torch.no_grad():
            # incoming Win rows: reinit
            for j in dead:
                jv = int(j)
                core.Win.data[jv] = torch.empty_like(core.Win.data[jv]).uniform_(
                    -self.win_scale, self.win_scale)
            # incoming Wrec rows: reinit
            for j in dead:
                jv = int(j)
                core.Wrec.data[jv] = torch.empty_like(core.Wrec.data[jv]).uniform_(
                    -self.wrec_scale, self.wrec_scale)
            core.Wrec.data.fill_diagonal_(0.0)
            # outgoing: zero readout columns (actor + critic) + Wrec columns
            readout.Wout_actor.data[:, dead] = 0.0
            readout.Wout_critic.data[:, dead] = 0.0
            core.Wrec.data[:, dead] = 0.0
            core.Wrec.data.fill_diagonal_(0.0)
        self.since_spike_ms[dead.numpy()] = 0.0
        self.last_n_regen = n_regen
        return n_regen

    def dormant_frac(self) -> float:
        return float((self.since_spike_ms >= self.dormant_silence_ms).mean())
