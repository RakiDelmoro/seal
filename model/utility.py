"""Utility tracker + dead-unit regeneration (spec §2.7).

  utility[p] = decay * utility[p] + (1-decay) * (td_error * trace[p]).abs().mean()
  update_gate = utility[p] > tau_low        # weights below gate: no update
  regenerate_dead_units(every=25_000):      # bottom 1% utility: reinit incoming,
                                            #  zero outgoing

Two distinct plasticity mechanisms:
  1. Per-parameter UTILITY GATE: a scalar running stat per parameter tensor
     (mean |delta*trace|). If it falls below tau_low, that parameter gets NO
     ObGD update this step (spec §2.8: `if utility_gate[p]: obgd_update(...)`).
     This is a coarse "this layer is currently useless" gate.
  2. Per-UNIT REGENERATION: at fixed intervals, the bottom `regen_frac` of
     TRUNK units by a per-unit utility proxy have their incoming weights
     reinitialized and outgoing weights zeroed. This is the dead-unit
     resurrection mechanism (spec §2.7, §2.4 LeakyReLU mitigation).

Per-unit utility proxy: we keep a running per-unit activation magnitude for
the trunk (256 units). Units with low activation utility AND long silence
(dormant) are regeneration candidates.
"""
from __future__ import annotations
import numpy as np
import torch


class UtilityTracker:
    """Per-parameter scalar utility + gate; per-unit trunk utility."""
    def __init__(self, params, decay: float = 0.9999, tau_low: float = 1e-6,
                 n_trunk_units: int = 256):
        self.params = list(params)
        self.decay = float(decay)
        self.tau_low = float(tau_low)
        # scalar utility per parameter
        self.utility = [torch.zeros(1, dtype=torch.float32) for _ in self.params]
        # per-unit trunk utility (running mean |activation|)
        self.unit_utility = np.zeros(n_trunk_units, dtype=np.float64)
        self.unit_count = 0

    def update_param_utility(self, td_error: float, traces):
        """Spec §2.7 per-parameter utility. Returns the gate list (bool per param)."""
        gates = []
        with torch.no_grad():
            for i, t in enumerate(traces):
                inst = (float(td_error) * t.detach()).abs().mean()
                self.utility[i].mul_(self.decay).add_((1.0 - self.decay) * inst)
                gates.append(bool(self.utility[i].item() > self.tau_low))
        return gates

    def update_unit_utility(self, trunk_acts: np.ndarray):
        """Running mean |activation| per trunk unit (batch-1)."""
        a = np.abs(trunk_acts.astype(np.float64))
        self.unit_count += 1
        # incremental mean
        self.unit_utility += (a - self.unit_utility) / self.unit_count

    def dormant_units(self, silence_counts: np.ndarray, silence_steps: int) -> np.ndarray:
        """Indices of units silent > silence_steps AND in bottom utility quartile."""
        silent = silence_counts > silence_steps
        if not silent.any():
            return np.array([], dtype=int)
        # bottom 25% utility among silent units
        util = self.unit_utility
        thr = np.quantile(util[silent], 0.75) if silent.sum() > 4 else util.max()
        return np.where(silent & (util <= thr))[0]


def regenerate_dead_units(module, trunk_incoming: torch.Tensor,
                          trunk_outgoing: torch.Tensor, dead_idx: np.ndarray):
    """Reinit incoming weights to dead units; zero outgoing weights from them.

  trunk_incoming: weight tensor [out_units, in_features] (e.g. EventLinear
  trunk_outgoing: weight tensor [out_features, out_units] (e.g. head Linear
                  reading the trunk hidden).
  dead_idx: unit indices to regenerate.

  Reinitialize the rows (incoming) of dead units and zero their columns
  (outgoing). Returns the count regenerated.
    """
    if len(dead_idx) == 0:
        return 0
    with torch.no_grad():
        fan_in = trunk_incoming.shape[1]
        bound = (1.0 / fan_in) ** 0.5
        for idx in dead_idx:
            trunk_incoming[idx] = torch.empty_like(trunk_incoming[idx]).uniform_(-bound, bound)
            trunk_outgoing[:, idx] = 0.0
    return len(dead_idx)
