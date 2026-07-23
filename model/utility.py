"""Utility tracker + dead-unit regeneration.

Two plasticity mechanisms:
  1. Per-parameter UTILITY GATE: a scalar running stat per parameter tensor
     (mean |delta*trace|). If it falls below tau_low, that parameter gets NO
     ObGD update this step.
  2. Per-UNIT REGENERATION (in agent._regenerate): at fixed intervals, the
     bottom utility quartile of TRUNK units that have been silent for a long
     time have their incoming weights reinitialized and outgoing weights
     zeroed (ReDo-style dead-unit resurrection).

Per-unit utility proxy: a running per-unit activation magnitude for the trunk
(256 units). Units with low activation utility AND long silence (dormant) are
regeneration candidates.
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
        """Per-parameter utility. Returns the gate list (bool per param)."""
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
        self.unit_utility += (a - self.unit_utility) / self.unit_count

    def dormant_units(self, silence_counts: np.ndarray, silence_steps: int) -> np.ndarray:
        """Indices of units silent > silence_steps AND in bottom utility quartile."""
        silent = silence_counts > silence_steps
        if not silent.any():
            return np.array([], dtype=int)
        util = self.unit_utility
        thr = np.quantile(util[silent], 0.75) if silent.sum() > 4 else util.max()
        return np.where(silent & (util <= thr))[0]
