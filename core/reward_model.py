"""Reward model r̂(s) — linear reward predictor learned by a local delta rule.

r̂(s) = w_R · s

Pong's ±1 reward arrives only on the frames where a point was actually
scored. Imagination needs to answer "would this *predicted* future state
bring reward?" — so we learn a simple reward predictor alongside everything
else, using only the observed (state, reward) pairs:

    err   = r − w_R · s              (prediction error on the real transition)
    Δw_R  = η · err · s / ‖s‖²       (normalized LMS — same local rule family
                                      as A, B, D: no backprop, fully online)

The prediction r̂ feeds the imagined-TD updates in imagination/imagined_td.py.
The reward is associated with the *arrival* state s_{t+1} (that is the state
the agent sees when the reward comes in), so imagined rollouts query r̂ on
their predicted next states too.
"""
from __future__ import annotations
import numpy as np

from config import N_STATE, R_INIT_STD, R_WEIGHT_DECAY, R_SEED


class RewardModel:
    """Linear reward predictor r̂(s) = w_R · s."""

    def __init__(self, n_state: int = N_STATE, init_std: float = R_INIT_STD,
                 seed: int = R_SEED):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, init_std, n_state).astype(np.float32)

    def forward(self, s: np.ndarray) -> float:
        """Predicted reward for (arriving at) state s."""
        return float(self.w @ s)

    def update(self, s: np.ndarray, reward: float, eta: float) -> float:
        """One normalized-LMS update from a real (arrival-state, reward) pair.

        Returns the prediction error *before* the update (for logging).
        """
        pred = self.forward(s)
        err = reward - pred
        norm_sq = max(float(s @ s), 1.0)   # normalized LMS (like the rest of SEAL)
        self.w += eta * err * s / norm_sq
        if R_WEIGHT_DECAY > 0:
            self.w *= (1.0 - R_WEIGHT_DECAY)
        return float(err)
