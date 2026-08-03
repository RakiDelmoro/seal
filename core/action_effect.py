"""Action effect matrix B: action → state change.

Part of the affine transition model:
  s_{t+1} = A·s_t + B·a_t + b

B ∈ ℝ^{N×A}. Each column is the state-space displacement caused by one action
(NOOP, UP, DOWN). The paddle's movement produces a consistent translating
signature in the state space, so this is the easiest component to learn.

Learning (normalized delta rule):
  ΔB = η_B · err · a_t^T / (‖s_t‖² + ε)

Normalizing by the state norm prevents B from absorbing large ball-motion
prediction errors. Weight decay keeps B bounded.
"""
from __future__ import annotations
import numpy as np

from config import (
    N_STATE, N_ACTIONS, B_INIT_STD, B_SEED, B_WEIGHT_DECAY, B_NORM_UPDATE,
)


class ActionEffect:
    """Learned action-effect matrix B."""

    def __init__(self, n_state: int = N_STATE, n_actions: int = N_ACTIONS,
                 init_std: float = B_INIT_STD, seed: int = B_SEED):
        rng = np.random.default_rng(seed)
        self.B = (rng.normal(0, init_std, (n_state, n_actions))
                  ).astype(np.float32)
        self.n_actions = n_actions

    def forward(self, action_onehot: np.ndarray) -> np.ndarray:
        """B @ a — the state change caused by this action."""
        return self.B @ action_onehot

    def update(self, err: np.ndarray, action_onehot: np.ndarray, eta: float,
               s_t: np.ndarray | None = None):
        """ΔB = η · err · a^T / (‖s_t‖² + ε) with optional weight decay."""
        scale = 1.0
        if B_NORM_UPDATE and s_t is not None:
            norm_sq = max(float(s_t @ s_t), 1.0)  # prevent zero-state explosion
            scale = 1.0 / norm_sq
        self.B += eta * scale * np.outer(err, action_onehot)
        if B_WEIGHT_DECAY > 0:
            self.B *= (1.0 - B_WEIGHT_DECAY)
