"""Inverse model D (the paper's W): state difference → action.

  u = D · (s* - s_t)     ← "I want to move toward s*; which action?"

D ∈ ℝ^{A×N}  (3×1000). This is the linear inverse model from the GCML paper
(eq 13). It maps a desired state change onto the action that would cause it,
providing the "sense of direction" on the cognitive map.

Learning rule (paper eq 13, Hebbian):
  ΔD = η_D · a_t · (s_{t+1} - s_t)^T

This learns the FORWARD map (action → state change) as an outer product, but
converges to the pseudo-inverse of the forward model V when V is linear and
the prediction is satisfied (proven in the paper). So applying D to a state
difference correctly inverts the forward map.

Note: this is the paper's exact rule. An earlier draft used a regression form
which is unnecessary — the Hebbian form is correct in the linear setting.
"""
from __future__ import annotations
import numpy as np

from config import N_STATE, N_ACTIONS, D_INIT_STD, D_SEED


class Direction:
    """Learned inverse model D (paper's W)."""

    def __init__(self, n_state: int = N_STATE, n_actions: int = N_ACTIONS,
                 init_std: float = D_INIT_STD, seed: int = D_SEED):
        rng = np.random.default_rng(seed)
        self.D = (rng.normal(0, init_std, (n_actions, n_state))
                  ).astype(np.float32)

    def forward(self, delta_s: np.ndarray) -> np.ndarray:
        """u = D @ (s* - s)  →  action utilities ∈ ℝ^A."""
        return self.D @ delta_s

    def update(self, action_onehot: np.ndarray, delta_s: np.ndarray, eta: float):
        """Error-driven inverse model (self-limiting).

        Predict the action from the observed state change, and descend the
        cross-entropy between the predicted action distribution and the action
        actually taken:

          ΔD = η · (a_onehot − softmax(D·Δs)) ⊗ Δs / (‖Δs‖² + ε)

        When D·Δs correctly predicts the action, softmax → a_onehot and the
        error → 0, so D stops growing — unlike the pure Hebbian rule, which
        adds ~η in a correlated direction every frame and diverges (observed:
        ‖D‖ 0.55 → 59 over 1620 episodes). This replaces the paper's
        `min(W, 1)` saturation (eq 21), which assumes nonneg grid-cell states
        and does not apply to Pong's signed Gabor features.
        """
        u = self.D @ delta_s               # (n_actions,) — predicted utilities
        u = u - u.max()                    # softmax (numerically stable)
        probs = np.exp(u)
        probs = probs / probs.sum()
        norm_sq = float(delta_s @ delta_s) + 1e-6
        grad = np.outer(action_onehot - probs, delta_s)   # (n_actions, n_state)
        self.D += eta * grad / norm_sq

    def decay(self, lam: float):
        """Oja-style weight decay: D *= (1 − λ). Bounds ‖D‖ as a safety net."""
        if lam > 0.0:
            self.D *= (1.0 - lam)
