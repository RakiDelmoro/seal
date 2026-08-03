"""Policy π(s) — stochastic softmax policy learned by imitation + streaming AC.

π(a|s) = exp(h_a) / Σ_b exp(h_b)    where    h = θ · s

Updates (both local delta rules — no backpropagation):
  1. Imitation: when imagination picks a good action, move π toward it.
       Δθ = η_imit · (target_action - softmax(h)) ⊗ s / (‖s‖² + ε)

  2. Streaming actor-critic: every step, reinforce the taken action by the
     TD(λ) error δ from the critic V (the streaming advantage).
       Δθ = η_ac · δ · (target_action - softmax(h)) ⊗ s / (‖s‖² + ε)
"""
from __future__ import annotations
import numpy as np

from config import N_STATE, N_ACTIONS, PI_INIT_STD, PI_WEIGHT_DECAY, PI_SEED


class Policy:
    """Linear softmax policy π(s)."""

    def __init__(self, n_state: int = N_STATE, n_actions: int = N_ACTIONS,
                 init_std: float = PI_INIT_STD, seed: int = PI_SEED):
        rng = np.random.default_rng(seed)
        self.theta = (rng.normal(0, init_std, (n_actions, n_state))
                      ).astype(np.float32)
        self.n_actions = n_actions

    def _logits(self, s: np.ndarray) -> np.ndarray:
        return self.theta @ s

    def forward(self, s: np.ndarray) -> np.ndarray:
        """Softmax action probabilities."""
        logits = self._logits(s)
        logits -= logits.max()  # numerically stable
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    def predict_action(self, s: np.ndarray) -> int:
        """Argmax action (greedy use of the policy)."""
        return int(np.argmax(self._logits(s)))

    def confidence(self, s: np.ndarray) -> float:
        """Max softmax probability — how confident is the policy?"""
        return float(self.forward(s).max())

    def update(self, s: np.ndarray, action: int, eta: float, scale: float = 1.0):
        """Softmax cross-entropy update toward the chosen action.

        Args:
            s: state vector.
            action: action index to reinforce.
            eta: learning rate.
            scale: scalar multiplier (e.g., TD error δ for actor-critic).
        """
        target = np.zeros(self.n_actions, dtype=np.float32)
        target[action] = 1.0
        probs = self.forward(s)
        norm_sq = max(float(s @ s), 1.0)  # prevent zero-state explosion
        grad = np.outer(target - probs, s) / norm_sq
        self.theta += eta * scale * grad
        if PI_WEIGHT_DECAY > 0:
            self.theta *= (1.0 - PI_WEIGHT_DECAY)

    def update_imitation(self, s: np.ndarray, action: int, eta: float):
        """Imitate a provided action (from imagination or another teacher)."""
        self.update(s, action, eta, scale=1.0)
