"""Feasibility gate F: state → valid action mask.

  ĝ = F · s_t        F ∈ ℝ^{A×N}

In the paper (eq 16), G is learned via  ΔG = η_G · (g - G·s) · s^T  where g
is the true feasibility mask. In Pong, all 3 actions (NOOP, UP, DOWN) are
always feasible, so F is frozen to all-ones — learning the constant wastes
computation.

For environments with constrained actions, F can be unfrozen and learned with
the delta rule. Gate this behind a domain check.
"""
from __future__ import annotations
import numpy as np

from config import N_ACTIONS


class FeasibilityGate:
    """Frozen all-ones feasibility gate (Pong: all actions always feasible)."""

    def __init__(self, n_actions: int = N_ACTIONS, frozen: bool = True):
        self.n_actions = n_actions
        self.frozen = frozen
        self.ones = np.ones(n_actions, dtype=np.float32)

    def forward(self, s: np.ndarray | None = None) -> np.ndarray:
        """Return the feasibility mask. For Pong, always all-ones."""
        return self.ones

    def update(self, *args, **kwargs):
        """No-op when frozen."""
        pass
