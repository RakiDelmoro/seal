"""Trajectory evaluator — score imagined rollouts by value + geometry.

score = α · Σ_t V(ŝ_t)  +  (1−α) · (−‖ŝ_H − s*‖₁)  −  danger_penalty · 𝟙[danger]

V is the learned value function (critic). α is the blend between value-based
and geometric scoring. During cold start α ≈ 0 because V is random; as V
gains signal, α grows toward IMAGINATION_ALPHA_V_MAX.

The geometric term is the paper's form (Lin et al. 2026, eq 28), adapted to Pong.
"""
from __future__ import annotations
import numpy as np

from config import (
    DANGER_PENALTY, IMAGINATION_ALPHA_V, IMAGINATION_ALPHA_V_MAX,
    IMAGINATION_ALPHA_V_GROWTH,
)
from imagination.geometric_goal import GeometricGoal

# Module-level scorer (stateless — safe to share). Lazily built so config
# changes are picked up.
_geometric: GeometricGoal | None = None


def _get_geometric() -> GeometricGoal:
    global _geometric
    if _geometric is None:
        _geometric = GeometricGoal()
    return _geometric


class ValueScorer:
    """Stateful scorer that blends learned value V with geometric goal scoring."""

    def __init__(self, alpha: float = IMAGINATION_ALPHA_V,
                 alpha_max: float = IMAGINATION_ALPHA_V_MAX,
                 alpha_growth: float = IMAGINATION_ALPHA_V_GROWTH):
        self.alpha = alpha
        self.alpha_max = alpha_max
        self.alpha_growth = alpha_growth
        self.geo = _get_geometric()

    def _value_signal(self, value, states: list) -> float:
        """Average predicted value along the imagined trajectory."""
        if value is None or len(states) == 0:
            return 0.0
        return float(np.mean([value.forward(s) for s in states]))

    def _geometric_score(self, states: list, s_star: np.ndarray | None,
                         danger_penalty: float) -> float:
        if s_star is None or len(states) == 0:
            return 0.0
        return self.geo.score_trajectory(states, s_star, danger_penalty)

    def score_trajectory(self, traj: dict, value=None,
                         s_star: np.ndarray | None = None,
                         danger_penalty: float = DANGER_PENALTY) -> float:
        """Score a single trajectory by blended value + geometry."""
        states = traj.get("states", [])
        if len(states) == 0:
            return 0.0
        v_score = self._value_signal(value, states)
        g_score = self._geometric_score(states, s_star, danger_penalty)
        return self.alpha * v_score + (1.0 - self.alpha) * g_score

    def score_trajectories(self, trajectories: list[dict], value=None,
                           s_star: np.ndarray | None = None,
                           danger_penalty: float = DANGER_PENALTY
                           ) -> tuple[list[float], int]:
        """Score all trajectories; return (scores, best_idx)."""
        scores = [self.score_trajectory(t, value, s_star, danger_penalty)
                  for t in trajectories]
        best_idx = int(np.argmax(scores))
        return scores, best_idx

    def grow_alpha(self, n_frames: int = 1):
        """Gradually increase reliance on the value function as it learns."""
        self.alpha = min(self.alpha_max,
                         self.alpha + self.alpha_growth * n_frames)


def evaluate_trajectory(traj: dict,
                        danger_penalty: float = DANGER_PENALTY,
                        s_star: np.ndarray | None = None,
                        value=None) -> float:
    """Backward-compatible single-trajectory scorer (geometry only)."""
    scorer = ValueScorer(alpha=0.0)
    return scorer.score_trajectory(traj, value=value, s_star=s_star,
                                   danger_penalty=danger_penalty)


def evaluate_trajectories(trajectories: list[dict],
                          danger_penalty: float = DANGER_PENALTY,
                          s_star: np.ndarray | None = None,
                          value=None
                          ) -> tuple[list[float], int]:
    """Backward-compatible batch scorer (geometry only)."""
    scorer = ValueScorer(alpha=0.0)
    return scorer.score_trajectories(trajectories, value=value, s_star=s_star,
                                     danger_penalty=danger_penalty)
