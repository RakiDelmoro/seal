"""Trajectory evaluator — score imagined rollouts GEOMETRICALLY.

  score = −‖ŝ_H − s*‖₁  −  danger_penalty · 𝟙[any ŝ_t on our side]

This is the paper's form (Lin et al. 2026, eq 28: `J = −‖s_{t+H} − o*‖ + β·χ`),
adapted to Pong: the goal s* is a recently-observed state with the ball on the
opponent's side (see imagination/geometric_goal.py), and "danger" means the
imagined trajectory leaves the ball on our side (we are about to be scored on).

Geometric scoring cannot diverge and is goal-directed from frame 1 — there is
no learned value function on the planning path.
"""
from __future__ import annotations
import numpy as np

from config import DANGER_PENALTY
from imagination.geometric_goal import GeometricGoal

# Module-level scorer (stateless — safe to share). Lazily built so config
# changes are picked up.
_geometric: GeometricGoal | None = None


def _get_geometric() -> GeometricGoal:
    global _geometric
    if _geometric is None:
        _geometric = GeometricGoal()
    return _geometric


def evaluate_trajectory(traj: dict,
                        danger_penalty: float = DANGER_PENALTY,
                        s_star: np.ndarray | None = None) -> float:
    """Score a single trajectory geometrically: −‖ŝ_H − s*‖₁ + danger penalty.

    `s_star` is the geometric goal (a recently-observed state with the ball on
    the opponent's side). Returns 0.0 if there is no goal or no trajectory.
    """
    states = traj.get("states", [])
    geo = _get_geometric()
    if s_star is None or len(states) == 0:
        return 0.0
    return geo.score_trajectory(states, s_star, danger_penalty)


def evaluate_trajectories(trajectories: list[dict],
                          danger_penalty: float = DANGER_PENALTY,
                          s_star: np.ndarray | None = None
                          ) -> tuple[list[float], int]:
    """Score all trajectories geometrically.

    Returns:
        (scores, best_idx) — list of scores and the index of the best one.
    """
    scores = [evaluate_trajectory(t, danger_penalty, s_star)
              for t in trajectories]
    best_idx = int(np.argmax(scores))
    return scores, best_idx
