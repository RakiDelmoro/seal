"""Success-rate tracker for adaptive exploration ε.

  success_rate = (points_scored + 1) / (points_scored + points_lost + 2)
                 (Laplace-smoothed, ∈ (0, 1), starts at 0.5)

  ε = max(ε_floor, ε_base · (1 - success_rate_ema))

A model that's losing explores more; a model that's winning trusts its
imagination. ε never drops below the floor (always keep seeing novel states
for online learning).

Used in Phase 2 (imaginative play); built here so the tracker accumulates
from the start.
"""
from __future__ import annotations
from collections import deque
from config import EPSILON_BASE, EPSILON_FLOOR, SUCCESS_EMA_EPISODES


class SuccessTracker:
    """Tracks Laplace-smoothed success rate over recent episodes."""

    def __init__(self, window: int = SUCCESS_EMA_EPISODES,
                 eps_base: float = EPSILON_BASE,
                 eps_floor: float = EPSILON_FLOOR):
        self.window = window
        self.eps_base = eps_base
        self.eps_floor = eps_floor
        self.recent: deque = deque(maxlen=window)  # (scored, lost) per episode

    def on_episode_end(self, points_scored: int, points_lost: int):
        self.recent.append((points_scored, points_lost))

    def success_rate(self) -> float:
        """Laplace-smoothed success rate over the recent window."""
        total_scored = sum(s for s, _ in self.recent)
        total_lost = sum(l for _, l in self.recent)
        return (total_scored + 1) / (total_scored + total_lost + 2)

    def epsilon(self) -> float:
        """Adaptive exploration rate."""
        return max(self.eps_floor, self.eps_base * (1.0 - self.success_rate()))

    def status(self) -> dict:
        return {
            "success_rate": self.success_rate(),
            "epsilon": self.epsilon(),
            "n_episodes": len(self.recent),
        }
