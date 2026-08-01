"""Imagination engine — the policy (a fixed planning procedure).

Action selection (two gates, no learned policy):

  Gate 1: ε-random? ──yes──→ random action (exploration)
            │ no
            ▼
  Gate 2: geometric goal s* exists? ──no──→ random action (no goal yet)
            │ yes
            ▼
  Imagination: sample 40 noisy rollouts toward s*, score each by geometric
               distance to the goal, pick the best → execute.

There is no learned policy π and no confidence gate. The game reward steers
exploration ε (adaptive from the success rate) — losing → explore more → see
more situations → A and D learn the physics better → imagination plans better.
"""
from __future__ import annotations
import numpy as np

from imagination.sampler import sample_trajectories
from imagination.evaluator import evaluate_trajectories
from imagination.geometric_goal import GeometricGoal

from config import (
    N_TRAJECTORIES, IMAGINATION_HORIZON, N_ACTIONS,
    EPSILON_BASE, EPSILON_FLOOR, TOP5_SAMPLING_PROB, DANGER_PENALTY,
)


class ImaginationEngine:
    """Pure imagination + exploration. No learned policy."""

    def __init__(self, n_trajectories: int = N_TRAJECTORIES,
                 horizon: int = IMAGINATION_HORIZON,
                 eps_base: float = EPSILON_BASE,
                 eps_floor: float = EPSILON_FLOOR,
                 top5_prob: float = TOP5_SAMPLING_PROB):
        self.n_trajectories = n_trajectories
        self.horizon = horizon
        self.eps_base = eps_base
        self.eps_floor = eps_floor
        self.top5_prob = top5_prob
        self.geometric = GeometricGoal()
        self.rng = np.random.default_rng()

        self.last_scores = None
        self.last_best_idx = None
        self._source_counts = {"imagination": 0, "random": 0,
                                "epsilon": 0, "no_goal": 0}

    def select_action(self, s_t: np.ndarray, core, success_tracker,
                      use_imagination: bool = True) -> tuple[int, dict]:
        """Select an action for the current state.

        Args:
            s_t: current state (N_STATE,).
            core: SEALCore (dynamics, action_effect, direction, gate, recent_states).
            success_tracker: SuccessTracker (provides adaptive ε from the
                game's ±1 reward — the only place reward enters).
            use_imagination: if False, force random (unused, kept for API compat).

        Returns:
            (action_index, diagnostics_dict)
        """
        epsilon = success_tracker.epsilon() if success_tracker else self.eps_floor

        # ── Gate 1: ε-random (exploration) ──────────────────────────
        # ε is adaptive from the game reward: losing → explore more.
        if self.rng.random() < epsilon:
            action = int(self.rng.integers(N_ACTIONS))
            self._source_counts["epsilon"] += 1
            return action, {
                "action": action, "source": "epsilon",
                "epsilon": epsilon, "n_trajectories": 0,
            }

        # ── Gate 2: geometric goal exists? ──────────────────────────
        # s* = recent state with the ball most on the opponent's side (read
        # geometrically from the state, not learned). If no such state exists
        # yet (not enough history, or the ball has never been seen on the
        # opponent's side), ranking rollouts is meaningless → act randomly.
        s_star = self.geometric.select_goal(core.recent_states)
        if s_star is None:
            action = int(self.rng.integers(N_ACTIONS))
            self._source_counts["no_goal"] += 1
            return action, {
                "action": action, "source": "no_goal",
                "epsilon": epsilon, "n_trajectories": 0,
            }

        # ── Imagination (the policy) ────────────────────────────────
        # Sample K noisy rollouts on the cognitive map toward s*.
        trajectories = sample_trajectories(
            s_t, s_star,
            core.dynamics, core.action_effect, core.direction, core.gate,
            n_trajectories=self.n_trajectories,
            horizon=self.horizon,
            rng=self.rng,
        )

        # Score each by geometric distance to s* + danger penalty.
        scores, best_idx = evaluate_trajectories(
            trajectories, danger_penalty=DANGER_PENALTY, s_star=s_star)
        self.last_scores = scores
        self.last_best_idx = best_idx
        mean_score = float(np.mean(scores))

        # Select action (greedy or top-5 soft selection for variety)
        if self.rng.random() < self.top5_prob and len(trajectories) >= 5:
            top5_idx = np.argsort(scores)[-5:]
            chosen_traj = int(self.rng.choice(top5_idx))
            action = trajectories[chosen_traj]["first_action"]
            source = "top5"
            chosen_score = scores[chosen_traj]
        else:
            action = trajectories[best_idx]["first_action"]
            source = "greedy"
            chosen_score = scores[best_idx]

        self._source_counts["imagination"] += 1
        return action, {
            "action": action,
            "source": source,
            "epsilon": epsilon,
            "n_trajectories": self.n_trajectories,
            "best_score": float(scores[best_idx]),
            "mean_score": mean_score,
            "chosen_score": chosen_score,
        }

    def source_counts(self) -> dict:
        return dict(self._source_counts)
