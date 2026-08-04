"""Imagination engine — System 2 planning + System 1 policy.

Action selection (three gates, no reflex, OaK-aligned):

  Gate 1: ε-random? ──yes──→ random action (exploration)
            │ no
            ▼
  Gate 2: π confident? ──yes──→ policy action (System 1, fast)
            │ no                    (20% override → imagination)
            ▼
  Gate 3: goal exists? ──yes──→ imagination (System 2)
            │                      score rollouts by α·V + (1−α)·geometry
            ▼
         random action (no goal yet)

The learned policy π is the only action chooser; imagination plans via the
transition model and teaches π by imitation. The value function V (if present)
critiques plans; if V is not yet useful, imagination falls back to pure
geometric scoring.
"""
from __future__ import annotations
import numpy as np

from imagination.sampler import sample_trajectories
from imagination.evaluator import ValueScorer
from imagination.geometric_goal import GeometricGoal

from config import (
    N_TRAJECTORIES, IMAGINATION_HORIZON, N_ACTIONS,
    EPSILON_BASE, EPSILON_FLOOR, TOP5_SAMPLING_PROB, DANGER_PENALTY,
    PI_CONFIDENCE_THRESHOLD, PI_FORCE_IMAGINATION,
)


class ImaginationEngine:
    """Policy + imagination action selection. No hardcoded reflex."""

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
        self.scorer = ValueScorer()
        self.rng = np.random.default_rng()

        self.last_scores = None
        self.last_best_idx = None
        self._source_counts = {
            "epsilon": 0, "policy": 0, "imagination": 0,
            "no_goal": 0, "random": 0,
        }

        # Recent value predictions to detect whether V has signal
        self._recent_values = []
        self._value_window = 100
        self._v_signal_threshold = 0.01

    def _v_has_signal(self, core) -> bool:
        """True if the SCORING value distinguishes states (std over recent
        window > threshold). Uses core.scorer_value() so it measures V_sf
        when SF is enabled."""
        # Keep only the most recent values to avoid unbounded growth
        if len(self._recent_values) > self._value_window * 2:
            self._recent_values = self._recent_values[-self._value_window:]
        if len(self._recent_values) < 20:
            return False
        vals = np.array(self._recent_values[-self._value_window:])
        return float(vals.std()) > self._v_signal_threshold

    def select_action(self, s_t: np.ndarray, core, success_tracker
                      ) -> tuple[int, dict]:
        """Select an action via the three-gate hierarchy.

        Args:
            s_t: current state.
            core: SEALCore (dynamics, action_effect, direction, value, policy).
            success_tracker: provides adaptive ε.

        Returns:
            (action_index, diagnostics_dict)
        """
        epsilon = success_tracker.epsilon() if success_tracker else self.eps_floor

        # ── Gate 1: exploration ─────────────────────────────────────
        if self.rng.random() < epsilon:
            action = int(self.rng.integers(N_ACTIONS))
            self._source_counts["epsilon"] += 1
            return action, {
                "action": action, "source": "epsilon",
                "epsilon": epsilon, "n_trajectories": 0,
            }

        # ── Gate 2: learned policy (System 1) ───────────────────────
        pi_conf = core.policy.confidence(s_t)
        if pi_conf > PI_CONFIDENCE_THRESHOLD:
            if self.rng.random() > PI_FORCE_IMAGINATION:
                action = core.policy.predict_action(s_t)
                self._source_counts["policy"] += 1
                return action, {
                    "action": action, "source": "policy",
                    "epsilon": epsilon, "n_trajectories": 0,
                    "pi_confidence": pi_conf,
                }
            # else fall through to imagination to keep teaching π

        # ── Gate 3: imagination (System 2) ────────────────────────────
        s_star = self.geometric.select_goal(core.recent_states,
                                            getattr(core, "pre_score_states", None))

        if s_star is not None:
            # Temporarily disable value-based scoring if V has no signal yet.
            v_signal = self._v_has_signal(core)
            if not v_signal:
                self.scorer.alpha = 0.0

            trajectories = sample_trajectories(
                s_t, s_star,
                core.dynamics, core.action_effect, core.direction, core.gate,
                n_trajectories=self.n_trajectories,
                horizon=self.horizon,
                rng=self.rng,
            )
            scores, best_idx = self.scorer.score_trajectories(
                trajectories, value=core.scorer_value(), s_star=s_star,
                danger_penalty=DANGER_PENALTY)
            self.last_scores = scores
            self.last_best_idx = best_idx

            # Top-5 soft selection for diversity
            if self.rng.random() < self.top5_prob and len(trajectories) >= 5:
                top5_idx = np.argsort(scores)[-5:]
                chosen_traj = int(self.rng.choice(top5_idx))
                action = trajectories[chosen_traj]["first_action"]
                source = "top5"
            else:
                action = trajectories[best_idx]["first_action"]
                source = "greedy"

            self._source_counts["imagination"] += 1
            if v_signal:
                self._recent_values.append(core.scorer_value().forward(s_t))
                self.scorer.grow_alpha(1)
            return action, {
                "action": action,
                "source": source,
                "epsilon": epsilon,
                "n_trajectories": self.n_trajectories,
                "best_score": float(scores[best_idx]),
                "mean_score": float(np.mean(scores)),
                "alpha_v": self.scorer.alpha,
                "pi_confidence": pi_conf,
            }

        # No goal available yet
        self._source_counts["no_goal"] += 1
        action = int(self.rng.integers(N_ACTIONS))
        return action, {
            "action": action,
            "source": "no_goal",
            "epsilon": epsilon,
            "n_trajectories": 0,
            "pi_confidence": pi_conf,
        }

    def source_counts(self) -> dict:
        return dict(self._source_counts)

    def last_score_stats(self) -> dict:
        """Stats over the most recent imagination's scores."""
        if self.last_scores is None or len(self.last_scores) == 0:
            return {"score_std": 0.0, "score_mean": 0.0, "score_best": 0.0}
        s = np.asarray(self.last_scores, dtype=np.float64)
        return {
            "score_std": float(np.std(s)),
            "score_mean": float(np.mean(s)),
            "score_best": float(np.max(s)),
        }
