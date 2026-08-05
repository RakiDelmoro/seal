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
    IMAGINATION_ALPHA_V_GROWTH, GAMMA,
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


class BootstrapScorer:
    """Dreamer/MuZero/TD-MPC-style scoring: "arrive OR be valued."

    The rollout does NOT need to reach the goal s*. Each trajectory is graded by

        score = Σ_t γᵗ · r̂(ŝ_t)  +  γᴴ · V_term(ŝ_H)  −  danger_penalty · 𝟙[danger]

    predicted reward along the way plus the LEARNED VALUE at the endpoint.
    A 5-step rollout walks ~3 units toward a ~150-unit-away goal — arrival is
    impossible (measured: best-of-40 got closer in 0/43 windows), so grading
    by absolute distance ties all plans into noise. Grading by the endpoint's
    learned value makes the effective horizon infinite: V_term(ŝ_H) carries
    "how good is the rest of the future from here?" without walking further.
    s* still steers rollout DIRECTION via the inverse model D; it is the
    grade, not the compass, that changes.

    Vectorized over all K trajectories: one matmul for the path rewards, one
    for the terminal values, one batched danger check — same cost class as
    the geometric scorer.
    """

    def __init__(self):
        self.geo = _get_geometric()

    def score_trajectories(self, trajectories: list[dict],
                           reward_model, terminal_value,
                           danger_penalty: float = DANGER_PENALTY
                           ) -> tuple[list[float], int]:
        """Score all trajectories; return (scores, best_idx).

        Args:
            trajectories: sampler output; each dict holds "states" (list of
                predicted states, length H).
            reward_model: r̂ — linear reward predictor (attr `w`).
            terminal_value: the value function to bootstrap with — pass
                core.scorer_value() (V_sf when SF is on, else V). Linear
                readout with attr `w`.
            danger_penalty: subtracted if any imagined state has the ball on
                our side.
        """
        K = len(trajectories)
        if K == 0:
            return [], 0
        S = np.stack([np.stack(t["states"]) for t in trajectories])  # (K, H, N)
        _, H, N = S.shape

        # (a) The trip: Σ_t γᵗ r̂(ŝ_t) — predicted reward along the rollout.
        r_seq = S.reshape(K * H, N) @ reward_model.w                 # (K*H,)
        discounts = GAMMA ** np.arange(H, dtype=np.float32)          # (H,)
        trip = (r_seq.reshape(K, H) @ discounts)                     # (K,)

        # (b) The landing: γᴴ · V_term(ŝ_H) — value at the endpoint.
        terminal = S[:, -1, :] @ terminal_value.w                    # (K,)
        landing = (GAMMA ** H) * terminal

        # (c) The cliff: danger penalty if any imagined state is on our side.
        E = self.geo._energies_batch(S.reshape(K * H, N))            # (K*H, 81)
        peaks = np.argmax(E, axis=1)
        best_e = E[np.arange(K * H), peaks]
        pxs = peaks % self.geo.grid
        danger = ((best_e >= self.geo.min_energy)
                  & (pxs <= self.geo.our_side_px)).reshape(K, H).any(axis=1)

        scores = trip + landing
        scores = scores - danger_penalty * danger.astype(np.float32)
        scores = [float(x) for x in scores]
        best_idx = int(np.argmax(scores))
        return scores, best_idx


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
