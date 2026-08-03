"""Geometric goal s* and trajectory scoring — the paper's mechanism, adapted to Pong.

The GCML paper (Lin et al. 2026) scores imagined trajectories by **geometric
distance to an explicit goal state s*** (eq 28: `J = −‖s_{t+H} − o*‖ + β·χ`),
where the goal `o*` is a target observation *given by the task*. Crucially, the
paper never uses a learned value function for imagination — the goal is
geometric/observational, so there is no deadly triad on the planning path.

For Pong, the goal is not handed to us but is **constructible from perception**:
the ball on the opponent's side (we are safe / about to score) vs the ball on
our side (danger). Both are visible in the state: the frozen CNN produces a
9×9 grid of 32-channel features, and the ball — the most salient moving object
— produces the highest activation at its grid position. `px` is horizontal:
  px = 0 → leftmost (our paddle)   →   px = 8 → rightmost (opponent).

We **select s* from recently observed states** by the geometric proxy "ball
furthest on the opponent's side". Rollouts are scored by the paper's form:
  score = −‖ŝ_H − s*‖₁  −  danger_penalty · 𝟙[any ŝ_t on our side]

This is geometry, not learning — it cannot diverge, and it is goal-directed
from frame 1.
"""
from __future__ import annotations
import numpy as np

from config import N_STATE, CNN_GRID, CNN_CHANNELS, N_POSITIONS


class GeometricGoal:
    """Pong goal s* by geometric proxy + geometric rollout scoring.

    All methods are pure functions of the state vector — no learning, no
    divergence possible. The only "domain knowledge" is that px = horizontal
    position and high px = opponent's side, which is the Pong analogue of the
    paper's task-given target observation o*.
    """

    def __init__(self, grid: int = CNN_GRID, channels: int = CNN_CHANNELS,
                 min_energy: float = 1e-3, our_side_px: int = 1):
        self.grid = grid
        self.channels = channels
        self.min_energy = min_energy
        self.our_side_px = our_side_px
        self.n_positions = grid * grid
        # Each position p occupies a contiguous block of `channels` state dims.
        self.ranges = [(p * channels, (p + 1) * channels)
                       for p in range(self.n_positions)]

    # ── Energy per position (fully vectorized — no Python loop) ───
    def _energies(self, s: np.ndarray) -> np.ndarray:
        """Energy per grid position (81,) for one state — vectorized.

        Reshapes the state into (81, 16) and computes the squared norm
        along axis 1 in one shot — no Python loop over positions.
        """
        s = np.asarray(s, dtype=np.float32)
        n = self.n_positions * self.channels
        s_reshaped = s[:n].reshape(self.n_positions, self.channels)
        return np.sum(s_reshaped ** 2, axis=1)

    def _energies_batch(self, S: np.ndarray) -> np.ndarray:
        """Energy per grid position (B, 81) for a batch of states — vectorized."""
        S = np.asarray(S, dtype=np.float32)
        n = self.n_positions * self.channels
        S_reshaped = S[:, :n].reshape(S.shape[0], self.n_positions, self.channels)
        return np.sum(S_reshaped ** 2, axis=2)

    # ── Ball position proxy ─────────────────────────────────────────
    def ball_px(self, s: np.ndarray) -> int | None:
        """Horizontal column (0=our side .. 8=opponent side) of the energy peak.

        Returns None if energy is too low to localize the ball.
        """
        e = self._energies(s)
        peak = int(np.argmax(e))
        if e[peak] < self.min_energy:
            return None
        return peak % self.grid               # px = col

    # ── Goal selection: s* = proven goal (pre-score) or geometric proxy ──
    def select_goal(self, recent_states, pre_score_states=None) -> np.ndarray | None:
        """s* = the goal to aim imagination toward.

        Preference order:
          1. Pre-score states (if any): states that preceded an actual +1.
             These are *proven* goal states — they led to scoring. Pick the
             most recent one (the ball/paddle config that just worked).
          2. Geometric proxy (fallback): the recent state with the highest
             ball px ("ball on the opponent's side"). Used during cold start
             before any +1 has occurred.

        The +1 reward is used here as a GOAL LABEL ("this state scored → aim
        there"), not as a learning signal — no weights are updated from it.
        """
        # 1. Prefer proven goal states (from actual +1s)
        if pre_score_states is not None and len(pre_score_states) > 0:
            return np.asarray(list(pre_score_states)[-1], dtype=np.float32).copy()

        # 2. Fallback: geometric proxy (ball most on opponent side)
        states = [np.asarray(s, dtype=np.float32) for s in recent_states]
        if len(states) < 10:
            return None
        S = np.stack(states)                              # (B, N)
        E = self._energies_batch(S)                      # (B, n_positions)
        peaks = np.argmax(E, axis=1)                      # (B,)
        best_energies = E[np.arange(len(S)), peaks]
        pxs = peaks % self.grid                           # (B,)
        valid = (best_energies >= self.min_energy) & (pxs >= self.our_side_px)
        if not np.any(valid):
            return None
        score = pxs.astype(np.float64) + 1e-6 * best_energies
        score[~valid] = -1.0
        best = int(np.argmax(score))
        if pxs[best] < self.our_side_px:
            return None
        return states[best].copy()

    # ── Danger: ball on our side ────────────────────────────────────
    def in_danger(self, s: np.ndarray) -> bool:
        """True if the ball is on our side (px ≤ our_side_px) and visible."""
        e = self._energies(s)
        peak = int(np.argmax(e))
        if e[peak] < self.min_energy:
            return False
        return (peak % self.grid) <= self.our_side_px

    # ── Geometric trajectory scoring (the paper's J = −‖s_H − o*‖) ──
    def score_trajectory(self, states: list[np.ndarray],
                         s_star: np.ndarray,
                         danger_penalty: float) -> float:
        """Score an imagined rollout by geometric distance to s* + danger.

          score = −‖ŝ_H − s*‖₁  −  danger_penalty · 𝟙[any ŝ_t on our side]
        """
        if len(states) == 0 or s_star is None:
            return 0.0
        terminal = np.asarray(states[-1], dtype=np.float32)
        s_star = np.asarray(s_star, dtype=np.float32)
        dist = float(np.sum(np.abs(terminal - s_star)))
        score = -dist
        # Vectorized danger over all trajectory states.
        S = np.stack([np.asarray(st, dtype=np.float32) for st in states])
        E = self._energies_batch(S)
        peaks = np.argmax(E, axis=1)
        best_e = E[np.arange(len(S)), peaks]
        pxs = peaks % self.grid
        danger = np.any((best_e >= self.min_energy) & (pxs <= self.our_side_px))
        if danger:
            score -= danger_penalty
        return score
