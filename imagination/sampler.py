"""Trajectory sampler — generate K noisy imagined rollouts (batched).

All K trajectories are advanced in parallel as a (K, N) batch, giving a
K× speedup over the per-trajectory loop. For each imagination step:

  Δ = s* - ŝ                          (K, N) — direction to goal
  u = D · Δ^T → (K, A)                inverse model: action utilities
  ε ~ N(0, σ²)                         (K, A) — noise for diversity
  e = F ⊙ (u + ε)                     (K, A) — eligibility
  a = argmax(e, axis=1)                (K,) — winner-take-all
  ŝ = A·ŝ + B·a + b                   (K, N) — PREDICT next state

Adaptive noise: σ = max(floor, scale · ‖u‖) per trajectory.
"""
from __future__ import annotations
import numpy as np

from config import (
    N_TRAJECTORIES, IMAGINATION_HORIZON, N_ACTIONS,
    NOISE_SIGMA_FLOOR, NOISE_SIGMA_SCALE,
)


def sample_trajectories(s_0: np.ndarray, s_star: np.ndarray | None,
                        dynamics, action_effect, direction, gate,
                        n_trajectories: int = N_TRAJECTORIES,
                        horizon: int = IMAGINATION_HORIZON,
                        noise_floor: float = NOISE_SIGMA_FLOOR,
                        noise_scale: float = NOISE_SIGMA_SCALE,
                        rng: np.random.Generator | None = None
                        ) -> list[dict]:
    """Generate K noisy imagined rollouts from s_0 toward s* (batched).

    Returns a list of K trajectory dicts. Each dict carries the predicted
    state trajectory (`states`) so the evaluator can score it geometrically
    by distance to s* (no learned value is used for ranking).
    """
    if rng is None:
        rng = np.random.default_rng()

    K = n_trajectories
    N = s_0.shape[0]
    F = gate.forward()  # (n_actions,) all-ones for Pong
    D_mat = direction.D        # (n_actions, N)
    B_mat = action_effect.B    # (N, n_actions)

    # Initialize batch: all trajectories start from s_0
    S = np.tile(s_0, (K, 1))   # (K, N)
    # Target norm: keep rollouts at the real state's magnitude so the
    # geometric scorer can compare them to the full-size goal s*. Without this,
    # A (a shrinkage operator, ‖A‖_op≈0.96) shrinks 5-step rollouts to ~25% of
    # their starting norm — making all 40 futures look equally far from the
    # goal regardless of their direction. The rollouts are accurate and
    # diverse (measured); renormalizing preserves that good information.
    target_norm = float(np.linalg.norm(s_0)) + 1e-8

    # Per-trajectory storage: first action + full predicted-state trajectory
    # (the evaluator scores these geometrically against s*).
    all_states = np.empty((K, horizon, N), dtype=np.float32)
    first_actions = np.full(K, -1, dtype=np.int64)

    for step in range(horizon):
        # Direction to goal: (K, N)
        if s_star is not None:
            delta = s_star[None, :] - S   # (K, N)
        else:
            delta = np.zeros_like(S)

        # Inverse model: u = D @ delta^T  → (N, K) → transpose → (K, n_actions)
        u = delta @ D_mat.T              # (K, n_actions)

        # Adaptive noise per trajectory
        u_norms = np.linalg.norm(u, axis=1)  # (K,)
        sigmas = np.maximum(noise_floor, noise_scale * u_norms)  # (K,)

        epsilon = rng.normal(0, 1, (K, N_ACTIONS)).astype(np.float32) * sigmas[:, None]

        # Eligibility
        e = F[None, :] * (u + epsilon)   # (K, n_actions)

        # Winner-take-all
        actions = np.argmax(e, axis=1)    # (K,)

        if step == 0:
            first_actions = actions.copy()

        # One-hot actions: (K, n_actions)
        a_onehot = np.zeros((K, N_ACTIONS), dtype=np.float32)
        a_onehot[np.arange(K), actions] = 1.0

        # Predict next state (batched): ŝ = A·ŝ + B·a
        S = dynamics.predict_batch(S, a_onehot, B_mat)  # (K, N)
        # Renormalize: keep each rollout at the real state's magnitude so the
        # geometric scorer isn't dominated by shrinkage. (Fixes the measured
        # rollout_norm_ratio collapse to 0.25 — the diagnosed bottleneck.)
        norms = np.linalg.norm(S, axis=1, keepdims=True) + 1e-8
        S = S * (target_norm / norms)
        all_states[:, step, :] = S

    # Convert to list of dicts. `states` holds the predicted trajectory so the
    # evaluator can score it geometrically (−‖ŝ_H − s*‖ + danger).
    trajectories = []
    for k in range(K):
        trajectories.append({
            "first_action": int(first_actions[k]),
            "states": [all_states[k, t, :] for t in range(horizon)],
        })
    return trajectories
